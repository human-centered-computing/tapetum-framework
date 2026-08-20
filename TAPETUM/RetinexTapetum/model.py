"""
RetinexTapetum.

Fast Retinex + Tapetum model designed to beat RetinexFormer on parameter
count and FPS while keeping strong color fidelity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def rgb_to_luminance(x):
    """Convert RGB tensors to luminance while leaving single-channel maps unchanged."""
    if x.size(1) == 1:
        return x
    return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]


def high_frequency_luminance(x, pool_size=15):
    """Extract local detail by subtracting a blurred luminance map from luminance."""
    y = rgb_to_luminance(x)
    y_low = F.avg_pool2d(y, kernel_size=pool_size, stride=1, padding=pool_size // 2)
    return y - y_low


class DepthwiseSeparableConv(nn.Module):
    """
    Efficient convolution block.

    The depthwise convolution extracts spatial patterns per channel, then the
    1x1 pointwise convolution mixes channels. This keeps the model fast while
    preserving most of the representational power needed for enhancement.
    """

    def __init__(self, in_ch, out_ch, activation=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, in_ch, 3, 1, 1, groups=in_ch),
            nn.Conv2d(in_ch, out_ch, 1),
        ]
        if activation:
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    """Residual feature block used to refine features without losing input detail."""

    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(
            DepthwiseSeparableConv(channels, channels),
            DepthwiseSeparableConv(channels, channels, activation=False),
        )

    def forward(self, x):
        return F.relu(x + self.body(x), inplace=True)


class ChannelGate(nn.Module):
    """Channel attention block that learns which feature channels matter most."""

    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.gate(x)


class DecomNet(nn.Module):
    """
    Retinex decomposition network.

    Given a low-light RGB image, it predicts reflectance R and illumination L.
    Both outputs are RGB maps in [0, 1], so the reconstruction can be written as
    image ~= R * L.
    """


    def __init__(self, in_ch=3, base=32):
        super().__init__()
        self.head = nn.Conv2d(in_ch, base, 3, 1, 1)
        self.body = nn.Sequential(
            ResidualBlock(base),
            ResidualBlock(base),
            ResidualBlock(base),
            ChannelGate(base),
        )
        self.r_out = nn.Conv2d(base, 3, 3, 1, 1)
        self.l_out = nn.Conv2d(base, 3, 3, 1, 1)

    def forward(self, x):
        f = F.relu(self.head(x), inplace=True)
        f = self.body(f)
        return torch.sigmoid(self.r_out(f)), torch.sigmoid(self.l_out(f))


class TapetumAttention(nn.Module):
    """
    Predict where the tapetum amplification should be active.

    Input channels:
        low RGB + RGB L + dark prior = 7 channels.

    Output:
        3-channel attention map T in [0, 1]. Larger values mean the model wants
        more photon-reuse-inspired illumination amplification in that region.
    """

    def __init__(self, in_ch=7, base=32):
        super().__init__()
        self.enc = nn.Sequential(
            DepthwiseSeparableConv(in_ch, base),
            ResidualBlock(base),
            ChannelGate(base),
        )
        self.down = nn.Sequential(
            nn.AvgPool2d(2),
            DepthwiseSeparableConv(base, base * 2),
            ResidualBlock(base * 2),
        )
        self.context = nn.Sequential(
            nn.Conv2d(base * 2, base * 2, 3, 1, 2, dilation=2, groups=base * 2),
            nn.Conv2d(base * 2, base * 2, 1),
            nn.ReLU(inplace=True),
            ChannelGate(base * 2),
        )
        self.up = nn.Conv2d(base * 2, base, 1)
        self.fuse = nn.Sequential(
            DepthwiseSeparableConv(base * 2, base),
            nn.Conv2d(base, 3, 1),
        )

    def forward(self, x):
        e = self.enc(x)
        c = self.context(self.down(e))
        c = F.interpolate(c, size=e.shape[-2:], mode="bilinear", align_corners=False)
        c = self.up(c)
        raw = self.fuse(torch.cat([e, c], dim=1))
        return torch.sigmoid(raw)


class LambdaMap(nn.Module):
    """
    Predict the spatial amplification strength.

    The lambda map controls how much the illumination is amplified. It is bounded
    by lambda_max and gated by dark_prior so bright regions are not boosted
    unnecessarily.
    """

    def __init__(self, in_ch=4, base=16, lambda_max=1.65):
        super().__init__()
        self.lambda_max = lambda_max
        self.net = nn.Sequential(
            DepthwiseSeparableConv(in_ch, base),
            ResidualBlock(base),
            nn.Conv2d(base, 3, 1),
            nn.Sigmoid(),
        )

    def forward(self, L, dark_prior):
        x = torch.cat([L, dark_prior], dim=1)
        return self.lambda_max * self.net(x) * dark_prior


class ColorRefinement(nn.Module):
    """
    Residual head that repairs color and fine detail after Retinex reconstruction.

    The tanh output is scaled by 0.08 so this branch makes controlled corrections
    instead of overwriting the physically motivated R * L_t reconstruction.
    """

    def __init__(self, in_ch=13, base=24):
        super().__init__()
        self.net = nn.Sequential(
            DepthwiseSeparableConv(in_ch, base),
            ResidualBlock(base),
            ChannelGate(base),
            nn.Conv2d(base, 3, 3, 1, 1),
        )

    def forward(self, x):
        return 0.08 * torch.tanh(self.net(x))


class RetinexTapetum(nn.Module):
    """
    End-to-end RetinexTapetum enhancement model.

    Pipeline:
        1. Decompose low-light input into reflectance R_low and illumination L_low.
        2. Estimate detail and darkness cues from the input/illumination.
        3. Predict tapetum attention T and spatial amplification lambda_map.
        4. Build enhanced illumination L_t = L_low * (1 + lambda_map * T).
        5. Reconstruct base image R_low * L_t and refine it with a small residual.
    """

    def __init__(self, base=32, lambda_init=0.0, lambda_max=1.65):
        super().__init__()
        del lambda_init
        self.decom_net = DecomNet(in_ch=3, base=base)
        self.tapetum_net = TapetumAttention(in_ch=7, base=base)
        self.lambda_map_net = LambdaMap(
            in_ch=4,
            base=max(base // 2, 12),
            lambda_max=lambda_max,
        )
        self.refine_net = ColorRefinement(
            in_ch=13,
            base=max(base // 2, 16),
        )

    def forward(self, low, high=None):
        """Run enhancement; when high is provided, also return training-only terms."""
        # Retinex decomposition separates scene content from illumination.
        R_low, L_low = self.decom_net(low)

        # Detail and darkness cues guide where amplification should happen.
        y_high = high_frequency_luminance(low)
        dark_prior = torch.clamp(1.0 - rgb_to_luminance(L_low), 0.0, 1.0)

        # Attention says where to amplify; lambda_map says how strongly.
        attention_input = torch.cat([low, L_low, dark_prior], dim=1)
        T = self.tapetum_net(attention_input)
        lambda_map = self.lambda_map_net(L_low, dark_prior)

        # Tapetum illumination update: active, spatially adaptive light reuse.
        L_t = L_low * (1.0 + lambda_map * T)
        base_enh = R_low * L_t

        # Final residual correction restores small color/detail errors.
        refine_input = torch.cat([low, base_enh, T, lambda_map, y_high], dim=1)
        residual = self.refine_net(refine_input)
        enhanced = torch.clamp(base_enh + residual, 0.0, 1.0)

        # Keep intermediate maps for visualization, debugging, and losses.
        out = {
            "enhanced": enhanced,
            "base_enhanced": base_enh,
            "residual": residual,
            "reflectance_low": R_low,
            "illumination_low": L_low,
            "tapetum_attention": T,
            "frequency_high": y_high,
            "dark_prior": dark_prior,
            "illumination_t": L_t,
            "lambda_map": lambda_map,
            "lambda": lambda_map.mean(),
        }

        if high is not None:
            # Training uses the normal-light pair to regularize decomposition.
            R_high, L_high = self.decom_net(high)
            out.update(
                {
                    "reflectance_high": R_high,
                    "illumination_high": L_high,
                    "recon_low": torch.clamp(R_low * L_low, 0.0, 1.0),
                    "recon_high": torch.clamp(R_high * L_high, 0.0, 1.0),
                }
            )

        return out

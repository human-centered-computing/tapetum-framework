"""
Retinex-Tapetum model implementation.

Compact darkness-aware Retinex framework with explicit, bounded illumination
control and bounded residual color-detail refinement.
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

    Given a low-light RGB image, it predicts task-oriented reflectance-like
    and illumination-like RGB representations in [0, 1]. The decomposition is
    used computationally and is not interpreted as a physically unique scene
    separation.
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
    Predict the three-channel Tapetum Attention Map T.

    Input channels:
        low RGB + illumination-like RGB + darkness prior = 7 channels.

    Output:
        T in [0, 1], used as a learned spatial/chromatic modulation map. This is
        not Transformer self-attention or query-key-value attention.
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

    The Darkness-Gated Spatial Amplification Map controls the available
    illumination-update magnitude. It is bounded by lambda_max and explicitly
    gated by dark_prior so relatively bright regions receive less amplification.
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
    Residual head for bounded color-detail refinement after base reconstruction.

    The tanh output is scaled by 0.08 so this branch makes controlled corrections
    instead of replacing the explicit Retinex-guided illumination pathway.
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
    End-to-end Retinex-Tapetum enhancement model.

    Pipeline:
        1. Estimate reflectance-like R_low and illumination-like L_low.
        2. Derive the high-frequency luminance cue Y_HF and darkness prior D.
        3. Predict Tapetum Attention Map T and spatial amplification Lambda.
        4. Update illumination: L_t = L_low * (1 + Lambda * T).
        5. Reconstruct I_base = R_low * L_t and apply a bounded RGB residual.
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
        # Task-oriented Retinex decomposition into learned intermediate maps.
        R_low, L_low = self.decom_net(low)

        # Auxiliary guidance cues. Y_HF is used only by the final refinement
        # branch; D guides the Tapetum attention and amplification branches.
        y_hf = high_frequency_luminance(low)
        dark_prior = torch.clamp(1.0 - rgb_to_luminance(L_low), 0.0, 1.0)

        # T modulates the update spatially/chromatically; lambda_map determines
        # the darkness-gated available amplification magnitude.
        attention_input = torch.cat([low, L_low, dark_prior], dim=1)
        T = self.tapetum_net(attention_input)
        lambda_map = self.lambda_map_net(L_low, dark_prior)

        # Bounded Tapetum illumination update.
        L_t = L_low * (1.0 + lambda_map * T)
        base_enh = R_low * L_t

        # Final bounded residual color-detail refinement.
        refine_input = torch.cat([low, base_enh, T, lambda_map, y_hf], dim=1)
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
            "frequency_high": y_hf,
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

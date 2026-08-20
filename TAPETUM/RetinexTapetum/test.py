"""
Inference script for RetinexTapetum.

Loads the best checkpoint, runs the model on all low-light test images, and
writes every enhanced output to RESULT_DIR as a lossless PNG file.
"""

import os
from PIL import Image
from tqdm import tqdm

import torch
from torchvision import transforms

from config import (
    DEVICE,
    DATA_NAME,
    DATA_VARIANT,
    TEST_LOW_DIR,
    CKPT_DIR,
    BASE_CHANNELS,
    LAMBDA_INIT,
    LAMBDA_MAX,
    COLOR_RESTORE,
    COLOR_RESTORE_STRENGTH,
    COLOR_RESTORE_EPS,
)
from model import RetinexTapetum

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")

# -----------------------------------------------------------------------------
# Standard RESULT directory
# -----------------------------------------------------------------------------
#
# Dataset without variant:
#   RESULT/<DATA_NAME>/RetinexTapetum/results/Test
#
# Dataset with variant:
#   RESULT/<DATA_NAME>/<DATA_VARIANT>/RetinexTapetum/results/Test
#
RESULT_ROOT = "/content/drive/MyDrive/TAPETUM/RESULT"
MODEL_NAME = "RetinexTapetum"

if DATA_VARIANT is None:
    MODEL_RESULT_ROOT = os.path.join(
        RESULT_ROOT,
        DATA_NAME,
        MODEL_NAME,
    )
else:
    MODEL_RESULT_ROOT = os.path.join(
        RESULT_ROOT,
        DATA_NAME,
        DATA_VARIANT,
        MODEL_NAME,
    )

RESULT_DIR = os.path.join(
    MODEL_RESULT_ROOT,
    "results",
    "Test",
)

METRICS_DIR = os.path.join(
    MODEL_RESULT_ROOT,
    "paper_metrics",
)

ANALYSIS_DIR = os.path.join(
    MODEL_RESULT_ROOT,
    "analysis",
)

RUN_SUMMARY_PATH = os.path.join(
    MODEL_RESULT_ROOT,
    "colab_run_summary.json",
)

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(ANALYSIS_DIR, exist_ok=True)

print("DEVICE      :", DEVICE)
print("DATA_NAME   :", DATA_NAME)
print("DATA_VARIANT:", DATA_VARIANT)
print("TEST_LOW_DIR:", TEST_LOW_DIR)
print("CKPT_DIR    :", CKPT_DIR)
print("MODEL_ROOT  :", MODEL_RESULT_ROOT)
print("RESULT_DIR  :", RESULT_DIR)
print("METRICS_DIR :", METRICS_DIR)
print("ANALYSIS_DIR:", ANALYSIS_DIR)
print("COLOR_RESTORE:", COLOR_RESTORE)


def list_images(folder):
    """List supported test images in sorted order."""
    return sorted([f for f in os.listdir(folder) if f.lower().endswith(IMG_EXTS)])


def infer_base_channels(checkpoint):
    """
    Infer the model width from the checkpoint.

    This makes inference tolerant to config changes: if BASE_CHANNELS changed
    after training, the saved weights still decide the correct architecture.
    """
    state = checkpoint.get("model", checkpoint)
    head_weight = state.get("decom_net.head.weight")
    if head_weight is None:
        return BASE_CHANNELS
    return int(head_weight.shape[0])


def load_model():
    """Load the best saved checkpoint and rebuild the model."""
    ckpt_path = os.path.join(CKPT_DIR, "best.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    checkpoint_base = infer_base_channels(checkpoint)
    if checkpoint_base != BASE_CHANNELS:
        print(
            "BASE_CHANNELS mismatch:",
            f"config={BASE_CHANNELS}, checkpoint={checkpoint_base}.",
            "Using checkpoint value.",
        )

    model = RetinexTapetum(
        base=checkpoint_base,
        lambda_init=LAMBDA_INIT,
        lambda_max=LAMBDA_MAX,
    ).to(DEVICE)

    try:
        model.load_state_dict(checkpoint["model"])
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint architecture mismatch. Train a fresh RetinexTapetum "
            "checkpoint with the current V2-quality config/model.py."
        ) from exc
    model.eval()

    print(f"Loaded checkpoint: {ckpt_path}")
    print("Model BASE_CHANNELS:", checkpoint_base)
    print("Model LAMBDA_MAX:", LAMBDA_MAX)
    if "best_psnr" in checkpoint:
        print(f"Best PSNR: {checkpoint['best_psnr']:.4f}")
    if "best_epoch" in checkpoint:
        print(f"Best Epoch: {checkpoint['best_epoch']}")
    if "lambda" in checkpoint:
        print("Saved lambda:", checkpoint["lambda"])

    return model


def tensor_to_pil(x):
    """Convert a model output tensor in [0, 1] into a PIL image."""
    x = x.squeeze(0).detach().cpu().clamp(0.0, 1.0)
    return transforms.ToPILImage()(x)


def rgb_to_luminance(x):
    return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]


def restore_input_chroma(enhanced, low, strength=0.75, eps=1e-4):
    """
    Re-inject input hue/chroma into the enhanced luminance.

    This optional inference-side repair keeps the model's enhanced brightness
    while borrowing RGB channel ratios from the original low-light image. It can
    help when a checkpoint produces desaturated outputs.
    """
    y_low = rgb_to_luminance(low)
    y_enh = rgb_to_luminance(enhanced)
    input_chroma = low / (y_low + eps)
    chroma_restored = torch.clamp(y_enh * input_chroma, 0.0, 1.0)
    return torch.clamp((1.0 - strength) * enhanced + strength * chroma_restored, 0.0, 1.0)


@torch.no_grad()
def run_test():
    model = load_model()
    to_tensor = transforms.ToTensor()

    files = list_images(TEST_LOW_DIR)
    print("Test image count:", len(files))
    print("Output format   : PNG")

    for fname in tqdm(files, desc="Testing"):
        in_path = os.path.join(TEST_LOW_DIR, fname)

        # Preserve the source filename stem, but always save the model output
        # as lossless PNG so JPEG/WebP compression does not affect metrics.
        output_name = os.path.splitext(fname)[0] + ".png"
        out_path = os.path.join(RESULT_DIR, output_name)

        # Convert PIL RGB image to a normalized tensor with batch dimension.
        img = Image.open(in_path).convert("RGB")
        inp = to_tensor(img).unsqueeze(0).to(DEVICE)

        # The model returns intermediate maps too; only enhanced is saved here.
        output = model(inp)
        enh = output["enhanced"]
        if COLOR_RESTORE:
            enh = restore_input_chroma(
                enh,
                inp,
                strength=COLOR_RESTORE_STRENGTH,
                eps=COLOR_RESTORE_EPS,
            )

        out_img = tensor_to_pil(enh).convert("RGB")
        out_img.save(
            out_path,
            format="PNG",
            optimize=False,
            compress_level=6,
        )

    print("===== TEST FINISHED =====")
    print("Dataset       :", DATA_NAME)
    print("Variant       :", DATA_VARIANT)
    print("Model         :", MODEL_NAME)
    print("Results saved :", RESULT_DIR)
    print("Metrics path  :", METRICS_DIR)
    print("Analysis path :", ANALYSIS_DIR)


if __name__ == "__main__":
    run_test()
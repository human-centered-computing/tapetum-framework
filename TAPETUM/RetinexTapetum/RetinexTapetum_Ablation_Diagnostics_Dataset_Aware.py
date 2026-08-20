# ============================================================
# @title RetinexTapetum — Ablation + Diagnostics
# Dataset Colab üzerinden seçilir.
# Tüm sonuçlar seçilen dataset'in RUN_ROOT klasörüne kaydedilir.
# ============================================================

from google.colab import drive
drive.mount("/content/drive")

import os
import sys
import csv
import math
import shutil
import subprocess
import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from tqdm.auto import tqdm
from IPython.display import display

import torch
import torch.nn.functional as F
from torchvision import transforms


# ============================================================
# 0. USER SETTINGS
# ============================================================

#@markdown ### Dataset seçimi

DATA_NAME = "LOL-v2" #@param ["LOL-v1", "LOL-v2", "UHD-LL down4", "SICE", "LoLI-Street"]

DATA_VARIANT = "Real_captured" #@param ["Real_captured", "Synthetic", "None"]

# Belirli bir test görüntüsü seçmek için dosya adını yaz.
# None bırakılırsa SAMPLE_INDEX kullanılır.
SAMPLE_FILENAME = None

# SAMPLE_FILENAME None ise kullanılacak test görüntüsü indeksi.
SAMPLE_INDEX = 0

# Görsellerin kayıt çözünürlüğü.
FIGURE_DPI = 220

# Internal behavior bölge eşikleri.
DARK_THRESHOLD = 0.50
BRIGHT_THRESHOLD = 0.20


# ============================================================
# 1. PROJECT AND DATASET PATHS
# ============================================================

PROJECT_DIR = Path(
    "/content/drive/MyDrive/TAPETUM/RetinexTapetum"
).resolve()

DATASETS_BASE_DIR = Path(
    "/content/drive/MyDrive/TAPETUM/datasets"
).resolve()

SELECTED_VARIANT = (
    None
    if str(DATA_VARIANT).strip().lower() in {
        "",
        "none",
        "null",
    }
    else DATA_VARIANT
)

if SELECTED_VARIANT is None:
    EXPECTED_DATASET_DIR = DATASETS_BASE_DIR / DATA_NAME
else:
    EXPECTED_DATASET_DIR = (
        DATASETS_BASE_DIR
        / DATA_NAME
        / SELECTED_VARIANT
    )


# ============================================================
# 2. PASS SELECTION TO config.py
# ============================================================

# config.py bu ortam değişkenlerini okumalıdır.
os.environ["RETINEX_DATA_NAME"] = DATA_NAME
os.environ["RETINEX_DATA_VARIANT"] = (
    "None"
    if SELECTED_VARIANT is None
    else SELECTED_VARIANT
)
os.environ["PYTHONUNBUFFERED"] = "1"


# ============================================================
# 3. VALIDATE PROJECT
# ============================================================

if not PROJECT_DIR.is_dir():
    raise FileNotFoundError(
        "Proje klasörü bulunamadı:\n"
        f"{PROJECT_DIR}"
    )

if not EXPECTED_DATASET_DIR.is_dir():
    raise FileNotFoundError(
        "Seçilen dataset klasörü bulunamadı:\n"
        f"{EXPECTED_DATASET_DIR}"
    )

os.chdir(PROJECT_DIR)

project_path = str(PROJECT_DIR)

if project_path in sys.path:
    sys.path.remove(project_path)

sys.path.insert(0, project_path)


# ============================================================
# 4. CLEAR PYTHON CACHE
# ============================================================

for cache_dir in PROJECT_DIR.rglob("__pycache__"):
    if cache_dir.is_dir():
        shutil.rmtree(
            cache_dir,
            ignore_errors=True,
        )

for pyc_file in PROJECT_DIR.rglob("*.pyc"):
    try:
        pyc_file.unlink()
    except OSError:
        pass

for module_name in [
    "config",
    "model",
    "dataset",
    "losses",
    "utils",
]:
    sys.modules.pop(
        module_name,
        None,
    )

importlib.invalidate_caches()


# ============================================================
# 5. IMPORT CONFIG AND MODEL
# ============================================================

import config
importlib.reload(config)

from model import RetinexTapetum


# ============================================================
# 6. VERIFY CONFIG SELECTION
# ============================================================

if config.DATA_NAME != DATA_NAME:
    raise RuntimeError(
        "DATA_NAME config.py dosyasına aktarılmadı.\n"
        f"Colab seçimi : {DATA_NAME!r}\n"
        f"Config değeri: {config.DATA_NAME!r}"
    )

if config.DATA_VARIANT != SELECTED_VARIANT:
    raise RuntimeError(
        "DATA_VARIANT config.py dosyasına aktarılmadı.\n"
        f"Colab seçimi : {SELECTED_VARIANT!r}\n"
        f"Config değeri: {config.DATA_VARIANT!r}"
    )


# ============================================================
# 7. CONFIG VALUES
# ============================================================

DEVICE = torch.device(
    config.DEVICE
    if torch.cuda.is_available()
    else "cpu"
)

TEST_LOW_DIR = Path(
    config.TEST_LOW_DIR
)

TEST_HIGH_DIR = Path(
    config.TEST_HIGH_DIR
)

CKPT_DIR = Path(
    config.CKPT_DIR
)

RUN_ROOT = Path(
    config.RUN_ROOT
)

BASE_CHANNELS = config.BASE_CHANNELS
LAMBDA_INIT = config.LAMBDA_INIT
LAMBDA_MAX = config.LAMBDA_MAX

LPIPS_LOSS_RESIZE = getattr(
    config,
    "LPIPS_LOSS_RESIZE",
    256,
)


# ============================================================
# 8. DATASET-SPECIFIC OUTPUT PATHS
# ============================================================

# Örnek:
# /content/drive/MyDrive/TAPETUM/LOL-v2/
# RetinexTapetum-Real_captured/analysis/
#     ablation/
#     diagnostics/
#     visual_diagnostics/

ANALYSIS_ROOT = RUN_ROOT / "analysis"

ABLATION_DIR = ANALYSIS_ROOT / "ablation"
DIAGNOSTICS_DIR = ANALYSIS_ROOT / "diagnostics"
VISUAL_DIR = ANALYSIS_ROOT / "visual_diagnostics"

for directory in [
    ANALYSIS_ROOT,
    ABLATION_DIR,
    DIAGNOSTICS_DIR,
    VISUAL_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


ABLATION_SUMMARY_PATH = (
    ABLATION_DIR
    / "RetinexTapetum_ablation_summary.csv"
)

ABLATION_DETAIL_PATH = (
    ABLATION_DIR
    / "RetinexTapetum_ablation_detail.csv"
)

ABLATION_LATEX_PATH = (
    ABLATION_DIR
    / "RetinexTapetum_ablation_latex.txt"
)

DIAGNOSTICS_SUMMARY_PATH = (
    DIAGNOSTICS_DIR
    / "RetinexTapetum_internal_diagnostics_summary.csv"
)

DIAGNOSTICS_DETAIL_PATH = (
    DIAGNOSTICS_DIR
    / "RetinexTapetum_internal_diagnostics_detail.csv"
)

DIAGNOSTICS_GAIN_PATH = (
    DIAGNOSTICS_DIR
    / "RetinexTapetum_internal_gain_summary.csv"
)

DIAGNOSTICS_LATEX_PATH = (
    DIAGNOSTICS_DIR
    / "RetinexTapetum_internal_diagnostics_latex.txt"
)

METADATA_PATH = (
    ANALYSIS_ROOT
    / "RetinexTapetum_analysis_metadata.txt"
)


# ============================================================
# 9. PRINT ENVIRONMENT
# ============================================================

print("=" * 90)
print("RETINEXTAPETUM — ABLATION + DIAGNOSTICS")
print("=" * 90)

print("PROJECT_DIR       :", PROJECT_DIR)
print("CONFIG FILE       :", config.__file__)
print("DATA_NAME         :", config.DATA_NAME)
print("DATA_VARIANT      :", config.DATA_VARIANT)
print("DATA_ROOT         :", config.DATA_ROOT)
print("TEST_LOW_DIR      :", TEST_LOW_DIR)
print("TEST_HIGH_DIR     :", TEST_HIGH_DIR)
print("RUN_ROOT          :", RUN_ROOT)
print("CKPT_DIR          :", CKPT_DIR)
print("ANALYSIS_ROOT     :", ANALYSIS_ROOT)
print("ABLATION_DIR      :", ABLATION_DIR)
print("DIAGNOSTICS_DIR   :", DIAGNOSTICS_DIR)
print("VISUAL_DIR        :", VISUAL_DIR)
print("DEVICE            :", DEVICE)
print("LAMBDA_MAX        :", LAMBDA_MAX)
print("LPIPS RESIZE      :", LPIPS_LOSS_RESIZE)

print("=" * 90)


# ============================================================
# 10. REQUIRED DIRECTORY CHECK
# ============================================================

required_dirs = {
    "TEST_LOW_DIR": TEST_LOW_DIR,
    "TEST_HIGH_DIR": TEST_HIGH_DIR,
    "CKPT_DIR": CKPT_DIR,
}

for name, directory in required_dirs.items():
    if not directory.is_dir():
        raise FileNotFoundError(
            f"{name} bulunamadı:\n"
            f"{directory}"
        )


# ============================================================
# 11. INSTALL / LOAD LPIPS
# ============================================================

def ensure_lpips():
    try:
        import lpips
    except ImportError:
        print("lpips kuruluyor...")

        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "lpips",
            ]
        )

        import lpips

    return lpips


lpips_module = ensure_lpips()

lpips_fn = lpips_module.LPIPS(
    net="alex"
).to(DEVICE).eval()

for parameter in lpips_fn.parameters():
    parameter.requires_grad_(False)


# ============================================================
# 12. GENERAL HELPERS
# ============================================================

IMG_EXTS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
)


def list_images(folder):
    folder = Path(folder)

    if not folder.is_dir():
        return []

    return sorted(
        file.name
        for file in folder.iterdir()
        if (
            file.is_file()
            and file.suffix.lower() in IMG_EXTS
        )
    )


def count_params(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def infer_base_channels(checkpoint):
    state_dict = checkpoint.get(
        "model",
        checkpoint,
    )

    head_weight = state_dict.get(
        "decom_net.head.weight"
    )

    if head_weight is None:
        return BASE_CHANNELS

    return int(
        head_weight.shape[0]
    )


def load_image_as_tensor(path):
    image = Image.open(
        path
    ).convert("RGB")

    return (
        transforms.ToTensor()(image)
        .unsqueeze(0)
        .to(DEVICE)
    )


def safe_mean_tensor(x):
    if x.numel() == 0:
        return float("nan")

    return (
        x.detach()
        .float()
        .mean()
        .item()
    )


def safe_median_tensor(x):
    if x.numel() == 0:
        return float("nan")

    return (
        x.detach()
        .float()
        .median()
        .item()
    )


def finite_mean(values):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return float("nan")

    return float(
        np.mean(values)
    )


def finite_median(values):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return float("nan")

    return float(
        np.median(values)
    )


def pearson_corr_torch(
    a,
    b,
    eps=1e-8,
):
    a = (
        a.detach()
        .float()
        .reshape(-1)
    )

    b = (
        b.detach()
        .float()
        .reshape(-1)
    )

    if (
        a.numel() < 2
        or b.numel() < 2
        or a.numel() != b.numel()
    ):
        return float("nan")

    a = a - a.mean()
    b = b - b.mean()

    denominator = (
        torch.sqrt(
            (a * a).mean()
            * (b * b).mean()
        )
        + eps
    )

    return (
        (a * b).mean()
        / denominator
    ).item()


def expand_like(
    source,
    target,
):
    if source.shape[-2:] != target.shape[-2:]:
        source = F.interpolate(
            source,
            size=target.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    if source.shape[1] == target.shape[1]:
        return source

    if source.shape[1] == 1:
        return source.expand(
            -1,
            target.shape[1],
            -1,
            -1,
        )

    source = source.mean(
        dim=1,
        keepdim=True,
    )

    return source.expand(
        -1,
        target.shape[1],
        -1,
        -1,
    )


# ============================================================
# 13. METRIC HELPERS
# ============================================================

def calc_psnr(
    pred,
    target,
):
    pred = pred.clamp(
        0.0,
        1.0,
    )

    target = target.clamp(
        0.0,
        1.0,
    )

    mse = F.mse_loss(
        pred,
        target,
    ).item()

    if mse <= 1e-12:
        return 100.0

    return (
        10.0
        * math.log10(
            1.0 / mse
        )
    )


def create_gaussian_window(
    window_size,
    channel,
    device,
    dtype,
):
    sigma = 1.5

    coordinates = torch.arange(
        window_size,
        dtype=dtype,
        device=device,
    )

    coordinates = (
        coordinates
        - window_size // 2
    )

    gaussian = torch.exp(
        -(coordinates ** 2)
        / (2 * sigma ** 2)
    )

    gaussian = (
        gaussian
        / gaussian.sum()
    )

    window_2d = (
        gaussian[:, None]
        @ gaussian[None, :]
    )

    window_2d = (
        window_2d
        .unsqueeze(0)
        .unsqueeze(0)
    )

    return window_2d.expand(
        channel,
        1,
        window_size,
        window_size,
    ).contiguous()


def calc_ssim(
    pred,
    target,
    window_size=11,
):
    pred = pred.clamp(
        0.0,
        1.0,
    )

    target = target.clamp(
        0.0,
        1.0,
    )

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    channel = pred.shape[1]

    window = create_gaussian_window(
        window_size=window_size,
        channel=channel,
        device=pred.device,
        dtype=pred.dtype,
    )

    mu_pred = F.conv2d(
        pred,
        window,
        padding=window_size // 2,
        groups=channel,
    )

    mu_target = F.conv2d(
        target,
        window,
        padding=window_size // 2,
        groups=channel,
    )

    mu_pred_sq = mu_pred.pow(2)
    mu_target_sq = mu_target.pow(2)
    mu_pred_target = (
        mu_pred
        * mu_target
    )

    sigma_pred_sq = (
        F.conv2d(
            pred * pred,
            window,
            padding=window_size // 2,
            groups=channel,
        )
        - mu_pred_sq
    )

    sigma_target_sq = (
        F.conv2d(
            target * target,
            window,
            padding=window_size // 2,
            groups=channel,
        )
        - mu_target_sq
    )

    sigma_pred_target = (
        F.conv2d(
            pred * target,
            window,
            padding=window_size // 2,
            groups=channel,
        )
        - mu_pred_target
    )

    numerator = (
        (2 * mu_pred_target + c1)
        * (
            2 * sigma_pred_target
            + c2
        )
    )

    denominator = (
        (
            mu_pred_sq
            + mu_target_sq
            + c1
        )
        * (
            sigma_pred_sq
            + sigma_target_sq
            + c2
        )
        + 1e-8
    )

    return (
        numerator
        / denominator
    ).mean().item()


def calc_lpips(
    pred,
    target,
    resize=None,
):
    pred = pred.clamp(
        0.0,
        1.0,
    )

    target = target.clamp(
        0.0,
        1.0,
    )

    if (
        resize is not None
        and resize > 0
    ):
        pred = F.interpolate(
            pred,
            size=(resize, resize),
            mode="bilinear",
            align_corners=False,
        )

        target = F.interpolate(
            target,
            size=(resize, resize),
            mode="bilinear",
            align_corners=False,
        )

    pred = pred * 2.0 - 1.0
    target = target * 2.0 - 1.0

    return (
        lpips_fn(
            pred,
            target,
        )
        .mean()
        .item()
    )


# ============================================================
# 14. LOAD MODEL
# ============================================================

def load_best_model():
    checkpoint_path = (
        CKPT_DIR
        / "best.pth"
    )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "best.pth bulunamadı:\n"
            f"{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
    )

    checkpoint_base = infer_base_channels(
        checkpoint
    )

    model = RetinexTapetum(
        base=checkpoint_base,
        lambda_init=LAMBDA_INIT,
        lambda_max=LAMBDA_MAX,
    ).to(DEVICE)

    state_dict = checkpoint.get(
        "model",
        checkpoint,
    )

    try:
        model.load_state_dict(
            state_dict,
            strict=True,
        )
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint ile mevcut model.py mimarisi uyumlu değil."
        ) from exc

    model.eval()

    print("\n===== CHECKPOINT =====")
    print("Path             :", checkpoint_path)
    print("BASE_CHANNELS    :", checkpoint_base)
    print("Trainable params :", f"{count_params(model):,}")

    for field in [
        "epoch",
        "best_epoch",
        "best_metric",
        "best_score",
        "best_psnr",
        "best_ssim",
        "best_lpips",
        "best_lpips_full",
    ]:
        if field in checkpoint:
            print(
                f"{field:18}:",
                checkpoint[field],
            )

    return (
        model,
        checkpoint,
        checkpoint_path,
    )


# ============================================================
# 15. ABLATION FORWARD VARIANTS
# ============================================================

@torch.inference_mode()
def forward_variant(
    model,
    low,
    variant,
):
    output = model(
        low
    )

    reflectance = output[
        "reflectance_low"
    ]

    illumination = output[
        "illumination_low"
    ]

    attention = output[
        "tapetum_attention"
    ]

    frequency_high = output[
        "frequency_high"
    ]

    darkness = output[
        "dark_prior"
    ]

    lambda_map = output[
        "lambda_map"
    ]

    base_enhanced = output[
        "base_enhanced"
    ]

    if variant == "full":
        prediction = output[
            "enhanced"
        ]

    elif variant == "retinex_base_only":
        prediction = torch.clamp(
            reflectance
            * illumination,
            0.0,
            1.0,
        )

    elif variant == "without_residual_refinement":
        prediction = torch.clamp(
            base_enhanced,
            0.0,
            1.0,
        )

    elif variant == "without_tapetum_update":
        illumination_new = illumination

        base_new = torch.clamp(
            reflectance
            * illumination_new,
            0.0,
            1.0,
        )

        refine_input = torch.cat(
            [
                low,
                base_new,
                attention,
                lambda_map,
                frequency_high,
            ],
            dim=1,
        )

        residual = model.refine_net(
            refine_input
        )

        prediction = torch.clamp(
            base_new
            + residual,
            0.0,
            1.0,
        )

    elif variant == "without_attention_modulation":
        attention_new = torch.ones_like(
            attention
        )

        illumination_new = (
            illumination
            * (
                1.0
                + lambda_map
                * attention_new
            )
        )

        base_new = (
            reflectance
            * illumination_new
        )

        refine_input = torch.cat(
            [
                low,
                base_new,
                attention_new,
                lambda_map,
                frequency_high,
            ],
            dim=1,
        )

        residual = model.refine_net(
            refine_input
        )

        prediction = torch.clamp(
            base_new
            + residual,
            0.0,
            1.0,
        )

    elif variant == "without_spatial_amplification":
        lambda_new = torch.zeros_like(
            lambda_map
        )

        illumination_new = (
            illumination
            * (
                1.0
                + lambda_new
                * attention
            )
        )

        base_new = (
            reflectance
            * illumination_new
        )

        refine_input = torch.cat(
            [
                low,
                base_new,
                attention,
                lambda_new,
                frequency_high,
            ],
            dim=1,
        )

        residual = model.refine_net(
            refine_input
        )

        prediction = torch.clamp(
            base_new
            + residual,
            0.0,
            1.0,
        )

    elif variant == "without_high_frequency_cue":
        frequency_new = torch.zeros_like(
            frequency_high
        )

        refine_input = torch.cat(
            [
                low,
                base_enhanced,
                attention,
                lambda_map,
                frequency_new,
            ],
            dim=1,
        )

        residual = model.refine_net(
            refine_input
        )

        prediction = torch.clamp(
            base_enhanced
            + residual,
            0.0,
            1.0,
        )

    elif variant == "without_darkness_gate":
        lambda_input = torch.cat(
            [
                illumination,
                darkness,
            ],
            dim=1,
        )

        lambda_no_gate = (
            model.lambda_map_net.lambda_max
            * model.lambda_map_net.net(
                lambda_input
            )
        )

        lambda_no_gate = torch.clamp(
            lambda_no_gate,
            0.0,
            model.lambda_map_net.lambda_max,
        )

        illumination_new = (
            illumination
            * (
                1.0
                + lambda_no_gate
                * attention
            )
        )

        base_new = (
            reflectance
            * illumination_new
        )

        refine_input = torch.cat(
            [
                low,
                base_new,
                attention,
                lambda_no_gate,
                frequency_high,
            ],
            dim=1,
        )

        residual = model.refine_net(
            refine_input
        )

        prediction = torch.clamp(
            base_new
            + residual,
            0.0,
            1.0,
        )

    else:
        raise ValueError(
            f"Unknown ablation variant: {variant}"
        )

    return prediction.clamp(
        0.0,
        1.0,
    )


# ============================================================
# 16. ABLATION EVALUATION
# ============================================================

@torch.inference_mode()
def evaluate_ablation(
    model,
    variants,
):
    to_tensor = transforms.ToTensor()

    low_files = list_images(
        TEST_LOW_DIR
    )

    high_files = set(
        list_images(
            TEST_HIGH_DIR
        )
    )

    matched_files = [
        filename
        for filename in low_files
        if filename in high_files
    ]

    if not matched_files:
        raise RuntimeError(
            "Test Low ve High klasörleri arasında "
            "eşleşen görüntü bulunamadı."
        )

    metric_store = {
        variant: {
            "psnr": [],
            "ssim": [],
            "lpips": [],
            "lpips_resize": [],
        }
        for variant in variants
    }

    detail_rows = []

    print("\n===== ABLATION START =====")
    print("Matched images:", len(matched_files))

    for filename in tqdm(
        matched_files,
        desc="Ablation",
    ):
        low_path = (
            TEST_LOW_DIR
            / filename
        )

        high_path = (
            TEST_HIGH_DIR
            / filename
        )

        low_image = Image.open(
            low_path
        ).convert("RGB")

        high_image = Image.open(
            high_path
        ).convert("RGB")

        if low_image.size != high_image.size:
            raise ValueError(
                "Low ve High görüntü boyutları farklı:\n"
                f"{filename}"
            )

        low = (
            to_tensor(low_image)
            .unsqueeze(0)
            .to(DEVICE)
        )

        high = (
            to_tensor(high_image)
            .unsqueeze(0)
            .to(DEVICE)
        )

        for variant in variants:
            prediction = forward_variant(
                model,
                low,
                variant,
            )

            psnr_value = calc_psnr(
                prediction,
                high,
            )

            ssim_value = calc_ssim(
                prediction,
                high,
            )

            lpips_value = calc_lpips(
                prediction,
                high,
                resize=None,
            )

            lpips_resize_value = calc_lpips(
                prediction,
                high,
                resize=LPIPS_LOSS_RESIZE,
            )

            metric_store[
                variant
            ][
                "psnr"
            ].append(
                psnr_value
            )

            metric_store[
                variant
            ][
                "ssim"
            ].append(
                ssim_value
            )

            metric_store[
                variant
            ][
                "lpips"
            ].append(
                lpips_value
            )

            metric_store[
                variant
            ][
                "lpips_resize"
            ].append(
                lpips_resize_value
            )

            detail_rows.append(
                {
                    "dataset": config.DATA_NAME,
                    "dataset_variant": config.DATA_VARIANT,
                    "file": filename,
                    "ablation_variant": variant,
                    "psnr": psnr_value,
                    "ssim": ssim_value,
                    "lpips": lpips_value,
                    "lpips_resize": lpips_resize_value,
                }
            )

    summary_rows = []

    for variant in variants:
        count = len(
            metric_store[
                variant
            ][
                "psnr"
            ]
        )

        if count == 0:
            raise RuntimeError(
                f"{variant} için sonuç üretilemedi."
            )

        summary_rows.append(
            {
                "dataset": config.DATA_NAME,
                "dataset_variant": config.DATA_VARIANT,
                "ablation_variant": variant,
                "matched": count,
                "psnr": finite_mean(
                    metric_store[
                        variant
                    ][
                        "psnr"
                    ]
                ),
                "ssim": finite_mean(
                    metric_store[
                        variant
                    ][
                        "ssim"
                    ]
                ),
                "lpips": finite_mean(
                    metric_store[
                        variant
                    ][
                        "lpips"
                    ]
                ),
                "lpips_resize": finite_mean(
                    metric_store[
                        variant
                    ][
                        "lpips_resize"
                    ]
                ),
            }
        )

    return (
        summary_rows,
        detail_rows,
    )


# ============================================================
# 17. INTERNAL DIAGNOSTICS
# ============================================================

@torch.inference_mode()
def evaluate_internal_diagnostics(
    model,
):
    to_tensor = transforms.ToTensor()

    low_files = list_images(
        TEST_LOW_DIR
    )

    high_files = set(
        list_images(
            TEST_HIGH_DIR
        )
    )

    matched_files = [
        filename
        for filename in low_files
        if filename in high_files
    ]

    if not matched_files:
        raise RuntimeError(
            "Internal diagnostics için eşleşen test görüntüsü bulunamadı."
        )

    detail_rows = []

    summary_store = {
        "T_mean": [],
        "T_median": [],
        "Td_mean": [],
        "Td_median": [],
        "Td_dark_mean": [],
        "Td_dark_median": [],
        "Td_bright_mean": [],
        "Td_bright_median": [],
        "lambda_dark_mean": [],
        "lambda_dark_median": [],
        "lambda_bright_mean": [],
        "lambda_bright_median": [],
        "boost_dark_mean": [],
        "boost_dark_median": [],
        "boost_bright_mean": [],
        "boost_bright_median": [],
        "corr_Td_darkness": [],
        "corr_boost_darkness": [],
        "gain_mean": [],
        "gain_median": [],
        "gain_dark_mean": [],
        "gain_bright_mean": [],
    }

    print("\n===== INTERNAL DIAGNOSTICS START =====")
    print("Matched images  :", len(matched_files))
    print("DARK_THRESHOLD :", DARK_THRESHOLD)
    print("BRIGHT_THRESHOLD:", BRIGHT_THRESHOLD)

    for filename in tqdm(
        matched_files,
        desc="Diagnostics",
    ):
        low_path = (
            TEST_LOW_DIR
            / filename
        )

        low = (
            to_tensor(
                Image.open(
                    low_path
                ).convert("RGB")
            )
            .unsqueeze(0)
            .to(DEVICE)
        )

        output = model(
            low
        )

        attention = (
            output[
                "tapetum_attention"
            ]
            .detach()
            .float()
        )

        darkness = (
            output[
                "dark_prior"
            ]
            .detach()
            .float()
        )

        lambda_map = (
            output[
                "lambda_map"
            ]
            .detach()
            .float()
        )

        darkness_for_attention = expand_like(
            darkness,
            attention,
        )

        darkness_for_lambda = expand_like(
            darkness,
            lambda_map,
        )

        darkness_response = (
            darkness_for_attention
            * attention
        )

        boost = (
            lambda_map
            * attention
        )

        gain = (
            1.0
            + boost
        )

        darkness_for_boost = expand_like(
            darkness,
            boost,
        )

        darkness_for_gain = expand_like(
            darkness,
            gain,
        )

        dark_attention_mask = (
            darkness_for_attention
            >= DARK_THRESHOLD
        )

        bright_attention_mask = (
            darkness_for_attention
            < BRIGHT_THRESHOLD
        )

        dark_lambda_mask = (
            darkness_for_lambda
            >= DARK_THRESHOLD
        )

        bright_lambda_mask = (
            darkness_for_lambda
            < BRIGHT_THRESHOLD
        )

        dark_boost_mask = (
            darkness_for_boost
            >= DARK_THRESHOLD
        )

        bright_boost_mask = (
            darkness_for_boost
            < BRIGHT_THRESHOLD
        )

        dark_gain_mask = (
            darkness_for_gain
            >= DARK_THRESHOLD
        )

        bright_gain_mask = (
            darkness_for_gain
            < BRIGHT_THRESHOLD
        )

        values = {
            "T_mean": safe_mean_tensor(
                attention
            ),

            "T_median": safe_median_tensor(
                attention
            ),

            "Td_mean": safe_mean_tensor(
                darkness_response
            ),

            "Td_median": safe_median_tensor(
                darkness_response
            ),

            "Td_dark_mean": (
                safe_mean_tensor(
                    darkness_response[
                        dark_attention_mask
                    ]
                )
                if dark_attention_mask.any()
                else float("nan")
            ),

            "Td_dark_median": (
                safe_median_tensor(
                    darkness_response[
                        dark_attention_mask
                    ]
                )
                if dark_attention_mask.any()
                else float("nan")
            ),

            "Td_bright_mean": (
                safe_mean_tensor(
                    darkness_response[
                        bright_attention_mask
                    ]
                )
                if bright_attention_mask.any()
                else float("nan")
            ),

            "Td_bright_median": (
                safe_median_tensor(
                    darkness_response[
                        bright_attention_mask
                    ]
                )
                if bright_attention_mask.any()
                else float("nan")
            ),

            "lambda_dark_mean": (
                safe_mean_tensor(
                    lambda_map[
                        dark_lambda_mask
                    ]
                )
                if dark_lambda_mask.any()
                else float("nan")
            ),

            "lambda_dark_median": (
                safe_median_tensor(
                    lambda_map[
                        dark_lambda_mask
                    ]
                )
                if dark_lambda_mask.any()
                else float("nan")
            ),

            "lambda_bright_mean": (
                safe_mean_tensor(
                    lambda_map[
                        bright_lambda_mask
                    ]
                )
                if bright_lambda_mask.any()
                else float("nan")
            ),

            "lambda_bright_median": (
                safe_median_tensor(
                    lambda_map[
                        bright_lambda_mask
                    ]
                )
                if bright_lambda_mask.any()
                else float("nan")
            ),

            "boost_dark_mean": (
                safe_mean_tensor(
                    boost[
                        dark_boost_mask
                    ]
                )
                if dark_boost_mask.any()
                else float("nan")
            ),

            "boost_dark_median": (
                safe_median_tensor(
                    boost[
                        dark_boost_mask
                    ]
                )
                if dark_boost_mask.any()
                else float("nan")
            ),

            "boost_bright_mean": (
                safe_mean_tensor(
                    boost[
                        bright_boost_mask
                    ]
                )
                if bright_boost_mask.any()
                else float("nan")
            ),

            "boost_bright_median": (
                safe_median_tensor(
                    boost[
                        bright_boost_mask
                    ]
                )
                if bright_boost_mask.any()
                else float("nan")
            ),

            "corr_Td_darkness": pearson_corr_torch(
                darkness_response,
                darkness_for_attention,
            ),

            "corr_boost_darkness": pearson_corr_torch(
                boost,
                darkness_for_boost,
            ),

            "gain_mean": safe_mean_tensor(
                gain
            ),

            "gain_median": safe_median_tensor(
                gain
            ),

            "gain_dark_mean": (
                safe_mean_tensor(
                    gain[
                        dark_gain_mask
                    ]
                )
                if dark_gain_mask.any()
                else float("nan")
            ),

            "gain_bright_mean": (
                safe_mean_tensor(
                    gain[
                        bright_gain_mask
                    ]
                )
                if bright_gain_mask.any()
                else float("nan")
            ),
        }

        for key, value in values.items():
            summary_store[
                key
            ].append(
                value
            )

        detail_rows.append(
            {
                "dataset": config.DATA_NAME,
                "dataset_variant": config.DATA_VARIANT,
                "filename": filename,
                "dark_pixels": int(
                    (
                        darkness
                        >= DARK_THRESHOLD
                    )
                    .sum()
                    .detach()
                    .cpu()
                ),
                "bright_pixels": int(
                    (
                        darkness
                        < BRIGHT_THRESHOLD
                    )
                    .sum()
                    .detach()
                    .cpu()
                ),
                **values,
            }
        )

    table_rows = [
        {
            "quantity": "Sigmoid tapetum response T(x)",
            "mean": finite_mean(
                summary_store[
                    "T_mean"
                ]
            ),
            "median": finite_median(
                summary_store[
                    "T_median"
                ]
            ),
        },
        {
            "quantity": "Darkness-regulated response Td(x)=D(x)*T(x)",
            "mean": finite_mean(
                summary_store[
                    "Td_mean"
                ]
            ),
            "median": finite_median(
                summary_store[
                    "Td_median"
                ]
            ),
        },
        {
            "quantity": "Td(x) in dark regions",
            "mean": finite_mean(
                summary_store[
                    "Td_dark_mean"
                ]
            ),
            "median": finite_median(
                summary_store[
                    "Td_dark_median"
                ]
            ),
        },
        {
            "quantity": "Td(x) in bright regions",
            "mean": finite_mean(
                summary_store[
                    "Td_bright_mean"
                ]
            ),
            "median": finite_median(
                summary_store[
                    "Td_bright_median"
                ]
            ),
        },
        {
            "quantity": "Lambda(x) in dark regions",
            "mean": finite_mean(
                summary_store[
                    "lambda_dark_mean"
                ]
            ),
            "median": finite_median(
                summary_store[
                    "lambda_dark_median"
                ]
            ),
        },
        {
            "quantity": "Lambda(x) in bright regions",
            "mean": finite_mean(
                summary_store[
                    "lambda_bright_mean"
                ]
            ),
            "median": finite_median(
                summary_store[
                    "lambda_bright_median"
                ]
            ),
        },
        {
            "quantity": "Effective boost Lambda(x)*T(x) in dark regions",
            "mean": finite_mean(
                summary_store[
                    "boost_dark_mean"
                ]
            ),
            "median": finite_median(
                summary_store[
                    "boost_dark_median"
                ]
            ),
        },
        {
            "quantity": "Effective boost Lambda(x)*T(x) in bright regions",
            "mean": finite_mean(
                summary_store[
                    "boost_bright_mean"
                ]
            ),
            "median": finite_median(
                summary_store[
                    "boost_bright_median"
                ]
            ),
        },
        {
            "quantity": "Corr.(Td(x), darkness)",
            "mean": finite_mean(
                summary_store[
                    "corr_Td_darkness"
                ]
            ),
            "median": finite_median(
                summary_store[
                    "corr_Td_darkness"
                ]
            ),
        },
        {
            "quantity": "Corr.(Lambda(x)*T(x), darkness)",
            "mean": finite_mean(
                summary_store[
                    "corr_boost_darkness"
                ]
            ),
            "median": finite_median(
                summary_store[
                    "corr_boost_darkness"
                ]
            ),
        },
    ]

    gain_summary = {
        "dataset": config.DATA_NAME,
        "dataset_variant": config.DATA_VARIANT,
        "matched": len(detail_rows),
        "dark_threshold": DARK_THRESHOLD,
        "bright_threshold": BRIGHT_THRESHOLD,
        "gain_mean": finite_mean(
            summary_store[
                "gain_mean"
            ]
        ),
        "gain_median": finite_median(
            summary_store[
                "gain_median"
            ]
        ),
        "gain_dark_mean": finite_mean(
            summary_store[
                "gain_dark_mean"
            ]
        ),
        "gain_bright_mean": finite_mean(
            summary_store[
                "gain_bright_mean"
            ]
        ),
        "dark_pixels_total": sum(
            row[
                "dark_pixels"
            ]
            for row in detail_rows
        ),
        "bright_pixels_total": sum(
            row[
                "bright_pixels"
            ]
            for row in detail_rows
        ),
    }

    return (
        table_rows,
        detail_rows,
        gain_summary,
    )


# ============================================================
# 18. VISUAL HELPERS
# ============================================================

def tensor_to_rgb(x):
    if x.dim() == 4:
        x = x[0]

    if (
        x.dim() == 3
        and x.shape[0] == 1
    ):
        x = x.repeat(
            3,
            1,
            1,
        )

    x = (
        x.detach()
        .float()
        .cpu()
        .clamp(0, 1)
    )

    return (
        x.permute(
            1,
            2,
            0,
        )
        .numpy()
    )


def tensor_to_gray(
    x,
    percentile_clip=(1, 99),
):
    if x.dim() == 4:
        x = x[0, 0]
    elif x.dim() == 3:
        x = x[0]

    array = (
        x.detach()
        .float()
        .cpu()
        .numpy()
    )

    low_value = np.percentile(
        array,
        percentile_clip[0],
    )

    high_value = np.percentile(
        array,
        percentile_clip[1],
    )

    if (
        high_value
        - low_value
        < 1e-8
    ):
        return np.zeros_like(
            array
        )

    return np.clip(
        (
            array
            - low_value
        )
        / (
            high_value
            - low_value
        ),
        0,
        1,
    )


def tensor_to_map(x):
    if x.dim() == 4:
        x = x[0]

    if x.dim() == 2:
        x = x.unsqueeze(0)

    array = (
        x.detach()
        .float()
        .cpu()
        .numpy()
    )

    low_value = np.percentile(
        array,
        1,
    )

    high_value = np.percentile(
        array,
        99,
    )

    if (
        high_value
        - low_value
        < 1e-8
    ):
        array = np.zeros_like(
            array
        )
    else:
        array = np.clip(
            (
                array
                - low_value
            )
            / (
                high_value
                - low_value
            ),
            0,
            1,
        )

    if array.shape[0] == 1:
        array = np.repeat(
            array,
            3,
            axis=0,
        )

    return np.transpose(
        array,
        (
            1,
            2,
            0,
        ),
    )


def tensor_to_diverging(
    x,
    vmax_percentile=99,
):
    if x.dim() == 4:
        x = x[0]

    if (
        x.dim() == 3
        and x.shape[0] == 3
    ):
        x = (
            0.299 * x[0]
            + 0.587 * x[1]
            + 0.114 * x[2]
        )
    elif x.dim() == 3:
        x = x[0]

    array = (
        x.detach()
        .float()
        .cpu()
        .numpy()
    )

    vmax = np.percentile(
        np.abs(array),
        vmax_percentile,
    )

    vmax = max(
        vmax,
        1e-6,
    )

    return np.clip(
        array / vmax,
        -1,
        1,
    )


def choose_sample_filename():
    low_files = list_images(
        TEST_LOW_DIR
    )

    high_files = set(
        list_images(
            TEST_HIGH_DIR
        )
    )

    matched_files = [
        filename
        for filename in low_files
        if filename in high_files
    ]

    if not matched_files:
        raise RuntimeError(
            "Görsel tanı için eşleşen görüntü bulunamadı."
        )

    if SAMPLE_FILENAME is not None:
        if SAMPLE_FILENAME not in matched_files:
            raise ValueError(
                f"{SAMPLE_FILENAME} test setinde bulunamadı."
            )

        return SAMPLE_FILENAME

    selected_index = max(
        0,
        min(
            SAMPLE_INDEX,
            len(matched_files) - 1,
        ),
    )

    return matched_files[
        selected_index
    ]


# ============================================================
# 19. VISUAL DIAGNOSTICS
# ============================================================

@torch.inference_mode()
def create_visual_diagnostics(
    model,
):
    sample_name = choose_sample_filename()

    low_path = (
        TEST_LOW_DIR
        / sample_name
    )

    high_path = (
        TEST_HIGH_DIR
        / sample_name
    )

    low = load_image_as_tensor(
        low_path
    )

    high = load_image_as_tensor(
        high_path
    )

    output = model(
        low
    )

    reflectance = output[
        "reflectance_low"
    ]

    illumination = output[
        "illumination_low"
    ]

    darkness = output[
        "dark_prior"
    ]

    attention = output[
        "tapetum_attention"
    ]

    lambda_map = output[
        "lambda_map"
    ]

    illumination_t = output[
        "illumination_t"
    ]

    base_enhanced = output[
        "base_enhanced"
    ]

    frequency_high = output[
        "frequency_high"
    ]

    residual = output[
        "residual"
    ]

    enhanced = output[
        "enhanced"
    ]

    boost = (
        lambda_map
        * attention
    )

    panels = [
        (
            "Input",
            tensor_to_rgb(
                low
            ),
            "rgb",
        ),
        (
            "Reflectance",
            tensor_to_rgb(
                reflectance
            ),
            "rgb",
        ),
        (
            "Illumination",
            tensor_to_rgb(
                illumination
            ),
            "rgb",
        ),
        (
            "Dark prior",
            tensor_to_gray(
                darkness
            ),
            "gray",
        ),
        (
            "Attention",
            tensor_to_map(
                attention
            ),
            "rgb",
        ),
        (
            "Lambda",
            tensor_to_map(
                lambda_map
            ),
            "rgb",
        ),
        (
            "Lambda x T",
            tensor_to_map(
                boost
            ),
            "rgb",
        ),
        (
            "Updated illumination",
            tensor_to_rgb(
                illumination_t
            ),
            "rgb",
        ),
        (
            "Base enhanced",
            tensor_to_rgb(
                base_enhanced
            ),
            "rgb",
        ),
        (
            "High frequency",
            tensor_to_diverging(
                frequency_high
            ),
            "diverging",
        ),
        (
            "Residual",
            tensor_to_diverging(
                residual
            ),
            "diverging",
        ),
        (
            "Enhanced",
            tensor_to_rgb(
                enhanced
            ),
            "rgb",
        ),
        (
            "Ground truth",
            tensor_to_rgb(
                high
            ),
            "rgb",
        ),
    ]

    columns = 4
    rows = math.ceil(
        len(panels)
        / columns
    )

    figure = plt.figure(
        figsize=(
            4.2 * columns,
            3.6 * rows,
        )
    )

    for index, (
        title,
        image,
        mode,
    ) in enumerate(
        panels,
        start=1,
    ):
        axis = plt.subplot(
            rows,
            columns,
            index,
        )

        if mode == "rgb":
            axis.imshow(
                image
            )
        elif mode == "gray":
            axis.imshow(
                image,
                cmap="gray",
                vmin=0,
                vmax=1,
            )
        else:
            axis.imshow(
                image,
                cmap="seismic",
                vmin=-1,
                vmax=1,
            )

        axis.set_title(
            title,
            fontsize=10,
        )

        axis.axis("off")

    figure.suptitle(
        (
            "RetinexTapetum Internal Diagnostics — "
            f"{sample_name}"
        ),
        fontsize=14,
    )

    plt.tight_layout(
        rect=[
            0,
            0,
            1,
            0.97,
        ]
    )

    internal_path = (
        VISUAL_DIR
        / (
            f"{Path(sample_name).stem}"
            "_internal_diagnostics.png"
        )
    )

    plt.savefig(
        internal_path,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )

    plt.show()
    plt.close(
        figure
    )

    comparison_variants = [
        (
            "Input",
            low,
        ),
        (
            "Full",
            forward_variant(
                model,
                low,
                "full",
            ),
        ),
        (
            "w/o tapetum",
            forward_variant(
                model,
                low,
                "without_tapetum_update",
            ),
        ),
        (
            "w/o lambda",
            forward_variant(
                model,
                low,
                "without_spatial_amplification",
            ),
        ),
        (
            "w/o dark gate",
            forward_variant(
                model,
                low,
                "without_darkness_gate",
            ),
        ),
        (
            "w/o residual",
            forward_variant(
                model,
                low,
                "without_residual_refinement",
            ),
        ),
        (
            "GT",
            high,
        ),
    ]

    figure = plt.figure(
        figsize=(
            3.6
            * len(
                comparison_variants
            ),
            4.2,
        )
    )

    for index, (
        title,
        image_tensor,
    ) in enumerate(
        comparison_variants,
        start=1,
    ):
        axis = plt.subplot(
            1,
            len(
                comparison_variants
            ),
            index,
        )

        axis.imshow(
            tensor_to_rgb(
                image_tensor
            )
        )

        axis.set_title(
            title,
            fontsize=10,
        )

        axis.axis("off")

    figure.suptitle(
        (
            "RetinexTapetum Ablation Comparison — "
            f"{sample_name}"
        ),
        fontsize=14,
    )

    plt.tight_layout(
        rect=[
            0,
            0,
            1,
            0.92,
        ]
    )

    comparison_path = (
        VISUAL_DIR
        / (
            f"{Path(sample_name).stem}"
            "_ablation_comparison.png"
        )
    )

    plt.savefig(
        comparison_path,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )

    plt.show()
    plt.close(
        figure
    )

    full = forward_variant(
        model,
        low,
        "full",
    )

    effect_variants = [
        (
            "Full - w/o tapetum",
            full
            - forward_variant(
                model,
                low,
                "without_tapetum_update",
            ),
        ),
        (
            "Full - w/o lambda",
            full
            - forward_variant(
                model,
                low,
                "without_spatial_amplification",
            ),
        ),
        (
            "Full - w/o dark gate",
            full
            - forward_variant(
                model,
                low,
                "without_darkness_gate",
            ),
        ),
        (
            "Full - w/o residual",
            full
            - forward_variant(
                model,
                low,
                "without_residual_refinement",
            ),
        ),
    ]

    figure = plt.figure(
        figsize=(
            4.5
            * len(
                effect_variants
            ),
            4.5,
        )
    )

    for index, (
        title,
        difference,
    ) in enumerate(
        effect_variants,
        start=1,
    ):
        axis = plt.subplot(
            1,
            len(
                effect_variants
            ),
            index,
        )

        axis.imshow(
            tensor_to_diverging(
                difference
            ),
            cmap="seismic",
            vmin=-1,
            vmax=1,
        )

        axis.set_title(
            title,
            fontsize=10,
        )

        axis.axis("off")

    figure.suptitle(
        (
            "Effect Maps Relative to Full Model — "
            f"{sample_name}"
        ),
        fontsize=14,
    )

    plt.tight_layout(
        rect=[
            0,
            0,
            1,
            0.92,
        ]
    )

    effect_path = (
        VISUAL_DIR
        / (
            f"{Path(sample_name).stem}"
            "_effect_maps.png"
        )
    )

    plt.savefig(
        effect_path,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )

    plt.show()
    plt.close(
        figure
    )

    return {
        "sample_name": sample_name,
        "internal_figure": internal_path,
        "ablation_figure": comparison_path,
        "effect_figure": effect_path,
    }


# ============================================================
# 20. RUN ANALYSIS
# ============================================================

model, checkpoint, checkpoint_path = load_best_model()

ABLATION_VARIANTS = [
    "full",
    "retinex_base_only",
    "without_residual_refinement",
    "without_tapetum_update",
    "without_attention_modulation",
    "without_spatial_amplification",
    "without_high_frequency_cue",
    "without_darkness_gate",
]

ablation_summary_rows, ablation_detail_rows = (
    evaluate_ablation(
        model,
        ABLATION_VARIANTS,
    )
)

diagnostic_table_rows, diagnostic_detail_rows, gain_summary = (
    evaluate_internal_diagnostics(
        model
    )
)

visual_outputs = create_visual_diagnostics(
    model
)


# ============================================================
# 21. SAVE ABLATION RESULTS
# ============================================================

with open(
    ABLATION_SUMMARY_PATH,
    "w",
    newline="",
    encoding="utf-8",
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=list(
            ablation_summary_rows[
                0
            ].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(
        ablation_summary_rows
    )

with open(
    ABLATION_DETAIL_PATH,
    "w",
    newline="",
    encoding="utf-8",
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=list(
            ablation_detail_rows[
                0
            ].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(
        ablation_detail_rows
    )


ablation_name_map = {
    "full": "Full RetinexTapetum",
    "retinex_base_only": "Retinex base only",
    "without_residual_refinement": "w/o residual refinement",
    "without_tapetum_update": "w/o tapetum update",
    "without_attention_modulation": "w/o attention modulation",
    "without_spatial_amplification": "w/o spatial amplification",
    "without_high_frequency_cue": "w/o high-frequency cue",
    "without_darkness_gate": "w/o darkness gate",
}

ablation_latex_rows = [
    (
        f"{ablation_name_map[row['ablation_variant']]}"
        f" & {row['psnr']:.2f}"
        f" & {row['ssim']:.3f}"
        f" & {row['lpips']:.3f} \\\\"
    )
    for row in ablation_summary_rows
]

with open(
    ABLATION_LATEX_PATH,
    "w",
    encoding="utf-8",
) as file:
    file.write(
        "\n".join(
            ablation_latex_rows
        )
    )


# ============================================================
# 22. SAVE DIAGNOSTIC RESULTS
# ============================================================

with open(
    DIAGNOSTICS_SUMMARY_PATH,
    "w",
    newline="",
    encoding="utf-8",
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "quantity",
            "mean",
            "median",
        ],
    )

    writer.writeheader()
    writer.writerows(
        diagnostic_table_rows
    )

with open(
    DIAGNOSTICS_DETAIL_PATH,
    "w",
    newline="",
    encoding="utf-8",
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=list(
            diagnostic_detail_rows[
                0
            ].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(
        diagnostic_detail_rows
    )

with open(
    DIAGNOSTICS_GAIN_PATH,
    "w",
    newline="",
    encoding="utf-8",
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=list(
            gain_summary.keys()
        ),
    )

    writer.writeheader()
    writer.writerow(
        gain_summary
    )

diagnostics_latex_rows = [
    (
        f"{row['quantity']}"
        f" & {row['mean']:.4f}"
        f" & {row['median']:.4f} \\\\"
    )
    for row in diagnostic_table_rows
]

with open(
    DIAGNOSTICS_LATEX_PATH,
    "w",
    encoding="utf-8",
) as file:
    file.write(
        "\n".join(
            diagnostics_latex_rows
        )
    )


# ============================================================
# 23. SAVE METADATA
# ============================================================

metadata_lines = [
    f"dataset={config.DATA_NAME}",
    f"dataset_variant={config.DATA_VARIANT}",
    f"data_root={config.DATA_ROOT}",
    f"test_low_dir={TEST_LOW_DIR}",
    f"test_high_dir={TEST_HIGH_DIR}",
    f"run_root={RUN_ROOT}",
    f"checkpoint={checkpoint_path}",
    f"analysis_root={ANALYSIS_ROOT}",
    f"ablation_dir={ABLATION_DIR}",
    f"diagnostics_dir={DIAGNOSTICS_DIR}",
    f"visual_dir={VISUAL_DIR}",
    f"visual_sample={visual_outputs['sample_name']}",
    f"device={DEVICE}",
    f"base_channels={infer_base_channels(checkpoint)}",
    f"parameters={count_params(model)}",
    f"lambda_init={LAMBDA_INIT}",
    f"lambda_max={LAMBDA_MAX}",
    f"lpips_resize={LPIPS_LOSS_RESIZE}",
    f"dark_threshold={DARK_THRESHOLD}",
    f"bright_threshold={BRIGHT_THRESHOLD}",
    f"gain_mean={gain_summary['gain_mean']}",
    f"gain_median={gain_summary['gain_median']}",
    f"gain_dark_mean={gain_summary['gain_dark_mean']}",
    f"gain_bright_mean={gain_summary['gain_bright_mean']}",
]

for field in [
    "epoch",
    "best_epoch",
    "best_metric",
    "best_score",
    "best_psnr",
    "best_ssim",
    "best_lpips",
    "best_lpips_full",
]:
    if field in checkpoint:
        metadata_lines.append(
            f"{field}={checkpoint[field]}"
        )

with open(
    METADATA_PATH,
    "w",
    encoding="utf-8",
) as file:
    file.write(
        "\n".join(
            metadata_lines
        )
    )


# ============================================================
# 24. PRINT AND DISPLAY RESULTS
# ============================================================

print("\n===== ABLATION SUMMARY =====")

for row in ablation_summary_rows:
    print(
        f"{row['ablation_variant']:35s} | "
        f"PSNR {row['psnr']:.4f} | "
        f"SSIM {row['ssim']:.4f} | "
        f"LPIPS {row['lpips']:.4f}"
    )

print("\n===== INTERNAL DIAGNOSTICS SUMMARY =====")

for row in diagnostic_table_rows:
    print(
        f"{row['quantity']:<60} | "
        f"Mean {row['mean']:.4f} | "
        f"Median {row['median']:.4f}"
    )

print("\n===== GAIN SUMMARY =====")

for key, value in gain_summary.items():
    if isinstance(
        value,
        float,
    ):
        print(
            f"{key:30s}: {value:.4f}"
        )
    else:
        print(
            f"{key:30s}: {value}"
        )


ablation_summary_df = pd.read_csv(
    ABLATION_SUMMARY_PATH
)

diagnostics_summary_df = pd.read_csv(
    DIAGNOSTICS_SUMMARY_PATH
)

gain_summary_df = pd.read_csv(
    DIAGNOSTICS_GAIN_PATH
)

print("\n===== ABLATION TABLE =====")
display(
    ablation_summary_df
)

print("\n===== DIAGNOSTICS TABLE =====")
display(
    diagnostics_summary_df
)

print("\n===== GAIN TABLE =====")
display(
    gain_summary_df
)


# ============================================================
# 25. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 90)
print("RETINEXTAPETUM ANALYSIS COMPLETED")
print("=" * 90)

print("Dataset               :", config.DATA_NAME)
print("Dataset variant       :", config.DATA_VARIANT)
print("Run root              :", RUN_ROOT)
print("Analysis root         :", ANALYSIS_ROOT)

print("\nAblation:")
print("Summary CSV           :", ABLATION_SUMMARY_PATH)
print("Detail CSV            :", ABLATION_DETAIL_PATH)
print("LaTeX                 :", ABLATION_LATEX_PATH)

print("\nDiagnostics:")
print("Summary CSV           :", DIAGNOSTICS_SUMMARY_PATH)
print("Detail CSV            :", DIAGNOSTICS_DETAIL_PATH)
print("Gain CSV              :", DIAGNOSTICS_GAIN_PATH)
print("LaTeX                 :", DIAGNOSTICS_LATEX_PATH)

print("\nVisuals:")
print("Internal diagnostics  :", visual_outputs["internal_figure"])
print("Ablation comparison   :", visual_outputs["ablation_figure"])
print("Effect maps           :", visual_outputs["effect_figure"])

print("\nMetadata:")
print("Metadata file         :", METADATA_PATH)

print("=" * 90)

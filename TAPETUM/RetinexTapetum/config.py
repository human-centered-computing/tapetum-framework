"""Configuration file for RetinexTapetum."""

import os
import torch


# -----------------------------------------------------------------------------
# Environment helpers
# -----------------------------------------------------------------------------

def env_to_bool(
    name: str,
    default: bool = False,
) -> bool:
    """Safely convert an environment variable to a boolean value."""

    value = os.environ.get(name)

    if value is None:
        return default

    normalized = str(value).strip().lower()

    if normalized in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }:
        return True

    if normalized in {
        "",
        "0",
        "false",
        "no",
        "n",
        "off",
        "none",
        "null",
    }:
        return False

    raise ValueError(
        f"Invalid boolean environment variable: "
        f"{name}={value!r}"
    )


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

PROJECT_ROOT = os.path.abspath(
    os.path.dirname(__file__)
)

WORKSPACE_ROOT = os.path.abspath(
    os.path.join(
        PROJECT_ROOT,
        os.pardir,
    )
)


# -----------------------------------------------------------------------------
# Dataset selection
# -----------------------------------------------------------------------------

DATA_NAME = os.environ.get(
    "RETINEX_DATA_NAME",
    "LOL-v2",
).strip()

_data_variant_env = os.environ.get(
    "RETINEX_DATA_VARIANT",
    "Real_captured",
)

DATA_VARIANT = (
    None
    if str(_data_variant_env).strip().lower()
    in {
        "",
        "none",
        "null",
    }
    else str(_data_variant_env).strip()
)

VALID_DATASET_SELECTIONS = {
    ("LOL-v1", None),
    ("LOL-v2", "Real_captured"),
    ("LOL-v2", "Synthetic"),
    ("UHD-LL down4", None),
    ("SICE", None),
    ("LoLI-Street", None),
    ("DCIM", None),
    ("LIME", None),
    ("MEF", None),
    ("NPE", None),
    ("VV", None),
}

SELECTED_DATASET = (
    DATA_NAME,
    DATA_VARIANT,
)

if SELECTED_DATASET not in VALID_DATASET_SELECTIONS:
    valid_text = "\n".join(
        f"  - DATA_NAME={name!r}, DATA_VARIANT={variant!r}"
        for name, variant in sorted(
            VALID_DATASET_SELECTIONS,
            key=lambda item: (
                item[0],
                str(item[1]),
            ),
        )
    )

    raise ValueError(
        "Invalid dataset and variant combination.\n"
        f"Selected: DATA_NAME={DATA_NAME!r}, "
        f"DATA_VARIANT={DATA_VARIANT!r}\n"
        f"Valid selections:\n{valid_text}"
    )


# -----------------------------------------------------------------------------
# Local dataset root
# -----------------------------------------------------------------------------

if DATA_VARIANT is None:
    LOCAL_DATA_ROOT = os.path.join(
        WORKSPACE_ROOT,
        "datasets",
        DATA_NAME,
    )
else:
    LOCAL_DATA_ROOT = os.path.join(
        WORKSPACE_ROOT,
        "datasets",
        DATA_NAME,
        DATA_VARIANT,
    )


# -----------------------------------------------------------------------------
# Google Drive dataset root
# -----------------------------------------------------------------------------

COLAB_BASE_DATA_ROOT = (
    "/content/drive/MyDrive/TAPETUM/datasets"
)

if DATA_VARIANT is None:
    COLAB_DATA_ROOT = os.path.join(
        COLAB_BASE_DATA_ROOT,
        DATA_NAME,
    )
else:
    COLAB_DATA_ROOT = os.path.join(
        COLAB_BASE_DATA_ROOT,
        DATA_NAME,
        DATA_VARIANT,
    )


# -----------------------------------------------------------------------------
# Final dataset root
# -----------------------------------------------------------------------------

DATA_ROOT = (
    COLAB_DATA_ROOT
    if os.path.exists(COLAB_DATA_ROOT)
    else LOCAL_DATA_ROOT
)


# -----------------------------------------------------------------------------
# Run root
# -----------------------------------------------------------------------------

if DATA_VARIANT is None:
    RUN_ROOT = os.path.join(
        WORKSPACE_ROOT,
        DATA_NAME,
        "RetinexTapetum",
    )
else:
    RUN_ROOT = os.path.join(
        WORKSPACE_ROOT,
        DATA_NAME,
        f"RetinexTapetum-{DATA_VARIANT}",
    )


# -----------------------------------------------------------------------------
# Train / validation / test paths
# -----------------------------------------------------------------------------

TRAIN_LOW_DIR = os.path.join(
    DATA_ROOT,
    "Train",
    "Low",
)

TRAIN_HIGH_DIR = os.path.join(
    DATA_ROOT,
    "Train",
    "Normal",
)

TEST_LOW_DIR = os.path.join(
    DATA_ROOT,
    "Test",
    "Low",
)

TEST_HIGH_DIR = os.path.join(
    DATA_ROOT,
    "Test",
    "Normal",
)

USE_TRAIN_VAL_SPLIT = True
VAL_RATIO = 0.10
SPLIT_SEED = 42

VAL_LOW_DIR = os.path.join(
    DATA_ROOT,
    "Val",
    "Low",
)

VAL_HIGH_DIR = os.path.join(
    DATA_ROOT,
    "Val",
    "Normal",
)


# -----------------------------------------------------------------------------
# Output paths
# -----------------------------------------------------------------------------

# Keep checkpoints and generated outputs under the common RESULT tree.
# This makes training, resume, and test use the same dataset-specific location:
#   RESULT/<DATA_NAME>/RetinexTapetum/checkpoints
# or
#   RESULT/<DATA_NAME>/<DATA_VARIANT>/RetinexTapetum/checkpoints
RESULT_ROOT = os.environ.get(
    "RETINEX_RESULT_ROOT",
    "/content/drive/MyDrive/TAPETUM/RESULT",
).strip()

if DATA_VARIANT is None:
    MODEL_RESULT_ROOT = os.path.join(
        RESULT_ROOT,
        DATA_NAME,
        "RetinexTapetum",
    )
else:
    MODEL_RESULT_ROOT = os.path.join(
        RESULT_ROOT,
        DATA_NAME,
        DATA_VARIANT,
        "RetinexTapetum",
    )

CKPT_DIR = os.path.join(
    MODEL_RESULT_ROOT,
    "checkpoints",
)

RESULT_DIR = os.path.join(
    MODEL_RESULT_ROOT,
    "results",
    "Test",
)


# -----------------------------------------------------------------------------
# Runtime
# -----------------------------------------------------------------------------

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# -----------------------------------------------------------------------------
# Dataset-specific hyperparameter profiles
# -----------------------------------------------------------------------------

# Each dataset has an independent profile. Change only the relevant profile
# when a new hyperparameter-search result is selected for that dataset.
DATASET_PROFILES = {
    ("LOL-v1", None): {
        "BATCH_SIZE": 2,
        "NUM_WORKERS": 2,
        "CROP_SIZE": 256,
        "EPOCHS": 120,
        "LR": 0.00046948014087030786,
        "MIN_LR": 2.12826357198832e-05,
        "GRAD_CLIP_NORM": 3.0,
        "BASE_CHANNELS": 128,
        "LAMBDA_MAX": 1.35,
        "LAMBDA_INIT": 0.0,
        "W_L1": 1.0,
        "W_SSIM": 0.35,
        "W_COLOR": 0.06,
        "W_CHROMA": 0.12,
        "W_ATTN": 0.012,
        "W_EDGE": 0.04,
        "W_DARK_NOISE": 0.01,
        "USE_LPIPS_LOSS": True,
        "LPIPS_NET": "alex",
        "W_LPIPS": 0.08,
        "LPIPS_LOSS_RESIZE": 256,
        "LPIPS_METRIC_RESIZE": 0,
        "BEST_MODEL_METRIC": "psnr",
        "W_RECON_LOW": 1.0,
        "W_RECON_HIGH": 0.85,
        "W_REFLECT": 0.06,
        "W_SMOOTH_LOW": 0.08,
        "W_SMOOTH_HIGH": 0.08,
        "W_SMOOTH_ENH": 0.06,
        "PATIENCE": 45,
    },
    ("LOL-v2", "Real_captured"): {
        "BATCH_SIZE": 2,
        "NUM_WORKERS": 2,
        "CROP_SIZE": 256,
        "EPOCHS": 120,
        "LR": 0.00046948014087030786,
        "MIN_LR": 2.12826357198832e-05,
        "GRAD_CLIP_NORM": 3.0,
        "BASE_CHANNELS": 128,
        "LAMBDA_MAX": 1.35,
        "LAMBDA_INIT": 0.0,
        "W_L1": 1.0,
        "W_SSIM": 0.35,
        "W_COLOR": 0.06,
        "W_CHROMA": 0.12,
        "W_ATTN": 0.012,
        "W_EDGE": 0.04,
        "W_DARK_NOISE": 0.01,
        "USE_LPIPS_LOSS": True,
        "LPIPS_NET": "alex",
        "W_LPIPS": 0.08,
        "LPIPS_LOSS_RESIZE": 256,
        "LPIPS_METRIC_RESIZE": 0,
        "BEST_MODEL_METRIC": "psnr",
        "W_RECON_LOW": 1.0,
        "W_RECON_HIGH": 0.85,
        "W_REFLECT": 0.06,
        "W_SMOOTH_LOW": 0.08,
        "W_SMOOTH_HIGH": 0.08,
        "W_SMOOTH_ENH": 0.06,
        "PATIENCE": 45,
    },
    ("LOL-v2", "Synthetic"): {
        "BATCH_SIZE": 2,
        "NUM_WORKERS": 2,
        "CROP_SIZE": 256,
        "EPOCHS": 120,
        "LR": 0.00046948014087030786,
        "MIN_LR": 2.12826357198832e-05,
        "GRAD_CLIP_NORM": 3.0,
        "BASE_CHANNELS": 128,
        "LAMBDA_MAX": 1.35,
        "LAMBDA_INIT": 0.0,
        "W_L1": 1.0,
        "W_SSIM": 0.35,
        "W_COLOR": 0.06,
        "W_CHROMA": 0.12,
        "W_ATTN": 0.012,
        "W_EDGE": 0.04,
        "W_DARK_NOISE": 0.01,
        "USE_LPIPS_LOSS": True,
        "LPIPS_NET": "alex",
        "W_LPIPS": 0.08,
        "LPIPS_LOSS_RESIZE": 256,
        "LPIPS_METRIC_RESIZE": 0,
        "BEST_MODEL_METRIC": "psnr",
        "W_RECON_LOW": 1.0,
        "W_RECON_HIGH": 0.85,
        "W_REFLECT": 0.06,
        "W_SMOOTH_LOW": 0.08,
        "W_SMOOTH_HIGH": 0.08,
        "W_SMOOTH_ENH": 0.06,
        "PATIENCE": 45,
    },
    ("UHD-LL down4", None): {
        "BATCH_SIZE": 2,
        "NUM_WORKERS": 2,
        "CROP_SIZE": 256,
        "EPOCHS": 120,
        "LR": 0.00046948014087030786,
        "MIN_LR": 2.12826357198832e-05,
        "GRAD_CLIP_NORM": 3.0,
        "BASE_CHANNELS": 128,
        "LAMBDA_MAX": 1.35,
        "LAMBDA_INIT": 0.0,
        "W_L1": 1.0,
        "W_SSIM": 0.35,
        "W_COLOR": 0.06,
        "W_CHROMA": 0.12,
        "W_ATTN": 0.012,
        "W_EDGE": 0.04,
        "W_DARK_NOISE": 0.01,
        "USE_LPIPS_LOSS": True,
        "LPIPS_NET": "alex",
        "W_LPIPS": 0.08,
        "LPIPS_LOSS_RESIZE": 256,
        "LPIPS_METRIC_RESIZE": 0,
        "BEST_MODEL_METRIC": "psnr",
        "W_RECON_LOW": 1.0,
        "W_RECON_HIGH": 0.85,
        "W_REFLECT": 0.06,
        "W_SMOOTH_LOW": 0.08,
        "W_SMOOTH_HIGH": 0.08,
        "W_SMOOTH_ENH": 0.06,
        "PATIENCE": 45,
    },
    ("SICE", None): {
        "BATCH_SIZE": 2,
        "NUM_WORKERS": 2,
        "CROP_SIZE": 256,
        "EPOCHS": 120,
        "LR": 0.00046948014087030786,
        "MIN_LR": 2.12826357198832e-05,
        "GRAD_CLIP_NORM": 3.0,
        "BASE_CHANNELS": 128,
        "LAMBDA_MAX": 1.35,
        "LAMBDA_INIT": 0.0,
        "W_L1": 1.0,
        "W_SSIM": 0.35,
        "W_COLOR": 0.06,
        "W_CHROMA": 0.12,
        "W_ATTN": 0.012,
        "W_EDGE": 0.04,
        "W_DARK_NOISE": 0.01,
        "USE_LPIPS_LOSS": True,
        "LPIPS_NET": "alex",
        "W_LPIPS": 0.08,
        "LPIPS_LOSS_RESIZE": 256,
        "LPIPS_METRIC_RESIZE": 0,
        "BEST_MODEL_METRIC": "psnr",
        "W_RECON_LOW": 1.0,
        "W_RECON_HIGH": 0.85,
        "W_REFLECT": 0.06,
        "W_SMOOTH_LOW": 0.08,
        "W_SMOOTH_HIGH": 0.08,
        "W_SMOOTH_ENH": 0.06,
        "PATIENCE": 45,
    },
    ("LoLI-Street", None): {
        "BATCH_SIZE": 2,
        "NUM_WORKERS": 2,
        "CROP_SIZE": 256,
        "EPOCHS": 120,
        "LR": 0.00046948014087030786,
        "MIN_LR": 2.12826357198832e-05,
        "GRAD_CLIP_NORM": 3.0,
        "BASE_CHANNELS": 128,
        "LAMBDA_MAX": 1.35,
        "LAMBDA_INIT": 0.0,
        "W_L1": 1.0,
        "W_SSIM": 0.35,
        "W_COLOR": 0.06,
        "W_CHROMA": 0.12,
        "W_ATTN": 0.012,
        "W_EDGE": 0.04,
        "W_DARK_NOISE": 0.01,
        "USE_LPIPS_LOSS": True,
        "LPIPS_NET": "alex",
        "W_LPIPS": 0.08,
        "LPIPS_LOSS_RESIZE": 256,
        "LPIPS_METRIC_RESIZE": 0,
        "BEST_MODEL_METRIC": "psnr",
        "W_RECON_LOW": 1.0,
        "W_RECON_HIGH": 0.85,
        "W_REFLECT": 0.06,
        "W_SMOOTH_LOW": 0.08,
        "W_SMOOTH_HIGH": 0.08,
        "W_SMOOTH_ENH": 0.06,
        "PATIENCE": 45,
    },
}

# -----------------------------------------------------------------------------
# Hyperparameter / checkpoint profile selection
# -----------------------------------------------------------------------------

# DATA_NAME selects the dataset to read. PROFILE_DATA_NAME and
# PROFILE_DATA_VARIANT select the trained configuration whose weights and model
# settings will be used. This separation allows unpaired external test sets
# such as DCIM, LIME, MEF, NPE, and VV to be evaluated with any trained profile.
#
# Examples:
#   RETINEX_DATA_NAME=DCIM
#   RETINEX_PROFILE_DATA_NAME=LOL-v1
#   RETINEX_PROFILE_DATA_VARIANT=None
#
#   RETINEX_DATA_NAME=MEF
#   RETINEX_PROFILE_DATA_NAME=LOL-v2
#   RETINEX_PROFILE_DATA_VARIANT=Real_captured

DEFAULT_PROFILE_DATA_NAME = (
    DATA_NAME
    if SELECTED_DATASET in DATASET_PROFILES
    else "LOL-v1"
)

DEFAULT_PROFILE_DATA_VARIANT = (
    DATA_VARIANT
    if SELECTED_DATASET in DATASET_PROFILES
    else None
)

PROFILE_DATA_NAME = os.environ.get(
    "RETINEX_PROFILE_DATA_NAME",
    DEFAULT_PROFILE_DATA_NAME,
).strip()

_profile_variant_env = os.environ.get(
    "RETINEX_PROFILE_DATA_VARIANT",
    "None"
    if DEFAULT_PROFILE_DATA_VARIANT is None
    else DEFAULT_PROFILE_DATA_VARIANT,
)

PROFILE_DATA_VARIANT = (
    None
    if str(_profile_variant_env).strip().lower()
    in {
        "",
        "none",
        "null",
    }
    else str(_profile_variant_env).strip()
)

PROFILE_KEY = (
    PROFILE_DATA_NAME,
    PROFILE_DATA_VARIANT,
)

if PROFILE_KEY not in DATASET_PROFILES:
    available_profiles = "\n".join(
        f"  - PROFILE_DATA_NAME={name!r}, "
        f"PROFILE_DATA_VARIANT={variant!r}"
        for name, variant in sorted(
            DATASET_PROFILES,
            key=lambda item: (
                item[0],
                str(item[1]),
            ),
        )
    )

    raise ValueError(
        "Invalid RetinexTapetum profile selection.\n"
        f"Selected: PROFILE_DATA_NAME={PROFILE_DATA_NAME!r}, "
        f"PROFILE_DATA_VARIANT={PROFILE_DATA_VARIANT!r}\n"
        f"Available trained profiles:\n{available_profiles}"
    )

ACTIVE_PROFILE = DATASET_PROFILES[PROFILE_KEY]

EXTERNAL_TEST_DATASETS = {
    "DCIM",
    "LIME",
    "MEF",
    "NPE",
    "VV",
}

IS_EXTERNAL_TEST_DATASET = (
    DATA_NAME in EXTERNAL_TEST_DATASETS
)


def profile_value(name: str):
    """Return a required value from the selected dataset profile."""

    if name not in ACTIVE_PROFILE:
        raise KeyError(
            f"Missing configuration value {name!r} "
            f"in dataset profile {PROFILE_KEY!r}."
        )

    return ACTIVE_PROFILE[name]


# -----------------------------------------------------------------------------
# Active data loading / optimization hyperparameters
# -----------------------------------------------------------------------------

IMG_EXTS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
)

BATCH_SIZE = profile_value("BATCH_SIZE")
NUM_WORKERS = profile_value("NUM_WORKERS")
CROP_SIZE = profile_value("CROP_SIZE")
EPOCHS = profile_value("EPOCHS")
LR = profile_value("LR")
MIN_LR = profile_value("MIN_LR")
GRAD_CLIP_NORM = profile_value("GRAD_CLIP_NORM")
BASE_CHANNELS = profile_value("BASE_CHANNELS")

SHOW_PROGRESS_BARS = False
TRAIN_LOG_INTERVAL = 50
VAL_LOG_INTERVAL = 25


# -----------------------------------------------------------------------------
# Active Tapetum illumination enhancement hyperparameters
# -----------------------------------------------------------------------------

LAMBDA_MAX = profile_value("LAMBDA_MAX")
LAMBDA_INIT = profile_value("LAMBDA_INIT")


# -----------------------------------------------------------------------------
# Active main enhancement loss weights
# -----------------------------------------------------------------------------

W_L1 = profile_value("W_L1")
W_SSIM = profile_value("W_SSIM")
W_COLOR = profile_value("W_COLOR")
W_CHROMA = profile_value("W_CHROMA")
W_ATTN = profile_value("W_ATTN")
W_EDGE = profile_value("W_EDGE")
W_DARK_NOISE = profile_value("W_DARK_NOISE")


# -----------------------------------------------------------------------------
# Active perceptual loss configuration
# -----------------------------------------------------------------------------

USE_LPIPS_LOSS = profile_value("USE_LPIPS_LOSS")
LPIPS_NET = profile_value("LPIPS_NET")
W_LPIPS = profile_value("W_LPIPS")
LPIPS_LOSS_RESIZE = profile_value("LPIPS_LOSS_RESIZE")
LPIPS_METRIC_RESIZE = profile_value("LPIPS_METRIC_RESIZE")
BEST_MODEL_METRIC = profile_value("BEST_MODEL_METRIC")


# -----------------------------------------------------------------------------
# Active Retinex decomposition loss weights
# -----------------------------------------------------------------------------

W_RECON_LOW = profile_value("W_RECON_LOW")
W_RECON_HIGH = profile_value("W_RECON_HIGH")
W_REFLECT = profile_value("W_REFLECT")
W_SMOOTH_LOW = profile_value("W_SMOOTH_LOW")
W_SMOOTH_HIGH = profile_value("W_SMOOTH_HIGH")
W_SMOOTH_ENH = profile_value("W_SMOOTH_ENH")


# -----------------------------------------------------------------------------
# Training control
# -----------------------------------------------------------------------------

PATIENCE = profile_value("PATIENCE")
SEED = 42

RESUME_TRAINING = env_to_bool(
    "RETINEX_TAPETUM_RESUME",
    default=False,
)

RESUME_CKPT_NAME = os.environ.get(
    "RETINEX_TAPETUM_RESUME_CKPT",
    "last.pth",
)


def print_active_profile() -> None:
    """Print the selected dataset and its active hyperparameter profile."""

    print("=" * 80)
    print("ACTIVE RETINEXTAPETUM CONFIGURATION")
    print("=" * 80)
    print(f"DATA_NAME       : {DATA_NAME}")
    print(f"DATA_VARIANT    : {DATA_VARIANT}")
    print(f"PROFILE_DATA_NAME    : {PROFILE_DATA_NAME}")
    print(f"PROFILE_DATA_VARIANT : {PROFILE_DATA_VARIANT}")
    print(f"PROFILE_KEY          : {PROFILE_KEY}")
    print(f"EXTERNAL_TEST_ONLY   : {IS_EXTERNAL_TEST_DATASET}")
    print(f"DATA_ROOT       : {DATA_ROOT}")
    print(f"RUN_ROOT        : {RUN_ROOT}")

    for key, value in ACTIVE_PROFILE.items():
        print(f"{key:<20}: {value}")

    print("=" * 80)


# -----------------------------------------------------------------------------
# Inference color restoration
# -----------------------------------------------------------------------------

COLOR_RESTORE = False
COLOR_RESTORE_STRENGTH = 0.0
COLOR_RESTORE_EPS = 1e-4
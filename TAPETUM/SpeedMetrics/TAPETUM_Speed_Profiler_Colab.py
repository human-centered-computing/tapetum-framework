"""
TAPETUM computational-efficiency profiler for Google Colab.

Run one model at a time. The script profiles a fixed 256x256, batch-1, FP32
inference and saves every result under:

    MyDrive/TAPETUM/SpeedMetrics/<dataset>/<model>/

Generated metrics:
    parameters, MACs, FLOPs, latency statistics, FPS, peak GPU memory,
    checkpoint size, runtime/device metadata.

Notes:
    - Image reading and result writing are excluded from latency.
    - LIME is a non-learned CPU method. Parameters are 0 and MACs/FLOPs are N/A.
    - TensorFlow models are profiled with TensorFlow v1 graph statistics.
    - PyTorch models are profiled with THOP using the same 256x256 input.
"""

# =============================================================================
# COLAB FORM: change these fields, then run the cell/script
# =============================================================================

MODEL_NAME = "RetinexTapetum (Proposed)"  # @param ["URetinex-Net++", "RetinexFormer", "KinD++", "RUAS", "RetinexNet", "Zero-DCE", "LIME", "RetinexTapetum (Proposed)"]
DATASET_NAME = "LOL-v2 Real-Captured"  # @param ["LOL-v1", "LOL-v2 Real-Captured", "LOL-v2 Synthetic", "UHD-LL down4"]
PROFILE_SIZE = 256  # @param {type:"integer"}
WARMUP_RUNS = 50  # @param {type:"integer"}
TIMED_RUNS = 500  # @param {type:"integer"}
PRECISION = "FP32"  # @param ["FP32"]
OVERWRITE_MODEL_RESULT = True  # @param {type:"boolean"}

# =============================================================================
# SETUP
# =============================================================================

import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import sys
import gc
import csv
import json
import math
import time
import glob
import shutil
import socket
import types
import threading
import traceback
import subprocess
import importlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime, timezone


def run_command(command, check=True):
    completed = subprocess.run(
        command,
        shell=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if check and completed.returncode != 0:
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {command}\n{completed.stdout}"
        )
    return completed.stdout.strip()


def install_dependencies():
    packages = [
        "thop",
        "fvcore",
        "tf_slim",
        "tf_keras",
        "natsort",
        "pynvml",
        "pyyaml",
        "opencv-python-headless",
        "timm",
        "einops",
    ]
    run_command(
        f"{sys.executable} -m pip install -q " + " ".join(packages),
        check=True,
    )


install_dependencies()

import numpy as np
import pandas as pd
import cv2
from PIL import Image

try:
    from google.colab import drive
except ImportError as exc:
    raise RuntimeError("This script must be executed in Google Colab.") from exc

if not os.path.ismount("/content/drive"):
    drive.mount("/content/drive")
else:
    print("Google Drive is already mounted.")


# =============================================================================
# PATH CONFIGURATION
# =============================================================================

DRIVE_ROOT = Path("/content/drive/MyDrive")
TAPETUM_ROOT = DRIVE_ROOT / "TAPETUM"
DATASETS_ROOT = TAPETUM_ROOT / "datasets"
SPEED_ROOT = TAPETUM_ROOT / "SpeedMetrics"
WORK_ROOT = Path("/content/TAPETUM_SPEED_WORK")

MODEL_DIR_NAMES = {
    "URetinex-Net++": "URetinex-Net-PLUS",
    "RetinexFormer": "RetinexFormer",
    "KinD++": "KinD_plus",
    "RUAS": "RUAS",
    "RetinexNet": "RetinexNet",
    "Zero-DCE": "Zero-DCE",
    "LIME": "LIME",
    "RetinexTapetum (Proposed)": "RetinexTapetum",
}

RESULT_MODEL_NAMES = {
    "URetinex-Net++": "URetinex-Net++",
    "RetinexFormer": "RetinexFormer",
    "KinD++": "KinD++",
    "RUAS": "RUAS",
    "RetinexNet": "RetinexNet",
    "Zero-DCE": "Zero-DCE",
    "LIME": "LIME",
    "RetinexTapetum (Proposed)": "RetinexTapetum (Proposed)",
}

DATASET_CONFIG = {
    "LOL-v1": {
        "dataset_key": "LOL-v1",
        "dataset_root": DATASETS_ROOT / "LOL-v1",
        "run_root": TAPETUM_ROOT / "LOL-v1",
    },
    "LOL-v2 Real-Captured": {
        "dataset_key": "LOL-v2 Real-Captured",
        "dataset_root": DATASETS_ROOT / "LOL-v2" / "Real_captured",
        "run_root": TAPETUM_ROOT / "LOL-v2",
    },
    "LOL-v2 Synthetic": {
        "dataset_key": "LOL-v2 Synthetic",
        "dataset_root": DATASETS_ROOT / "LOL-v2" / "Synthetic",
        "run_root": TAPETUM_ROOT / "LOL-v2",
    },
    "UHD-LL down4": {
        "dataset_key": "UHD-LL down4",
        "dataset_root": DATASETS_ROOT / "UHD-LL down4",
        "run_root": TAPETUM_ROOT / "UHD-LL down4",
    },
}

if MODEL_NAME not in MODEL_DIR_NAMES:
    raise ValueError(f"Unsupported model: {MODEL_NAME}")
if DATASET_NAME not in DATASET_CONFIG:
    raise ValueError(f"Unsupported dataset: {DATASET_NAME}")
if PROFILE_SIZE <= 0 or PROFILE_SIZE % 4 != 0:
    raise ValueError("PROFILE_SIZE must be a positive multiple of 4.")
if WARMUP_RUNS < 1 or TIMED_RUNS < 10:
    raise ValueError("Use at least 1 warm-up and 10 timed runs.")

dataset_cfg = DATASET_CONFIG[DATASET_NAME]
DATASET_KEY = dataset_cfg["dataset_key"]
DATASET_ROOT = Path(dataset_cfg["dataset_root"])
RUN_ROOT = Path(dataset_cfg["run_root"])
PROJECT_SOURCE = TAPETUM_ROOT / MODEL_DIR_NAMES[MODEL_NAME]
MODEL_RESULT_NAME = MODEL_NAME.replace(" (Proposed)", "").replace("/", "_")
RESULT_DIR = SPEED_ROOT / DATASET_KEY / MODEL_RESULT_NAME
MASTER_CSV = SPEED_ROOT / DATASET_KEY / "speed_metrics_all.csv"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
WORK_ROOT.mkdir(parents=True, exist_ok=True)


def print_configuration():
    print("=" * 88)
    print("TAPETUM SPEED PROFILER")
    print("=" * 88)
    print(f"MODEL_NAME       : {MODEL_NAME}")
    print(f"DATASET_NAME     : {DATASET_NAME}")
    print(f"DATASET_ROOT     : {DATASET_ROOT}")
    print(f"PROJECT_SOURCE   : {PROJECT_SOURCE}")
    print(f"RUN_ROOT         : {RUN_ROOT}")
    print(f"RESULT_DIR       : {RESULT_DIR}")
    print(f"INPUT            : 1 x 3 x {PROFILE_SIZE} x {PROFILE_SIZE}, {PRECISION}")
    print(f"WARMUP/TIMED     : {WARMUP_RUNS}/{TIMED_RUNS}")
    print("=" * 88)


print_configuration()

if not TAPETUM_ROOT.exists():
    raise FileNotFoundError(f"TAPETUM folder not found: {TAPETUM_ROOT}")
if not DATASET_ROOT.exists():
    raise FileNotFoundError(f"Dataset folder not found: {DATASET_ROOT}")
if not PROJECT_SOURCE.exists():
    raise FileNotFoundError(f"Model project folder not found: {PROJECT_SOURCE}")


# =============================================================================
# FILE AND CHECKPOINT HELPERS
# =============================================================================

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
CHECKPOINT_EXTENSIONS = {".pth", ".pt", ".ckpt", ".index", ".meta", ".data"}


def copy_project_to_local(source, destination):
    if destination.exists():
        shutil.rmtree(destination)
    ignore = shutil.ignore_patterns(
        ".git", "__pycache__", "results", "result", "metrics", "logs", "*.png", "*.jpg"
    )
    shutil.copytree(source, destination, ignore=ignore)
    return destination


def find_first_test_image(root):
    preferred_tokens = ("test/low", "test/input", "test/low-light", "test/dark")
    all_images = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            normalized = str(path).lower().replace("\\", "/")
            score = 0
            if any(token in normalized for token in preferred_tokens):
                score += 100
            if "/test/" in normalized:
                score += 50
            if any(token in path.parent.name.lower() for token in ("low", "input", "dark")):
                score += 20
            if any(token in normalized for token in ("normal", "high", "target", "gt")):
                score -= 100
            all_images.append((score, str(path), path))
    if not all_images:
        raise FileNotFoundError(f"No image was found under {root}")
    all_images.sort(key=lambda item: (-item[0], item[1]))
    return all_images[0][2]


def prepare_profile_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((PROFILE_SIZE, PROFILE_SIZE), Image.Resampling.BICUBIC)
    output_path = RESULT_DIR / "profile_input.png"
    image.save(output_path)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return image, array, output_path


def checkpoint_search_roots(project_dir):
    model_aliases = [
        MODEL_DIR_NAMES[MODEL_NAME],
        MODEL_RESULT_NAME,
        MODEL_NAME.replace(" (Proposed)", ""),
    ]
    variant = ""
    if "Real-Captured" in DATASET_NAME:
        variant = "Real_captured"
    elif "Synthetic" in DATASET_NAME:
        variant = "Synthetic"

    roots = [project_dir, PROJECT_SOURCE]
    for alias in model_aliases:
        roots.extend([
            RUN_ROOT / alias,
            RUN_ROOT / f"{alias}-{variant}" if variant else RUN_ROOT / alias,
            RUN_ROOT / f"{alias}_{variant}" if variant else RUN_ROOT / alias,
        ])
    unique = []
    seen = set()
    for root in roots:
        root = Path(root)
        if str(root) not in seen and root.exists():
            unique.append(root)
            seen.add(str(root))
    return unique


def checkpoint_score(path):
    name = path.name.lower()
    score = 0
    if "best" in name:
        score += 1000
    if "latest" in name or "last" in name:
        score += 800
    if "epoch99" in name:
        score += 700
    if "final" in name:
        score += 600
    if "optimizer" in name or "training_state" in str(path).lower():
        score -= 1000
    try:
        score += path.stat().st_mtime / 1e12
    except OSError:
        pass
    return score


def find_torch_checkpoint(project_dir, required_tokens=(), preferred_tokens=()):
    candidates = []
    for root in checkpoint_search_roots(project_dir):
        for extension in ("*.pth", "*.pt"):
            for path in root.rglob(extension):
                normalized = str(path).lower()
                if required_tokens and not all(token.lower() in normalized for token in required_tokens):
                    continue
                score = checkpoint_score(path)
                score += 150 * sum(token.lower() in normalized for token in preferred_tokens)
                candidates.append((score, path))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def strip_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("params_ema", "params", "model", "state_dict", "net_g", "network"):
            value = checkpoint.get(key)
            if isinstance(value, dict) and value:
                checkpoint = value
                break
    if not isinstance(checkpoint, dict):
        raise TypeError("The selected checkpoint does not contain a state dictionary.")
    clean = {}
    for key, value in checkpoint.items():
        new_key = key
        for prefix in ("module.", "model.", "net_g."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        clean[new_key] = value
    return clean


def load_state_flexible(model, checkpoint_path, torch_module, strict_first=True):
    checkpoint = torch_module.load(checkpoint_path, map_location="cpu")
    state = strip_state_dict(checkpoint)
    errors = []
    if strict_first:
        try:
            model.load_state_dict(state, strict=True)
            return "strict"
        except Exception as exc:
            errors.append(str(exc))
    missing, unexpected = model.load_state_dict(state, strict=False)
    loaded_count = len(state) - len(unexpected)
    if loaded_count <= 0:
        raise RuntimeError("No checkpoint weights matched the model. " + " | ".join(errors))
    print(f"Non-strict checkpoint load: missing={len(missing)}, unexpected={len(unexpected)}")
    return "non-strict"


def checkpoint_size_mb(path):
    if path is None:
        return 0.0
    path = Path(path)
    if path.is_file():
        prefix = str(path)
        if path.suffix == ".index":
            prefix = prefix[:-len(".index")]
        related = glob.glob(prefix + "*")
        if related:
            return sum(Path(item).stat().st_size for item in related if Path(item).is_file()) / 1e6
        return path.stat().st_size / 1e6
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) / 1e6


def import_file_module(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def clear_conflicting_modules():
    for name in list(sys.modules):
        if name in {"model", "utils", "config", "operations", "genotypes", "network"}:
            sys.modules.pop(name, None)
        elif name.startswith(("basicsr", "network.", "loss.", "data.")):
            sys.modules.pop(name, None)
    importlib.invalidate_caches()


def safe_output_tensor(output):
    if isinstance(output, dict):
        for key in ("enhanced", "output", "out", "result"):
            if key in output:
                return output[key]
        return next(iter(output.values()))
    if isinstance(output, (list, tuple)):
        if len(output) == 3 and hasattr(output[1], "shape"):
            return output[1]
        return safe_output_tensor(output[-1])
    return output


# =============================================================================
# GPU MEMORY AND TIMING
# =============================================================================

class NvidiaProcessMemorySampler:
    def __init__(self, interval=0.002):
        self.interval = interval
        self.stop_event = threading.Event()
        self.thread = None
        self.baseline = 0.0
        self.peak = 0.0

    @staticmethod
    def current_process_memory_mb():
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            pid = os.getpid()
            processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            used_bytes = sum(
                process.usedGpuMemory
                for process in processes
                if process.pid == pid and process.usedGpuMemory is not None
            )
            return used_bytes / 1e6
        except Exception:
            return 0.0

    def _sample(self):
        while not self.stop_event.is_set():
            self.peak = max(self.peak, self.current_process_memory_mb())
            self.stop_event.wait(self.interval)

    def start(self):
        self.baseline = self.current_process_memory_mb()
        self.peak = self.baseline
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        return max(0.0, self.peak - self.baseline)


def summarize_latencies(latencies_ms):
    values = np.asarray(latencies_ms, dtype=np.float64)
    mean_ms = float(np.mean(values))
    return {
        "latency_mean_ms": mean_ms,
        "latency_median_ms": float(np.median(values)),
        "latency_std_ms": float(np.std(values)),
        "latency_min_ms": float(np.min(values)),
        "latency_p95_ms": float(np.percentile(values, 95)),
        "fps": float(1000.0 / mean_ms),
    }


def benchmark_torch(model, input_tensor, forward_fn=None):
    import torch

    if forward_fn is None:
        forward_fn = lambda: model(input_tensor)

    model.eval()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    with torch.inference_mode():
        for _ in range(WARMUP_RUNS):
            forward_fn()
        torch.cuda.synchronize()

        sampler = NvidiaProcessMemorySampler()
        sampler.start()
        latencies = []
        for _ in range(TIMED_RUNS):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = forward_fn()
            end.record()
            torch.cuda.synchronize()
            latencies.append(float(start.elapsed_time(end)))
        nvml_delta = sampler.stop()

    peak_allocated = torch.cuda.max_memory_allocated() / 1e6
    peak_reserved = torch.cuda.max_memory_reserved() / 1e6
    stats = summarize_latencies(latencies)
    stats.update({
        "peak_gpu_memory_mb": float(max(peak_allocated, nvml_delta)),
        "peak_gpu_allocated_mb": float(peak_allocated),
        "peak_gpu_reserved_mb": float(peak_reserved),
    })
    return output, stats


def profile_torch_complexity(model, input_tensor, forward_model=None):
    import torch
    from thop import profile

    target = forward_model if forward_model is not None else model
    try:
        with torch.inference_mode():
            macs, params = profile(target, inputs=(input_tensor,), verbose=False)
        return float(params), float(macs), float(2.0 * macs), "THOP"
    except Exception as thop_error:
        print(f"THOP warning: {thop_error}")
        try:
            from fvcore.nn import FlopCountAnalysis
            params = sum(parameter.numel() for parameter in model.parameters())
            flops = float(FlopCountAnalysis(target, input_tensor).total())
            return float(params), float(flops / 2.0), flops, "fvcore fallback"
        except Exception as fvcore_error:
            print(f"fvcore warning: {fvcore_error}")
            params = sum(parameter.numel() for parameter in model.parameters())
            return float(params), math.nan, math.nan, "unavailable"


def benchmark_python_cpu(forward_fn):
    for _ in range(WARMUP_RUNS):
        forward_fn()
    latencies = []
    output = None
    for _ in range(TIMED_RUNS):
        start = time.perf_counter()
        output = forward_fn()
        latencies.append((time.perf_counter() - start) * 1000.0)
    stats = summarize_latencies(latencies)
    stats.update({
        "peak_gpu_memory_mb": math.nan,
        "peak_gpu_allocated_mb": math.nan,
        "peak_gpu_reserved_mb": math.nan,
    })
    return output, stats


# =============================================================================
# PYTORCH MODEL ADAPTERS
# =============================================================================

def prepare_torch_input(rgb_array, device):
    import torch
    return torch.from_numpy(rgb_array).permute(2, 0, 1).unsqueeze(0).contiguous().to(device)


def load_retinextapetum(project_dir, device):
    import torch
    clear_conflicting_modules()
    sys.path.insert(0, str(project_dir))
    module = import_file_module("tapetum_profile_model", project_dir / "model.py")
    checkpoint = find_torch_checkpoint(
        project_dir,
        preferred_tokens=("retinextapetum", "best", "checkpoints"),
    )
    if checkpoint is None:
        raise FileNotFoundError("RetinexTapetum checkpoint was not found.")
    raw = torch.load(checkpoint, map_location="cpu")
    state = raw.get("model", raw) if isinstance(raw, dict) else raw
    head = state.get("decom_net.head.weight") if isinstance(state, dict) else None
    base = int(head.shape[0]) if head is not None else 128
    model = module.RetinexTapetum(base=base, lambda_init=0.0, lambda_max=1.65)
    load_state_flexible(model, checkpoint, torch)
    model = model.to(device).eval()
    return model, checkpoint, {"base_channels": base}


def load_zero_dce(project_dir, device):
    import torch
    clear_conflicting_modules()
    code_dir = project_dir / "Zero-DCE_code"
    if not code_dir.exists():
        code_dir = project_dir
    sys.path.insert(0, str(code_dir))
    module = import_file_module("zero_dce_profile_model", code_dir / "model.py")
    model = module.enhance_net_nopool()
    checkpoint = find_torch_checkpoint(
        code_dir,
        preferred_tokens=("epoch99", "snapshots_lolv2", "snapshots"),
    )
    if checkpoint is None:
        raise FileNotFoundError("Zero-DCE checkpoint was not found.")
    load_state_flexible(model, checkpoint, torch)
    return model.to(device).eval(), checkpoint, {}


def load_retinexformer(project_dir, device):
    import torch
    clear_conflicting_modules()
    sys.path.insert(0, str(project_dir))
    arch_file = project_dir / "basicsr" / "models" / "archs" / "RetinexFormer_arch.py"
    module = import_file_module("retinexformer_profile_arch", arch_file)
    model = module.RetinexFormer(
        in_channels=3,
        out_channels=3,
        n_feat=40,
        stage=1,
        num_blocks=[1, 2, 2],
    )
    checkpoint = find_torch_checkpoint(
        project_dir,
        preferred_tokens=("retinexformer", "best", "net_g", "models"),
    )
    if checkpoint is None:
        raise FileNotFoundError(
            "RetinexFormer checkpoint was not found in the project or selected dataset run folder."
        )
    load_state_flexible(model, checkpoint, torch)
    return model.to(device).eval(), checkpoint, {
        "n_feat": 40,
        "stage": 1,
        "num_blocks": [1, 2, 2],
    }


def load_ruas(project_dir, device):
    import torch
    clear_conflicting_modules()
    sys.path.insert(0, str(project_dir))
    previous = Path.cwd()
    os.chdir(project_dir)
    try:
        module = import_file_module("ruas_profile_model", project_dir / "model.py")
        model = module.Network()
    finally:
        os.chdir(previous)
    preferred = ("lol",) if DATASET_NAME == "LOL-v1" else ("upe", "best")
    checkpoint = find_torch_checkpoint(project_dir, preferred_tokens=preferred)
    if checkpoint is None:
        raise FileNotFoundError("RUAS checkpoint was not found.")
    try:
        load_state_flexible(model, checkpoint, torch)
    except Exception:
        # Some RUAS folders contain separate enhance/denoise weights. The model already
        # loads denoise.pt; use the selected file for the enhancement branch if necessary.
        state = strip_state_dict(torch.load(checkpoint, map_location="cpu"))
        model.enhance_net.load_state_dict(state, strict=False)
    return model.to(device).eval(), checkpoint, {}


def load_uretinex(project_dir, device):
    import torch
    clear_conflicting_modules()
    sys.path.insert(0, str(project_dir))
    test_module = import_file_module("uretinex_profile_test", project_dir / "test.py")
    opts = SimpleNamespace(
        real_high=str(project_dir / "evaluate_data" / "real_high"),
        low_dir=str(project_dir / "evaluate_data" / "low-source"),
        gpu_id="0",
        loop_time_for_img=1,
        alg_name="URetinex-Net++",
        ratio=5.0,
        evaluate=False,
        test="metric",
        save_dir=str(RESULT_DIR),
        high_dir_root=str(project_dir / "evaluate_data" / "real_high"),
        Decom_model_low_path=str(project_dir / "pretrained_model" / "decom" / "decom_low_light.pth"),
        Decom_model_high_path=str(project_dir / "pretrained_model" / "decom" / "decom_high_light.pth"),
        pretrain_unfolding_model_path=None,
        fusion_model_A_path=None,
    )
    previous = Path.cwd()
    os.chdir(project_dir)
    try:
        model = test_module.Inference(opts)
    finally:
        os.chdir(previous)
    checkpoint = project_dir / "pretrained_model" / "fusion_enhance" / "fusion.pth"
    if not checkpoint.exists():
        checkpoint = find_torch_checkpoint(project_dir, preferred_tokens=("fusion", "best"))
    return model.to(device).eval(), checkpoint, {"ratio": 5.0}


def load_torch_model(project_dir, device):
    loaders = {
        "RetinexTapetum (Proposed)": load_retinextapetum,
        "Zero-DCE": load_zero_dce,
        "RetinexFormer": load_retinexformer,
        "RUAS": load_ruas,
        "URetinex-Net++": load_uretinex,
    }
    if MODEL_NAME not in loaders:
        raise ValueError(f"No PyTorch adapter for {MODEL_NAME}")
    return loaders[MODEL_NAME](project_dir, device)


def torch_forward_wrapper(model, input_tensor):
    if MODEL_NAME == "RetinexTapetum (Proposed)":
        return model(input_tensor)["enhanced"]
    if MODEL_NAME == "Zero-DCE":
        return model(input_tensor)[1]
    if MODEL_NAME == "RetinexFormer":
        return model(input_tensor)
    if MODEL_NAME == "RUAS":
        outputs, _ = model(input_tensor)
        return outputs[-1] if DATASET_NAME == "LOL-v1" else outputs[-2]
    if MODEL_NAME == "URetinex-Net++":
        return model(input_tensor)[0]
    raise ValueError(MODEL_NAME)


class TorchOutputAdapter:
    def __init__(self, model):
        import torch.nn as nn
        self._module = nn.Module()
        self._module.model = model

        def forward(instance, x):
            return torch_forward_wrapper(instance.model, x)

        self._module.forward = types.MethodType(forward, self._module)

    @property
    def module(self):
        return self._module


def run_pytorch_profile(project_dir, rgb_array):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU runtime is required for PyTorch profiling.")
    device = torch.device("cuda:0")
    model, checkpoint, extra = load_torch_model(project_dir, device)
    input_tensor = prepare_torch_input(rgb_array, device)
    adapter = TorchOutputAdapter(model).module.to(device).eval()

    params, macs, flops, complexity_tool = profile_torch_complexity(
        model,
        input_tensor,
        forward_model=adapter,
    )
    output, timing = benchmark_torch(
        model,
        input_tensor,
        forward_fn=lambda: adapter(input_tensor),
    )
    output = safe_output_tensor(output)
    output_array = output.detach().float().clamp(0, 1).cpu().squeeze(0).permute(1, 2, 0).numpy()

    return {
        "framework": "PyTorch",
        "device_type": "GPU",
        "parameters": params,
        "macs": macs,
        "flops": flops,
        "complexity_tool": complexity_tool,
        "checkpoint_path": str(checkpoint) if checkpoint else "",
        "model_size_mb": checkpoint_size_mb(checkpoint),
        "output_array": output_array,
        "timing": timing,
        "extra": extra,
    }


# =============================================================================
# TENSORFLOW V1 MODEL ADAPTERS: KinD++ and RetinexNet
# =============================================================================

def load_tf_v1():
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    return tf


def tensorflow_graph_complexity(tf, graph):
    params_profile = tf.profiler.profile(
        graph,
        options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter(),
    )
    flops_profile = tf.profiler.profile(
        graph,
        options=tf.profiler.ProfileOptionBuilder.float_operation(),
    )
    params = float(params_profile.total_parameters if params_profile else 0)
    flops = float(flops_profile.total_float_ops if flops_profile else math.nan)
    macs = flops / 2.0 if math.isfinite(flops) else math.nan
    return params, macs, flops


def find_tf_checkpoint(tf, directories):
    for directory in directories:
        directory = Path(directory)
        if not directory.exists():
            continue
        latest = tf.train.latest_checkpoint(str(directory))
        if latest:
            return latest
        index_files = sorted(directory.rglob("*.index"), key=checkpoint_score, reverse=True)
        if index_files:
            return str(index_files[0])[:-len(".index")]
    return None


def run_tf_benchmark(session, forward_fn):
    for _ in range(WARMUP_RUNS):
        forward_fn()
    sampler = NvidiaProcessMemorySampler()
    sampler.start()
    latencies = []
    output = None
    for _ in range(TIMED_RUNS):
        start = time.perf_counter()
        output = forward_fn()
        latencies.append((time.perf_counter() - start) * 1000.0)
    peak_delta = sampler.stop()
    stats = summarize_latencies(latencies)
    stats.update({
        "peak_gpu_memory_mb": float(peak_delta),
        "peak_gpu_allocated_mb": math.nan,
        "peak_gpu_reserved_mb": math.nan,
    })
    return output, stats


def run_kind_plus_profile(project_dir, rgb_array):
    tf = load_tf_v1()
    tf.reset_default_graph()
    clear_conflicting_modules()
    sys.path.insert(0, str(project_dir))
    model_module = import_file_module("kind_plus_profile_model", project_dir / "model.py")

    graph = tf.get_default_graph()
    input_decom = tf.placeholder(tf.float32, [1, PROFILE_SIZE, PROFILE_SIZE, 3], name="profile_input")
    input_low_r = tf.placeholder(tf.float32, [1, PROFILE_SIZE, PROFILE_SIZE, 3])
    input_low_i = tf.placeholder(tf.float32, [1, PROFILE_SIZE, PROFILE_SIZE, 1])
    input_ratio = tf.placeholder(tf.float32, [1, PROFILE_SIZE, PROFILE_SIZE, 1])
    training = tf.placeholder_with_default(False, shape=())

    decom_r, decom_i = model_module.DecomNet(input_decom)
    restored_r = model_module.Restoration_net(input_low_r, input_low_i, training)
    adjusted_i = model_module.Illumination_adjust_net(input_low_i, input_ratio)

    decom_vars = [v for v in tf.trainable_variables() if "DecomNet" in v.name]
    adjust_vars = [v for v in tf.trainable_variables() if "I_enhance_Net" in v.name]
    restore_vars = [v for v in tf.global_variables() if "Denoise_Net" in v.name]
    saver_decom = tf.train.Saver(var_list=decom_vars)
    saver_adjust = tf.train.Saver(var_list=adjust_vars)
    saver_restore = tf.train.Saver(var_list=restore_vars)

    directories = checkpoint_search_roots(project_dir)
    decom_ckpt = find_tf_checkpoint(tf, [root / "checkpoint" / "decom_model" for root in directories])
    adjust_ckpt = find_tf_checkpoint(tf, [root / "checkpoint" / "illu_model" for root in directories])
    restore_ckpt = find_tf_checkpoint(tf, [root / "checkpoint" / "restoration_model" for root in directories])
    if not all((decom_ckpt, adjust_ckpt, restore_ckpt)):
        raise FileNotFoundError(
            f"KinD++ checkpoints incomplete: decom={decom_ckpt}, adjust={adjust_ckpt}, restoration={restore_ckpt}"
        )

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    session.run(tf.global_variables_initializer())
    saver_decom.restore(session, decom_ckpt)
    saver_adjust.restore(session, adjust_ckpt)
    saver_restore.restore(session, restore_ckpt)

    input_batch = rgb_array[None, ...].astype(np.float32)
    ratio_batch = np.full((1, PROFILE_SIZE, PROFILE_SIZE, 1), 5.0, dtype=np.float32)

    def forward():
        r_low, i_low = session.run(
            [decom_r, decom_i],
            feed_dict={input_decom: input_batch},
        )
        r_restored = session.run(
            restored_r,
            feed_dict={input_low_r: r_low, input_low_i: i_low, training: False},
        )
        i_adjusted = session.run(
            adjusted_i,
            feed_dict={input_low_i: i_low, input_ratio: ratio_batch},
        )
        gray = cv2.cvtColor(np.squeeze(r_low), cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (0, 0), 3)
        low_i = np.minimum(np.sqrt(np.maximum(blurred * 2.0, 0.0)), 1.0)[None, ..., None]
        denoised = r_restored * low_i
        return np.clip(i_low * input_batch + (1.0 - i_low) * denoised * i_adjusted, 0, 1)

    params, macs, flops = tensorflow_graph_complexity(tf, graph)
    output, timing = run_tf_benchmark(session, forward)
    model_size = sum(checkpoint_size_mb(path) for path in (decom_ckpt, adjust_ckpt, restore_ckpt))
    session.close()

    return {
        "framework": "TensorFlow v1 compatibility",
        "device_type": "GPU",
        "parameters": params,
        "macs": macs,
        "flops": flops,
        "complexity_tool": "TensorFlow graph profiler",
        "checkpoint_path": "; ".join((decom_ckpt, adjust_ckpt, restore_ckpt)),
        "model_size_mb": model_size,
        "output_array": np.squeeze(output),
        "timing": timing,
        "extra": {"illumination_ratio": 5.0},
    }


def run_retinexnet_profile(project_dir, rgb_array):
    tf = load_tf_v1()
    tf.reset_default_graph()
    clear_conflicting_modules()
    sys.path.insert(0, str(project_dir))

    source_path = project_dir / "model.py"
    source = source_path.read_text(encoding="utf-8", errors="replace")
    source = source.replace(
        "import tensorflow as tf",
        "import tensorflow.compat.v1 as tf\ntf.disable_v2_behavior()",
    )
    module = types.ModuleType("retinexnet_profile_model")
    module.__file__ = str(source_path)
    sys.modules[module.__name__] = module
    exec(compile(source, str(source_path), "exec"), module.__dict__)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    model = module.lowlight_enhance(session)

    roots = checkpoint_search_roots(project_dir)
    decom_ckpt = find_tf_checkpoint(tf, [root / "model" / "Decom" for root in roots] + [root / "checkpoint" / "Decom" for root in roots])
    relight_ckpt = find_tf_checkpoint(tf, [root / "model" / "Relight" for root in roots] + [root / "checkpoint" / "Relight" for root in roots])
    if not decom_ckpt or not relight_ckpt:
        raise FileNotFoundError(
            f"RetinexNet checkpoints incomplete: Decom={decom_ckpt}, Relight={relight_ckpt}"
        )
    model.saver_Decom.restore(session, decom_ckpt)
    model.saver_Relight.restore(session, relight_ckpt)

    input_batch = rgb_array[None, ...].astype(np.float32)
    forward = lambda: session.run(model.output_S, feed_dict={model.input_low: input_batch})
    params, macs, flops = tensorflow_graph_complexity(tf, tf.get_default_graph())
    output, timing = run_tf_benchmark(session, forward)
    model_size = checkpoint_size_mb(decom_ckpt) + checkpoint_size_mb(relight_ckpt)
    session.close()

    return {
        "framework": "TensorFlow v1 compatibility",
        "device_type": "GPU",
        "parameters": params,
        "macs": macs,
        "flops": flops,
        "complexity_tool": "TensorFlow graph profiler",
        "checkpoint_path": f"{decom_ckpt}; {relight_ckpt}",
        "model_size_mb": model_size,
        "output_array": np.squeeze(output),
        "timing": timing,
        "extra": {},
    }


# =============================================================================
# LIME CPU ADAPTER
# =============================================================================

def lime_enhance_bgr(image_bgr):
    image = image_bgr.astype(np.float32) / 255.0
    illumination = np.max(image, axis=2)
    illumination = cv2.GaussianBlur(illumination, (15, 15), 0)
    illumination = np.clip(illumination, 0.1, 1.0)
    enhanced = image / illumination[..., None]
    return np.clip(enhanced, 0.0, 1.0)


def run_lime_profile(rgb_array):
    image_bgr = cv2.cvtColor((rgb_array * 255.0).round().astype(np.uint8), cv2.COLOR_RGB2BGR)
    output, timing = benchmark_python_cpu(lambda: lime_enhance_bgr(image_bgr))
    output_rgb = cv2.cvtColor((output * 255.0).round().astype(np.uint8), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return {
        "framework": "OpenCV/NumPy",
        "device_type": "CPU",
        "parameters": 0.0,
        "macs": math.nan,
        "flops": math.nan,
        "complexity_tool": "N/A for non-learned iterative/image-processing method",
        "checkpoint_path": "",
        "model_size_mb": 0.0,
        "output_array": output_rgb,
        "timing": timing,
        "extra": {"note": "LIME is profiled on CPU; compare its latency only with CPU results."},
    }


# =============================================================================
# SYSTEM INFORMATION AND RESULT WRITING
# =============================================================================

def get_gpu_name():
    value = run_command("nvidia-smi --query-gpu=name --format=csv,noheader", check=False)
    return value.splitlines()[0].strip() if value else "N/A"


def get_cuda_driver():
    value = run_command("nvidia-smi --query-gpu=driver_version --format=csv,noheader", check=False)
    return value.splitlines()[0].strip() if value else "N/A"


def safe_number(value, scale=1.0):
    try:
        number = float(value)
        return number / scale if math.isfinite(number) else None
    except Exception:
        return None


def save_output_image(array):
    array = np.asarray(array)
    if array.ndim == 4:
        array = array[0]
    array = np.clip(array, 0.0, 1.0)
    image = Image.fromarray((array * 255.0).round().astype(np.uint8), mode="RGB")
    path = RESULT_DIR / "profile_output.png"
    image.save(path)
    return path


def build_result(profile, input_image_path, output_image_path, elapsed_total):
    timing = profile["timing"]
    result = {
        "method": RESULT_MODEL_NAMES[MODEL_NAME],
        "dataset": DATASET_NAME,
        "dataset_key": DATASET_KEY,
        "framework": profile["framework"],
        "device_type": profile["device_type"],
        "gpu": get_gpu_name(),
        "cuda_driver": get_cuda_driver(),
        "precision": PRECISION,
        "batch_size": 1,
        "input_height": PROFILE_SIZE,
        "input_width": PROFILE_SIZE,
        "parameters": int(profile["parameters"]) if math.isfinite(float(profile["parameters"])) else None,
        "params_m": safe_number(profile["parameters"], 1e6),
        "macs": int(profile["macs"]) if math.isfinite(float(profile["macs"])) else None,
        "macs_g": safe_number(profile["macs"], 1e9),
        "flops": int(profile["flops"]) if math.isfinite(float(profile["flops"])) else None,
        "flops_g": safe_number(profile["flops"], 1e9),
        "latency_mean_ms": timing["latency_mean_ms"],
        "latency_median_ms": timing["latency_median_ms"],
        "latency_std_ms": timing["latency_std_ms"],
        "latency_min_ms": timing["latency_min_ms"],
        "latency_p95_ms": timing["latency_p95_ms"],
        "fps": timing["fps"],
        "peak_gpu_memory_mb": safe_number(timing["peak_gpu_memory_mb"]),
        "peak_gpu_allocated_mb": safe_number(timing["peak_gpu_allocated_mb"]),
        "peak_gpu_reserved_mb": safe_number(timing["peak_gpu_reserved_mb"]),
        "model_size_mb": float(profile["model_size_mb"]),
        "complexity_tool": profile["complexity_tool"],
        "warmup_runs": WARMUP_RUNS,
        "timed_runs": TIMED_RUNS,
        "input_image": str(input_image_path),
        "output_image": str(output_image_path),
        "checkpoint_path": profile["checkpoint_path"],
        "project_source": str(PROJECT_SOURCE),
        "result_directory": str(RESULT_DIR),
        "total_script_seconds": float(elapsed_total),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "python_version": sys.version.split()[0],
        "extra": profile.get("extra", {}),
    }
    return result


def flatten_for_csv(result):
    return {
        key: value
        for key, value in result.items()
        if key != "extra" and not isinstance(value, (dict, list, tuple))
    }


def update_master_csv(row):
    MASTER_CSV.parent.mkdir(parents=True, exist_ok=True)
    if MASTER_CSV.exists():
        table = pd.read_csv(MASTER_CSV)
    else:
        table = pd.DataFrame()

    new_row = pd.DataFrame([row])
    if not table.empty and {"method", "dataset_key"}.issubset(table.columns):
        same = (table["method"] == row["method"]) & (table["dataset_key"] == row["dataset_key"])
        if OVERWRITE_MODEL_RESULT:
            table = table.loc[~same].copy()
    table = pd.concat([table, new_row], ignore_index=True)

    method_order = [
        "URetinex-Net++",
        "RetinexFormer",
        "KinD++",
        "RUAS",
        "RetinexNet",
        "Zero-DCE",
        "LIME",
        "RetinexTapetum (Proposed)",
    ]
    if "method" in table.columns:
        order_map = {name: index for index, name in enumerate(method_order)}
        table["_order"] = table["method"].map(order_map).fillna(999)
        table = table.sort_values(["_order", "timestamp_utc"]).drop(columns=["_order"])
    table.to_csv(MASTER_CSV, index=False)


def save_results(result):
    json_path = RESULT_DIR / "speed_metrics.json"
    csv_path = RESULT_DIR / "speed_metrics.csv"
    system_path = RESULT_DIR / "run_summary.txt"

    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    row = flatten_for_csv(result)
    pd.DataFrame([row]).to_csv(csv_path, index=False)
    update_master_csv(row)

    lines = [
        f"Method: {result['method']}",
        f"Dataset: {result['dataset']}",
        f"Device: {result['device_type']} / {result['gpu']}",
        f"Input: {result['batch_size']}x3x{result['input_height']}x{result['input_width']} {result['precision']}",
        f"Params (M): {result['params_m']}",
        f"MACs (G): {result['macs_g']}",
        f"FLOPs (G): {result['flops_g']}",
        f"Latency mean (ms): {result['latency_mean_ms']:.6f}",
        f"Latency median (ms): {result['latency_median_ms']:.6f}",
        f"Latency p95 (ms): {result['latency_p95_ms']:.6f}",
        f"FPS: {result['fps']:.6f}",
        f"Peak GPU memory (MB): {result['peak_gpu_memory_mb']}",
        f"Model size (MB): {result['model_size_mb']:.6f}",
        f"Checkpoint: {result['checkpoint_path']}",
        f"Timestamp UTC: {result['timestamp_utc']}",
    ]
    system_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, csv_path, system_path


# =============================================================================
# MAIN
# =============================================================================

def main():
    total_start = time.perf_counter()
    profile_source_image = find_first_test_image(DATASET_ROOT)
    _, rgb_array, saved_input_path = prepare_profile_image(profile_source_image)

    print(f"Profile source image: {profile_source_image}")
    print("Copying the selected project to local Colab storage...")
    local_project = copy_project_to_local(
        PROJECT_SOURCE,
        WORK_ROOT / MODEL_DIR_NAMES[MODEL_NAME],
    )

    if MODEL_NAME == "LIME":
        profile = run_lime_profile(rgb_array)
    elif MODEL_NAME == "KinD++":
        profile = run_kind_plus_profile(local_project, rgb_array)
    elif MODEL_NAME == "RetinexNet":
        profile = run_retinexnet_profile(local_project, rgb_array)
    else:
        profile = run_pytorch_profile(local_project, rgb_array)

    output_path = save_output_image(profile.pop("output_array"))
    result = build_result(
        profile,
        saved_input_path,
        output_path,
        time.perf_counter() - total_start,
    )
    json_path, csv_path, summary_path = save_results(result)

    print("\n" + "=" * 88)
    print("PROFILE COMPLETED")
    print("=" * 88)
    print(f"Method           : {result['method']}")
    print(f"Dataset          : {result['dataset']}")
    print(f"Params (M)       : {result['params_m']}")
    print(f"MACs (G)         : {result['macs_g']}")
    print(f"FLOPs (G)        : {result['flops_g']}")
    print(f"Latency mean (ms): {result['latency_mean_ms']:.4f}")
    print(f"Latency p95 (ms) : {result['latency_p95_ms']:.4f}")
    print(f"FPS              : {result['fps']:.4f}")
    print(f"Peak memory (MB) : {result['peak_gpu_memory_mb']}")
    print(f"Model size (MB)  : {result['model_size_mb']:.4f}")
    print(f"JSON             : {json_path}")
    print(f"CSV              : {csv_path}")
    print(f"Master CSV       : {MASTER_CSV}")
    print(f"Summary          : {summary_path}")
    print(f"Output image     : {output_path}")
    print("=" * 88)


try:
    main()
except Exception as error:
    error_path = RESULT_DIR / "profile_error.txt"
    error_text = (
        f"Model: {MODEL_NAME}\n"
        f"Dataset: {DATASET_NAME}\n"
        f"Time: {datetime.now(timezone.utc).isoformat()}\n\n"
        f"{traceback.format_exc()}"
    )
    error_path.write_text(error_text, encoding="utf-8")
    print(error_text)
    print(f"Error report saved to: {error_path}")
    raise

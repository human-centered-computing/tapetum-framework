import csv
import os
import subprocess
import sys
import time

from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torchvision import transforms

from config import (
    BASE_CHANNELS,
    CKPT_DIR,
    DEVICE,
    LAMBDA_INIT,
    LAMBDA_MAX,
    LPIPS_LOSS_RESIZE,
    PROJECT_ROOT,
    TEST_HIGH_DIR,
    TEST_LOW_DIR,
)
from model import RetinexTapetum


IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


def ensure_lpips():
    try:
        import lpips  # noqa: F401
    except ImportError:
        print("lpips not found. Installing lpips...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "lpips"])
    import lpips

    return lpips


def list_images(folder):
    return sorted([f for f in os.listdir(folder) if f.lower().endswith(IMG_EXTS)])


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def calc_psnr(pred, target):
    mse = F.mse_loss(pred, target).item()
    if mse == 0:
        return 100.0
    return 10.0 * torch.log10(torch.tensor(1.0 / mse)).item()


def create_gaussian_window(window_size, channel, device):
    sigma = 1.5
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window_2d = g[:, None] @ g[None, :]
    window_2d = window_2d.unsqueeze(0).unsqueeze(0)
    return window_2d.expand(channel, 1, window_size, window_size).contiguous()


def calc_ssim(pred, target, window_size=11):
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    channel = pred.size(1)
    window = create_gaussian_window(window_size, channel, pred.device)

    mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(target, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size // 2, groups=channel) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + 1e-8
    )
    return ssim_map.mean().item()


def calc_lpips(lpips_fn, pred, target, resize=None):
    pred_lpips = pred.clamp(0.0, 1.0)
    target_lpips = target.clamp(0.0, 1.0)

    if resize and resize > 0:
        size = (resize, resize)
        pred_lpips = F.interpolate(pred_lpips, size=size, mode="bilinear", align_corners=False)
        target_lpips = F.interpolate(target_lpips, size=size, mode="bilinear", align_corners=False)

    return lpips_fn(pred_lpips * 2.0 - 1.0, target_lpips * 2.0 - 1.0).mean().item()


def infer_base_channels(checkpoint):
    state = checkpoint.get("model", checkpoint)
    head_weight = state.get("decom_net.head.weight")
    if head_weight is None:
        return BASE_CHANNELS
    return int(head_weight.shape[0])


def load_model(device):
    ckpt_path = os.path.join(CKPT_DIR, "best.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
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
    ).to(device)

    try:
        model.load_state_dict(checkpoint["model"])
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint architecture mismatch. Train a fresh RetinexTapetum "
            "checkpoint with the current V2-quality config/model.py."
        ) from exc
    model.eval()

    print("Loaded checkpoint:", ckpt_path)
    print("Model BASE_CHANNELS:", checkpoint_base)
    print("Model LAMBDA_MAX:", LAMBDA_MAX)
    if "best_epoch" in checkpoint:
        print("Best epoch:", checkpoint["best_epoch"])
    if "best_metric" in checkpoint:
        print("Best metric from train:", checkpoint["best_metric"])
    if "best_score" in checkpoint:
        print("Best score from train:", checkpoint["best_score"])
    if "best_psnr" in checkpoint:
        print("Best checkpoint PSNR:", checkpoint["best_psnr"])
    if "best_lpips" in checkpoint:
        print("Best checkpoint LPIPS:", checkpoint["best_lpips"])

    return model, checkpoint, ckpt_path


@torch.inference_mode()
def benchmark_fps(model, device, size=256, warmup=20, runs=100):
    x = torch.rand(1, 3, size, size, device=device)
    for _ in range(warmup):
        model(x)
    sync(device)

    start = time.perf_counter()
    for _ in range(runs):
        model(x)
    sync(device)

    elapsed = time.perf_counter() - start
    return runs / elapsed


@torch.inference_mode()
def evaluate(model, lpips_fn, device):
    to_tensor = transforms.ToTensor()
    low_files = list_images(TEST_LOW_DIR)

    rows = []
    psnr_values = []
    ssim_values = []
    lpips_values = []
    lpips_train_values = []
    lambda_means = []
    lambda_mins = []
    lambda_maxs = []

    print("\n===== FULL TEST EVALUATION START =====")
    for fname in tqdm(low_files):
        low_path = os.path.join(TEST_LOW_DIR, fname)
        gt_path = os.path.join(TEST_HIGH_DIR, fname)
        if not os.path.exists(gt_path):
            continue

        low = to_tensor(Image.open(low_path).convert("RGB")).unsqueeze(0).to(device)
        gt = to_tensor(Image.open(gt_path).convert("RGB")).unsqueeze(0).to(device)

        output = model(low)
        pred = output["enhanced"].clamp(0.0, 1.0)

        psnr = calc_psnr(pred, gt)
        ssim = calc_ssim(pred, gt)
        lpips_val = calc_lpips(lpips_fn, pred, gt)
        lpips_train_val = calc_lpips(lpips_fn, pred, gt, resize=LPIPS_LOSS_RESIZE)

        lambda_map = output.get("lambda_map")
        if lambda_map is not None:
            lam_mean = lambda_map.mean().item()
            lam_min = lambda_map.min().item()
            lam_max = lambda_map.max().item()
        else:
            lam = output["lambda"].item()
            lam_mean = lam
            lam_min = lam
            lam_max = lam

        psnr_values.append(psnr)
        ssim_values.append(ssim)
        lpips_values.append(lpips_val)
        lpips_train_values.append(lpips_train_val)
        lambda_means.append(lam_mean)
        lambda_mins.append(lam_min)
        lambda_maxs.append(lam_max)

        rows.append(
            {
                "file": fname,
                "psnr": psnr,
                "ssim": ssim,
                "lpips": lpips_val,
                "lpips_train_resize": lpips_train_val,
                "lambda_mean": lam_mean,
                "lambda_min": lam_min,
                "lambda_max": lam_max,
            }
        )

    if not rows:
        raise RuntimeError("No matched low/ground-truth image pairs found.")

    mean = lambda values: sum(values) / len(values)
    return {
        "rows": rows,
        "matched": len(rows),
        "psnr": mean(psnr_values),
        "ssim": mean(ssim_values),
        "lpips": mean(lpips_values),
        "lpips_train_resize": mean(lpips_train_values),
        "lambda_mean": mean(lambda_means),
        "lambda_min": mean(lambda_mins),
        "lambda_max": mean(lambda_maxs),
    }


def save_csv(metrics, params_m, fps):
    metrics_dir = os.path.join(PROJECT_ROOT, "archive/metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    detail_path = os.path.join(metrics_dir, "RetinexTapetum_detail_metrics.csv")
    with open(detail_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file",
                "psnr",
                "ssim",
                "lpips",
                "lpips_train_resize",
                "lambda_mean",
                "lambda_min",
                "lambda_max",
            ],
        )
        writer.writeheader()
        writer.writerows(metrics["rows"])

    summary_path = os.path.join(metrics_dir, "RetinexTapetum_metrics.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "params_m",
                "fps",
                "psnr",
                "ssim",
                "lpips",
                "lpips_train_resize",
                "matched",
                "lambda_mean",
                "lambda_min",
                "lambda_max",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "method": "RetinexTapetum",
                "params_m": params_m,
                "fps": fps,
                "psnr": metrics["psnr"],
                "ssim": metrics["ssim"],
                "lpips": metrics["lpips"],
                "lpips_train_resize": metrics["lpips_train_resize"],
                "matched": metrics["matched"],
                "lambda_mean": metrics["lambda_mean"],
                "lambda_min": metrics["lambda_min"],
                "lambda_max": metrics["lambda_max"],
            }
        )

    return summary_path, detail_path


def main():
    device = torch.device(DEVICE)
    print("DEVICE:", device)
    print("TEST_LOW_DIR:", TEST_LOW_DIR)
    print("TEST_HIGH_DIR:", TEST_HIGH_DIR)
    print("CKPT_DIR:", CKPT_DIR)

    lpips = ensure_lpips()
    lpips_fn = lpips.LPIPS(net="alex").to(device).eval()

    model, _, _ = load_model(device)
    params_m = count_params(model) / 1_000_000
    fps = benchmark_fps(model, device)
    metrics = evaluate(model, lpips_fn, device)
    summary_path, detail_path = save_csv(metrics, params_m, fps)

    print("\n===== RetinexTapetum FINAL METRICS =====")
    print(f"Params (M) : {params_m:.4f}")
    print(f"FPS        : {fps:.4f}")
    print(f"PSNR       : {metrics['psnr']:.4f}")
    print(f"SSIM       : {metrics['ssim']:.4f}")
    print(f"LPIPS      : {metrics['lpips']:.4f}")
    print(f"LPIPS@{LPIPS_LOSS_RESIZE:<4}: {metrics['lpips_train_resize']:.4f}")
    print(f"Matched    : {metrics['matched']}")
    print(f"Lambda mean: {metrics['lambda_mean']:.4f}")
    print(f"Lambda min : {metrics['lambda_min']:.4f}")
    print(f"Lambda max : {metrics['lambda_max']:.4f}")
    print("\nCSV saved to:")
    print(summary_path)
    print(detail_path)
    print("\n===== LATEX TABLE ROW =====")
    print(
        "RetinexTapetum (Ours) & "
        f"{params_m:.4f} & {fps:.2f} & {metrics['psnr']:.2f} & "
        f"{metrics['ssim']:.3f} & {metrics['lpips']:.3f} \\\\"
    )


if __name__ == "__main__":
    main()

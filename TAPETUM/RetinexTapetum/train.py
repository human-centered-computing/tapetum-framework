"""
Training script for RetinexTapetum.

This script follows the same training/reporting structure as the reference
Retinex+Tapetum code so model outputs can be compared more fairly.
"""

import os
import random
import shutil
import subprocess
import sys
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader

from config import (
    TRAIN_LOW_DIR,
    TRAIN_HIGH_DIR,
    VAL_LOW_DIR,
    VAL_HIGH_DIR,
    USE_TRAIN_VAL_SPLIT,
    VAL_RATIO,
    SPLIT_SEED,
    CKPT_DIR,
    DEVICE,
    BATCH_SIZE,
    NUM_WORKERS,
    CROP_SIZE,
    EPOCHS,
    LR,
    MIN_LR,
    SEED,
    BASE_CHANNELS,
    LAMBDA_INIT,
    LAMBDA_MAX,
    PATIENCE,
    GRAD_CLIP_NORM,
    SHOW_PROGRESS_BARS,
    TRAIN_LOG_INTERVAL,
    VAL_LOG_INTERVAL,
    USE_LPIPS_LOSS,
    W_LPIPS,
    LPIPS_NET,
    LPIPS_METRIC_RESIZE,
    BEST_MODEL_METRIC,
    RESUME_TRAINING,
    RESUME_CKPT_NAME,
)
from dataset import LOLPairDataset, list_images
from model import RetinexTapetum
from losses import total_loss_fn
from utils import seed_everything, calc_psnr


def mirror_checkpoint_to_drive(src_path, mirror_dir):
    """Copy a checkpoint to Drive during Colab training when requested."""
    if not mirror_dir:
        return
    os.makedirs(mirror_dir, exist_ok=True)
    dst_path = os.path.join(mirror_dir, os.path.basename(src_path))
    shutil.copy2(src_path, dst_path)


def build_train_val_file_split(low_dir, high_dir, val_ratio, split_seed):
    """Create a deterministic file-level train/validation split from paired files."""
    low_files = list_images(low_dir)
    high_files = set(list_images(high_dir))
    files = [f for f in low_files if f in high_files]

    if not files:
        raise RuntimeError(
            "No paired train images found for train/validation split: "
            f"low_dir={low_dir}, high_dir={high_dir}"
        )
    if len(files) < 2:
        raise RuntimeError(
            "At least two paired train images are required for a train/validation split; "
            f"found {len(files)}."
        )

    rng = random.Random(split_seed)
    rng.shuffle(files)

    val_count = max(1, int(len(files) * val_ratio))
    val_count = min(val_count, len(files) - 1)
    val_files = sorted(files[:val_count])
    train_files = sorted(files[val_count:])

    print(f"Total paired train images: {len(files)}")
    print(f"Train split images: {len(train_files)}")
    print(f"Val split images: {len(val_files)}")
    print(f"VAL_RATIO: {val_ratio:.2f}")
    print(f"SPLIT_SEED: {split_seed}")

    return train_files, val_files


def build_perceptual_loss(device):
    """Create the frozen LPIPS network used by the perceptual training loss."""
    if not USE_LPIPS_LOSS or W_LPIPS <= 0:
        print("LPIPS training loss: disabled")
        return None

    try:
        import lpips
    except ImportError:
        print("lpips not found. Installing lpips...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "lpips"])
        import lpips

    perceptual_fn = lpips.LPIPS(net=LPIPS_NET).to(device).eval()
    for param in perceptual_fn.parameters():
        param.requires_grad_(False)

    print(f"LPIPS training loss: enabled ({LPIPS_NET}, weight={W_LPIPS})")
    return perceptual_fn


def calc_lpips_metric(pred, target, perceptual_fn, resize=0):
    """LPIPS metric for validation/checkpoint selection."""
    if perceptual_fn is None:
        return 0.0

    pred_metric = pred.clamp(0.0, 1.0)
    target_metric = target.clamp(0.0, 1.0)

    if resize and resize > 0:
        size = (resize, resize)
        pred_metric = F.interpolate(pred_metric, size=size, mode="bilinear", align_corners=False)
        target_metric = F.interpolate(target_metric, size=size, mode="bilinear", align_corners=False)

    return perceptual_fn(pred_metric * 2.0 - 1.0, target_metric * 2.0 - 1.0).mean().item()


def is_better_metric(metric_name, current, best):
    """Return True when current validation score improves the configured metric."""
    if metric_name.startswith("lpips"):
        return current < best
    return current > best


def initial_best_score(metric_name):
    return float("inf") if metric_name.startswith("lpips") else -1.0


def best_state_from_history(history, metric_name):
    """Rebuild best-metric fields from checkpoint history when possible."""
    best_score = initial_best_score(metric_name)
    best_epoch = 0
    best_psnr = -1.0
    best_lpips = float("inf")
    best_lpips_full = float("inf")

    for item in history:
        if not isinstance(item, dict):
            continue
        val_logs = item.get("val", {})
        if not isinstance(val_logs, dict):
            continue
        if metric_name not in val_logs:
            continue
        current_score = val_logs[metric_name]
        if is_better_metric(metric_name, current_score, best_score):
            best_score = current_score
            best_epoch = int(item.get("epoch", 0))
            best_psnr = val_logs.get("psnr", -1.0)
            best_lpips = val_logs.get("lpips", float("inf"))
            best_lpips_full = val_logs.get("lpips_full", float("inf"))

    if best_epoch == 0:
        return None

    return {
        "best_score": best_score,
        "best_epoch": best_epoch,
        "best_psnr": best_psnr,
        "best_lpips": best_lpips,
        "best_lpips_full": best_lpips_full,
    }


def load_resume_checkpoint(model, optimizer, scheduler, device):
    """Load training state from last.pth so Colab reruns continue safely."""
    resume_path = os.path.join(CKPT_DIR, RESUME_CKPT_NAME)
    state = {
        "start_epoch": 1,
        "best_score": initial_best_score(BEST_MODEL_METRIC),
        "best_epoch": 0,
        "best_psnr": -1.0,
        "best_lpips": float("inf"),
        "best_lpips_full": float("inf"),
        "no_improve": 0,
        "history": [],
        "resume_path": None,
    }

    if not RESUME_TRAINING:
        print("Resume disabled: RETINEX_TAPETUM_RESUME=0/config.RESUME_TRAINING=False")
        return state

    if not os.path.exists(resume_path):
        print(f"No resume checkpoint found at {resume_path}; starting from epoch 1.")
        return state

    print(f"Loading resume checkpoint: {resume_path}")
    checkpoint = torch.load(resume_path, map_location=device)
    if "model" not in checkpoint:
        raise ValueError(f"Resume checkpoint has no model state: {resume_path}")

    try:
        model.load_state_dict(checkpoint["model"])
    except RuntimeError as exc:
        raise RuntimeError(
            "Resume checkpoint architecture mismatch. Use a matching model/config "
            "or set RETINEX_TAPETUM_RESUME=0 for a clean restart."
        ) from exc

    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        print("Resume checkpoint has no optimizer state; optimizer starts fresh.")

    if "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
        # Let a changed config.EPOCHS remain the new total training target.
        scheduler.T_max = EPOCHS
    else:
        print("Resume checkpoint has no scheduler state; scheduler starts fresh.")

    completed_epoch = int(checkpoint.get("epoch", 0))
    history = checkpoint.get("history", [])
    if not isinstance(history, list):
        history = []

    best_state = best_state_from_history(history, BEST_MODEL_METRIC)
    if best_state is None and checkpoint.get("best_metric") == BEST_MODEL_METRIC:
        best_state = {
            "best_score": checkpoint.get("best_score", state["best_score"]),
            "best_epoch": int(checkpoint.get("best_epoch", 0)),
            "best_psnr": checkpoint.get("best_psnr", -1.0),
            "best_lpips": checkpoint.get("best_lpips", float("inf")),
            "best_lpips_full": checkpoint.get("best_lpips_full", float("inf")),
        }

    if best_state is None:
        print(
            "Could not rebuild previous best state for metric "
            f"{BEST_MODEL_METRIC}; best tracking will restart from the next validation."
        )
    else:
        state.update(best_state)

    state["start_epoch"] = completed_epoch + 1
    state["history"] = history
    state["resume_path"] = resume_path
    state["no_improve"] = max(0, completed_epoch - state["best_epoch"]) if state["best_epoch"] else 0

    print(
        "Resume state -> "
        f"completed_epoch: {completed_epoch}, next_epoch: {state['start_epoch']}, "
        f"best_epoch: {state['best_epoch']}, best_{BEST_MODEL_METRIC}: {state['best_score']:.4f}, "
        f"no_improve: {state['no_improve']}"
    )
    return state


def train_one_epoch(model, loader, optimizer, device, perceptual_fn=None, epoch=0):
    """Run one full training epoch and return averaged logs."""
    model.train()
    running = {
        "total": 0.0,
        "l1": 0.0,
        "ssim": 0.0,
        "color": 0.0,
        "chroma": 0.0,
        "attn": 0.0,
        "edge": 0.0,
        "dark_noise": 0.0,
        "lpips": 0.0,
        "decomp": 0.0,
        "recon_low": 0.0,
        "recon_high": 0.0,
        "reflect": 0.0,
        "smooth_low": 0.0,
        "smooth_high": 0.0,
        "smooth_enh": 0.0,
    }

    pbar = tqdm(loader, total=len(loader), desc="Train", leave=False, disable=not SHOW_PROGRESS_BARS)
    for batch_idx, batch in enumerate(pbar, start=1):
        low = batch["low"].to(device, non_blocking=True)
        high = batch["high"].to(device, non_blocking=True)

        # Forward with high image enabled so the model returns decomposition terms.
        optimizer.zero_grad()
        output = model(low, high)
        loss, logs = total_loss_fn(output, low, high, perceptual_fn=perceptual_fn)
        loss.backward()

        # Optional stabilization for harder runs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()

        for k in running:
            running[k] += logs[k]

        if TRAIN_LOG_INTERVAL and (
            batch_idx == 1 or batch_idx % TRAIN_LOG_INTERVAL == 0 or batch_idx == len(loader)
        ):
            avg_loss = running["total"] / batch_idx
            print(
                f"Epoch {epoch:03d} train {batch_idx:03d}/{len(loader):03d} | "
                f"avg_loss {avg_loss:.4f} | "
                f"batch_loss {logs['total']:.4f} | "
                f"lpips {logs['lpips']:.4f} | "
                f"edge {logs['edge']:.4f} | "
                f"chroma {logs['chroma']:.4f} | "
                f"lambda {output['lambda'].item():.4f}",
                flush=True,
            )

        pbar.set_postfix(
            {
                "loss": f"{logs['total']:.4f}",
                "edge": f"{logs['edge']:.4f}",
                "dark": f"{logs['dark_noise']:.4f}",
                "lpips": f"{logs['lpips']:.4f}",
                "chroma": f"{logs['chroma']:.4f}",
                "decomp": f"{logs['decomp']:.4f}",
                "sEnh": f"{logs['smooth_enh']:.4f}",
                "lam": f"{output['lambda'].item():.3f}",
            }
        )

    n = len(loader)
    for k in running:
        running[k] /= n
    return running


@torch.no_grad()
def validate(model, loader, device, perceptual_fn=None, epoch=0):
    """Run validation on the full validation split and return averaged logs."""
    model.eval()
    running = {
        "total": 0.0,
        "l1": 0.0,
        "ssim": 0.0,
        "color": 0.0,
        "chroma": 0.0,
        "attn": 0.0,
        "edge": 0.0,
        "dark_noise": 0.0,
        "lpips": 0.0,
        "lpips_full": 0.0,
        "decomp": 0.0,
        "recon_low": 0.0,
        "recon_high": 0.0,
        "reflect": 0.0,
        "smooth_low": 0.0,
        "smooth_high": 0.0,
        "smooth_enh": 0.0,
        "psnr": 0.0,
        "lambda_mean": 0.0,
        "lambda_min": 0.0,
        "lambda_max": 0.0,
    }

    pbar = tqdm(loader, total=len(loader), desc="Val", leave=False, disable=not SHOW_PROGRESS_BARS)
    for batch_idx, batch in enumerate(pbar, start=1):
        low = batch["low"].to(device, non_blocking=True)
        high = batch["high"].to(device, non_blocking=True)

        # Validation uses the full paired image and does not update weights.
        output = model(low, high)
        _, logs = total_loss_fn(output, low, high, perceptual_fn=perceptual_fn)
        psnr = calc_psnr(output["enhanced"], high)
        lpips_full = calc_lpips_metric(
            output["enhanced"],
            high,
            perceptual_fn,
            resize=LPIPS_METRIC_RESIZE,
        )
        lambda_map = output["lambda_map"]

        for k in running:
            if k not in ("psnr", "lpips_full", "lambda_mean", "lambda_min", "lambda_max"):
                running[k] += logs[k]
        running["psnr"] += psnr
        running["lpips_full"] += lpips_full
        running["lambda_mean"] += lambda_map.mean().item()
        running["lambda_min"] += lambda_map.min().item()
        running["lambda_max"] += lambda_map.max().item()

        if VAL_LOG_INTERVAL and (
            batch_idx == 1 or batch_idx % VAL_LOG_INTERVAL == 0 or batch_idx == len(loader)
        ):
            avg_psnr = running["psnr"] / batch_idx
            avg_lpips_full = running["lpips_full"] / batch_idx
            print(
                f"Epoch {epoch:03d} val {batch_idx:03d}/{len(loader):03d} | "
                f"avg_psnr {avg_psnr:.2f} | "
                f"avg_lpips_full {avg_lpips_full:.4f} | "
                f"sample_psnr {psnr:.2f} | "
                f"sample_lpips_full {lpips_full:.4f}",
                flush=True,
            )

        pbar.set_postfix(
            {
                "loss": f"{logs['total']:.4f}",
                "psnr": f"{psnr:.2f}",
                "edge": f"{logs['edge']:.4f}",
                "dark": f"{logs['dark_noise']:.4f}",
                "lpips": f"{logs['lpips']:.4f}",
                "lpipsFull": f"{lpips_full:.4f}",
                "chroma": f"{logs['chroma']:.4f}",
                "decomp": f"{logs['decomp']:.4f}",
                "sEnh": f"{logs['smooth_enh']:.4f}",
                "lam": f"{lambda_map.mean().item():.3f}",
            }
        )

    n = len(loader)
    for k in running:
        running[k] /= n
    return running


def main():
    seed_everything(SEED)
    os.makedirs(CKPT_DIR, exist_ok=True)

    print("DEVICE:", DEVICE)
    print("TRAIN_LOW_DIR :", TRAIN_LOW_DIR)
    print("TRAIN_HIGH_DIR:", TRAIN_HIGH_DIR)
    print("VAL_LOW_DIR   :", VAL_LOW_DIR)
    print("VAL_HIGH_DIR  :", VAL_HIGH_DIR)
    print("CKPT_DIR      :", CKPT_DIR)
    print("USE_TRAIN_VAL_SPLIT:", USE_TRAIN_VAL_SPLIT)
    print("VAL_RATIO     :", VAL_RATIO)
    print("SPLIT_SEED    :", SPLIT_SEED)
    print("BASE_CHANNELS :", BASE_CHANNELS)
    print("BATCH_SIZE    :", BATCH_SIZE)
    print("EPOCHS        :", EPOCHS)
    print("LR            :", LR)
    print("MIN_LR        :", MIN_LR)
    print("W_LPIPS       :", W_LPIPS)
    print("BEST_METRIC   :", BEST_MODEL_METRIC)
    print("RESUME        :", RESUME_TRAINING)
    print("RESUME_CKPT   :", RESUME_CKPT_NAME)
    print("TRAIN_LOG_INTERVAL:", TRAIN_LOG_INTERVAL)
    print("VAL_LOG_INTERVAL  :", VAL_LOG_INTERVAL)
    if RESUME_TRAINING:
        print(
            "WARNING: When changing train/validation split settings, use "
            "RETINEX_TAPETUM_RESUME=0 for a clean experimental run."
        )
    drive_ckpt_mirror_dir = os.environ.get("DRIVE_CKPT_MIRROR_DIR")
    if drive_ckpt_mirror_dir:
        print("DRIVE_CKPT_MIRROR_DIR:", drive_ckpt_mirror_dir)

    if USE_TRAIN_VAL_SPLIT:
        train_files, val_files = build_train_val_file_split(
            TRAIN_LOW_DIR,
            TRAIN_HIGH_DIR,
            VAL_RATIO,
            SPLIT_SEED,
        )
        train_dataset = LOLPairDataset(
            low_dir=TRAIN_LOW_DIR,
            high_dir=TRAIN_HIGH_DIR,
            crop_size=CROP_SIZE,
            training=True,
            file_list=train_files,
        )
        val_dataset = LOLPairDataset(
            low_dir=TRAIN_LOW_DIR,
            high_dir=TRAIN_HIGH_DIR,
            crop_size=CROP_SIZE,
            training=False,
            file_list=val_files,
        )
    else:
        train_dataset = LOLPairDataset(
            low_dir=TRAIN_LOW_DIR,
            high_dir=TRAIN_HIGH_DIR,
            crop_size=CROP_SIZE,
            training=True,
        )

        val_dataset = LOLPairDataset(
            low_dir=VAL_LOW_DIR,
            high_dir=VAL_HIGH_DIR,
            crop_size=CROP_SIZE,
            training=False,
        )

    split_info = {
        "use_train_val_split": USE_TRAIN_VAL_SPLIT,
        "val_ratio": VAL_RATIO,
        "split_seed": SPLIT_SEED,
        "train_files_count": len(train_dataset),
        "val_files_count": len(val_dataset),
    }

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    # Validation keeps batch_size=1 so metrics are averaged image-by-image.
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    print("Train batches:", len(train_loader))
    print("Val batches  :", len(val_loader))

    model = RetinexTapetum(
        base=BASE_CHANNELS,
        lambda_init=LAMBDA_INIT,
        lambda_max=LAMBDA_MAX,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=MIN_LR)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable params:", f"{num_params:,}")

    perceptual_fn = build_perceptual_loss(DEVICE)
    if BEST_MODEL_METRIC.startswith("lpips") and perceptual_fn is None:
        raise ValueError("LPIPS-based BEST_MODEL_METRIC requires USE_LPIPS_LOSS=True and W_LPIPS > 0.")

    resume_state = load_resume_checkpoint(model, optimizer, scheduler, DEVICE)
    start_epoch = resume_state["start_epoch"]
    best_score = resume_state["best_score"]
    best_epoch = resume_state["best_epoch"]
    best_psnr = resume_state["best_psnr"]
    best_lpips = resume_state["best_lpips"]
    best_lpips_full = resume_state["best_lpips_full"]
    no_improve = resume_state["no_improve"]
    history = resume_state["history"]

    if start_epoch > EPOCHS:
        print("\n===== TRAIN SKIPPED =====")
        print(f"Resume checkpoint already reached epoch {start_epoch - 1}; target EPOCHS is {EPOCHS}.")
        print("Increase EPOCHS or set RETINEX_TAPETUM_RESUME=0 for a clean restart.")
        print("Best epoch:", best_epoch)
        print(f"Best {BEST_MODEL_METRIC}:", best_score)
        print("Best PSNR:", best_psnr)
        print("Best LPIPS@loss:", best_lpips)
        print("Best LPIPS full:", best_lpips_full)
        print("Best checkpoint:", os.path.join(CKPT_DIR, "best.pth"))
        print("Last checkpoint:", os.path.join(CKPT_DIR, "last.pth"))
        return

    print("\n===== TRAIN START =====")
    for epoch in range(start_epoch, EPOCHS + 1):
        print(f"\nEpoch {epoch:03d}/{EPOCHS:03d} train start", flush=True)
        train_logs = train_one_epoch(
            model,
            train_loader,
            optimizer,
            DEVICE,
            perceptual_fn=perceptual_fn,
            epoch=epoch,
        )
        print(f"Epoch {epoch:03d}/{EPOCHS:03d} validation start", flush=True)
        val_logs = validate(
            model,
            val_loader,
            DEVICE,
            perceptual_fn=perceptual_fn,
            epoch=epoch,
        )
        scheduler.step()

        lam = val_logs["lambda_mean"]

        # First append current epoch history, then write checkpoints.
        # This avoids checkpoints lagging one epoch behind in their history field.
        history.append({
            "epoch": epoch,
            "train": train_logs,
            "val": val_logs,
            "lambda": lam,
            "use_train_val_split": split_info["use_train_val_split"],
            "val_ratio": split_info["val_ratio"],
            "split_seed": split_info["split_seed"],
            "train_files_count": split_info["train_files_count"],
            "val_files_count": split_info["val_files_count"],
        })

        log_str = (
            f"Epoch {epoch:03d} | "
            f"train_loss {train_logs['total']:.4f} | "
            f"val_loss {val_logs['total']:.4f} | "
            f"val_psnr {val_logs['psnr']:.2f} | "
            f"val_lpips {val_logs['lpips']:.4f} | "
            f"val_lpips_full {val_logs['lpips_full']:.4f} | "
            f"edge {val_logs['edge']:.4f} | "
            f"dark_noise {val_logs['dark_noise']:.4f} | "
            f"chroma {val_logs['chroma']:.4f} | "
            f"decomp {val_logs['decomp']:.4f} | "
            f"smooth_enh {val_logs['smooth_enh']:.4f} | "
            f"lambda_mean {lam:.4f} | "
            f"lambda_min {val_logs['lambda_min']:.4f} | "
            f"lambda_max {val_logs['lambda_max']:.4f}"
        )
        print(log_str)

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_metric": BEST_MODEL_METRIC,
            "best_score": best_score,
            "best_psnr": best_psnr,
            "best_lpips": best_lpips,
            "best_lpips_full": best_lpips_full,
            "best_epoch": best_epoch,
            "history": history,
            "lambda_mean": val_logs["lambda_mean"],
            "lambda_min": val_logs["lambda_min"],
            "lambda_max": val_logs["lambda_max"],
            "use_train_val_split": split_info["use_train_val_split"],
            "val_ratio": split_info["val_ratio"],
            "split_seed": split_info["split_seed"],
            "train_files_count": split_info["train_files_count"],
            "val_files_count": split_info["val_files_count"],
        }

        # Select the best checkpoint using the metric configured in config.py.
        current_score = val_logs[BEST_MODEL_METRIC]
        if is_better_metric(BEST_MODEL_METRIC, current_score, best_score):
            best_score = current_score
            best_epoch = epoch
            best_psnr = val_logs["psnr"]
            best_lpips = val_logs["lpips"]
            best_lpips_full = val_logs["lpips_full"]
            no_improve = 0
            ckpt["best_score"] = best_score
            ckpt["best_psnr"] = best_psnr
            ckpt["best_lpips"] = best_lpips
            ckpt["best_lpips_full"] = best_lpips_full
            ckpt["best_epoch"] = best_epoch
            best_path = os.path.join(CKPT_DIR, "best.pth")
            torch.save(ckpt, best_path)
            mirror_checkpoint_to_drive(best_path, drive_ckpt_mirror_dir)
            print(
                "Best model updated -> "
                f"Epoch: {best_epoch}, {BEST_MODEL_METRIC}: {best_score:.4f}, "
                f"PSNR: {best_psnr:.4f}, LPIPS@loss: {best_lpips:.4f}, "
                f"LPIPS full: {best_lpips_full:.4f}"
            )
        else:
            no_improve += 1

        last_path = os.path.join(CKPT_DIR, "last.pth")
        torch.save(ckpt, last_path)
        mirror_checkpoint_to_drive(last_path, drive_ckpt_mirror_dir)

        if no_improve >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            print(
                f"Best epoch: {best_epoch} | Best {BEST_MODEL_METRIC}: {best_score:.4f} | "
                f"Best PSNR: {best_psnr:.4f} | Best LPIPS@loss: {best_lpips:.4f} | "
                f"Best LPIPS full: {best_lpips_full:.4f}"
            )
            break

    print("===== TRAIN FINISHED =====")
    print("Best epoch:", best_epoch)
    print(f"Best {BEST_MODEL_METRIC}:", best_score)
    print("Best PSNR:", best_psnr)
    print("Best LPIPS@loss:", best_lpips)
    print("Best LPIPS full:", best_lpips_full)
    print("Best checkpoint:", os.path.join(CKPT_DIR, "best.pth"))
    print("Last checkpoint:", os.path.join(CKPT_DIR, "last.pth"))


if __name__ == "__main__":
    main()
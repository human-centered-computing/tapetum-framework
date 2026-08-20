"""Small helper utilities shared by train and test scripts."""

import math
import random
import numpy as np
import torch
import torch.nn.functional as F


def seed_everything(seed=42):
    """Seed Python, NumPy, and PyTorch for repeatable experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def calc_psnr(pred, target):
    """Compute PSNR assuming tensors are already in [0, 1]."""
    mse = F.mse_loss(pred, target).item()
    if mse == 0:
        return 100.0
    return 10 * math.log10(1.0 / mse)

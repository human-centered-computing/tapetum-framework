"""
Dataset utilities for paired LOL-v2 training.

This module ensures that low-light and normal-light images are matched by the
same filename so that every training sample is a valid pair.
"""

import os
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from config import IMG_EXTS


def list_images(folder):
    """Return all supported image filenames in sorted order."""
    return sorted([f for f in os.listdir(folder) if f.lower().endswith(IMG_EXTS)])


class LOLPairDataset(Dataset):
    """
    Paired dataset for LOL-v2 Real_captured.

    Each item returns:
        - low  : low-light RGB tensor  [3, H, W]
        - high : normal-light RGB tensor [3, H, W]
        - name : original filename

    During training:
        - images are resized if smaller than crop_size
        - random crop is applied
        - flip / rotation augmentation is applied

    During validation / test-style loading:
        - no random augmentation is applied
        - full image is returned
    """

    def __init__(self, low_dir, high_dir, crop_size=256, training=True, file_list=None):
        self.low_dir = low_dir
        self.high_dir = high_dir
        self.crop_size = crop_size
        self.training = training
        self.to_tensor = transforms.ToTensor()

        low_files = list_images(low_dir)
        high_files = set(list_images(high_dir))
        if file_list is not None:
            requested_files = sorted(file_list)
            low_file_set = set(low_files)
            self.files = [f for f in requested_files if f in low_file_set and f in high_files]
        else:
            # Pair images strictly by shared filename.
            self.files = [f for f in low_files if f in high_files]

        mode = "train" if self.training else "val/test"

        print(f"[Dataset] {low_dir}")
        print(f"Mode: {mode}")
        print(f"Split file_list: {'yes' if file_list is not None else 'no'}")
        print(f"Paired samples: {len(self.files)}")

    def __len__(self):
        return len(self.files)

    def _resize_if_needed(self, low, high):
        """
        Ensure both images are large enough for random cropping.

        If either spatial dimension is smaller than crop_size, both images are
        resized together to preserve pairing.
        """
        w, h = low.size
        if w < self.crop_size or h < self.crop_size:
            new_w = max(w, self.crop_size)
            new_h = max(h, self.crop_size)
            low = low.resize((new_w, new_h), Image.BICUBIC)
            high = high.resize((new_w, new_h), Image.BICUBIC)
        return low, high

    def _random_crop(self, low, high):
        """Apply the same random crop to the paired low/high images."""
        w, h = low.size
        x = random.randint(0, w - self.crop_size)
        y = random.randint(0, h - self.crop_size)
        low = low.crop((x, y, x + self.crop_size, y + self.crop_size))
        high = high.crop((x, y, x + self.crop_size, y + self.crop_size))
        return low, high

    def _augment(self, low, high):
        """Apply the same geometric augmentations to the paired images."""
        if random.random() < 0.5:
            low = low.transpose(Image.FLIP_LEFT_RIGHT)
            high = high.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            low = low.transpose(Image.FLIP_TOP_BOTTOM)
            high = high.transpose(Image.FLIP_TOP_BOTTOM)
        k = random.randint(0, 3)
        if k:
            low = low.rotate(90 * k)
            high = high.rotate(90 * k)
        return low, high

    def __getitem__(self, idx):
        fname = self.files[idx]
        low = Image.open(os.path.join(self.low_dir, fname)).convert("RGB")
        high = Image.open(os.path.join(self.high_dir, fname)).convert("RGB")

        if self.training:
            low, high = self._resize_if_needed(low, high)
            low, high = self._random_crop(low, high)
            low, high = self._augment(low, high)

        low = self.to_tensor(low)
        high = self.to_tensor(high)

        return {"low": low, "high": high, "name": fname}

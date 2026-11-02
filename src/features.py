"""Precompute frozen CNN features once so LSTM training is laptop-friendly."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    resnet18,
    resnet34,
    resnet50,
)

from .utils import ensure_dir, save_json

LOGGER = logging.getLogger(__name__)

_BACKBONES = {
    "resnet18": (resnet18, ResNet18_Weights.DEFAULT, 512),
    "resnet34": (resnet34, ResNet34_Weights.DEFAULT, 512),
    "resnet50": (resnet50, ResNet50_Weights.DEFAULT, 2048),
}


def build_backbone(name: str = "resnet18") -> tuple[nn.Module, int]:
    """Return a frozen residual backbone that outputs a 7x7 spatial grid."""
    if name not in _BACKBONES:
        raise ValueError(f"Unsupported backbone {name!r}; choose from {sorted(_BACKBONES)}")
    factory, weights, dim = _BACKBONES[name]
    model = factory(weights=weights)
    # Drop global average pooling and the classifier, keep layer4 features.
    backbone = nn.Sequential(*list(model.children())[:-2])
    backbone.eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    return backbone, int(dim)


def build_transform(image_size: int = 224) -> transforms.Compose:
    resize = int(round(image_size * 256 / 224))
    return transforms.Compose(
        [
            transforms.Resize(resize, antialias=True),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class FeatureImageDataset(Dataset):
    def __init__(self, image_ids: list[int], image_index: dict, raw_images: str | Path, image_size: int = 224):
        self.image_ids = image_ids
        self.image_index = image_index
        self.raw_images = Path(raw_images)
        self.transform = build_transform(image_size)

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> torch.Tensor:
        image_id = self.image_ids[index]
        path = self.raw_images / self.image_index[str(image_id)]["file_name"]
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            return self.transform(image)


def extract_features(
    image_ids: list[int],
    image_index: dict,
    raw_images: str | Path,
    output_dir: str | Path,
    backbone_name: str = "resnet18",
    batch_size: int = 32,
    image_size: int = 224,
    device: str | torch.device = "cpu",
) -> dict:
    """Extract and memory-map ``[N, 7, 7, C]`` spatial and ``[N, C]`` global features."""
    output_dir = ensure_dir(output_dir)
    device = torch.device(device)
    backbone, dim = build_backbone(backbone_name)
    backbone = backbone.to(device)
    dataset = FeatureImageDataset(image_ids, image_index, raw_images, image_size=image_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    count = len(image_ids)
    spatial_path = output_dir / "spatial.npy"
    global_path = output_dir / "global.npy"
    spatial = np.lib.format.open_memmap(spatial_path, mode="w+", dtype=np.float16, shape=(count, 7, 7, dim))
    global_features = np.lib.format.open_memmap(global_path, mode="w+", dtype=np.float16, shape=(count, dim))

    started = time.perf_counter()
    row = 0
    with torch.inference_mode():
        for batch_index, images in enumerate(loader):
            images = images.to(device, non_blocking=False)
            features = backbone(images)  # [B, C, 7, 7]
            spatial_batch = features.permute(0, 2, 3, 1).float().cpu().numpy().astype(np.float16)
            global_batch = features.mean(dim=(2, 3)).float().cpu().numpy().astype(np.float16)
            end = row + spatial_batch.shape[0]
            spatial[row:end] = spatial_batch
            global_features[row:end] = global_batch
            row = end
            if (batch_index + 1) % 20 == 0 or row == count:
                elapsed = time.perf_counter() - started
                rate = row / elapsed if elapsed else 0.0
                LOGGER.info("Features %d/%d (%.1f img/s, %.1f%% done)", row, count, rate, 100.0 * row / count)

    spatial.flush()
    global_features.flush()
    del spatial, global_features
    if device.type == "mps":
        torch.mps.empty_cache()

    elapsed = time.perf_counter() - started
    save_json(image_ids, output_dir / "feature_ids.json")
    meta = {
        "backbone": backbone_name,
        "spatial_dim": int(dim),
        "grid_size": 7,
        "image_size": int(image_size),
        "count": int(count),
        "dtype": "float16",
        "seconds": round(elapsed, 2),
        "images_per_second": round(count / elapsed, 2) if elapsed else None,
        "shape_spatial": [int(count), 7, 7, int(dim)],
        "shape_global": [int(count), int(dim)],
    }
    save_json(meta, output_dir / "feature_meta.json")
    LOGGER.info("Saved features to %s in %.1fs", output_dir, elapsed)
    return meta


def validate_features(directory: str | Path) -> dict:
    directory = Path(directory)
    meta = json.loads((directory / "feature_meta.json").read_text())
    spatial = np.load(directory / "spatial.npy", mmap_mode="r")
    global_features = np.load(directory / "global.npy", mmap_mode="r")
    ids = json.loads((directory / "feature_ids.json").read_text())
    if spatial.shape[0] != len(ids) or global_features.shape[0] != len(ids):
        raise RuntimeError("Feature arrays and ids disagree")
    if not np.isfinite(np.asarray(spatial[:2], dtype=np.float32)).all():
        raise RuntimeError("Non-finite spatial features detected")
    return meta

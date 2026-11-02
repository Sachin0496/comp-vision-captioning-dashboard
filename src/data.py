"""COCO data preparation, feature stores, and captioning datasets."""

from __future__ import annotations

import json
import logging
import random
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .text import Vocab, caption_length_stats
from .utils import ensure_dir, load_json, project_path, save_json

LOGGER = logging.getLogger(__name__)

CAPTIONS_FILE = "captions_val2017.json"
INSTANCES_FILE = "instances_val2017.json"


# ---------------------------------------------------------------------------
# Raw COCO loading
# ---------------------------------------------------------------------------
def _extract_if_needed(images_dir: Path, annotation_dir: Path) -> None:
    """Extract the COCO archives if their contents are absent."""
    if not images_dir.exists() or not any(images_dir.glob("*.jpg")):
        archive = project_path("data", "raw", "val2017.zip")
        if not archive.exists():
            raise FileNotFoundError(
                f"COCO images not found at {images_dir} and archive missing at {archive}. "
                "Run scripts/00_download_data.sh first."
            )
        LOGGER.info("Extracting %s ...", archive)
        ensure_dir(images_dir.parent)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(images_dir.parent)

    if not (annotation_dir / CAPTIONS_FILE).exists():
        archive = project_path("data", "raw", "annotations_trainval2017.zip")
        if not archive.exists():
            raise FileNotFoundError(f"Annotations missing and archive not found at {archive}")
        LOGGER.info("Extracting %s ...", archive)
        ensure_dir(annotation_dir.parent)
        # The archive's root folder is ``annotations/``.
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(annotation_dir.parent)
        if not (annotation_dir / CAPTIONS_FILE).exists():
            raise RuntimeError(f"Extraction did not produce {annotation_dir / CAPTIONS_FILE}")


def load_captions(path: str | Path) -> dict[int, list[str]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    captions: dict[int, list[str]] = defaultdict(list)
    for annotation in payload["annotations"]:
        captions[int(annotation["image_id"])].append(annotation["caption"].strip())
    for image_id in captions:
        captions[image_id] = sorted(captions[image_id])
    return dict(captions)


def load_instances(path: str | Path) -> tuple[dict[int, list[str]], list[str], dict[str, str], dict[int, dict]]:
    """Return ground-truth object names, canonical classes, mapping and image info."""
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    categories = sorted(payload["categories"], key=lambda item: int(item["id"]))
    names = [str(category["name"]) for category in categories]
    coco_id_to_name = {str(category["id"]): str(category["name"]) for category in categories}
    category_id_to_name = {int(category["id"]): str(category["name"]) for category in categories}

    objects: dict[int, set[str]] = defaultdict(set)
    for image in payload["images"]:
        objects.setdefault(int(image["id"]), set())
    for annotation in payload["annotations"]:
        name = category_id_to_name.get(int(annotation["category_id"]))
        if name:
            objects[int(annotation["image_id"])].add(name)
    image_info = {
        int(image["id"]): {
            "file_name": str(image["file_name"]),
            "width": int(image["width"]),
            "height": int(image["height"]),
        }
        for image in payload["images"]
    }
    return {key: sorted(value) for key, value in objects.items()}, names, coco_id_to_name, image_info


def make_splits(image_ids: Sequence[int], sizes: Sequence[int], seed: int = 42) -> dict[str, list[int]]:
    """Deterministic train/val/test split over the available images."""
    ids = sorted(int(image_id) for image_id in image_ids)
    if sum(sizes) != len(ids):
        raise ValueError(f"Split sizes {sizes} sum to {sum(sizes)}, but {len(ids)} images are available")
    rng = random.Random(seed)
    rng.shuffle(ids)
    train_end = sizes[0]
    val_end = train_end + sizes[1]
    return {
        "train": sorted(ids[:train_end]),
        "val": sorted(ids[train_end:val_end]),
        "test": sorted(ids[val_end:]),
    }


def prepare_dataset(config: dict, force: bool = False) -> dict:
    """Create splits, vocabulary, captions, GT objects and metadata on disk."""
    raw_images = project_path(config["data"]["raw_images"])
    annotation_dir = project_path(config["data"]["annotations"])
    processed = ensure_dir(project_path(config["data"]["processed"]))
    split_path = processed / "splits.json"

    _extract_if_needed(raw_images, annotation_dir)

    captions = load_captions(annotation_dir / CAPTIONS_FILE)
    objects_gt, class_names, coco_id_to_name, image_info = load_instances(annotation_dir / INSTANCES_FILE)
    common_ids = sorted(set(captions) & set(objects_gt) & set(image_info))
    if len(common_ids) != 5000:
        raise RuntimeError(f"Expected 5000 COCO val images, found {len(common_ids)}")

    missing_files = [image_id for image_id in common_ids if not (raw_images / image_info[image_id]["file_name"]).exists()]
    if missing_files:
        raise FileNotFoundError(f"{len(missing_files)} image files are missing, e.g. {missing_files[:5]}")

    sizes = [int(value) for value in config["data"]["splits"]]
    splits = make_splits(common_ids, sizes, seed=int(config["seed"]))
    vocab = Vocab.build(
        (caption for image_id in splits["train"] for caption in captions[image_id]),
        min_freq=int(config["data"]["min_word_freq"]),
        max_len=int(config["data"]["max_caption_len"]),
    )

    all_captions = {str(image_id): captions[image_id] for image_id in common_ids}
    all_objects = {str(image_id): objects_gt[image_id] for image_id in common_ids}
    split_lookup = {image_id: name for name, ids in splits.items() for image_id in ids}
    image_index = {
        str(image_id): {
            **image_info[image_id],
            "split": split_lookup[image_id],
        }
        for image_id in common_ids
    }
    categories = {
        "names": class_names,
        "name_to_idx": {name: idx + 1 for idx, name in enumerate(class_names)},
        "coco_id_to_name": coco_id_to_name,
    }

    train_captions = [caption for image_id in splits["train"] for caption in captions[image_id]]
    stats = {
        "n_images": len(common_ids),
        "splits": {name: len(ids) for name, ids in splits.items()},
        "vocab_size": len(vocab),
        "vocab_coverage_train": vocab.coverage(train_captions),
        "caption_lengths": caption_length_stats(train_captions),
        "captions_per_image": dict(Counter(len(captions[image_id]) for image_id in common_ids)),
        "objects_per_image": {
            "mean": float(np.mean([len(objects_gt[image_id]) for image_id in common_ids])),
            "min": int(min(len(objects_gt[image_id]) for image_id in common_ids)),
            "max": int(max(len(objects_gt[image_id]) for image_id in common_ids)),
        },
        "object_class_count": len(class_names),
    }

    save_json(splits, split_path)
    save_json(all_captions, processed / "captions.json")
    save_json(all_objects, processed / "objects_gt.json")
    save_json(categories, processed / "categories.json")
    save_json(image_index, processed / "image_index.json")
    save_json(stats, processed / "stats.json")
    vocab.save(processed / "vocab.json")
    LOGGER.info(
        "Prepared %d images (%s); vocab=%d; train coverage=%.2f%%",
        len(common_ids),
        ", ".join(f"{name}={len(ids)}" for name, ids in splits.items()),
        len(vocab),
        100.0 * (1.0 - stats["vocab_coverage_train"]["unknown_rate"]),
    )
    return {"splits": splits, "vocab": vocab, "stats": stats, "categories": categories}


# ---------------------------------------------------------------------------
# Feature store
# ---------------------------------------------------------------------------
class FeatureStore:
    """Read-only access to precomputed ResNet spatial/global features."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        meta_path = self.directory / "feature_meta.json"
        ids_path = self.directory / "feature_ids.json"
        if not meta_path.exists() or not ids_path.exists():
            raise FileNotFoundError(
                f"Features not found in {self.directory}. Run scripts/02_extract_features.py first."
            )
        self.meta = load_json(meta_path)
        self.image_ids: list[int] = [int(value) for value in load_json(ids_path)]
        self.id_to_row = {image_id: row for row, image_id in enumerate(self.image_ids)}
        spatial_path = self.directory / "spatial.npy"
        self.spatial = np.load(spatial_path, mmap_mode="r")
        self.global_features = np.load(self.directory / "global.npy")
        if self.spatial.shape[0] != len(self.image_ids):
            raise RuntimeError("Feature row count does not match feature_ids.json")

    def __len__(self) -> int:
        return len(self.image_ids)

    def get(self, image_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.id_to_row[int(image_id)]
        spatial = torch.from_numpy(np.asarray(self.spatial[row], dtype=np.float32)).view(-1, self.spatial.shape[-1])
        global_feature = torch.from_numpy(np.asarray(self.global_features[row], dtype=np.float32))
        return spatial, global_feature


# ---------------------------------------------------------------------------
# Datasets and collation
# ---------------------------------------------------------------------------
def _load_detection_arrays(
    image_id: int,
    detections: dict | None,
    name_to_idx: dict[str, int],
    max_objects: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not detections:
        return torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.float32)
    record = detections.get(str(image_id)) or {}
    labels: list[int] = []
    scores: list[float] = []
    for name, score in zip(record.get("labels", []), record.get("scores", [])):
        idx = name_to_idx.get(name)
        if idx is None:
            continue
        labels.append(int(idx))
        scores.append(float(score))
    if not labels:
        return torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.float32)
    order = np.argsort(scores)[::-1][:max_objects]
    return (
        torch.tensor([labels[i] for i in order], dtype=torch.long),
        torch.tensor([scores[i] for i in order], dtype=torch.float32),
    )


class CaptionDataset(Dataset):
    """Caption dataset backed by precomputed CNN features.

    Each item is one (image, caption) pair.  Setting ``captions_per_image=1``
    yields one deterministic caption per image, used for loss evaluation and
    generation; ``None`` yields all five training captions.
    """

    def __init__(
        self,
        image_ids: Sequence[int],
        captions: dict,
        feature_store: FeatureStore,
        vocab: Vocab,
        detections: dict | None = None,
        name_to_idx: dict[str, int] | None = None,
        captions_per_image: int | None = None,
        max_len: int = 22,
        max_objects: int = 12,
        training: bool = False,
    ) -> None:
        self.image_ids = [int(value) for value in image_ids]
        self.captions = captions
        self.features = feature_store
        self.vocab = vocab
        self.detections = detections
        self.name_to_idx = name_to_idx or {}
        self.max_len = int(max_len)
        self.max_objects = int(max_objects)
        self.training = bool(training)
        self.items: list[tuple[int, str]] = []
        for image_id in self.image_ids:
            image_captions = list(captions[str(image_id)])
            if captions_per_image is not None:
                image_captions = image_captions[: int(captions_per_image)]
            self.items.extend((image_id, caption) for caption in image_captions)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        image_id, caption = self.items[index]
        spatial, global_feature = self.features.get(image_id)
        tokens = torch.tensor(self.vocab.encode(caption, self.max_len), dtype=torch.long)
        obj_labels, obj_scores = _load_detection_arrays(
            image_id, self.detections, self.name_to_idx, self.max_objects
        )
        return {
            "image_id": int(image_id),
            "caption": caption,
            "tokens": tokens,
            "spatial": spatial,
            "global": global_feature,
            "obj_labels": obj_labels,
            "obj_scores": obj_scores,
        }


def collate_captions(batch: list[dict]) -> dict:
    """Pad captions and object lists into dense tensors."""
    max_len = max(int(item["tokens"].numel()) for item in batch)
    tokens = torch.zeros(len(batch), max_len, dtype=torch.long)
    lengths = torch.zeros(len(batch), dtype=torch.long)
    for row, item in enumerate(batch):
        seq = item["tokens"]
        tokens[row, : seq.numel()] = seq
        lengths[row] = seq.numel()

    max_objects = max(1, max(int(item["obj_labels"].numel()) for item in batch))
    obj_labels = torch.zeros(len(batch), max_objects, dtype=torch.long)
    obj_scores = torch.zeros(len(batch), max_objects, dtype=torch.float32)
    obj_mask = torch.zeros(len(batch), max_objects, dtype=torch.bool)
    for row, item in enumerate(batch):
        count = int(item["obj_labels"].numel())
        if count:
            obj_labels[row, :count] = item["obj_labels"]
            obj_scores[row, :count] = item["obj_scores"]
            obj_mask[row, :count] = True

    return {
        "image_id": torch.tensor([item["image_id"] for item in batch], dtype=torch.long),
        "captions": [item["caption"] for item in batch],
        "tokens": tokens,
        "lengths": lengths,
        "spatial": torch.stack([item["spatial"] for item in batch]),
        "global": torch.stack([item["global"] for item in batch]),
        "obj_labels": obj_labels,
        "obj_scores": obj_scores,
        "obj_mask": obj_mask,
    }


class GenerationDataset(Dataset):
    """One item per image with references and features for caption generation."""

    def __init__(
        self,
        image_ids: Sequence[int],
        captions: dict,
        feature_store: FeatureStore,
        detections: dict | None = None,
        name_to_idx: dict[str, int] | None = None,
        max_objects: int = 12,
    ) -> None:
        self.image_ids = [int(value) for value in image_ids]
        self.captions = captions
        self.features = feature_store
        self.detections = detections
        self.name_to_idx = name_to_idx or {}
        self.max_objects = int(max_objects)

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> dict:
        image_id = self.image_ids[index]
        spatial, global_feature = self.features.get(image_id)
        obj_labels, obj_scores = _load_detection_arrays(
            image_id, self.detections, self.name_to_idx, self.max_objects
        )
        return {
            "image_id": int(image_id),
            "references": list(self.captions[str(image_id)]),
            "spatial": spatial,
            "global": global_feature,
            "obj_labels": obj_labels,
            "obj_scores": obj_scores,
        }


def collate_generation(batch: list[dict]) -> dict:
    max_objects = max(1, max(int(item["obj_labels"].numel()) for item in batch))
    obj_labels = torch.zeros(len(batch), max_objects, dtype=torch.long)
    obj_scores = torch.zeros(len(batch), max_objects, dtype=torch.float32)
    obj_mask = torch.zeros(len(batch), max_objects, dtype=torch.bool)
    for row, item in enumerate(batch):
        count = int(item["obj_labels"].numel())
        if count:
            obj_labels[row, :count] = item["obj_labels"]
            obj_scores[row, :count] = item["obj_scores"]
            obj_mask[row, :count] = True
    return {
        "image_id": torch.tensor([item["image_id"] for item in batch], dtype=torch.long),
        "references": [item["references"] for item in batch],
        "spatial": torch.stack([item["spatial"] for item in batch]),
        "global": torch.stack([item["global"] for item in batch]),
        "obj_labels": obj_labels,
        "obj_scores": obj_scores,
        "obj_mask": obj_mask,
    }


def load_processed_bundle(config: dict) -> dict:
    """Load splits, captions, GT objects, categories, vocab and feature store."""
    processed = project_path(config["data"]["processed"])
    splits = load_json(processed / "splits.json")
    captions = load_json(processed / "captions.json")
    gt_objects = load_json(processed / "objects_gt.json")
    categories = load_json(processed / "categories.json")
    vocab = Vocab.load(processed / "vocab.json")
    features = FeatureStore(project_path("data", "processed", "features"))
    detections_path = processed / "detections.json"
    detections = load_json(detections_path) if detections_path.exists() else None
    return {
        "splits": splits,
        "captions": captions,
        "gt_objects": gt_objects,
        "categories": categories,
        "vocab": vocab,
        "features": features,
        "detections": detections,
    }


def load_image(image_id: int, image_index: dict, raw_images: str | Path) -> Image.Image:
    record = image_index[str(image_id)]
    return Image.open(Path(raw_images) / record["file_name"]).convert("RGB")

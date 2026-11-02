"""Object detection used to produce auxiliary object-label inputs.

The detector is a small, COCO-pretrained Faster R-CNN (MobileNetV3-320).  It is
deliberately independent of the captioning model: its labels are the only
object information the object-aware captioner sees, so detector errors are part
of the experiment rather than hidden by ground-truth annotations.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_320_fpn,
)

from .utils import ensure_dir, load_json, save_json

LOGGER = logging.getLogger(__name__)

DETECTOR_NAME = "fasterrcnn_mobilenet_v3_large_320_fpn"


class ObjectDetector:
    def __init__(self, device: str | torch.device = "cpu", max_side: int = 512) -> None:
        self.device = torch.device(device)
        self.weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
        self.model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=self.weights)
        self.model.to(self.device).eval()
        self.categories = list(self.weights.meta["categories"])
        self.max_side = int(max_side)
        self.to_tensor = transforms.ToTensor()

    def _load(self, path: Path) -> torch.Tensor:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            width, height = image.size
            scale = min(1.0, self.max_side / max(width, height))
            if scale < 1.0:
                image = image.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.BILINEAR)
            return self.to_tensor(image)

    @torch.inference_mode()
    def predict(self, paths: list[Path], score_thresh: float = 0.3, max_detections: int = 12) -> list[dict]:
        tensors = [self._load(path).to(self.device) for path in paths]
        outputs = self.model(tensors)
        results: list[dict] = []
        for output in outputs:
            scores = output["scores"].detach().cpu()
            labels = output["labels"].detach().cpu()
            boxes = output["boxes"].detach().cpu()
            keep = scores >= float(score_thresh)
            scores, labels, boxes = scores[keep], labels[keep], boxes[keep]
            if scores.numel() > max_detections:
                top = torch.topk(scores, max_detections).indices
                scores, labels, boxes = scores[top], labels[top], boxes[top]
            names = [self.categories[int(label)] for label in labels]
            results.append(
                {
                    "labels": names,
                    "scores": [round(float(score), 4) for score in scores],
                    "boxes": [[round(float(value), 1) for value in box] for box in boxes],
                }
            )
        return results


def detect_dataset(
    image_ids: list[int],
    image_index: dict,
    raw_images: str | Path,
    output_path: str | Path,
    device: str | torch.device = "cpu",
    score_thresh: float = 0.3,
    max_detections: int = 12,
    batch_size: int = 4,
    max_side: int = 512,
    limit: int | None = None,
    resume: bool = True,
) -> dict:
    """Run the detector and store labels/scores/boxes per image as JSON."""
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    raw_images = Path(raw_images)
    ids = image_ids[:limit] if limit else image_ids
    detections: dict[str, dict] = {}
    if resume and output_path.exists():
        detections = load_json(output_path)
        LOGGER.info("Resuming: %d images already detected", len(detections))

    pending = [image_id for image_id in ids if str(image_id) not in detections]
    if not pending:
        LOGGER.info("All requested images already have detections")
        return detections

    detector = ObjectDetector(device=device, max_side=max_side)
    started = time.perf_counter()
    processed = 0
    for start in range(0, len(pending), batch_size):
        batch_ids = pending[start : start + batch_size]
        paths = [raw_images / image_index[str(image_id)]["file_name"] for image_id in batch_ids]
        results = detector.predict(paths, score_thresh=score_thresh, max_detections=max_detections)
        for image_id, result in zip(batch_ids, results):
            detections[str(image_id)] = result
        processed += len(batch_ids)
        if processed % 100 == 0 or processed == len(pending):
            elapsed = time.perf_counter() - started
            rate = processed / elapsed if elapsed else 0.0
            remaining = (len(pending) - processed) / rate if rate else 0.0
            LOGGER.info(
                "Detector %d/%d (%.1f img/s, ~%.0f min left)",
                processed,
                len(pending),
                rate,
                remaining / 60.0,
            )
        if processed % 500 == 0:
            save_json(detections, output_path)
        if detector.device.type == "mps":
            torch.mps.empty_cache()

    save_json(detections, output_path)
    meta = {
        "detector": DETECTOR_NAME,
        "score_threshold": float(score_thresh),
        "max_detections": int(max_detections),
        "max_side": int(max_side),
        "images": len(detections),
        "seconds": round(time.perf_counter() - started, 1),
    }
    save_json(meta, output_path.with_name("detections_meta.json"))
    LOGGER.info("Saved %d detection records to %s", len(detections), output_path)
    return detections


def detections_to_objects(detections: dict, min_score: float = 0.3) -> dict[str, list[str]]:
    """Unique detected labels per image, for descriptive statistics."""
    summary: dict[str, list[str]] = {}
    for image_id, record in detections.items():
        labels = [label for label, score in zip(record["labels"], record["scores"]) if score >= min_score]
        summary[image_id] = sorted(set(labels))
    return summary

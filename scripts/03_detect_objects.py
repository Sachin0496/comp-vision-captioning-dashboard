#!/usr/bin/env python
"""Precompute detector object labels for every image."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.detector import detect_dataset  # noqa: E402
from src.utils import configure_runtime, load_config, load_json, project_path, save_json, setup_logging  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default=None, choices=["auto", "cpu", "mps"])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    setup_logging(log_file=ROOT / "logs" / "detect_objects.log")
    config = load_config(args.config)
    device = configure_runtime(
        args.device or config["runtime"]["device"],
        mps_memory_fraction=float(config["runtime"]["mps_memory_fraction"]),
        torch_threads=int(config["runtime"]["torch_threads"]),
    )
    processed = project_path(config["data"]["processed"])
    splits = load_json(processed / "splits.json")
    image_index = load_json(processed / "image_index.json")
    image_ids = sorted({int(value) for ids in splits.values() for value in ids})
    output_path = Path(args.output) if args.output else processed / "detections.json"
    detections = detect_dataset(
        image_ids=image_ids,
        image_index=image_index,
        raw_images=project_path(config["data"]["raw_images"]),
        output_path=output_path,
        device=device,
        score_thresh=float(config["detector"]["score_thresh"]),
        max_detections=int(config["detector"]["max_detections"]),
        batch_size=int(args.batch_size or config["detector"]["batch_size"]),
        max_side=int(config["detector"]["max_side"]),
        limit=args.limit,
        resume=not args.no_resume,
    )
    if args.limit:
        save_json(detections, output_path)


if __name__ == "__main__":
    main()

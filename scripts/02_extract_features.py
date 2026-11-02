#!/usr/bin/env python
"""Extract frozen ResNet features for all images (once)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features import extract_features, validate_features  # noqa: E402
from src.utils import configure_runtime, load_config, load_json, project_path, setup_logging  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--limit", type=int, default=None, help="Smoke-test on the first N images")
    parser.add_argument("--output", default=None, help="Override output directory")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default=None, choices=["auto", "cpu", "mps"])
    args = parser.parse_args()

    setup_logging(log_file=ROOT / "logs" / "extract_features.log")
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
    if args.limit:
        image_ids = image_ids[: args.limit]

    output_dir = Path(args.output) if args.output else processed / "features"
    meta_path = output_dir / "feature_meta.json"
    if meta_path.exists() and not args.overwrite and not args.limit:
        meta = validate_features(output_dir)
        if meta["count"] == len(image_ids) and meta.get("backbone") == config["features"]["backbone"]:
            print(f"Features already complete at {output_dir}: {meta}")
            return

    extract_features(
        image_ids=image_ids,
        image_index=image_index,
        raw_images=project_path(config["data"]["raw_images"]),
        output_dir=output_dir,
        backbone_name=config["features"]["backbone"],
        batch_size=int(config["features"]["batch_size"]),
        image_size=int(config["data"]["image_size"]),
        device=device,
    )


if __name__ == "__main__":
    main()

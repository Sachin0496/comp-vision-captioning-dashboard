#!/usr/bin/env python
"""Prepare COCO splits, vocabulary, ground-truth objects and metadata."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import prepare_dataset  # noqa: E402
from src.utils import load_config, save_json, setup_logging  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--force", action="store_true", help="Recompute even if processed files exist")
    args = parser.parse_args()

    setup_logging(log_file=ROOT / "logs" / "prepare_data.log")
    config = load_config(args.config)
    split_path = ROOT / config["data"]["processed"] / "splits.json"
    if split_path.exists() and not args.force:
        print(f"Processed data already exists at {split_path.parent} (use --force to rebuild)")
        return

    bundle = prepare_dataset(config, force=args.force)
    save_json(bundle["stats"], ROOT / config["data"]["processed"] / "stats.json")
    print("Preparation complete.")


if __name__ == "__main__":
    main()

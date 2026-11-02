#!/usr/bin/env python
"""Create the report figures: attention maps, comparisons and examples."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
sys.path.insert(0, str(ROOT))

from src.hallucination import extract_mentions  # noqa: E402
from src.utils import load_config, load_json, project_path, setup_logging  # noqa: E402
from src.visualize import (  # noqa: E402
    plot_bleu_hallucination,
    plot_model_comparison,
    plot_qualitative_examples,
    plot_training_curves,
    visualize_attention_samples,
)

MODELS = ["global", "attention", "object"]


def find_highlight_indices(caption: str, words: list[str]) -> list[int]:
    """Map CHAIR object mentions back to generated-word indices."""
    indices: list[int] = []
    lowered = [word.lower() for word in words]
    for mention in extract_mentions(caption):
        surface = mention["surface"].split()
        for start in range(0, len(lowered) - len(surface) + 1):
            if lowered[start : start + len(surface)] == surface:
                indices.extend(range(start, start + len(surface)))
                break
    # Preserve order and cap at five highlighted words.
    seen: list[int] = []
    for index in indices:
        if index not in seen:
            seen.append(index)
    return seen[:5]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--predictions-dir", default=str(ROOT / "outputs" / "predictions"))
    parser.add_argument("--metrics-dir", default=str(ROOT / "outputs" / "metrics"))
    parser.add_argument("--checkpoint-dir", default=str(ROOT / "outputs" / "checkpoints"))
    parser.add_argument("--figures-dir", default=str(ROOT / "report" / "figures"))
    args = parser.parse_args()

    setup_logging(log_file=ROOT / "logs" / "visualize.log")
    config = load_config(args.config)
    figures = Path(args.figures_dir)
    figures.mkdir(parents=True, exist_ok=True)
    image_index = load_json(project_path(config["data"]["processed"], "image_index.json"))
    raw_images = project_path(config["data"]["raw_images"])

    summary_path = Path(args.metrics_dir) / "model_comparison.json"
    if summary_path.exists():
        summary = load_json(summary_path)
        print("Model comparison ->", plot_model_comparison(summary, figures / "model_comparison.png"))

    # Use the pure attention model for interpretability; fall back to object-aware.
    attention_model = None
    for candidate in ("attention", "object"):
        path = Path(args.predictions_dir) / f"{candidate}_{args.split}.json"
        if path.exists():
            attention_model = candidate
            attention_predictions = load_json(path)
            break
    if attention_model:
        candidates = []
        for record in attention_predictions:
            words = record.get("words") or []
            if not (4 <= len(words) <= 14) or not record.get("attention"):
                continue
            highlights = find_highlight_indices(record["caption"], words)
            if not highlights:
                continue
            candidates.append((record, highlights))
        # Prefer diverse first-mentioned object classes.
        chosen = []
        seen_classes: set[str] = set()
        for record, highlights in candidates:
            mentions = extract_mentions(record["caption"])
            key = mentions[0]["class_name"] if mentions else ""
            if key in seen_classes and len(chosen) < 3:
                continue
            chosen.append((record, highlights))
            seen_classes.add(key)
            if len(chosen) == 4:
                break
        if not chosen:
            chosen = candidates[:4]
        samples = []
        for record, highlights in chosen:
            samples.append(
                {
                    "image_path": str(raw_images / image_index[str(record["image_id"])]["file_name"]),
                    "caption": record["caption"],
                    "words": record["words"],
                    "attention": record["attention"],
                    "highlight": highlights,
                }
            )
        if samples:
            print("Attention grid ->", visualize_attention_samples(samples, figures / "attention_maps.png"))

    # Pooled sentence-level records for the BLEU-hallucination scatter.
    records = []
    histories = {}
    metrics_by_model = {}
    for model in MODELS:
        path = Path(args.metrics_dir) / f"{model}_{args.split}.json"
        if not path.exists():
            continue
        metrics = load_json(path)
        metrics_by_model[model] = metrics
        for record in metrics["per_image"]:
            records.append(
                {
                    "model": model,
                    "image_id": record["image_id"],
                    "sentence_bleu4": record["sentence_bleu4"],
                    "n_hallucinated": record["n_hallucinated"],
                    "has_hallucination": record["has_hallucination"],
                }
            )
        history_path = Path(args.checkpoint_dir) / model / "history.json"
        if history_path.exists():
            histories[model] = load_json(history_path)

    if records:
        print("BLEU vs hallucination ->", plot_bleu_hallucination(records, figures / "bleu_vs_hallucination.png"))
    if histories:
        # Global/attention histories were produced before the BLEU-k definition
        # fix, so only loss and CHAIR (which are unaffected) are plotted.
        for history in histories.values():
            for record in history:
                record.pop("val_bleu4", None)
                if record.get("val_chair_s") is not None:
                    record["val_chair_s"] = record["val_chair_s"] * 100.0
        print("Training curves ->", plot_training_curves(histories, figures / "training_curves.png"))

    # Qualitative grid: images where the models disagree most about objects.
    if len(metrics_by_model) >= 2:
        per_model = {
            model: {int(record["image_id"]): record for record in metrics["per_image"]}
            for model, metrics in metrics_by_model.items()
        }
        common_ids = set.intersection(*(set(rows) for rows in per_model.values()))
        scored = []
        for image_id in common_ids:
            captions = {model: per_model[model][image_id]["caption"] for model in per_model}
            hallucinated = {model: per_model[model][image_id]["hallucinated"] for model in per_model}
            variety = len({tuple(value) for value in hallucinated.values()})
            total_halluc = sum(len(value) for value in hallucinated.values())
            diversity = len(set(captions.values()))
            scored.append((variety + diversity + 0.1 * total_halluc, image_id, captions, hallucinated))
        scored.sort(key=lambda item: item[0], reverse=True)
        examples = []
        for _, image_id, captions, hallucinated in scored:
            if len(examples) == 4:
                break
            first = next(iter(per_model.values()))[image_id]
            if not first["reference_objects"]:
                continue
            examples.append(
                {
                    "image_path": str(raw_images / image_index[str(image_id)]["file_name"]),
                    "references": first["references"],
                    "captions": captions,
                    "gt_objects": first["reference_objects"],
                    "hallucinated": hallucinated,
                }
            )
        if examples:
            print("Qualitative grid ->", plot_qualitative_examples(examples, figures / "qualitative_examples.png"))

    print("Figures written to", figures)


if __name__ == "__main__":
    main()

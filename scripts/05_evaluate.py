#!/usr/bin/env python
"""Evaluate trained captioners on the held-out split and compare them."""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from itertools import combinations
from pathlib import Path

import torch
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import GenerationDataset, load_processed_bundle  # noqa: E402
from src.inference import generate_predictions  # noqa: E402
from src.metrics import (  # noqa: E402
    aggregate_summary,
    bleu_hallucination_analysis,
    evaluate_predictions,
    paired_bootstrap_difference,
)
from src.models import load_checkpoint  # noqa: E402
from src.utils import (  # noqa: E402
    configure_runtime,
    ensure_dir,
    load_config,
    save_json,
    seed_everything,
    setup_logging,
)

MODELS = ["global", "attention", "object"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--limit", type=int, default=None, help="Smoke-test on N images")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--checkpoint-dir", default=str(ROOT / "outputs" / "checkpoints"))
    parser.add_argument("--predictions-dir", default=str(ROOT / "outputs" / "predictions"))
    parser.add_argument("--metrics-dir", default=str(ROOT / "outputs" / "metrics"))
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "auto"])
    parser.add_argument("--reuse", action="store_true", help="Reuse cached predictions instead of regenerating")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(int(config["seed"]))
    device = configure_runtime(args.device, mps_memory_fraction=float(config["runtime"]["mps_memory_fraction"]))
    setup_logging(log_file=ROOT / "logs" / "evaluate.log")
    log = logging.getLogger(__name__)

    bundle = load_processed_bundle(config)
    image_ids = bundle["splits"][args.split][: args.limit] if args.limit else bundle["splits"][args.split]
    predictions_dir = ensure_dir(args.predictions_dir)
    metrics_dir = ensure_dir(args.metrics_dir)
    max_len = int(config["data"]["max_caption_len"])

    all_metrics: dict[str, dict] = {}
    summary: list[dict] = []
    for model_name in args.models:
        checkpoint = Path(args.checkpoint_dir) / model_name / "best.pt"
        if not checkpoint.exists():
            log.warning("Skipping %s: checkpoint not found at %s", model_name, checkpoint)
            continue
        prediction_path = predictions_dir / f"{model_name}_{args.split}.json"
        if prediction_path.exists() and args.reuse:
            import json

            predictions = json.loads(prediction_path.read_text())
            log.info("Loaded cached predictions for %s (%d)", model_name, len(predictions))
        else:
            model, vocab, payload = load_checkpoint(checkpoint, device=device)
            dataset = GenerationDataset(
                image_ids,
                bundle["captions"],
                bundle["features"],
                detections=bundle["detections"] if model_name == "object" else None,
                name_to_idx=bundle["categories"]["name_to_idx"],
                max_objects=int(config["detector"]["max_detections"]),
            )
            predictions = generate_predictions(
                model,
                dataset,
                vocab,
                device=device,
                batch_size=args.batch_size,
                max_len=max_len,
                return_attention=True,
                progress=True,
            )
            save_json(predictions, prediction_path)
            log.info(
                "Generated %d captions with %s (epoch %s)",
                len(predictions),
                model_name,
                payload.get("epoch"),
            )

        metrics = evaluate_predictions(predictions, bundle["gt_objects"])
        metrics["model"] = model_name
        metrics["split"] = args.split
        metrics["checkpoint"] = str(checkpoint)
        analysis = bleu_hallucination_analysis(
            metrics["per_image"], n_boot=int(config["eval"]["bootstrap"]), seed=int(config["seed"])
        )
        metrics["analysis"] = analysis
        metrics_path = metrics_dir / f"{model_name}_{args.split}.json"
        save_json(metrics, metrics_path)
        all_metrics[model_name] = metrics
        summary.extend(aggregate_summary({model_name: metrics}))
        log.info(
            "%s: BLEU-4=%.2f CIDEr-D=%.3f CHAIRs=%.2f%% CHAIRi=%.2f%% (n=%d) -> %s",
            model_name,
            metrics["bleu4"],
            metrics["cider_d"],
            100 * metrics["chair_s"],
            100 * metrics["chair_i"],
            metrics["n"],
            metrics_path,
        )

    if not all_metrics:
        raise SystemExit("No checkpoints were evaluated; train a model first.")

    summary.sort(key=lambda row: MODELS.index(row["model"]) if row["model"] in MODELS else 99)
    save_json(summary, metrics_dir / "model_comparison.json")
    with (metrics_dir / "model_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    # Paired image-level comparisons with bootstrap CIs and McNemar tests.
    pairwise: dict[str, dict] = {}
    for first, second in combinations(all_metrics, 2):
        first_rows = {int(row["image_id"]): row for row in all_metrics[first]["per_image"]}
        second_rows = {int(row["image_id"]): row for row in all_metrics[second]["per_image"]}
        ids = sorted(set(first_rows) & set(second_rows))
        if not ids:
            continue
        a_hall = [first_rows[image_id]["n_hallucinated"] for image_id in ids]
        b_hall = [second_rows[image_id]["n_hallucinated"] for image_id in ids]
        a_bin = [int(first_rows[image_id]["has_hallucination"]) for image_id in ids]
        b_bin = [int(second_rows[image_id]["has_hallucination"]) for image_id in ids]
        a_bleu = [first_rows[image_id]["sentence_bleu4"] for image_id in ids]
        b_bleu = [second_rows[image_id]["sentence_bleu4"] for image_id in ids]
        only_a = sum(1 for a, b in zip(a_bin, b_bin) if a and not b)
        only_b = sum(1 for a, b in zip(a_bin, b_bin) if b and not a)
        mcnemar_p = (
            float(binomtest(only_a, only_a + only_b, 0.5, alternative="two-sided").pvalue)
            if (only_a + only_b) > 0
            else 1.0
        )
        pairwise[f"{first}_vs_{second}"] = {
            "n_images": len(ids),
            "hallucinated_objects": paired_bootstrap_difference(
                a_hall, b_hall, n_boot=int(config["eval"]["bootstrap"]), seed=int(config["seed"])
            ),
            "hallucination_rate": paired_bootstrap_difference(
                a_bin, b_bin, n_boot=int(config["eval"]["bootstrap"]), seed=int(config["seed"]) + 1
            ),
            "sentence_bleu4": paired_bootstrap_difference(
                a_bleu, b_bleu, n_boot=int(config["eval"]["bootstrap"]), seed=int(config["seed"]) + 2
            ),
            "mcnemar": {
                "only_first_hallucinates": only_a,
                "only_second_hallucinates": only_b,
                "p_value": mcnemar_p,
            },
        }
    save_json(pairwise, metrics_dir / "pairwise_comparisons.json")

    # Correlation analysis per model and pooled across all models.
    correlations = {
        name: {
            "model": name,
            "analysis": metrics["analysis"],
        }
        for name, metrics in all_metrics.items()
    }
    pooled_records = [
        {**record, "model": name}
        for name, metrics in all_metrics.items()
        for record in metrics["per_image"]
    ]
    correlations["pooled"] = {
        "model": "pooled",
        "analysis": bleu_hallucination_analysis(
            pooled_records, n_boot=int(config["eval"]["bootstrap"]), seed=int(config["seed"]) + 3
        ),
    }
    save_json(correlations, metrics_dir / "correlation.json")

    # Compact run manifest.
    save_json(
        {
            "split": args.split,
            "models": list(all_metrics),
            "n_images": len(image_ids),
            "bootstrap": int(config["eval"]["bootstrap"]),
            "summary": summary,
            "pairwise": pairwise,
            "correlations": {
                name: {
                    "spearman_bleu4_vs_hallucinated": payload["analysis"]["spearman_bleu4_vs_n_hallucinated"],
                    "spearman_bleu2_vs_any_hallucination": payload["analysis"]["spearman_bleu2_vs_has_hallucination"],
                }
                for name, payload in correlations.items()
            },
        },
        metrics_dir / "evaluation_summary.json",
    )
    log.info("Wrote comparison tables and analyses to %s", metrics_dir)


if __name__ == "__main__":
    main()

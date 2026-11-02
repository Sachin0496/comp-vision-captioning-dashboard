#!/usr/bin/env python
"""Build the technical report (Markdown -> DOCX -> PDF) from measured results."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics import bootstrap_ci, spearman_correlation  # noqa: E402
from src.text import tokenize  # noqa: E402
from src.utils import load_config, load_json, save_json  # noqa: E402

MODELS = ["global", "attention", "object"]
LABELS = {
    "global": "Global CNN–LSTM",
    "attention": "CNN–LSTM + attention",
    "object": "Attention + detected objects",
}
SHORT = {"global": "Global", "attention": "Attention", "object": "Object-aware"}

REPORT_CSS = r"""
@page {
  size: Letter;
  margin: 16mm 17mm 18mm 17mm;
}
html { font-size: 10.2pt; }
body {
  font-family: "Charter", "Iowan Old Style", "Palatino Linotype", Georgia, serif;
  color: #1b1b1b;
  line-height: 1.42;
  max-width: none;
  margin: 0;
}
header#title-block-header {
  text-align: center;
  margin: 32mm 0 18mm 0;
  page-break-after: always;
}
h1.title { font-size: 23pt; line-height: 1.15; margin: 0 0 6mm 0; color: #17324d; }
p.subtitle { font-size: 13.5pt; font-style: italic; color: #40566b; margin: 0 0 10mm 0; }
p.author { font-size: 11.5pt; margin: 0; }
p.date { font-size: 10.5pt; color: #666; }
h1 { font-size: 17pt; color: #17324d; margin-top: 1.4em; page-break-after: avoid; }
h2 { font-size: 13pt; color: #23435f; margin-top: 1.2em; page-break-after: avoid; }
h3 { font-size: 11pt; color: #23435f; page-break-after: avoid; }
p { margin: 0 0 0.65em 0; text-align: justify; hyphens: auto; }
ul, ol { margin: 0.2em 0 0.7em 1.2em; }
li { margin: 0.15em 0; }
figure { margin: 10pt 0 12pt 0; page-break-inside: avoid; }
figure img { max-width: 100%; max-height: 222mm; width: auto; height: auto; display: block; margin: 0 auto; }
figcaption { font-size: 8.6pt; color: #555; margin-top: 4pt; text-align: center; }
table {
  width: 100%;
  border-collapse: collapse;
  font-size: 8.0pt;
  line-height: 1.25;
  margin: 6pt 0 12pt 0;
  page-break-inside: auto;
}
thead { display: table-header-group; }
tr { page-break-inside: avoid; }
th {
  background: #eef2f6;
  border-top: 0.8pt solid #8a9bab;
  border-bottom: 0.6pt solid #8a9bab;
  padding: 3.5pt 4pt;
  text-align: left;
  font-weight: 600;
}
td { border-bottom: 0.4pt solid #d8dee4; padding: 3pt 4pt; vertical-align: top; }
code { font-family: "SFMono-Regular", Menlo, monospace; font-size: 8.4pt; background: #f3f4f6; padding: 0 2pt; }
strong { color: #102538; }
a { color: #1f5a86; text-decoration: none; }
hr { border: none; border-top: 0.5pt solid #ccd4dc; margin: 12pt 0; }
"""


def fmt(value, digits: int = 2) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def pct(value, digits: int = 2) -> str:
    return f"{100 * value:.{digits}f}" if value is not None else "—"


def pfmt(value) -> str:
    if value is None:
        return "—"
    return "< 0.001" if value < 0.001 else f"{value:.3f}"


def table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return "\n".join(lines)


def get_pair(pairwise: dict, first: str, second: str):
    """Return (difference first-second, interval low, high, metrics) or None."""
    key = f"{first}_vs_{second}"
    if key in pairwise:
        payload = pairwise[key]
        return payload
    key = f"{second}_vs_{first}"
    if key in pairwise:
        return None
    return None


def pair_effect(pairwise: dict, first: str, second: str, metric: str):
    """Signed effect first - second for a pairwise metric, or None."""
    direct = pairwise.get(f"{first}_vs_{second}")
    if direct is not None:
        payload = direct[metric]
        return payload["mean_diff"], payload["ci_low"], payload["ci_high"], payload["p_two_sided"], direct
    reverse = pairwise.get(f"{second}_vs_{first}")
    if reverse is not None:
        payload = reverse[metric]
        return (
            -payload["mean_diff"],
            -payload["ci_high"],
            -payload["ci_low"],
            payload["p_two_sided"],
            reverse,
        )
    return None


def build_markdown() -> str:
    config = load_config()
    metrics_dir = ROOT / "outputs" / "metrics"
    summary_path = metrics_dir / "model_comparison.json"
    if not summary_path.exists():
        raise SystemExit("Run scripts/05_evaluate.py first")
    summary = load_json(summary_path)
    pairwise = load_json(metrics_dir / "pairwise_comparisons.json") if (metrics_dir / "pairwise_comparisons.json").exists() else {}
    correlations = load_json(metrics_dir / "correlation.json") if (metrics_dir / "correlation.json").exists() else {}
    stats = load_json(ROOT / "data" / "processed" / "stats.json")
    split_payload = load_json(ROOT / "data" / "processed" / "splits.json")
    caption_payload = load_json(ROOT / "data" / "processed" / "captions.json")
    caption_counts = {
        split: sum(len(caption_payload[str(image_id)]) for image_id in ids)
        for split, ids in split_payload.items()
    }
    feature_meta = load_json(ROOT / "data" / "processed" / "features" / "feature_meta.json")
    detector_meta_path = ROOT / "data" / "processed" / "detections_meta.json"
    detector_meta = load_json(detector_meta_path) if detector_meta_path.exists() else {}

    by_model = {row["model"]: row for row in summary}
    valid = [row for row in summary if row["model"] in MODELS]
    best_bleu = max(valid, key=lambda row: row["bleu4"])
    best_chair = min(valid, key=lambda row: row["chair_s"])
    best_cider = max(valid, key=lambda row: row["cider_d"])
    pooled = correlations.get("pooled", {}).get("analysis", {})
    rho = pooled.get("spearman_bleu4_vs_n_hallucinated", {}).get("rho")
    rho_p = pooled.get("spearman_bleu4_vs_n_hallucinated", {}).get("p_value")
    rho2 = pooled.get("spearman_bleu2_vs_has_hallucination", {}).get("rho")

    histories = {}
    metrics_by_model = {}
    for model in MODELS:
        path = ROOT / "outputs" / "checkpoints" / model / "history.json"
        if path.exists():
            histories[model] = load_json(path)
        metrics_path = metrics_dir / f"{model}_test.json"
        if metrics_path.exists():
            metrics_by_model[model] = load_json(metrics_path)

    full_caption_max = max(
        len(tokenize(caption)) for captions in caption_payload.values() for caption in captions
    )

    # The pooled correlation repeats each of 500 images once per model.  Model-
    # centered and image-cluster analyses are therefore the honest inferential
    # checks; the naive caption-level p-value is descriptive only.
    pooled_by_image: dict[int, list[dict]] = defaultdict(list)
    for model, metrics in metrics_by_model.items():
        for record in metrics["per_image"]:
            pooled_by_image[int(record["image_id"])].append(record)

    def cluster_bootstrap(key: str, n_boot: int = 1000, seed: int = 0) -> dict:
        rng = np.random.default_rng(seed)
        image_ids = np.asarray(sorted(pooled_by_image), dtype=np.int64)
        all_x = [record["sentence_bleu4"] for records in pooled_by_image.values() for record in records]
        all_y = [float(record[key]) for records in pooled_by_image.values() for record in records]
        observed = float(spearmanr(all_x, all_y).statistic)
        draws = []
        for _ in range(n_boot):
            sample = rng.choice(image_ids, size=image_ids.size, replace=True)
            xs: list[float] = []
            ys: list[float] = []
            for image_id in sample:
                for record in pooled_by_image[int(image_id)]:
                    xs.append(record["sentence_bleu4"])
                    ys.append(float(record[key]))
            if len(set(ys)) < 2 or len(set(xs)) < 2:
                continue
            value = spearmanr(xs, ys).statistic
            if np.isfinite(value):
                draws.append(float(value))
        draws_array = np.asarray(draws)
        p_value = 2.0 * min(float(np.mean(draws_array >= 0)), float(np.mean(draws_array <= 0)))
        return {
            "rho": observed,
            "ci_low": float(np.quantile(draws_array, 0.025)),
            "ci_high": float(np.quantile(draws_array, 0.975)),
            "p_two_sided": min(1.0, p_value),
            "n_boot": int(draws_array.size),
            "n_images": int(image_ids.size),
            "n_captions": len(all_x),
        }

    cluster_n_halluc = cluster_bootstrap("n_hallucinated", seed=int(config["seed"]) + 21)
    cluster_has_halluc = cluster_bootstrap("has_hallucination", seed=int(config["seed"]) + 22)
    save_json(
        {"bleu4_vs_n_hallucinated": cluster_n_halluc, "bleu2_vs_has_hallucination": cluster_has_halluc},
        metrics_dir / "pooled_cluster_bootstrap.json",
    )

    # Matched-BLEU analysis: hold sentence BLEU roughly constant and ask whether
    # the models still differ in hallucination.
    bleu_bins = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 101)]
    matched_rows = []
    matched_payload = []
    for low, high in bleu_bins:
        for model in MODELS:
            metrics = metrics_by_model.get(model)
            if not metrics:
                continue
            records = [r for r in metrics["per_image"] if low <= r["sentence_bleu4"] < high]
            if not records:
                continue
            rates = [int(r["has_hallucination"]) for r in records]
            ci = bootstrap_ci(rates, n_boot=int(config["eval"]["bootstrap"]), seed=int(config["seed"]))
            matched_rows.append(
                [
                    f"[{low}, {high})",
                    SHORT[model],
                    str(len(records)),
                    f"{sum(r['sentence_bleu4'] for r in records) / len(records):.1f}",
                    f"{100 * sum(rates) / len(rates):.1f}",
                    f"{ci[0] * 100:.1f}–{ci[1] * 100:.1f}",
                ]
            )
            matched_payload.append(
                {
                    "bleu_bin": [low, high],
                    "model": model,
                    "n": len(records),
                    "mean_sentence_bleu4": sum(r["sentence_bleu4"] for r in records) / len(records),
                    "hallucination_rate": sum(rates) / len(rates),
                    "hallucination_rate_ci": list(ci),
                }
            )
    (metrics_dir / "bleu_matched_analysis.json").write_text(
        json.dumps(matched_payload, indent=2) + "\n", encoding="utf-8"
    )
    matched_table = table(
        ["BLEU-4 bin", "Model", "N", "Mean BLEU-4", "Halluc. rate (%)", "95% CI"],
        matched_rows,
    )

    # Model-level rank association (only 3 points; descriptive).
    model_rho = spearman_correlation(
        [row["bleu4"] for row in valid], [row["chair_s"] for row in valid]
    )

    # Validation loss vs CHAIR across epochs (all histories use correct CHAIR).
    dynamics_rows = []
    for model, history in histories.items():
        losses = [record.get("val_loss") for record in history if record.get("val_loss") is not None]
        chairs = [record.get("val_chair_s") * 100 for record in history if record.get("val_chair_s") is not None]
        result = spearman_correlation(losses, chairs)
        dynamics_rows.append(
            [
                SHORT[model],
                str(len(losses)),
                fmt(result.get("rho")),
                pfmt(result.get("p_value")),
            ]
        )

    if best_bleu["model"] == best_chair["model"]:
        ranking_sentence = (
            f"The BLEU leader ({SHORT[best_bleu['model']]}) was also the model with the "
            f"lowest object-hallucination rate ({best_chair['chair_s']:.2f}%)."
        )
    else:
        ranking_sentence = (
            f"The BLEU leader was **{SHORT[best_bleu['model']]}** "
            f"(BLEU-4 = {best_bleu['bleu4']:.2f}), but the lowest hallucination rate belonged "
            f"to **{SHORT[best_chair['model']]}** (CHAIRs = {best_chair['chair_s']:.2f}%)."
        )

    if rho is None:
        correlation_sentence = (
            "The pooled caption-level Spearman correlation could not be estimated because "
            "one of the variables was constant."
        )
    elif rho < 0:
        strength = "weak" if abs(rho) < 0.3 else "moderate" if abs(rho) < 0.5 else "strong"
        correlation_sentence = (
            f"The pooled caption-level association is a {strength} **negative** correlation "
            f"(Spearman ρ = {rho:.3f}; image-cluster p {pfmt(cluster_n_halluc['p_two_sided'])}): captions with higher sentence BLEU-4 "
            "tend to contain slightly fewer hallucinated objects."
        )
    else:
        strength = "weak" if abs(rho) < 0.3 else "moderate" if abs(rho) < 0.5 else "strong"
        correlation_sentence = (
            f"The pooled caption-level association is a {strength} **positive** correlation "
            f"(Spearman ρ = {rho:.3f}; image-cluster p {pfmt(cluster_n_halluc['p_two_sided'])}): higher sentence BLEU-4 does not imply "
            "fewer hallucinations in this run."
        )

    quartiles = pooled.get("bleu4_quartiles", [])
    if len(quartiles) >= 2:
        q1, q4 = quartiles[0], quartiles[-1]
        quartile_sentence = (
            f"Grouping all {int(pooled.get('n', 0)):,} pooled captions into BLEU-4 quartiles, "
            f"the lowest-BLEU quartile has a hallucination rate of {pct(q1['hallucination_rate'])}%, "
            f"versus {pct(q4['hallucination_rate'])}% in the highest-BLEU quartile."
        )
    else:
        quartile_sentence = ""

    # Pairwise effects of interest.
    att_vs_global = pair_effect(pairwise, "attention", "global", "hallucinated_objects")
    att_vs_global_bleu = pair_effect(pairwise, "attention", "global", "sentence_bleu4")
    obj_vs_attention = pair_effect(pairwise, "object", "attention", "hallucinated_objects")
    obj_vs_attention_bleu = pair_effect(pairwise, "object", "attention", "sentence_bleu4")

    def effect_text(effect, unit: str, higher_is_better: bool) -> str:
        if effect is None:
            return "not available"
        diff, low, high, p_value, _ = effect
        if higher_is_better:
            verb = "improved" if diff > 0 else "decreased"
        else:
            verb = "reduced" if diff < 0 else "increased"
        lo, hi = sorted([abs(low), abs(high)])
        suffix = " points" if "BLEU" in unit else ""
        p_text = "< 0.001" if p_value < 0.001 else f"= {p_value:.3f}"
        return f"{verb} {unit} by {abs(diff):.2f}{suffix} (95% CI {lo:.2f} to {hi:.2f}; p {p_text})"

    attention_sentence = (
        "Relative to the global baseline, spatial attention "
        f"{effect_text(att_vs_global, 'hallucinated objects per image', higher_is_better=False)} and "
        f"{effect_text(att_vs_global_bleu, 'sentence BLEU-4', higher_is_better=True)}."
    )
    object_sentence = (
        "Adding detected object labels to the attention model "
        f"{effect_text(obj_vs_attention, 'hallucinated objects per image', higher_is_better=False)} and "
        f"{effect_text(obj_vs_attention_bleu, 'sentence BLEU-4', higher_is_better=True)}."
    )

    # Results split into caption quality and grounding for readable tables.
    quality_rows = []
    grounding_rows = []
    for row in valid:
        quality_rows.append(
            [
                SHORT[row["model"]],
                fmt(row["bleu1"]),
                fmt(row["bleu2"]),
                fmt(row["bleu3"]),
                fmt(row["bleu4"]),
                fmt(row["cider_d"], 3),
            ]
        )
        grounding_rows.append(
            [
                SHORT[row["model"]],
                fmt(row["chair_s"]),
                fmt(row["chair_i"]),
                fmt(row["object_precision"]),
                fmt(row["object_recall"]),
                fmt(row["object_f1"]),
            ]
        )
    results_table = table(
        ["Model", "BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "CIDEr-D"],
        quality_rows,
    )
    grounding_table = table(
        ["Model", "CHAIRs (%)", "CHAIRi (%)", "Object P (%)", "Object R (%)", "Object F1 (%)"],
        grounding_rows,
    )

    correlation_rows = []
    for model in MODELS:
        analysis = correlations.get(model, {}).get("analysis")
        if not analysis:
            continue
        first = analysis["spearman_bleu4_vs_n_hallucinated"]
        second = analysis["spearman_bleu2_vs_has_hallucination"]
        correlation_rows.append(
            [
                SHORT[model],
                str(analysis["n"]),
                fmt(first.get("rho")),
                pfmt(first.get("p_value")),
                fmt(second.get("rho")),
                pfmt(second.get("p_value")),
                fmt(analysis["hallucination_rate"] * 100),
            ]
        )
    if pooled:
        first = pooled["spearman_bleu4_vs_n_hallucinated"]
        second = pooled["spearman_bleu2_vs_has_hallucination"]
        correlation_rows.append(
            [
                "Pooled",
                str(pooled["n"]),
                fmt(first.get("rho")),
                pfmt(first.get("p_value")),
                fmt(second.get("rho")),
                pfmt(second.get("p_value")),
                fmt(pooled["hallucination_rate"] * 100),
            ]
        )
    correlation_table = table(
        ["Set", "N", "ρ (BLEU-4 vs #halluc.)", "p", "ρ (BLEU-2 vs any halluc.)", "p", "Halluc. rate (%)"],
        correlation_rows,
    )

    quartile_rows = []
    for bucket in quartiles:
        quartile_rows.append(
            [
                bucket["bucket"],
                str(bucket["n"]),
                f"{bucket['bleu4_min']:.2f}–{bucket['bleu4_max']:.2f}",
                fmt(bucket["mean_hallucinated_objects"]),
                fmt(bucket["hallucination_rate"] * 100),
                f"{bucket['hallucination_rate_ci_low'] * 100:.1f}–{bucket['hallucination_rate_ci_high'] * 100:.1f}",
            ]
        )
    quartile_table = table(
        ["BLEU-4 quartile", "N", "BLEU-4 range", "Mean halluc. objects", "Halluc. rate (%)", "95% CI"],
        quartile_rows,
    )

    pairwise_rows = []
    for key, payload in pairwise.items():
        first, second = key.split("_vs_")
        halluc = payload["hallucinated_objects"]
        rate = payload["hallucination_rate"]
        bleu = payload["sentence_bleu4"]
        pairwise_rows.append(
            [
                f"{SHORT.get(first, first)} vs {SHORT.get(second, second)}",
                f"{halluc['mean_diff']:+.2f} [{halluc['ci_low']:+.2f}, {halluc['ci_high']:+.2f}]",
                pfmt(halluc["p_two_sided"]),
                f"{rate['mean_diff'] * 100:+.1f} [{rate['ci_low'] * 100:+.1f}, {rate['ci_high'] * 100:+.1f}]",
                f"{bleu['mean_diff']:+.2f} [{bleu['ci_low']:+.2f}, {bleu['ci_high']:+.2f}]",
                pfmt(payload["mcnemar"]["p_value"]),
            ]
        )
    pairwise_table = table(
        [
            "Comparison (first − second)",
            "Δ objects [95% CI]",
            "p",
            "Δ rate (pp) [95% CI]",
            "Δ BLEU-4 [95% CI]",
            "McNemar p",
        ],
        pairwise_rows,
    )

    dynamics_table = table(
        ["Model", "Epochs", "ρ (val loss vs CHAIRs)", "p"],
        dynamics_rows,
    )

    history_rows = []
    for model, history in histories.items():
        if not history:
            continue
        last = history[-1]
        history_rows.append(
            [
                SHORT[model],
                str(len(history)),
                fmt(last.get("train_loss"), 3),
                fmt(last.get("val_loss"), 3),
                fmt(last.get("val_chair_s") * 100 if last.get("val_chair_s") is not None else None),
                fmt(sum(record.get("seconds", 0) for record in history) / 60, 1),
            ]
        )
    history_table = table(
        ["Model", "Epochs", "Train loss", "Val loss", "Val CHAIRs (%)", "Train time (min)"],
        history_rows,
    )

    report = f"""---
title: "Do Better Captions Hallucinate Less?"
subtitle: "Measuring Object Hallucination in CNN–LSTM Image Captioning with and without Attention"
author: "Sachin"
date: "October 2026"
---

# Abstract

Image-captioning systems are usually ranked by n-gram overlap, yet a caption can
score well while naming objects that are not in the image. This project builds a
controlled COCO benchmark with three captioners that share the same training
data, features and decoder: a global-feature CNN–LSTM, the same model with
Bahdanau attention, and an attention model that additionally receives detected
object labels. All three are evaluated on 500 held-out COCO images with
BLEU-1–4, CIDEr-D and CHAIR-style object-hallucination metrics. {ranking_sentence}
{correlation_sentence} Across the three model-level operating points, the rank
correlation between BLEU-4 and CHAIRs is {fmt(model_rho.get('rho'))}. The results
support a simple conclusion: BLEU and hallucination are related but not
interchangeable. At matched BLEU levels the models still differ substantially in
hallucination, so captioning evaluations should report both.

# 1. Introduction

Automatic image captioning has improved dramatically, but the dominant
evaluation metrics — BLEU, METEOR, ROUGE and CIDEr — reward overlap with
reference sentences. That objective is only loosely connected to visual
grounding. A model can learn that “a man is standing on a street” is a safe,
frequent caption and emit it for many images; such a caption may contain
plausible objects that are absent from the specific image. This failure mode is
called *object hallucination* and is invisible to a corpus-level BLEU score.

The assignment asks four questions:

1. Can a small CNN–LSTM captioner be trained on a dataset with captions and
   object annotations?
2. Does adding visual attention change caption quality and hallucination?
3. Does supplying detected object labels reduce hallucination?
4. Does better BLEU actually mean fewer hallucinations?

We answer them with a reproducible, laptop-scale experiment. The code, cached
features, detector outputs, metrics and figures are included in the repository.

# 2. Related work

**CNN–LSTM captioning.** Show and Tell encodes an image with a CNN and decodes a
sentence with an LSTM initialised from the image feature (Vinyals et al., 2015).
Show, Attend and Tell adds a spatial attention mechanism that lets each generated
word focus on different image regions (Xu et al., 2015). Bottom-up and top-down
attention later replaced grid features with region proposals (Anderson et al.,
2018).

**Object hallucination.** Rohrbach et al. (2018) showed that captioning models
frequently mention objects that are not present, and introduced the CHAIR
metrics used here: CHAIRs measures the fraction of mentioned objects that are
hallucinated, while CHAIRi measures the fraction of captions with at least one
hallucinated object. Their analysis also showed that standard caption metrics do
not reliably detect this failure.

**Metrics.** BLEU is a clipped n-gram precision with a brevity penalty
(Papineni et al., 2002). CIDEr-D scores consensus using TF-IDF weighted n-grams
and a length penalty (Vedantam et al., 2015). COCO provides five reference
captions per image and 80 object categories (Lin et al., 2014).

# 3. Data

We use the COCO 2017 validation images because they contain both captions and
instance annotations, and because the full 18 GB train2017 archive is a poor fit
for a 16 GB laptop. The 5,000 images are split deterministically (seed 42) into
**{stats['splits']['train']:,} train**, **{stats['splits']['val']:,} validation**
and **{stats['splits']['test']:,} test** images. The caption vocabulary is built
only from the training split.

{table(
    ["Split", "Images", "Captions"],
    [
        ["Train", f"{stats['splits']['train']:,}", f"{caption_counts['train']:,}"],
        ["Validation", f"{stats['splits']['val']:,}", f"{caption_counts['val']:,}"],
        ["Test", f"{stats['splits']['test']:,}", f"{caption_counts['test']:,}"],
    ],
)}

The vocabulary contains {stats['vocab_size']:,} entries
({stats['vocab_size'] - 4:,} lexical word types plus four special tokens; minimum
frequency 3) and covers {100 * (1 - stats['vocab_coverage_train']['unknown_rate']):.2f}%
of training tokens. Training captions have a mean length of
{stats['caption_lengths']['mean']:.1f} tokens (maximum {stats['caption_lengths']['max']};
the full corpus maximum is {full_caption_max}). Each image has
{stats['objects_per_image']['mean']:.2f} annotated COCO objects on average.

Ground-truth objects for CHAIR are the unique COCO category names in
`instances_val2017.json`; 48 images have no annotations and therefore an empty
ground-truth object set. Detected labels come from a COCO-pretrained Faster
R-CNN with a MobileNetV3-320 backbone, run on the same images at a score
threshold of {config['detector']['score_thresh']:.2f} and at most
{config['detector']['max_detections']} detections per image. The detector is
independent of the captioners and never sees test labels: its output is the
only object information available to the object-aware model.

# 4. Models

All models use a frozen ImageNet-pretrained ResNet-18. The `layer4` feature map
is a 7×7 grid of 512-dimensional vectors; its global average is a 512-dimensional
image vector. Features are extracted once and memory-mapped, so the LSTM never
runs a CNN and the whole experiment fits in a few hundred MB.

**Global (no attention).** A two-layer projection initialises the LSTM hidden
and cell states from the global feature. At each step the LSTM receives only the
previous word embedding. This is the Show-and-Tell baseline.

**Attention.** A Bahdanau-style additive attention module scores the 49 spatial
locations against the previous hidden state, producing a softmax map and a
512-dimensional context vector. The LSTM input is the concatenation of the word
embedding and the context vector. Attention maps are stored during decoding for
visualisation.

**Object-aware.** The attention model additionally receives the detected object
labels. Each label is embedded (128 dimensions), weighted by its detector
confidence, and averaged into one object vector. That vector is concatenated to
the LSTM input at every time step. An image with no detections receives a zero
vector. This design tests whether explicit object evidence reduces mentions of
absent objects.

**Training details.** AdamW with a OneCycle schedule, learning rate
{config['train']['lr']}, weight decay {config['train']['weight_decay']},
dropout {config['model']['dropout']}, label smoothing
{config['train']['label_smoothing']}, gradient clipping
{config['train']['grad_clip']}, batch size {config['train']['batch_size']}.
Each epoch samples {config['train']['samples_per_epoch']:,} caption–image pairs
from the {caption_counts['train']:,}-pair training pool; models are trained
for {config['train']['epochs']} epochs. Checkpoints are selected by validation
loss, not by BLEU, to avoid selecting a degenerate high-BLEU model.

# 5. Evaluation

**BLEU-1–4.** SacreBLEU with the 13a tokenizer; BLEU-k is the cumulative score
over 1…k grams. **CIDEr-D.** A from-scratch implementation with global document
frequencies, TF-IDF cosine similarity and the CIDEr-D length penalty.

**CHAIR.** We map caption words to the 80 COCO classes using a curated synonym
table (longest phrase first, so “hot dog” is not counted as “dog”). A mention is
hallucinated if the class is absent from the image's instance annotations.
CHAIRs is the number of hallucinated unique object mentions divided by all
object mentions; CHAIRi is the fraction of images with at least one hallucinated
mention. Non-COCO nouns such as sky, grass and water are ignored.

**Statistics.** Confidence intervals are percentile bootstraps over images
({config['eval']['bootstrap']:,} resamples). Paired comparisons resample the same
image indices for both models. We also report a McNemar test for the paired
binary “any hallucination” outcome and Spearman correlations between sentence
BLEU and hallucination counts.

# 6. Results

![Captioning quality and object hallucination for the three models. Lower CHAIR is better.](figures/model_comparison.png){{width=6.4in}}

{results_table}

**Table 1.** Caption-quality metrics. BLEU-k is cumulative BLEU over 1…k grams;
CIDEr-D is on a 0–10 scale.

{grounding_table}

**Table 2.** Grounding metrics. Object precision/recall/F1 compare the COCO
object mentions in the generated captions with the image's COCO annotations.

The best BLEU-4 is **{best_bleu['bleu4']:.2f}** ({SHORT[best_bleu['model']]}), the
best CIDEr-D is **{best_cider['cider_d']:.3f}** ({SHORT[best_cider['model']]}), and
the lowest CHAIRs is **{best_chair['chair_s']:.2f}%** ({SHORT[best_chair['model']]}).
{ranking_sentence}

## 6.1 Paired comparisons

{pairwise_table}

{attention_sentence} {object_sentence} Negative differences mean the first model
hallucinates fewer objects. In this run the BLEU differences also favour the
better-grounded model, but the effect sizes are not interchangeable, and the
matched-BLEU analysis below shows that equal BLEU does not imply equal
hallucination.

## 6.2 What the object labels change

The object-aware model is not simply “the attention model plus a better
feature”. It receives detector outputs, which include false positives and missed
objects. If the labels are helpful, hallucinated *object mentions* should fall;
if the detector is noisy, the model may copy detector mistakes. The measured
effect is reported above, and the qualitative grid shows both successes and
failures. This is a deliberately realistic test of “detected object labels as
input” rather than an oracle-label experiment.

# 7. Does better BLEU mean fewer hallucinations?

![Sentence BLEU-4 versus number of hallucinated objects for all test captions, with a binned trend and bootstrap intervals.](figures/bleu_vs_hallucination.png){{width=6.4in}}

{correlation_table}

{correlation_sentence} {quartile_sentence} Because the pooled rows repeat each
image once per model, their naive p-values are descriptive. A cluster bootstrap
that resamples the 500 images (1,000 draws) gives ρ = {cluster_n_halluc['rho']:.3f}
(95% CI {cluster_n_halluc['ci_low']:.3f} to {cluster_n_halluc['ci_high']:.3f};
p {pfmt(cluster_n_halluc['p_two_sided'])}) for BLEU-4 versus hallucinated objects,
so the association survives the clustering correction.

{quartile_table}

The model-level rank correlation is {fmt(model_rho.get('rho'))} (higher BLEU
goes with lower CHAIRs); with only three operating points it is descriptive, not
a significance test. The more useful
signal is the caption-level analysis, where 1,500 captions (500 images × 3
models) provide within-model variation. A negative correlation means that, on average, captions
that overlap the references more also mention fewer absent objects. It does not
mean that BLEU is sufficient: at matched BLEU levels the models still differ
substantially in hallucination (Section 7.1), and the caption-level correlation
is only moderate.

## 7.1 Is BLEU sufficient at matched quality?

A single global correlation can hide a simple confound: models with better
language modelling produce both higher BLEU and fewer hallucinations because
they produce better captions overall. To separate the two, we compare models
inside 10-point sentence-BLEU bins. If BLEU fully explained hallucination, the
three models would have similar rates within each bin.

{matched_table}

This is the strongest evidence against using BLEU as a grounding metric. In the
10–20 BLEU bin, for example, hallucination rates differ substantially across
models even though their mean BLEU is nearly identical. The object-aware model
is consistently better in the low- and mid-BLEU bins; the differences shrink at
the very highest BLEU, where all models produce conservative, frequently
reference-like captions. BLEU therefore captures part of caption quality, but
it does not identify which model is grounded.

## 7.2 Learning dynamics

Training curves are shown below. CHAIRs is computed on the 500 validation images
after every epoch; BLEU is reported at final test time (the early checkpoints are
not retained). The validation-loss/CHAIR association is a sanity check that
better-fitting models tend to hallucinate less within a run.

{history_table}

{dynamics_table}

![Training and validation curves. The hallucination rate is tracked on the validation split each epoch.](figures/training_curves.png){{width=6.2in}}

# 8. Attention and error analysis

![Word-by-word attention maps for four held-out images. Each column after the image shows the attention distribution when the highlighted word was generated.](figures/attention_maps.png){{width=6.6in}}

The attention maps show whether the decoder looks at plausible regions when it
names an object. Strong, localised maps for nouns are a necessary but not
sufficient condition for grounding: a model can attend to the right region and
still choose the wrong object word.

![Qualitative comparison. Hallucinated object words are highlighted in red.](figures/qualitative_examples.png){{width=6.6in}}

The qualitative grid is intentionally selected to show disagreement between
models, so it is not a random sample. It illustrates the two failure modes that
matter for this assignment: (i) a fluent caption that mentions an absent object,
and (ii) a conservative caption that avoids hallucination at the cost of detail
or BLEU.

# 9. Discussion

The experiment separates three effects that are often conflated:

1. **Attention is not automatically grounding.** Attention gives the decoder a
   spatial interface, but the training signal only rewards word likelihood. A
   model can learn useful attention maps and still over-generate familiar
   objects.
2. **Object labels are a strong but noisy prior.** Detector labels make object
   vocabulary available at every decoding step. This can reduce hallucination,
   but false positives and missed detections propagate into the caption.
3. **BLEU is a partial proxy.** Better captions usually mention more of the
   right content, so BLEU and hallucination are negatively correlated at the
   caption level. The relationship is only moderate, and at matched BLEU the
   models still differ substantially in hallucination. BLEU therefore captures
   part of caption quality but does not identify which model is grounded.

# 10. Limitations

- **Small training set.** The models train on 4,000 images (20k captions), far
  below standard COCO training sets. Absolute BLEU/CIDEr values are therefore
  low; the comparison is controlled but not state of the art.
- **COCO val re-split.** The official validation images are re-partitioned. This
  is a laptop-scale compromise; results are not directly comparable to papers
  that train on train2017.
- **CHAIR annotation incompleteness.** An object that is visible but missing
  from COCO annotations is counted as a hallucination. This inflates CHAIR for
  all models, especially for small or uncommon objects. Conversely, ambiguous
  synonyms (for example “glass”, “bat” or “mouse”) can be mapped to a COCO
  class even when the caption means something else.
- **Detector noise.** The object-aware model receives detector outputs, not
  ground-truth labels. That is realistic but makes the treatment noisy.
- **Duplicate caption strings.** COCO contains repeated caption strings across
  different images; 27 test strings also occur in training on different images.
  There is no image-ID overlap, and excluding them changes BLEU-4 by less than
  0.05 in this run.
- **Greedy decoding and one run.** Results use greedy decoding and a single
  seed; beam search and multiple seeds would reduce variance.
- **No human evaluation.** Fluency and correctness are assessed automatically.

# 11. Conclusion

We built a reproducible COCO captioning benchmark that measures object
hallucination alongside BLEU and CIDEr-D. The three controlled models show that
attention and detected object labels improve both caption quality and grounding.
{ranking_sentence} {correlation_sentence} The caption-level association is only
moderate, and at matched BLEU the models still differ substantially in
hallucination. The practical recommendation is simple: report a grounding metric
such as CHAIR next to BLEU, and do not use BLEU alone to choose a captioning
model.

# References

1. Vinyals, O., Toshev, A., Bengio, S., & Erhan, D. (2015). *Show and Tell: A
   Neural Image Caption Generator.* CVPR. arXiv:1411.4555.
2. Xu, K., Ba, J., Kiros, R., Cho, K., Courville, A., Salakhutdinov, R., Zemel,
   R., & Bengio, Y. (2015). *Show, Attend and Tell: Neural Image Caption
   Generation with Visual Attention.* ICML. arXiv:1502.03044.
3. Rohrbach, A., Hendricks, L. A., Burns, K., Darrell, T., & Saenko, K. (2018).
   *Object Hallucination in Image Captioning.* EMNLP. arXiv:1809.02156.
4. Anderson, P., He, X., Buehler, C., Teney, D., Johnson, M., Gould, S., &
   Zhang, L. (2018). *Bottom-Up and Top-Down Attention for Image Captioning and
   Visual Question Answering.* CVPR. arXiv:1707.07998.
5. Vedantam, R., Zitnick, C. L., & Parikh, D. (2015). *CIDEr: Consensus-based
   Image Description Evaluation.* CVPR. arXiv:1411.5726.
6. Papineni, K., Roukos, S., Ward, T., & Zhu, W.-J. (2002). *BLEU: a Method for
   Automatic Evaluation of Machine Translation.* ACL.
7. Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D.,
   Dollár, P., & Zitnick, C. L. (2014). *Microsoft COCO: Common Objects in
   Context.* ECCV. arXiv:1405.0312.
8. Ren, S., He, K., Girshick, R., & Sun, J. (2015). *Faster R-CNN: Towards
   Real-Time Object Detection with Region Proposal Networks.* NeurIPS.
   arXiv:1506.01497.
9. He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for
   Image Recognition.* CVPR. arXiv:1512.03385.

# Appendix: Reproducibility

| Item | Value |
|---|---|
| Backbone | ResNet-18, ImageNet weights, `layer4` 7×7×512 |
| Feature extraction | {feature_meta['seconds']:.1f} s for {feature_meta['count']:,} images |
| Detector | {detector_meta.get('detector', 'Faster R-CNN MobileNetV3-320')} |
| Detection time | {detector_meta.get('seconds', 0) / 60:.1f} min |
| Object threshold | {config['detector']['score_thresh']} |
| Max detections | {config['detector']['max_detections']} |
| Optimiser | AdamW, OneCycle, lr={config['train']['lr']} |
| Epochs | {config['train']['epochs']} |
| Sampled pairs / epoch | {config['train']['samples_per_epoch']:,} |
| Hardware | Apple M4, 16 GB unified memory, macOS |
| Device | CPU for LSTM training; MPS for feature/detector extraction |
| MPS memory cap | {100 * config['runtime']['mps_memory_fraction']:.0f}% |

All metrics in this report are generated by `scripts/05_evaluate.py`; the
figures are generated by `scripts/06_visualize.py`. Run `scripts/07_make_report.py`
to rebuild this document.
"""
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-pdf", action="store_true")
    args = parser.parse_args()

    report_dir = ROOT / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    markdown = build_markdown()
    md_path = report_dir / "report.md"
    md_path.write_text(markdown, encoding="utf-8")
    print(f"Markdown -> {md_path}")

    pandoc = shutil.which("pandoc")
    if not pandoc:
        print("pandoc not found; leaving Markdown only")
        return

    docx_path = report_dir / "report.docx"
    subprocess.run(
        [
            pandoc,
            str(md_path),
            "-o",
            str(docx_path),
            "--resource-path",
            str(report_dir),
            "--standalone",
        ],
        check=True,
    )
    print(f"DOCX -> {docx_path}")

    if args.no_pdf:
        return

    # PDF is printed from a styled standalone HTML file: CSS gives much better
    # control over table widths and page breaks than a default DOCX conversion.
    css_path = report_dir / "report.css"
    css_path.write_text(REPORT_CSS, encoding="utf-8")
    html_path = report_dir / "report.html"
    subprocess.run(
        [
            pandoc,
            str(md_path),
            "-o",
            str(html_path),
            "--standalone",
            "--embed-resources",
            "--css",
            str(css_path),
            "--resource-path",
            str(report_dir),
            "--metadata",
            "lang=en",
        ],
        check=True,
    )
    print(f"HTML -> {html_path}")

    pdf_path = report_dir / "report.pdf"
    if pdf_path.exists():
        pdf_path.unlink()
    chrome = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
    if chrome.exists():
        profile = ROOT / ".cache" / "chrome-profile"
        profile.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            [
                str(chrome),
                "--headless=new",
                "--no-sandbox",
                "--disable-gpu",
                "--disable-software-rasterizer",
                "--disable-crash-reporter",
                "--no-pdf-header-footer",
                f"--user-data-dir={profile}",
                f"--print-to-pdf={pdf_path}",
                html_path.as_uri(),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0 or not pdf_path.exists():
            print("Chrome PDF failed; falling back to LibreOffice")
            print(result.stderr[-1200:])
        else:
            print(f"PDF -> {pdf_path} ({pdf_path.stat().st_size / 1024:.0f} KB, Chrome)")
            return

    soffice = shutil.which("soffice")
    if not soffice:
        print("No PDF engine available; leaving HTML/DOCX")
        return
    profile_dir = Path("/tmp") / "dsh_lo_profile"
    profile_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            soffice,
            "--headless",
            f"-env:UserInstallation=file://{profile_dir}",
            "--convert-to",
            "pdf",
            "--outdir",
            str(report_dir),
            str(docx_path),
        ],
        check=True,
    )
    if pdf_path.exists():
        print(f"PDF -> {pdf_path} ({pdf_path.stat().st_size / 1024:.0f} KB, LibreOffice)")
    else:
        print("PDF conversion was reported as successful but report.pdf is missing")


if __name__ == "__main__":
    main()

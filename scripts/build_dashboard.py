#!/usr/bin/env python
"""Build the static dashboard payload and assets from the project artifacts.

The dashboard is a plain static site: ``dashboard/data/dashboard-data.js`` is the
only data source it reads (``window.DASHBOARD_DATA``).  This script never
recomputes a metric -- every number is copied or derived arithmetically (percent
scaling, differences, means of saved per-image scores) from the artifacts that
the submitted report pipeline produced:

    data/processed/*.json
    outputs/metrics/*.json
    outputs/predictions/*_test.json
    outputs/checkpoints/*/history.json
    data/raw/val2017/*.jpg
    report/figures/*.png, report/*.docx, report/*.pdf

Usage
-----
    python scripts/build_dashboard.py
    python scripts/build_dashboard.py --config configs/default.yaml
    python scripts/build_dashboard.py --limit 25          # smoke run -> dashboard/data/smoke/
    python scripts/build_dashboard.py --out /tmp/d/data --assets-dir /tmp/d/assets

Required artifacts that are missing abort the build with a non-zero exit code.
Optional artifacts that are missing produce ``null`` / empty lists and a warning
on stderr -- never a fabricated number.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

MODELS: tuple[str, ...] = ("global", "attention", "object")

MODEL_META: dict[str, dict[str, Any]] = {
    "global": {
        "label": "Global CNN-LSTM",
        "short": "Global",
        "color": "#4C72B0",
        "description": "A single pooled image vector initialises the LSTM; no spatial access.",
        "uses_attention": False,
        "uses_objects": False,
    },
    "attention": {
        "label": "CNN-LSTM + Bahdanau Attention",
        "short": "Attention",
        "color": "#DD8452",
        "description": "Bahdanau attention over a 7x7 spatial grid at every decoding step.",
        "uses_attention": True,
        "uses_objects": False,
    },
    "object": {
        "label": "Object-aware Attention",
        "short": "Object-aware",
        "color": "#55A868",
        "description": "Attention plus score-weighted detected object-label embeddings.",
        "uses_attention": True,
        "uses_objects": True,
    },
}

BLEU_BINS: tuple[str, ...] = ("0-10", "10-20", "20-30", "30-40", "40+")

# Figures copied into dashboard/assets/figures/, in report order.
FIGURE_SPECS: tuple[tuple[str, str, str, str], ...] = (
    (
        "model_comparison",
        "Model comparison",
        "report/figures/model_comparison.png",
        "Caption quality (BLEU-1-4, CIDEr-D) and CHAIR hallucination for the three models on the 500-image test split; lower CHAIR is better.",
    ),
    (
        "bleu_vs_hallucination",
        "BLEU vs hallucination",
        "report/figures/bleu_vs_hallucination.png",
        "Sentence BLEU-4 against the number of hallucinated objects for all 1,500 pooled test captions, with a binned trend and bootstrap intervals.",
    ),
    (
        "training_curves",
        "Training curves",
        "report/figures/training_curves.png",
        "Training/validation loss and validation CHAIRs per epoch for each model (12 epochs, validation CHAIRs measured on the 500 validation images).",
    ),
    (
        "attention_maps",
        "Attention maps",
        "report/figures/attention_maps.png",
        "Word-by-word 7x7 attention distributions for four held-out images; each column after the image is the attention at one generated word.",
    ),
    (
        "qualitative_examples",
        "Qualitative examples",
        "report/figures/qualitative_examples.png",
        "Side-by-side captions for held-out images with hallucinated object words highlighted in red.",
    ),
)

# Timing that only exists in the run logs (no machine-readable artifact). The
# values are read back from those logs when present, so the payload cannot drift
# from the recorded run; ``None`` omits the runtime rather than inventing one.
FEATURE_LOG = "logs/extract_features.log"

MAX_IMAGE_WIDTH = 900
JPEG_QUALITY = 82


class BuildError(RuntimeError):
    """Raised when a required artifact is missing or unusable."""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def display_path(path: Path) -> str:
    """Repository-relative path when possible, absolute path otherwise."""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def require(path: Path, what: str) -> Any:
    if not path.exists():
        raise BuildError(f"missing required {what}: {display_path(path)}")
    try:
        return load_json(path)
    except json.JSONDecodeError as exc:  # pragma: no cover - corrupt input
        raise BuildError(f"invalid JSON in {display_path(path)}: {exc}") from exc


def optional(path: Path, warnings: list[str], what: str | None = None) -> Any:
    if not path.exists():
        warnings.append(f"optional {what or path.name} not found: {display_path(path)}")
        return None
    try:
        return load_json(path)
    except json.JSONDecodeError as exc:
        warnings.append(f"optional {path.name} is not valid JSON ({exc}); skipped")
        return None


def num(value: Any, digits: int = 2) -> float | None:
    """Round a finite number; ``None``/NaN/inf become ``None`` (never fabricate)."""
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    out = round(out, digits)
    return 0.0 if out == 0 else out


def pct(value: Any, digits: int = 2) -> float | None:
    """Convert a 0-1 fraction to a 0-100 percentage."""
    out = num(value, digits + 2)
    if out is None:
        return None
    return num(out * 100.0, digits)


def sig(value: Any) -> float | None:
    """Keep a p-value honest across the whole range instead of rounding it to 0.

    Values at or above 1e-4 keep four decimal places; smaller values keep three
    significant digits (``8.43e-16``) so no stored p-value silently becomes zero.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    if out == 0.0:
        return 0.0
    if abs(out) >= 1e-4:
        return round(out, 4)
    return float(f"{out:.3g}")


def as_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def rel(path: Path, base: Path) -> str:
    """Path relative to ``base`` rendered with forward slashes."""
    try:
        relpath = os.path.relpath(path.resolve(), base.resolve())
    except ValueError:  # different drives on Windows
        relpath = str(path.resolve())
    return Path(relpath).as_posix()


def params_label(count: int | None) -> str | None:
    if not count:
        return None
    if count >= 1_000_000:
        return f"{count / 1_000_000:.2f} M"
    if count >= 1_000:
        return f"{count / 1_000:.1f} K"
    return str(count)


def history_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("history"), list):
        return [row for row in payload["history"] if isinstance(row, dict)]
    return []


def seed_from_config(path: Path, default: int = 42) -> int:
    """Read ``seed:`` from the YAML config without importing a YAML parser."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return default
    match = re.search(r"^seed:\s*(\d+)\s*$", text, flags=re.MULTILINE)
    return int(match.group(1)) if match else default


def pdf_pages(path: Path) -> int | None:
    """Page count via ``pdfinfo``; ``None`` when the tool is unavailable."""
    if not path.exists():
        return None
    binary = shutil.which("pdfinfo")
    if binary is None:
        return None
    try:
        result = subprocess.run(
            [binary, str(path)],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"^Pages:\s+(\d+)\s*$", result.stdout, flags=re.MULTILINE)
    return int(match.group(1)) if match else None


def evaluation_runtime(log_path: Path) -> str | None:
    """Span of the last evaluation run in ``logs/evaluate.log``.

    Timestamps are minute-resolution, so this reports a coarse ``< N s`` value
    and is only used when the log exists.
    """
    if not log_path.exists():
        return None
    stamps: list[int] = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = re.match(r"^(\d{2}):(\d{2}):(\d{2})\s*\|", line)
        if match:
            hours, minutes, seconds = (int(part) for part in match.groups())
            stamps.append(hours * 3600 + minutes * 60 + seconds)
    if len(stamps) < 2:
        return None
    # Keep the final contiguous block (split on gaps longer than two minutes).
    start = len(stamps) - 1
    while start > 0 and stamps[start] - stamps[start - 1] <= 120:
        start -= 1
    span = stamps[-1] - stamps[start]
    return f"< 10 s" if span < 10 else f"~{span} s"


def log_seconds(path: Path, pattern: str) -> float | None:
    """Last ``<number>s`` matched by ``pattern`` in a run log, in seconds."""
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    matches = re.findall(pattern, text)
    if not matches:
        return None
    return float(matches[-1])


def resize_rgb(path: Path, max_width: int = MAX_IMAGE_WIDTH) -> Image.Image:
    with Image.open(path) as handle:
        image = handle.convert("RGB")
    if image.width > max_width:
        height = max(1, round(image.height * max_width / image.width))
        image = image.resize((max_width, height), Image.LANCZOS)
    return image


def write_jpeg(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="JPEG", quality=JPEG_QUALITY, optimize=True)


def render_overlay(base: Image.Image, grid: Sequence[Sequence[float]], path: Path) -> None:
    """Render a 7x7 attention grid over ``base`` with the inferno colormap."""
    array = np.asarray(grid, dtype=float)
    if array.ndim != 2 or array.size == 0:
        raise BuildError(f"attention grid is not a 2-D array (shape={array.shape})")
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        raise BuildError("attention grid contains no finite values")
    low = float(np.min(finite))
    high = float(np.percentile(finite, 99))
    if not math.isfinite(high) or high <= low:
        high = low + 1e-9
    normalised = np.clip((array - low) / (high - low), 0.0, 1.0)

    width, height = base.size
    dpi = 100
    figure = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi, frameon=False)
    axis = figure.add_axes((0.0, 0.0, 1.0, 1.0))
    axis.set_axis_off()
    axis.imshow(base, interpolation="bilinear")
    axis.imshow(
        normalised,
        cmap="inferno",
        alpha=0.5,
        extent=(0, width, height, 0),
        interpolation="bilinear",
        aspect="auto",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, format="png", dpi=dpi, facecolor="white", pad_inches=0)
    plt.close(figure)


def copy_file(source: Path, destination: Path) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return destination.stat().st_size


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


class Builder:
    def __init__(self, out_dir: Path, assets_dir: Path, limit: int | None, config: Path):
        self.out_dir = out_dir
        self.assets_dir = assets_dir
        # Every payload path is relative to the directory that contains the
        # data directory (``dashboard/`` for the real build).
        self.base_dir = out_dir.parent
        self.limit = limit
        self.config = config
        self.warnings: list[str] = []
        self.used_images: set[int] = set()
        self.overlay_count = 0

        self.paths = {
            "feature_log": ROOT / FEATURE_LOG,
            "stats": ROOT / "data/processed/stats.json",
            "splits": ROOT / "data/processed/splits.json",
            "captions": ROOT / "data/processed/captions.json",
            "objects_gt": ROOT / "data/processed/objects_gt.json",
            "image_index": ROOT / "data/processed/image_index.json",
            "categories": ROOT / "data/processed/categories.json",
            "model_comparison": ROOT / "outputs/metrics/model_comparison.json",
            "evaluation_summary": ROOT / "outputs/metrics/evaluation_summary.json",
            "correlation": ROOT / "outputs/metrics/correlation.json",
            "model_comparison_csv": ROOT / "outputs/metrics/model_comparison.csv",
            "raw_images": ROOT / "data/raw/val2017",
        }
        for model in MODELS:
            self.paths[f"metrics_{model}"] = ROOT / f"outputs/metrics/{model}_test.json"
            self.paths[f"predictions_{model}"] = ROOT / f"outputs/predictions/{model}_test.json"
            self.paths[f"history_{model}"] = ROOT / f"outputs/checkpoints/{model}/history.json"
            self.paths[f"summary_{model}"] = ROOT / f"outputs/checkpoints/{model}/training_summary.json"

    # -- loading ---------------------------------------------------------

    def load(self) -> None:
        self.stats = require(self.paths["stats"], "dataset statistics")
        self.splits = require(self.paths["splits"], "dataset splits")
        self.captions = require(self.paths["captions"], "captions")
        self.objects_gt = require(self.paths["objects_gt"], "ground-truth objects")
        self.image_index = require(self.paths["image_index"], "image index")
        categories = require(self.paths["categories"], "category list")
        self.category_names = [str(name) for name in categories.get("names", [])]
        if not self.category_names:
            raise BuildError("categories.json has no 'names' list")

        self.comparison = require(self.paths["model_comparison"], "model comparison table")
        self.evaluation = require(self.paths["evaluation_summary"], "evaluation summary")
        self.correlation = require(self.paths["correlation"], "correlation analysis")
        self.per_image: dict[str, dict[int, dict[str, Any]]] = {}
        self.records: dict[str, list[dict[str, Any]]] = {}
        for model in MODELS:
            payload = require(self.paths[f"metrics_{model}"], f"{model} per-image metrics")
            records = payload.get("per_image") if isinstance(payload, dict) else None
            if not isinstance(records, list) or not records:
                raise BuildError(
                    f"outputs/metrics/{model}_test.json has no non-empty 'per_image' list"
                )
            self.records[model] = records
            self.per_image[model] = {int(row["image_id"]): row for row in records}

        self.histories = {}
        for model in MODELS:
            history = require(self.paths[f"history_{model}"], f"{model} training history")
            rows = history_records(history)
            if not rows:
                raise BuildError(f"outputs/checkpoints/{model}/history.json has no epoch records")
            self.histories[model] = rows

        self.attention_predictions = require(
            self.paths["predictions_attention"], "attention predictions"
        )
        self.detections = optional(ROOT / "data/processed/detections.json", self.warnings)
        self.detections_meta = optional(
            ROOT / "data/processed/detections_meta.json", self.warnings
        )
        self.matched = optional(ROOT / "outputs/metrics/bleu_matched_analysis.json", self.warnings)
        self.cluster = optional(
            ROOT / "outputs/metrics/pooled_cluster_bootstrap.json", self.warnings
        )
        self.pairwise_raw = optional(
            ROOT / "outputs/metrics/pairwise_comparisons.json", self.warnings
        )
        self.train_summaries = {
            model: optional(self.paths[f"summary_{model}"], self.warnings) for model in MODELS
        }

        # Test images, limited for smoke runs.
        test_ids = sorted(int(value) for value in self.splits.get("test", []))
        if not test_ids:
            raise BuildError("splits.json contains no 'test' split")
        if self.limit is not None:
            test_ids = test_ids[: max(0, self.limit)]
            if not test_ids:
                raise BuildError("--limit must be a positive integer")
        self.test_ids = test_ids

    def available_ids(self) -> list[int]:
        """Test-image ids that every per-image artifact agrees on."""
        return [
            image_id
            for image_id in self.test_ids
            if all(image_id in self.per_image[model] for model in MODELS)
        ]

    # -- payload sections ------------------------------------------------

    def build_meta(self) -> dict[str, Any]:
        stats = self.stats
        splits = {key: as_int(stats.get("splits", {}).get(key)) for key in ("train", "val", "test")}
        train_captions = sum(
            len(self.captions.get(str(image_id), []))
            for image_id in self._split_ids("train")
        )
        test_captions = sum(
            len(self.captions.get(str(image_id), []))
            for image_id in self._split_ids("test")
        )
        return {
            "title": "Do Better Captions Hallucinate Less?",
            "subtitle": (
                "Measuring Object Hallucination in CNN-LSTM Image Captioning "
                "with and without Attention"
            ),
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "split": "test",
            "split_note": (
                "500 held-out COCO val2017 images, never seen during training or "
                "model selection; each image has about 5 reference captions."
            ),
            "n_test": splits["test"],
            "n_train": splits["train"],
            "n_val": splits["val"],
            "n_total": as_int(stats.get("n_images")),
            "vocab_size": as_int(stats.get("vocab_size")),
            "train_captions": train_captions,
            "test_captions": test_captions,
            "seed": seed_from_config(self.config),
            "device_note": (
                "Apple M4 / 16 GB: CPU for LSTM training, MPS for feature and "
                "detector extraction (MPS memory fraction 0.55)."
            ),
            "dataset_note": (
                "COCO val2017 images re-split deterministically into "
                f"{splits['train']:,}/{splits['val']:,}/{splits['test']:,} "
                "(train/val/test); the vocabulary is built from training captions only."
            ),
        }

    def _split_ids(self, name: str) -> list[int]:
        return [int(value) for value in self.splits.get(name, [])]

    def build_models(self) -> list[dict[str, Any]]:
        rows: dict[str, dict[str, Any]] = {}
        for entry in self.comparison:
            rows[str(entry.get("model"))] = entry
        missing = [model for model in MODELS if model not in rows]
        if missing:
            raise BuildError(f"model_comparison.json is missing rows for: {', '.join(missing)}")

        models: list[dict[str, Any]] = []
        for model in MODELS:
            entry = rows[model]
            meta = MODEL_META[model]
            history = self.histories[model]
            seconds = sum(float(row.get("seconds") or 0.0) for row in history)
            # Copied from the saved evaluation run: --limit only restricts which
            # images get copied and rendered, never these corpus-level numbers.
            records = self.records[model]

            sentence_bleu4 = [
                float(row["sentence_bleu4"])
                for row in records
                if isinstance(row.get("sentence_bleu4"), (int, float))
            ]
            unique_captions = len({str(row.get("caption", "")).strip() for row in records} - {""})
            summary = self.train_summaries.get(model) or {}
            params = as_int(summary.get("parameters")) if isinstance(summary, dict) else None
            bleu4 = num(entry.get("bleu4"), 2)
            models.append(
                {
                    "id": model,
                    "label": meta["label"],
                    "short": meta["short"],
                    "color": meta["color"],
                    "description": meta["description"],
                    "params": params,
                    "params_label": params_label(params),
                    "train_minutes": num(seconds / 60.0, 1),
                    "bleu1": num(entry.get("bleu1"), 2),
                    "bleu2": num(entry.get("bleu2"), 2),
                    "bleu3": num(entry.get("bleu3"), 2),
                    "bleu4": bleu4,
                    "bleu": bleu4,
                    "cider_d": num(entry.get("cider_d"), 3),
                    "chair_s": num(entry.get("chair_s"), 2),
                    "chair_i": num(entry.get("chair_i"), 2),
                    "object_precision": num(entry.get("object_precision"), 2),
                    "object_recall": num(entry.get("object_recall"), 2),
                    "object_f1": num(entry.get("object_f1"), 2),
                    "n_mentions": as_int(entry.get("n_mentions")),
                    "n_hallucinated": as_int(entry.get("n_hallucinated")),
                    "sentence_bleu4_mean": (
                        num(sum(sentence_bleu4) / len(sentence_bleu4), 2)
                        if sentence_bleu4
                        else None
                    ),
                    "unique_captions": unique_captions if records else None,
                    "uses_attention": meta["uses_attention"],
                    "uses_objects": meta["uses_objects"],
                }
            )
        return models

    def build_pairwise(self) -> list[dict[str, Any]]:
        summary = self.evaluation.get("pairwise") if isinstance(self.evaluation, dict) else None
        if not isinstance(summary, dict) or not summary:
            summary = self.pairwise_raw if isinstance(self.pairwise_raw, dict) else {}
        if not summary:
            self.warnings.append("no pairwise comparison data available")
            return []

        spec = (
            ("global_vs_attention", "global", "attention", "Global vs Attention"),
            ("global_vs_object", "global", "object", "Global vs Object-aware"),
            ("attention_vs_object", "attention", "object", "Attention vs Object-aware"),
        )
        out: list[dict[str, Any]] = []
        for key, first, second, label in spec:
            entry = summary.get(key)
            if not isinstance(entry, dict):
                self.warnings.append(f"pairwise entry missing: {key}")
                continue
            objects = entry.get("hallucinated_objects", {}) or {}
            rate = entry.get("hallucination_rate", {}) or {}
            bleu = entry.get("sentence_bleu4", {}) or {}
            mcnemar = entry.get("mcnemar", {}) or {}
            rate_diff = num(rate.get("mean_diff"), 4)
            out.append(
                {
                    "id": key,
                    "first": first,
                    "second": second,
                    "label": label,
                    "objects_diff": num(objects.get("mean_diff"), 3),
                    "objects_ci": [num(objects.get("ci_low"), 3), num(objects.get("ci_high"), 3)],
                    "objects_p": sig(objects.get("p_two_sided")),
                    "rate_diff_pp": (
                        num(rate_diff * 100.0, 1) if rate_diff is not None else None
                    ),
                    "rate_ci_pp": [pct(rate.get("ci_low"), 1), pct(rate.get("ci_high"), 1)],
                    "bleu4_diff": num(bleu.get("mean_diff"), 3),
                    "bleu4_ci": [num(bleu.get("ci_low"), 3), num(bleu.get("ci_high"), 3)],
                    "mcnemar_p": sig(mcnemar.get("p_value")),
                    "only_first": as_int(mcnemar.get("only_first_hallucinates")),
                    "only_second": as_int(mcnemar.get("only_second_hallucinates")),
                }
            )
        return out

    def build_correlations(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key in (*MODELS, "pooled"):
            entry = self.correlation.get(key) if isinstance(self.correlation, dict) else None
            analysis = entry.get("analysis") if isinstance(entry, dict) else None
            if not isinstance(analysis, dict):
                self.warnings.append(f"correlation entry missing: {key}")
                out[key] = {"n": None, "rho": None, "p": None, "rate": None}
                continue
            spearman = analysis.get("spearman_bleu4_vs_n_hallucinated", {}) or {}
            out[key] = {
                "n": as_int(analysis.get("n")),
                "rho": num(spearman.get("rho"), 3),
                "p": sig(spearman.get("p_value")),
                "rate": pct(analysis.get("hallucination_rate")),
            }
        return out

    def build_quartiles(self) -> list[dict[str, Any]]:
        pooled = (self.correlation.get("pooled") or {}).get("analysis") or {}
        buckets = pooled.get("bleu4_quartiles") or []
        out: list[dict[str, Any]] = []
        for bucket in buckets:
            out.append(
                {
                    "bucket": str(bucket.get("bucket")),
                    "n": as_int(bucket.get("n")),
                    "bleu_min": num(bucket.get("bleu4_min"), 2),
                    "bleu_max": num(bucket.get("bleu4_max"), 2),
                    "bleu_mean": num(bucket.get("bleu4_mean"), 2),
                    "rate": pct(bucket.get("hallucination_rate")),
                    "ci": [
                        pct(bucket.get("hallucination_rate_ci_low")),
                        pct(bucket.get("hallucination_rate_ci_high")),
                    ],
                    "mean_hallucinated": num(bucket.get("mean_hallucinated_objects"), 2),
                }
            )
        return out

    def build_matched_bleu(self) -> list[dict[str, Any]]:
        if not isinstance(self.matched, list):
            return []
        indexed: dict[tuple[int, int, str], dict[str, Any]] = {}
        for row in self.matched:
            bounds = row.get("bleu_bin") or []
            if len(bounds) != 2:
                continue
            low, high = int(bounds[0]), int(bounds[1])
            label = "40+" if low >= 40 else f"{low}-{high}"
            indexed[(low, high, str(row.get("model")))] = (label, row)

        # Order bin then model, using the canonical bin/model orders.
        out: list[dict[str, Any]] = []
        ordered_keys = [
            (0, 10),
            (10, 20),
            (20, 30),
            (30, 40),
            (40, 101),
        ]
        for bounds in ordered_keys:
            for model in MODELS:
                entry = indexed.get((bounds[0], bounds[1], model))
                if entry is None:
                    continue
                label, row = entry
                out.append(
                    {
                        "bin": label,
                        "model": model,
                        "n": as_int(row.get("n")),
                        "mean_bleu4": num(row.get("mean_sentence_bleu4"), 2),
                        "rate": pct(row.get("hallucination_rate")),
                        "ci": [
                            pct((row.get("hallucination_rate_ci") or [None, None])[0]),
                            pct((row.get("hallucination_rate_ci") or [None, None])[1]),
                        ],
                    }
                )
        if len(out) != 15:
            self.warnings.append(f"matched-BLEU analysis produced {len(out)} rows (expected 15)")
        return out

    def build_training(self) -> dict[str, list[dict[str, Any]]]:
        out: dict[str, list[dict[str, Any]]] = {}
        for model in MODELS:
            rows: list[dict[str, Any]] = []
            for record in self.histories[model]:
                rows.append(
                    {
                        "epoch": as_int(record.get("epoch")),
                        "train_loss": num(record.get("train_loss"), 3),
                        "val_loss": num(record.get("val_loss"), 3),
                        "val_chair_s": pct(record.get("val_chair_s")),
                        "val_bleu4": num(record.get("val_bleu4"), 3),
                    }
                )
            out[model] = rows
        return out

    def build_dataset(self) -> dict[str, Any]:
        stats = self.stats
        split_sizes = {
            name: as_int(stats.get("splits", {}).get(name))
            for name in ("train", "val", "test")
        }
        caption_counts = {
            name: sum(len(self.captions.get(str(image_id), [])) for image_id in self._split_ids(name))
            for name in ("train", "val", "test")
        }
        lengths = stats.get("caption_lengths", {}) or {}
        objects = stats.get("objects_per_image", {}) or {}

        object_counts: Counter[str] = Counter()
        for names in self.objects_gt.values():
            object_counts.update(str(name) for name in names)
        canonical = set(self.category_names)
        top_objects = [
            {"name": name, "count": count}
            for name, count in sorted(
                object_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )
            if name in canonical
        ][:12]

        detector: dict[str, Any] | None = None
        if isinstance(self.detections, dict) and self.detections:
            lengths_detections = [
                len(record.get("labels", [])) for record in self.detections.values()
            ]
            meta = self.detections_meta if isinstance(self.detections_meta, dict) else {}
            total = sum(lengths_detections)
            detector = {
                "name": meta.get("detector"),
                "threshold": num(meta.get("score_threshold"), 3),
                "max_detections": as_int(meta.get("max_detections")),
                "mean_detections": num(total / len(lengths_detections), 2),
                "empty_images": sum(1 for value in lengths_detections if value == 0),
                "seconds": num(meta.get("seconds"), 1),
                "images": as_int(meta.get("images")) or len(lengths_detections),
            }

        return {
            "splits": split_sizes,
            "captions": caption_counts,
            "caption_length": {
                "mean": num(lengths.get("mean")),
                "p50": as_int(lengths.get("p50")),
                "p95": as_int(lengths.get("p95")),
                "max": as_int(lengths.get("max")),
            },
            "objects_per_image": {
                "mean": num(objects.get("mean")),
                "max": as_int(objects.get("max")),
            },
            "top_objects": top_objects,
            "detector": detector,
        }

    # -- per-image selections --------------------------------------------

    def _image_id_list(self) -> list[int]:
        return self.available_ids()

    def _image_rel(self, image_id: int) -> str:
        return f"{rel(self.assets_dir / 'images', self.base_dir)}/{image_id}.jpg"

    def _copy_image(self, image_id: int) -> None:
        if image_id in self.used_images:
            return
        index = self.image_index.get(str(image_id))
        if not isinstance(index, dict) or "file_name" not in index:
            raise BuildError(f"image_index.json has no entry for test image {image_id}")
        source = self.paths["raw_images"] / index["file_name"]
        if not source.exists():
            raise BuildError(f"missing raw image for test image {image_id}: {source}")
        write_jpeg(resize_rgb(source), self.assets_dir / "images" / f"{image_id}.jpg")
        self.used_images.add(image_id)

    def _caption_block(self, image_id: int) -> dict[str, dict[str, Any]]:
        block: dict[str, dict[str, Any]] = {}
        for model in MODELS:
            record = self.per_image[model][image_id]
            block[model] = {
                "text": str(record.get("caption", "")),
                "hallucinated": [str(word) for word in record.get("hallucinated", [])],
            }
        return block

    def _example(self, image_id: int, record: dict[str, Any]) -> dict[str, Any]:
        return {
            "image_id": image_id,
            "image": self._image_rel(image_id),
            "references": [str(text) for text in record.get("references", [])],
            "gt_objects": [str(name) for name in record.get("reference_objects", [])],
            "captions": self._caption_block(image_id),
        }

    def build_hallucination(self) -> dict[str, Any]:
        canonical = set(self.category_names)
        counts: Counter[str] = Counter()
        for model in MODELS:
            for record in self.records[model][: len(self.test_ids)]:
                for word in record.get("hallucinated", []):
                    if str(word) in canonical:
                        counts[str(word)] += 1
        top = [
            {"name": name, "count": count}
            for name, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:12]
        ]

        candidates: list[tuple[int, int]] = []
        for image_id in self._image_id_list():
            hallucinating = sum(
                1 for model in MODELS if self.per_image[model][image_id].get("hallucinated")
            )
            if 1 <= hallucinating < len(MODELS):
                candidates.append((hallucinating, image_id))
        candidates.sort(key=lambda item: (item[0], item[1]))

        examples: list[dict[str, Any]] = []
        for _, image_id in candidates[:6]:
            record = self.per_image[MODELS[0]][image_id]
            self._copy_image(image_id)
            examples.append(self._example(image_id, record))
        return {"top_hallucinated": top, "examples": examples}

    def build_gallery(self) -> list[dict[str, Any]]:
        ranked: list[tuple[int, int, int]] = []
        for image_id in self._image_id_list():
            sets = [
                frozenset(self.per_image[model][image_id].get("hallucinated", []))
                for model in MODELS
            ]
            ranked.append((-len(set(sets)), -sum(1 for entry in sets if entry), image_id))
        ranked.sort()
        gallery: list[dict[str, Any]] = []
        for _, _, image_id in ranked[:8]:
            record = self.per_image[MODELS[0]][image_id]
            self._copy_image(image_id)
            gallery.append(self._example(image_id, record))
        return gallery

    def _attention_highlight_indices(self, words: Sequence[str]) -> list[int]:
        """Word indices that spell a COCO object class name (greedy longest match)."""
        lowered = [str(word).lower() for word in words]
        canonical = set(self.category_names)
        indices: list[int] = []
        position = 0
        while position < len(lowered):
            for span in (3, 2, 1):
                if position + span > len(lowered):
                    continue
                phrase = " ".join(lowered[position : position + span])
                if phrase in canonical:
                    indices.extend(range(position, position + span))
                    position += span
                    break
            else:
                position += 1
        seen: list[int] = []
        for index in indices:
            if index not in seen:
                seen.append(index)
        return seen

    def build_attention_examples(self) -> list[dict[str, Any]]:
        predictions = [
            row
            for row in self.attention_predictions
            if isinstance(row, dict) and row.get("words") and row.get("attention")
        ]
        predictions.sort(key=lambda row: int(row.get("image_id", 0)))
        predictions = [
            row
            for row in predictions
            if int(row.get("image_id", 0)) in set(self.test_ids)
        ]

        preferred: list[tuple[dict[str, Any], list[int]]] = []
        fallback: list[tuple[dict[str, Any], list[int]]] = []
        for row in predictions:
            words = [str(word) for word in row["words"]]
            highlights = self._attention_highlight_indices(words)
            if not highlights:
                continue
            entry = (row, highlights)
            fallback.append(entry)
            if 4 <= len(words) <= 14 and 2 <= len(highlights) <= 4:
                preferred.append(entry)
        chosen = preferred[:4] if len(preferred) >= 4 else fallback[:4]

        examples: list[dict[str, Any]] = []
        for row, highlights in chosen:
            image_id = int(row["image_id"])
            grid = row["attention"]
            if len(grid) != len(row["words"]):
                raise BuildError(
                    f"attention predictions for image {image_id} have "
                    f"{len(grid)} maps for {len(row['words'])} words"
                )
            self._copy_image(image_id)
            base = resize_rgb(self.paths["raw_images"] / self.image_index[str(image_id)]["file_name"])
            overlays: list[dict[str, Any]] = []
            for index in highlights:
                destination = self.assets_dir / "attention" / f"{image_id}_{index}.png"
                render_overlay(base, grid[index], destination)
                self.overlay_count += 1
                overlays.append(
                    {
                        "index": index,
                        "word": str(row["words"][index]),
                        "path": f"{rel(self.assets_dir / 'attention', self.base_dir)}/{destination.name}",
                    }
                )
            examples.append(
                {
                    "image_id": image_id,
                    "image": self._image_rel(image_id),
                    "caption": str(row.get("caption", "")),
                    "words": [str(word) for word in row["words"]],
                    "overlays": overlays,
                    "highlight": sorted(highlights),
                }
            )
        return examples

    # -- static assets ----------------------------------------------------

    def build_figures(self) -> list[dict[str, Any]]:
        figures: list[dict[str, Any]] = []
        for figure_id, label, source_rel, caption in FIGURE_SPECS:
            source = ROOT / source_rel
            if not source.exists():
                self.warnings.append(f"optional figure not found: {source_rel}")
                continue
            destination = self.assets_dir / "figures" / f"{figure_id}.png"
            copy_file(source, destination)
            figures.append(
                {
                    "id": figure_id,
                    "label": label,
                    "path": f"{rel(self.assets_dir / 'figures', self.base_dir)}/{destination.name}",
                    "caption": caption,
                }
            )
        return figures

    def build_artifacts(self) -> list[dict[str, Any]]:
        spec = (
            (
                "MAD-format report (DOCX)",
                ROOT / "report/MAD_Report_Image_Captioning.docx",
                "docx",
            ),
            (
                "MAD-format report (PDF)",
                ROOT / "report/MAD_Report_Image_Captioning.pdf",
                "pdf",
            ),
            (
                "Model comparison table (CSV)",
                self.paths["model_comparison_csv"],
                "csv",
            ),
        )
        artifacts: list[dict[str, Any]] = []
        for label, source, kind in spec:
            if not source.exists():
                self.warnings.append(f"optional download not found: {rel(source, ROOT)}")
                continue
            destination = self.assets_dir / "downloads" / source.name
            size = copy_file(source, destination)
            artifacts.append(
                {
                    "label": label,
                    "path": f"{rel(self.assets_dir / 'downloads', self.base_dir)}/{destination.name}",
                    "kind": kind,
                    "size": size,
                }
            )
        return artifacts

    # -- headline / tasks / pipeline --------------------------------------

    def build_headline(
        self,
        models: list[dict[str, Any]],
        matched: Sequence[dict[str, Any]],
        correlations: dict[str, Any],
    ) -> dict[str, Any]:
        by_id = {model["id"]: model for model in models}
        global_model = by_id["global"]
        object_model = by_id["object"]
        # Differences are computed from the unrounded source values and only
        # then rounded, so the headline is not a difference of two roundings.
        raw = {str(entry.get("model")): entry for entry in self.comparison}
        raw_global = raw.get("global", {})
        raw_object = raw.get("object", {})
        reduction = None
        if raw_global.get("chair_s") is not None and raw_object.get("chair_s") is not None:
            reduction = num(float(raw_global["chair_s"]) - float(raw_object["chair_s"]), 2)
        bleu_gain = None
        if raw_global.get("bleu4") is not None and raw_object.get("bleu4") is not None:
            bleu_gain = num(float(raw_object["bleu4"]) - float(raw_global["bleu4"]), 2)

        pooled = correlations.get("pooled", {})
        cluster = {}
        if isinstance(self.cluster, dict):
            cluster = self.cluster.get("bleu4_vs_n_hallucinated") or {}

        matched_bin = "10-20"
        matched_rates: dict[str, float | None] = {}
        for row in matched:
            if row["bin"] == matched_bin:
                matched_rates[row["model"]] = row["rate"]

        answer = self._headline_answer(
            object_model, global_model, pooled, cluster, matched_rates, reduction
        )
        return {
            "question": "Do better captions hallucinate less?",
            "answer": answer,
            "best_model": "object",
            "best_bleu4": object_model["bleu4"],
            "best_chair_s": object_model["chair_s"],
            "pooled_rho": pooled.get("rho"),
            "pooled_p": pooled.get("p"),
            "cluster_rho": num(cluster.get("rho"), 3),
            "cluster_ci": [num(cluster.get("ci_low"), 3), num(cluster.get("ci_high"), 3)],
            "cluster_p": sig(cluster.get("p_two_sided")),
            "matched_bin": matched_bin,
            "matched_rates": matched_rates,
            "hallucination_reduction_pp": reduction,
            "bleu_gain_points": bleu_gain,
        }

    def _headline_answer(
        self,
        object_model: dict[str, Any],
        global_model: dict[str, Any],
        pooled: dict[str, Any],
        cluster: dict[str, Any],
        matched_rates: dict[str, float | None],
        reduction: float | None,
    ) -> str:
        parts: list[str] = []
        n_images = as_int(self.comparison[0].get("n")) if self.comparison else None
        scope = f"{n_images} held-out COCO images" if n_images else "held-out COCO images"
        if object_model["bleu4"] is not None and object_model["chair_s"] is not None:
            parts.append(
                f"On {scope} the object-aware model has the best "
                f"BLEU-4 ({object_model['bleu4']:.2f}) and the lowest CHAIRs "
                f"({object_model['chair_s']:.2f}%)"
            )
            if reduction is not None:
                parts[-1] += (
                    f", {abs(reduction):.2f} percentage points below the global model's "
                    f"{global_model['chair_s']:.2f}%"
                )
            parts[-1] += "."
        if pooled.get("rho") is not None:
            pooled_n = pooled.get("n")
            pool_scope = f"{pooled_n:,} pooled captions" if pooled_n else "pooled captions"
            sentence = (
                f"Across {pool_scope}, higher sentence BLEU-4 is associated "
                f"with fewer hallucinated objects (Spearman ρ = {pooled['rho']:.3f}"
            )
            if cluster.get("ci_low") is not None and cluster.get("ci_high") is not None:
                sentence += (
                    f"; image-cluster bootstrap 95% CI "
                    f"{cluster['ci_low']:.3f} to {cluster['ci_high']:.3f}"
                )
            sentence += ")"
            parts.append(sentence + ".")
        rates = [rate for rate in matched_rates.values() if rate is not None]
        if len(rates) == 3:
            parts.append(
                f"At matched BLEU-4 in the 10-20 bin the models still range from "
                f"{min(rates):.1f}% to {max(rates):.1f}% hallucinated object mentions, "
                f"so BLEU overlap alone does not identify a grounded captioner."
            )
        return " ".join(parts) if parts else "Metrics unavailable."

    def build_tasks(
        self,
        models: list[dict[str, Any]],
        examples: Sequence[dict[str, Any]],
        report_pages: int | None,
        correlations: dict[str, Any],
        matched: Sequence[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        by_id = {model["id"]: model for model in models}
        reduction = None
        if by_id["global"]["chair_s"] is not None and by_id["object"]["chair_s"] is not None:
            reduction = num(by_id["global"]["chair_s"] - by_id["object"]["chair_s"], 2)
        total_captions = sum(as_int(value) or 0 for value in (self._split_captions().values()))
        overlay_total = sum(len(example["overlays"]) for example in examples)
        pooled = correlations.get("pooled", {})
        rates = [row["rate"] for row in matched if row["bin"] == "10-20" and row["rate"] is not None]
        gap = num(max(rates) - min(rates), 1) if len(rates) == 3 else None
        pages = f"{report_pages}-page report" if report_pages else "report"

        def evidence(*items: tuple[str, str | Path]) -> list[dict[str, str]]:
            out: list[dict[str, str]] = []
            for label, target in items:
                path = target if isinstance(target, Path) else ROOT / target
                if path.exists():
                    out.append({"label": label, "path": rel(path, self.base_dir)})
            return out

        downloads = self.assets_dir / "downloads"
        figures = self.assets_dir / "figures"

        return [
            {
                "id": "dataset",
                "label": "Dataset with captions + object annotations",
                "description": (
                    "COCO val2017 images re-split into train/val/test with a fixed seed; "
                    "captions tokenised into a training-only vocabulary and per-image "
                    "ground-truth object classes extracted from the COCO annotations."
                ),
                "status": "complete",
                "metric": (
                    f"{as_int(self.stats.get('n_images')):,} images · "
                    f"{len(self.category_names)} classes · {total_captions:,} captions"
                ),
                "evidence": evidence(
                    ("download", "scripts/00_download_data.sh"),
                    ("prepare data", "scripts/01_prepare_data.py"),
                    ("stats", "data/processed/stats.json"),
                    ("captions", "data/processed/captions.json"),
                    ("object annotations", "data/processed/objects_gt.json"),
                    ("categories", "data/processed/categories.json"),
                ),
            },
            {
                "id": "global",
                "label": "CNN-LSTM captioner without attention",
                "description": (
                    "ResNet-18 features are pooled to one vector that initialises an "
                    "LSTM decoder; the model has no spatial access and serves as the "
                    "baseline for both caption quality and hallucination."
                ),
                "status": "complete",
                "metric": (
                    f"BLEU-4 {by_id['global']['bleu4']:.2f} · CIDEr-D "
                    f"{by_id['global']['cider_d']:.3f} · CHAIRs "
                    f"{by_id['global']['chair_s']:.2f}%"
                ),
                "evidence": evidence(
                    ("train", "scripts/04_train.py"),
                    ("training history", "outputs/checkpoints/global/history.json"),
                    ("test metrics", "outputs/metrics/global_test.json"),
                    ("predictions", "outputs/predictions/global_test.json"),
                ),
            },
            {
                "id": "attention",
                "label": "CNN-LSTM with Bahdanau attention",
                "description": (
                    "The same backbone and decoder, but each decoding step attends over "
                    "the 7x7 spatial grid, letting nouns focus on image regions."
                ),
                "status": "complete",
                "metric": (
                    f"BLEU-4 {by_id['attention']['bleu4']:.2f} · CIDEr-D "
                    f"{by_id['attention']['cider_d']:.3f} · CHAIRs "
                    f"{by_id['attention']['chair_s']:.2f}%"
                ),
                "evidence": evidence(
                    ("train", "scripts/04_train.py"),
                    ("training history", "outputs/checkpoints/attention/history.json"),
                    ("test metrics", "outputs/metrics/attention_test.json"),
                    ("predictions with attention", "outputs/predictions/attention_test.json"),
                ),
            },
            {
                "id": "object",
                "label": "Object-label variant",
                "description": (
                    "Attention plus score-weighted embeddings of the detected object "
                    "labels (Faster R-CNN, threshold 0.3, at most 12 boxes), giving the "
                    "decoder grounded object vocabulary at every step."
                ),
                "status": "complete",
                "metric": (
                    f"BLEU-4 {by_id['object']['bleu4']:.2f} · CIDEr-D "
                    f"{by_id['object']['cider_d']:.3f} · CHAIRs "
                    f"{by_id['object']['chair_s']:.2f}%"
                ),
                "evidence": evidence(
                    ("detector", "scripts/03_detect_objects.py"),
                    ("train", "scripts/04_train.py"),
                    ("training history", "outputs/checkpoints/object/history.json"),
                    ("test metrics", "outputs/metrics/object_test.json"),
                ),
            },
            {
                "id": "metrics",
                "label": "BLEU-1-4 and CIDEr-D evaluation",
                "description": (
                    "Corpus-level BLEU-1-4 and CIDEr-D on the 500-image test split, with "
                    "1,000-sample bootstrap confidence intervals for paired differences."
                ),
                "status": "complete",
                "metric": (
                    f"best BLEU-4 {by_id['object']['bleu4']:.2f} · best CIDEr-D "
                    f"{by_id['object']['cider_d']:.3f} (object-aware)"
                ),
                "evidence": evidence(
                    ("evaluate", "scripts/05_evaluate.py"),
                    ("comparison table", "outputs/metrics/model_comparison.json"),
                    ("comparison CSV", "outputs/metrics/model_comparison.csv"),
                    ("comparison figure", "report/figures/model_comparison.png"),
                ),
            },
            {
                "id": "chair",
                "label": "CHAIR hallucination measurement",
                "description": (
                    "Every generated caption is mined for COCO object mentions and "
                    "matched against the image's ground-truth objects; CHAIRs is the "
                    "share of mentioned objects that are absent."
                ),
                "status": "complete",
                "metric": (
                    f"CHAIRs {by_id['global']['chair_s']:.2f}% to "
                    f"{by_id['object']['chair_s']:.2f}%"
                    + (f" · {reduction:.2f} pp reduction" if reduction is not None else "")
                ),
                "evidence": evidence(
                    ("hallucination logic", "src/hallucination.py"),
                    ("global metrics", "outputs/metrics/global_test.json"),
                    ("object metrics", "outputs/metrics/object_test.json"),
                    ("correlation analysis", "outputs/metrics/correlation.json"),
                ),
            },
            {
                "id": "attention_viz",
                "label": "Attention-map visualisation",
                "description": (
                    "Per-word 7x7 attention grids are normalised robustly and blended "
                    "over the original image so the decoder's focus can be inspected."
                ),
                "status": "complete",
                "metric": f"{len(examples)} attention examples · {overlay_total} word overlays",
                "evidence": evidence(
                    ("visualise", "scripts/06_visualize.py"),
                    ("attention grid figure", "report/figures/attention_maps.png"),
                    ("attention predictions", "outputs/predictions/attention_test.json"),
                    ("rendered overlays", figures.parent / "attention"),
                ),
            },
            {
                "id": "analysis",
                "label": "BLEU ↔ hallucination analysis and final report",
                "description": (
                    "Caption-level Spearman correlations, BLEU-4 quartiles, an "
                    "image-cluster bootstrap and a matched-BLEU comparison test whether "
                    "higher overlap implies fewer hallucinations; the findings are "
                    "written up in the MAD-format report."
                ),
                "status": "complete",
                "metric": (
                    (f"pooled rho = {pooled['rho']:.3f}" if pooled.get("rho") is not None else "pooled rho = n/a")
                    + (f" · {gap:.1f} pp matched-BLEU spread" if gap is not None else "")
                    + f" · {pages}"
                ),
                "evidence": evidence(
                    ("matched-BLEU analysis", "outputs/metrics/bleu_matched_analysis.json"),
                    ("cluster bootstrap", "outputs/metrics/pooled_cluster_bootstrap.json"),
                    ("scatter figure", "report/figures/bleu_vs_hallucination.png"),
                    ("make report", "scripts/07_make_report.py"),
                    ("report DOCX", downloads / "MAD_Report_Image_Captioning.docx"),
                    ("report PDF", downloads / "MAD_Report_Image_Captioning.pdf"),
                ),
            },
        ]

    def _split_captions(self) -> dict[str, int]:
        return {
            name: sum(len(self.captions.get(str(image_id), [])) for image_id in self._split_ids(name))
            for name in ("train", "val", "test")
        }

    def build_pipeline(self, report_pages: int | None) -> list[dict[str, Any]]:
        images = as_int(self.stats.get("n_images")) or 0
        training = " / ".join(
            f"{(num(sum(float(row.get('seconds') or 0.0) for row in self.histories[model]) / 60.0, 1)):.1f}"
            for model in MODELS
        )
        feature_seconds = log_seconds(
            self.paths["feature_log"], r"Saved features to .* in ([0-9.]+)s"
        )
        detector_seconds = None
        if isinstance(self.detections_meta, dict):
            raw_seconds = self.detections_meta.get("seconds")
            if isinstance(raw_seconds, (int, float)):
                detector_seconds = float(raw_seconds)
        if feature_seconds is None:
            self.warnings.append("feature extraction log missing: runtime omitted")
        if detector_seconds is None:
            self.warnings.append("detections_meta.json has no timing: detector runtime omitted")
        report_runtime = f"{report_pages} pages" if report_pages else None
        if report_runtime is None:
            self.warnings.append("pdfinfo unavailable or report PDF missing: report page count omitted")

        return [
            {
                "id": "data",
                "label": "Dataset",
                "detail": (
                    "Download COCO val2017 images and annotations, re-split "
                    f"{as_int(self.stats.get('splits', {}).get('train')):,}/"
                    f"{as_int(self.stats.get('splits', {}).get('val')):,}/"
                    f"{as_int(self.stats.get('splits', {}).get('test')):,}, build the "
                    "training-only vocabulary and extract ground-truth objects."
                ),
                "status": "complete",
                "runtime": f"{images:,} images",
                "file": rel(ROOT / "scripts/01_prepare_data.py", self.base_dir),
            },
            {
                "id": "features",
                "label": "CNN features",
                "detail": (
                    "ResNet-18 (ImageNet) layer4 activations cached once and reused by "
                    "all three captioners: a pooled vector and a 7x7 spatial grid."
                ),
                "status": "complete",
                "runtime": (
                    f"{feature_seconds:.1f} s" if feature_seconds is not None else None
                ),
                "file": rel(ROOT / "scripts/02_extract_features.py", self.base_dir),
            },
            {
                "id": "detector",
                "label": "Object detector",
                "detail": (
                    "fasterrcnn_mobilenet_v3_large_320_fpn at score threshold 0.3, at "
                    "most 12 boxes per image, feeding the object-aware variant."
                ),
                "status": "complete",
                "runtime": (
                    f"{detector_seconds / 60.0:.1f} min" if detector_seconds is not None else None
                ),
                "file": rel(ROOT / "scripts/03_detect_objects.py", self.base_dir),
            },
            {
                "id": "training",
                "label": "Training",
                "detail": (
                    "AdamW with OneCycle over 12 epochs and 8,000 sampled caption pairs "
                    "per epoch on CPU; the best validation-loss checkpoint is kept."
                ),
                "status": "complete",
                "runtime": f"{training} min",
                "file": rel(ROOT / "scripts/04_train.py", self.base_dir),
            },
            {
                "id": "evaluation",
                "label": "Evaluation",
                "detail": (
                    "Greedy decoding of 500 test images and corpus-level BLEU-1-4, "
                    "CIDEr-D, CHAIR and object precision/recall/F1."
                ),
                "status": "complete",
                "runtime": evaluation_runtime(ROOT / "logs/evaluate.log"),
                "file": rel(ROOT / "scripts/05_evaluate.py", self.base_dir),
            },
            {
                "id": "report",
                "label": "Figures and report",
                "detail": (
                    "Report figures from scripts/06_visualize.py, then the MAD-format "
                    "DOCX/PDF report from scripts/07_make_report.py."
                ),
                "status": "complete",
                "runtime": report_runtime,
                "file": rel(ROOT / "scripts/07_make_report.py", self.base_dir),
            },
        ]

    # -- assembly ---------------------------------------------------------

    def build(self) -> dict[str, Any]:
        self.load()
        if not self.available_ids():
            raise BuildError("no test images are present in all three per-image metric files")

        models = self.build_models()
        examples_attention = self.build_attention_examples()
        gallery = self.build_gallery()
        hallucination = self.build_hallucination()
        report_pages = pdf_pages(ROOT / "report/MAD_Report_Image_Captioning.pdf")
        figures = self.build_figures()
        artifacts = self.build_artifacts()
        # Every panel below is copied from a saved artifact, so it is identical
        # under --limit; only the per-image selections above depend on the scope.
        matched = self.build_matched_bleu()
        pairwise = self.build_pairwise()
        correlations = self.build_correlations()
        quartiles = self.build_quartiles()
        training = self.build_training()

        payload = {
            "meta": self.build_meta(),
            "headline": self.build_headline(models, matched, correlations),
            "tasks": self.build_tasks(
                models, examples_attention, report_pages, correlations, matched
            ),
            "pipeline": self.build_pipeline(report_pages),
            "models": models,
            "pairwise": pairwise,
            "correlations": correlations,
            "quartiles": quartiles,
            "matched_bleu": matched,
            "training": training,
            "dataset": self.build_dataset(),
            "hallucination": hallucination,
            "attention_examples": examples_attention,
            "gallery": gallery,
            "figures": figures,
            "artifacts": artifacts,
        }
        return payload


# ---------------------------------------------------------------------------
# Writing and validation
# ---------------------------------------------------------------------------


def dump_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2)


def dump_js(payload: Any) -> str:
    body = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    return f"window.DASHBOARD_DATA = {body};\n"


def _reject_constant(token: str) -> None:
    raise ValueError(f"non-finite JSON constant: {token}")


def verify_payload_roundtrip(text: str, source: str) -> Any:
    """Prove the written text parses back as strict JSON.

    ``json.dumps(..., allow_nan=False)`` already guarantees that no NaN or
    Infinity literal is emitted; ``parse_constant`` makes the reload fail loudly
    if that ever regresses.  Scanning the raw text is deliberately avoided so a
    caption that happens to contain "NaN" is not mistaken for a bad number.
    """
    try:
        return json.loads(text, parse_constant=_reject_constant)
    except (json.JSONDecodeError, ValueError) as exc:
        raise BuildError(f"{source} is not strict JSON: {exc}") from exc


def write_outputs(payload: dict[str, Any], out_dir: Path) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_text = dump_json(payload) + "\n"
    json_path = out_dir / "dashboard.json"
    json_path.write_text(json_text, encoding="utf-8")
    verify_payload_roundtrip(json_text, json_path.name)

    js_text = dump_js(payload)
    js_path = out_dir / "dashboard-data.js"
    js_path.write_text(js_text, encoding="utf-8")
    if not js_text.startswith("window.DASHBOARD_DATA = ") or not js_text.endswith(";\n"):
        raise BuildError("internal error: malformed JS payload")

    return {"json": json_path.stat().st_size, "js": js_path.stat().st_size}


def tree_size(path: Path) -> tuple[int, int]:
    files = [item for item in path.rglob("*") if item.is_file()]
    return len(files), sum(item.stat().st_size for item in files)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build dashboard/data/dashboard.json, dashboard/data/dashboard-data.js and "
            "the dashboard asset copies from the saved project artifacts."
        )
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "default.yaml"),
        help="config file used only to read the seed (default: configs/default.yaml)",
    )
    parser.add_argument(
        "-o",
        "--out",
        default=None,
        help="payload directory (default: dashboard/data; with --limit: dashboard/data/smoke)",
    )
    parser.add_argument(
        "--assets-dir",
        "--assets",
        dest="assets_dir",
        default=None,
        help="asset directory (default: dashboard/assets; with --limit: dashboard/assets/smoke)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help=(
            "smoke-test mode: consider only the first N test images. With the default "
            "paths this writes to dashboard/data/smoke and dashboard/assets/smoke so the "
            "full payload is never overwritten."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        print("build_dashboard: error: --limit must be a positive integer", file=sys.stderr)
        return 2

    out_given = args.out is not None
    assets_given = args.assets_dir is not None
    out_dir = Path(args.out) if out_given else ROOT / "dashboard" / "data"
    assets_dir = Path(args.assets_dir) if assets_given else ROOT / "dashboard" / "assets"
    if args.limit is not None:
        if not out_given:
            out_dir = out_dir / "smoke"
        if not assets_given:
            assets_dir = assets_dir / "smoke"
    # Resolve before deriving relative payload paths: a relative --out such as
    # ``build/data`` would otherwise produce paths relative to the wrong base.
    out_dir = out_dir.expanduser().resolve()
    assets_dir = assets_dir.expanduser().resolve()

    builder = Builder(out_dir=out_dir, assets_dir=assets_dir, limit=args.limit, config=Path(args.config))
    try:
        payload = builder.build()
        sizes = write_outputs(payload, out_dir)
    except BuildError as exc:
        print(f"build_dashboard: error: {exc}", file=sys.stderr)
        return 2

    for warning in builder.warnings:
        print(f"build_dashboard: warning: {warning}", file=sys.stderr)

    image_count, image_bytes = tree_size(assets_dir / "images")
    overlay_count, overlay_bytes = tree_size(assets_dir / "attention")
    figure_count, figure_bytes = tree_size(assets_dir / "figures")
    download_count, download_bytes = tree_size(assets_dir / "downloads")
    total_bytes = sizes["json"] + sizes["js"] + image_bytes + overlay_bytes + figure_bytes + download_bytes

    scope = f"first {args.limit} test images" if args.limit is not None else "full test split"
    print(f"Dashboard payload  -> {out_dir / 'dashboard.json'} ({sizes['json'] / 1024:.1f} KB)")
    print(f"JS payload         -> {out_dir / 'dashboard-data.js'} ({sizes['js'] / 1024:.1f} KB)")
    print(
        "Assets             -> "
        f"{image_count} images, {overlay_count} attention overlays, "
        f"{figure_count} figures, {download_count} downloads ({total_bytes / 1e6:.2f} MB total)"
    )
    print(
        "Coverage           -> "
        f"{len(payload['models'])} models, {len(payload['tasks'])} tasks, "
        f"{len(payload['pipeline'])} pipeline steps, {len(payload['pairwise'])} pairwise, "
        f"{len(payload['matched_bleu'])} matched-BLEU rows, "
        f"{len(payload['hallucination']['examples'])} hallucination examples, "
        f"{len(payload['attention_examples'])} attention examples, "
        f"{len(payload['gallery'])} gallery items"
    )
    print(f"Scope              -> {scope}; generated_at {payload['meta']['generated_at']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

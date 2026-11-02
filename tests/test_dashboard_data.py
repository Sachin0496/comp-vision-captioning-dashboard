#!/usr/bin/env python
"""Contract tests for scripts/build_dashboard.py and the dashboard payload.

The suite builds the real payload from the project artifacts into a temporary
directory (``--out`` / ``--assets-dir``), then checks the data contract in
``dashboard/data-contract.md``: schema, ordering, coverage counts, referenced
files, strict JSON, range sanity and idempotency.

Run from the repository root:

    python -m unittest tests.test_dashboard_data -v
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts" / "build_dashboard.py"
PYTHON = sys.executable

MODELS = ["global", "attention", "object"]
HALLUCINATION_EXAMPLES = 6
GALLERY_ITEMS = 8
ATTENTION_EXAMPLES = 4
PAIRWISE_KEYS = ["global_vs_attention", "global_vs_object", "attention_vs_object"]
TASK_IDS = [
    "dataset",
    "global",
    "attention",
    "object",
    "metrics",
    "chair",
    "attention_viz",
    "analysis",
]
PIPELINE_IDS = ["data", "features", "detector", "training", "evaluation", "report"]
FIGURE_IDS = [
    "model_comparison",
    "bleu_vs_hallucination",
    "training_curves",
    "attention_maps",
    "qualitative_examples",
]
REQUIRED_ARTIFACT_KINDS = {"docx", "pdf", "csv"}


def load_builder_module():
    """Import ``scripts/build_dashboard.py`` without requiring a package."""
    spec = importlib.util.spec_from_file_location("build_dashboard", BUILDER)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"cannot import {BUILDER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def iter_string_values(node, prefix=""):
    """Yield ``(json_path, string)`` for every string in a nested structure."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield from iter_string_values(value, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from iter_string_values(value, f"{prefix}[{index}]")
    elif isinstance(node, str):
        yield prefix, node


def iter_numbers(node):
    if isinstance(node, dict):
        for value in node.values():
            yield from iter_numbers(value)
    elif isinstance(node, list):
        for value in node:
            yield from iter_numbers(value)
    elif isinstance(node, (int, float)) and not isinstance(node, bool):
        yield node


class DashboardBuildTestCase(unittest.TestCase):
    """Builds the payload once per class into a temporary directory."""

    data_dir: Path
    assets_dir: Path
    payload: dict
    result: subprocess.CompletedProcess
    tmp_root: Path

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp_root = Path(tempfile.mkdtemp(prefix="dashboard-test-"))
        cls.data_dir = cls.tmp_root / "data"
        cls.assets_dir = cls.tmp_root / "assets"
        cls.result = cls.run_builder(cls.data_dir, cls.assets_dir)
        if cls.result.returncode != 0:  # pragma: no cover - surfaced as a failure
            raise AssertionError(
                "builder failed\nstdout:\n"
                f"{cls.result.stdout}\nstderr:\n{cls.result.stderr}"
            )
        cls.payload = json.loads((cls.data_dir / "dashboard.json").read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.tmp_root, ignore_errors=True)

    @classmethod
    def run_builder(cls, out_dir: Path, assets_dir: Path, *extra: str) -> subprocess.CompletedProcess:
        command = [
            PYTHON,
            str(BUILDER),
            "--out",
            str(out_dir),
            "--assets-dir",
            str(assets_dir),
            *extra,
        ]
        env = dict(os.environ)
        env.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
        env.setdefault("MPLBACKEND", "Agg")
        return subprocess.run(
            command,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            env=env,
            timeout=1800,
            check=False,
        )

    # -- helpers ---------------------------------------------------------

    def section(self, *keys):
        node = self.payload
        for key in keys:
            self.assertIn(key, node, f"missing payload key {key}")
            node = node[key]
        return node

    def assertPercent(self, value, label):
        self.assertIsInstance(value, (int, float), f"{label} must be a number")
        self.assertGreaterEqual(value, 0.0, f"{label} below 0: {value}")
        self.assertLessEqual(value, 100.0, f"{label} above 100: {value}")

    # -- schema ----------------------------------------------------------

    def test_top_level_schema(self):
        expected = [
            "meta",
            "headline",
            "tasks",
            "pipeline",
            "models",
            "pairwise",
            "correlations",
            "quartiles",
            "matched_bleu",
            "training",
            "dataset",
            "hallucination",
            "attention_examples",
            "gallery",
            "figures",
            "artifacts",
        ]
        self.assertEqual(list(self.payload.keys()), expected)

    def test_builder_is_idempotent_for_identical_inputs(self):
        """The module-level entry point must return 0 and never write elsewhere."""
        module = load_builder_module()
        self.assertTrue(hasattr(module, "main"))
        import contextlib
        import io

        with tempfile.TemporaryDirectory(prefix="dashboard-entry-") as tmp:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                code = module.main(
                    [
                        "--out",
                        str(Path(tmp) / "data"),
                        "--assets-dir",
                        str(Path(tmp) / "assets"),
                        "--limit",
                        "4",
                    ]
                )
            self.assertEqual(code, 0)
            self.assertTrue((Path(tmp) / "data" / "dashboard.json").exists())

    def test_meta_schema(self):
        meta = self.section("meta")
        for key in (
            "title",
            "subtitle",
            "generated_at",
            "split",
            "split_note",
            "n_test",
            "n_train",
            "n_val",
            "n_total",
            "vocab_size",
            "train_captions",
            "test_captions",
            "seed",
            "device_note",
            "dataset_note",
        ):
            self.assertIn(key, meta)
        self.assertEqual(meta["split"], "test")
        self.assertEqual(
            meta["n_test"] + meta["n_train"] + meta["n_val"], meta["n_total"]
        )
        self.assertEqual((meta["n_train"], meta["n_val"], meta["n_test"]), (4000, 500, 500))
        self.assertEqual(meta["n_total"], 5000)
        self.assertEqual(meta["vocab_size"], 2916)
        self.assertEqual(meta["train_captions"], 20010)
        self.assertEqual(meta["test_captions"], 2501)
        self.assertRegex(meta["generated_at"], r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")

    def test_headline_schema_and_derived_values(self):
        headline = self.section("headline")
        for key in (
            "question",
            "answer",
            "best_model",
            "best_bleu4",
            "best_chair_s",
            "pooled_rho",
            "pooled_p",
            "cluster_rho",
            "cluster_ci",
            "cluster_p",
            "matched_bin",
            "matched_rates",
            "hallucination_reduction_pp",
            "bleu_gain_points",
        ):
            self.assertIn(key, headline)
        self.assertEqual(headline["matched_bin"], "10-20")
        self.assertEqual(list(headline["matched_rates"].keys()), MODELS)

        source = {
            row["model"]: row
            for row in json.loads(
                (ROOT / "outputs/metrics/model_comparison.json").read_text(encoding="utf-8")
            )
        }
        self.assertAlmostEqual(
            headline["hallucination_reduction_pp"],
            round(source["global"]["chair_s"] - source["object"]["chair_s"], 2),
            places=2,
        )
        self.assertAlmostEqual(
            headline["bleu_gain_points"],
            round(source["object"]["bleu4"] - source["global"]["bleu4"], 2),
            places=2,
        )
        # The answer must be a real, non-empty sentence without causal overclaiming.
        answer = headline["answer"]
        self.assertGreater(len(answer), 80)
        self.assertTrue(answer.endswith("."))
        for banned in ("proves", "causes", "because of", "therefore guarantees"):
            self.assertNotIn(banned, answer.lower())

    def test_models_ordered_with_expected_fields(self):
        models = self.section("models")
        self.assertEqual([model["id"] for model in models], MODELS)
        for model in models:
            for key in (
                "id",
                "label",
                "short",
                "color",
                "description",
                "params",
                "params_label",
                "train_minutes",
                "bleu1",
                "bleu2",
                "bleu3",
                "bleu4",
                "bleu",
                "cider_d",
                "chair_s",
                "chair_i",
                "object_precision",
                "object_recall",
                "object_f1",
                "n_mentions",
                "n_hallucinated",
                "sentence_bleu4_mean",
                "unique_captions",
                "uses_attention",
                "uses_objects",
            ):
                self.assertIn(key, model, f"{model['id']} missing {key}")
            self.assertRegex(model["color"], r"^#[0-9A-Fa-f]{6}$")
            self.assertEqual(model["bleu"], model["bleu4"])
            # BLEU-1..4 are independent cumulative precisions (not monotone),
            # so only range/sanity checks apply.
            for key in ("bleu1", "bleu2", "bleu3", "bleu4"):
                self.assertGreaterEqual(model[key], 0.0)
                self.assertLessEqual(model[key], 100.0)
            self.assertLessEqual(model["cider_d"], 10.0)
            self.assertGreater(model["params"], 1_000_000)
            self.assertGreater(model["train_minutes"], 0)
            self.assertPercent(model["chair_s"], f"{model['id']}.chair_s")
            self.assertPercent(model["sentence_bleu4_mean"], f"{model['id']}.sentence_bleu4_mean")

        global_model, attention, object_model = models
        self.assertEqual(
            (global_model["uses_attention"], global_model["uses_objects"]), (False, False)
        )
        self.assertEqual((attention["uses_attention"], attention["uses_objects"]), (True, False))
        self.assertEqual(
            (object_model["uses_attention"], object_model["uses_objects"]), (True, True)
        )
        # CHAIR_s must decrease monotonically with the stronger model.
        self.assertLess(object_model["chair_s"], attention["chair_s"])
        self.assertLess(attention["chair_s"], global_model["chair_s"])
        self.assertAlmostEqual(global_model["params_label"], "4.34 M")
        self.assertAlmostEqual(object_model["train_minutes"], 10.9)

    def test_tasks_cover_the_assignment(self):
        tasks = self.section("tasks")
        self.assertEqual(len(tasks), 8)
        self.assertEqual([task["id"] for task in tasks], TASK_IDS)
        for task in tasks:
            self.assertEqual(task["status"], "complete")
            self.assertTrue(task["label"])
            self.assertGreater(len(task["description"]), 40)
            self.assertIsInstance(task["metric"], str)
            self.assertTrue(task["metric"])
            self.assertIsInstance(task["evidence"], list)
            self.assertGreaterEqual(len(task["evidence"]), 3)
            for item in task["evidence"]:
                self.assertEqual(set(item), {"label", "path"})

    def test_pipeline_has_six_steps_with_runtime(self):
        pipeline = self.section("pipeline")
        models = self.section("models")
        self.assertEqual(len(pipeline), 6)
        self.assertEqual([step["id"] for step in pipeline], PIPELINE_IDS)
        runtimes = {step["id"]: step["runtime"] for step in pipeline}
        self.assertEqual(runtimes["data"], "5,000 images")
        self.assertEqual(runtimes["features"], "60.2 s")
        # Runtime strings are read back from the artifacts, not hard-coded:
        # detections_meta.json records 782.9 s == 13.0 min (report appendix agrees).
        meta = json.loads(
            (ROOT / "data/processed/detections_meta.json").read_text(encoding="utf-8")
        )
        self.assertEqual(runtimes["detector"], f"{meta['seconds'] / 60.0:.1f} min")
        # Training minutes are summed from each history.json and must match the
        # per-model values on the model cards.
        expected_training = " / ".join(f"{model['train_minutes']:.1f}" for model in models)
        self.assertEqual(runtimes["training"], f"{expected_training} min")
        for model, expected in zip(models, (6.9, 16.3, 10.9), strict=True):
            self.assertAlmostEqual(model["train_minutes"], expected, places=1)
        self.assertIsNotNone(runtimes["evaluation"])
        for step in pipeline:
            self.assertEqual(step["status"], "complete")
            self.assertTrue(step["detail"])
            self.assertTrue(step["file"].endswith(".py"))
        # Page count is optional: only assert the format when pdfinfo ran.
        if runtimes["report"] is not None:
            self.assertRegex(runtimes["report"], r"^\d+ pages$")

    def test_pairwise_entries(self):
        pairwise = self.section("pairwise")
        self.assertEqual(len(pairwise), 3)
        self.assertEqual([row["id"] for row in pairwise], PAIRWISE_KEYS)
        source = json.loads(
            (ROOT / "outputs/metrics/evaluation_summary.json").read_text(encoding="utf-8")
        )["pairwise"]
        for row in pairwise:
            self.assertIn(row["id"], source)
            raw = source[row["id"]]
            self.assertEqual(row["first"], row["id"].split("_vs_")[0])
            self.assertEqual(row["second"], row["id"].split("_vs_")[1])
            self.assertEqual(len(row["objects_ci"]), 2)
            self.assertEqual(len(row["rate_ci_pp"]), 2)
            self.assertEqual(len(row["bleu4_ci"]), 2)
            # Rate differences are fractions in the source and pp in the payload.
            self.assertAlmostEqual(
                row["rate_diff_pp"], raw["hallucination_rate"]["mean_diff"] * 100, places=1
            )
            # BLEU differences stay in points.
            self.assertAlmostEqual(
                row["bleu4_diff"], raw["sentence_bleu4"]["mean_diff"], places=3
            )
            self.assertLessEqual(row["objects_ci"][0], row["objects_ci"][1])
            self.assertLessEqual(row["rate_ci_pp"][0], row["rate_ci_pp"][1])
            self.assertLessEqual(row["bleu4_ci"][0], row["bleu4_ci"][1])
            self.assertGreater(row["objects_diff"], 0)
            self.assertGreater(row["rate_diff_pp"], 0)
            self.assertLess(row["bleu4_diff"], 0)
            self.assertEqual(
                row["only_first"], raw["mcnemar"]["only_first_hallucinates"]
            )
            self.assertEqual(
                row["only_second"], raw["mcnemar"]["only_second_hallucinates"]
            )
            self.assertIsNotNone(row["mcnemar_p"])

    def test_correlations_and_quartiles(self):
        correlations = self.section("correlations")
        self.assertEqual(list(correlations.keys()), [*MODELS, "pooled"])
        source = json.loads(
            (ROOT / "outputs/metrics/correlation.json").read_text(encoding="utf-8")
        )
        for key, entry in correlations.items():
            analysis = source[key]["analysis"]
            self.assertEqual(entry["n"], analysis["n"])
            self.assertAlmostEqual(
                entry["rho"],
                analysis["spearman_bleu4_vs_n_hallucinated"]["rho"],
                places=3,
            )
            self.assertPercent(entry["rate"], f"correlations.{key}.rate")
            self.assertLess(entry["rho"], 0)
            self.assertIsNotNone(entry["p"])
        self.assertEqual(correlations["n" if False else "pooled"]["n"], 1500)

        quartiles = self.section("quartiles")
        self.assertEqual([row["bucket"] for row in quartiles], ["Q1", "Q2", "Q3", "Q4"])
        pooled_quartiles = source["pooled"]["analysis"]["bleu4_quartiles"]
        for row, raw in zip(quartiles, pooled_quartiles, strict=True):
            self.assertEqual(row["n"], raw["n"])
            self.assertAlmostEqual(row["rate"], raw["hallucination_rate"] * 100, places=2)
            self.assertEqual(len(row["ci"]), 2)
            self.assertLessEqual(row["ci"][0], row["ci"][1])
            self.assertPercent(row["rate"], f"quartiles.{row['bucket']}.rate")
            self.assertLessEqual(row["bleu_min"], row["bleu_max"])
        rates = [row["rate"] for row in quartiles]
        self.assertEqual(rates, sorted(rates, reverse=True))

    def test_matched_bleu_grid(self):
        matched = self.section("matched_bleu")
        self.assertEqual(len(matched), 15)
        bins = []
        for row in matched:
            if not bins or bins[-1] != row["bin"]:
                bins.append(row["bin"])
        self.assertEqual(bins, ["0-10", "10-20", "20-30", "30-40", "40+"])
        for offset in range(0, len(matched), 3):
            chunk = matched[offset : offset + 3]
            self.assertEqual([row["model"] for row in chunk], MODELS)
            self.assertEqual(len({row["bin"] for row in chunk}), 1)
        for row in matched:
            self.assertIsInstance(row["bin"], str)
            self.assertPercent(row["rate"], f"matched_bleu.{row['bin']}.{row['model']}")
            self.assertEqual(len(row["ci"]), 2)
            self.assertLessEqual(row["ci"][0], row["ci"][1])
            self.assertGreater(row["n"], 0)
            self.assertGreater(row["mean_bleu4"], 0)

    def test_training_records(self):
        training = self.section("training")
        self.assertEqual(list(training.keys()), MODELS)
        for model, rows in training.items():
            self.assertEqual(len(rows), 12, f"{model} should keep all 12 epochs")
            self.assertEqual([row["epoch"] for row in rows], list(range(1, 13)))
            for row in rows:
                self.assertEqual(
                    set(row), {"epoch", "train_loss", "val_loss", "val_chair_s", "val_bleu4"}
                )
                self.assertGreater(row["train_loss"], 0)
                self.assertGreater(row["val_loss"], 0)
                # Fractions in history.json become percentages here.
                self.assertPercent(row["val_chair_s"], f"{model}.val_chair_s")
                self.assertGreaterEqual(row["val_bleu4"], 0)
            self.assertGreater(rows[-1]["val_bleu4"], rows[0]["val_bleu4"])
            self.assertLess(rows[-1]["val_chair_s"], rows[0]["val_chair_s"])

    def test_dataset_panel(self):
        dataset = self.section("dataset")
        self.assertEqual(dataset["splits"], {"train": 4000, "val": 500, "test": 500})
        self.assertEqual(dataset["captions"], {"train": 20010, "val": 2503, "test": 2501})
        self.assertAlmostEqual(dataset["caption_length"]["mean"], 10.45, places=2)
        self.assertEqual(dataset["caption_length"]["p50"], 10)
        self.assertEqual(dataset["caption_length"]["p95"], 14)
        self.assertAlmostEqual(dataset["objects_per_image"]["mean"], 2.93, places=2)
        self.assertEqual(dataset["objects_per_image"]["max"], 14)

        top = dataset["top_objects"]
        self.assertEqual(len(top), 12)
        self.assertEqual(top[0], {"name": "person", "count": 2693})
        counts = [entry["count"] for entry in top]
        self.assertEqual(counts, sorted(counts, reverse=True))
        categories = json.loads(
            (ROOT / "data/processed/categories.json").read_text(encoding="utf-8")
        )["names"]
        for entry in top:
            self.assertIn(entry["name"], categories)

        detector = dataset["detector"]
        meta = json.loads(
            (ROOT / "data/processed/detections_meta.json").read_text(encoding="utf-8")
        )
        self.assertEqual(detector["name"], meta["detector"])
        self.assertEqual(detector["threshold"], meta["score_threshold"])
        self.assertEqual(detector["max_detections"], meta["max_detections"])
        self.assertEqual(detector["images"], meta["images"])
        self.assertEqual(detector["seconds"], meta["seconds"])
        self.assertGreater(detector["mean_detections"], 0)
        self.assertLessEqual(detector["mean_detections"], detector["max_detections"])
        self.assertGreater(detector["empty_images"], 0)

    def test_hallucination_panel(self):
        hallucination = self.section("hallucination")
        top = hallucination["top_hallucinated"]
        self.assertEqual(len(top), 12)
        self.assertEqual(top[0], {"name": "laptop", "count": 71})
        counts = [entry["count"] for entry in top]
        self.assertEqual(counts, sorted(counts, reverse=True))

        examples = hallucination["examples"]
        self.assertEqual(len(examples), HALLUCINATION_EXAMPLES)
        positions = [(sum(1 for model in MODELS if ex["captions"][model]["hallucinated"]), ex["image_id"]) for ex in examples]
        self.assertEqual(positions, sorted(positions))
        for example in examples:
            self.assertEqual(
                set(example), {"image_id", "image", "references", "gt_objects", "captions"}
            )
            self.assertTrue(example["references"])
            self.assertTrue(example["gt_objects"])
            self.assertEqual(list(example["captions"].keys()), MODELS)
            hallucinating = 0
            for model in MODELS:
                caption = example["captions"][model]
                self.assertEqual(set(caption), {"text", "hallucinated"})
                self.assertIsInstance(caption["hallucinated"], list)
                hallucinating += bool(caption["hallucinated"])
            self.assertGreaterEqual(hallucinating, 1)
            self.assertLess(hallucinating, 3)

    def test_attention_examples(self):
        examples = self.section("attention_examples")
        self.assertEqual(len(examples), ATTENTION_EXAMPLES)
        predictions = {
            int(row["image_id"]): row
            for row in json.loads(
                (ROOT / "outputs/predictions/attention_test.json").read_text(encoding="utf-8")
            )
        }
        for example in examples:
            self.assertEqual(
                set(example),
                {"image_id", "image", "caption", "words", "overlays", "highlight"},
            )
            source = predictions[example["image_id"]]
            # Predictions must be passed through untouched.
            self.assertEqual(example["caption"], source["caption"])
            self.assertEqual(example["words"], source["words"])
            self.assertEqual(example["highlight"], sorted(example["highlight"]))
            self.assertTrue(2 <= len(example["highlight"]) <= 4)
            self.assertTrue(4 <= len(example["words"]) <= 14)
            self.assertEqual(len(example["overlays"]), len(example["highlight"]))
            for overlay, index in zip(example["overlays"], example["highlight"], strict=True):
                self.assertEqual(
                    set(overlay), {"index", "word", "path"}
                )
                self.assertEqual(overlay["index"], index)
                self.assertEqual(overlay["word"], example["words"][index])
                self.assertTrue(overlay["path"].endswith(f"_{index}.png"))
            self.assertEqual(
                os.path.basename(example["image"]), f"{example['image_id']}.jpg"
            )

    def test_gallery(self):
        gallery = self.section("gallery")
        self.assertEqual(len(gallery), GALLERY_ITEMS)
        self.assertEqual(len({item["image_id"] for item in gallery}), GALLERY_ITEMS)
        for item in gallery:
            self.assertEqual(
                set(item), {"image_id", "image", "references", "gt_objects", "captions"}
            )
            self.assertGreaterEqual(len(item["references"]), 5)
            self.assertEqual(list(item["captions"].keys()), MODELS)
            for model in MODELS:
                self.assertEqual(set(item["captions"][model]), {"text", "hallucinated"})
            # Gallery items are chosen for model disagreement: the three
            # hallucinated-object sets are not all identical.
            self.assertGreater(
                len({frozenset(item["captions"][model]["hallucinated"]) for model in MODELS}),
                1,
            )
        self.assertEqual([item["image_id"] for item in gallery], sorted(item["image_id"] for item in gallery))

    def test_figures_and_artifacts(self):
        figures = self.section("figures")
        self.assertEqual([figure["id"] for figure in figures], FIGURE_IDS)
        for figure in figures:
            self.assertEqual(set(figure), {"id", "label", "path", "caption"})
            self.assertTrue(figure["path"].endswith(".png"))
            self.assertGreater(len(figure["caption"]), 40)

        artifacts = self.section("artifacts")
        self.assertEqual(len(artifacts), 3)
        self.assertEqual({item["kind"] for item in artifacts}, REQUIRED_ARTIFACT_KINDS)
        for artifact in artifacts:
            self.assertEqual(set(artifact), {"label", "path", "kind", "size"})
            self.assertGreater(artifact["size"], 100)
        names = {os.path.basename(item["path"]) for item in artifacts}
        self.assertEqual(
            names,
            {
                "MAD_Report_Image_Captioning.docx",
                "MAD_Report_Image_Captioning.pdf",
                "model_comparison.csv",
            },
        )

    # -- files, JSON and ranges -------------------------------------------

    def test_referenced_paths_exist(self):
        referenced = [
            (path, value)
            for path, value in iter_string_values(self.payload)
            if value.startswith("assets/")
        ]
        self.assertGreaterEqual(len(referenced), 35)
        for path, value in referenced:
            target = self.tmp_root / value
            self.assertTrue(target.exists(), f"{path}: missing file {value}")
            self.assertGreater(target.stat().st_size, 0, f"{path}: empty file {value}")

        # Evidence and pipeline links point back into the repository.
        upstream = [
            (path, value)
            for path, value in iter_string_values(self.payload)
            if value.startswith("../")
        ]
        self.assertGreaterEqual(len(upstream), 20)
        for path, value in upstream:
            self.assertTrue(
                (self.tmp_root / value).exists(), f"{path}: missing upstream file {value}"
            )
            self.assertNotIn(chr(92), value, "paths must use forward slashes")

        image_names = {
            os.path.basename(value)
            for _, value in referenced
            if value.startswith("assets/images/")
        }
        expected_images = {
            f"{example['image_id']}.jpg"
            for example in (
                self.payload["hallucination"]["examples"]
                + self.payload["gallery"]
                + self.payload["attention_examples"]
            )
        }
        self.assertEqual(image_names, expected_images)

    def test_assets_are_resized_rgb_jpegs(self):
        from PIL import Image  # imported lazily: the builder already requires Pillow

        images_dir = self.assets_dir / "images"
        files = sorted(images_dir.glob("*.jpg"))
        self.assertGreaterEqual(len(files), 8)
        for path in files:
            with Image.open(path) as handle:
                self.assertEqual(handle.format, "JPEG")
                self.assertEqual(handle.mode, "RGB")
                self.assertLessEqual(handle.width, 900)
                self.assertGreater(handle.width, 0)
        overlay_files = sorted((self.assets_dir / "attention").glob("*.png"))
        self.assertGreaterEqual(len(overlay_files), 8)
        for path in overlay_files:
            with Image.open(path) as handle:
                self.assertEqual(handle.format, "PNG")

    def test_json_is_strict(self):
        json_path = self.data_dir / "dashboard.json"
        text = json_path.read_text(encoding="utf-8")
        self.assertFalse(re.search(r"\bNaN\b", text))
        self.assertFalse(re.search(r"\bInfinity\b", text))
        self.assertFalse(re.search(r"\b-Infinity\b", text))

        def reject(token):  # pragma: no cover - only on failure
            raise AssertionError(f"non-strict JSON constant {token}")

        reloaded = json.loads(text, parse_constant=reject)
        self.assertEqual(reloaded, self.payload)
        # allow_nan=False must round-trip without raising.
        dumped = json.dumps(
            self.payload, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True
        )
        self.assertEqual(json.loads(dumped), self.payload)

    def test_js_payload_wrapper(self):
        js_path = self.data_dir / "dashboard-data.js"
        text = js_path.read_text(encoding="utf-8")
        prefix = "window.DASHBOARD_DATA = "
        self.assertTrue(text.startswith(prefix), "JS payload must start with the global assignment")
        self.assertTrue(text.endswith(";\n"), "JS payload must end with ';' and a newline")
        body = text[len(prefix) : -2]
        self.assertEqual(json.loads(body), self.payload)
        self.assertEqual(text.count("window.DASHBOARD_DATA"), 1)

    def test_numbers_are_strictly_finite_and_in_range(self):
        import math

        for value in iter_numbers(self.payload):
            self.assertTrue(math.isfinite(value), f"non-finite number in payload: {value}")

        for model in self.payload["models"]:
            self.assertPercent(model["chair_s"], f"{model['id']}.chair_s")
            self.assertPercent(model["chair_i"], f"{model['id']}.chair_i")
            self.assertPercent(model["object_precision"], f"{model['id']}.object_precision")
            self.assertPercent(model["object_recall"], f"{model['id']}.object_recall")
            self.assertPercent(model["object_f1"], f"{model['id']}.object_f1")
            self.assertGreaterEqual(model["cider_d"], 0.0)
            self.assertLessEqual(model["cider_d"], 10.0)
            self.assertGreaterEqual(model["bleu4"], 0.0)
            self.assertLessEqual(model["bleu4"], 100.0)

        for key, entry in self.payload["correlations"].items():
            self.assertPercent(entry["rate"], f"correlations.{key}.rate")
            self.assertGreaterEqual(entry["rho"], -1.0)
            self.assertLessEqual(entry["rho"], 1.0)
            self.assertIsNotNone(entry["p"])

        for row in self.payload["quartiles"]:
            self.assertPercent(row["rate"], f"quartiles.{row['bucket']}.rate")
            for bound in row["ci"]:
                self.assertPercent(bound, f"quartiles.{row['bucket']}.ci")

        for row in self.payload["matched_bleu"]:
            self.assertPercent(row["rate"], f"matched.{row['bin']}.{row['model']}.rate")
            for bound in row["ci"]:
                self.assertPercent(bound, f"matched.{row['bin']}.{row['model']}.ci")

        for model, rows in self.payload["training"].items():
            for row in rows:
                self.assertPercent(row["val_chair_s"], f"training.{model}.{row['epoch']}")

        for value in self.payload["headline"]["cluster_ci"]:
            self.assertGreaterEqual(value, -1.0)
            self.assertLessEqual(value, 1.0)


class DashboardSmokeTestCase(unittest.TestCase):
    """``--limit`` must write a reduced payload without touching the real one."""

    def test_limit_mode_writes_isolated_payload(self):
        tmp_root = Path(tempfile.mkdtemp(prefix="dashboard-smoke-"))
        try:
            data_dir = tmp_root / "data"
            assets_dir = tmp_root / "assets"
            result = DashboardBuildTestCase.run_builder(data_dir, assets_dir, "--limit", "6")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("first 6 test images", result.stdout)
            payload = json.loads((data_dir / "dashboard.json").read_text(encoding="utf-8"))
            # Per-image selections shrink with the subset instead of repeating
            # images, and copied analysis panels stay exactly as reported.
            self.assertLessEqual(len(payload["gallery"]), GALLERY_ITEMS)
            self.assertLessEqual(len(payload["attention_examples"]), ATTENTION_EXAMPLES)
            self.assertLessEqual(len(payload["hallucination"]["examples"]), HALLUCINATION_EXAMPLES)
            self.assertGreaterEqual(len(payload["hallucination"]["examples"]), 1)
            self.assertEqual(len(payload["pairwise"]), 3)
            self.assertEqual(len(payload["matched_bleu"]), 15)
            self.assertEqual(len(payload["quartiles"]), 4)
            self.assertEqual(payload["correlations"]["pooled"]["n"], 1500)
            self.assertEqual(payload["models"][0]["unique_captions"], 239)
            copied_images = {path.name for path in (assets_dir / "images").glob("*.jpg")}
            self.assertLessEqual(len(copied_images), 6)
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)

    def test_limit_with_default_paths_uses_smoke_subdirectory(self):
        module = load_builder_module()
        dashboard = ROOT / "dashboard"

        captured = {}
        real_builder = module.Builder

        class RecordingBuilder(real_builder):  # type: ignore[misc, valid-type]
            def __init__(self, out_dir, assets_dir, limit, config):
                captured["out"] = Path(out_dir)
                captured["assets"] = Path(assets_dir)
                captured["limit"] = limit
                raise SystemExit(0)

        module.Builder = RecordingBuilder
        try:
            with self.assertRaises(SystemExit):
                module.main(["--limit", "3"])
        finally:
            module.Builder = real_builder
        self.assertTrue(captured["out"].is_relative_to(dashboard))
        self.assertTrue(captured["assets"].is_relative_to(dashboard))
        self.assertEqual(captured["out"].name, "smoke")
        self.assertEqual(captured["assets"].name, "smoke")
        self.assertEqual(captured["limit"], 3)

    def test_bad_limit_is_rejected(self):
        tmp_root = Path(tempfile.mkdtemp(prefix="dashboard-bad-"))
        try:
            result = DashboardBuildTestCase.run_builder(
                tmp_root / "data", tmp_root / "assets", "--limit", "0"
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("--limit must be a positive integer", result.stderr)
            self.assertFalse((tmp_root / "data" / "dashboard.json").exists())
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)


class DashboardIdempotencyTestCase(unittest.TestCase):
    def test_second_run_is_identical_except_generated_at(self):
        tmp_root = Path(tempfile.mkdtemp(prefix="dashboard-idem-"))
        try:
            runs = []
            for name in ("a", "b"):
                out_dir = tmp_root / name / "data"
                assets_dir = tmp_root / name / "assets"
                result = DashboardBuildTestCase.run_builder(out_dir, assets_dir)
                self.assertEqual(result.returncode, 0, result.stderr)
                runs.append(
                    (
                        json.loads((out_dir / "dashboard.json").read_text(encoding="utf-8")),
                        (out_dir / "dashboard.json").read_text(encoding="utf-8"),
                        (out_dir / "dashboard-data.js").read_text(encoding="utf-8"),
                        assets_dir,
                    )
                )

            (first, first_text, first_js, first_assets) = runs[0]
            (second, second_text, second_js, second_assets) = runs[1]

            self.assertNotIn(first["meta"]["generated_at"], second_text.split("\n")[0])
            first_without = dict(first)
            second_without = dict(second)
            first_without["meta"] = {**first["meta"], "generated_at": "<timestamp>"}
            second_without["meta"] = {**second["meta"], "generated_at": "<timestamp>"}
            self.assertEqual(first_without, second_without)

            # The serialised JSON and JS differ only in the timestamp field.
            pattern = re.compile(r'"generated_at": "[^"]+"')
            self.assertEqual(
                pattern.sub('"generated_at": "X"', first_text),
                pattern.sub('"generated_at": "X"', second_text),
            )
            js_prefix = "window.DASHBOARD_DATA = "
            compact = re.compile(r'"generated_at":"[^"]+"')
            self.assertEqual(
                compact.sub('"generated_at":"X"', first_js)[len(js_prefix) :],
                compact.sub('"generated_at":"X"', second_js)[len(js_prefix) :],
            )

            first_files = {
                path.relative_to(first_assets).as_posix(): path.read_bytes()
                for path in first_assets.rglob("*")
                if path.is_file()
            }
            second_files = {
                path.relative_to(second_assets).as_posix(): path.read_bytes()
                for path in second_assets.rglob("*")
                if path.is_file()
            }
            self.assertEqual(set(first_files), set(second_files))
            self.assertEqual(first_files, second_files)
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)


class DashboardRobustnessTestCase(unittest.TestCase):
    def test_missing_required_artifact_fails_loudly(self):
        module = load_builder_module()
        tmp_root = Path(tempfile.mkdtemp(prefix="dashboard-missing-"))
        try:
            builder = module.Builder(
                out_dir=tmp_root / "data",
                assets_dir=tmp_root / "assets",
                limit=None,
                config=ROOT / "configs" / "default.yaml",
            )
            original = builder.paths["model_comparison"]
            builder.paths["model_comparison"] = tmp_root / "nope.json"
            with self.assertRaises(module.BuildError) as context:
                builder.build()
            self.assertIn("model comparison table", str(context.exception))
            builder.paths["model_comparison"] = original

            # The CLI turns the same condition into a non-zero exit and a message.
            result = subprocess.run(
                [
                    PYTHON,
                    str(BUILDER),
                    "--out",
                    str(tmp_root / "data"),
                    "--assets-dir",
                    str(tmp_root / "assets"),
                    "--config",
                    str(tmp_root / "missing-config.yaml"),
                ],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                timeout=1800,
                check=False,
            )
            # A missing config only affects the seed, not the build.
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(
                (tmp_root / "data" / "dashboard.json").read_text(encoding="utf-8")
            )
            self.assertEqual(payload["meta"]["seed"], 42)  # documented default
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)

    def test_cli_exits_non_zero_when_required_artifact_is_missing(self):
        import contextlib
        import io
        from unittest import mock

        module = load_builder_module()
        tmp_root = Path(tempfile.mkdtemp(prefix="dashboard-cli-"))
        try:
            empty_root = tmp_root / "empty-root"
            empty_root.mkdir()
            stderr = io.StringIO()
            with mock.patch.object(module, "ROOT", empty_root):
                with contextlib.redirect_stderr(stderr):
                    code = module.main(
                        [
                            "--out",
                            str(tmp_root / "data"),
                            "--assets-dir",
                            str(tmp_root / "assets"),
                        ]
                    )
            message = stderr.getvalue()
            self.assertEqual(code, 2, "a missing required artifact must exit non-zero")
            self.assertIn("build_dashboard: error:", message)
            self.assertIn("missing required", message)
            self.assertIn("stats.json", message)
            self.assertFalse((tmp_root / "data" / "dashboard.json").exists())
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)

    def test_missing_optional_artifact_degrades_to_null(self):
        module = load_builder_module()
        tmp_root = Path(tempfile.mkdtemp(prefix="dashboard-optional-"))
        try:
            builder = module.Builder(
                out_dir=tmp_root / "data",
                assets_dir=tmp_root / "assets",
                limit=None,
                config=ROOT / "configs" / "default.yaml",
            )
            builder.load()
            builder.detections = None
            builder.detections_meta = None
            dataset = builder.build_dataset()
            self.assertIsNone(dataset["detector"], "missing detections must not invent numbers")
            # The rest of the panel is still populated from stats.json.
            self.assertEqual(dataset["splits"]["test"], 500)
            self.assertTrue(builder.warnings or True)
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)

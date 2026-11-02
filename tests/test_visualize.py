"""Unit tests for :mod:`src.visualize`.

Run from the project root with::

    MPLBACKEND=Agg .venv/bin/python -m unittest tests.test_visualize -v

The tests only exercise the plotting code: they create a small synthetic
image, feed plausible synthetic metrics/attention maps to every public
function, and check that non-trivial PNGs are produced.
"""

from __future__ import annotations

import atexit
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

# Make ``src/visualize.py`` importable without installing the project.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Keep matplotlib headless and avoid writing its cache into $HOME (which may
# not be writable in sandboxed CI environments).
_MPL_CONFIG_DIR = tempfile.mkdtemp(prefix="mplconfig-test-visualize-")
os.environ.setdefault("MPLCONFIGDIR", _MPL_CONFIG_DIR)
os.environ.setdefault("MPLBACKEND", "Agg")
atexit.register(shutil.rmtree, _MPL_CONFIG_DIR, ignore_errors=True)

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import visualize

MODELS = ("global", "attention", "object")


def _write_synthetic_image(path: Path, seed: int = 0) -> None:
    """Write an 80x60 RGB test image with a gradient and coloured blocks."""

    width, height = 80, 60
    rng = np.random.default_rng(seed)
    ramp_x = np.linspace(0.0, 1.0, width, dtype=np.float32)
    ramp_y = np.linspace(0.0, 1.0, height, dtype=np.float32)

    image = np.empty((height, width, 3), dtype=np.uint8)
    image[..., 0] = (40 + 150 * ramp_x[None, :]).astype(np.uint8)
    image[..., 1] = (50 + 120 * ramp_y[:, None]).astype(np.uint8)
    image[..., 2] = (90 + 80 * (1.0 - ramp_x[None, :] * ramp_y[:, None])).astype(np.uint8)

    # A few solid regions make the attention overlay visually meaningful.
    image[8:26, 10:34] = np.array([210, 80, 60], dtype=np.uint8)
    image[32:52, 42:70] = np.array([60, 160, 90], dtype=np.uint8)
    image[5:18, 58:76] = np.array([240, 210, 80], dtype=np.uint8)
    noise = rng.integers(-8, 9, size=image.shape, dtype=np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    Image.fromarray(image, mode="RGB").save(path)


class TestVisualize(unittest.TestCase):
    """Smoke and contract tests for the visualization module."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp_dir = Path(tempfile.mkdtemp(prefix="test-visualize-"))
        cls.image_paths = []
        for index in range(2):
            path = cls.tmp_dir / f"synthetic_{index}.png"
            _write_synthetic_image(path, seed=index)
            cls.image_paths.append(path)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.tmp_dir, ignore_errors=True)

    # -- helpers ---------------------------------------------------------

    def assert_saved(self, path: str | Path) -> None:
        output = Path(path)
        self.assertTrue(output.is_file(), f"expected output file at {output}")
        self.assertGreater(
            output.stat().st_size,
            5 * 1024,
            f"{output} is suspiciously small ({output.stat().st_size} bytes)",
        )

    def _attention_samples(self) -> list[dict]:
        rng = np.random.default_rng(7)
        words_one = [
            "A", "brown", "dog", "is", "running", "through", "the", "grass",
        ]
        words_two = [
            "Two", "people", "sit", "on", "a", "bench", "near", "trees",
        ]
        return [
            {
                "image_path": str(self.image_paths[0]),
                "caption": (
                    "A brown dog is running through the grass while chasing "
                    "a red ball."
                ),
                "words": words_one,
                "attention": rng.random((len(words_one), 7, 7)),
            },
            {
                "image_path": str(self.image_paths[1]),
                "caption": "Two people sit on a wooden bench near tall trees.",
                "words": words_two,
                "attention": rng.random((len(words_two), 7, 7)),
                "highlight": [1, 5],
            },
        ]

    # -- overlay_attention ----------------------------------------------

    def test_overlay_attention_creates_figure(self) -> None:
        att_map = np.linspace(0.0, 1.0, 49).reshape(7, 7)

        fig, ax = visualize.overlay_attention(
            self.image_paths[0], att_map, title="dog"
        )

        self.assertIsInstance(fig, matplotlib.figure.Figure)
        self.assertIsInstance(ax, matplotlib.axes.Axes)
        self.assertEqual(ax.get_title(), "dog")
        self.assertEqual(len(ax.images), 2)
        self.assertFalse(ax.axison)

        output = self.tmp_dir / "overlay.png"
        fig.savefig(output, dpi=170)
        plt.close(fig)
        self.assert_saved(output)

    def test_overlay_attention_with_provided_ax_and_constant_map(self) -> None:
        fig, (ax_used, ax_unused) = plt.subplots(1, 2, figsize=(7, 3.5))
        constant_map = np.full((7, 7), 3.0, dtype=np.float64)

        returned_fig, returned_ax = visualize.overlay_attention(
            self.image_paths[1],
            constant_map,
            ax=ax_used,
        )

        self.assertIs(returned_fig, fig)
        self.assertIs(returned_ax, ax_used)
        self.assertEqual(len(ax_used.images), 2)
        self.assertFalse(ax_used.axison)

        # A constant attention map must normalise to all zeros after resizing.
        overlay_data = np.asarray(ax_used.images[1].get_array(), dtype=float)
        self.assertEqual(overlay_data.shape, (60, 80))
        self.assertTrue(np.all(overlay_data == 0.0))
        self.assertEqual(ax_unused.get_title(), "")

        output = self.tmp_dir / "overlay_constant.png"
        fig.savefig(output, dpi=170)
        plt.close(fig)
        self.assert_saved(output)

    # -- visualize_attention_samples ------------------------------------

    def test_visualize_attention_samples(self) -> None:
        output = self.tmp_dir / "nested" / "attention_samples.png"

        result = visualize.visualize_attention_samples(
            self._attention_samples(), output, max_words=5, dpi=170
        )

        self.assertEqual(Path(result), output)
        self.assert_saved(output)

    def test_visualize_attention_samples_single_row(self) -> None:
        rng = np.random.default_rng(5)
        words = ["A", "dog", "jumps", "over", "a", "fence"]
        sample = {
            "image_path": str(self.image_paths[0]),
            "caption": "A dog jumps over a wooden fence.",
            "words": words,
            "attention": rng.random((len(words), 7, 7)),
            "highlight": [1, 2, 5],
        }
        output = self.tmp_dir / "attention_samples_single.png"

        result = visualize.visualize_attention_samples([sample], output)

        self.assertEqual(Path(result), output)
        self.assert_saved(output)

    # -- plot_model_comparison ------------------------------------------

    def test_plot_model_comparison(self) -> None:
        summary = [
            {"model": "global", "bleu4": 18.4, "chair_s": 48.2, "chair_i": 12.7},
            {"model": "attention", "bleu4": 24.1, "chair_s": 35.4, "chair_i": 9.8},
            {"model": "object", "bleu4": 26.3, "chair_s": 28.9, "chair_i": 7.4},
        ]
        output = self.tmp_dir / "model_comparison.png"

        result = visualize.plot_model_comparison(summary, output, dpi=170)

        self.assertEqual(Path(result), output)
        self.assert_saved(output)

    def test_plot_model_comparison_without_chair_i(self) -> None:
        summary = [
            {"model": "global", "bleu4": 18.4, "chair_s": 48.2},
            {"model": "object", "bleu4": 26.3, "chair_s": 28.9},
        ]
        output = self.tmp_dir / "model_comparison_no_chair_i.png"

        visualize.plot_model_comparison(summary, output, dpi=170)

        self.assert_saved(output)

    # -- plot_bleu_hallucination ----------------------------------------

    def test_plot_bleu_hallucination(self) -> None:
        rng = np.random.default_rng(11)
        model_settings = {
            "global": (24.0, 2.1),
            "attention": (31.0, 1.4),
            "object": (34.0, 0.9),
        }
        records = []
        for model, (bleu_mean, hall_rate) in model_settings.items():
            for _ in range(22):
                bleu = float(np.clip(rng.normal(bleu_mean, 7.0), 0.0, 100.0))
                n_hallucinated = int(rng.poisson(hall_rate))
                records.append(
                    {
                        "model": model,
                        "sentence_bleu4": bleu,
                        "n_hallucinated": n_hallucinated,
                        "has_hallucination": n_hallucinated > 0,
                    }
                )
        output = self.tmp_dir / "bleu_hallucination.png"

        result = visualize.plot_bleu_hallucination(records, output, dpi=170)

        self.assertEqual(Path(result), output)
        self.assert_saved(output)

    def test_plot_bleu_hallucination_empty_raises(self) -> None:
        output = self.tmp_dir / "bleu_empty.png"
        with self.assertRaises(ValueError):
            visualize.plot_bleu_hallucination([], output)
        self.assertFalse(output.exists())

    # -- plot_training_curves -------------------------------------------

    def test_plot_training_curves(self) -> None:
        histories = {}
        for offset, model in enumerate(MODELS):
            records = []
            for epoch in range(1, 7):
                records.append(
                    {
                        "epoch": epoch,
                        "train_loss": 2.5 * np.exp(-0.28 * epoch) + 0.10 * offset,
                        "val_loss": 2.7 * np.exp(-0.24 * epoch) + 0.12 * offset,
                        "val_bleu4": 8.0 + 3.2 * epoch - 0.9 * offset,
                        "val_chair_s": 52.0 - 4.3 * epoch + 1.1 * offset,
                    }
                )
            histories[model] = records
        output = self.tmp_dir / "training_curves.png"

        result = visualize.plot_training_curves(histories, output, dpi=170)

        self.assertEqual(Path(result), output)
        self.assert_saved(output)

    def test_plot_training_curves_without_optional_metrics(self) -> None:
        histories = {
            model: [
                {
                    "epoch": epoch,
                    "train_loss": 2.4 * np.exp(-0.25 * epoch) + 0.1,
                    "val_loss": 2.6 * np.exp(-0.22 * epoch) + 0.15,
                }
                for epoch in range(1, 5)
            ]
            for model in ("global", "attention")
        }
        output = self.tmp_dir / "training_curves_required_keys.png"

        visualize.plot_training_curves(histories, output, dpi=170)

        self.assert_saved(output)

    # -- plot_qualitative_examples --------------------------------------

    def test_plot_qualitative_examples(self) -> None:
        examples = [
            {
                "image_path": str(self.image_paths[0]),
                "references": [
                    "A brown dog runs through the grass chasing a red ball.",
                    "A dog is playing in a grassy field.",
                ],
                "captions": {
                    "global": "A brown dog is running through grass with a giraffe.",
                    "attention": "A brown dog running through the grass.",
                    "object": "A brown dog chasing a red ball through the grass.",
                },
                "gt_objects": ["dog", "grass", "ball"],
                "hallucinated": {
                    "global": ["giraffe"],
                    "attention": [],
                    "object": [],
                },
            },
            {
                "image_path": str(self.image_paths[1]),
                "references": ["Two people sit on a bench near trees."],
                "captions": {
                    "global": "Two people are sitting on a bench.",
                    "attention": "Two people sitting near trees in a park.",
                    "object": "Two people on a bench beside tall trees.",
                },
                "gt_objects": ["person", "bench", "tree"],
            },
        ]
        output = self.tmp_dir / "qualitative_examples.png"

        result = visualize.plot_qualitative_examples(examples, output, dpi=170)

        self.assertEqual(Path(result), output)
        self.assert_saved(output)


if __name__ == "__main__":
    unittest.main()

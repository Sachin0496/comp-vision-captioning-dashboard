"""Unit tests for the evaluation core (stdlib ``unittest`` only).

Run with::

    .venv/bin/python -m unittest tests.test_metrics -v

Coverage:
* longest-phrase-first mention extraction (``hot dog`` is not ``dog``)
* deduplication of repeated mentions per caption
* CHAIR-S / CHAIR-I / object precision-recall arithmetic
* BLEU and CIDEr-D ranges and edge cases
* bootstrap, paired bootstrap and Spearman helpers
* JSON serializability of the end-to-end evaluation payloads
"""

from __future__ import annotations

import json
import math
import pathlib
import sys
import unittest

import numpy as np

# Make ``import src`` work no matter where the test runner is invoked from.
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.hallucination import (  # noqa: E402
    COCO_CLASSES,
    COCO_CLASS_SET,
    SYNONYMS,
    extract_mentions,
    score_hallucination,
)
from src.metrics import (  # noqa: E402
    bleu_hallucination_analysis,
    bootstrap_ci,
    cider_d,
    corpus_bleu,
    evaluate_predictions,
    paired_bootstrap_difference,
    sentence_bleu,
    spearman_correlation,
)


class TestCocoClasses(unittest.TestCase):
    """The taxonomy itself."""

    def test_eighty_unique_classes(self):
        self.assertEqual(len(COCO_CLASSES), 80)
        self.assertEqual(len(set(COCO_CLASSES)), 80)

    def test_canonical_order(self):
        # Spot-check the standard COCO ordering at both ends.
        self.assertEqual(COCO_CLASSES[0], "person")
        self.assertEqual(COCO_CLASSES[1], "bicycle")
        self.assertEqual(COCO_CLASSES[-1], "toothbrush")
        self.assertEqual(COCO_CLASSES[-2], "hair drier")

    def test_synonym_targets_are_canonical(self):
        self.assertTrue(SYNONYMS)
        targets = set(SYNONYMS.values())
        self.assertTrue(targets <= COCO_CLASS_SET)
        self.assertTrue(targets)  # non-empty

    def test_required_synonyms_present(self):
        expected = {
            "man": "person",
            "woman": "person",
            "people": "person",
            "bike": "bicycle",
            "motorbike": "motorcycle",
            "plane": "airplane",
            "jet": "airplane",
            "sofa": "couch",
            "tv": "tv",
            "television": "tv",
            "phone": "cell phone",
            "cellphone": "cell phone",
            "fridge": "refrigerator",
            "doughnut": "donut",
            "teddy": "teddy bear",
            "potted plant": "potted plant",
            "dining table": "dining table",
            "wine glass": "wine glass",
            "sports ball": "sports ball",
            "ball": "sports ball",
            "baseball bat": "baseball bat",
            "bat": "baseball bat",
            "baseball glove": "baseball glove",
            "glove": "baseball glove",
            "tennis racket": "tennis racket",
            "racket": "tennis racket",
        }
        for surface, canonical in expected.items():
            with self.subTest(surface=surface):
                self.assertIn(surface, SYNONYMS)
                self.assertEqual(SYNONYMS[surface], canonical)

    def test_plurals_present(self):
        for surface, canonical in {
            "dogs": "dog",
            "bicycles": "bicycle",
            "men": "person",
            "people": "person",
            "cookies": None,
            "mice": "mouse",
            "knives": "knife",
            "teddy bears": "teddy bear",
        }.items():
            if canonical is None:
                self.assertNotIn(surface, SYNONYMS)
            else:
                self.assertEqual(SYNONYMS.get(surface), canonical)


class TestMentionExtraction(unittest.TestCase):
    """Longest-phrase matching, boundaries, offsets and deduplication."""

    def classes(self, caption: str) -> list[str]:
        return [m["class_name"] for m in extract_mentions(caption)]

    def test_hot_dog_is_not_a_dog(self):
        classes = self.classes("a hot dog on a dining table")
        self.assertEqual(classes, ["hot dog", "dining table"])
        self.assertNotIn("dog", classes)

    def test_teddy_bear_is_not_a_bear(self):
        self.assertEqual(self.classes("a teddy bear on the bed"), ["teddy bear", "bed"])
        # ...but a real bear still counts.
        self.assertEqual(self.classes("a brown bear in the woods"), ["bear"])

    def test_mentions_have_valid_offsets(self):
        caption = "a hot dog on a dining table"
        for mention in extract_mentions(caption):
            with self.subTest(surface=mention["surface"]):
                self.assertEqual(caption[mention["start"] : mention["end"]], mention["surface"])
                self.assertLess(mention["start"], mention["end"])
                self.assertIn(mention["class_name"], COCO_CLASS_SET)

    def test_case_insensitive(self):
        self.assertEqual(self.classes("A HOT DOG on a Dining Table"), ["hot dog", "dining table"])

    def test_word_boundaries(self):
        # Substrings inside other words must not trigger mentions.
        self.assertEqual(self.classes("a personally interesting day"), [])
        self.assertEqual(self.classes("dogmatic behavior"), [])
        self.assertEqual(self.classes("grassy hill"), [])

    def test_plurals_and_dedupe(self):
        mentions = extract_mentions("two men riding bicycles next to a woman")
        classes = [m["class_name"] for m in mentions]
        # Every *occurrence* is returned: "men" once plus "woman" once.
        self.assertEqual(classes.count("person"), 2)
        self.assertEqual(classes.count("bicycle"), 1)
        # ...but the scorer deduplicates per caption.
        result = score_hallucination(
            [{"image_id": 1, "caption": "two men riding bicycles next to a woman"}],
            {"1": ["person", "bicycle"]},
        )
        self.assertEqual(result["per_image"][0]["mentioned"], ["bicycle", "person"])
        self.assertEqual(result["per_image"][0]["n_mentions"], 2)

    def test_non_coco_nouns_ignored(self):
        self.assertEqual(self.classes("the sky above the grass and water"), [])

    def test_empty_caption(self):
        self.assertEqual(extract_mentions(""), [])
        self.assertEqual(extract_mentions("   "), [])

    def test_baseball_bat_without_sports_ball(self):
        classes = self.classes("a baseball player swinging a bat")
        self.assertIn("baseball bat", classes)
        self.assertIn("person", classes)
        self.assertNotIn("sports ball", classes)


class TestHallucinationScoring(unittest.TestCase):
    """CHAIR-S / CHAIR-I / precision-recall arithmetic."""

    def setUp(self):
        self.predictions = [
            {
                "image_id": 1,
                "caption": "a man riding a bike next to a dog",
                "references": ["a man on a bike", "a cyclist with a dog"],
            },
            {
                "image_id": 2,
                "caption": "a cat on a dining table",
                "references": ["a cat on a table"],
            },
        ]
        self.gt = {"1": ["person", "bicycle"], "2": ["cat", "dining table"]}

    def test_chair_math(self):
        result = score_hallucination(self.predictions, self.gt)

        # Image 1: person, bicycle (correct) + dog (hallucinated) -> 3 mentions.
        # Image 2: cat, dining table (both correct)              -> 2 mentions.
        self.assertEqual(result["n_images"], 2)
        self.assertEqual(result["n_mentions"], 5)
        self.assertEqual(result["n_hallucinated"], 1)

        self.assertAlmostEqual(result["chair_s"], 1 / 5)
        self.assertAlmostEqual(result["chair_i"], 1 / 2)
        # precision = 4 correct / 5 mentioned
        self.assertAlmostEqual(result["object_precision"], 4 / 5)
        # recall = 4 correct / 4 ground-truth classes
        self.assertAlmostEqual(result["object_recall"], 1.0)
        expected_f1 = 2 * 0.8 * 1.0 / (0.8 + 1.0)
        self.assertAlmostEqual(result["object_f1"], expected_f1)

    def test_per_image_records(self):
        result = score_hallucination(self.predictions, self.gt)
        first, second = result["per_image"]

        self.assertEqual(first["image_id"], 1)
        self.assertEqual(first["mentioned"], ["bicycle", "dog", "person"])
        self.assertEqual(first["hallucinated"], ["dog"])
        self.assertEqual(first["n_mentions"], 3)
        self.assertEqual(first["n_hallucinated"], 1)
        self.assertTrue(first["has_hallucination"])
        self.assertEqual(first["reference_objects"], ["bicycle", "person"])

        self.assertEqual(second["hallucinated"], [])
        self.assertEqual(second["n_hallucinated"], 0)
        self.assertFalse(second["has_hallucination"])

    def test_repeated_mentions_do_not_inflate_chair(self):
        # CHAIR counts unique object mentions per caption.
        predictions = [{"image_id": 1, "caption": "dog dog dog and another dog"}]
        result = score_hallucination(predictions, {"1": []})
        self.assertEqual(result["n_mentions"], 1)
        self.assertEqual(result["n_hallucinated"], 1)
        self.assertAlmostEqual(result["chair_s"], 1.0)

    def test_empty_mentions_give_zero_chair(self):
        result = score_hallucination(
            [{"image_id": 1, "caption": "a beautiful sunny day", "references": []}],
            {"1": ["cat"]},
        )
        self.assertEqual(result["chair_s"], 0.0)
        self.assertEqual(result["chair_i"], 0.0)
        self.assertEqual(result["n_mentions"], 0)
        self.assertEqual(result["object_precision"], 0.0)
        self.assertEqual(result["object_recall"], 0.0)
        self.assertEqual(result["object_f1"], 0.0)

    def test_no_hallucination(self):
        result = score_hallucination(
            [{"image_id": 1, "caption": "a cat on a bed"}], {"1": ["cat", "bed"]}
        )
        self.assertEqual(result["chair_s"], 0.0)
        self.assertEqual(result["chair_i"], 0.0)
        self.assertAlmostEqual(result["object_precision"], 1.0)
        self.assertAlmostEqual(result["object_recall"], 1.0)

    def test_unknown_image_is_scored_against_empty_gt(self):
        result = score_hallucination(
            [{"image_id": 999, "caption": "a dog"}], {"1": ["dog"]}
        )
        self.assertEqual(result["per_image"][0]["reference_objects"], [])
        self.assertEqual(result["per_image"][0]["hallucinated"], ["dog"])
        # The single image hallucinates, so CHAIR-I is 1.0 and CHAIR-S is 1.0.
        self.assertEqual(result["chair_i"], 1.0)
        self.assertEqual(result["chair_s"], 1.0)

    def test_gt_lookup_accepts_int_or_str_keys(self):
        predictions = [{"image_id": "139", "caption": "a person"}]
        for gt in ({"139": ["person"]}, {139: ["person"]}):
            with self.subTest(gt=gt):
                result = score_hallucination(predictions, gt)
                self.assertEqual(result["per_image"][0]["mentioned"], ["person"])
                self.assertEqual(result["chair_s"], 0.0)

    def test_non_coco_gt_names_are_ignored(self):
        result = score_hallucination(
            [{"image_id": 1, "caption": "a dog"}], {"1": ["dog", "sky", "grass"]}
        )
        self.assertEqual(result["per_image"][0]["reference_objects"], ["dog"])
        self.assertEqual(result["chair_s"], 0.0)

    def test_empty_predictions(self):
        result = score_hallucination([], {})
        self.assertEqual(result["n_images"], 0)
        self.assertEqual(result["chair_s"], 0.0)
        self.assertEqual(result["chair_i"], 0.0)


class TestBleu(unittest.TestCase):
    """Corpus and sentence BLEU."""

    def setUp(self):
        self.predictions = [
            {"image_id": 1, "caption": "a man riding a bike next to a dog"},
            {"image_id": 2, "caption": "a cat on a dining table"},
        ]
        self.references = [
            ["a man rides a bike", "a cyclist with a dog"],
            ["a cat sits on a table", "a cat on a table"],
        ]

    def test_keys_and_ranges(self):
        result = corpus_bleu(self.predictions, self.references)
        for key in ("bleu1", "bleu2", "bleu3", "bleu4"):
            self.assertIn(key, result)
            self.assertIsInstance(result[key], float)
            self.assertGreaterEqual(result[key], 0.0)
            self.assertLessEqual(result[key], 100.0)

    def test_perfect_corpus_scores_100(self):
        predictions = [{"caption": "a cat"}, {"caption": "a dog"}]
        references = [["a cat"], ["a dog"]]
        result = corpus_bleu(predictions, references)
        self.assertAlmostEqual(result["bleu1"], 100.0, places=6)
        self.assertEqual(result["sys_len"], 4)
        self.assertEqual(result["ref_len"], 4)

    def test_reference_transposition(self):
        # Two references per item must be passed as sacrebleu "streams".
        predictions = [{"caption": "a cat"}, {"caption": "a dog"}]
        references = [["a cat", "the cat"], ["a dog", "the dog"]]
        result = corpus_bleu(predictions, references)
        self.assertAlmostEqual(result["bleu1"], 100.0, places=6)

    def test_sentence_bleu_range_and_order(self):
        exact = sentence_bleu("a cat on a table", ["a cat on a table"])
        unrelated = sentence_bleu("something entirely different", ["a cat on a table"])
        self.assertAlmostEqual(exact, 100.0, places=6)
        self.assertLess(unrelated, exact)
        self.assertGreaterEqual(unrelated, 0.0)

        bleu2 = sentence_bleu("a cat on a table", ["a cat on a table"], max_order=2)
        self.assertTrue(math.isfinite(bleu2))
        self.assertGreaterEqual(bleu2, 0.0)
        self.assertLessEqual(bleu2, 100.0)

    def test_sentence_bleu_without_references(self):
        self.assertEqual(sentence_bleu("a cat", []), 0.0)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            corpus_bleu(self.predictions, self.references[:1])


class TestCiderD(unittest.TestCase):
    """From-scratch CIDEr-D."""

    def setUp(self):
        self.predictions = [
            {"caption": "a man riding a bike next to a dog"},
            {"caption": "a cat on a dining table"},
        ]
        self.references = [
            ["a man rides a bike", "a cyclist with a dog"],
            ["a cat sits on a table", "a cat on a table"],
        ]

    def test_finite_and_bounded(self):
        score = cider_d(self.predictions, self.references)
        self.assertIsInstance(score, float)
        self.assertTrue(math.isfinite(score))
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 10.0)

    def test_perfect_caption_scores_ten(self):
        score = cider_d(
            [{"caption": "a cat sits on a table"}],
            [["a cat sits on a table", "a cat sits on a table"]],
        )
        self.assertAlmostEqual(score, 10.0, places=6)

    def test_unrelated_caption_scores_lower(self):
        exact = cider_d([{"caption": "a cat sits on a table"}], [["a cat sits on a table"]])
        unrelated = cider_d([{"caption": "zzz qqq"}], [["a cat sits on a table"]])
        self.assertGreater(exact, unrelated)
        self.assertGreaterEqual(unrelated, 0.0)

    def test_empty_inputs(self):
        self.assertEqual(cider_d([], []), 0.0)
        self.assertEqual(cider_d([{"caption": ""}], [["a cat"]]), 0.0)
        self.assertEqual(cider_d([{"caption": "a cat"}], [[]]), 0.0)

    def test_deterministic(self):
        first = cider_d(self.predictions, self.references)
        second = cider_d(self.predictions, self.references)
        self.assertEqual(first, second)

    def test_order_independence_of_mean(self):
        # The score is a mean over images, so permuting image order must not
        # change it.  (Note: duplicating the corpus is *not* an invariance --
        # document frequency is corpus-relative by construction.)
        order = [1, 0]
        shuffled_predictions = [self.predictions[i] for i in order]
        shuffled_references = [self.references[i] for i in order]
        self.assertAlmostEqual(
            cider_d(shuffled_predictions, shuffled_references),
            cider_d(self.predictions, self.references),
            places=9,
        )

    def test_punctuation_and_case_insensitive(self):
        plain = cider_d([{"caption": "a cat on a table"}], [["a cat on a table"]])
        noisy = cider_d([{"caption": "A cat, on a TABLE!"}], [["a cat on a table"]])
        self.assertAlmostEqual(plain, noisy, places=6)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            cider_d(self.predictions, self.references[:1])


class TestBootstrapAndCorrelation(unittest.TestCase):
    """Resampling helpers and rank correlation."""

    def test_bootstrap_ci_shape(self):
        low, high = bootstrap_ci([1, 2, 3, 4, 5], n_boot=300, seed=0)
        self.assertIsInstance(low, float)
        self.assertIsInstance(high, float)
        self.assertLessEqual(low, high)
        # The interval should bracket the sample mean for a well-behaved input.
        self.assertLessEqual(low, 3.0)
        self.assertGreaterEqual(high, 3.0)

    def test_bootstrap_ci_edge_cases(self):
        self.assertTrue(all(math.isnan(v) for v in bootstrap_ci([], n_boot=10, seed=0)))
        self.assertEqual(bootstrap_ci([7.0], n_boot=10, seed=0), (7.0, 7.0))

    def test_bootstrap_ci_deterministic(self):
        self.assertEqual(
            bootstrap_ci([1, 2, 3, 4], n_boot=50, seed=3),
            bootstrap_ci([1, 2, 3, 4], n_boot=50, seed=3),
        )

    def test_paired_bootstrap_keys_and_values(self):
        result = paired_bootstrap_difference([1, 0, 1, 0], [0, 0, 1, 0], n_boot=300, seed=0)
        for key in ("mean_diff", "ci_low", "ci_high", "p_two_sided"):
            self.assertIn(key, result)
            self.assertIsInstance(result[key], float)
        self.assertAlmostEqual(result["mean_diff"], 0.25)
        self.assertLessEqual(result["ci_low"], result["ci_high"])
        self.assertGreaterEqual(result["p_two_sided"], 0.0)
        self.assertLessEqual(result["p_two_sided"], 1.0)

    def test_paired_bootstrap_identical_models(self):
        values = [1, 0, 1, 0, 1]
        result = paired_bootstrap_difference(values, values, n_boot=200, seed=1)
        self.assertAlmostEqual(result["mean_diff"], 0.0)
        self.assertGreaterEqual(result["p_two_sided"], 0.9)

    def test_paired_bootstrap_length_mismatch(self):
        with self.assertRaises(ValueError):
            paired_bootstrap_difference([1, 2, 3], [1, 2], n_boot=10)

    def test_spearman_perfect(self):
        result = spearman_correlation([1, 2, 3, 4], [10, 20, 30, 40])
        self.assertAlmostEqual(result["rho"], 1.0, places=6)
        self.assertEqual(result["n"], 4)
        self.assertGreaterEqual(result["p_value"], 0.0)

    def test_spearman_constant_is_none(self):
        result = spearman_correlation([1, 1, 1], [1, 2, 3])
        self.assertIsNone(result["rho"])
        self.assertIsNone(result["p_value"])

    def test_spearman_too_short(self):
        result = spearman_correlation([1, 2], [1, 2])
        self.assertIsNone(result["rho"])

    def test_spearman_native_types(self):
        result = spearman_correlation([1, 2, 3, 4], [4, 3, 2, 1])
        self.assertIsInstance(result["rho"], float)
        self.assertIsInstance(result["n"], int)
        json.dumps(result)


class TestEvaluatePredictions(unittest.TestCase):
    """End-to-end evaluation payload."""

    def setUp(self):
        self.predictions = [
            {
                "image_id": 1,
                "caption": "a man riding a bike next to a dog",
                "references": ["a man rides a bike", "a cyclist with a dog"],
            },
            {
                "image_id": 2,
                "caption": "a cat on a dining table",
                "references": ["a cat sits on a table", "a cat on a table"],
            },
            {
                "image_id": 3,
                "caption": "a bird in the sky",
                "references": ["a bird flying"],
            },
        ]
        self.gt = {
            "1": ["person", "bicycle"],
            "2": ["cat", "dining table"],
            "3": ["bird"],
        }

    def test_score_keys(self):
        result = evaluate_predictions(self.predictions, self.gt)
        for key in (
            "n",
            "bleu1",
            "bleu2",
            "bleu3",
            "bleu4",
            "cider_d",
            "chair_s",
            "chair_i",
            "object_precision",
            "object_recall",
            "object_f1",
            "per_image",
        ):
            self.assertIn(key, result)
        self.assertEqual(result["n"], 3)

    def test_per_image_length_and_extra_fields(self):
        result = evaluate_predictions(self.predictions, self.gt)
        self.assertEqual(len(result["per_image"]), 3)
        for record in result["per_image"]:
            self.assertIn("sentence_bleu4", record)
            self.assertIn("sentence_bleu2", record)
            self.assertIn("bleu_bucket", record)
            self.assertGreaterEqual(record["sentence_bleu4"], 0.0)
            self.assertLessEqual(record["sentence_bleu4"], 100.0)

    def test_json_serializable(self):
        result = evaluate_predictions(self.predictions, self.gt)
        payload = json.dumps(result)  # must not raise
        self.assertGreater(len(payload), 0)
        # Round-trip preserves the per-image count.
        self.assertEqual(len(json.loads(payload)["per_image"]), 3)

    def test_empty_predictions_raise(self):
        with self.assertRaises(ValueError):
            evaluate_predictions([], {})


class TestBleuHallucinationAnalysis(unittest.TestCase):
    """The research-question analysis."""

    @classmethod
    def setUpClass(cls):
        # Build a deterministic, varied per-image table.
        cls.records = []
        for index in range(40):
            hallucinated = index % 3
            cls.records.append(
                {
                    "image_id": index,
                    "caption": "a cat",
                    "references": ["a cat"],
                    "mentioned": ["cat"],
                    "hallucinated": ["dog"] * hallucinated,
                    "n_mentions": 1 + hallucinated,
                    "n_hallucinated": hallucinated,
                    "has_hallucination": hallucinated > 0,
                    "reference_objects": ["cat"],
                    "sentence_bleu4": float(40 - index),
                    "sentence_bleu2": float(60 - index),
                }
            )

    def test_structure_and_types(self):
        result = bleu_hallucination_analysis(self.records, n_boot=100, seed=0)
        self.assertEqual(result["n"], 40)
        self.assertIn("spearman_bleu4_vs_n_hallucinated", result)
        self.assertIn("spearman_bleu2_vs_has_hallucination", result)
        self.assertIn("bleu4_quartiles", result)
        self.assertFalse(math.isnan(result["hallucination_rate"]))

        self.assertIsInstance(result["hallucination_rate"], float)
        self.assertIsInstance(result["n"], int)
        self.assertEqual(len(result["hallucination_rate_ci"]), 2)

    def test_quartiles(self):
        result = bleu_hallucination_analysis(self.records, n_boot=50, seed=0)
        buckets = [row["bucket"] for row in result["bleu4_quartiles"]]
        self.assertEqual(buckets, ["Q1", "Q2", "Q3", "Q4"])
        total = sum(row["n"] for row in result["bleu4_quartiles"])
        self.assertEqual(total, 40)
        for row in result["bleu4_quartiles"]:
            self.assertLessEqual(row["bleu4_min"], row["bleu4_max"])
            self.assertGreaterEqual(row["hallucination_rate"], 0.0)
            self.assertLessEqual(row["hallucination_rate"], 1.0)

    def test_json_serializable(self):
        result = bleu_hallucination_analysis(self.records, n_boot=50, seed=0)
        payload = json.dumps(result)  # must not raise
        self.assertIn("bleu4_quartiles", json.loads(payload))

    def test_constant_bleu_handled(self):
        records = [dict(record, sentence_bleu4=10.0) for record in self.records]
        result = bleu_hallucination_analysis(records, n_boot=20, seed=0)
        # Constant input => undefined correlation, reported as None (not NaN).
        self.assertIsNone(result["spearman_bleu4_vs_n_hallucinated"]["rho"])
        json.dumps(result)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            bleu_hallucination_analysis([], n_boot=10)


class TestNativeTypes(unittest.TestCase):
    """No numpy scalars may leak into JSON-facing payloads."""

    def test_no_numpy_scalars(self):
        def walk(value):
            if isinstance(value, dict):
                for item in value.values():
                    walk(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    walk(item)
            else:
                self.assertNotIsInstance(
                    value, np.generic, f"numpy scalar leaked: {value!r}"
                )

        predictions = [
            {
                "image_id": 1,
                "caption": "a dog and a cat",
                "references": ["a dog", "a cat"],
            },
            {
                "image_id": 2,
                "caption": "a bird",
                "references": ["a bird"],
            },
        ]
        walk(evaluate_predictions(predictions, {"1": ["dog"], "2": ["bird"]}))
        walk(bleu_hallucination_analysis(
            evaluate_predictions(predictions, {"1": ["dog"], "2": ["bird"]})["per_image"],
            n_boot=20,
        ))


if __name__ == "__main__":
    unittest.main()

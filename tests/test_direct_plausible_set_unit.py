"""Unit tests for direct plausible-candidate set selection."""

import json
import os
import sys
import tempfile
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from plausible_candidate_selection.pipeline import (
    aggregate_plausible_set_evaluations,
    run_direct_plausible_set,
)
from plausible_candidate_selection.schemas import validate_plausible_set_result


def make_item_info(root):
    data_dir = os.path.join(root, "pog_dense")
    os.makedirs(data_dir, exist_ok=True)
    item_info = {
        "1": {"title": "Panama beach hat", "cate_id": "opaque_hat"},
        "2": {"title": "casual leather watch", "cate_id": "opaque_watch"},
        "3": {"title": "summer sandals", "cate_id": "opaque_shoe"},
        "4": {"title": "winter wool coat", "cate_id": "opaque_coat"},
        "5": {"title": "lightweight resort dress", "cate_id": "opaque_dress"},
        "6": {"title": "formal office briefcase", "cate_id": "opaque_bag"},
    }
    with open(
        os.path.join(data_dir, "item_info.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(item_info, handle)


class DirectPlausibleSetTest(unittest.IsolatedAsyncioTestCase):
    async def test_one_call_selects_a_set_and_hides_evaluator_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            make_item_info(temporary)
            conf = {
                "dataset": "pog_dense",
                "data_path": temporary,
            }
            sample = {
                "bundle_id": 99,
                "input_indices": [1, 2],
                "candidate_indices": [3, 4, 5, 6],
                "true_indice": 5,
                "true_option_char": "C",
            }
            calls = []

            async def call_text(prompt, step_name):
                calls.append((prompt, step_name))
                return json.dumps(
                    {
                        "plausible_candidates": [
                            {
                                "label": "A",
                                "completion_hypothesis": (
                                    "A relaxed warm-weather vacation outfit."
                                ),
                                "reason": (
                                    "Sandals provide practical beach footwear."
                                ),
                            },
                            {
                                "label": "C",
                                "completion_hypothesis": (
                                    "A complete resort outfit centered on light apparel."
                                ),
                                "reason": (
                                    "The dress supplies the main warm-weather garment."
                                ),
                            },
                        ],
                        "ranking": ["A", "B", "C", "D"],
                    }
                )

            result = await run_direct_plausible_set(
                sample,
                conf,
                call_text,
            )

        self.assertEqual(len(calls), 1)
        prompt = calls[0][0]
        self.assertIn("summer sandals", prompt)
        self.assertIn("lightweight resort dress", prompt)
        self.assertNotIn('"item_id"', prompt)
        self.assertNotIn("true_indice", prompt)
        self.assertNotIn("ground_truth", prompt)
        self.assertNotIn("opaque_hat", prompt)
        self.assertEqual(
            result["evaluation"]["plausible_labels"],
            ["A", "C"],
        )
        self.assertEqual(result["evaluation"]["plausible_set_size"], 2)
        self.assertEqual(result["evaluation"]["selection_fraction"], 0.5)
        self.assertTrue(result["evaluation"]["gt_in_plausible_set"])
        self.assertEqual(result["evaluation"]["ranking"], ["A", "B", "C", "D"])
        self.assertEqual(
            result["evaluation"]["plausible_rank_top_k"],
            ["A", "B"],
        )
        self.assertFalse(
            result["evaluation"]["plausible_ranking_consistent"]
        )
        self.assertEqual(result["evaluation"]["gt_rank"], 3)
        self.assertAlmostEqual(result["evaluation"]["reciprocal_rank"], 1 / 3)
        self.assertFalse(result["evaluation"]["hit_at_1"])
        self.assertTrue(result["evaluation"]["hit_at_3"])
        self.assertTrue(result["evaluation"]["hit_at_5"])
        self.assertEqual(result["evaluation"]["llm_calls"], 1)


class DirectPlausibleSetSchemaTest(unittest.TestCase):
    def test_duplicate_labels_are_rejected(self):
        candidate = {
            "label": "A",
            "completion_hypothesis": "A coherent outfit.",
            "reason": "A concrete compositional reason.",
        }
        issues = validate_plausible_set_result(
            {
                "plausible_candidates": [candidate, dict(candidate)],
                "ranking": ["A", "B"],
            },
            ["A", "B"],
        )
        self.assertIn("plausible candidate labels must be unique", issues)

    def test_ranking_must_be_a_complete_permutation(self):
        issues = validate_plausible_set_result(
            {
                "plausible_candidates": [],
                "ranking": ["A", "A"],
            },
            ["A", "B"],
        )
        self.assertIn("ranking labels must be unique", issues)
        self.assertIn(
            "ranking must contain every supplied answer-option label exactly once",
            issues,
        )


class DirectPlausibleSetAggregationTest(unittest.TestCase):
    def test_coverage_is_reported_with_same_size_random_baseline(self):
        summary = aggregate_plausible_set_evaluations(
            [
                {
                    "plausible_set_size": 2,
                    "selection_fraction": 0.2,
                    "gt_in_plausible_set": True,
                    "gt_rank": 2,
                    "plausible_ranking_consistent": True,
                    "error": "",
                },
                {
                    "plausible_set_size": 4,
                    "selection_fraction": 0.4,
                    "gt_in_plausible_set": False,
                    "gt_rank": 4,
                    "plausible_ranking_consistent": False,
                    "error": "",
                },
                {
                    "plausible_set_size": 0,
                    "selection_fraction": 0.0,
                    "gt_in_plausible_set": False,
                    "gt_rank": None,
                    "plausible_ranking_consistent": None,
                    "error": "invalid response",
                },
            ]
        )

        self.assertEqual(summary["completed_sample_count"], 3)
        self.assertEqual(summary["valid_sample_count"], 2)
        self.assertEqual(summary["error_sample_count"], 1)
        self.assertEqual(summary["gt_coverage_count"], 1)
        self.assertAlmostEqual(summary["gt_plausible_coverage"], 0.5)
        self.assertAlmostEqual(summary["average_plausible_set_size"], 3.0)
        self.assertAlmostEqual(summary["average_selection_fraction"], 0.3)
        self.assertAlmostEqual(summary["random_same_size_expected_coverage"], 0.3)
        self.assertAlmostEqual(summary["coverage_above_random_same_size"], 0.2)
        self.assertEqual(
            summary["plausible_set_size_distribution"],
            {"2": 1, "4": 1},
        )
        self.assertEqual(summary["plausible_ranking_consistency_count"], 1)
        self.assertAlmostEqual(
            summary["plausible_ranking_consistency_rate"],
            0.5,
        )
        self.assertEqual(summary["valid_ranking_sample_count"], 2)
        self.assertAlmostEqual(summary["mean_gt_rank"], 3.0)
        self.assertAlmostEqual(summary["mean_reciprocal_rank"], 0.375)
        self.assertAlmostEqual(summary["hit_rate_at_1"], 0.0)
        self.assertAlmostEqual(summary["hit_rate_at_3"], 0.5)
        self.assertAlmostEqual(summary["hit_rate_at_5"], 1.0)
        self.assertEqual(summary["gt_rank_distribution"], {"2": 1, "4": 1})


if __name__ == "__main__":
    unittest.main()

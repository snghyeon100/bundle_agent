"""Unit tests for two-step counterfactual set reinterpretation."""

import json
import os
import sys
import tempfile
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from counterfactual_reinterpretation.pipeline import (
    aggregate_reinterpretation_evaluations,
    run_counterfactual_reinterpretation,
)
from counterfactual_reinterpretation.schemas import (
    validate_adjudication,
    validate_reinterpretations,
)


def _make_item_info(root):
    data_dir = os.path.join(root, "pog_dense")
    os.makedirs(data_dir, exist_ok=True)
    with open(
        os.path.join(data_dir, "item_info.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            {
                "1": {"title": "summer dress", "cate_id": "hidden_dress"},
                "2": {"title": "sunglasses", "cate_id": "hidden_glasses"},
                "3": {"title": "beach sandals", "cate_id": "hidden_shoes"},
                "4": {"title": "formal briefcase", "cate_id": "hidden_bag"},
            },
            handle,
        )


def _reinterpretation(label, candidate):
    return {
        "label": label,
        "completed_set_interpretation": f"A coherent reading with {candidate}.",
        "partial_member_contributions": [
            {"partial_label": "P1", "contribution": "Main garment."},
            {"partial_label": "P2", "contribution": "Outdoor accessory."},
        ],
        "candidate_contribution": f"{candidate} contributes a completion.",
        "role_closure": "It fills or tests the remaining role.",
        "counterfactual_necessity": "Removing it leaves that role unresolved.",
        "conflicts_or_redundancies": [],
    }


class CounterfactualPipelineTest(unittest.IsolatedAsyncioTestCase):
    async def test_two_calls_reinterpret_then_rank_without_ids_or_gt(self):
        with tempfile.TemporaryDirectory() as temporary:
            _make_item_info(temporary)
            conf = {"dataset": "pog_dense", "data_path": temporary}
            sample = {
                "bundle_id": 9,
                "input_indices": [1, 2],
                "candidate_indices": [3, 4],
                "true_indice": 3,
                "true_option_char": "A",
            }
            calls = []

            async def call_analysis(prompt, step_name):
                calls.append(("analysis", prompt, step_name))
                return json.dumps(
                    {
                        "reinterpretations": [
                            _reinterpretation("A", "sandals"),
                            _reinterpretation("B", "briefcase"),
                        ]
                    }
                )

            async def call_decision(prompt, step_name):
                calls.append(("decision", prompt, step_name))
                return json.dumps(
                    {
                        "ranking": ["A", "B"],
                        "prediction": "A",
                        "decisive_comparison": (
                            "Sandals close the seasonal composition more coherently."
                        ),
                        "decision_basis": {
                            "explanatory_coverage": "A explains both partial members.",
                            "role_closure": "A fills complementary footwear.",
                            "counterfactual_necessity": "Removing A restores the gap.",
                            "conflict_or_redundancy": "B introduces a context conflict.",
                        },
                    }
                )

            result = await run_counterfactual_reinterpretation(
                sample,
                conf,
                call_analysis,
                call_decision,
            )

        self.assertEqual([call[0] for call in calls], ["analysis", "decision"])
        self.assertIn("Do not choose, rank, compare", calls[0][1])
        self.assertIn("candidate_reinterpretations", calls[1][1])
        self.assertNotIn('"item_id"', calls[0][1])
        self.assertNotIn("true_indice", calls[0][1])
        self.assertNotIn("hidden_dress", calls[0][1])
        self.assertEqual(result["validation_issues"], [])
        self.assertTrue(result["evaluation"]["valid"])
        self.assertEqual(result["evaluation"]["prediction"], "A")
        self.assertEqual(result["evaluation"]["gt_rank"], 1)
        self.assertTrue(result["evaluation"]["hit"])
        self.assertEqual(result["evaluation"]["llm_calls"], 2)


class CounterfactualSchemaTest(unittest.TestCase):
    def test_stage_one_requires_every_partial_for_every_candidate(self):
        entry = _reinterpretation("A", "sandals")
        entry["partial_member_contributions"] = entry[
            "partial_member_contributions"
        ][:1]
        issues = validate_reinterpretations(
            {"reinterpretations": [entry]},
            ["A"],
            ["P1", "P2"],
        )
        self.assertTrue(
            any("must explain every supplied partial member" in issue for issue in issues)
        )

    def test_stage_two_prediction_must_match_rank_one(self):
        issues = validate_adjudication(
            {
                "ranking": ["A", "B"],
                "prediction": "B",
                "decisive_comparison": "A is stronger.",
                "decision_basis": {
                    "explanatory_coverage": "A",
                    "role_closure": "A",
                    "counterfactual_necessity": "A",
                    "conflict_or_redundancy": "A",
                },
            },
            ["A", "B"],
        )
        self.assertIn("prediction must equal the first ranking label", issues)


class CounterfactualAggregationTest(unittest.TestCase):
    def test_only_structurally_valid_rows_contribute_to_ranking_metrics(self):
        summary = aggregate_reinterpretation_evaluations(
            [
                {
                    "valid": True,
                    "error": "",
                    "gt_rank": 1,
                },
                {
                    "valid": True,
                    "error": "",
                    "gt_rank": 4,
                },
                {
                    "valid": False,
                    "error": "stage_1 invalid",
                    "gt_rank": 2,
                },
            ]
        )
        self.assertEqual(summary["completed_sample_count"], 3)
        self.assertEqual(summary["valid_sample_count"], 2)
        self.assertEqual(summary["invalid_or_error_sample_count"], 1)
        self.assertAlmostEqual(summary["hit_rate_at_1"], 0.5)
        self.assertAlmostEqual(summary["hit_rate_at_3"], 0.5)
        self.assertAlmostEqual(summary["hit_rate_at_5"], 1.0)
        self.assertAlmostEqual(summary["mean_reciprocal_rank"], 0.625)
        self.assertAlmostEqual(summary["mean_gt_rank"], 2.5)
        self.assertEqual(summary["gt_rank_distribution"], {"1": 1, "4": 1})


if __name__ == "__main__":
    unittest.main()

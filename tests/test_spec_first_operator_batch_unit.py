"""Unit tests for the spec-first execution and prediction bridge."""

import json
import os
import sys
import tempfile
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from operator_learning.spec_first_prediction import (
    aggregate_prediction_rows,
    build_strategy_evidence,
    evaluate_full_ranking,
)
from operator_learning.spec_first_runtime import (
    execute_strategy_program,
    validate_context_result,
)


class SpecFirstBatchBridgeTest(unittest.TestCase):
    def test_generated_contexts_are_rendered_unchanged(self):
        specs = [
            {
                "strategy_id": "S1",
                "intent": "intent",
                "reference_construction": "reference",
                "candidate_relation": "relation",
            }
        ]
        reports = [
            {
                "strategy_id": "S1",
                "success": True,
                "result": [
                    {
                        "label": "A",
                        "item_id": 1,
                        "contexts": [
                            {
                                "sources": ["item_metadata"],
                                "text": "exact generated context",
                                "supporting_item_ids": [99],
                            }
                        ],
                    }
                ],
            }
        ]
        evidence = build_strategy_evidence(
            specs=specs,
            execution_reports=reports,
            candidate_labels=["A"],
        )
        self.assertEqual(
            evidence[0]["candidate_evidence"][0]["contexts"],
            [
                {
                    "sources": ["item_metadata"],
                    "text": "exact generated context",
                }
            ],
        )

    def test_context_accepts_multiple_declared_sources(self):
        candidates = [{"label": "A", "item_id": 1}]
        result = [
            {
                "label": "A",
                "item_id": 1,
                "contexts": [
                    {
                        "sources": [
                            "item_description_embedding",
                            "item_metadata",
                        ],
                        "text": "selected related item text",
                    }
                ],
            }
        ]
        self.assertEqual(
            validate_context_result(
                result,
                candidates,
                {"item_description_embedding", "item_metadata"},
            ),
            [],
        )

    def test_guarded_worker_executes_run_contract(self):
        code = """
def run(partial_items, candidate_items, source_paths, max_contexts_per_candidate=5):
    assert "item_metadata" in source_paths
    return [
        {
            "label": candidate["label"],
            "item_id": candidate["item_id"],
            "contexts": [
                {
                    "sources": ["item_metadata"],
                    "text": "context-" + candidate["label"],
                    "supporting_item_ids": [],
                    "supporting_bundle_ids": [],
                }
            ],
        }
        for candidate in candidate_items
    ]
"""
        partial = [{"item_id": 7, "text": "partial"}]
        candidates = [
            {"label": "A", "item_id": 1, "text": "one"},
            {"label": "B", "item_id": 2, "text": "two"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            source_path = os.path.join(directory, "item_info.json")
            with open(source_path, "w", encoding="utf-8") as handle:
                json.dump({}, handle)
            report = execute_strategy_program(
                code=code,
                strategy_id="S1",
                required_sources=["item_metadata"],
                partial_items=partial,
                candidate_items=candidates,
                all_source_paths={"item_metadata": source_path},
                case_dir=directory,
                conf={
                    "operator_program_timeout_seconds": 10,
                    "operator_program_max_contexts_per_candidate": 5,
                },
            )
        self.assertTrue(report["success"], report)
        self.assertEqual(
            [row["label"] for row in report["result"]],
            ["A", "B"],
        )

    def test_ranking_evaluation_and_aggregation(self):
        row = {
            "sample_idx": 0,
            "valid": True,
            "error": "",
            **evaluate_full_ranking(
                {
                    "prediction": "B",
                    "ranking": ["B", "A", "C"],
                    "rationale": "test",
                },
                "A",
            ),
        }
        summary = aggregate_prediction_rows([row])
        self.assertEqual(row["gt_rank"], 2)
        self.assertEqual(summary["hit_rate_at_1"], 0.0)
        self.assertEqual(summary["hit_rate_at_3"], 1.0)
        self.assertEqual(summary["mean_reciprocal_rank"], 0.5)


if __name__ == "__main__":
    unittest.main()

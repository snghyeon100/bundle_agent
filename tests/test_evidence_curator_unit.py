"""Unit tests for strategy curation and completion explanations."""

import os
import sys
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from operator_learning.evidence_curator import (
    candidate_completion_explanations,
    evidence_curator_prompt,
    normalize_evidence_curation,
    select_curated_strategy_evidence,
    validate_evidence_curation,
)


class EvidenceCuratorTest(unittest.TestCase):
    def setUp(self):
        self.labels = ["A", "B"]
        self.evidence = [
            {
                "strategy_id": "S1",
                "intent": "intent one",
                "reference_construction": "reference one",
                "candidate_relation": "relation one",
                "candidate_evidence": [
                    {
                        "label": "A",
                        "contexts": [
                            {
                                "text": "context A",
                                "sources": ["bundle_item_history"],
                                "supporting_item_ids": [1],
                                "supporting_bundle_ids": [10],
                            }
                        ],
                    },
                    {"label": "B", "contexts": []},
                ],
            },
            {
                "strategy_id": "S2",
                "intent": "intent two",
                "reference_construction": "reference two",
                "candidate_relation": "relation two",
                "candidate_evidence": [
                    {"label": "A", "contexts": []},
                    {
                        "label": "B",
                        "contexts": [
                            {
                                "text": "context B",
                                "sources": ["item_metadata"],
                                "supporting_item_ids": [2],
                                "supporting_bundle_ids": [],
                            }
                        ],
                    },
                ],
            },
            {
                "strategy_id": "S3",
                "intent": "empty intent",
                "reference_construction": "empty reference",
                "candidate_relation": "empty relation",
                "candidate_evidence": [
                    {"label": "A", "contexts": []},
                    {"label": "B", "contexts": []},
                ],
            },
        ]

    def test_valid_selection_keeps_complete_selected_strategy_results(self):
        curation = {
            "selected_strategies": ["S1", "S2"],
            "selection_reasons": {
                "S1": "useful relation one",
                "S2": "useful relation two",
            },
            "candidate_explanations": {
                "A": (
                    "This item could complete the bundle as an accessory. "
                    "A direct historical bundle relation was observed."
                ),
                "B": (
                    "This item could complete the bundle by adding another role. "
                    "Additional metadata evidence was observed."
                ),
            },
        }
        self.assertEqual(
            validate_evidence_curation(
                curation,
                strategy_evidence=self.evidence,
                candidate_labels=self.labels,
            ),
            [],
        )
        selected = select_curated_strategy_evidence(
            self.evidence,
            curation,
        )
        self.assertEqual(
            [strategy["strategy_id"] for strategy in selected],
            ["S1", "S2"],
        )
        self.assertEqual(
            selected[0]["candidate_evidence"][0]["contexts"][0]["text"],
            "context A",
        )
        self.assertEqual(
            candidate_completion_explanations(curation),
            [
                {
                    "label": "A",
                    "summary": (
                        "This item could complete the bundle as an accessory. "
                        "A direct historical bundle relation was observed."
                    ),
                },
                {
                    "label": "B",
                    "summary": (
                        "This item could complete the bundle by adding another "
                        "role. Additional metadata evidence was observed."
                    ),
                },
            ],
        )

    def test_rejects_empty_strategy_and_identifier_leakage(self):
        curation = {
            "selected_strategies": ["S3"],
            "selection_reasons": {"S3": "reason"},
            "candidate_explanations": {
                "A": "S3:A:0 supports this relation.",
                "B": "No useful source-grounded relationship was found.",
            },
        }
        issues = validate_evidence_curation(
            curation,
            strategy_evidence=self.evidence,
            candidate_labels=self.labels,
        )
        self.assertTrue(any("no executed evidence" in issue for issue in issues))
        self.assertTrue(any("must not mention" in issue for issue in issues))

    def test_normalizes_identifier_leakage_without_losing_explanation(self):
        curation = {
            "selected_strategies": ["S1"],
            "selection_reasons": {"S1": "reason"},
            "candidate_explanations": {
                "A": "The relation is direct (S1 evidence concerns the candidate).",
                "B": "No additional source evidence was found.",
            },
        }
        normalized = normalize_evidence_curation(curation)
        self.assertNotIn("S1", normalized["candidate_explanations"]["A"])
        self.assertIn(
            "the selected evidence concerns the candidate",
            normalized["candidate_explanations"]["A"],
        )
        self.assertEqual(
            validate_evidence_curation(
                normalized,
                strategy_evidence=self.evidence,
                candidate_labels=self.labels,
            ),
            [],
        )

    def test_malformed_selected_strategy_type_returns_issue(self):
        curation = {
            "selected_strategies": [{"bad": "type"}],
            "selection_reasons": {},
            "candidate_explanations": {
                "A": "No useful source-grounded relationship was found.",
                "B": "No useful source-grounded relationship was found.",
            },
        }
        issues = validate_evidence_curation(
            curation,
            strategy_evidence=self.evidence,
            candidate_labels=self.labels,
        )
        self.assertTrue(any("only strings" in issue for issue in issues))

    def test_prompt_strips_internal_entity_and_evidence_ids(self):
        prompt = evidence_curator_prompt(
            partial_items=[{"item_id": 9, "text": "partial"}],
            candidate_items=[
                {"label": "A", "item_id": 1, "text": "a"},
                {"label": "B", "item_id": 2, "text": "b"},
            ],
            strategy_specs=[
                {
                    "strategy_id": "S1",
                    "intent": "intent",
                    "description": "declared strategy",
                    "reference_construction": "reference",
                    "candidate_relation": "relation",
                    "evidence_route": ["route"],
                    "required_sources": ["item_metadata"],
                    "pseudocode": ["step"],
                }
            ],
            strategy_evidence=self.evidence[:1],
        )
        self.assertIn("not to choose the missing item or rank candidates", prompt)
        self.assertIn('"strategy_specifications"', prompt)
        self.assertIn('"program_results"', prompt)
        self.assertIn('"candidate_explanations"', prompt)
        self.assertNotIn('"evidence_id"', prompt)
        self.assertNotIn('"supporting_item_ids"', prompt)
        self.assertNotIn('"supporting_bundle_ids"', prompt)
        self.assertNotIn('"item_id"', prompt)
        self.assertNotIn("ground_truth", prompt)


if __name__ == "__main__":
    unittest.main()

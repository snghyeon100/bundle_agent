import os
import sys
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from operator_learning.evidence_summary import (
    candidate_evidence_summary_prompt,
    candidate_summary_prediction_prompt,
    validate_candidate_summaries,
)


PARTIAL_ITEMS = [{"item_id": 1, "text": "partial one"}]
CANDIDATE_ITEMS = [
    {"label": "A", "item_id": 2, "text": "candidate a"},
    {"label": "B", "item_id": 3, "text": "candidate b"},
]
STRATEGY_EVIDENCE = [
    {
        "strategy_id": "S1",
        "intent": "one intent",
        "reference_construction": "must not be rendered",
        "candidate_relation": "historical co-bundle relation",
        "candidate_evidence": [
            {
                "label": "A",
                "contexts": [
                    {
                        "sources": ["bundle_item_history"],
                        "text": "bundle composition",
                    }
                ],
            },
            {"label": "B", "contexts": []},
        ],
    }
]


class EvidenceSummaryTest(unittest.TestCase):
    def test_summary_prompt_keeps_candidate_evidence_but_drops_reference(self):
        prompt = candidate_evidence_summary_prompt(
            partial_items=PARTIAL_ITEMS,
            candidate_items=CANDIDATE_ITEMS,
            strategy_evidence=STRATEGY_EVIDENCE,
        )
        self.assertIn("candidate a", prompt)
        self.assertIn("bundle composition", prompt)
        self.assertIn("historical co-bundle relation", prompt)
        self.assertNotIn("must not be rendered", prompt)

    def test_summary_validation_requires_complete_ordered_labels(self):
        valid = {
            "candidate_summaries": [
                {"label": "A", "summary": "evidence for A"},
                {"label": "B", "summary": "no evidence found"},
            ]
        }
        self.assertEqual(
            validate_candidate_summaries(valid, ["A", "B"]),
            [],
        )
        reversed_value = {
            "candidate_summaries": list(
                reversed(valid["candidate_summaries"])
            )
        }
        self.assertTrue(
            validate_candidate_summaries(reversed_value, ["A", "B"])
        )

    def test_prediction_prompt_places_summary_beside_candidate(self):
        prompt = candidate_summary_prediction_prompt(
            dataset="pog_dense",
            partial_items=PARTIAL_ITEMS,
            candidate_items=CANDIDATE_ITEMS,
            candidate_summaries=[
                {"label": "A", "summary": "summary A"},
                {"label": "B", "summary": "summary B"},
            ],
        )
        self.assertIn("[A]\nItem: candidate a", prompt)
        self.assertIn("Source-grounded evidence summary: summary A", prompt)
        self.assertLess(prompt.index("candidate a"), prompt.index("summary A"))

    def test_prediction_prompt_can_render_curator_interpretations(self):
        prompt = candidate_summary_prediction_prompt(
            dataset="pog",
            partial_items=PARTIAL_ITEMS,
            candidate_items=CANDIDATE_ITEMS,
            candidate_summaries=[
                {"label": "A", "summary": "interpretation A"},
                {"label": "B", "summary": "interpretation B"},
            ],
            evidence_mode="interpretation",
        )
        self.assertIn("Evidence interpretation: interpretation A", prompt)
        self.assertNotIn("Source-grounded evidence summary:", prompt)

    def test_prediction_prompt_can_render_completion_explanations(self):
        prompt = candidate_summary_prediction_prompt(
            dataset="pog",
            partial_items=PARTIAL_ITEMS,
            candidate_items=CANDIDATE_ITEMS,
            candidate_summaries=[
                {"label": "A", "summary": "completion explanation A"},
                {"label": "B", "summary": "completion explanation B"},
            ],
            evidence_mode="completion_explanation",
        )
        self.assertIn(
            "Bundle-completion explanation: completion explanation A",
            prompt,
        )
        self.assertIn(
            "CANDIDATES WITH BUNDLE-COMPLETION EXPLANATIONS",
            prompt,
        )


if __name__ == "__main__":
    unittest.main()

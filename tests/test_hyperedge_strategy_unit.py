"""Unit tests for the isolated hyperedge strategy prompt variant."""

import os
import sys
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from hyperedge_strategy.prompts import induction_prompt as hyperedge_prompt
from operator_learning.prompts import induction_prompt as base_prompt


CASE = {
    "dataset": "pog",
    "partial_items": [
        {"item_id": 1, "text": "partial one", "metadata": {"cate": 10}},
        {"item_id": 2, "text": "partial two", "metadata": {"cate": 20}},
    ],
    "candidate_items": [
        {
            "label": "A",
            "item_id": 3,
            "text": "candidate",
            "metadata": {"cate": 30},
        }
    ],
    "source_diagnostics": {
        "bundle_item_history": {
            "availability": "available",
            "partial_coverage": "full",
        },
        "item_metadata": {
            "availability": "available",
            "partial_coverage": "full",
        },
    },
}

SOURCES = {
    "dataset": "pog",
    "components": [
        {
            "id": "bundle_item_history",
            "format": {"path": "data/bi_train.txt"},
        },
        {
            "id": "item_metadata",
            "format": {"path": "data/item_info.json"},
        },
    ],
}


class HyperedgeStrategyPromptTest(unittest.TestCase):
    def test_variant_preserves_base_and_adds_hyperedge_requirements(self):
        base = base_prompt(CASE, SOURCES, [], 3, text_only=False)
        variant = hyperedge_prompt(CASE, SOURCES, [], 3, text_only=False)

        self.assertNotIn("Treat a bundle as one hyperedge", base)
        self.assertIn("Treat a bundle as one hyperedge", variant)
        self.assertIn("Hi = P union {ci}", variant)
        self.assertIn(
            "examines every candidate-augmented hyperedge Hi as a whole",
            variant,
        )
        self.assertNotIn("HYPEREDGE STRATEGY REQUIREMENTS", variant)
        self.assertNotIn("reusable", variant.lower())
        self.assertNotIn("must declare and read bundle_item_history", variant)
        self.assertIn(
            "Every final context must be either a related item text",
            variant,
        )
        self.assertLess(
            variant.index("Treat a bundle as one hyperedge"),
            variant.index("Infer exactly three distinct"),
        )

    def test_variant_keeps_current_spec_first_contract(self):
        prompt = hyperedge_prompt(CASE, SOURCES, [], 3, text_only=False)

        self.assertIn("strategy_specs", prompt)
        self.assertIn("programs", prompt)
        self.assertIn("def run(partial_items, candidate_items, source_paths", prompt)
        self.assertIn("Use strategy IDs S1, S2, and S3 exactly once", prompt)
        self.assertIn('"supporting_bundle_ids": []', prompt)


if __name__ == "__main__":
    unittest.main()

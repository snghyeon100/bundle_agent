"""Unit tests for code-free macro-strategy induction diagnostics."""

import os
import sys
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from online_hypothesis_program.strategy_diagnostic import (
    STRATEGY_SCHEMA_VERSION,
    strategy_generation_prompt,
    validate_strategy_result,
)


PARTIAL_ITEMS = [
    {"item_id": 11, "text": "observed item one"},
    {"item_id": 17, "text": "observed item two"},
]
MANIFEST = {
    "components": [
        {
            "id": "item_metadata",
            "meaning": "item-side semantic metadata",
            "runtime_format": "mapping from item_id to metadata",
            "observed_field_schema": {"title": ["string"]},
        },
        {
            "id": "bundle_item_history",
            "meaning": "historical bundle-item membership",
            "runtime_format": "forward and reverse bundle/item mappings",
        },
        {
            "id": "user_item_history",
            "meaning": "historical user-item interactions",
            "runtime_format": "forward and reverse user/item mappings",
        },
    ]
}


def valid_result():
    strategies = []
    specifications = [
        (
            "S1",
            "The observed items form a recurring composition whose omitted member "
            "is stable across related historical bundles.",
            ["bundle_item_history", "item_metadata"],
        ),
        (
            "S2",
            "The observed items bridge user communities and the completion is an "
            "item preferred by the shared but otherwise dissimilar users.",
            ["user_item_history", "item_metadata"],
        ),
        (
            "S3",
            "The observed items occupy contrasting semantic roles and the missing "
            "item should connect their respective historical neighborhoods.",
            ["bundle_item_history", "user_item_history"],
        ),
    ]
    for strategy_id, hypothesis, sources in specifications:
        strategies.append(
            {
                "id": strategy_id,
                "hypothesis": hypothesis,
                "partial_roles": [
                    {"item_id": 11, "role": f"{strategy_id} anchor"},
                    {"item_id": 17, "role": f"{strategy_id} counterpart"},
                ],
                "required_sources": sources,
                "pseudocode": [
                    f"1. Build a {strategy_id}-specific reference for both roles.",
                    "2. Expand that reference through the selected historical relation.",
                    "3. Intersect and contrast the expanded neighborhoods.",
                    "4. Retrieve items supported by the resulting completion relation.",
                ],
                "retrieval_target": "corpus items satisfying this completion relation",
                "provenance_plan": "retain the intermediate records linking each result",
            }
        )
    return {
        "schema_version": STRATEGY_SCHEMA_VERSION,
        "strategies": strategies,
    }


class StrategyDiagnosticTest(unittest.TestCase):
    def test_prompt_is_code_free_and_exposes_partial_and_runtime_sources(self):
        prompt = strategy_generation_prompt(
            dataset="pog_dense",
            partial_items=PARTIAL_ITEMS,
            workspace_manifest=MANIFEST,
            strategy_count=3,
        )

        self.assertIn('"item_id": 11', prompt)
        self.assertIn("observed item one", prompt)
        self.assertIn("Produce exactly 3 genuinely different interpretations", prompt)
        self.assertIn('"runtime_format"', prompt)
        self.assertIn('"observed_field_schema"', prompt)
        self.assertIn("3 to 8 causally dependent pseudocode steps", prompt)
        self.assertIn("Return structured pseudocode, not Python code", prompt)
        self.assertNotIn("def retrieve(", prompt)

    def test_valid_three_strategy_result_is_accepted(self):
        issues = validate_strategy_result(
            valid_result(),
            available_sources=[
                "item_metadata",
                "bundle_item_history",
                "user_item_history",
            ],
            strategy_count=3,
            partial_item_ids=[11, 17],
        )
        self.assertEqual(issues, [])

    def test_rejects_wrong_count_unknown_source_and_incomplete_roles(self):
        result = valid_result()
        result["strategies"] = result["strategies"][:2]
        result["strategies"][0]["required_sources"] = ["invented_source"]
        result["strategies"][1]["partial_roles"] = [
            {"item_id": 11, "role": "only one partial item"}
        ]

        issues = validate_strategy_result(
            result,
            available_sources=[
                "item_metadata",
                "bundle_item_history",
                "user_item_history",
            ],
            strategy_count=3,
            partial_item_ids=[11, 17],
        )
        joined = " | ".join(issues)
        self.assertIn("exactly 3 entries", joined)
        self.assertIn("unavailable sources", joined)
        self.assertIn("cover exactly the partial item IDs", joined)

    def test_rejects_duplicate_or_python_pseudocode(self):
        result = valid_result()
        result["strategies"][1]["hypothesis"] = result["strategies"][0][
            "hypothesis"
        ]
        result["strategies"][2]["pseudocode"][1] = (
            "def retrieve(partial_item_ids): return []"
        )

        issues = validate_strategy_result(
            result,
            available_sources=[
                "item_metadata",
                "bundle_item_history",
                "user_item_history",
            ],
            strategy_count=3,
            partial_item_ids=[11, 17],
        )
        joined = " | ".join(issues)
        self.assertIn("hypotheses must be distinct", joined)
        self.assertIn("must not contain Python code", joined)


if __name__ == "__main__":
    unittest.main()

"""Unit tests for offline reusable-program compilation."""

import os
import sys
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from operator_learning.prompts import compilation_prompt
from operator_learning.runtime import (
    assert_implementation_unchanged,
    make_compiled_program,
    validate_compilation_result,
)
from operator_learning.schemas import CANDIDATE_PROPOSAL_OUTPUT_CONTRACT


OPERATOR = {
    "name": "RetrieveHistoricalCompanions",
    "hypothesis": "Related historical bundles contain plausible missing items.",
    "required_sources": ["bundle_item_history"],
    "applicability": ["partial items have historical bundle coverage"],
    "evidence_types": ["historical_bundle_context"],
    "pseudocode": [
        "retrieve bundles containing partial items",
        "collect non-partial items",
        "select bounded candidate proposals",
        "return representative bundle provenance",
    ],
    "output_contract": CANDIDATE_PROPOSAL_OUTPUT_CONTRACT,
    "operator_id": "program_1",
}

SOURCE_CAPABILITIES = {
    "dataset": "pog_dense",
    "components": [
        {
            "id": "bundle_item_history",
            "format": {"path": "data/bi_train.txt"},
        },
        {
            "id": "item_content_embedding",
            "format": {"path": "data/content_feature.pt"},
        },
    ],
}

CODE = '''
def execute(partial_item_ids, source_api, candidate_budget, evidence_budget):
    return {
        "schema_version": "candidate_proposal_set_v1",
        "program_id": "program_1",
        "hypothesis": "Related historical bundles contain plausible missing items.",
        "candidate_proposals": [],
        "evidence_records": [],
        "execution_trace": {
            "used_sources": [],
            "candidate_budget": candidate_budget,
            "evidence_budget": evidence_budget,
        },
    }
'''.strip()


class OfflineCompilationTest(unittest.TestCase):
    def test_prompt_is_case_free_source_scoped_and_program_specific(self):
        prompt = compilation_prompt(OPERATOR, SOURCE_CAPABILITIES)

        self.assertIn("offline compiler", prompt)
        self.assertIn("def execute(partial_item_ids", prompt)
        self.assertIn("candidate_proposal_set_v1", prompt)
        self.assertIn("bundle_item_history", prompt)
        self.assertNotIn("item_content_embedding", prompt)
        self.assertNotIn("ground truth", prompt.lower().split("not available")[1])
        self.assertNotIn("TYPED OPERATOR GRAPH", prompt)
        self.assertNotIn("SELECTED_WORKFLOW", prompt)

    def test_compilation_response_requires_exact_program_name(self):
        issues = validate_compilation_result(
            {"program_name": OPERATOR["name"], "code": CODE},
            OPERATOR,
        )
        self.assertEqual(issues, [])

        issues = validate_compilation_result(
            {"program_name": "DifferentProgram", "code": CODE},
            OPERATOR,
        )
        self.assertIn("program_name must exactly match", " | ".join(issues))

    def test_code_hash_detects_post_validation_mutation(self):
        compiled = make_compiled_program(OPERATOR, CODE)
        self.assertEqual(
            assert_implementation_unchanged(compiled),
            compiled["implementation"]["sha256"],
        )
        compiled["implementation"]["code"] += "\n# changed"
        with self.assertRaisesRegex(ValueError, "code hash changed"):
            assert_implementation_unchanged(compiled)


if __name__ == "__main__":
    unittest.main()

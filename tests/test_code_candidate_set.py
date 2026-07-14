import os
import sys
import unittest
from unittest.mock import AsyncMock, patch


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from code.pipeline import run_code_agent, validate_adaptive_item_evidence
from code.prompts import code_generation_prompt, decision_prompt


CASE_VIEW = {
    "case_id": "bundle_1",
    "dataset": "pog",
    "bundle_id": 1,
    "partial_item_ids": [20, 10],
    "candidates": [
        {"label": "A", "item_id": 30},
        {"label": "B", "item_id": 40},
    ],
}


def valid_evidence():
    return {
        "schema_version": "adaptive_item_evidence_v1",
        "strategy": {
            "name": "season_and_use_context",
            "description": (
                "Extract season and use-context observations for every item because those "
                "properties distinguish the visible candidates in this problem."
            ),
        },
        "partial_evidence": {
            "partial_20": {
                "item_id": 20,
                "evidence": ["description_feature.pt: neighbors -> casual shirt context"],
            },
            "partial_10": {
                "item_id": 10,
                "evidence": ["description_feature.pt: neighbors -> everyday pants context"],
            },
        },
        "candidate_evidence": {
            "A": {
                "item_id": 30,
                "evidence": ["description_feature.pt: neighbors -> casual footwear context"],
            },
            "B": {
                "item_id": 40,
                "evidence": ["description_feature.pt: neighbors -> winter accessory context"],
            },
        },
    }


class CandidateSetEvidenceTests(unittest.TestCase):
    def test_valid_schema_is_accepted(self):
        self.assertEqual(validate_adaptive_item_evidence(valid_evidence(), CASE_VIEW), [])

    def test_extra_strategy_field_is_rejected(self):
        evidence = valid_evidence()
        evidence["strategy"]["data_sources"] = ["bi_train.txt"]
        issues = validate_adaptive_item_evidence(evidence, CASE_VIEW)
        self.assertIn("strategy must contain exactly name and description", issues)

    def test_missing_strategy_is_rejected(self):
        evidence = valid_evidence()
        del evidence["strategy"]
        issues = validate_adaptive_item_evidence(evidence, CASE_VIEW)
        self.assertIn("strategy must be an object", issues)

    def test_missing_strategy_description_is_rejected(self):
        evidence = valid_evidence()
        del evidence["strategy"]["description"]
        issues = validate_adaptive_item_evidence(evidence, CASE_VIEW)
        self.assertIn("strategy.description must be non-empty", issues)

    def test_all_sparse_evidence_is_accepted(self):
        evidence = valid_evidence()
        payloads = list(evidence["partial_evidence"].values()) + list(
            evidence["candidate_evidence"].values()
        )
        for payload in payloads:
            payload["evidence"] = ["SPARSE: no supporting context"]
        self.assertEqual(validate_adaptive_item_evidence(evidence, CASE_VIEW), [])

    def test_wrong_partial_item_id_is_rejected(self):
        evidence = valid_evidence()
        evidence["partial_evidence"]["partial_20"]["item_id"] = 999
        issues = validate_adaptive_item_evidence(evidence, CASE_VIEW)
        self.assertIn("partial partial_20 item ID mismatch", issues)

    def test_empty_candidate_evidence_is_rejected(self):
        evidence = valid_evidence()
        evidence["candidate_evidence"]["A"]["evidence"] = []
        issues = validate_adaptive_item_evidence(evidence, CASE_VIEW)
        self.assertIn("candidate A evidence must be a non-empty string list", issues)

    def test_prompts_use_one_strategy_for_per_item_evidence(self):
        semantic_case = {
            "dataset": "pog",
            "partial_items": [
                {"item_id": 20, "text": "shirt"},
                {"item_id": 10, "text": "pants"},
            ],
            "candidates": [
                {"label": "A", "item_id": 30, "text": "shoe"},
                {"label": "B", "item_id": 40, "text": "hat"},
            ],
        }
        generation = code_generation_prompt(
            CASE_VIEW,
            {"sources": []},
            "output/evidence.json",
            semantic_case=semantic_case,
        )
        self.assertIn("adaptive_item_evidence_v1", generation)
        self.assertIn(
            "useful for determining which candidate item most appropriately completes the partial bundle",
            generation,
        )
        self.assertIn("extracts source-derived contextual observations", generation)
        self.assertIn("exactly one instance-adaptive contextual-evidence strategy", generation)
        self.assertIn("IB x BI: item -> bundles containing the item", generation)
        self.assertIn("IU x UI: item -> users interacting with the item", generation)
        self.assertIn("BI x IB: for a target item", generation)
        self.assertIn(
            "Before committing to a strategy, determine which feasible source relation is expected to provide",
            generation,
        )
        self.assertIn(
            "the most informative and non-sparse context across the current partial and candidate items",
            generation,
        )
        self.assertIn("Then implement exactly one strategy", generation)
        self.assertIn("These are examples only", generation)
        self.assertIn("Adapt one of them or invent exactly one strategy", generation)
        self.assertNotIn("Select, adapt, or invent", generation)
        self.assertIn("Apply the strategy independently and consistently", generation)
        self.assertIn("every partial item and every candidate item", generation)
        self.assertIn("retrieve context from source records", generation)
        self.assertIn("concrete external records or contextual patterns", generation)
        self.assertIn("report their available text or title rather than their item IDs", generation)
        self.assertIn("Item IDs may be used internally for source lookup", generation)
        self.assertIn("should not be used as the final contextual evidence", generation)
        self.assertNotIn("own metadata or feature values are not contextual evidence", generation)
        self.assertNotIn("do not report similarity scores, vector values, or distances", generation)
        self.assertIn("Do not directly compare candidates with partial items", generation)
        self.assertIn("construct a partial-item aggregate", generation)
        self.assertIn("Read at least one listed source at runtime", generation)
        self.assertNotIn("item or bundle context", generation)
        self.assertNotIn("P union {c}", generation)
        self.assertNotIn("exclude the current evaluation bundle", generation)
        self.assertNotIn("Useful evidence may involve", generation)
        self.assertIn('"strategy":', generation)
        self.assertIn('"partial_evidence":', generation)
        self.assertIn('"candidate_evidence":', generation)
        self.assertNotIn("bundle_composition_plan", generation)
        self.assertNotIn("selected_bundle_question", generation)
        self.assertNotIn("operator_steps", generation)
        self.assertNotIn("HIGH-LEVEL SCRIPT SKELETON", generation)

        prediction = decision_prompt(semantic_case, valid_evidence())
        self.assertIn("Instance-adaptive strategy: season_and_use_context", prediction)
        self.assertIn("1. shirt\nEvidence:", prediction)
        self.assertIn("A. shoe\nEvidence:", prediction)
        self.assertNotIn("partial bundle plus option A", prediction)


class FailClosedPipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_stage1_failure_skips_prediction(self):
        sample = {
            "bundle_id": 1,
            "input_indices": [20, 10],
            "candidate_indices": [30, 40],
        }
        decision_case = {
            "case_id": "bundle_1",
            "dataset": "pog",
            "bundle_id": 1,
            "partial_items": [],
            "candidates": [],
        }
        inputs = {
            "case_view": CASE_VIEW,
            "workspace": {"workspace_dir": "workspace", "files": []},
            "source_manifest": {},
            "decision_case": decision_case,
            "evidence_output_file": "output/evidence.json",
            "prompt": "generate code",
        }
        failed_result = {
            "raw_response": "bad code",
            "generated_code": "bad code",
            "execution_summary": {"returncode": 1},
            "validation_issues": ["execution failed"],
            "accepted_evidence": None,
        }
        prediction_call = AsyncMock(return_value="A")

        with patch("code.pipeline.build_code_generation_inputs", return_value=inputs), patch(
            "code.pipeline.generate_code_evidence_once",
            new=AsyncMock(return_value=failed_result),
        ):
            row, prediction, raw = await run_code_agent(
                sample,
                {},
                {"code_generation": object(), "prediction": object()},
                prediction_call,
                lambda value: value,
            )

        self.assertEqual(prediction, "ERR_CODE")
        self.assertIn("ERR_CODE", raw)
        self.assertEqual(row["code_stage1_status"], "failed")
        self.assertFalse(row["code_evidence_accepted"])
        self.assertNotIn("code_prediction_prompt", row)
        prediction_call.assert_not_awaited()

    async def test_accepted_stage1_runs_prediction(self):
        sample = {
            "bundle_id": 1,
            "input_indices": [20, 10],
            "candidate_indices": [30, 40],
        }
        decision_case = {
            "case_id": "bundle_1",
            "dataset": "pog",
            "bundle_id": 1,
            "partial_items": [
                {"item_id": 20, "text": "shirt"},
                {"item_id": 10, "text": "pants"},
            ],
            "candidates": [
                {"label": "A", "item_id": 30, "text": "shoe"},
                {"label": "B", "item_id": 40, "text": "hat"},
            ],
        }
        inputs = {
            "case_view": CASE_VIEW,
            "workspace": {"workspace_dir": "workspace", "files": []},
            "source_manifest": {},
            "decision_case": decision_case,
            "evidence_output_file": "output/evidence.json",
            "prompt": "generate code",
        }
        accepted_result = {
            "raw_response": "working code",
            "generated_code": "working code",
            "execution_summary": {"returncode": 0},
            "validation_issues": [],
            "accepted_evidence": valid_evidence(),
        }
        prediction_call = AsyncMock(return_value="A")

        with patch("code.pipeline.build_code_generation_inputs", return_value=inputs), patch(
            "code.pipeline.generate_code_evidence_once",
            new=AsyncMock(return_value=accepted_result),
        ):
            row, prediction, raw = await run_code_agent(
                sample,
                {},
                {
                    "code_generation": {"model": "test-model"},
                    "prediction": {"model": "test-model"},
                },
                prediction_call,
                lambda value: value.strip(),
            )

        self.assertEqual(prediction, "A")
        self.assertEqual(raw, "A")
        self.assertEqual(row["code_stage1_status"], "accepted")
        self.assertTrue(row["code_evidence_accepted"])
        self.assertIn("1. shirt\nEvidence:", row["code_prediction_prompt"])
        self.assertIn("A. shoe\nEvidence:", row["code_prediction_prompt"])
        prediction_call.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()

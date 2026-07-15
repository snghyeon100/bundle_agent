import os
import sys
import unittest
from unittest.mock import AsyncMock, patch


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from code.pipeline import run_code_agent, validate_adaptive_bundle_evidence
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
        "schema_version": "adaptive_bundle_evidence_v2",
        "intent": "A casual warm-weather outfit with complementary everyday pieces.",
        "strategy": {
            "name": "shared outfit neighborhood",
            "description": (
                "Use item-to-outfit-to-item relations to retrieve shared warm-weather outfit "
                "records for the partial bundle and the same contextual records for each candidate."
            ),
        },
        "partial_bundle_evidence": {
            "evidence": ["bi_train.txt: shared outfits -> casual shirt and pants context"],
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
        self.assertEqual(validate_adaptive_bundle_evidence(valid_evidence(), CASE_VIEW), [])

    def test_extra_strategy_field_is_rejected(self):
        evidence = valid_evidence()
        evidence["strategy"]["data_sources"] = ["bi_train.txt"]
        issues = validate_adaptive_bundle_evidence(evidence, CASE_VIEW)
        self.assertIn("strategy must contain exactly name and description", issues)

    def test_missing_intent_is_rejected(self):
        evidence = valid_evidence()
        del evidence["intent"]
        issues = validate_adaptive_bundle_evidence(evidence, CASE_VIEW)
        self.assertIn("intent must be a non-empty string", issues)

    def test_missing_strategy_is_rejected(self):
        evidence = valid_evidence()
        del evidence["strategy"]
        issues = validate_adaptive_bundle_evidence(evidence, CASE_VIEW)
        self.assertIn("strategy must be an object", issues)

    def test_missing_strategy_description_is_rejected(self):
        evidence = valid_evidence()
        del evidence["strategy"]["description"]
        issues = validate_adaptive_bundle_evidence(evidence, CASE_VIEW)
        self.assertIn("strategy.description must be non-empty", issues)

    def test_empty_evidence_lists_are_accepted(self):
        evidence = valid_evidence()
        evidence["partial_bundle_evidence"]["evidence"] = []
        for payload in evidence["candidate_evidence"].values():
            payload["evidence"] = []
        self.assertEqual(validate_adaptive_bundle_evidence(evidence, CASE_VIEW), [])

    def test_more_than_five_bundle_evidence_items_are_rejected(self):
        evidence = valid_evidence()
        evidence["partial_bundle_evidence"]["evidence"] = [f"record {i}" for i in range(6)]
        issues = validate_adaptive_bundle_evidence(evidence, CASE_VIEW)
        self.assertIn(
            "partial_bundle_evidence.evidence must contain at most 5 items",
            issues,
        )

    def test_more_than_five_candidate_evidence_items_are_rejected(self):
        evidence = valid_evidence()
        evidence["candidate_evidence"]["A"]["evidence"] = [f"record {i}" for i in range(6)]
        issues = validate_adaptive_bundle_evidence(evidence, CASE_VIEW)
        self.assertIn("candidate A evidence must contain at most 5 items", issues)

    def test_wrong_candidate_item_id_is_rejected(self):
        evidence = valid_evidence()
        evidence["candidate_evidence"]["A"]["item_id"] = 999
        issues = validate_adaptive_bundle_evidence(evidence, CASE_VIEW)
        self.assertIn("candidate A item ID mismatch", issues)

    def test_non_string_candidate_evidence_is_rejected(self):
        evidence = valid_evidence()
        evidence["candidate_evidence"]["A"]["evidence"] = [{"record": "not allowed"}]
        issues = validate_adaptive_bundle_evidence(evidence, CASE_VIEW)
        self.assertIn("candidate A evidence must be a string list", issues)

    def test_prompt_uses_minimal_high_level_intent_to_strategy_guidance(self):
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
        self.assertIn("adaptive_bundle_evidence_v2", generation)
        self.assertIn("Before writing code, inspect only the partial items", generation)
        self.assertIn("infer one concise intent", generation)
        self.assertIn("Do not use the candidates to infer or revise the intent", generation)
        self.assertIn("design exactly one evidence-retrieval code strategy", generation)
        self.assertIn("for this specific bundle-completion task", generation)
        self.assertIn("reflects the distinctive characteristics expressed by the intent", generation)
        self.assertIn("Choose the sources, relations, and retrieval operations yourself", generation)
        self.assertIn("an instance-specific multi-step strategy", generation)
        self.assertIn("rather than a single generic lookup", generation)
        self.assertIn("would be unchanged for another bundle", generation)
        self.assertIn("Do not implement a fallback strategy", generation)
        self.assertIn("Minimal example of Partial Items -> Intent -> Strategy", generation)
        self.assertIn("A waterproof shell jacket, flexible utility pants, and trail shoes", generation)
        self.assertIn("A functional, weather-ready outdoor outfit designed for mobility", generation)
        self.assertIn("weather protection, mobility, and outdoor use", generation)
        self.assertIn("rather than reusing the outdoor attributes or steps above", generation)
        self.assertIn("partial bundle as a whole", generation)
        self.assertIn("derive a reusable completion pattern", generation)
        self.assertIn("apply the same fixed strategy to every candidate", generation)
        self.assertIn("connects it to both the actual partial items and the derived completion pattern", generation)
        self.assertIn("direct or may use a source-grounded multi-step relation", generation)
        self.assertIn("must consume the actual partial-bundle context", generation)
        self.assertIn("independently recreating search criteria is insufficient", generation)
        self.assertIn("Do not score, rank, select, or predict candidates", generation)
        self.assertIn("Evidence is not a description or score of the target item itself", generation)
        self.assertIn("a concrete source-grounded item or bundle record", generation)
        self.assertIn("intent-specific completion pattern established from the current partial bundle", generation)
        self.assertNotIn("semantic properties X and Y", generation)
        self.assertNotIn("the bundle's distinctive semantic signature", generation)
        self.assertNotIn("Spotify-specific conceptual guidance", generation)
        self.assertNotIn("POG-specific high-level intent-to-strategy requirement", generation)
        self.assertIn("concrete external records or contextual patterns", generation)
        self.assertIn("report their available text or title rather than only item IDs", generation)
        self.assertIn("Item IDs may be used internally for lookup", generation)
        self.assertIn("emit an empty evidence list instead of inventing evidence", generation)
        self.assertNotIn("Evidence arrays are support-only", generation)
        self.assertNotIn("diagnostics, not evidence", generation)
        self.assertIn("between zero and 5 strings", generation)
        self.assertIn("most semantically relevant to the inferred intent", generation)
        self.assertIn("Do not select evidence merely by item ID, bundle ID, source order", generation)
        self.assertIn("retrieve_partial_bundle_context(partial_items, sources)", generation)
        self.assertIn(
            "retrieve_candidate_evidence(candidate, partial_items, partial_bundle_context, sources)",
            generation,
        )
        self.assertIn("actually use partial_bundle_context", generation)
        self.assertIn("connect this candidate to the actual partial items", generation)
        self.assertIn("demonstrate the derived completion_pattern", generation)
        self.assertIn(
            "candidate, PARTIAL_ITEMS, partial_bundle_context, sources",
            generation,
        )
        self.assertIn('partial_bundle_context["evidence"]', generation)
        self.assertIn("completion_pattern is internal", generation)
        self.assertNotIn("derive_completion_patterns(partial_items, sources)", generation)
        self.assertIn("read at least one listed source at runtime", generation)
        self.assertIn('"intent":', generation)
        self.assertIn('"strategy":', generation)
        self.assertIn('"partial_bundle_evidence":', generation)
        self.assertIn('"candidate_evidence":', generation)

        prediction = decision_prompt(semantic_case, valid_evidence())
        self.assertIn("Inferred bundle intent: A casual warm-weather outfit", prediction)
        self.assertIn("Retrieval strategy: shared outfit neighborhood", prediction)
        self.assertIn("Partial-bundle evidence:", prediction)
        self.assertIn("1. shirt\n2. pants", prediction)
        self.assertIn("A. shoe\nEvidence:", prediction)

    def test_spotify_prompt_uses_same_minimal_high_level_guidance(self):
        spotify_case = {**CASE_VIEW, "dataset": "spotify"}
        generation = code_generation_prompt(
            spotify_case,
            {"sources": []},
            "output/evidence.json",
            semantic_case={"partial_items": [], "candidates": []},
        )
        self.assertIn("Before writing code, inspect only the partial items", generation)
        self.assertIn("design exactly one evidence-retrieval code strategy", generation)
        self.assertIn("Several upbeat synth-pop tracks", generation)
        self.assertIn("An energetic retro synth-pop playlist for nighttime driving", generation)
        self.assertIn("era, genre, energy, and listening context", generation)
        self.assertNotIn("A waterproof shell jacket", generation)


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
        self.assertIn("1. shirt\n2. pants", row["code_prediction_prompt"])
        self.assertIn("Inferred bundle intent:", row["code_prediction_prompt"])
        self.assertIn("Partial-bundle evidence:", row["code_prediction_prompt"])
        self.assertIn("A. shoe\nEvidence:", row["code_prediction_prompt"])
        prediction_call.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()

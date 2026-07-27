"""Unit tests for the two-call online hypothesis-program pipeline."""

import json
import os
import sys
import tempfile
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from online_hypothesis_program.pipeline import run_online_hypothesis_program
from online_hypothesis_program.runtime import (
    execute_code_in_process,
    execute_program_subprocess,
)
from online_hypothesis_program.schemas import DISCOVERY_SCHEMA_VERSION


PROGRAM_CODE = """
def execute(partial_item_ids, source_api, candidate_budget, evidence_budget):
    return {
        "candidate_proposals": [],
        "evidence_records": [],
        "used_sources": ["bundle_item_history"],
    }
""".strip()


class FakeSourceAPI:
    available_sources = ("bundle_item_history",)


def discovery_response():
    return {
        "schema_version": DISCOVERY_SCHEMA_VERSION,
        "hypotheses": [
            {
                "id": "H1",
                "observed_cues": ["soft knit texture", "neutral color"],
                "intent": "The bundle may form a soft coordinated layered outfit.",
                "missing_role": "A complementary lower or outer layer.",
            }
        ],
        "programs": [
            {
                "hypothesis_id": "H1",
                "program_id": "P1",
                "name": "LayeredContextSearch",
                "required_sources": ["bundle_item_history"],
                "evidence_types": ["historical_bundle_context"],
                "code": PROGRAM_CODE,
            }
        ],
    }


def make_dataset(root):
    data_dir = os.path.join(root, "pog_dense")
    os.makedirs(data_dir, exist_ok=True)
    with open(os.path.join(data_dir, "count.json"), "w", encoding="utf-8") as handle:
        json.dump({"#B": 20, "#I": 6, "#U": 10}, handle)
    item_info = {
        "1": {"title": "cream knit top", "cate_id": "c1"},
        "2": {"title": "soft neutral scarf", "cate_id": "c2"},
        "3": {"title": "wide-leg wool trousers", "cate_id": "c3"},
        "4": {"title": "sport sandals", "cate_id": "c4"},
        "5": {"title": "layered wool cardigan", "cate_id": "c5"},
    }
    with open(
        os.path.join(data_dir, "item_info.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(item_info, handle)
    with open(
        os.path.join(data_dir, "bi_train.txt"),
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write("10, 1, 2, 5\n")
    with open(
        os.path.join(data_dir, "ui_full.txt"),
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write("7, 1, 5\n")


class OnlineHypothesisProgramTest(unittest.IsolatedAsyncioTestCase):
    async def test_two_calls_keep_llm1_candidate_blind_and_render_ids_out(self):
        with tempfile.TemporaryDirectory() as temporary:
            make_dataset(temporary)
            conf = {
                "dataset": "pog_dense",
                "data_path": temporary,
                "online_hypothesis_max_count": 3,
                "online_candidate_budget_per_hypothesis": 5,
                "online_evidence_budget_per_hypothesis": 8,
                "online_total_candidate_budget": 10,
            }
            sample = {
                "bundle_id": 99,
                "input_indices": [1, 2],
                "candidate_indices": [3, 4],
                "true_indice": 3,
                "true_option_char": "A",
            }
            prompts = []

            async def call_program(prompt, step_name):
                prompts.append(("program", prompt, step_name))
                return json.dumps(discovery_response())

            async def call_prediction(prompt, step_name):
                prompts.append(("prediction", prompt, step_name))
                return json.dumps(
                    {
                        "prediction": "A",
                        "rationale": "The trousers best fit the layered composition.",
                    }
                )

            def executor(**kwargs):
                return {
                    "status": "success",
                    "result": {
                        "schema_version": "candidate_proposal_set_v1",
                        "program_id": "P1",
                        "hypothesis": "unused by renderer",
                        "candidate_proposals": [
                            {"item_id": 5, "evidence_refs": ["E1"]}
                        ],
                        "evidence_records": [
                            {
                                "evidence_id": "E1",
                                "type": "historical_bundle_context",
                                "source": "bundle_item_history",
                                "anchor_item_ids": [1, 2],
                                "related_item_ids": [5],
                                "related_bundle_ids": [10],
                                "attributes": {"score": 0.91},
                            }
                        ],
                        "execution_trace": {
                            "used_sources": ["bundle_item_history"],
                            "candidate_budget": 5,
                            "evidence_budget": 8,
                        },
                    },
                    "validation_issues": [],
                }

            result = await run_online_hypothesis_program(
                sample,
                conf,
                call_program,
                call_prediction,
                program_executor=executor,
            )

        self.assertEqual(len(prompts), 2)
        llm1_prompt = prompts[0][1]
        llm2_prompt = prompts[1][1]
        self.assertIn("cream knit top", llm1_prompt)
        self.assertNotIn("wide-leg wool trousers", llm1_prompt)
        self.assertNotIn("sport sandals", llm1_prompt)
        self.assertNotIn('"answer_options"', llm1_prompt)
        self.assertIn("wide-leg wool trousers", llm2_prompt)
        self.assertIn("layered wool cardigan", llm2_prompt)
        self.assertNotIn('"item_id": 5', llm2_prompt)
        self.assertNotIn("related_bundle_ids", llm2_prompt)
        self.assertNotIn("0.91", llm2_prompt)
        self.assertEqual(result["evaluation"]["llm_calls"], 2)
        self.assertEqual(result["evaluation"]["prediction"], "A")
        self.assertEqual(result["evaluation"]["prediction_hit"], 1)

    async def test_program_failure_still_calls_prediction(self):
        with tempfile.TemporaryDirectory() as temporary:
            make_dataset(temporary)
            conf = {
                "dataset": "pog_dense",
                "data_path": temporary,
                "online_hypothesis_max_count": 3,
            }
            sample = {
                "bundle_id": 100,
                "input_indices": [1, 2],
                "candidate_indices": [3, 4],
                "true_indice": 4,
                "true_option_char": "B",
            }
            call_count = 0

            async def call_program(prompt, step_name):
                nonlocal call_count
                call_count += 1
                return json.dumps(discovery_response())

            async def call_prediction(prompt, step_name):
                nonlocal call_count
                call_count += 1
                self.assertIn('"search_status": "execution_error"', prompt)
                return json.dumps(
                    {
                        "prediction": "B",
                        "rationale": "Fallback comparison favors option B.",
                    }
                )

            def executor(**kwargs):
                return {
                    "status": "execution_error",
                    "result": None,
                    "validation_issues": [],
                }

            result = await run_online_hypothesis_program(
                sample,
                conf,
                call_program,
                call_prediction,
                program_executor=executor,
            )

        self.assertEqual(call_count, 2)
        self.assertEqual(result["evaluation"]["successful_program_count"], 0)
        self.assertEqual(result["evaluation"]["prediction"], "B")


class OnlineProgramRuntimeTest(unittest.TestCase):
    def test_generated_code_executes_with_restricted_boundary(self):
        result = execute_code_in_process(
            PROGRAM_CODE,
            source_api=FakeSourceAPI(),
            partial_item_ids=[1, 2],
            candidate_budget=5,
            evidence_budget=8,
        )
        self.assertEqual(result["candidate_proposals"], [])
        self.assertEqual(result["used_sources"], ["bundle_item_history"])

    def test_generated_code_executes_in_child_process_with_real_source_api(self):
        code = """
def execute(partial_item_ids, source_api, candidate_budget, evidence_budget):
    bundle_map = source_api.get_bundles_for_items(partial_item_ids)
    bundle_ids = []
    for values in bundle_map.values():
        bundle_ids.extend(values)
    bundle_ids = list(dict.fromkeys(bundle_ids))[:evidence_budget]
    item_map = source_api.get_items_for_bundles(bundle_ids)
    partial = set(partial_item_ids)
    records = []
    proposals = []
    for bundle_id in bundle_ids:
        related = [
            item_id
            for item_id in item_map.get(bundle_id, [])
            if item_id not in partial
        ]
        if not related:
            continue
        evidence_id = "E" + str(len(records) + 1)
        records.append({
            "evidence_id": evidence_id,
            "type": "historical_bundle_context",
            "source": "bundle_item_history",
            "anchor_item_ids": list(partial_item_ids),
            "related_item_ids": related,
            "related_bundle_ids": [bundle_id],
            "attributes": {},
        })
        for item_id in related:
            if len(proposals) >= candidate_budget:
                break
            if not any(value["item_id"] == item_id for value in proposals):
                proposals.append({
                    "item_id": item_id,
                    "evidence_refs": [evidence_id],
                })
    return {
        "candidate_proposals": proposals,
        "evidence_records": records,
        "used_sources": ["bundle_item_history"],
    }
""".strip()
        with tempfile.TemporaryDirectory() as temporary:
            make_dataset(temporary)
            execution = execute_program_subprocess(
                program={
                    "hypothesis_id": "H1",
                    "program_id": "P1",
                    "name": "HistoricalContextSearch",
                    "required_sources": ["bundle_item_history"],
                    "evidence_types": ["historical_bundle_context"],
                    "code": code,
                },
                hypothesis={
                    "id": "H1",
                    "observed_cues": ["soft knit texture"],
                    "intent": "The bundle may form a layered outfit.",
                    "missing_role": "A complementary layer.",
                },
                conf={
                    "dataset": "pog_dense",
                    "data_path": temporary,
                    "online_program_timeout_seconds": 10,
                },
                partial_item_ids=[1, 2],
                candidate_budget=5,
                evidence_budget=8,
            )

        self.assertEqual(execution["status"], "success")
        proposals = execution["result"]["candidate_proposals"]
        self.assertEqual([proposal["item_id"] for proposal in proposals], [5])


if __name__ == "__main__":
    unittest.main()

"""Unit tests for hypothesis-conditioned completion-exemplar retrieval."""

import json
import os
import sys
import tempfile
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from online_hypothesis_program.pipeline import (
    _admit_valid_discovery_entries,
    _normalize_discovery_contract,
    _parse_discovery_response,
    run_online_hypothesis_program,
)
from online_hypothesis_program.raw_workspace import build_dataset_workspace
from online_hypothesis_program.renderer import (
    _representative_user_contexts,
    render_retrieval_evidence,
)
from online_hypothesis_program.runtime import (
    compile_program_in_process,
    execute_program_subprocess,
    retrieve_in_process,
    wrap_and_validate_result,
)
from online_hypothesis_program.schemas import (
    DISCOVERY_SCHEMA_VERSION,
    validate_discovery_result,
    validate_online_program_source,
)
from online_hypothesis_program.source_api import DatasetSourceAPI


PROGRAM_CODE = """
def retrieve(partial_item_ids, dataset_workspace, parameters, budget):
    history = dataset_workspace["bundle_item_history"]
    partial_set = set(partial_item_ids)
    bundle_ids = []
    for item_id in partial_item_ids:
        for bundle_id in history["items_to_bundles"].get(item_id, ()):
            if bundle_id not in bundle_ids:
                bundle_ids.append(bundle_id)
    counts = {}
    first_bundle = {}
    for bundle_id in bundle_ids:
        for item_id in history["bundles_to_items"].get(bundle_id, ()):
            if item_id not in partial_set:
                counts[item_id] = counts.get(item_id, 0) + 1
                first_bundle.setdefault(item_id, bundle_id)
    ranked = sorted(counts, key=lambda item_id: (-counts[item_id], item_id))
    result = []
    for item_id in ranked[:budget["max_retrieved_items"]]:
        result.append({
            "item_id": item_id,
            "provenance": [{
                "source": "bundle_item_history",
                "relation": "co-occurs in partial-conditioned historical bundles",
                "supporting_context": {
                    "item_ids": list(partial_item_ids),
                    "bundle_ids": [first_bundle[item_id]],
                    "user_ids": []
                }
            }]
        })
    return result
""".strip()


def discovery_response():
    specifications = [
        (
            "P1",
            (
                "The partial bundle follows a coordinated composition and an "
                "additional item should extend a recurring historical combination."
            ),
        ),
        (
            "P2",
            (
                "The partial bundle leaves a complementary role that may be filled "
                "by items repeatedly observed around the same anchors."
            ),
        ),
    ]
    return {
        "schema_version": DISCOVERY_SCHEMA_VERSION,
        "programs": [
            {
                "id": program_id,
                "hypothesis": hypothesis,
                "strategy": {
                    "reference": (
                        "Build historical bundles reached from partial items."
                    ),
                    "retrieval": (
                        "Rank non-partial corpus items by repeated co-occurrence."
                    ),
                },
                "required_sources": ["bundle_item_history"],
                "parameters": {},
                "code": PROGRAM_CODE,
            }
            for program_id, hypothesis in specifications
        ],
    }


def make_dataset(root):
    data_dir = os.path.join(root, "pog_dense")
    os.makedirs(data_dir, exist_ok=True)
    with open(os.path.join(data_dir, "count.json"), "w", encoding="utf-8") as handle:
        json.dump({"#B": 20, "#I": 6, "#U": 10}, handle)
    with open(
        os.path.join(data_dir, "item_info.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            {
                "1": {"title": "cream knit top", "cate_id": "c1"},
                "2": {"title": "soft neutral scarf", "cate_id": "c2"},
                "3": {"title": "wide-leg wool trousers", "cate_id": "c3"},
                "4": {"title": "sport sandals", "cate_id": "c4"},
                "5": {"title": "layered wool cardigan", "cate_id": "c5"},
            },
            handle,
        )
    with open(
        os.path.join(data_dir, "bi_train.txt"),
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write("10, 1, 2, 3, 5\n")
    with open(
        os.path.join(data_dir, "ui_full.txt"),
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write("7, 1, 3, 5\n")


def successful_execution(program):
    return {
        "status": "success",
        "result": {
            "schema_version": "completion_exemplar_retrieval_v2",
            "program_id": program["id"],
            "completion_hypothesis": program["hypothesis"],
            "retrieved_items": [
                {
                    "item_id": 3,
                    "provenance": [
                        {
                            "source": "bundle_item_history",
                            "relation": "historically completes the partial anchors",
                            "supporting_context": {
                                "item_ids": [1, 2, 5],
                                "bundle_ids": [10],
                                "user_ids": [],
                            },
                        }
                    ],
                }
            ],
            "execution_trace": {
                "required_sources": ["bundle_item_history"],
                "max_retrieved_items": 5,
                "max_supporting_contexts_per_item": 2,
            },
        },
        "validation_issues": [],
    }


class OnlineHypothesisPipelineTest(unittest.IsolatedAsyncioTestCase):
    async def test_two_calls_execute_each_retriever_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            make_dataset(temporary)
            conf = {
                "dataset": "pog_dense",
                "data_path": temporary,
                "online_hypothesis_count": 2,
                "online_retrieved_item_budget_per_hypothesis": 5,
                "online_max_supporting_contexts_per_item": 2,
            }
            sample = {
                "bundle_id": 99,
                "input_indices": [1, 2],
                "candidate_indices": [3, 4],
                "true_indice": 3,
                "true_option_char": "A",
            }
            prompts = []
            executor_calls = []

            async def call_program(prompt, step_name):
                prompts.append(("program", prompt, step_name))
                return json.dumps(discovery_response())

            async def call_prediction(prompt, step_name):
                prompts.append(("prediction", prompt, step_name))
                return json.dumps(
                    {
                        "prediction": "A",
                        "ranking": ["A", "B"],
                        "rationale": "A matches the retrieved historical exemplar.",
                    }
                )

            def executor(**kwargs):
                executor_calls.append(kwargs)
                return successful_execution(kwargs["program"])

            result = await run_online_hypothesis_program(
                sample,
                conf,
                call_program,
                call_prediction,
                program_executor=executor,
            )

        self.assertEqual(len(prompts), 2)
        self.assertEqual(len(executor_calls), 2)
        for call in executor_calls:
            self.assertNotIn("candidate_ids", call)
            self.assertEqual(call["partial_item_ids"], [1, 2])
            self.assertEqual(call["retrieved_item_budget"], 5)
        llm1_prompt, llm2_prompt = prompts[0][1], prompts[1][1]
        self.assertIn("cream knit top", llm1_prompt)
        self.assertIn('"item_id": 1', llm1_prompt)
        self.assertIn('"item_id": 2', llm1_prompt)
        self.assertNotIn("wide-leg wool trousers", llm1_prompt)
        self.assertNotIn("sport sandals", llm1_prompt)
        self.assertNotIn('"answer_options"', llm1_prompt)
        self.assertLess(
            llm1_prompt.index("PARTIAL BUNDLE"),
            llm1_prompt.index("Interpret the partial bundle"),
        )
        self.assertIn('"source_record_format"', llm1_prompt)
        self.assertIn('"runtime_format"', llm1_prompt)
        self.assertIn('"observed_field_schema"', llm1_prompt)
        self.assertIn("bundle_id, item_id_1, item_id_2", llm1_prompt)
        self.assertIn('"supporting_context"', llm1_prompt)
        self.assertIn("max_supporting_contexts_per_item", llm1_prompt)
        self.assertIn(
            "Interpret the partial bundle in exactly 2 genuinely different ways.",
            llm1_prompt,
        )
        self.assertIn("def retrieve(", llm1_prompt)
        self.assertIn("wide-leg wool trousers", llm2_prompt)
        self.assertIn('"matching_answer_options": [', llm2_prompt)
        self.assertIn('"A"', llm2_prompt)
        self.assertNotIn('"item_id": 3', llm2_prompt)
        self.assertEqual(result["evaluation"]["gt_retrieved"], True)
        self.assertEqual(result["evaluation"]["gt_retrieval_rank"], 1)
        self.assertEqual(result["evaluation"]["program_count"], 2)
        self.assertEqual(result["evaluation"]["proposed_program_count"], 2)

    async def test_invalid_discovery_still_calls_prediction(self):
        with tempfile.TemporaryDirectory() as temporary:
            make_dataset(temporary)
            conf = {"dataset": "pog_dense", "data_path": temporary}
            sample = {
                "bundle_id": 99,
                "input_indices": [1, 2],
                "candidate_indices": [3, 4],
                "true_indice": 3,
                "true_option_char": "A",
            }
            calls = []

            async def bad_program(prompt, step_name):
                calls.append(step_name)
                return "{}"

            async def prediction(prompt, step_name):
                calls.append(step_name)
                return json.dumps(
                    {
                        "prediction": "B",
                        "ranking": ["B", "A"],
                        "rationale": "No retrieved context was available.",
                    }
                )

            result = await run_online_hypothesis_program(
                sample,
                conf,
                bad_program,
                prediction,
            )

        self.assertEqual(len(calls), 2)
        self.assertEqual(result["evaluation"]["program_count"], 0)
        self.assertTrue(result["llm1"]["validation_issues"])
        self.assertTrue(result["evaluation"]["valid_prediction"])


class RetrievalRuntimeTest(unittest.TestCase):
    def test_generated_retriever_receives_budget_and_returns_items(self):
        namespace = compile_program_in_process(PROGRAM_CODE)
        workspace = {
            "item_ids": tuple(range(6)),
            "bundle_item_history": {
                "items_to_bundles": {1: (10,), 2: (10,)},
                "bundles_to_items": {10: (1, 2, 3, 5)},
            },
        }
        result = retrieve_in_process(
            namespace,
            partial_item_ids=[1, 2],
            dataset_workspace=workspace,
            parameters={},
            retrieved_item_budget=1,
            supporting_context_budget=2,
        )
        self.assertEqual([item["item_id"] for item in result], [3])

    def test_result_validator_rejects_partial_item(self):
        discovery = discovery_response()
        raw = [
            {
                "item_id": 1,
                "provenance": [
                    {
                        "source": "bundle_item_history",
                        "relation": "invalid self retrieval",
                        "supporting_context": {
                            "item_ids": [2],
                            "bundle_ids": [10],
                            "user_ids": [],
                        },
                    }
                ],
            }
        ]
        _, issues = wrap_and_validate_result(
            raw,
            program=discovery["programs"][0],
            partial_item_ids=[1, 2],
            retrieved_item_budget=5,
            supporting_context_budget=2,
        )
        self.assertIn(
            "retrieved_items[0].item_id must exclude partial items",
            issues,
        )

    def test_supporting_bundle_context_may_include_retrieved_item(self):
        discovery = discovery_response()
        raw = [
            {
                "item_id": 3,
                "provenance": [
                    {
                        "source": "bundle_item_history",
                        "relation": "member of the supporting historical bundle",
                        "supporting_context": {
                            "item_ids": [1, 2, 3, 5],
                            "bundle_ids": [10],
                            "user_ids": [],
                        },
                    }
                ],
            }
        ]
        _, issues = wrap_and_validate_result(
            raw,
            program=discovery["programs"][0],
            partial_item_ids=[1, 2],
            retrieved_item_budget=5,
            supporting_context_budget=2,
        )
        self.assertEqual(issues, [])

    def test_subprocess_executes_one_partial_only_retriever(self):
        with tempfile.TemporaryDirectory() as temporary:
            make_dataset(temporary)
            discovery = discovery_response()
            result = execute_program_subprocess(
                program=discovery["programs"][0],
                conf={
                    "dataset": "pog_dense",
                    "data_path": temporary,
                    "online_program_timeout_seconds": 10,
                },
                partial_item_ids=[1, 2],
                retrieved_item_budget=2,
                supporting_context_budget=1,
            )
        self.assertEqual(result["status"], "success")
        self.assertEqual(
            [item["item_id"] for item in result["result"]["retrieved_items"]],
            [3, 5],
        )

    def test_workspace_mappings_are_immutable(self):
        with tempfile.TemporaryDirectory() as temporary:
            make_dataset(temporary)
            workspace = build_dataset_workspace(
                {"dataset": "pog_dense", "data_path": temporary},
                allowed_sources=["bundle_item_history"],
            )
            with self.assertRaises(TypeError):
                workspace["bundle_item_history"]["bundles_to_items"][10] = ()

    def test_metadata_runtime_accepts_integer_and_numeric_string_lookup(self):
        with tempfile.TemporaryDirectory() as temporary:
            make_dataset(temporary)
            workspace = build_dataset_workspace(
                {"dataset": "pog_dense", "data_path": temporary},
                allowed_sources=["item_metadata"],
            )
            metadata = workspace["item_metadata"]
            self.assertEqual(metadata.get(1), metadata.get("1"))
            self.assertTrue(all(isinstance(key, int) for key in metadata))
            with self.assertRaises(TypeError):
                metadata[1] = {}


class RetrievalSchemaAndRendererTest(unittest.TestCase):
    def test_contract_normalization_removes_item_ids_source_declaration(self):
        response = discovery_response()
        response["programs"][0]["required_sources"].append("item_ids")
        normalized, repairs = _normalize_discovery_contract(response)
        self.assertEqual(
            normalized["programs"][0]["required_sources"],
            ["bundle_item_history"],
        )
        self.assertEqual(len(repairs), 1)
        self.assertIn(
            "item_ids",
            response["programs"][0]["required_sources"],
        )

    def test_admission_keeps_valid_pair_when_another_program_is_invalid(self):
        response = discovery_response()
        response["programs"][1]["code"] = (
            "def retrieve(partial_item_ids, dataset_workspace, parameters, budget):\n"
            "    return []"
        )
        admitted, rejected = _admit_valid_discovery_entries(
            response,
            available_sources=["bundle_item_history"],
        )
        self.assertEqual(
            [program["id"] for program in admitted["programs"]],
            ["P1"],
        )
        self.assertEqual(rejected[0]["program_id"], "P2")
        self.assertTrue(rejected[0]["validation_issues"])

    def test_discovery_accepts_retrieve_contract(self):
        issues = validate_discovery_result(
            discovery_response(),
            available_sources=[
                "bundle_item_history",
                "item_metadata",
                "user_item_history",
            ],
        )
        self.assertEqual(issues, [])

    def test_source_validator_rejects_candidate_argument(self):
        code = """
def retrieve(partial_item_ids, candidate_id, dataset_workspace, parameters, budget):
    return []
""".strip()
        issues = validate_online_program_source(code)
        self.assertTrue(
            any("retrieve arguments must be exactly" in issue for issue in issues)
        )

    def test_renderer_resolves_text_and_exact_option_overlap(self):
        with tempfile.TemporaryDirectory() as temporary:
            make_dataset(temporary)
            reader = DatasetSourceAPI(
                {"dataset": "pog_dense", "data_path": temporary}
            )
            discovery = discovery_response()
            executions = {
                "P1": successful_execution(discovery["programs"][0]),
                "P2": successful_execution(discovery["programs"][1]),
            }
            rendered = render_retrieval_evidence(
                programs=discovery["programs"],
                executions=executions,
                source_api=reader,
                answer_options=[
                    {"label": "A", "item_id": 3},
                    {"label": "B", "item_id": 4},
                ],
            )
        exemplar = rendered["model_view"][0]["retrieved_exemplars"][0]
        self.assertEqual(exemplar["item_text"], "wide-leg wool trousers")
        self.assertEqual(exemplar["matching_answer_options"], ["A"])
        self.assertIn(
            "layered wool cardigan",
            exemplar["provenance"][0]["related_item_texts"],
        )
        self.assertEqual(
            rendered["retrieval_counts"]["unique_retrieved_item_count"],
            1,
        )

    def test_user_provenance_context_is_readable(self):
        with tempfile.TemporaryDirectory() as temporary:
            make_dataset(temporary)
            reader = DatasetSourceAPI(
                {"dataset": "pog_dense", "data_path": temporary}
            )
            contexts = _representative_user_contexts(
                reader,
                {
                    "supporting_context": {
                        "item_ids": [],
                        "bundle_ids": [],
                        "user_ids": [7],
                    }
                },
            )
        self.assertIn("wide-leg wool trousers", contexts[0])

    def test_discovery_json_repair(self):
        value = json.dumps(discovery_response(), indent=2)
        parsed, repairs = _parse_discovery_response(value)
        self.assertEqual(parsed["schema_version"], DISCOVERY_SCHEMA_VERSION)
        self.assertEqual(repairs, [])


if __name__ == "__main__":
    unittest.main()

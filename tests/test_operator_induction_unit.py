"""Unit tests for candidate-program induction, deduplication, and admission."""

import json
import os
import sys
import unittest
from unittest.mock import patch


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from operator_learning.memory import deduplicate_operator_pool
from operator_learning.pipeline import (
    admit_verified_programs,
    compile_operator_programs,
    deduplicate_raw_operators,
    enrich_case_source_diagnostics,
    induce_raw_operators,
    verify_compiled_programs,
)
from operator_learning.runtime import (
    evaluate_candidate_proposal_set,
    make_compiled_program,
    validate_candidate_proposal_set,
    validate_program_source,
)
from operator_learning.schemas import (
    CANDIDATE_PROPOSAL_OUTPUT_CONTRACT,
    OPERATOR_LIBRARY_SCHEMA_VERSION,
)


SOURCE_CAPABILITIES = {
    "dataset": "pog_dense",
    "description": "Fashion bundle completion.",
    "components": [
        {
            "id": "item_metadata",
            "description": "Item text and opaque category identifiers.",
            "format": {"path": "data/item_info.json"},
        },
        {
            "id": "bundle_item_history",
            "description": "Historical bundle-to-item relations.",
            "format": {"path": "data/bi_train.txt"},
        },
    ],
}


def program_spec(
    *,
    name="RetrieveRecurringBundleCompanions",
    hypothesis="Items recurring in related historical bundles are plausible completions.",
):
    return {
        "name": name,
        "hypothesis": hypothesis,
        "required_sources": ["bundle_item_history"],
        "applicability": ["partial items have historical bundle coverage"],
        "evidence_types": ["historical_bundle_context"],
        "pseudocode": [
            "retrieve historical bundles containing partial items",
            "collect non-partial items from the related bundles",
            "retain recurring candidate items under the candidate budget",
            "return representative bundle provenance for every candidate",
        ],
        "output_contract": CANDIDATE_PROPOSAL_OUTPUT_CONTRACT,
    }


def strategy_spec(
    *,
    strategy_id="S1",
    intent="The completed set forms a coherent historical composition.",
    name="HistoricalBundleOverlap",
    description=(
        "Compare each candidate through historical bundles shared with the partial items."
    ),
    source="bundle_item_history",
):
    return {
        "strategy_id": strategy_id,
        "intent": intent,
        "name": name,
        "description": description,
        "reference_construction": (
            "Collect historical bundles connected to the partial items."
        ),
        "candidate_relation": (
            "Check how each candidate connects to the partial-conditioned bundles."
        ),
        "evidence_route": [
            "partial items",
            "historical bundles",
            "candidate-conditioned bundle contexts",
        ],
        "required_sources": [source],
        "pseudocode": [
            "build a partial-conditioned historical bundle reference",
            "apply the same candidate relation to every candidate",
            "return bounded textual bundle contexts",
        ],
    }


def induction_program(strategy_id="S1"):
    return {
        "strategy_id": strategy_id,
        "code": (
            "def run(partial_items, candidate_items, source_paths, "
            "max_contexts_per_candidate=5):\n"
            "    return [\n"
            "        {\n"
            "            'label': candidate['label'],\n"
            "            'item_id': candidate['item_id'],\n"
            "            'contexts': [],\n"
            "        }\n"
            "        for candidate in candidate_items\n"
            "    ]\n"
        ),
    }


def induction_response(*strategies):
    return {
        "strategy_specs": list(strategies),
        "programs": [
            induction_program(strategy["strategy_id"])
            for strategy in strategies
        ],
    }


def valid_program_code():
    return '''
def execute(partial_item_ids, source_api, candidate_budget, evidence_budget):
    partial = {str(item_id) for item_id in partial_item_ids}
    bundle_map = source_api.get_bundles_for_items(partial_item_ids)
    bundle_ids = []
    for values in bundle_map.values():
        bundle_ids.extend(values)
    bundle_ids = list(dict.fromkeys(bundle_ids))[:evidence_budget]
    item_map = source_api.get_items_for_bundles(bundle_ids)
    evidence_records = []
    candidate_refs = {}
    for index, bundle_id in enumerate(bundle_ids, start=1):
        related = [
            item_id
            for item_id in item_map.get(bundle_id, [])
            if str(item_id) not in partial
        ]
        evidence_id = f"E{index}"
        evidence_records.append({
            "evidence_id": evidence_id,
            "type": "historical_bundle_context",
            "source": "bundle_item_history",
            "anchor_item_ids": list(partial_item_ids),
            "related_item_ids": related,
            "related_bundle_ids": [bundle_id],
            "attributes": {},
        })
        for item_id in related:
            candidate_refs.setdefault(str(item_id), [item_id, []])[1].append(evidence_id)
    proposals = [
        {"item_id": value[0], "evidence_refs": value[1]}
        for value in list(candidate_refs.values())[:candidate_budget]
    ]
    return {
        "schema_version": "candidate_proposal_set_v1",
        "program_id": "program_1",
        "hypothesis": "Historical recurrence supports completion candidates.",
        "candidate_proposals": proposals,
        "evidence_records": evidence_records,
        "execution_trace": {
            "used_sources": ["bundle_item_history"],
            "candidate_budget": candidate_budget,
            "evidence_budget": evidence_budget,
        },
    }
'''.strip()


class CandidateProgramInductionTest(unittest.IsolatedAsyncioTestCase):
    async def test_induction_prompt_exposes_candidates_but_hides_the_gt_identity(self):
        case = {
            "case_id": "bundle_1",
            "dataset": "pog_dense",
            "bundle_id": 1,
            "partial_items": [
                {
                    "item_id": 10,
                    "text": "black tailored jacket",
                    "metadata": {"cate_id": "outerwear"},
                }
            ],
            "candidate_items": [
                {"label": "A", "item_id": 20, "text": "black trousers"},
                {"label": "B", "item_id": 21, "text": "white sneakers"},
            ],
            "source_diagnostics": {
                "partial_item_count": 1,
                "partial_metadata_coverage": {"covered": 1, "total": 1},
            },
            "evaluation": {
                "ground_truth_item_id": 999999,
                "ground_truth_profile": {"text": "SECRET_GT_ITEM"},
                "benchmark_candidate_item_ids": [999999, 888888],
            },
        }
        response = induction_response(
            strategy_spec(),
            strategy_spec(
                strategy_id="S2",
                name="CategoryCompositionGap",
                description=(
                    "Compare candidate categories with the category composition "
                    "of historical bundles related to the partial set."
                ),
                source="item_metadata",
            ),
            strategy_spec(
                strategy_id="S3",
                name="ConditionalCompanionConsistency",
                description=(
                    "Measure whether a candidate repeatedly completes historical "
                    "contexts containing multiple partial anchors."
                ),
            ),
        )
        response["programs"][0]["code"] = (
            "import json\n\n"
            "def run(partial_items, candidate_items, source_paths, "
            "max_contexts_per_candidate=5):\n"
            "    return []\n"
        )
        response["programs"].reverse()
        calls = []

        async def call_text(prompt, step_name):
            calls.append((prompt, step_name))
            return json.dumps(response)

        conf = {
            "dataset": "pog_dense",
            "operator_induction_count": 3,
            "operator_prompt_text_only": True,
        }
        with patch(
            "operator_learning.pipeline.build_discovery_case",
            return_value=case,
        ):
            result = await induce_raw_operators(
                [{}],
                conf,
                call_text,
                source_capabilities=SOURCE_CAPABILITIES,
            )

        self.assertEqual(len(calls), 1)
        prompt, step_name = calls[0]
        self.assertIn("Strategy Designer", prompt)
        self.assertIn("INPUT CASE", prompt)
        self.assertIn("SOURCE DIAGNOSTICS", prompt)
        self.assertIn(
            '"availability": "available" means that the source exists',
            prompt,
        )
        self.assertIn(
            '"none": no partial item has a direct source record or relation',
            prompt,
        )
        self.assertIn("SOURCE MANIFEST", prompt)
        self.assertIn("PROGRAM CONTRACT", prompt)
        self.assertIn("construct Bi = P union {ci}", prompt)
        self.assertIn("difficult to distinguish", prompt)
        self.assertEqual(prompt.count("Bi = P union {ci}"), 1)
        self.assertIn("from a bundle-completion perspective", prompt)
        self.assertIn(
            "Exactly one candidate item is the actual missing item to add to "
            "the given partial bundle",
            prompt,
        )
        self.assertIn("competing completion hypotheses", prompt)
        self.assertIn(
            "Infer exactly three distinct and plausible completion intents",
            prompt,
        )
        self.assertIn("fits every hypothetical bundle equally", prompt)
        self.assertIn("STRATEGY DEFINITION", prompt)
        self.assertIn(
            "This shared evaluation basis is called the reference",
            prompt,
        )
        self.assertIn(
            "A reference is not the correct candidate or any particular candidate",
            prompt,
        )
        self.assertIn(
            "use the same reference to evaluate every candidate",
            prompt,
        )
        self.assertIn(
            "return an empty contexts list for that candidate",
            prompt,
        )
        self.assertIn(
            "must show a concrete connection between that candidate "
            "and the shared reference",
            prompt,
        )
        self.assertIn(
            "Do not return shared reference contexts selected "
            "independently of the candidate",
            prompt,
        )
        self.assertNotIn('"sparse_fallback"', prompt)
        self.assertIn('"intent": "one plausible interpretation', prompt)
        self.assertNotIn("retrieve_intent_conditioned_context", prompt)
        self.assertIn(
            '"code": "complete Python source implementing the exact S1 specification"',
            prompt,
        )
        self.assertIn("provide all imports and helper", prompt)
        self.assertIn('"pseudocode"', prompt)
        self.assertIn("source-grounded textual contexts", prompt)
        self.assertIn("Numerical computations may be used internally", prompt)
        self.assertIn(
            "do not output numerical scores, similarities, distances, counts, "
            "or diagnostic messages as contexts",
            prompt,
        )
        self.assertIn(
            "related item text or historical bundle item-text composition",
            prompt,
        )
        self.assertIn('"contexts": [', prompt)
        self.assertIn(
            "def run(partial_items, candidate_items, source_paths,",
            prompt,
        )
        self.assertIn(
            '"sources": ["one or more exact declared source IDs"]',
            prompt,
        )
        self.assertIn("source_paths maps each exact", prompt)
        self.assertIn("historical bundle item-text composition", prompt)
        self.assertNotIn('"signals": signals', prompt)
        self.assertIn('"strategy_specs"', prompt)
        self.assertIn('"programs"', prompt)
        self.assertLess(prompt.index('"strategy_specs"'), prompt.index('"programs"'))
        self.assertIn("black tailored jacket", prompt)
        self.assertIn("black trousers", prompt)
        self.assertIn('"item_id": 20', prompt)
        self.assertNotIn("SECRET_GT_ITEM", prompt)
        self.assertNotIn("999999", prompt)
        self.assertNotIn("888888", prompt)
        self.assertNotIn("ground_truth", prompt)
        self.assertNotIn("SOURCE GROUNDING", prompt)
        self.assertNotIn("FORBIDDEN PREVIOUS PROGRAM SIGNATURES", prompt)
        self.assertNotIn("semantic completion hypothesis", prompt)
        self.assertIn("candidate-relation strategy induction", step_name)
        self.assertIn(
            "Intent: " + response["strategy_specs"][0]["intent"],
            result["raw_operators"][0]["hypothesis"],
        )
        self.assertIn(
            "Candidate relation: "
            + response["strategy_specs"][0]["candidate_relation"],
            result["raw_operators"][0]["hypothesis"],
        )
        self.assertEqual(
            result["raw_operators"][0]["generated_code"],
            next(
                program["code"]
                for program in response["programs"]
                if program["strategy_id"] == "S1"
            ),
        )
        self.assertEqual(result["induction_traces"][0]["validation_issues"], [])
        self.assertNotIn("inputs", result["raw_operators"][0])
        self.assertNotIn("output", result["raw_operators"][0])

    def test_source_diagnostics_are_categorical_without_density_or_counts(self):
        case = {
            "partial_items": [
                {"item_id": 10, "text": "black jacket", "metadata": {}},
                {"item_id": 11, "text": "white shirt", "metadata": {}},
            ],
            "source_diagnostics": {
                "partial_item_count": 2,
                "partial_metadata_coverage": {"covered": 2, "total": 2},
            },
        }
        source_capabilities = {
            "components": [
                {"id": "item_metadata"},
                {"id": "bundle_item_history"},
                {"id": "user_item_history"},
                {"id": "item_content_embedding"},
            ]
        }
        diagnostic_indices = {
            "bundle_item_history": {
                "item_to_anchors": {"10": ["100"], "11": []},
                "anchor_to_items": {"100": ["10", "20"]},
            },
            "user_item_history": {
                "item_to_anchors": {},
                "anchor_to_items": {},
            },
        }

        enrich_case_source_diagnostics(
            case,
            source_capabilities,
            diagnostic_indices,
        )

        self.assertEqual(
            case["source_diagnostics"],
            {
                "bundle_item_history": {
                    "availability": "available",
                    "partial_coverage": "partial",
                },
                "item_content_embedding": {
                    "availability": "available",
                },
                "item_metadata": {
                    "availability": "available",
                    "partial_coverage": "full",
                },
                "user_item_history": {
                    "availability": "available",
                    "partial_coverage": "none",
                },
            },
        )
        serialized = json.dumps(case["source_diagnostics"])
        self.assertNotIn("density", serialized)
        self.assertNotIn("count", serialized)

    async def test_induction_memory_is_not_included_in_the_prompt(self):
        case = {
            "case_id": "bundle_2",
            "dataset": "pog_dense",
            "partial_items": [{"item_id": 11, "text": "cream knit top"}],
            "candidate_items": [
                {"label": "A", "item_id": 30, "text": "cream skirt"},
                {"label": "B", "item_id": 31, "text": "blue jeans"},
            ],
            "source_diagnostics": {},
            "evaluation": {
                "ground_truth_item_id": 12,
                "ground_truth_profile": {},
                "benchmark_candidate_item_ids": [],
            },
        }
        response = induction_response(
            strategy_spec(name="RetrieveLayeringCompanions"),
            strategy_spec(
                strategy_id="S2",
                name="CategoryCompositionGap",
                source="item_metadata",
            ),
            strategy_spec(
                strategy_id="S3",
                name="ConditionalCompanionConsistency",
            ),
        )
        prompts = []

        async def call_text(prompt, step_name):
            prompts.append(prompt)
            return json.dumps(response)

        with patch(
            "operator_learning.pipeline.build_discovery_case",
            return_value=case,
        ):
            await induce_raw_operators(
                [{}],
                {
                    "dataset": "pog_dense",
                    "operator_induction_count": 3,
                    "operator_prompt_text_only": True,
                },
                call_text,
                source_capabilities=SOURCE_CAPABILITIES,
                initial_operator_memory=[program_spec()],
            )

        prompt = prompts[0]
        self.assertNotIn("FORBIDDEN PREVIOUS PROGRAM SIGNATURES", prompt)
        self.assertNotIn("RetrieveRecurringBundleCompanions", prompt)
        self.assertNotIn("candidate_proposals_with_source_provenance", prompt)

    async def test_compilation_is_one_call_per_unique_program(self):
        operator = {
            **program_spec(),
            "operator_id": "candidate_program_1",
            "origin_case_id": "bundle_1",
        }
        library = {
            "schema_version": OPERATOR_LIBRARY_SCHEMA_VERSION,
            "operators": [operator],
        }
        code = valid_program_code()
        calls = []

        async def call_text(prompt, step_name):
            calls.append((prompt, step_name))
            return json.dumps(
                {
                    "program_name": operator["name"],
                    "code": code,
                }
            )

        result = await compile_operator_programs(
            library,
            {},
            call_text,
            source_capabilities=SOURCE_CAPABILITIES,
        )

        self.assertEqual(len(calls), 1)
        self.assertIn("offline program compilation", calls[0][1])
        self.assertIn("def execute", calls[0][0])
        self.assertNotIn("bundle_1", calls[0][0])
        compiled = result["compiled_library"]["programs"][0]
        self.assertEqual(compiled["implementation"]["code"], code)
        self.assertEqual(compiled["admission_status"], "unverified")
        self.assertEqual(len(compiled["implementation"]["sha256"]), 64)


class CandidateProgramDeterminismTest(unittest.TestCase):
    def test_deduplication_uses_no_llm_and_preserves_provenance(self):
        first = {
            **program_spec(),
            "operator_id": "bundle_1__op1",
            "origin_case_id": "bundle_1",
        }
        second = {
            **program_spec(
                hypothesis=(
                    "Retrieve plausible companions recurring in historical bundles."
                )
            ),
            "operator_id": "bundle_2__op1",
            "origin_case_id": "bundle_2",
        }
        second["pseudocode"] = [
            "find historical bundles linked to the partial items",
            "collect companion items outside the partial bundle",
            "keep a bounded recurring candidate set",
            "attach representative source bundle provenance",
        ]
        result = deduplicate_operator_pool(
            [first, second],
            similarity_threshold=0.9,
        )

        self.assertEqual(result["deduplicated_operator_count"], 1)
        self.assertEqual(
            result["operators"][0]["member_operator_ids"],
            ["bundle_1__op1", "bundle_2__op1"],
        )
        self.assertEqual(
            result["operators"][0]["origin_case_ids"],
            ["bundle_1", "bundle_2"],
        )

    def test_pipeline_dedup_builds_graph_free_library(self):
        result = deduplicate_raw_operators(
            [
                {
                    **program_spec(),
                    "operator_id": "bundle_1__op1",
                    "origin_case_id": "bundle_1",
                }
            ],
            {
                "operator_dedup_similarity_threshold": 0.9,
                "operator_library_max_size": 20,
            },
        )

        self.assertEqual(
            result["library"]["schema_version"],
            OPERATOR_LIBRARY_SCHEMA_VERSION,
        )
        self.assertNotIn("compatibility_graph", result)
        self.assertNotIn("clustering_prompt", result)

    def test_program_source_requires_stable_execute_boundary(self):
        self.assertEqual(validate_program_source(valid_program_code()), [])
        issues = validate_program_source(
            "def execute(sample, gt):\n    return open('secret').read()\n"
        )
        self.assertIn("execute arguments must be exactly", " | ".join(issues))
        self.assertIn("forbidden call: open", issues)

    def test_candidate_proposal_requires_candidate_linked_provenance(self):
        output = {
            "schema_version": "candidate_proposal_set_v1",
            "program_id": "program_1",
            "hypothesis": "Historical contexts support completion candidates.",
            "candidate_proposals": [
                {"item_id": 42, "evidence_refs": ["E1"]}
            ],
            "evidence_records": [
                {
                    "evidence_id": "E1",
                    "type": "historical_bundle_context",
                    "source": "bundle_item_history",
                    "anchor_item_ids": [1, 2],
                    "related_item_ids": [42, 43],
                    "related_bundle_ids": [9],
                    "attributes": {},
                }
            ],
            "execution_trace": {
                "used_sources": ["bundle_item_history"],
                "candidate_budget": 10,
                "evidence_budget": 5,
            },
        }
        issues = validate_candidate_proposal_set(
            output,
            allowed_sources=["bundle_item_history"],
            candidate_budget=10,
            evidence_budget=5,
        )
        self.assertEqual(issues, [])
        metrics = evaluate_candidate_proposal_set(output, 42)
        self.assertTrue(metrics["hit"])
        self.assertEqual(metrics["retrieval_rank"], 1)

        output["evidence_records"][0]["related_item_ids"] = [43]
        issues = validate_candidate_proposal_set(
            output,
            allowed_sources=["bundle_item_history"],
            candidate_budget=10,
            evidence_budget=5,
        )
        self.assertIn(
            "candidate_proposals[0] must reference evidence containing its item_id",
            issues,
        )

    def test_verification_and_admission_reuse_exact_code(self):
        operator = {**program_spec(), "operator_id": "program_1"}
        compiled = make_compiled_program(operator, valid_program_code())
        library = {"programs": [compiled]}
        cases = [
            {
                "case_id": "heldout_1",
                "evaluation": {"ground_truth_item_id": 42},
            }
        ]

        def execute_program(compiled_program, case, candidate_budget, evidence_budget):
            return {
                "schema_version": "candidate_proposal_set_v1",
                "program_id": "program_1",
                "hypothesis": operator["hypothesis"],
                "candidate_proposals": [
                    {"item_id": 42, "evidence_refs": ["E1"]}
                ],
                "evidence_records": [
                    {
                        "evidence_id": "E1",
                        "type": "historical_bundle_context",
                        "source": "bundle_item_history",
                        "anchor_item_ids": [1],
                        "related_item_ids": [42],
                        "related_bundle_ids": [7],
                        "attributes": {},
                    }
                ],
                "execution_trace": {
                    "used_sources": ["bundle_item_history"],
                    "candidate_budget": candidate_budget,
                    "evidence_budget": evidence_budget,
                },
            }

        verification = verify_compiled_programs(
            library,
            cases,
            execute_program,
            candidate_budget=10,
            evidence_budget=5,
        )
        admitted = admit_verified_programs(
            library,
            verification,
            min_execution_success_rate=1.0,
            min_candidate_recall=1.0,
        )

        self.assertEqual(len(admitted["verified_programs"]), 1)
        self.assertEqual(len(admitted["rejected_programs"]), 0)
        self.assertEqual(
            admitted["verified_programs"][0]["implementation"]["sha256"],
            compiled["implementation"]["sha256"],
        )


if __name__ == "__main__":
    unittest.main()

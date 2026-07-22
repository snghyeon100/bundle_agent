"""LLM prompts for rank-free operator induction, clustering, and composition."""

from code.common import pretty_json, task_semantics


def _semantic_case(case, text_only):
    if not text_only:
        return case
    return {
        "dataset": case.get("dataset"),
        "partial_items": [
            str(item.get("text", ""))
            for item in case.get("partial_items", [])
            if isinstance(item, dict) and str(item.get("text", "")).strip()
        ],
        "ground_truth": str(case.get("ground_truth", {}).get("text", "")),
    }


def induction_prompt(case, source_manifest, operator_count, *, text_only=True):
    semantic_case = _semantic_case(case, text_only)
    return (
        "You are the Operator Discovery Agent for a retrieval-based bundle-completion system.\n\n"
        f"{task_semantics(case.get('dataset'))}\n\n"
        "TASK DEFINITION\n"
        "A bundle is a set of items that belong together. At test time, the system receives an "
        "incomplete bundle and a finite candidate set, and must select the candidate that best completes "
        "the bundle. The system may retrieve evidence from the available data sources before a separate "
        "prediction component makes the final choice.\n\n"
        "This offline discovery step is different from test-time prediction. The validation case below "
        "contains only the partial bundle and its held-out ground-truth completion. The ground truth is "
        "shown solely as hindsight evidence for discovering useful retrieval behavior. It will not be "
        "available when an operator is later used. No negative candidates are provided in this step.\n\n"
        "OPERATOR DEFINITION\n"
        "A retrieval operator is one reusable evidence-acquisition or evidence-transformation primitive "
        "that can become a node inside a later multi-step workflow. At runtime, an operator may consume "
        "the partial bundle, candidate items when relevant, intermediate evidence from an earlier node, "
        "and one or more data sources. It must produce evidence or an intermediate signal that another "
        "operator or the final predictor can use.\n\n"
        "An operator is not a complete workflow, not a final candidate answer, and not a vague semantic "
        "observation. It must specify one central retrieval or transformation principle precisely enough "
        "for a later code-generation agent to implement. Supporting parsing or aggregation may be "
        "described, but do not combine several independent retrieval ideas into one operator.\n\n"
        "DISCOVERY OBJECTIVE\n"
        "First infer the semantic intent of the partial bundle and analyze why the ground-truth item is a "
        "valid completion. Use that hindsight relation to propose diverse and creative atomic operators. "
        "Do not hard-code the current item titles, IDs, exact ground-truth identity, or a rule that works "
        "only for this case. Generalize each insight into a reusable operation for other bundles with a "
        "similar situation. Do not force the operators into a predefined taxonomy; let the case semantics "
        "and useful retrieval structures determine them.\n\n"
        "The operators must be genuinely different retrieval ideas, not renamed paraphrases. Do not "
        "perform final completion selection or ranking.\n\n"
        f"Validation case:\n{pretty_json(semantic_case)}\n\n"
        "The following source capability manifest describes the data currently available to the "
        "system. Use it as practical grounding, but do not treat it as a closed list. A creative "
        "operator may also propose a useful derived signal or an additional conceptual source that is "
        "not currently available.\n\n"
        f"Source capability manifest:\n{pretty_json(source_manifest)}\n\n"
        f"Return JSON only with exactly {int(operator_count)} operators:\n"
        "{\n"
        '  "operators": [\n'
        "    {\n"
        '      "name": "ConcisePascalCaseName",\n'
        '      "purpose": "the single reusable evidence objective",\n'
        '      "anchor": "the runtime entity or observable condition that starts this operator",\n'
        '      "inputs": ["runtime inputs required by this operator"],\n'
        '      "sources": ["source name or additional source idea used to obtain evidence"],\n'
        '      "relation_path": "how the operator traverses or connects entities and evidence",\n'
        '      "operation": "one central retrieval, comparison, filtering, aggregation, or transformation principle",\n'
        '      "output": "evidence or intermediate signal, never the final candidate answer",\n'
        '      "when_useful": "observable test-time condition indicating this operator should be used"\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Return no explanation outside the JSON object."
    )


def clustering_prompt(raw_operators, dataset, min_operators, max_operators):
    return (
        "You are building a middle-level retrieval operator library from validation-induced raw "
        "operators. Cluster by functional equivalence, not merely by shared data source or wording.\n\n"
        "Two operators may merge only when their retrieval objective, anchor, relation path, "
        "computation/filtering principle, and output evidence type are functionally compatible. "
        "Keep operators separate when any of those structural roles differ. Avoid abstractions such "
        "as Analyze, Retrieve, Process, or Validate.\n\n"
        f"Dataset: {dataset}\n"
        f"Raw operators:\n{pretty_json(raw_operators)}\n\n"
        f"Create between {int(min_operators)} and {int(max_operators)} refined operators when the raw "
        "pool supports that many genuinely distinct functions. Return JSON only:\n"
        "{\n"
        '  "clusters": [\n'
        "    {\n"
        '      "name": "functional cluster name",\n'
        '      "rationale": "why these members are functionally equivalent",\n'
        '      "member_ids": ["raw operator_id"],\n'
        '      "representative": "RefinedOperatorName"\n'
        "    }\n"
        "  ],\n"
        '  "operators": [\n'
        "    {\n"
        '      "name": "RefinedOperatorName",\n'
        '      "purpose": "middle-level reusable retrieval objective",\n'
        '      "anchor": "generalized retrieval anchor",\n'
        '      "inputs": ["required logical input"],\n'
        '      "sources": ["allowed source name or source class"],\n'
        '      "relation_path": "generalized relation path",\n'
        '      "operation": "specific reusable computation or filtering principle",\n'
        '      "output": "evidence type produced",\n'
        '      "when_useful": "observable applicability condition",\n'
        '      "derived_from": ["raw operator_id"]\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Every raw operator_id must belong to exactly one cluster. Each refined operator must match "
        "exactly one cluster representative and preserve source-grounded executability."
    )


def composition_prompt(case, source_manifest, library, workflow_count):
    public_case = {
        "case_id": case.get("case_id"),
        "dataset": case.get("dataset"),
        "bundle_id": case.get("bundle_id"),
        "partial_items": case.get("partial_items", []),
        "candidates": case.get("candidates", []),
    }
    return (
        "You compose retrieval workflows for a new bundle-completion case using only the supplied "
        "operator library. This is rank-free test-time composition: no ground truth, validation label, "
        "candidate rank, or historical operator score is available.\n\n"
        f"{task_semantics(case.get('dataset'))}\n\n"
        "Infer the bundle intent only from partial_items. Candidates may affect retrieval comparison "
        "steps but must not revise the inferred intent.\n\n"
        f"Case:\n{pretty_json(public_case)}\n\n"
        f"Available sources:\n{pretty_json(source_manifest)}\n\n"
        f"Operator library:\n{pretty_json(library)}\n\n"
        f"Create exactly {int(workflow_count)} structurally distinct workflows. Each workflow must use "
        "2 to 4 library operators, specify concrete data flow, and be executable using available "
        "sources. Reusing an identical operator sequence with renamed prose is not distinct. Choose one "
        "recommended workflow only by case/operator/source applicability, never by candidate ranking.\n\n"
        "Return JSON only:\n"
        "{\n"
        '  "schema_version": "operator_workflow_candidates_v1",\n'
        '  "intent": "one concise intent inferred only from partial items",\n'
        '  "workflows": [\n'
        "    {\n"
        '      "name": "workflow name",\n'
        '      "operator_sequence": ["ExactLibraryOperatorName", "ExactLibraryOperatorName"],\n'
        '      "rationale": "why this composition fits the current case",\n'
        '      "source_plan": "which available sources support which steps",\n'
        '      "steps": [\n'
        "        {\n"
        '          "operator": "ExactLibraryOperatorName",\n'
        '          "objective": "case-specific use of the operator",\n'
        '          "input": "input data or prior step output",\n'
        '          "output": "evidence passed to the next step"\n'
        "        }\n"
        "      ]\n"
        "    }\n"
        "  ],\n"
        '  "recommended_workflow": "one workflow name from workflows"\n'
        "}"
    )

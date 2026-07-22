"""LLM prompts for source-free semantic induction and later workflow composition."""

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
        "candidates": {
            str(candidate.get("label", "")): str(candidate.get("text", ""))
            for candidate in case.get("candidates", [])
            if isinstance(candidate, dict)
            and str(candidate.get("label", "")).strip()
            and str(candidate.get("text", "")).strip()
        },
        "ground_truth_candidate": str(case.get("ground_truth", {}).get("label", "")),
    }


def induction_prompt(case, operator_count, *, text_only=True):
    semantic_case = _semantic_case(case, text_only)
    return (
        "You are the Semantic Operator Discovery Agent for a bundle-completion system.\n\n"
        f"{task_semantics(case.get('dataset'))}\n\n"
        "TASK DEFINITION\n"
        "A bundle is a set of items that belong together. At test time, the system receives an "
        "incomplete partial bundle and a finite candidate set, and must select the candidate that best completes "
        "the bundle. A later system will gather evidence before a separate prediction component makes "
        "the final choice.\n\n"
        "This offline discovery step is different from test-time prediction. The validation case below "
        "contains the partial bundle, candidate texts, and the label of the ground-truth candidate. The "
        "ground-truth label is shown solely as hindsight evidence for discovering useful retrieval "
        "behavior. It will not be available when an operator is later used.\n\n"
        "OPERATOR DEFINITION\n"
        "A semantic operator is one reusable evidence question or evidence-transformation primitive that "
        "can become a node inside a later multi-step workflow. At runtime, it may consume the partial "
        "bundle, candidate items when relevant, and intermediate evidence from an earlier node. It must "
        "produce semantic evidence or an intermediate signal that another operator or the final predictor "
        "can use.\n\n"
        "An operator is not a complete workflow, not a final candidate answer, and not a vague semantic "
        "observation. It must specify one central comparison or transformation principle. Do not combine "
        "several independent ideas into one operator. This phase is deliberately source-free: do not "
        "choose ideas based on files, databases, embeddings, models, or other implementation resources. "
        "Describe what evidence should be obtained, not which source will provide it. Source grounding "
        "will happen only after the semantic library has been clustered.\n\n"
        "DISCOVERY OBJECTIVE\n"
        "First infer the semantic intent of the partial bundle. Then compare the ground-truth candidate "
        "with the alternatives and analyze what semantic evidence could distinguish the correct "
        "completion from plausible but incorrect candidates. Use that hindsight contrast to propose "
        "diverse and creative atomic operators. Do not merely explain why the GT is good, and do not "
        "hard-code the current item titles, candidate labels, IDs, exact ground-truth identity, or a rule "
        "that works only for this case. Generalize each insight into a reusable operation for other "
        "bundles with a similar discriminative situation. Do not force the operators into a predefined "
        "taxonomy; let the case semantics and useful retrieval structures determine them.\n\n"
        "The operators must be genuinely different semantic ideas, not renamed paraphrases. Do not "
        "perform final completion selection or ranking.\n\n"
        f"Validation case:\n{pretty_json(semantic_case)}\n\n"
        f"Return JSON only with exactly {int(operator_count)} operators:\n"
        "{\n"
        '  "operators": [\n'
        "    {\n"
        '      "name": "ConcisePascalCaseName",\n'
        '      "purpose": "the single reusable evidence objective",\n'
        '      "anchor": "the semantic entity, role, relation, or contrast that starts this operator",\n'
        '      "inputs": ["logical runtime inputs required by this operator"],\n'
        '      "operation": "one central semantic comparison, filtering, aggregation, or transformation principle",\n'
        '      "output": "evidence or intermediate signal, never the final candidate answer",\n'
        '      "when_useful": "observable test-time condition indicating this operator should be used"\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Return no explanation outside the JSON object."
    )


def clustering_prompt(raw_operators, dataset, min_operators, max_operators):
    return (
        "You are building a middle-level semantic operator library from validation-induced raw "
        "operators. No source capability manifest is available in this phase. Cluster by semantic and "
        "functional equivalence, never by presumed files, models, databases, or implementation methods.\n\n"
        "Two operators may merge only when their evidence objective, semantic anchor, comparison or "
        "transformation principle, and output evidence type are compatible. Keep operators separate "
        "when those roles differ. Preserve useful semantic distinctions while removing paraphrases and "
        "case-specific wording. Avoid empty abstractions such as Analyze, Retrieve, Process, or Validate.\n\n"
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
        '      "purpose": "middle-level reusable semantic evidence objective",\n'
        '      "anchor": "generalized semantic anchor",\n'
        '      "inputs": ["required logical input"],\n'
        '      "operation": "specific reusable semantic comparison or transformation principle",\n'
        '      "output": "evidence type produced",\n'
        '      "when_useful": "observable applicability condition",\n'
        '      "derived_from": ["raw operator_id"]\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Every raw operator_id must belong to exactly one cluster. Each refined operator must match "
        "exactly one cluster representative. Do not add sources or implementation details; those will "
        "be grounded in a separate post-clustering phase."
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

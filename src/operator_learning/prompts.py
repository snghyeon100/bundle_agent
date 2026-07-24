"""LLM prompts for compact operator induction, clustering, and composition."""

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


def induction_prompt(
    case,
    source_capabilities,
    operator_count,
    *,
    text_only=True,
):
    semantic_case = _semantic_case(case, text_only)
    return (
        "You are the Operator Discovery Agent for a bundle-completion system.\n\n"
        f"{task_semantics(case.get('dataset'))}\n\n"
        "TASK DEFINITION\n"
        "A bundle is a set of items that belong together. At test time, the system receives an "
        "incomplete partial bundle and a finite candidate set, gathers evidence, and later predicts "
        "the best completion. This offline discovery case includes the ground-truth label solely as "
        "hindsight evidence for discovering reusable operations. The label will not be available at "
        "test time.\n\n"
        "OPERATOR DEFINITION\n"
        "An operator is a small reusable function used as one node when constructing an evidence "
        "strategy. It receives one logical artifact, performs one evidence-related transformation, "
        "and returns one logical artifact. A strategy exists only after a later composer connects "
        "multiple operators. Describe input and output in concise natural language; there is "
        "deliberately no predefined type catalog. The description must be concrete enough for a later "
        "workflow composer and code-generation agent to infer compatibility and implementation.\n\n"
        "An operator is not a complete strategy, does not name another operator, and does not specify "
        "execution order. Do not coordinate the returned operators into a prebuilt workflow. A source "
        "lookup by itself is not a discovered operator: when source data is needed, the operator must "
        "state the semantic transformation performed with that data. The output must be reusable "
        "intermediate evidence, a hypothesis, a request, a contrast, or a diagnostic—not a final "
        "candidate choice, rank, prediction, or score-only result.\n\n"
        "EXAMPLE\n"
        "A valid operator is: candidate-indexed evidence plus a bundle-intent hypothesis → separate "
        "shared support, candidate-exclusive support, and conflicts → candidate-indexed contrastive "
        "evidence. This is one function that can become a node in many strategies. Merely retrieving "
        "metadata is a source operation, not a discovered operator. Retrieving several sources, "
        "combining them, and choosing the best candidate is a complete strategy, not one operator.\n\n"
        "SOURCE CAPABILITIES\n"
        "The following manifest describes available evidence relations and operations, not mandatory "
        "implementation choices. In each operator's sources list, use only capability IDs from this "
        "manifest. Use [] when the operation needs no external capability. Sources support an "
        "operation; they do not define the operator.\n\n"
        f"{pretty_json(source_capabilities)}\n\n"
        "DISCOVERY OBJECTIVE\n"
        "Use the ground-truth candidate only as hindsight for discovering why some evidence would have "
        "been discriminative. Ground truth, gold labels, and the correct answer must never appear in "
        "an operator's name, objective, input, operation, or output because they are unavailable at "
        "test time. Discover independent atomic operations that could transform useful evidence in "
        "other cases. Generalize away from current product names, labels, IDs, and the exact answer. "
        "Avoid duplicate operations and avoid producing BundleCase-to-score evaluators.\n\n"
        f"Validation case:\n{pretty_json(semantic_case)}\n\n"
        f"Return JSON only with exactly {int(operator_count)} operators:\n"
        "{\n"
        '  "operators": [\n'
        "    {\n"
        '      "name": "ConcisePascalCaseName",\n'
        '      "objective": "single reusable semantic objective",\n'
        '      "input": "one concise logical input artifact",\n'
        '      "operation": "one central semantic transformation",\n'
        '      "output": "one concise reusable intermediate artifact",\n'
        '      "sources": ["exact capability ID from the manifest"]\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Use [] for sources when no external capability is needed. Return no explanation "
        "outside the JSON object."
    )


def clustering_prompt(raw_operators, dataset, min_operators, max_operators):
    return (
        "You are building a reusable operator library from compact case-induced operators. "
        "Cluster by semantic transformation and interface compatibility, not merely by shared source or "
        "surface wording.\n\n"
        "Two operators may merge only when their objective, input artifact, central operation, and "
        "output artifact express the same reusable transition. Similar sources alone do not justify "
        "a merge. Preserve useful source alternatives in sources. Input and output were written in "
        "natural language without a predefined catalog; generalize wording only when the underlying "
        "artifacts are genuinely equivalent. Avoid empty abstractions such as Analyze, Retrieve, "
        "Process, or Validate.\n\n"
        f"Dataset: {dataset}\n"
        f"Raw operators:\n{pretty_json(raw_operators)}\n\n"
        f"Create between {int(min_operators)} and {int(max_operators)} refined operators when the raw "
        "pool supports that many genuinely distinct transformations. Every raw operator_id must "
        "belong to exactly one cluster. Return JSON only:\n"
        "{\n"
        '  "clusters": [\n'
        "    {\n"
        '      "name": "functional and interface-compatible cluster name",\n'
        '      "rationale": "why the semantic function and contracts are equivalent",\n'
        '      "member_ids": ["raw operator_id"],\n'
        '      "representative": "RefinedOperatorName"\n'
        "    }\n"
        "  ],\n"
        '  "operators": [\n'
        "    {\n"
        '      "name": "RefinedOperatorName",\n'
        '      "objective": "single reusable semantic objective",\n'
        '      "input": "one concise logical input artifact",\n'
        '      "operation": "one central reusable transformation",\n'
        '      "output": "one concise reusable intermediate artifact",\n'
        '      "sources": ["capability ID preserved from raw members"],\n'
        '      "derived_from": ["raw operator_id"]\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Each refined operator must match exactly one cluster representative. The operator's "
        "derived_from values must exactly equal that cluster's member_ids."
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
        "operator library. This is rank-free test-time composition: no ground truth, validation "
        "label, candidate rank, or historical operator score is available.\n\n"
        f"{task_semantics(case.get('dataset'))}\n\n"
        "Infer bundle intent only from partial_items. Treat each operator's natural-language input and "
        "output as semantic interfaces. Connect operators when an earlier output supplies the "
        "artifact required by a later input. When descriptions differ but artifacts are semantically "
        "compatible, state the required adaptation explicitly in the step input; do not pretend that "
        "an incompatible interface is connected. Operator sources must be realizable from the "
        "available sources.\n\n"
        f"Case:\n{pretty_json(public_case)}\n\n"
        f"Available sources:\n{pretty_json(source_manifest)}\n\n"
        f"Operator library:\n{pretty_json(library)}\n\n"
        f"Create exactly {int(workflow_count)} structurally distinct workflows. Each workflow must use "
        "2 to 4 library operators and specify concrete artifact flow. Reusing an identical operator "
        "sequence with renamed prose is not distinct. Choose one recommended workflow only by case, "
        "contract, and source applicability.\n\n"
        "Return JSON only:\n"
        "{\n"
        '  "schema_version": "operator_workflow_candidates_v1",\n'
        '  "intent": "one concise intent inferred only from partial items",\n'
        '  "workflows": [\n'
        "    {\n"
        '      "name": "workflow name",\n'
        '      "operator_sequence": ["ExactLibraryOperatorName", "ExactLibraryOperatorName"],\n'
        '      "rationale": "why this operator composition fits the case",\n'
        '      "source_plan": "which available sources satisfy operator requirements",\n'
        '      "steps": [\n'
        "        {\n"
        '          "operator": "ExactLibraryOperatorName",\n'
        '          "objective": "case-specific use of the operator",\n'
        '          "input": "case data or exact prior artifact and any explicit adaptation",\n'
        '          "output": "artifact matching the operator output contract"\n'
        "        }\n"
        "      ]\n"
        "    }\n"
        "  ],\n"
        '  "recommended_workflow": "one workflow name from workflows"\n'
        "}"
    )

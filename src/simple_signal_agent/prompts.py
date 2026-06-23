import json

from progressive_signal_agent.common import candidate_labels, task_semantics


def _dump(value):
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def signal_code_prompt(
    case_view,
    source_manifest,
    output_file,
    max_evidence_chars,
    refinement_context=None,
):
    labels = ", ".join(candidate_labels(case_view))
    refinement_block = ""
    if refinement_context:
        refinement_block = (
            "\n\nThis is a refinement round. Produce a complete replacement evidence pack, not a delta. "
            "Preserve previous reliable signals when they remain useful, and address the evaluator's information "
            "requirements. You choose how to implement the improvement; do not copy the evaluator's wording as evidence."
            f"\n\nRefinement context:\n{_dump(refinement_context)}"
        )

    return (
        "You are the Signal Python Code Generator in a training-free bundle-completion system. Generate only complete "
        "executable Python code without markdown fences or explanation. The script runs with the allowed workspace as "
        "its current directory.\n\n"
        f"{task_semantics(case_view['dataset'])}\n\n"
        "Use the listed train-safe sources to discover and measure useful sample-specific signals for the partial bundle "
        "and every candidate. You may inspect, transform, join, aggregate, compare, or retrieve compact representative "
        "examples from the available sources. Signals must be factual outputs of executed code. Do not choose, rank, "
        "recommend, imply a preferred candidate, or create a final recommendation score. A large count or similarity is "
        "not by itself proof of compatibility. Use bundle_id only as a bundle entity, user IDs only as user entities, and "
        "item IDs only as item entities. Follow the current-bundle policy in the manifest.\n\n"
        f"Write UTF-8 JSON to exactly: {output_file}\n"
        f"Required candidate labels: {labels}\n"
        f"Keep the serialized JSON below approximately {int(max_evidence_chars)} characters. Include at most three compact "
        "representative evidence entries per candidate per signal. Attempt useful sources without crashing when an optional "
        "source is unavailable or has an unexpected serialized shape.\n\n"
        "Output exactly this minimal structure:\n"
        "{\n"
        '  "signals": [\n'
        "    {\n"
        '      "signal_name": "stable descriptive identifier",\n'
        '      "description": "what factual relationship or quantity was measured",\n'
        '      "sources": ["exact available source filename"],\n'
        '      "candidate_observations": {\n'
        '        "A": {"value": null, "evidence": []}\n'
        "      }\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Every signal must apply the same computation to every candidate label. `value` may be a JSON number, string, "
        "boolean, compact object/list, or null. `evidence` must be a JSON list of short factual observations or representative "
        "examples. Source names must exactly match names in the manifest. Do not add prediction, winner, ranking, preferred "
        "candidate, recommendation, or final-score fields.\n\n"
        f"ID-only case:\n{_dump(case_view)}\n\n"
        f"Source Capability Manifest:\n{_dump(source_manifest)}"
        f"{refinement_block}"
    )


def code_repair_prompt(
    case_view,
    source_manifest,
    previous_code,
    execution_and_validation,
    output_file,
):
    return (
        "You are repairing Python signal-extraction code. Return only complete executable Python code without markdown "
        "fences or explanation. Fix execution, safety-compatible implementation, JSON serialization, output path, or evidence "
        "schema defects. Preserve the intended signal investigation and candidate-symmetric computation. Do not turn this "
        "repair into a new research plan, and do not predict or rank candidates.\n\n"
        f"The repaired script must write UTF-8 JSON to exactly: {output_file}\n\n"
        f"ID-only case:\n{_dump(case_view)}\n\n"
        f"Source Capability Manifest:\n{_dump(source_manifest)}\n\n"
        f"Execution and validation defects:\n{_dump(execution_and_validation)}\n\n"
        f"Previous code:\n{previous_code}"
    )


def sufficiency_evaluation_prompt(
    case_view,
    source_manifest,
    generated_code,
    execution_summary,
    evidence,
    iteration,
    remaining_refinement_rounds,
    previous_evaluations,
):
    return (
        "You are the Signal Sufficiency Evaluator. Evaluate executed, source-grounded evidence only. Do not choose, rank, "
        "recommend, or imply a preferred candidate. Do not rewrite Python code. Your role is to decide whether the evidence "
        "can be passed to a separate Decision Agent or whether one evidence refinement is warranted.\n\n"
        "Assess candidate coverage, actual discrimination, zeros and ties, relevance to bundle compatibility, provenance, "
        "direct versus indirect grounding, redundancy, popularity or single-source dominance, conflicts, missingness, and "
        "possible confusion between similarity, compatibility, and redundancy. Do not require every source to be used. A "
        "large numeric margin alone is not sufficient.\n\n"
        "Use exactly one status:\n"
        "- SUFFICIENT: grounded evidence is adequate for a forced-choice Decision Agent.\n"
        "- REFINE: evidence is insufficient, a concrete unresolved information need exists, an allowed investigation can "
        "feasibly address it, and at least one refinement round remains.\n"
        "- INCONCLUSIVE: evidence is insufficient and another allowed refinement is unlikely to resolve it, or no refinement "
        "round remains.\n\n"
        "REFINE must not mean only that evidence is weak. State what new factual information is required and why different "
        "possible findings could change the evidence assessment. `required_improvements` must describe information needs, not "
        "Python code, fixed algorithms, or a prescribed retrieval recipe.\n\n"
        "Return JSON only:\n"
        "{\n"
        '  "status": "SUFFICIENT|REFINE|INCONCLUSIVE",\n'
        '  "evidence_quality": "NONE|LOW|MEDIUM|HIGH",\n'
        '  "reliable_signals": [],\n'
        '  "weak_or_failed_signals": [],\n'
        '  "coverage_problems": [],\n'
        '  "redundancy_problems": [],\n'
        '  "conflicts": [],\n'
        '  "evidence_gaps": [],\n'
        '  "required_improvements": [],\n'
        '  "expected_new_information": "",\n'
        '  "reason": ""\n'
        "}\n\n"
        f"Iteration index: {int(iteration)}\n"
        f"Remaining refinement rounds after this evaluation: {int(remaining_refinement_rounds)}\n\n"
        f"ID-only case:\n{_dump(case_view)}\n\n"
        f"Source Capability Manifest:\n{_dump(source_manifest)}\n\n"
        f"Generated code:\n{generated_code}\n\n"
        f"Execution summary:\n{_dump(execution_summary)}\n\n"
        f"Validated Evidence JSON:\n{_dump(evidence)}\n\n"
        f"Previous evaluations:\n{_dump(previous_evaluations)}"
    )


def decision_prompt(decision_case, evidence, evaluation, refinement_history):
    labels = ", ".join(candidate_labels(decision_case))
    return (
        "You are the final Decision Agent for bundle completion and the only component allowed to select a candidate. Use "
        "the deterministic item text and metadata together with the validated source-grounded evidence. Account for the "
        "sufficiency status, weak or conflicting signals, sparsity, and the difference between compatibility and mere "
        "similarity. This is forced choice even when evidence is inconclusive.\n\n"
        f"Choose exactly one label from: {labels}. Return JSON only with no explanation and no additional fields:\n"
        '{"prediction":"A"}\n\n'
        f"{task_semantics(decision_case['dataset'])}\n\n"
        f"Case with deterministic item text and metadata:\n{_dump(decision_case)}\n\n"
        f"Final validated Evidence JSON:\n{_dump(evidence)}\n\n"
        f"Signal Sufficiency Evaluation:\n{_dump(evaluation)}\n\n"
        f"Compact refinement history:\n{_dump(refinement_history)}"
    )

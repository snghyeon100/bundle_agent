import json

from progressive_signal_agent.common import candidate_labels, task_semantics


def _dump(value):
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def signal_code_prompt(
    case_view,
    source_manifest,
    affordance_graph,
    output_file,
    max_evidence_chars,
    refinement_context=None,
):
    labels = ", ".join(candidate_labels(case_view))
    refinement_block = ""
    if refinement_context:
        refinement_block = (
            "\n\nThis is a refinement round. Produce the new or revised signals discovered in this round. The runner "
            "will deterministically carry forward signals that the previous evaluator named in `reliable_signals`. If you "
            "recompute or improve a carried signal, reuse its exact `signal_name` so the current version replaces the old "
            "version instead of creating a duplicate. Re-examine the connectivity, grounding, dependencies, and risks in "
            "the Evidence Affordance Graph, then independently choose or derive an investigation that addresses the unresolved "
            "information need. No graph path or operation is prescribed. Make the investigation qualitatively deeper by "
            "reaching new grounded context; merely switching to a parallel raw count or similarity score is not refinement. "
            "Prefer expected information gain over path length. Do not copy the evaluator's wording as evidence."
            f"\n\nRefinement context:\n{_dump(refinement_context)}"
        )

    return (
        "You are the Signal Python Code Generator in a training-free bundle-completion system. Generate only complete "
        "executable Python code without markdown fences or explanation. The script runs with the allowed workspace as "
        "its current directory.\n\n"
        f"{task_semantics(case_view['dataset'])}\n\n"
        "For POG datasets, the evidence goal is fashion outfit compatibility and complementarity, not item "
        "resemblance.\n\n"
        "Use the listed train-safe sources to discover and measure useful sample-specific signals for the partial bundle "
        "and every candidate. You may inspect, transform, join, aggregate, compare, or retrieve compact representative "
        "examples from the available sources. Signals must be factual outputs of executed code. Do not choose, rank, "
        "recommend, imply a preferred candidate, or create a final recommendation score. A large count or similarity is "
        "not by itself proof of compatibility. Use bundle_id only as a bundle entity, user IDs only as user entities, and "
        "item IDs only as item entities. Follow the current-bundle policy in the manifest. The execution environment may be "
        "CPU-only: every .pt file must be loaded with torch.load(..., map_location=\"cpu\").\n\n"
        "Use the Evidence Affordance Graph as a soft semantic map of connectivity, grounding, source dependencies, and risks, "
        "not as a checklist, required workflow, or path allowlist. Infer the investigation yourself and freely derive other "
        "source-grounded relations when useful. Do not converge early on the first easy score when other reachable evidence "
        "could materially reduce uncertainty, but do not add sources or hops without a case-specific reason. Multiple parallel "
        "similarities or variants of the same count are not independent corroboration. The graph does not require every source "
        "to be used.\n\n"
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
        f"Source Capability Manifest:\n{_dump(source_manifest)}\n\n"
        f"Evidence Affordance Graph:\n{_dump(affordance_graph)}"
        f"{refinement_block}"
    )


def code_repair_prompt(
    case_view,
    source_manifest,
    previous_code,
    execution_and_validation,
    output_file,
):
    labels = ", ".join(candidate_labels(case_view))
    return (
        "You are repairing Python signal-extraction code. Return only complete executable Python code without markdown "
        "fences or explanation. Fix execution, safety-compatible implementation, JSON serialization, output path, or evidence "
        "schema defects. Preserve the intended signal investigation and candidate-symmetric computation. Do not turn this "
        "repair into a new research plan, and do not predict or rank candidates. The execution environment may be CPU-only: "
        "every .pt file must be loaded with torch.load(..., map_location=\"cpu\").\n\n"
        f"The repaired script must write UTF-8 JSON to exactly: {output_file}\n"
        f"Required candidate labels in every signal: {labels}\n"
        "The JSON object may contain only the top-level `signals` field. Preserve or restore this exact minimal schema:\n"
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
        "}\n"
        "Each signal object may contain only signal_name, description, sources, and candidate_observations. Every candidate "
        "observation may contain only value and evidence. `description` and `sources` are mandatory and must not be removed. "
        "Do not add case_id, dataset, bundle_id, partial_item_ids, candidates, prediction, ranking, recommendation, winner, "
        "or final_score to the evidence JSON.\n\n"
        f"ID-only case:\n{_dump(case_view)}\n\n"
        f"Source Capability Manifest:\n{_dump(source_manifest)}\n\n"
        f"Execution and validation defects:\n{_dump(execution_and_validation)}\n\n"
        f"Previous code:\n{previous_code}"
    )


def sufficiency_evaluation_prompt(
    case_view,
    source_manifest,
    affordance_graph,
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
        "For POG datasets, the evidence goal is fashion outfit compatibility and complementarity, not item "
        "resemblance.\n\n"
        "Assess candidate coverage, actual discrimination, zeros and ties, relevance to bundle compatibility, provenance, "
        "direct versus indirect grounding, redundancy, popularity or single-source dominance, conflicts, missingness, and "
        "possible confusion between similarity, compatibility, and redundancy. Do not require every source to be used. A "
        "large numeric margin alone is not sufficient. Treat computations from parallel embeddings or repeated variants of "
        "the same statistic as one signal family rather than independent corroboration. A signal counts as pointing to a "
        "candidate only when its measured direction has a grounded, decision-relevant interpretation; ties, all-zero values, "
        "raw popularity, and category equality without a compatibility rationale do not establish a preferred direction.\n\n"
        "Use the Evidence Affordance Graph only to understand what relational context was reachable. It is not a checklist: "
        "do not penalize an investigation merely for leaving sources or paths unused. Assess whether the code stopped at a "
        "shallow relation despite a feasible composition that could materially resolve the current uncertainty, and whether "
        "apparently independent signals actually reuse the same underlying source relation.\n\n"
        "Apply this strict approval gate: SUFFICIENT is allowed only when at least two reliable, genuinely independent, "
        "candidate-discriminating signal families converge on the same single candidate and no other reliable signal "
        "materially contradicts that convergence. Assess convergence internally without naming, ranking, recommending, or "
        "implying the candidate in your response. If reliable signals point to different candidates, only weak or tied "
        "signals agree, or fewer than two independent families provide a grounded direction, the evidence is not "
        "SUFFICIENT. Return REFINE when a deeper allowed investigation can feasibly resolve the disagreement or missing "
        "corroboration and refinement budget remains; otherwise return INCONCLUSIVE.\n\n"
        "Use exactly one status:\n"
        "- SUFFICIENT: grounded evidence is adequate for a forced-choice Decision Agent.\n"
        "- REFINE: evidence is insufficient, a concrete unresolved information need exists, an allowed investigation can "
        "feasibly address it, and at least one refinement round remains.\n"
        "- INCONCLUSIVE: evidence is insufficient and another allowed refinement is unlikely to resolve it, or no refinement "
        "round remains.\n\n"
        "REFINE must not mean only that evidence is weak. State what deeper factual information is required to test, "
        "corroborate, or overturn the current signals and why different possible findings could change the evidence "
        "assessment. Describe the unresolved evidence relationship or contextual gap, not a filename, similarity metric, "
        "Python implementation, fixed algorithm, or prescribed graph path. The next generator may navigate or derive any "
        "source-grounded relation that addresses that information need.\n\n"
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
        f"Evidence Affordance Graph:\n{_dump(affordance_graph)}\n\n"
        f"Generated code:\n{generated_code}\n\n"
        f"Execution summary:\n{_dump(execution_summary)}\n\n"
        f"Validated Evidence JSON:\n{_dump(evidence)}\n\n"
        f"Previous evaluations:\n{_dump(previous_evaluations)}"
    )


def decision_prompt(decision_case, evidence):
    labels = ", ".join(candidate_labels(decision_case))
    return (
        "You are the final Decision Agent for bundle completion and the only component allowed to select a candidate. Use "
        "only the deterministic item text and metadata together with the validated source-grounded Evidence JSON below. "
        "No evaluator conclusion, quality label, preferred candidate, or refinement summary is provided. Independently "
        "assess each signal's relevance, provenance, candidate coverage, sparsity, ties, conflicts, directness, and possible "
        "popularity bias. Distinguish bundle compatibility and complementary roles from mere item similarity or redundancy. "
        "Weak, all-zero, tied, indirect, or contradictory evidence should not override coherent item semantics. This remains "
        "a forced choice even when the evidence itself is sparse or uninformative.\n\n"
        f"Choose exactly one label from: {labels}. Return JSON only with no explanation and no additional fields:\n"
        '{"prediction":"A"}\n\n'
        f"{task_semantics(decision_case['dataset'])}\n\n"
        f"Case with deterministic item text and metadata:\n{_dump(decision_case)}\n\n"
        f"Final merged Evidence JSON:\n{_dump(evidence)}"
    )

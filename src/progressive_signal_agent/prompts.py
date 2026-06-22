import json

from .common import candidate_labels, task_semantics


def _dump(value):
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def broad_planning_prompt(case_view, source_manifest):
    return (
        "You are the Broad Signal Planner in a training-free bundle-completion system.\n"
        "Your role is coverage planning, not importance estimation. You have not observed the data yet, so do not "
        "claim that any signal is reliable, choose a winner, rank candidates, or write Python.\n\n"
        f"{task_semantics(case_view['dataset'])}\n\n"
        "Design a compact source-by-source plan that lets later code enrich the ID-only case with item metadata "
        "and extract broad factual candidate-scoped surface observations from every available source when feasible. "
        "The same factual computation must cover every candidate. Separate candidate occurrence from candidate-with-input "
        "relationships, and separate source absence from a zero observation. Treat the current bundle_id as a typed bundle "
        "entity and follow the manifest's current-bundle policy.\n\n"
        "Return JSON only:\n"
        "{\n"
        '  "coverage_goal": "...",\n'
        '  "source_tasks": [{"source":"...","factual_question":"...","candidate_scope":"all",'
        '"required_outputs":["..."],"limitations_to_record":["..."]}],\n'
        '  "output_requirements": ["candidate-scoped observations", "provenance", "representative examples", "limitations"]\n'
        "}\n\n"
        f"ID-only case:\n{_dump(case_view)}\n\n"
        f"Source Capability Manifest:\n{_dump(source_manifest)}"
    )


def broad_code_prompt(case_view, source_manifest, plan, output_file):
    labels = ", ".join(candidate_labels(case_view))
    return (
        "You are the Broad Signal Python Code Generator. Generate only executable Python code, without markdown fences "
        "or explanation. The script will run with the workspace as its current directory.\n\n"
        "Implement the fixed coverage plan over the listed train-safe sources. Do not predict, rank, recommend, create a "
        "winner, or combine observations into a final recommendation score. Use bundle_id only as a bundle entity and item "
        "IDs only as item entities. Enrich all input and candidate item IDs from item_info.json when available. Attempt every "
        "planned source and record failures rather than crashing.\n\n"
        f"Write UTF-8 JSON to exactly: {output_file}\n"
        f"Required candidate labels: {labels}\n"
        "Use this output schema:\n"
        "{\n"
        '  "case_profile": {"bundle_id": 0, "input_items": [], "candidate_items": []},\n'
        '  "source_attempts": [{"source":"...","status":"used|unavailable|failed","details":"..."}],\n'
        '  "observations": [{"observation_id":"...","source":"...","scope":"input|candidate:A|context",'
        '"kind":"...","value":null,"examples":[],"basis":"...","limitations":[]}],\n'
        '  "warnings": []\n'
        "}\n"
        "Every candidate must have one or more separate observation objects using exact scopes candidate:A, candidate:B, "
        "and so on. Do not hide all candidate values inside one aggregate dictionary. Keep examples compact. Loading a file, "
        "reporting row counts, or reporting tensor shape is only a diagnostic and does not replace sample-specific extraction.\n\n"
        f"Case:\n{_dump(case_view)}\n\n"
        f"Manifest:\n{_dump(source_manifest)}\n\n"
        f"Fixed broad plan:\n{_dump(plan)}"
    )


def code_repair_prompt(case_view, source_manifest, fixed_specification, previous_code, execution, output_file, stage):
    return (
        f"You are repairing {stage} Python evidence code. Generate only complete executable Python code without markdown "
        "fences or explanation. Preserve the fixed research/coverage specification exactly; change only implementation or "
        "output-contract defects. Do not replace the investigation with an easier count, lookup, similarity, or new plan. "
        "Do not predict or rank candidates.\n\n"
        f"The script must write valid UTF-8 JSON to: {output_file}\n\n"
        f"Case:\n{_dump(case_view)}\n\n"
        f"Manifest:\n{_dump(source_manifest)}\n\n"
        f"Fixed specification:\n{_dump(fixed_specification)}\n\n"
        f"Execution/validation issues:\n{_dump(execution)}\n\n"
        f"Previous code:\n{previous_code}"
    )


def diagnosis_prompt(case_view, source_manifest, accumulated_evidence, execution_summaries):
    return (
        "You are the Signal Diagnosis Agent. Assess executed evidence, not item-text intuition. Do not choose, rank, "
        "recommend, or imply a preferred candidate. Report what is reliable, what failed, and what remains unresolved. "
        "Do not prescribe an investigation method: do not tell the next planner which retrieval algorithm, graph path, "
        "similarity method, smoothing method, or source join to use. Frame evidence gaps as factual questions with competing "
        "explanations.\n\n"
        "Check candidate/source coverage, missingness, zeros and ties, discrimination, direct versus indirect evidence, "
        "redundancy, popularity dominance, source conflict, provenance, plan fulfillment, and possible confusion between "
        "similarity, compatibility, and redundancy.\n\n"
        "Return JSON only:\n"
        "{\n"
        '  "status": "USABLE|NEEDS_DEEPENING|STOP_INCONCLUSIVE",\n'
        '  "evidence_quality": "none|low|medium|high",\n'
        '  "reliable_observations": [],\n'
        '  "observed_failures": [],\n'
        '  "unresolved_questions": [{"question":"...","competing_explanations":["...","..."]}],\n'
        '  "evidence_gaps": [],\n'
        '  "conflicts": [],\n'
        '  "signals_to_downweight": [],\n'
        '  "candidate_coverage": {},\n'
        '  "stop_reason": ""\n'
        "}\n\n"
        f"Case:\n{_dump(case_view)}\n\n"
        f"Manifest:\n{_dump(source_manifest)}\n\n"
        f"Executed evidence:\n{_dump(accumulated_evidence)}\n\n"
        f"Execution summaries:\n{_dump(execution_summaries)}"
    )


def deep_planning_prompt(case_view, source_manifest, accumulated_evidence, diagnosis, previous_plans):
    labels = ", ".join(candidate_labels(case_view))
    return (
        "You are the Open-Ended Deep Research Planner. Design executable investigations that resolve the diagnosis's "
        "specific evidence gaps. Do not write Python, choose a winner, rank candidates, or prescribe a final score.\n\n"
        "Do not merely produce something called a deep signal. Internally generate multiple distinct proposals, compare them, "
        "and select a small non-redundant portfolio. Judge proposals by novelty from existing observations, expected information "
        "gain, candidate discrimination, grounding, robustness, independence, coverage, and execution feasibility. Use the "
        "manifest's entities, relations, and generic transformations as compositional primitives, not as completed recipes. "
        "Invent the investigation for this case.\n\n"
        "Each selected investigation must state competing explanations and how positive versus negative outcomes would change "
        "the interpretation. Reject a proposal if both outcomes mean the same thing. Do not repeat file diagnostics, row counts, "
        "tensor shapes, existing direct counts, unchanged cosine similarity, renamed/reweighted surface observations, or aggregate "
        "values without candidate-scoped provenance. A simple investigation is allowed if it adds genuinely new grounded evidence.\n\n"
        f"Cover all candidate labels with the same computation: {labels}. Return JSON only:\n"
        "{\n"
        '  "research_objective": "...",\n'
        '  "investigations": [{\n'
        '    "investigation_id":"I1","question":"...","competing_explanations":["...","..."],\n'
        '    "why_needed":"...","sources_used":["..."],"derivation_path":["..."],\n'
        '    "method_design":"...","new_information":"...",\n'
        '    "possible_outcomes":{"positive":"...","negative":"..."},\n'
        '    "distinction_from_surface":"...","expected_candidate_scope":"all candidates",\n'
        '    "failure_condition":"..."\n'
        '  }],\n'
        '  "portfolio_rationale":"...",\n'
        '  "stop_condition":"..."\n'
        "}\n\n"
        f"Case:\n{_dump(case_view)}\n\n"
        f"Manifest:\n{_dump(source_manifest)}\n\n"
        f"Existing executed evidence:\n{_dump(accumulated_evidence)}\n\n"
        f"Diagnosis:\n{_dump(diagnosis)}\n\n"
        f"Previous deep plans to avoid repeating:\n{_dump(previous_plans)}"
    )


def deep_code_prompt(case_view, source_manifest, accumulated_evidence, diagnosis, plan, output_file):
    labels = ", ".join(candidate_labels(case_view))
    return (
        "You are the Deep Signal Python Code Generator. Generate only complete executable Python code without markdown fences "
        "or explanation. Implement the fixed research specification faithfully. Do not replace a difficult investigation with "
        "an easier lookup or an existing surface observation. Do not predict, rank, recommend, or create a final score.\n\n"
        f"Write UTF-8 JSON to exactly: {output_file}\n"
        f"Required candidate labels: {labels}\n"
        "Output schema:\n"
        "{\n"
        '  "investigations": [{"investigation_id":"I1","question":"...","status":"completed|partial|failed",\n'
        '    "method_summary":"...","sources_used":["..."],\n'
        '    "observations":[{"source":"...","scope":"candidate:A|input|context","kind":"...",\n'
        '      "value":null,"related_ids":[],"examples":[],"basis":"...","limitations":[]}],\n'
        '    "limitations":[]}],\n'
        '  "plan_fulfillment": [{"investigation_id":"I1","status":"completed|partial|failed","details":"..."}],\n'
        '  "warnings": []\n'
        "}\n"
        "Every completed or partial investigation must emit a separate observation for every candidate label using exact "
        "candidate scopes. Shared context observations may be added but cannot replace candidate observations. Keep retrieved "
        "examples compact and human-readable by enriching item IDs from item_info when available. Record factual absence rather "
        "than inventing variation.\n\n"
        f"Case:\n{_dump(case_view)}\n\n"
        f"Manifest:\n{_dump(source_manifest)}\n\n"
        f"Existing evidence (do not repeat):\n{_dump(accumulated_evidence)}\n\n"
        f"Diagnosis:\n{_dump(diagnosis)}\n\n"
        f"Fixed deep research specification:\n{_dump(plan)}"
    )


def decision_prompt(case_view, evidence_json):
    labels = ", ".join(candidate_labels(case_view))
    return (
        "You are the final Decision Agent for bundle completion. You are the only stage allowed to choose a candidate. "
        "Use the source-grounded case profile and verified evidence. Account for evidence quality, sparsity, conflicts, "
        "downweighted observations, provenance, and the difference between compatibility and mere similarity. Do not treat "
        "the number of observations as evidence strength. If evidence is weak, use task semantics and enriched item metadata "
        "carefully and lower confidence.\n\n"
        f"Choose exactly one of these labels: {labels}. Return JSON only:\n"
        "{\n"
        '  "prediction":"A",\n'
        '  "reasoning":"...",\n'
        '  "confidence":"low|medium|high",\n'
        '  "evidence_quality_used":"none|low|medium|high",\n'
        '  "observations_used":["..."],\n'
        '  "downweighted_or_ignored":["..."]\n'
        "}\n\n"
        f"{task_semantics(case_view['dataset'])}\n\n"
        f"ID-only case:\n{_dump(case_view)}\n\n"
        f"Verified Progressive Signal Evidence JSON:\n{_dump(evidence_json)}"
    )

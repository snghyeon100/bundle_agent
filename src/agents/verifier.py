import json

from agents.common import build_agent_sample_view, task_semantics


def format_prompt_value(value):
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=2)


def generate_verifier_prompt(sample, planning_text, evidence_json, execution_summary, round_number, remaining_rounds):
    sample_view = build_agent_sample_view(sample)
    return (
        "You are the evidence and abstraction critic for a four-stage recommendation agent.\n"
        "Your job is to evaluate whether the compact evidence JSON is complete, discriminative, and reliable. "
        "Do not choose the final answer.\n\n"
        "Dataset/task semantics:\n"
        f"{task_semantics(sample.get('dataset', ''))}\n\n"
        f"Round: {round_number}\n"
        f"Remaining retrieval rounds after this verifier decision: {remaining_rounds}\n\n"
        "Current sample JSON:\n"
        f"{json.dumps(sample_view, ensure_ascii=False, indent=2)}\n\n"
        "Planner output:\n"
        f"{format_prompt_value(planning_text)}\n\n"
        "Code execution summary:\n"
        f"{json.dumps(execution_summary, ensure_ascii=False, indent=2)}\n\n"
        "Retrieved evidence JSON:\n"
        f"{json.dumps(evidence_json, ensure_ascii=False, indent=2)}\n\n"
        "Critique the actual evidence JSON, not just the natural-language summary. "
        "Inspect candidate signals, numeric gaps, ties, all-zero outputs, missing values, failed analyses, and provenance. "
        "Embedding-only or single-signal evidence should usually be at most medium unless the numeric separation is clear and the analysis is well aligned with the task. "
        "If a signal ranks candidates but the top candidates are very close, mark the signal as tie-heavy or low-margin. "
        "If the code failed or produced no usable evidence, request repair or replanning through the JSON fields.\n\n"
        "Check for these failure modes: missing candidate evidence, code failure, all-zero or tie-heavy signals, "
        "low-margin numeric comparisons, contradictory evidence, and provenance that is too weak or missing.\n\n"
        "When needs_replanning is true, suggest deeper or more creative follow-up work that remains implementable from allowed files: new joins, alternate abstraction levels, "
        "candidate subsets to inspect, cross-source combinations, or transformations that could make weak broad-sweep signals more discriminative.\n\n"
        "Return only valid JSON using this schema:\n"
        "{\n"
        '  "round": 1,\n'
        '  "evidence_sufficient": false,\n'
        '  "evidence_quality": "none|weak|medium|strong",\n'
        '  "failure_modes": ["bi_scores_all_zero"],\n'
        '  "candidate_coverage": {"all_candidates_present": true, "missing_candidates": []},\n'
        '  "numeric_diagnostics": [\n'
        '    {"signal": "...", "status": "failed|all_zero|tie_heavy|low_margin|useful", "best_labels": ["A"], "reason": "..."}\n'
        '  ],\n'
        '  "useful_signals": ["..."],\n'
        '  "weak_signals": ["..."],\n'
        '  "downweight_recommendations": ["..."],\n'
        '  "needs_replanning": true,\n'
        '  "suggested_investigation_focus": ["..."],\n'
        '  "verifier_summary": "..."\n'
        "}\n"
    )

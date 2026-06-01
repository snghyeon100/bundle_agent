import json

from agents.common import build_agent_sample_view, task_semantics
from agents.workspace import workspace_view


def generate_planning_prompt(sample, workspace, evidence_output_file, previous_rounds=None, verifier_feedback=None):
    sample_view = build_agent_sample_view(sample)
    previous_rounds = previous_rounds or []
    round_number = len(previous_rounds) + 1
    if previous_rounds:
        task_limit_instruction = (
            "This is a focused follow-up round. Plan a small number of deeper or more creative analyses based on verifier feedback. "
            "Prefer source combinations, alternate abstraction levels, and targeted transformations over repeating broad scans."
        )
    else:
        task_limit_instruction = (
            "This is the first round. Plan a broad but still practical sweep over useful sources and signals. "
            "Do not treat each task as one source; prefer analyses that combine sources when the combination could produce a stronger candidate-level signal."
        )
    replan_instruction = ""
    if verifier_feedback:
        replan_instruction = (
            "\nThis is a re-planning round. Do not simply repeat the previous plan. "
            "Use the initial plan, all previous evidence, and verifier feedback to revise the investigation design.\n"
            "Treat suggested_investigation_focus or suggested_retrieval_focus as the main clue for what to try next. "
            "Avoid or explicitly downweight weak_signals or downweight_recommendations unless the verifier specifically asks to repair them. "
            "If the verifier identified all-zero, tie-heavy, non-discriminative, or failed signals, change the abstraction level or analysis idea rather than repeating the same computation. "
            "Look for deeper candidate-level evidence through cross-source joins, graph inversions, neighborhood aggregation, category/style abstractions, or other transformations that remain grounded in allowed files. "
            "Explain how the revised analysis_tasks address each verifier failure mode.\n"
        )

    return (
        "You are the investigation designer for a bundle completion research system.\n"
        "Your job is to design sample-specific analyses that a later code-writing agent can run over raw allowed files. "
        "Do not write code and do not choose the final answer. "
        "Be creative and deep within the available raw files: design signals that transform, join, invert, aggregate, or re-abstract the data "
        "to expose candidate-level differences that are not obvious from a single file. "
        "Keep the plan practical, but collect enough diverse evidence for the verifier to decide what deserves deeper follow-up.\n\n"
        "Dataset/task semantics:\n"
        f"{task_semantics(sample.get('dataset', ''))}\n\n"
        f"Retrieval round: {round_number}\n"
        "Current sample JSON:\n"
        f"{json.dumps(sample_view, ensure_ascii=False, indent=2)}\n\n"
        "Allowed workspace. Later generated code may inspect files only under this workspace:\n"
        f"{json.dumps(workspace_view(workspace, evidence_output_file), ensure_ascii=False, indent=2)}\n\n"
        f"Previous compact retrieval rounds:\n{json.dumps(previous_rounds, ensure_ascii=False, indent=2)}\n\n"
        f"Verifier feedback for re-planning:\n{json.dumps(verifier_feedback or {}, ensure_ascii=False, indent=2)}\n"
        f"{replan_instruction}\n"
        "Important restrictions:\n"
        "- Do not use or request bi_full.txt, any test/validation ground-truth file such as bi_test_gt.txt, or any true labels.\n"
        "- Do not use prior result CSV files, predictions, hits, or true option labels.\n"
        "- When planning BI/UI analysis, remember that interaction files are comma-space separated rows: context_id, item_id, item_id, ...\n"
        "- item_info.json is keyed by string item_id, and tensor feature caches are indexed by integer item_id when dimensions match item count.\n"
        "- Prefer sample-specific evidence. Different samples may need different abstractions and data combinations.\n"
        "- Prefer fused analyses when useful, such as BI co-affiliation x item metadata, UI overlap x item category, metadata groupings x train-bundle frequency, or embeddings x relational evidence. "
        "You may also consider multi-hop graph-style patterns such as BI x IB (bundle -> items -> neighboring bundles/items), "
        "IB x BI (item -> bundles -> neighboring items), or IU x UI (item -> users -> neighboring items). "
        "These are examples of the style of thinking, not fixed recipes.\n\n"
        f"{task_limit_instruction}\n"
        "Return only valid compact JSON. Keep it concise so the object is complete. "
        "Do not use markdown, comments, trailing commas, or extra text outside JSON. "
        "Each string should be short enough that the full JSON can close properly.\n"
        "Schema:\n"
        "{\n"
        '  "round": 1,\n'
        '  "goal": "what evidence would distinguish the candidates",\n'
        '  "analysis_tasks": [\n'
        '    {\n'
        '      "name": "...",\n'
        '      "question": "what this tests",\n'
        '      "files": ["item_info.json", "bi_train.txt"],\n'
        '      "method": "short operation description",\n'
        '      "candidate_signal": "what candidate-level value or text should be produced"\n'
        '    }\n'
        '  ],\n'
        '  "avoid": ["weak or repeated analyses to avoid"]\n'
        "}\n"
    )

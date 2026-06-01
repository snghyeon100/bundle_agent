import json

from agents.common import build_agent_sample_view, candidate_labels, task_semantics
from agents.workspace import workspace_view


def generate_code_prompt(sample, planning_text, workspace, evidence_output_file, conf):
    sample_view = build_agent_sample_view(sample)
    max_stdout_chars = int(conf.get("agent_code_max_stdout_chars", 20000))
    labels = candidate_labels(sample)
    return (
        "You are the programmatic evidence builder for a bundle completion task.\n"
        "Write Python code that implements the planner's investigation ideas over the allowed raw files. "
        "The code will be saved and executed by the runner inside the allowed workspace.\n"
        f"The code must write JSON to {evidence_output_file} and also print the same JSON object to stdout.\n\n"
        "Dataset/task semantics:\n"
        f"{task_semantics(sample.get('dataset', ''))}\n\n"
        "Planning agent output:\n"
        f"{planning_text}\n\n"
        "Current sample JSON:\n"
        f"{json.dumps(sample_view, ensure_ascii=False, indent=2)}\n\n"
        "Allowed workspace files. Use these relative paths only:\n"
        f"{json.dumps(workspace_view(workspace, evidence_output_file), ensure_ascii=False, indent=2)}\n\n"
        "Rules:\n"
        "- You may use Python standard library and installed scientific packages if available.\n"
        "- Parse bi_train.txt and ui_full.txt with line.strip().split(', '), not split(). The first value is the context id; remaining values are item ids.\n"
        "- For direct BI co-affiliation, a candidate co-occurs with an input item when both item ids appear in the same bi_train.txt line.\n"
        "- item_info.json is a dict keyed by string item_id; do not treat it as a list of objects.\n"
        "- If a .pt feature file is a torch.Tensor with shape [num_items, dim], use features[item_id] directly.\n"
        "- You may create fused analyses by joining metadata, train bundle affiliations, user interactions, and embedding caches. "
        "Do not limit yourself to one source per signal if combining files gives a better sample-specific test.\n"
        "- Favor deeper candidate-level evidence when feasible: transform raw rows into inverted indexes, neighborhoods, category/style abstractions, "
        "or aggregated compatibility signals instead of only reporting direct counts.\n"
        "- For graph-style fused analyses, you may build inverted maps such as item->train_bundles from bi_train.txt or item->users from ui_full.txt, "
        "then aggregate neighboring items, categories, metadata fields, or embedding similarities before scoring candidates. "
        "Use these as optional patterns, not mandatory recipes.\n"
        "- Treat common operations such as direct co-affiliation, co-purchase, category/attribute aggregation, and cosine similarity as possible tools, not a fixed recipe.\n"
        "- In the first retrieval round, compute a broad set of distinct candidate-level signals when the plan asks for them, including cross-source signals when useful. "
        "In later rounds, prioritize deeper or more creative follow-up analyses suggested by verifier feedback.\n"
        "- Do not read bi_full.txt, bi_test_gt.txt, validation/test ground truth files, result CSV files, or any file containing predictions/hits/true labels.\n"
        "- Do not use item_cf_feature.pt unless it is explicitly listed in the allowed workspace files.\n"
        "- Do not access files outside data/ and output/.\n"
        "- Be robust: if a source cannot be loaded, record that in warnings and continue.\n"
        f"- Keep the printed JSON under about {max_stdout_chars} characters.\n"
        "- Do not choose the final answer; only provide evidence and optional preliminary evidence ranking.\n"
        "- Do not add generic evidence_for claims when a signal is zero, tied, unavailable, or failed; put those issues in evidence_against or warnings.\n"
        f"- Include every candidate label exactly once: {', '.join(labels)}.\n\n"
        "The JSON must use this compact schema. Keep values short and candidate-focused:\n"
        "{\n"
        '  "summary": "...",\n'
        '  "raw_files_used": ["item_info.json", "bi_train.txt"],\n'
        '  "numeric_comparisons": [\n'
        '    {"signal": "...", "values": {"A": 0.0, "B": 0.0}, "best_labels": ["A"], "note": "..."}\n'
        '  ],\n'
        '  "candidate_evidence": {\n'
        '    "A": {\n'
        '      "summary": "...",\n'
        '      "evidence_for": ["..."],\n'
        '      "evidence_against": ["..."],\n'
        '      "signals": {"custom_signal_name": "numeric or short text"},\n'
        '      "provenance": ["bi_train.txt: co_affiliation"]\n'
        "    }\n"
        "  },\n"
        '  "warnings": ["..."]\n'
        "}\n\n"
        "Return only the Python code. Do not wrap it in explanation."
    )


def generate_code_repair_prompt(sample, planning_text, previous_code, execution_result, workspace, evidence_output_file):
    sample_view = build_agent_sample_view(sample)
    labels = candidate_labels(sample)
    return (
        "You are repairing Python evidence-building code for a bundle completion task.\n"
        "The previous code failed, was blocked, or did not produce valid JSON. Write a corrected complete Python script.\n"
        f"Write JSON to {evidence_output_file} and print the same JSON object to stdout.\n\n"
        "Dataset/task semantics:\n"
        f"{task_semantics(sample.get('dataset', ''))}\n\n"
        "Current sample JSON:\n"
        f"{json.dumps(sample_view, ensure_ascii=False, indent=2)}\n\n"
        "Planning agent output:\n"
        f"{planning_text}\n\n"
        "Allowed workspace files:\n"
        f"{json.dumps(workspace_view(workspace, evidence_output_file), ensure_ascii=False, indent=2)}\n\n"
        "Previous code:\n"
        f"```python\n{previous_code[:12000]}\n```\n\n"
        "Execution result:\n"
        f"{json.dumps(execution_result, ensure_ascii=False, indent=2)}\n\n"
        "Repair requirements:\n"
        "- Do not read bi_full.txt, bi_test_gt.txt, validation/test ground truth files, result CSV files, or true labels.\n"
        "- Parse bi_train.txt/ui_full.txt with split(', '), item_info.json as a dict keyed by string item_id, and tensor features by integer item_id.\n"
        "- Use robust fallbacks if optional embedding files cannot be loaded.\n"
        "- Do not access files outside data/ and output/.\n"
        f"- Include every candidate label exactly once: {', '.join(labels)}.\n"
        "- Keep candidate_evidence, numeric_comparisons, and warnings. Do not add extra large raw dumps.\n"
        "- If a previous planned analysis cannot be implemented, record the limitation in warnings instead of fabricating evidence.\n\n"
        "Return only the corrected Python code."
    )

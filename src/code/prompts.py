"""Prompt builders for the two-step code method."""

from .common import candidate_labels, pretty_json, task_semantics


def _unified_case_context(case_view, semantic_case=None):
    semantic_case = semantic_case or {}
    partial_text = {
        int(item.get("item_id")): item
        for item in semantic_case.get("partial_items", [])
        if isinstance(item, dict) and item.get("item_id") is not None
    }
    candidate_text = {
        str(item.get("label", "")): item
        for item in semantic_case.get("candidates", [])
        if isinstance(item, dict)
    }
    unified = {
        "case_id": case_view.get("case_id"),
        "dataset": case_view.get("dataset"),
        "bundle_id": case_view.get("bundle_id"),
        "partial_items": [],
        "candidates": [],
    }
    for item_id in case_view.get("partial_item_ids", []):
        source = partial_text.get(int(item_id), {})
        entry = {"item_id": int(item_id), "text": source.get("text", "")}
        unified["partial_items"].append(entry)
    for candidate in case_view.get("candidates", []):
        label = str(candidate.get("label", ""))
        item_id = int(candidate.get("item_id"))
        source = candidate_text.get(label, {})
        entry = {"label": label, "item_id": item_id, "text": source.get("text", "")}
        unified["candidates"].append(entry)
    return unified


def _schema_text(case_view):
    partial_entries = []
    for item_id in case_view.get("partial_item_ids", []):
        key = f"partial_{int(item_id)}"
        partial_entries.append(
            f'    "{key}": {{"item_id": {int(item_id)}, "evidence": ["short relation-grounded evidence string"]}}'
        )
    candidate_entries = []
    for candidate in case_view.get("candidates", []):
        label = str(candidate.get("label"))
        item_id = int(candidate.get("item_id"))
        candidate_entries.append(
            f'    "{label}": {{"item_id": {item_id}, "evidence": ["short relation-grounded evidence string"]}}'
        )
    return (
        "{\n"
        '  "schema_version": "code_evidence_v1",\n'
        '  "strategies": [\n'
        "    {\n"
        '      "name": "ib_x_bi_cobundle_context",\n'
        '      "relation_signal": "BI relation: item -> train bundles -> co-occurring items/context",\n'
        '      "data_sources": ["bi_train.txt", "item_info.json"],\n'
        '      "description": "one short sentence about what this strategy retrieves"\n'
        "    }\n"
        "  ],\n"
        '  "partial_evidence": {\n'
        + ",\n".join(partial_entries)
        + "\n  },\n"
        '  "candidate_evidence": {\n'
        + ",\n".join(candidate_entries)
        + "\n  },\n"
        '  "policy_trace": {\n'
        '    "implemented_strategies": ["strategy name -> concrete relation path implemented"],\n'
        '    "skipped_strategies": ["strategy/view -> concrete source or sparsity reason"],\n'
        '    "notes": ["short implementation or fallback note"]\n'
        "  }\n"
        "}"
    )


def code_generation_prompt(case_view, source_manifest, output_file, semantic_case=None):
    labels = ", ".join(candidate_labels(case_view))
    partial_keys = [f"partial_{int(item_id)}" for item_id in case_view.get("partial_item_ids", [])]
    candidate_keys = [str(candidate.get("label")) for candidate in case_view.get("candidates", [])]
    required_keys = ", ".join(partial_keys + candidate_keys)
    unified_case = _unified_case_context(case_view, semantic_case)

    return (
        "You are the Code Generation Agent for a bundle-completion evidence pipeline.\n\n"
        "Generate ONLY complete executable Python code. Do not use markdown fences and do not "
        "include explanations outside the code.\n\n"
        "The code will run in a prepared workspace whose current working directory contains "
        "the listed data sources under `data/`. The script must read only those listed sources "
        "and write UTF-8 JSON to exactly this relative path:\n"
        f"{output_file}\n\n"
        f"{task_semantics(case_view.get('dataset'))}\n\n"
        "The final system will later choose which candidate item should be added to the partial "
        "bundle. Your code must NOT choose, rank, score, recommend, or reveal a final prediction. "
        "Your code only retrieves compact source-grounded evidence that can be attached next to "
        "the partial item texts and candidate item texts in a later prediction prompt.\n\n"
        "Problem instance with IDs and text only. Do not assume or use hidden category labels:\n"
        f"{pretty_json(unified_case)}\n\n"
        "Available data sources and their relation contracts:\n"
        f"{pretty_json(source_manifest)}\n\n"
        "RELATION-SIGNAL STRATEGY REQUIREMENTS\n"
        "Use the data sources as typed relation signals. Derive the retrieval strategies yourself "
        "from this specific partial bundle, candidate set, dataset, and source manifest. Trust your "
        "own ability to decide which relation paths will expose useful evidence for this sample. "
        "The strategy logic should be shared across the whole bundle problem whenever possible; do "
        "not invent a separate private method for each item. It is fine for a strategy to return "
        "sparse evidence for some items.\n\n"
        "One example of the style of strategy you may use or adapt:\n"
        "- IB x BI co-bundle context: treat the BI train source as a relation signal "
        "item -> train bundles -> co-occurring items/context. For a target item, find train "
        "bundles containing that item, then retrieve representative co-occurring item "
        "titles/context from those bundles.\n\n"
        "That is only an example, not a required strategy and not a complete strategy list. "
        "Do not stop by copying only this example. Implement at least three strategies in total. "
        "In addition to any use of this example, inspect the source manifest and design at least "
        "two sample-adaptive strategies yourself for this specific bundle problem. Each "
        "self-designed strategy should name a concrete relation path from available sources and "
        "explain why that path is useful for this partial/candidate set. "
        "Possible source families include item text metadata, UI relations, content/description "
        "embeddings, CF/LightGCN features, Spotify artist/album context, or keyword-filtered "
        "bundle context, but use them only when they fit this problem and are available. "
        "Load .pt files on CPU with map_location='cpu'.\n\n"
        "EVIDENCE RULES\n"
        "- Evidence strings must be short, source-grounded, and parseable by a later pipeline.\n"
        "- Each evidence string should name the relation path/source signal, then give compact "
        "representative retrieved titles, counts, relation context, or a sparse-evidence note.\n"
        "- Do not use category metadata as a strategy input or evidence output. If item_info.json "
        "contains fields such as cate, cate_id, category, category_id, genre category, or item "
        "class labels, ignore those fields.\n"
        "- Include at most 5 representative titles/items per evidence string and append a compact "
        "count note when there are many more.\n"
        "- Keep evidence neutral. Do not say that a candidate is best, correct, compatible, "
        "incompatible, likely answer, or ranked above another candidate.\n"
        "- Populate evidence for every required partial key and candidate label. Use sparse notes "
        "instead of omitting keys when a relation signal has no support.\n\n"
        f"Required labels: {labels}\n"
        f"Required evidence keys: {required_keys}\n\n"
        "The JSON written by the script must match this schema exactly enough for deterministic "
        "parsing. Extra top-level fields are not allowed:\n"
        f"{_schema_text(case_view)}\n\n"
        "HIGH-LEVEL SCRIPT SKELETON TO FOLLOW\n"
        "You may change helper internals, but keep this shape and the same final JSON schema:\n"
        "```python\n"
        "import json, os\n"
        "\n"
        f"OUTPUT_PATH = r\"{output_file}\"\n"
        "PARTIAL_ITEMS = [{\"key\": \"partial_123\", \"item_id\": 123, \"text\": \"...\"}]\n"
        "CANDIDATES = [{\"label\": \"A\", \"item_id\": 456, \"text\": \"...\"}]\n"
        "STRATEGY_PLAN = [\n"
        "    {\n"
        "        \"name\": \"ib_x_bi_cobundle_context\",\n"
        "        \"relation_signal\": \"item -> BI train bundles -> co-items\",\n"
        "        \"data_sources\": [\"bi_train.txt\", \"item_info.json\"],\n"
        "        \"description\": \"Example strategy: retrieve train-bundle co-occurrence context for every target item.\",\n"
        "    },\n"
        "    # Add at least two self-designed, sample-adaptive strategies from the source manifest.\n"
        "    # The final STRATEGY_PLAN must contain at least three strategies in total.\n"
        "]\n"
        "\n"
        "def main():\n"
        "    sources = load_sources()\n"
        "    indexes = build_indexes(sources)\n"
        "    partial_evidence = {item['key']: {'item_id': item['item_id'], 'evidence': []} for item in PARTIAL_ITEMS}\n"
        "    candidate_evidence = {item['label']: {'item_id': item['item_id'], 'evidence': []} for item in CANDIDATES}\n"
        "    policy_trace = {'implemented_strategies': [], 'skipped_strategies': [], 'notes': []}\n"
        "    for strategy in STRATEGY_PLAN:\n"
        "        apply_strategy(strategy, indexes, partial_evidence, candidate_evidence, policy_trace)\n"
        "    fill_sparse_notes(partial_evidence, candidate_evidence)\n"
        "    write_json({\n"
        "        'schema_version': 'code_evidence_v1',\n"
        "        'strategies': STRATEGY_PLAN,\n"
        "        'partial_evidence': partial_evidence,\n"
        "        'candidate_evidence': candidate_evidence,\n"
        "        'policy_trace': policy_trace,\n"
        "    }, OUTPUT_PATH)\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
        "```\n\n"
        "In your actual code, set OUTPUT_PATH to the exact required output path above. "
        "Do not emit partial code."
    )


def _short_evidence_lines(values, max_items=4, max_chars=900):
    if not isinstance(values, list):
        values = []
    lines = []
    for value in values[:max_items]:
        text = " ".join(str(value or "").split())
        if text:
            lines.append(text[:max_chars])
    return lines


def _partial_evidence(evidence, item_id):
    partials = evidence.get("partial_evidence", {}) if isinstance(evidence, dict) else {}
    payload = partials.get(f"partial_{int(item_id)}") if isinstance(partials, dict) else None
    return _short_evidence_lines(payload.get("evidence", []) if isinstance(payload, dict) else [])


def _candidate_evidence(evidence, label):
    candidates = evidence.get("candidate_evidence", {}) if isinstance(evidence, dict) else {}
    payload = candidates.get(str(label)) if isinstance(candidates, dict) else None
    return _short_evidence_lines(payload.get("evidence", []) if isinstance(payload, dict) else [])


def _decision_task_names(dataset):
    if "spotify" in str(dataset or "").lower():
        return "playlist continuation", "music playlist", "song"
    return "bundle construction", "fashion outfit", "fashion item"


def decision_prompt(decision_case, evidence):
    task_name, bundle_name, item_name = _decision_task_names(decision_case.get("dataset"))
    partial_blocks = []
    for index, item in enumerate(decision_case.get("partial_items", [])):
        block = [f"{index + 1}. {item.get('text', '')}"]
        lines = _partial_evidence(evidence, item.get("item_id"))
        if lines:
            block.append("Evidence: " + " | ".join(lines))
        partial_blocks.append("\n".join(block))

    option_blocks = []
    for candidate in decision_case.get("candidates", []):
        label = str(candidate.get("label", ""))
        block = [f"{label}. {candidate.get('text', '')}"]
        lines = _candidate_evidence(evidence, label)
        if lines:
            block.append("Evidence: " + " | ".join(lines))
        option_blocks.append("\n".join(block))

    return (
        f"You are a helpful and honest assistant. The following are multiple choice questions about {task_name}. "
        "You should directly answer the question by choosing the letter of the correct option. Only provide the letter "
        "of your answer, without any explanation or mentioning the option content.\n"
        f"Question: Given the partial {bundle_name} below, which candidate {item_name} should be included into this {bundle_name}?\n"
        f"Partial {bundle_name}:\n"
        f"{chr(10).join(partial_blocks)}\n"
        f"Options:\n{chr(10).join(option_blocks)}\n"
        'Your answer should indicate your choice with a single letter (e.g., "A," "B," "C," etc.).\n'
        "Choice:"
    )

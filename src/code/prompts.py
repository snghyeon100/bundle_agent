"""Prompt builders for the adaptive per-item evidence method."""

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
        unified["partial_items"].append(
            {"item_id": int(item_id), "text": source.get("text", "")}
        )
    for candidate in case_view.get("candidates", []):
        label = str(candidate.get("label", ""))
        item_id = int(candidate.get("item_id"))
        source = candidate_text.get(label, {})
        unified["candidates"].append(
            {"label": label, "item_id": item_id, "text": source.get("text", "")}
        )
    return unified


def _schema_text(case_view):
    partial_entries = []
    for item_id in case_view.get("partial_item_ids", []):
        item_id = int(item_id)
        partial_entries.append(
            f'    "partial_{item_id}": {{\n'
            f'      "item_id": {item_id},\n'
            '      "evidence": ["source: retrieval relation -> external records -> contextual pattern"]\n'
            "    }"
        )
    candidate_entries = []
    for candidate in case_view.get("candidates", []):
        label = str(candidate.get("label"))
        item_id = int(candidate.get("item_id"))
        candidate_entries.append(
            f'    "{label}": {{\n'
            f'      "item_id": {item_id},\n'
            '      "evidence": ["source: retrieval relation -> external records -> contextual pattern"]\n'
            "    }"
        )
    return (
        "{\n"
        '  "schema_version": "adaptive_item_evidence_v1",\n'
        '  "strategy": {\n'
        '    "name": "short instance-specific contextual-evidence strategy name",\n'
        '    "description": "what instance-specific distinction is targeted and how source context makes it interpretable"\n'
        "  },\n"
        '  "partial_evidence": {\n'
        + ",\n".join(partial_entries)
        + "\n  },\n"
        '  "candidate_evidence": {\n'
        + ",\n".join(candidate_entries)
        + "\n  }\n"
        "}"
    )


def code_generation_prompt(
    case_view,
    source_manifest,
    output_file,
    semantic_case=None,
):
    labels = ", ".join(candidate_labels(case_view))
    unified_case = _unified_case_context(case_view, semantic_case)

    return (
        "You are the Code Generation Agent for a bundle-completion evidence pipeline.\n\n"
        "Generate only complete executable Python code, with no markdown or explanation.\n\n"
        f"{task_semantics(case_view.get('dataset'))}\n\n"
        "Problem instance:\n"
        f"{pretty_json(unified_case)}\n\n"
        "Your goal is to generate code that extracts source-derived contextual observations useful "
        "for determining which candidate item most appropriately completes the partial bundle.\n\n"
        "Invent and implement exactly one instance-adaptive contextual-evidence strategy. Inspect the "
        "partial items, candidate items, and available sources together to determine what strategy best "
        "fits this instance.\n\n"
        "Examples of item-context strategies:\n"
        "- IB x BI: item -> bundles containing the item -> other items in those bundles.\n"
        "- IU x UI: item -> users interacting with the item -> other items interacted with by those users.\n"
        "- BI x IB: for a target item, begin with its containing bundles, follow bundle -> items -> other "
        "bundles sharing those items, then retrieve representative items from the related bundles.\n\n"
        "Before committing to a strategy, determine which feasible source relation is expected to provide "
        "the most informative and non-sparse context across the current partial and candidate items. Then "
        "implement exactly one strategy.\n\n"
        "These are examples only. Adapt one of them or invent exactly one strategy that is best suited "
        "to this instance; do not implement all three by default.\n\n"
        "Apply the strategy independently and consistently to every partial item and every candidate item. "
        "For each item, retrieve context from source records.\n\n"
        "Final evidence must report the concrete external records or contextual patterns retrieved for "
        "each item. When retrieved context records are items, report their available text or title rather "
        "than their item IDs. Item IDs may be used internally for source lookup but should not be used as "
        "the final contextual evidence.\n\n"
        "Do not directly compare candidates with partial items, construct a partial-item aggregate, rank "
        "candidates, or make the final prediction. The Prediction Agent will compare the item-level "
        "contextual evidence. Read at least one listed source at runtime.\n\n"
        "Available data sources under `data/`:\n"
        f"{pretty_json(source_manifest)}\n\n"
        "Read only the listed sources. Load .pt files on CPU. Canonicalize item-ID sets with "
        "sorted(set(...)). Write UTF-8 JSON to exactly:\n"
        f"{output_file}\n\n"
        f"Required candidate labels: {labels}\n\n"
        "The written JSON must match this schema; replace the descriptive placeholders:\n"
        f"{_schema_text(case_view)}"
    )


def _short_evidence_lines(values, max_items=5, max_chars=900):
    if not isinstance(values, list):
        values = []
    lines = []
    for value in values[:max_items]:
        text = " ".join(str(value or "").split())
        if text:
            lines.append(text[:max_chars])
    return lines


def _partial_item_evidence(evidence, item_id):
    partials = evidence.get("partial_evidence", {}) if isinstance(evidence, dict) else {}
    payload = partials.get(f"partial_{int(item_id)}") if isinstance(partials, dict) else None
    return _short_evidence_lines(payload.get("evidence", []) if isinstance(payload, dict) else [])


def _candidate_item_evidence(evidence, label):
    candidates = evidence.get("candidate_evidence", {}) if isinstance(evidence, dict) else {}
    payload = candidates.get(str(label)) if isinstance(candidates, dict) else None
    return _short_evidence_lines(payload.get("evidence", []) if isinstance(payload, dict) else [])


def _strategy_lines(evidence, max_chars=700):
    if not isinstance(evidence, dict):
        return []
    strategy = evidence.get("strategy", {})
    if not isinstance(strategy, dict):
        return []
    name = " ".join(str(strategy.get("name", "")).split())
    description = " ".join(str(strategy.get("description", "")).split())
    if not name or not description:
        return []
    return [f"Instance-adaptive strategy: {name}. {description}"[:max_chars]]


def _decision_task_names(dataset):
    if "spotify" in str(dataset or "").lower():
        return "playlist continuation", "music playlist", "song"
    return "bundle construction", "fashion outfit", "fashion item"


def decision_prompt(decision_case, evidence):
    task_name, bundle_name, item_name = _decision_task_names(decision_case.get("dataset"))
    partial_blocks = []
    for index, item in enumerate(decision_case.get("partial_items", [])):
        block = [f"{index + 1}. {item.get('text', '')}"]
        lines = _partial_item_evidence(evidence, item.get("item_id"))
        if lines:
            block.append("Evidence: " + " | ".join(lines))
        partial_blocks.append("\n".join(block))

    option_blocks = []
    for candidate in decision_case.get("candidates", []):
        label = str(candidate.get("label", ""))
        block = [f"{label}. {candidate.get('text', '')}"]
        lines = _candidate_item_evidence(evidence, label)
        if lines:
            block.append("Evidence: " + " | ".join(lines))
        option_blocks.append("\n".join(block))

    strategy_lines = _strategy_lines(evidence)
    strategy_section = ""
    if strategy_lines:
        strategy_section = "\n".join(strategy_lines) + "\n"

    return (
        f"You are a helpful and honest assistant. The following is a multiple choice question about {task_name}. "
        "Choose the correct option using the item text and the source-grounded contextual evidence attached "
        "to the partial items and candidate items. Only provide one option letter, without explanation or "
        "option content.\n"
        f"Question: Given the partial {bundle_name} below, which candidate {item_name} should be included into this {bundle_name}?\n"
        f"Partial {bundle_name}:\n"
        f"{chr(10).join(partial_blocks)}\n"
        f"{strategy_section}"
        f"Options:\n{chr(10).join(option_blocks)}\n"
        'Your answer must be a single letter (for example, "A" or "B").\n'
        "Choice:"
    )

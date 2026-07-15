"""Prompt builders for adaptive bundle-level evidence retrieval."""

from .common import candidate_labels, pretty_json, task_semantics


MAX_EVIDENCE_ITEMS = 5


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


def _dataset_intent_strategy_example(dataset):
    name = str(dataset or "").lower()
    if "spotify" in name:
        return (
            "Minimal example of Partial Items -> Intent -> Strategy:\n\n"
            "Partial Items:\n"
            "Several upbeat synth-pop tracks with retro production and a night-driving mood.\n\n"
            "Intent:\n"
            "\"An energetic retro synth-pop playlist for nighttime driving.\"\n\n"
            "Strategy:\n"
            "Use the inferred era, genre, energy, and listening context to design a multi-step retrieval "
            "strategy. Retrieve source-grounded contexts that express those properties, identify recurring "
            "intent-relevant track patterns, and link every candidate to the same patterns while retaining "
            "the five most semantically relevant records."
        )
    return (
        "Minimal example of Partial Items -> Intent -> Strategy:\n\n"
        "Partial Items:\n"
        "A waterproof shell jacket, flexible utility pants, and trail shoes.\n\n"
        "Intent:\n"
        "\"A functional, weather-ready outdoor outfit designed for mobility.\"\n\n"
        "Strategy:\n"
        "Use the intent-specific properties of weather protection, mobility, and outdoor use to design a "
        "multi-step evidence-retrieval strategy. Retrieve source-grounded contexts that demonstrate those "
        "properties, identify the most relevant recurring patterns, and connect every candidate to the same "
        "intent-specific patterns. Retain at most five records by semantic relevance, not by item ID or "
        "retrieval order.\n\n"
        "This example demonstrates only the transformation from partial items to intent to strategy. For "
        "the current bundle, infer new characteristics and design a new strategy rather than reusing the "
        "outdoor attributes or steps above."
    )


def _schema_text(case_view):
    candidate_entries = []
    for candidate in case_view.get("candidates", []):
        label = str(candidate.get("label"))
        item_id = int(candidate.get("item_id"))
        candidate_entries.append(
            f'    "{label}": {{\n'
            f'      "item_id": {item_id},\n'
            '      "evidence": ["source: concrete record -> candidate-to-bundle contextual observation"]\n'
            "    }"
        )
    return (
        "{\n"
        '  "schema_version": "adaptive_bundle_evidence_v2",\n'
        '  "intent": "one concise sentence describing the inferred partial-bundle intent",\n'
        '  "strategy": {\n'
        '    "name": "short instance-specific retrieval strategy name",\n'
        '    "description": "the intent-specific evidence-retrieval strategy implemented by the code"\n'
        "  },\n"
        '  "partial_bundle_evidence": {\n'
        '    "evidence": ["source: concrete records -> shared partial-bundle contextual pattern"]\n'
        "  },\n"
        '  "candidate_evidence": {\n'
        + ",\n".join(candidate_entries)
        + "\n  }\n"
        "}"
    )


def _code_skeleton_text():
    return (
        "INTENT = <one concise intent inferred only from PARTIAL_ITEMS>\n"
        "STRATEGY = {\"name\": <name>, \"description\": <description>}\n\n"
        "def load_sources():\n"
        "    <load the listed sources needed by the selected strategy>\n\n"
        "def retrieve_partial_bundle_context(partial_items, sources):\n"
        "    <retrieve source-grounded records for the partial bundle and return an internal dictionary "
        "containing: (1) evidence, with at most five evidence strings, and (2) completion_pattern, a reusable "
        "strategy-specific representation derived from those records>\n\n"
        "def retrieve_candidate_evidence(candidate, partial_items, partial_bundle_context, sources):\n"
        "    <actually use partial_bundle_context to retrieve concrete item or bundle records that both connect "
        "this candidate to the actual partial items, directly or through a source-grounded multi-step relation, "
        "and demonstrate the derived completion_pattern; filter by semantic relevance and return at most five "
        "evidence strings>\n\n"
        "def main():\n"
        "    sources = load_sources()\n"
        "    partial_bundle_context = retrieve_partial_bundle_context(PARTIAL_ITEMS, sources)\n"
        "    candidate_evidence = {}\n"
        "    for candidate in CANDIDATES:\n"
        "        candidate_evidence[candidate[\"label\"]] = {\n"
        "            \"item_id\": candidate[\"item_id\"],\n"
        "            \"evidence\": retrieve_candidate_evidence(\n"
        "                candidate, PARTIAL_ITEMS, partial_bundle_context, sources\n"
        "            ),\n"
        "        }\n"
        "    <write INTENT, STRATEGY, partial_bundle_context[\"evidence\"], and candidate_evidence using the "
        "required JSON schema; completion_pattern is internal and must not be added to the JSON>\n"
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
        "Before writing code, inspect only the partial items and infer one concise intent describing the "
        "bundle they are trying to form. Do not use the candidates to infer or revise the intent.\n\n"
        "Based on that inferred intent, design exactly one evidence-retrieval code strategy for this specific "
        "bundle-completion task. The strategy must be capable of retrieving source-grounded evidence that "
        "reflects the distinctive characteristics expressed by the intent. Choose the sources, relations, "
        "and retrieval operations yourself from the listed data sources. It must be an instance-specific "
        "multi-step strategy in which the steps work together to capture the characteristics of this bundle, "
        "rather than a single generic lookup that would be unchanged for another bundle. Do not implement "
        "a fallback strategy.\n\n"
        f"{_dataset_intent_strategy_example(case_view.get('dataset'))}\n\n"
        "Implement the strategy to retrieve evidence for the partial bundle as a whole and derive a reusable "
        "completion pattern from those retrieved records, then apply the same fixed strategy to every candidate. "
        "For each candidate, retrieve evidence that connects it to both the actual partial items and the derived "
        "completion pattern. The connection to the partial items may be direct or may use a source-grounded "
        "multi-step relation. The candidate function must consume the actual partial-bundle context; merely "
        "accepting the argument while independently recreating search criteria is insufficient. Do not score, "
        "rank, select, or predict candidates.\n\n"
        "Evidence is not a description or score of the target item itself. It must be a concrete "
        "source-grounded item or bundle record that explains how the target item relates to an "
        "intent-specific completion pattern established from the current partial bundle.\n\n"
        "Final evidence must report concrete external records or contextual patterns retrieved from the "
        "listed sources. When records are items, report their available text or title rather than only item "
        "IDs. Item IDs may be used internally for lookup. If the selected strategy retrieves no evidence "
        "for a bundle or candidate, emit an empty evidence list instead of inventing evidence or writing a "
        "synthetic placeholder.\n\n"
        f"Every evidence array must contain between zero and {MAX_EVIDENCE_ITEMS} strings. The generated "
        "program may retrieve a broader context internally, but before writing JSON it must filter, "
        "deduplicate, and retain only the records most semantically relevant to the inferred intent and "
        "partial-bundle context. Do not select evidence merely by item ID, bundle ID, source order, or the "
        "first records encountered. Candidate evidence must additionally provide a positive connection from "
        "that candidate to the partial context through the selected relation chain. Selecting evidence "
        "records is not candidate scoring: do not rank or select candidates.\n\n"
        "Use the following high-level program skeleton. Replace every angle-bracket placeholder and "
        "generate working implementations. Do not leave placeholders, ellipses, pass statements, TODOs, "
        "pseudocode, or undefined helper functions. You may add strategy-specific helpers, but do not add "
        "the internal completion pattern to the written JSON and do not add a second strategy.\n\n"
        f"{_code_skeleton_text()}\n"
        "Available data sources under `data/`:\n"
        f"{pretty_json(source_manifest)}\n\n"
        "Read only the listed sources and read at least one listed source at runtime. Load .pt files on CPU. "
        "Canonicalize item-ID sets with sorted(set(...)). Write UTF-8 JSON to exactly:\n"
        f"{output_file}\n\n"
        f"Required candidate labels: {labels}\n\n"
        "The written JSON must match this schema exactly; replace the descriptive placeholders. Evidence "
        "arrays may be empty only when the selected retrieval produced no supporting source records:\n"
        f"{_schema_text(case_view)}"
    )


def _short_evidence_lines(values, max_items=MAX_EVIDENCE_ITEMS, max_chars=900):
    if not isinstance(values, list):
        values = []
    lines = []
    for value in values[:max_items]:
        text = " ".join(str(value or "").split())
        if text:
            lines.append(text[:max_chars])
    return lines


def _bundle_evidence(evidence):
    payload = evidence.get("partial_bundle_evidence", {}) if isinstance(evidence, dict) else {}
    return _short_evidence_lines(payload.get("evidence", []) if isinstance(payload, dict) else [])


def _candidate_item_evidence(evidence, label):
    candidates = evidence.get("candidate_evidence", {}) if isinstance(evidence, dict) else {}
    payload = candidates.get(str(label)) if isinstance(candidates, dict) else None
    return _short_evidence_lines(payload.get("evidence", []) if isinstance(payload, dict) else [])


def _intent_and_strategy_lines(evidence, max_chars=700):
    if not isinstance(evidence, dict):
        return []
    lines = []
    intent = " ".join(str(evidence.get("intent", "")).split())
    if intent:
        lines.append(f"Inferred bundle intent: {intent}"[:max_chars])
    strategy = evidence.get("strategy", {})
    if isinstance(strategy, dict):
        name = " ".join(str(strategy.get("name", "")).split())
        description = " ".join(str(strategy.get("description", "")).split())
        if name and description:
            lines.append(f"Retrieval strategy: {name}. {description}"[:max_chars])
    return lines


def _decision_task_names(dataset):
    if "spotify" in str(dataset or "").lower():
        return "playlist continuation", "music playlist", "song"
    return "bundle construction", "fashion outfit", "fashion item"


def decision_prompt(decision_case, evidence):
    task_name, bundle_name, item_name = _decision_task_names(decision_case.get("dataset"))
    partial_blocks = [
        f"{index + 1}. {item.get('text', '')}"
        for index, item in enumerate(decision_case.get("partial_items", []))
    ]

    option_blocks = []
    for candidate in decision_case.get("candidates", []):
        label = str(candidate.get("label", ""))
        block = [f"{label}. {candidate.get('text', '')}"]
        lines = _candidate_item_evidence(evidence, label)
        if lines:
            block.append("Evidence:\n" + "\n".join(f"- {line}" for line in lines))
        option_blocks.append("\n".join(block))

    context_lines = _intent_and_strategy_lines(evidence)
    bundle_lines = _bundle_evidence(evidence)
    if bundle_lines:
        context_lines.append(
            "Partial-bundle evidence:\n" + "\n".join(f"- {line}" for line in bundle_lines)
        )
    context_section = "\n\n".join(context_lines)
    if context_section:
        context_section += "\n\n"

    return (
        f"You are a helpful and honest assistant. The following is a multiple choice question about {task_name}. "
        "Choose the correct option using the item text, inferred bundle intent, and source-grounded evidence. "
        "Only provide one option letter, without explanation or option content.\n"
        f"Question: Given the partial {bundle_name} below, which candidate {item_name} should be included into this {bundle_name}?\n"
        f"Partial {bundle_name}:\n"
        f"{chr(10).join(partial_blocks)}\n\n"
        f"{context_section}"
        f"Options:\n\n{(chr(10) * 2).join(option_blocks)}\n\n"
        'Your answer must be a single letter (for example, "A" or "B").\n'
        "Choice:"
    )

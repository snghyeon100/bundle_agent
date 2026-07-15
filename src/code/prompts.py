"""Prompt builders for adaptive bundle-level evidence retrieval."""

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


def _dataset_concept_guidance(dataset):
    name = str(dataset or "").lower()
    if "spotify" in name:
        return (
            "Spotify-specific conceptual guidance (not a template or prescribed strategy):\n"
            "- A playlist intent may be organized by artist, genre, era, mood, theme, listening "
            "context, or a combination inferred from the partial tracks.\n"
            "- Potential source relations include track -> playlists -> co-occurring tracks, track "
            "-> users -> other tracks, and other feasible multi-hop relations exposed by the listed "
            "sources.\n"
            "Decide for the current partial playlist itself. Do not assume that any listed intent "
            "dimension or relation is appropriate."
        )
    return (
        "Illustrative POG example of an intent-specific multi-step strategy:\n\n"
        "Suppose the partial items describe a lightweight shirt, relaxed shorts, and breathable casual "
        "shoes. A plausible intent is \"A lightweight summer casual outfit.\"\n\n"
        "Translate that intent into retrieval targets such as lightweight materials, short silhouettes, "
        "summer use, casual style, and complementary category roles. Then use one coherent chain:\n\n"
        "1. Retrieve direct containing outfits for each partial item.\n"
        "2. Independently retrieve description- and image-nearest items, retain category-consistent "
        "neighbors with actual train-bundle support, and use them as semantic anchors for sparse partial "
        "items.\n"
        "3. Traverse both direct items and semantic anchors to their containing outfits, then retrieve "
        "co-occurring items, titles, and categories.\n"
        "4. Identify intent-specific patterns recurring across the combined outfit neighborhoods.\n"
        "5. Apply the same candidate -> semantic anchors -> containing outfits -> contextual metadata "
        "chain to every candidate and report concrete records, without scoring or selecting candidates.\n\n"
        "Direct and embedding-bridged retrieval are internal steps of one strategy, not separate fallback "
        "strategies. Neighbor IDs or similarity values alone are not final evidence; report the retrieved "
        "item metadata and outfit records.\n\n"
        "This example illustrates how to operationalize an inferred intent. Do not default to a "
        "seasonal intent unless the current partial items support it. Infer the appropriate intent "
        "and relation chain for the current instance."
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
        '    "description": "the source relation, retrieval operation, and intent-specific distinction targeted"\n'
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
        "def retrieve_partial_bundle_evidence(partial_items, sources):\n"
        "    <return a list of source-grounded evidence strings for the partial bundle as a whole>\n\n"
        "def retrieve_candidate_evidence(candidate, partial_items, sources):\n"
        "    <apply the same fixed strategy and return this candidate's evidence strings>\n\n"
        "def main():\n"
        "    sources = load_sources()\n"
        "    partial_bundle_evidence = retrieve_partial_bundle_evidence(PARTIAL_ITEMS, sources)\n"
        "    candidate_evidence = {}\n"
        "    for candidate in CANDIDATES:\n"
        "        candidate_evidence[candidate[\"label\"]] = {\n"
        "            \"item_id\": candidate[\"item_id\"],\n"
        "            \"evidence\": retrieve_candidate_evidence(candidate, PARTIAL_ITEMS, sources),\n"
        "        }\n"
        "    <write INTENT, STRATEGY, partial_bundle_evidence, and candidate_evidence as required JSON>\n"
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
        "The central task is to identify the distinctive characteristics of the current partial bundle "
        "and design an evidence-retrieval strategy specifically tailored to those characteristics, so "
        "that a separate Prediction Agent can identify which candidate best completes the bundle.\n\n"
        "Do not use a generic retrieval strategy that would remain unchanged for a partial bundle with "
        "a different intent.\n\n"
        "Infer intent exclusively from the partial items. Do not use candidate items to choose, revise, "
        "or broaden the intent. Keep the intent to one concise sentence.\n\n"
        "Choose exactly one strategy after considering the inferred intent and the feasible relations in "
        "the available sources. Fix that strategy before evaluating candidates. The strategy description "
        "must state the source relation, retrieval operation, and intent-specific distinction it targets. "
        "Do not produce or implement a fallback strategy.\n\n"
        f"{_dataset_concept_guidance(case_view.get('dataset'))}\n\n"
        "Retrieve source-grounded evidence for the partial bundle as a whole. This evidence should expose "
        "shared external context across the partial items rather than repeat their input text or create a "
        "separate evidence block for every partial item.\n\n"
        "Then apply the one fixed strategy consistently to every candidate. Candidate evidence may describe "
        "a concrete source-grounded relationship between that candidate and the complete partial bundle "
        "context. Do not assign compatibility scores, rank candidates, or make the final prediction.\n\n"
        "Final evidence must report concrete external records or contextual patterns retrieved from the "
        "listed sources. When records are items, report their available text or title rather than only item "
        "IDs. Item IDs may be used internally for lookup. If the selected strategy retrieves no evidence "
        "for a bundle or candidate, emit an empty evidence list instead of inventing evidence or writing a "
        "synthetic placeholder.\n\n"
        "Use the following high-level program skeleton. Replace every angle-bracket placeholder and "
        "generate working implementations. Do not leave placeholders, ellipses, pass statements, TODOs, "
        "pseudocode, or undefined helper functions. You may add strategy-specific helpers, but do not add "
        "an extra intermediate output or a second strategy.\n\n"
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


def _short_evidence_lines(values, max_items=5, max_chars=900):
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
            block.append("Evidence: " + " | ".join(lines))
        option_blocks.append("\n".join(block))

    context_lines = _intent_and_strategy_lines(evidence)
    bundle_lines = _bundle_evidence(evidence)
    if bundle_lines:
        context_lines.append("Partial-bundle evidence: " + " | ".join(bundle_lines))
    context_section = "\n".join(context_lines)
    if context_section:
        context_section += "\n"

    return (
        f"You are a helpful and honest assistant. The following is a multiple choice question about {task_name}. "
        "Choose the correct option using the item text, inferred bundle intent, and source-grounded evidence. "
        "Only provide one option letter, without explanation or option content.\n"
        f"Question: Given the partial {bundle_name} below, which candidate {item_name} should be included into this {bundle_name}?\n"
        f"Partial {bundle_name}:\n"
        f"{chr(10).join(partial_blocks)}\n"
        f"{context_section}"
        f"Options:\n{chr(10).join(option_blocks)}\n"
        'Your answer must be a single letter (for example, "A" or "B").\n'
        "Choice:"
    )

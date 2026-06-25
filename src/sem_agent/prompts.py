"""Prompt builders for the sem_agent pipeline.

Stage 1 — Item Evidence Expansion:
    Code generator retrieves item-level supporting evidence for every partial
    item and candidate item. The goal is to find other items that help explain
    each target item through bundle, user, metadata, or feature-neighbor paths.

Stage 2 — Bundle Context & Candidate Fit:
    Code generator uses Stage 1 supporting items as anchors to build a
    bundle-level context, then explains how each candidate's expanded evidence
    fits or conflicts with that context.

Both stages output ONLY string narratives in the `value` field; numeric values
are forbidden so the decision model cannot fall back to number comparison.
"""

import json

from .common import candidate_labels, task_semantics
from .affordance_graph import render_affordance_relation_map


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _dump(value):
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)



def _shared_rules(output_file, labels, max_evidence_chars):
    return (
        f"Write UTF-8 JSON to exactly: {output_file}\n"
        f"Required candidate labels for candidate-scoped signals: {labels}\n"
        f"Keep the serialized JSON below approximately {int(max_evidence_chars)} characters.\n\n"
        "Output schema (include `observation` only for partial_bundle signals, "
        "and `candidate_observations` only for candidate signals):\n"
        "{\n"
        '  "signals": [\n'
        "    {\n"
        '      "signal_name": "stable_snake_case_identifier",\n'
        '      "signal_scope": "partial_bundle | candidate",\n'
        '      "description": "what semantic goal was investigated and what was found",\n'
        '      "sources": ["exact source filename from manifest"],\n'
        '      "relation_path": ["typed hop 1", "typed hop 2"],\n'
        '      "observation": {\n'
        '        "value": "descriptive string narrative about the partial bundle",\n'
        '        "evidence": ["short factual sentence 1", "short factual sentence 2"]\n'
        "      },\n"
        '      "candidate_observations": {\n'
        '        "A": {\n'
        '          "value": "descriptive string narrative about candidate A",\n'
        '          "evidence": ["short factual sentence 1", "short factual sentence 2"]\n'
        "        }\n"
        "      }\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        "CRITICAL value rules:\n"
        "- `signal_scope` MUST be either `partial_bundle` or `candidate`.\n"
        "- For `signal_scope: partial_bundle`, write one shared `observation` object and "
        "do NOT repeat it under candidate labels.\n"
        "- For `signal_scope: candidate`, write `candidate_observations` with every "
        "required candidate label and do NOT write a shared `observation`.\n"
        "- Every `value` MUST be a non-empty descriptive STRING narrative.\n"
        "- Do NOT put a bare number (e.g. 0.71) as the sole content of `value`.\n"
        "- Numbers MAY appear inside a narrative string for context "
        '  (e.g. "co-appears in 5 bundles, mostly with bags and dresses").\n'
        "- `evidence` must be a list of ≤5 short factual strings (item titles, "
        "  category names, bundle IDs with context — not raw vector values).\n"
        "- Every candidate-scoped signal must compute the same logic for EVERY candidate label.\n"
        "- Source names must exactly match names in the manifest.\n"
        "- Do NOT add prediction, winner, ranking, recommendation, or final-score fields.\n"
        "- CPU-only environment: torch.load(..., map_location=\"cpu\").\n"
        "- Skip unavailable sources gracefully without crashing.\n"
    )


# ---------------------------------------------------------------------------
# Stage 1 prompt
# ---------------------------------------------------------------------------

def stage1_ecosystem_prompt(
    case_view,
    source_manifest,
    affordance_graph,
    output_file,
    max_evidence_chars,
    semantic_case=None,
):
    """Stage 1: Item Evidence Expansion Builder.

    The code generator retrieves item-level supporting evidence and encodes it as
    pure-text narratives — no bare numeric values allowed.
    """
    labels = ", ".join(candidate_labels(case_view))

    semantic_goals = (
        "STAGE 1 TASK — Unified Item Evidence Expansion\n"
        "For every target item in the case, including partial item(s) and candidate "
        "items, retrieve supporting or neighboring items from the available source "
        "relation graph using the same evidence-expansion logic.\n\n"

        "Question: Which other items from the available sources help explain each "
        "target item?\n"
        "Investigate: For each target item, search for supporting items through any "
        "available source-composition path. Supporting items may come from co-bundles, "
        "shared users, category anchors, or visual/textual/UI/BI LightGCN embedding neighbors.\n"
        "Output intent: For each target item, collect a list of supporting item titles "
        "and the paths that found them into the `evidence` array. Do NOT write long "
        "narratives or analysis in the `value` string; simply output a static string like "
        "'Extracted N supporting items'. This stage is strictly for data retrieval.\n\n"

        "STAGE 1 OUTPUT STRUCTURE\n"
        "Produce JSON with at least the following two signal shapes:\n"
        "{\n"
        '  "signals": [\n'
        "    {\n"
        '      "signal_name": "partial_item_evidence_expansion",\n'
        '      "signal_scope": "partial_bundle",\n'
        '      "description": "supporting items retrieved for the partial item(s)",\n'
        '      "sources": ["source filename(s) used"],\n'
        '      "relation_path": ["typed hop actually used"],\n'
        '      "observation": {\n'
        '        "value": "Extracted N supporting items.",\n'
        '        "evidence": [\n'
        '          "Path name: item title (brief rationale)"\n'
        "        ]\n"
        "      }\n"
        "    },\n"
        "    {\n"
        '      "signal_name": "candidate_item_evidence_expansion",\n'
        '      "signal_scope": "candidate",\n'
        '      "description": "supporting items retrieved for each candidate item",\n'
        '      "sources": ["source filename(s) used"],\n'
        '      "relation_path": ["typed hop actually used"],\n'
        '      "candidate_observations": {\n'
        '        "A": {\n'
        '          "value": "Extracted N supporting items.",\n'
        '          "evidence": [\n'
        '            "Path name: item title (brief rationale)"\n'
        "          ]\n"
        "        }\n"
        "      }\n"
        "    }\n"
        "  ]\n"
        "}\n"
        "For every evidence string, the prefix before ':' must describe the actual "
        "retrieval path or rationale used by the generated code.\n"
        "Do not write detailed narratives, fit judgments, rankings, "
        "predictions, or final recommendations in Stage 1.\n\n"

        "IMPLEMENTATION NOTES\n"
        "- You are FREE to choose any traversal paths through the available sources "
        "to implement this task. The task is a semantic target, not an algorithmic "
        "prescription.\n"
        "- Treat the listed sources as a relation graph, not as isolated files. You can compose relations to retrieve supporting items through multi-hop paths.\n"
        "- Include `relation_path` listing the typed hops your code actually executes.\n"
    )

    return (
        "You are the Stage 1 Item Evidence Expansion Code Generator in a bundle-completion system.\n"
        "Generate ONLY complete executable Python code — no markdown fences, no explanation.\n"
        "The script runs with the allowed workspace as its current directory.\n\n"
        f"{task_semantics(case_view['dataset'])}\n\n"
        f"{semantic_goals}\n"
        f"{_shared_rules(output_file, labels, max_evidence_chars)}\n"
        f"ID-only case:\n{_dump(case_view)}\n\n"
        f"Text-enriched case context:\n{_dump(semantic_case or {})}\n\n"
        "Use the text-enriched context as semantic context for interpreting items, "
        "but ground every signal in the listed workspace sources. Do not rely on "
        "item text alone for final evidence.\n\n"
        f"Source Capability Manifest:\n{_dump(source_manifest)}"
    )


# ---------------------------------------------------------------------------
# Stage 2 prompt
# ---------------------------------------------------------------------------

def _stage2_rules(labels):
    return (
        f"Required candidate labels for candidate-scoped signals: {labels}\n\n"
        "CRITICAL value rules:\n"
        "- `signal_scope` MUST be either `partial_bundle` or `candidate`.\n"
        "- For `signal_scope: partial_bundle`, write one shared `observation` object and "
        "do NOT repeat it under candidate labels.\n"
        "- For `signal_scope: candidate`, write `candidate_observations` with every "
        "required candidate label and do NOT write a shared `observation`.\n"
        "- Every `value` MUST be a non-empty descriptive STRING narrative.\n"
        "- Do NOT put a bare number (e.g. 0.71) as the sole content of `value`.\n"
        "- Do NOT add prediction, winner, ranking, recommendation, or final-score fields.\n"
        "- Output valid JSON ONLY. Do not write markdown text outside the JSON block.\n"
    )

def stage2_gap_prompt(
    case_view,
    source_manifest,
    affordance_graph,
    output_file,
    max_evidence_chars,
    stage1_evidence,
    semantic_case=None,
):
    """Stage 2: Bundle Context & Candidate Fit Narrator.

    Uses Stage 1 item evidence expansions as anchors to infer the bundle context
    and explain how each candidate fits that context.
    """
    labels = ", ".join(candidate_labels(case_view))
    relation_map = render_affordance_relation_map(affordance_graph)

    # Combine Stage 1 evidence into unified case
    partial_evidence = []
    for sig in stage1_evidence.get("signals", []) if isinstance(stage1_evidence, dict) else []:
        if str(sig.get("signal_scope", "")).strip() == "partial_bundle":
            obs = sig.get("observation")
            if isinstance(obs, dict):
                partial_evidence.append({
                    "signal_name": sig.get("signal_name"),
                    "value": obs.get("value"),
                    "evidence": obs.get("evidence")
                })

    cand_evidence_by_label = {}
    for sig in stage1_evidence.get("signals", []) if isinstance(stage1_evidence, dict) else []:
        if str(sig.get("signal_scope", "")).strip() == "candidate":
            c_obs = sig.get("candidate_observations")
            if isinstance(c_obs, dict):
                for lbl, obs in c_obs.items():
                    if not isinstance(obs, dict):
                        continue
                    if lbl not in cand_evidence_by_label:
                        cand_evidence_by_label[lbl] = []
                    cand_evidence_by_label[lbl].append({
                        "signal_name": sig.get("signal_name"),
                        "value": obs.get("value"),
                        "evidence": obs.get("evidence")
                    })

    unified_case = {
        "case_id": case_view.get("case_id"),
        "dataset": case_view.get("dataset"),
        "bundle_id": case_view.get("bundle_id"),
        "partial_items": [],
        "partial_bundle_evidence": partial_evidence,
        "candidates": []
    }

    if semantic_case:
        for item in semantic_case.get("partial_items", []):
            unified_case["partial_items"].append({
                "item_id": item.get("item_id"),
                "text": item.get("text", "")
            })
        for cand in semantic_case.get("candidates", []):
            lbl = str(cand.get("label", ""))
            unified_case["candidates"].append({
                "label": lbl,
                "item_id": cand.get("item_id"),
                "text": cand.get("text", ""),
                "stage1_evidence": cand_evidence_by_label.get(lbl, [])
            })
    else:
        unified_case["partial_items"] = case_view.get("partial_item_ids", [])
        for cand in case_view.get("candidates", []):
            lbl = str(cand.get("label", ""))
            unified_case["candidates"].append({
                "label": lbl,
                "item_id": cand.get("item_id"),
                "stage1_evidence": cand_evidence_by_label.get(lbl, [])
            })

    semantic_goals = (
        "Stage 1 has already expanded every partial item and candidate item into "
        "supporting items retrieved from the source relation graph. This evidence "
        "is attached directly to the partial bundle and each candidate in the Unified Case below. "
        "Your task now is bundle-level interpretation.\n\n"

        "GOAL 1 — Bundle Context Construction (required)\n"
        "  Question: What bundle-level context is implied by the partial item(s) and "
        "their Stage 1 supporting items?\n"
        "  Investigate: Use the Stage 1 partial-item supporting evidence as anchors. "
        "Infer the partial bundle's likely context, role structure, or missing direction.\n"
        "  Output: A narrative describing the constructed bundle context, the "
        "supporting item evidence behind it, and any ambiguity or competing signals.\n"
        "  Scope: Output this as `signal_scope: partial_bundle` with one shared "
        "`observation`; do not repeat the bundle-context narrative under candidate labels.\n\n"

        "GOAL 2 — Candidate Fit To Bundle Context (required)\n"
        "  Question: How does each candidate's Stage 1 item evidence relate to the "
        "bundle context constructed above?\n"
        "  Investigate: Cross-reference each candidate's supporting items against the bundle context. "
        "Look for overlap, complementarity, conflict, or sparse evidence.\n"
        "  Output: For each candidate, a narrative explaining how its expanded item "
        "evidence fits, partially fits, conflicts with, or remains ambiguous relative "
        "to the bundle context.\n\n"

        "STAGE 2 OUTPUT STRUCTURE\n"
        "Produce JSON with exactly the following two signal shapes:\n"
        "```json\n"
        "{\n"
        '  "signals": [\n'
        "    {\n"
        '      "signal_name": "bundle_context_construction",\n'
        '      "signal_scope": "partial_bundle",\n'
        '      "description": "bundle-level context inferred from Stage 1 supporting items",\n'
        '      "observation": {\n'
        '        "value": "bundle-context narrative using Stage 1 partial-item supporting evidence as anchors; mention context hypothesis, supporting patterns, competing signals, and uncertainty",\n'
        '        "evidence": [\n'
        '          "Stage 1 anchor used: short factual support"\n'
        "        ]\n"
        "      }\n"
        "    },\n"
        "    {\n"
        '      "signal_name": "candidate_fit_to_bundle_context",\n'
        '      "signal_scope": "candidate",\n'
        '      "description": "candidate evidence interpreted against the constructed bundle context",\n'
        '      "candidate_observations": {\n'
        '        "A": {\n'
        '          "value": "candidate-specific fit narrative: explain how candidate A\'s Stage 1 supporting evidence overlaps with, complements, conflicts with, or remains ambiguous relative to the bundle context; do not rank or recommend",\n'
        '          "evidence": [\n'
        '            "Stage 1 candidate anchor used: short factual support"\n'
        "          ]\n"
        "        }\n"
        "      }\n"
        "    }\n"
        "  ]\n"
        "}\n"
        "```\n"
        "Use Stage 1 partial-item supporting evidence as anchors for bundle-context construction. "
        "Use Stage 1 candidate supporting evidence as anchors for candidate-fit analysis.\n"
    )

    return (
        "You are the Stage 2 Bundle Context & Candidate Fit Reasoner in a bundle-completion system.\n"
        "Your task is to analyze the provided Unified Case and output a JSON response containing your analysis.\n\n"
        f"{task_semantics(case_view['dataset'])}\n\n"
        f"{semantic_goals}\n"
        f"{_stage2_rules(labels)}\n"
        f"Unified Case (ID, text, and Stage 1 evidence):\n{_dump(unified_case)}\n"
    )


# ---------------------------------------------------------------------------
# Code repair prompt (shared across stages)
# ---------------------------------------------------------------------------

def repair_prompt(
    case_view,
    source_manifest,
    previous_code,
    execution_context,
    output_file,
    affordance_graph=None,
    require_relation_path=False,
):
    labels = ", ".join(candidate_labels(case_view))
    relation_path_rule = ""
    graph_block = ""
    if require_relation_path:
        relation_path_rule = (
            " Every signal must retain `relation_path` with at least two non-empty "
            "typed transitions the repaired code actually executes."
        )
        if affordance_graph:
            graph_block = (
                f"\n\nCompact Evidence Relation Map:\n"
                f"{render_affordance_relation_map(affordance_graph)}"
            )

    return (
        "You are repairing Python signal-extraction code. Return ONLY complete executable "
        "Python code — no markdown fences, no explanation.\n"
        "Fix execution errors, JSON schema defects, or value-format violations. "
        "Preserve the original semantic investigation intent.\n"
        "CPU-only: torch.load(..., map_location=\"cpu\").\n\n"
        f"Script must write UTF-8 JSON to exactly: {output_file}\n"
        f"Required candidate labels for every candidate-scoped signal: {labels}\n\n"
        "CRITICAL: `signal_scope` must be either `partial_bundle` or `candidate`. "
        "For `partial_bundle`, use one shared `observation`. For `candidate`, use "
        "`candidate_observations` with every required candidate label. Every `value` "
        f"must be a non-empty STRING narrative. A bare number is not acceptable.{relation_path_rule}\n\n"
        "Preserve this exact schema. Include `observation` only for partial_bundle "
        "signals, and include `candidate_observations` only for candidate signals:\n"
        "{\n"
        '  "signals": [\n'
        "    {\n"
        '      "signal_name": "...",\n'
        '      "signal_scope": "partial_bundle | candidate",\n'
        '      "description": "...",\n'
        '      "sources": ["..."],\n'
        '      "relation_path": ["hop1", "hop2"],\n'
        '      "observation": {"value": "shared partial-bundle narrative", "evidence": ["..."]},\n'
        '      "candidate_observations": {\n'
        '        "A": {"value": "narrative string", "evidence": ["..."]}\n'
        "      }\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        f"ID-only case:\n{_dump(case_view)}\n\n"
        f"Source Capability Manifest:\n{_dump(source_manifest)}\n\n"
        f"Execution context and defects:\n{_dump(execution_context)}"
        f"{graph_block}\n\n"
        f"Previous code:\n{previous_code}"
    )


# ---------------------------------------------------------------------------
# Decision prompt
# ---------------------------------------------------------------------------

def _decision_task_names(dataset):
    if "spotify" in str(dataset or "").lower():
        return "playlist continuation", "music playlist", "song"
    return "bundle construction", "fashion outfit", "fashion item"


def _compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def _signal_line(signal, observation, indent="   - "):
    signal_name = str(signal.get("signal_name", "unnamed_signal"))
    sources = signal.get("sources", [])
    source_text = ", ".join(str(s) for s in sources) if isinstance(sources, list) else str(sources)
    relation_path = signal.get("relation_path", [])
    path_text = (
        " -> ".join(str(t) for t in relation_path)
        if isinstance(relation_path, list) and relation_path
        else ""
    )
    value_text = _compact(observation.get("value"))
    facts = observation.get("evidence", [])
    fact_text = _compact(facts) if isinstance(facts, list) and facts else "[]"
    path_suffix = f"; path={path_text}" if path_text else ""
    return f"{indent}{signal_name} [sources: {source_text}]: value={value_text}; evidence={fact_text}{path_suffix}"


def _partial_evidence_lines(evidence):
    lines = []
    signals = evidence.get("signals", []) if isinstance(evidence, dict) else []
    for signal in signals:
        if not isinstance(signal, dict):
            continue
        if str(signal.get("signal_scope", "")).strip() != "partial_bundle":
            continue
        observation = signal.get("observation")
        if isinstance(observation, dict):
            lines.append(_signal_line(signal, observation, indent=" - "))
    return lines


def _candidate_evidence_lines(evidence, label):
    lines = []
    signals = evidence.get("signals", []) if isinstance(evidence, dict) else []
    for signal in signals:
        if not isinstance(signal, dict):
            continue
        scope = str(signal.get("signal_scope", "candidate")).strip()
        if scope == "partial_bundle":
            continue
        observations = signal.get("candidate_observations", {})
        observation = observations.get(label) if isinstance(observations, dict) else None
        if not isinstance(observation, dict):
            continue
        lines.append(_signal_line(signal, observation))
    return lines


def decision_prompt(decision_case, evidence):
    task_name, bundle_name, item_name = _decision_task_names(decision_case.get("dataset"))
    input_str = "; ".join(
        f"{index + 1}. {item.get('text', '')}"
        for index, item in enumerate(decision_case.get("partial_items", []))
    )

    option_blocks = []
    partial_lines = _partial_evidence_lines(evidence)
    for candidate in decision_case.get("candidates", []):
        label = str(candidate.get("label", ""))
        block = [f"{label}. {candidate.get('text', '')}"]
        evidence_lines = _candidate_evidence_lines(evidence, label)
        if evidence_lines:
            block.append("   Evidence:")
            block.extend(evidence_lines)
        option_blocks.append("\n".join(block))
    target_str = "\n".join(option_blocks)
    partial_context = ""
    if partial_lines:
        partial_context = "Shared partial-bundle evidence:\n" + "\n".join(partial_lines) + "\n"

    dataset_name = str(decision_case.get("dataset", "")).lower()
    pog_guidance = ""
    if dataset_name in ["pog", "pog_dense"]:
        pog_guidance = (
            "Note: For fashion outfits, similar items are rarely put together. "
            "Therefore, you must prioritize compatibility and complementarity over item resemblance.\n"
        )

    return (
        f"You are a helpful and honest assistant. The following are multiple choice questions about {task_name}. "
        "You should directly answer the question by choosing the letter of the correct option. Only provide the letter "
        "of your answer, without any explanation or mentioning the option content.\n"
        #f"{pog_guidance}"
        f"Question: Given the partial {bundle_name}: {input_str}, which candidate {item_name} should be included into this "
        f"{bundle_name}?\n"
        f"{partial_context}"
        f"Options:\n{target_str}\n"
        'Your answer should indicate your choice with a single letter (e.g., "A," "B," "C," etc.).\n'
        "Choice:"
    )

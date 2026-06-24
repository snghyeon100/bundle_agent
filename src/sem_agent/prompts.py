"""Prompt builders for the sem_agent pipeline.

Stage 1 — Ecosystem Profile:
    Code generator extracts three semantic goals as pure-text narratives:
      (1) Candidate Ecosystem   — what world does each candidate belong to?
      (2) Partial Bundle Profile — what style/category world is already in play?
      (3) User Preference Context — what do users who engage with the partial item
          also tend to engage with?

Stage 2 — Gap + Cross-validation:
    Code generator uses Stage 1 profiles as anchors for two further goals:
      (1) Bundle Gap Analysis   — what is missing from the partial bundle?
      (2) Per-candidate Gap Fit — does each candidate fill the gap?

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


def _dataset_domain_note(dataset):
    """Return domain-specific compatibility framing for the given dataset."""
    name = str(dataset or "").lower()
    if "pog" in name:
        return (
            "Domain: fashion outfit completion. "
            "A well-formed outfit combines COMPLEMENTARY items across different "
            "categories (e.g. shoes + bag + earrings + top). "
            "Items from the SAME category as an existing bundle item are almost "
            "never the correct answer. Do NOT interpret high similarity as "
            "compatibility — similar items suggest redundancy, not fit."
        )
    if "spotify" in name:
        return (
            "Domain: music playlist continuation. "
            "A good continuation track shares mood, tempo, or genre context with "
            "the existing tracks rather than duplicating them exactly."
        )
    return (
        "Domain: bundle completion. "
        "Focus on items that complement the partial bundle, not items that "
        "merely resemble existing bundle members."
    )


def _shared_rules(output_file, labels, max_evidence_chars):
    return (
        f"Write UTF-8 JSON to exactly: {output_file}\n"
        f"Required candidate labels: {labels}\n"
        f"Keep the serialized JSON below approximately {int(max_evidence_chars)} characters.\n\n"
        "Output schema:\n"
        "{\n"
        '  "signals": [\n'
        "    {\n"
        '      "signal_name": "stable_snake_case_identifier",\n'
        '      "description": "what semantic goal was investigated and what was found",\n'
        '      "sources": ["exact source filename from manifest"],\n'
        '      "relation_path": ["typed hop 1", "typed hop 2"],\n'
        '      "candidate_observations": {\n'
        '        "A": {\n'
        '          "value": "descriptive string narrative — NO numbers as the sole content",\n'
        '          "evidence": ["short factual sentence 1", "short factual sentence 2"]\n'
        "        }\n"
        "      }\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        "CRITICAL value rules:\n"
        "- `value` MUST be a non-empty descriptive STRING narrative for every candidate.\n"
        "- Do NOT put a bare number (e.g. 0.71) as the sole content of `value`.\n"
        "- Numbers MAY appear inside a narrative string for context "
        '  (e.g. "co-appears in 5 bundles, mostly with bags and dresses").\n'
        "- `evidence` must be a list of ≤3 short factual strings (item titles, "
        "  category names, bundle IDs with context — not raw vector values).\n"
        "- Every signal must compute the same logic for EVERY candidate label.\n"
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
):
    """Stage 1: Ecosystem Profile Builder.

    The code generator pursues three semantic goals and encodes results as
    pure-text narratives — no bare numeric values allowed.
    """
    labels = ", ".join(candidate_labels(case_view))
    domain_note = _dataset_domain_note(case_view["dataset"])

    semantic_goals = (
        "Your code must investigate ALL THREE of the following semantic goals. "
        "Each goal should become one or more signals in the output JSON.\n\n"

        "GOAL 1 — Candidate Ecosystem Profile (required)\n"
        "  Question: What world does each candidate belong to in the training data?\n"
        "  Investigate: Which categories, styles, or item types co-appear with each "
        "candidate in bundles (bi_train.txt) or in user interaction histories "
        "(ui_full.txt)? Use feature tensors only as a retrieval bridge to find "
        "similar items, then look up THOSE items' bundle/user contexts.\n"
        "  Output: For each candidate, a narrative string describing its ecosystem "
        '  (e.g. "earrings — co-appears mainly with dresses, heels, and handbags '
        '  across 12 bundles; users who interacted also bought formal accessories").\n\n'

        "GOAL 2 — Partial Bundle Profile (required)\n"
        "  Question: What style and category world is already established by the "
        "partial bundle item(s)?\n"
        "  Investigate: Which categories and item types most frequently co-appear "
        "with the partial item(s) in bi_train.txt? What do users who interacted "
        "with the partial item also tend to engage with (ui_full.txt)?\n"
        "  Output: A single narrative describing the established ecosystem "
        '  (e.g. "high heels — bundle neighbors are mostly bags, earrings, dresses; '
        '  user neighbors lean toward formal/dressy items").\n'
        "  Note: This goal produces one signal whose `value` is the same for all "
        "candidates (it describes the partial bundle, not per-candidate). Still "
        "include all candidate labels in candidate_observations with the same value.\n\n"

        "GOAL 3 — User Preference Context (required if ui_full.txt is available)\n"
        "  Question: What do users who engaged with the partial item tend to "
        "also engage with, and does that overlap with each candidate?\n"
        "  Investigate: Find users who interacted with the partial item (ui_full.txt). "
        "Among those users' other interactions, which item categories or specific "
        "items appear most? Check whether each candidate falls into those preferred "
        "categories.\n"
        "  Output: For each candidate, a narrative string describing whether its "
        "category/style aligns with the revealed user preferences "
        '  (e.g. "earrings category matches Top-2 user preference category").\n\n'

        "IMPLEMENTATION NOTES\n"
        "- You are FREE to choose any traversal paths through the available sources "
        "to implement these goals. The goals are semantic targets, not algorithmic "
        "prescriptions.\n"
        "- Feature tensors (.pt files) may be used as the FIRST hop to retrieve "
        "similar items, which then serve as anchors for bundle/user lookups. "
        "Do NOT stop at the similarity score itself.\n"
        "- Prefer item titles and category names in evidence over raw IDs alone.\n"
        "- Include `relation_path` listing the typed hops your code actually executes "
        '  (e.g. ["item has content representation", "bundle contains item"]).\n'
    )

    return (
        "You are the Stage 1 Ecosystem Profile Code Generator in a bundle-completion system.\n"
        "Generate ONLY complete executable Python code — no markdown fences, no explanation.\n"
        "The script runs with the allowed workspace as its current directory.\n\n"
        f"{task_semantics(case_view['dataset'])}\n\n"
        f"{domain_note}\n\n"
        f"{semantic_goals}\n"
        f"{_shared_rules(output_file, labels, max_evidence_chars)}\n"
        f"ID-only case:\n{_dump(case_view)}\n\n"
        f"Source Capability Manifest:\n{_dump(source_manifest)}"
    )


# ---------------------------------------------------------------------------
# Stage 2 prompt
# ---------------------------------------------------------------------------

def stage2_gap_prompt(
    case_view,
    source_manifest,
    affordance_graph,
    output_file,
    max_evidence_chars,
    stage1_evidence,
):
    """Stage 2: Gap + Cross-validation Narrator.

    Uses Stage 1 ecosystem profiles as anchors to investigate what the bundle
    is missing and which candidate best fills that gap.
    """
    labels = ", ".join(candidate_labels(case_view))
    domain_note = _dataset_domain_note(case_view["dataset"])
    relation_map = render_affordance_relation_map(affordance_graph)

    semantic_goals = (
        "Stage 1 has already built ecosystem profiles for each candidate and for the "
        "partial bundle. Your task now is deeper cross-validation.\n\n"

        "GOAL 1 — Bundle Gap Analysis (required)\n"
        "  Question: What is the partial bundle MISSING?\n"
        "  Investigate: Given the partial bundle's ecosystem profile from Stage 1, "
        "identify which item categories or types frequently co-appear with the "
        "partial item(s) in bi_train.txt but are NOT yet represented in the partial "
        "bundle. Cross-check with user preference patterns from Stage 1.\n"
        "  Output: A narrative describing the gap "
        '  (e.g. "the partial bundle (high heels) is most commonly paired with '
        '  earrings/accessories (18% of bundles) and bags (28%) — neither is present").\n'
        "  Like Goal 2 in Stage 1, this is a shared signal — same value for all "
        "candidates.\n\n"

        "GOAL 2 — Per-candidate Gap Fit (required)\n"
        "  Question: Does each candidate fill the identified gap?\n"
        "  Investigate: Cross-reference each candidate's ecosystem profile (Stage 1) "
        "against the gap analysis (Goal 1 above). Use multi-hop paths, for example:\n"
        "    - candidate's category vs. gap categories\n"
        "    - bundles containing the candidate: do they also tend to contain the "
        "partial item or similar items?\n"
        "    - users who interact with the candidate: do they overlap with users who "
        "interact with the partial item?\n"
        "  Output: For each candidate, a narrative explaining how well (or poorly) it "
        "fills the bundle gap "
        '  (e.g. "earrings(E) — falls squarely in the top-gap category; '
        '  its bundle ecosystem aligns with the partial item\'s bundle world").\n\n'

        "IMPLEMENTATION NOTES\n"
        "- Every Stage 2 signal MUST traverse at least TWO typed relation hops.\n"
        "- The following are NOT valid as standalone Stage 2 signals: raw cosine "
        "similarity between two items, direct category equality, plain co-occurrence "
        "count (these are Stage 1 work).\n"
        "- Feature tensors MAY be used as the first hop to retrieve anchor items; "
        "the investigation must then continue into bundle, user, category, or "
        "attribute context.\n"
        "- Use a new `signal_name` for new signals. If you materially revise a Stage 1 "
        "signal, reuse its exact name.\n"
        "- Prefer item titles and category names in evidence.\n"
        "- Include `relation_path` listing the typed hops your code actually executes.\n"
    )

    return (
        "You are the Stage 2 Gap & Cross-validation Code Generator in a bundle-completion system.\n"
        "Generate ONLY complete executable Python code — no markdown fences, no explanation.\n"
        "The script runs with the allowed workspace as its current directory.\n\n"
        f"{task_semantics(case_view['dataset'])}\n\n"
        f"{domain_note}\n\n"
        f"{semantic_goals}\n"
        f"{_shared_rules(output_file, labels, max_evidence_chars)}\n"
        f"ID-only case:\n{_dump(case_view)}\n\n"
        f"Source Capability Manifest:\n{_dump(source_manifest)}\n\n"
        f"Compact Evidence Relation Map:\n{relation_map}\n\n"
        f"Stage 1 ecosystem profiles (use as anchors, do not re-compute):\n"
        f"{_dump(stage1_evidence)}"
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
        f"Required candidate labels in every signal: {labels}\n\n"
        "CRITICAL: `value` in every candidate_observation must be a non-empty STRING "
        "narrative. A bare number is not acceptable.{relation_path_rule}\n\n"
        "Preserve this exact schema:\n"
        "{\n"
        '  "signals": [\n'
        "    {\n"
        '      "signal_name": "...",\n'
        '      "description": "...",\n'
        '      "sources": ["..."],\n'
        '      "relation_path": ["hop1", "hop2"],\n'
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


def _candidate_evidence_lines(evidence, label):
    lines = []
    for signal in (evidence.get("signals", []) if isinstance(evidence, dict) else []):
        if not isinstance(signal, dict):
            continue
        obs = (signal.get("candidate_observations") or {}).get(label)
        if not isinstance(obs, dict):
            continue
        name = str(signal.get("signal_name", "signal"))
        sources = signal.get("sources", [])
        src_text = ", ".join(str(s) for s in sources) if isinstance(sources, list) else str(sources)
        path = signal.get("relation_path", [])
        path_text = " -> ".join(str(h) for h in path) if isinstance(path, list) and path else ""
        value_text = str(obs.get("value") or "")
        facts = obs.get("evidence", [])
        fact_text = "; ".join(str(f) for f in facts) if facts else ""
        path_suffix = f" [path: {path_text}]" if path_text else ""
        evidence_suffix = f" | facts: {fact_text}" if fact_text else ""
        lines.append(f"   - [{name}]{path_suffix}: {value_text}{evidence_suffix}")
    return lines


def decision_prompt(decision_case, evidence):
    task_name, bundle_name, item_name = _decision_task_names(decision_case.get("dataset"))
    domain_note = _dataset_domain_note(decision_case.get("dataset"))

    input_str = "; ".join(
        f"{i + 1}. {item.get('text', '')}"
        for i, item in enumerate(decision_case.get("partial_items", []))
    )

    option_blocks = []
    for candidate in decision_case.get("candidates", []):
        label = str(candidate.get("label", ""))
        block = [f"{label}. {candidate.get('text', '')}"]
        ev_lines = _candidate_evidence_lines(evidence, label)
        if ev_lines:
            block.append("   Evidence:")
            block.extend(ev_lines)
        option_blocks.append("\n".join(block))
    target_str = "\n".join(option_blocks)

    return (
        f"You are a helpful and honest assistant evaluating {task_name}.\n"
        f"{domain_note}\n\n"
        "The evidence below was extracted by automated code from training data. "
        "Use it to REASON about which candidate best COMPLEMENTS the partial bundle — "
        "do not simply pick the candidate with the most mentions or longest evidence.\n\n"
        f"Question: Given the partial {bundle_name}: {input_str}\n"
        f"Which candidate {item_name} should be added to complete the {bundle_name}?\n\n"
        f"Options:\n{target_str}\n\n"
        "Respond with ONLY a single letter (e.g. A, B, C ...).\n"
        "Choice:"
    )

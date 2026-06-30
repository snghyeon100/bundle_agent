"""Prompt builders for the sem_agent pipeline.

Stage 1 — Item Evidence Expansion:
    Code generator retrieves item-level supporting evidence for every partial
    item and candidate item. The goal is to find other items that help explain
    each target item through bundle, user, metadata, or feature-neighbor paths.

Stage 2 — Bundle Context & Candidate Fit:
    Code generator uses Stage 1 supporting items as anchors to build a
    bundle-level context, then explains how each candidate's expanded evidence
    fits or conflicts with that context.

Stage 1 stores only evidence strings. Stage 2 stores only summary strings.
"""

import json

from .common import candidate_labels, task_semantics
# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _dump(value):
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def _unified_case_context(case_view, semantic_case=None):
    unified = {
        "case_id": case_view.get("case_id"),
        "dataset": case_view.get("dataset"),
        "bundle_id": case_view.get("bundle_id"),
        "partial_items": [],
        "candidates": [],
    }
    semantic_case = semantic_case or {}
    semantic_partials = {
        int(item.get("item_id")): item
        for item in semantic_case.get("partial_items", [])
        if isinstance(item, dict) and item.get("item_id") is not None
    }
    semantic_candidates = {
        str(item.get("label", "")): item
        for item in semantic_case.get("candidates", [])
        if isinstance(item, dict)
    }
    for item_id in case_view.get("partial_item_ids", []):
        semantic_item = semantic_partials.get(int(item_id), {})
        entry = {"item_id": int(item_id)}
        if semantic_item:
            entry["text"] = semantic_item.get("text", "")
            if semantic_item.get("metadata"):
                entry["metadata"] = semantic_item.get("metadata")
        unified["partial_items"].append(entry)
    for candidate in case_view.get("candidates", []):
        label = str(candidate.get("label", ""))
        item_id = int(candidate.get("item_id"))
        semantic_item = semantic_candidates.get(label, {})
        entry = {"label": label, "item_id": item_id}
        if semantic_item:
            entry["text"] = semantic_item.get("text", "")
            if semantic_item.get("metadata"):
                entry["metadata"] = semantic_item.get("metadata")
        unified["candidates"].append(entry)
    return unified





# ---------------------------------------------------------------------------
# Problem analysis prompt
# ---------------------------------------------------------------------------

def problem_analysis_prompt(case_view, source_manifest, semantic_case=None):
    return (
        "You are a Problem Analysis Agent for a bundle-completion evidence pipeline.\n\n"
        "A partial bundle and multiple candidate items are given. The final task of the full "
        "system is to choose which candidate item should be added to the partial bundle.\n\n"
        "Your role is not to answer the task. Your role is to analyze this specific problem "
        "instance so that a later evidence-retrieval agent can retrieve sample-adaptive evidence.\n\n"
        "Analyze what needs to be understood about this instance in order to retrieve useful, "
        "source-grounded evidence for the partial item(s) and each candidate. Decide for yourself "
        "what aspects of the problem are important. The analysis should be specific to this sample, "
        "not generic advice for all bundle-completion tasks.\n\n"
        "You will also be given a manifest of available data sources. Use it only to reason about "
        "what kinds of evidence could be useful. Do not invent source evidence and do not claim "
        "that a source contains a fact unless it has actually been retrieved later.\n\n"
        "Important constraints:\n"
        "- Do not choose or rank candidates.\n"
        "- Do not judge candidate fit.\n"
        "- Do not produce final recommendations.\n"
        "- Do not invent evidence.\n"
        "- You may infer tentative item roles or problem structure from item text, but express "
        "uncertainty when appropriate.\n\n"
        "Return a concise free-form analysis that would help the evidence-retrieval agent decide "
        "what to retrieve.\n\n"
        f"{task_semantics(case_view['dataset'])}\n\n"
        f"Unified case context (IDs and text together):\n{_dump(_unified_case_context(case_view, semantic_case))}\n\n"
        f"Source Capability Manifest:\n{_dump(source_manifest)}"
    )


# ---------------------------------------------------------------------------
# Evidence retrieval prompt
# ---------------------------------------------------------------------------

def stage1_ecosystem_prompt(
    case_view,
    source_manifest,
    output_file,
    max_evidence_chars,
    semantic_case=None,
    problem_analysis=None,
):
    """Stage 1: Item Evidence Expansion Builder.

    The code generator retrieves item-level supporting evidence and encodes it as
    pure-text narratives — no bare numeric values allowed.
    """
    labels = ", ".join(candidate_labels(case_view))

    semantic_goals = (
        "STAGE 1 TASK - Sample-Adaptive Evidence Program Synthesis\n"
        "Generate executable Python code that performs sample-adaptive evidence retrieval "
        "for this specific bundle-completion instance. Your code should adapt both "
        "(1) which source paths are used and (2) how evidence from those paths is represented. "
        "This stage retrieves and formats source-grounded evidence only; it must not make "
        "candidate-fit judgments or choose an answer.\n\n"

        "AVAILABLE RETRIEVAL OPERATORS\n"
        "Use these operators as a library, not as a fixed checklist. You are not required "
        "to use every operator.\n"
        "1. Co-bundle expansion: use bundle-item data to follow item -> historical bundle -> "
        "other items in the same bundle. This is direct bundle-context evidence but may be sparse.\n"
        "2. Shared-user expansion: use user-item data to follow item -> users -> other items "
        "interacted by those users. This can reveal user/style context but may be noisy or high-fanout.\n"
        "3. Category/metadata anchor expansion: use item metadata to find same or related "
        "category anchors. This grounds item type/style but is not direct compatibility evidence.\n"
        "4. Representation-neighbor expansion: use image, text, content, UI-LightGCN, BI-LightGCN, "
        "or CF features to retrieve nearest-neighbor items. This is useful as a sparse-evidence fallback, "
        "but may retrieve substitutes rather than complementary items.\n"
        "5. Cross-source composition: when useful, compose relations such as item -> bundle -> "
        "co-item -> metadata, item -> user -> interacted item -> bundle, or item -> embedding "
        "neighbor -> metadata/category.\n\n"

        "SAMPLE-ADAPTIVE POLICY\n"
        "First infer a compact sample-level base retrieval policy from the partial item text, "
        "candidate item texts, source availability, item roles, and evidence sparsity. Apply this "
        "base policy to every candidate for comparability. Candidate-specific fallback paths are "
        "allowed only when the base policy returns sparse or empty evidence for that candidate, "
        "or when the candidate's item role makes the base operator insufficient for neutral item "
        "profiling. Do not use deeper retrieval for a candidate because it appears more compatible "
        "with the partial bundle. Your generated code must explicitly construct a `policy_trace` "
        "object before retrieval and then execute retrieval according to that trace. Do not simply "
        "run all operators for every candidate by default.\n\n"

        "SAMPLE-ADAPTIVE EVIDENCE VIEWS\n"
        "Adaptiveness also applies to evidence representation. Your code may compute "
        "sample-relevant evidence views from selected sources, such as grouped examples, "
        "category/role concentration, source sparsity, broad-source warnings, source agreement "
        "or disagreement, high-fanout-filtered contexts, or representative anchors. Use statistics "
        "inside the code to decide what is representative, but output compact qualitative evidence "
        "strings rather than bare numeric scores.\n\n"

        "EVIDENCE STRING RULES\n"
        "For every target item, output evidence strings grounded in retrieved item titles or metadata. "
        "Every evidence string must disclose its path/view in the prefix before ':', for example "
        "'base co-bundle profile via bundle 11186: item title 1; item title 2', "
        "'fallback text-neighbor profile: item title 1; item title 2', or "
        "'sparse co-bundle evidence: no historical co-bundle items found'. "
        "Group evidence by retrieval signal, view, or anchor. For each evidence string/group, include "
        "at most 5 representative item titles. For high-fanout groups, append a compact note such as "
        "'(+1432 more items not shown)'.\n\n"

        "Do not output `value`, `description`, `sources`, `relation_path`, raw scores, "
        "rankings, predictions, fit judgments, or final recommendations. Stage 1 output should "
        "keep only `signal_scope` and evidence arrays.\n\n"

        "STAGE 1 OUTPUT STRUCTURE\n"
        "Produce JSON with `policy_trace` for audit/debug and exactly these two signal shapes. "
        "`policy_trace` is not final decision evidence; it is used only to audit the generated "
        "retrieval program.\n"
        "{\n"
        '  "policy_trace": {\n'
        '    "sample_observation": "short description of item roles/source sparsity that guided retrieval, without ranking candidates",\n'
        '    "base_retrieval_policy": ["shared base retrieval path applied to all candidates"],\n'
        '    "fallback_rules": ["fallback rule used only for sparse evidence or item-role mismatch"],\n'
        '    "evidence_view_policy": ["how selected source evidence is represented, e.g. grouped examples or sparsity notes"]\n'
        "  },\n"
        '  "signals": [\n'
        "    {\n"
        '      "signal_scope": "partial_bundle",\n'
        '      "observation": {\n'
        '        "evidence": [\n'
        '          "Path or anchor name: item title 1; item title 2; item title 3"\n'
        "        ]\n"
        "      }\n"
        "    },\n"
        "    {\n"
        '      "signal_scope": "candidate",\n'
        '      "candidate_observations": {\n'
        '        "A": {\n'
        '          "evidence": [\n'
        '            "Path or anchor name: item title 1; item title 2; item title 3"\n'
        "          ]\n"
        "        }\n"
        "      }\n"
        "    }\n"
        "  ]\n"
        "}\n"
        "For every evidence string, the prefix before ':' must describe the actual "
        "retrieval path, source signal, or anchor used by the generated code.\n"
        "Do not write detailed narratives, fit judgments, rankings, "
        "predictions, or final recommendations in Stage 1.\n\n"

        "IMPLEMENTATION NOTES\n"
        "- You are FREE to choose any traversal paths through the available sources "
        "to implement this task. The task is a semantic target, not an algorithmic "
        "prescription.\n"
        "- Treat the listed sources as a relation graph, not as isolated files. You can compose relations to retrieve supporting items through multi-hop paths.\n"
        "- CPU-only: load every .pt file with torch.load(..., map_location=\"cpu\"). "
        "If a tensor requires gradients, call detach() before numpy().\n"
        "- Keep numpy arrays as numpy arrays while using numpy attributes such as `.size`, "
        "`.shape`, or boolean indexing. If you convert an array to a Python list, use "
        "`len(list_value)` instead of `.size` and do not call `.tolist()` on it again.\n"
    )

    return (
        "You are the Stage 1 Sample-Adaptive Evidence Retrieval Code Generator in a bundle-completion system.\n"
        "Generate ONLY complete executable Python code — no markdown fences, no explanation.\n"
        "The script runs with the allowed workspace as its current directory.\n"
        f"The script must write UTF-8 JSON to exactly this path: {output_file}\n\n"
        f"{task_semantics(case_view['dataset'])}\n\n"
        f"Problem Analysis Guidance (not evidence; do not copy it as evidence):\n"
        f"{problem_analysis or '(none provided)'}\n\n"
        f"{semantic_goals}\n"
        #f"{_shared_rules(output_file, labels, max_evidence_chars)}\n"
        f"Unified case context (IDs and text together):\n{_dump(_unified_case_context(case_view, semantic_case))}\n\n"
        "Use the item text as semantic context for interpreting items, "
        "but ground every signal in the listed workspace sources. Do not rely on "
        "item text alone for final evidence.\n\n"
        f"Source Capability Manifest:\n{_dump(source_manifest)}"
    )


# ---------------------------------------------------------------------------
# Stage 2 prompt
# ---------------------------------------------------------------------------


def stage2_gap_prompt(
    case_view,
    source_manifest,
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
    # Combine Stage 1 evidence into unified case
    partial_evidence = []
    for sig in stage1_evidence.get("signals", []) if isinstance(stage1_evidence, dict) else []:
        if str(sig.get("signal_scope", "")).strip() == "partial_bundle":
            obs = sig.get("observation")
            if isinstance(obs, dict):
                partial_evidence.append({
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
                        "evidence": obs.get("evidence")
                    })

    unified_case = {
        "case_id": case_view.get("case_id"),
        "dataset": case_view.get("dataset"),
        "bundle_id": case_view.get("bundle_id"),
        "partial_items": [],
        "candidates": []
    }

    if semantic_case:
        for item in semantic_case.get("partial_items", []):
            unified_case["partial_items"].append({
                "item_id": item.get("item_id"),
                "text": item.get("text", ""),
                "stage1_evidence": partial_evidence,
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
        unified_case["partial_items"] = [
            {"item_id": item_id, "stage1_evidence": partial_evidence}
            for item_id in case_view.get("partial_item_ids", [])
        ]
        for cand in case_view.get("candidates", []):
            lbl = str(cand.get("label", ""))
            unified_case["candidates"].append({
                "label": lbl,
                "item_id": cand.get("item_id"),
                "stage1_evidence": cand_evidence_by_label.get(lbl, [])
            })

    legacy_stage2_fit_goals = (
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

    semantic_goals = (
        "Stage 1 has already expanded every partial item and candidate item into "
        "supporting items retrieved from the source relation graph. In the Unified Case below, "
        "each item has its text placed next to its Stage 1 evidence. Your task is to build "
        "bundle-completion-aware item profiles, not fit judgments or candidate selections.\n\n"
        "GOAL 1 - Partial Bundle Item Profile (required)\n"
        "  Profile the partial bundle item(s) using the item text and attached Stage 1 evidence. "
        "Describe the observed item role(s), visible/style attributes, occasion or season cues, "
        "bundle-relevant category/role context, and uncertainty. Do not name a missing item type "
        "as the answer and do not recommend any candidate.\n\n"
        "GOAL 2 - Candidate Item Profiles (required)\n"
        "  For each candidate, profile only that candidate's item text and attached Stage 1 evidence. "
        "Describe the item's likely outfit role, style attributes, occasion or season cues, "
        "bundle-relevant category/role context, and mixed or sparse evidence. "
        "Do not compare candidates to each other. Do not describe whether a candidate fits the partial bundle.\n\n"
        "STRICT NEUTRALITY RULES\n"
        "- Do not rank candidates or choose a winner.\n"
        "- Do not use phrases such as strong fit, weak fit, fits well, best, worse, should be selected, "
        "more aligned, less aligned, likely correct, or final answer.\n"
        "- Do not output `value`, `evidence`, `observation`, or `candidate_observations` in Stage 2. "
        "Output one summary string for the partial bundle and one summary string per candidate.\n\n"
        "STAGE 2 OUTPUT STRUCTURE\n"
        "Produce JSON with exactly the following two signal shapes:\n"
        "```json\n"
        "{\n"
        '  "signals": [\n'
        "    {\n"
        '      "signal_name": "partial_bundle_item_summary",\n'
        '      "signal_scope": "partial_bundle",\n'
        '      "description": "bundle-completion-aware neutral item profile of the partial bundle and its Stage 1 evidence",\n'
        '      "summary": "partial bundle item profile using item text and Stage 1 evidence"\n'
        "    },\n"
        "    {\n"
        '      "signal_name": "candidate_item_summaries",\n'
        '      "signal_scope": "candidate",\n'
        '      "description": "bundle-completion-aware neutral item profiles for each candidate and its Stage 1 evidence",\n'
        '      "candidate_summaries": {\n'
        '        "A": "candidate A item profile using item text and Stage 1 evidence",\n'
        '        "B": "candidate B item profile using item text and Stage 1 evidence"\n'
        "      }\n"
        "    }\n"
        "  ]\n"
        "}\n"
        "```\n"
        f"Include every candidate label in `candidate_summaries`: {labels}.\n"
    )

    return (
        "You are the Stage 2 Bundle-Completion-Aware Item Profiler in a bundle-completion system.\n"
        "Your task is to compress item text and Stage 1 evidence into neutral item profiles.\n\n"
        f"{task_semantics(case_view['dataset'])}\n\n"
        f"{semantic_goals}\n"
        f"Unified Case (each item has text next to its Stage 1 evidence):\n{_dump(unified_case)}\n"
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
    require_relation_path=False,
):
    labels = ", ".join(candidate_labels(case_view))
    relation_path_rule = ""
    if require_relation_path:
        relation_path_rule = (
            ""
        )

    return (
        "You are repairing Python signal-extraction code. Return ONLY complete executable "
        "Python code — no markdown fences, no explanation.\n"
        "Fix execution errors or JSON schema defects. "
        "Preserve the original semantic investigation intent.\n"
        "CPU-only: torch.load(..., map_location=\"cpu\").\n\n"
        f"Script must write UTF-8 JSON to exactly: {output_file}\n"
        f"Required candidate labels for every candidate-scoped signal: {labels}\n\n"
        "CRITICAL: `signal_scope` must be either `partial_bundle` or `candidate`. "
        "For `partial_bundle`, use one shared `observation` containing only `evidence`. "
        "For `candidate`, use `candidate_observations` with every required candidate label, "
        "and each label object must contain only `evidence`. Do not output `value`, "
        "`description`, `sources`, `relation_path`, scores, rankings, or explanations. Evidence "
        f"strings should preserve the sample-adaptive retrieval intent and may describe "
        f"compact evidence views such as base paths, fallback paths, sparsity, source agreement, "
        f"or grouped representative anchors. Group items by the same retrieval path or view, such as "
        f"`base co-bundle profile via bundle 11186: item title 1; item title 2`. For high-fanout groups, "
        f"include up to 5 representative titles and append a count such as "
        f"`(+1432 more items not shown)`.{relation_path_rule}\n\n"
        "Preserve this exact schema. Include `policy_trace` for audit/debug only. "
        "Include `observation` only for partial_bundle signals, and include "
        "`candidate_observations` only for candidate signals:\n"
        "{\n"
        '  "policy_trace": {\n'
        '    "sample_observation": "...",\n'
        '    "base_retrieval_policy": ["..."],\n'
        '    "fallback_rules": ["..."],\n'
        '    "evidence_view_policy": ["..."]\n'
        "  },\n"
        '  "signals": [\n'
        "    {\n"
        '      "signal_scope": "partial_bundle | candidate",\n'
        '      "observation": {"evidence": ["..."]},\n'
        '      "candidate_observations": {\n'
        '        "A": {"evidence": ["..."]}\n'
        "      }\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        f"ID-only case:\n{_dump(case_view)}\n\n"
        f"Source Capability Manifest:\n{_dump(source_manifest)}\n\n"
        f"Execution context and defects:\n{_dump(execution_context)}\n\n"
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


def _summary_text(value):
    return " ".join(str(value or "").split())


def _partial_evidence_lines(evidence):
    lines = []
    signals = evidence.get("signals", []) if isinstance(evidence, dict) else []
    for signal in signals:
        if not isinstance(signal, dict):
            continue
        if str(signal.get("signal_scope", "")).strip() != "partial_bundle":
            continue
        summary = _summary_text(signal.get("summary"))
        if not summary and isinstance(signal.get("observation"), dict):
            summary = _summary_text(signal["observation"].get("value"))
        if summary:
            lines.append(f" - {summary}")
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
        summaries = signal.get("candidate_summaries", {})
        summary = summaries.get(label) if isinstance(summaries, dict) else ""
        if not summary:
            observations = signal.get("candidate_observations", {})
            observation = observations.get(label) if isinstance(observations, dict) else None
            if isinstance(observation, dict):
                summary = observation.get("value", "")
        summary = _summary_text(summary)
        if summary:
            lines.append(f"   Stage 2 summary: {summary}")
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
        block.extend(_candidate_evidence_lines(evidence, label))
        option_blocks.append("\n".join(block))
    target_str = "\n".join(option_blocks)
    partial_context = ""
    if partial_lines:
        partial_context = "Shared partial-bundle context:\n" + "\n".join(partial_lines) + "\n"

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

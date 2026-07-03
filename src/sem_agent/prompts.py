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


def _enriched_case_context(case_view, semantic_case=None, data_recon=None):
    unified = _unified_case_context(case_view, semantic_case)
    data_recon = data_recon or {}
    partial_recon = data_recon.get("partial_item_diagnostic") or data_recon.get("partial_items", {})
    candidate_recon = data_recon.get("candidate_level_diagnostic") or data_recon.get("candidates", {})

    for item in unified.get("partial_items", []):
        metadata = item.pop("metadata", {}) if isinstance(item.get("metadata"), dict) else {}
        cate = metadata.get("cate") or metadata.get("cate_id")
        if cate:
            item["cate"] = cate
        recon = partial_recon.get(str(item.get("item_id")), {})
        if recon:
            item["recon"] = recon

    for item in unified.get("candidates", []):
        metadata = item.pop("metadata", {}) if isinstance(item.get("metadata"), dict) else {}
        cate = metadata.get("cate") or metadata.get("cate_id")
        if cate:
            item["cate"] = cate
        recon = candidate_recon.get(str(item.get("label")), {})
        if recon:
            item["recon"] = recon

    if data_recon.get("sample_level_diagnostic"):
        unified["sample_level_diagnostic"] = data_recon.get("sample_level_diagnostic")
    elif data_recon.get("partial_set"):
        unified["partial_set_recon"] = data_recon.get("partial_set")
    if data_recon.get("partial_item_diagnostic"):
        unified["partial_item_diagnostic"] = data_recon.get("partial_item_diagnostic")
    if data_recon.get("candidate_level_diagnostic"):
        unified["candidate_level_diagnostic"] = data_recon.get("candidate_level_diagnostic")
    if data_recon.get("aggregate_candidate_diagnostic"):
        unified["aggregate_candidate_diagnostic"] = data_recon.get("aggregate_candidate_diagnostic")
    if data_recon.get("recon_legend"):
        unified["recon_legend"] = data_recon.get("recon_legend")
    if data_recon.get("sources_read"):
        unified["recon_sources_read"] = data_recon.get("sources_read")
    return unified





# ---------------------------------------------------------------------------
# Problem analysis prompt
# ---------------------------------------------------------------------------

def problem_analysis_prompt(case_view, source_manifest, semantic_case=None, data_recon=None):
    return (
        "You are a Problem Analysis Agent for a bundle-completion evidence pipeline.\n\n"
        "A partial bundle and multiple candidate items are given. The final task of the full "
        "system is to choose which candidate item should be added to the partial bundle.\n\n"
        "Your role is not to answer the task. Your role is to analyze this specific problem "
        "instance and specify what source-grounded evidence should be retrieved, from which "
        "sources, and through which retrieval patterns, so that a later code-generation agent "
        "can synthesize the evidence retrieval program.\n\n"
        "Analyze what needs to be understood about this instance in order to retrieve useful, "
        "source-grounded evidence for the partial item(s) and each candidate. Decide for yourself "
        "what aspects of the problem are important. The analysis should be specific to this "
        "sample while staying grounded in the bundle-completion task: identify what evidence "
        "would clarify compatibility, complementarity, item roles, and missing-bundle context "
        "for this partial bundle and its candidates. Avoid generic advice that would apply to "
        "any sample.\n\n"
        "You will also be given a manifest of available data sources and a small deterministic "
        "data reconnaissance summary. Use the reconnaissance only to calibrate retrieval strategy "
        "around sparsity, category structure, and source availability. It is not final compatibility "
        "evidence and should not be used to choose a candidate. Use recon findings to choose "
        "`operator_hint`, requested `views`, `filters_or_grouping`, caps/diversity rules, and "
        "`fallback` conditions for Stage 1 code synthesis; do not cite recon values as evidence.\n\n"
        "Important constraints:\n"
        "- Do not choose or rank candidates.\n"
        "- Do not judge candidate fit.\n"
        "- Do not produce final recommendations.\n"
        "- Do not invent evidence.\n"
        "- You may infer tentative item roles or problem structure from item text, but express "
        "uncertainty when appropriate.\n\n"
        "Return ONLY JSON with this compact schema. Keep it sample-adaptive and concise:\n"
        "{\n"
        '  "summary": {\n'
        '    "partial": "One short sentence about the partial item(s).",\n'
        '    "candidates": "One short sentence about the candidate roles/categories and notable contrasts.",\n'
        '    "recon": [\n'
        '      "Short implication for retrieval based on the deterministic recon."\n'
        "    ]\n"
        "  },\n"
        '  "strategy": {\n'
        '    "sample_context": [\n'
        "      {\n"
        '        "name": "short_unique_strategy_name",\n'
        '        "intent": "Why this evidence is needed for this sample, without judging fit.",\n'
        '        "paths": ["source_name.ext"],\n'
        '        "operator_hint": "Code-friendly retrieval pattern, e.g. category_pair_bundles, embedding_neighbors, neighbor_then_bundle_intersection, same_category_multiuse, keyword_filtered_bundles, co_bundle_context.",\n'
        '        "anchors": {\n'
        '          "partial_items": ["item ids or labels"],\n'
        '          "candidates": ["candidate labels if applicable"],\n'
        '          "categories": ["category ids or category roles if applicable"],\n'
        '          "keywords": ["title/style/season/motif cues if useful"]\n'
        "        },\n"
        '        "operation": "Executable retrieval instruction for Stage 1 code synthesis.",\n'
        '        "views": ["Separate evidence views to output, e.g. counts, representative_bundle_titles, neighbor_titles, co_occurring_item_titles, sparse_note"],\n'
        '        "filters_or_grouping": ["Sample-specific filters, grouping keys, caps, diversity rules, or category constraints."],\n'
        '        "fallback": "Fallback retrieval to use if the primary view is sparse or unavailable."\n'
        "      }\n"
        "    ],\n"
        '    "item": {\n'
        '      "partial_6606": [\n'
        "        {\n"
        '          "name": "short_unique_partial_item_strategy_name",\n'
        '          "intent": "Why this partial item needs item-level evidence, without judging fit.",\n'
        '          "paths": ["source_name.ext"],\n'
        '          "operator_hint": "Code-friendly retrieval pattern.",\n'
        '          "anchors": {\n'
        '            "role": "partial",\n'
        '            "item_id": "item id",\n'
        '            "item_category": "category id or role",\n'
        '            "keywords": ["partial-item title/style/season/motif cues"]\n'
        "          },\n"
        '          "operation": "Partial-item executable retrieval instruction, not a final-answer hint.",\n'
        '          "views": ["Separate evidence views to output for this partial item."],\n'
        '          "filters_or_grouping": ["Partial-item filters, grouping keys, caps, or diversity rules."],\n'
        '          "fallback": "Partial-item fallback retrieval if the primary view is sparse."\n'
        "        }\n"
        "      ],\n"
        '      "candidate_A": [\n'
        "        {\n"
        '          "name": "short_unique_candidate_strategy_name",\n'
        '          "intent": "Why this candidate needs this evidence, without judging fit.",\n'
        '          "paths": ["source_name.ext"],\n'
        '          "operator_hint": "Code-friendly retrieval pattern.",\n'
        '          "anchors": {\n'
        '            "role": "candidate",\n'
        '            "label": "A",\n'
        '            "candidate_item": "item id",\n'
        '            "candidate_category": "category id or role",\n'
        '            "partial_items": ["item ids"],\n'
        '            "partial_categories": ["category ids"],\n'
        '            "keywords": ["candidate-specific title/style/season/motif cues"]\n'
        "          },\n"
        '          "operation": "Candidate-specific executable retrieval instruction, not a fit judgment.",\n'
        '          "views": ["Separate evidence views to output for this candidate."],\n'
        '          "filters_or_grouping": ["Candidate-specific filters, grouping keys, caps, or diversity rules."],\n'
        '          "fallback": "Candidate-specific fallback retrieval if the primary view is sparse."\n'
        "        }\n"
        "      ]\n"
        "    }\n"
        "  }\n"
        "}\n\n"
        "The `strategy.sample_context` list should contain code-friendly retrieval specs for "
        "shared problem-level context that should be retrieved once for the whole sample. Each "
        "spec must name source paths, `operator_hint`, concrete `anchors`, "
        "`operation`, requested `views`, filters/grouping, and fallback. `operator_hint` is a "
        "hint for the later code generator, not a closed vocabulary; choose the simplest reusable "
        "retrieval pattern that fits the evidence need. Avoid generic operations like plain nearest "
        "neighbors unless you add sample-specific anchors, filters, clusters, or aggregation. "
        "Separate strategies into `sample_context` and `item`: use `sample_context` for shared "
        "problem context, and use `item` for evidence needs tied to one partial item or one "
        "candidate item. Use item keys like `partial_6606` and `candidate_A`. "
        "`strategy.item` should include partial-item strategies when the partial item itself needs "
        "item-level evidence, and candidate-item strategies for candidates that need special "
        "retrieval because of role, category overlap, sparsity, season/style cues, motif cues, "
        "or other sample-specific issues. Each entry must use the same code-friendly spec format. "
        "Use multiple `views` when one strategy needs multiple retrieval substeps; the later "
        "evidence stage should be able to output one evidence string per view.\n\n"
        f"{task_semantics(case_view['dataset'])}\n\n"
        "Enriched case context (IDs/text/category plus deterministic planning recon):\n"
        f"{_dump(_enriched_case_context(case_view, semantic_case, data_recon))}\n\n"
        "Recon fields are planning inputs, not final compatibility evidence. Use item-level "
        "`recon` fields and `candidate_level_diagnostic` patterns for item-level strategy; "
        "use `sample_level_diagnostic` and `aggregate_candidate_diagnostic` for sample-context strategy. "
        "Use `recon_legend` to interpret recon field names.\n\n"
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
    compact source-grounded evidence strings. Numeric counts/ratios are allowed
    only when tied to a named retrieval view, not as final scores.
    """
    labels = ", ".join(candidate_labels(case_view))
    partial_item_keys = [f"partial_{int(item_id)}" for item_id in case_view.get("partial_item_ids", [])]
    candidate_item_keys = [f"candidate_{label}" for label in candidate_labels(case_view)]
    item_keys = ", ".join(partial_item_keys + candidate_item_keys)

    semantic_goals = (
        "STAGE 1 TASK - Analysis-Conditioned Evidence Code Generation\n"
        "A partial bundle and multiple candidate items are given. The final task of the full "
        "system is to choose which candidate item should be added to the partial bundle.\n\n"
        "Your role is not to answer the task. Generate complete Python code that retrieves "
        "source-grounded evidence for the partial item(s) and every candidate. Do not choose, "
        "rank, recommend, or judge candidate fit.\n\n"
        "Treat the Problem Analysis `strategy` section as the retrieval program spec for this "
        "sample. For every named sample-context or item strategy, copy its exact name into "
        "`SAMPLE_CONTEXT_PLAN` or `ITEM_PLAN` and preserve its `operator_hint`, anchors, operation, "
        "views, filters/grouping, and fallback. Either implement each requested view with "
        "available workspace sources or record the exact strategy/view name and concrete "
        "data/source reason in `skipped_analysis_needs`.\n\n"
        "Do not replace sample-specific strategies with a generic nearest-neighbor, similarity, "
        "metadata, or co-bundle recipe unless the strategy asks for that view. Use "
        "`operator_hint` as the main dispatch hint in `run_strategy`; if an older analysis lacks "
        "`operator_hint`, infer the closest retrieval pattern from `operation`, `paths`, and "
        "anchors. Keep item evidence comparable: apply shared `sample_context` strategies once "
        "for the whole problem, and add item-specific views only where the Problem Analysis asks "
        "for them.\n\n"
        "Build `policy_trace` from the exact Problem Analysis strategy names before retrieval. "
        "For each name, record implemented retrieval paths, skipped subviews, and fallbacks used. "
        "Use grouped examples, category/role concentration, source sparsity, source agreement or "
        "disagreement, high-fanout-filtered contexts, representative anchors, counts, ratios, or "
        "similarities only when they are tied to a named retrieval view. A generated program "
        "should synthesize code from the strategy specs, not rewrite them into a different "
        "generic plan.\n\n"

        "EVIDENCE STRING RULES\n"
        "For every target item, output evidence strings grounded in retrieved item titles or metadata. "
        "Every evidence string must use this compact format: "
        "`retrieval path/source signal/anchor: grounded title list or sparse-evidence note`. "
        "If a strategy requests multiple `views` or contains multiple retrieval substeps, output "
        "separate evidence strings for the meaningful subviews instead of compressing the whole "
        "strategy into one title list. "
        "Group evidence by retrieval signal, view, or anchor. For each evidence string/group, include "
        "at most 5 representative item titles in total. If no grounded item titles are available, "
        "use a sparse-evidence note. For high-fanout groups, append a compact note such as "
        "'(+1432 more items not shown)'.\n\n"

        "Do not output `value`, `description`, `sources`, `relation_path`, rankings, "
        "predictions, fit judgments, or final recommendations. Do not output ungrounded raw "
        "scores. Counts, ratios, or similarities are allowed only when an analysis instruction "
        "asks for them and the evidence string names the retrieval path and what was counted or "
        "compared. Stage 1 output should keep only `signal_scope`, `observation`, "
        "`item_observations`, item identity fields, and evidence arrays.\n\n"

        "STAGE 1 OUTPUT STRUCTURE\n"
        "Produce JSON with `policy_trace` for audit/debug and exactly these two signal shapes. "
        "`policy_trace` is not final decision evidence; it is used only to audit the generated "
        "retrieval program.\n"
        "{\n"
        '  "policy_trace": {\n'
        '    "analysis_driven_needs": ["exact strategy instruction name from the Problem Analysis"],\n'
        '    "implemented_retrieval_paths": ["exact instruction name -> source-grounded retrieval path implemented in code"],\n'
        '    "skipped_analysis_needs": ["exact instruction name/subview -> concrete data or source reason skipped"],\n'
        '    "fallbacks": ["fallback rule used only for sparse evidence or item-role mismatch"],\n'
        '    "evidence_view_policy": ["how selected source evidence is represented, e.g. grouped examples or sparsity notes"]\n'
        "  },\n"
        '  "signals": [\n'
        "    {\n"
        '      "signal_scope": "sample_context",\n'
        '      "observation": {\n'
        '        "evidence": [\n'
        '          "shared sample-level evidence string following EVIDENCE STRING RULES"\n'
        "        ]\n"
        "      }\n"
        "    },\n"
        "    {\n"
        '      "signal_scope": "item",\n'
        '      "item_observations": {\n'
        '        "partial_6606": {\n'
        '          "role": "partial",\n'
        '          "item_id": 6606,\n'
        '          "evidence": [\n'
        '            "partial item evidence string following EVIDENCE STRING RULES"\n'
        "          ]\n"
        "        },\n"
        '        "candidate_A": {\n'
        '          "role": "candidate",\n'
        '          "label": "A",\n'
        '          "item_id": 43712,\n'
        '          "evidence": [\n'
        '            "candidate item evidence string following EVIDENCE STRING RULES"\n'
        "          ]\n"
        "        }\n"
        "      }\n"
        "    }\n"
        "  ]\n"
        "}\n"
        "Do not write detailed narratives, fit judgments, rankings, "
        "predictions, or final recommendations in Stage 1.\n\n"

        "IMPLEMENTATION NOTES\n"
        "- If the Problem Analysis provides strategy instructions, follow them as the "
        "sample-specific retrieval contract.\n"
        "- Preserve each strategy dict from the Problem Analysis as plan config when possible; "
        "`SAMPLE_CONTEXT_PLAN` and `ITEM_PLAN` should expose `name`, `operator_hint`, `anchors`, "
        "`views`, and fallback details to `run_strategy`.\n"
        "- Treat the listed sources as a relation graph, not as isolated files. You can compose relations when useful to retrieve supporting items.\n"
        "- CPU-only: load every .pt file with torch.load(..., map_location=\"cpu\"). "
        "If a tensor requires gradients, call detach() before numpy().\n"
        "- Keep numpy arrays as numpy arrays while using numpy attributes such as `.size`, "
        "`.shape`, or boolean indexing. If you convert an array to a Python list, use "
        "`len(list_value)` instead of `.size` and do not call `.tolist()` on it again.\n"
        "\n"
        "REQUIRED SCRIPT SHAPE\n"
        "Use this high-level script structure. You may adapt helper internals, but keep this shape:\n"
        "```python\n"
        "def main():\n"
        "    sources = load_sources()\n"
        "    indexes = build_indexes(sources)\n"
        "    policy_trace = init_policy_trace(SAMPLE_CONTEXT_PLAN, ITEM_PLAN)\n"
        "\n"
        "    sample_context_evidence = []\n"
        "    for cfg in SAMPLE_CONTEXT_PLAN:\n"
        "        sample_context_evidence.extend(run_strategy(cfg, indexes, scope='sample_context'))\n"
        "\n"
        "    item_evidence = {key: [] for key in REQUIRED_ITEM_KEYS}\n"
        "    for key in REQUIRED_ITEM_KEYS:\n"
        "        for cfg in ITEM_PLAN.get(key, []):\n"
        "            item_evidence[key].extend(run_strategy(cfg, indexes, scope='item', item_key=key))\n"
        "        if not item_evidence[key]:\n"
        "            item_evidence[key].append(make_sparse_note(key))\n"
        "\n"
        "    output_obj = build_output(policy_trace, sample_context_evidence, item_evidence)\n"
        "    write_json(output_obj, OUTPUT_PATH)\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
        "```\n"
        "`run_strategy` should dispatch primarily on `cfg['operator_hint']` and then on "
        "`cfg['name']` when needed for sample-specific details. For each requested view in "
        "`cfg['views']`, either append a separate evidence string or record that exact "
        "strategy/view in `policy_trace['skipped_analysis_needs']`.\n"
        "Do not emit partial code. If a strategy is complex, implement a simpler "
        "source-grounded approximation through `run_strategy` and record skipped subviews "
        "in `policy_trace`.\n"
    )

    return (
        "You are the Stage 1 Analysis-Conditioned Evidence Retrieval Code Generator in a bundle-completion system.\n"
        "Generate ONLY complete executable Python code — no markdown fences, no explanation.\n"
        "The script runs with the allowed workspace as its current directory.\n"
        f"The script must write UTF-8 JSON to exactly this path: {output_file}\n\n"
        f"{task_semantics(case_view['dataset'])}\n\n"
        f"Required item keys for Stage 1 item evidence: {item_keys}\n"
        f"Required candidate labels: {labels}\n\n"
        f"Problem Analysis Retrieval Contract (not evidence; do not copy it as evidence):\n"
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
    sample_context_evidence = []
    item_evidence_by_key = {}
    for sig in stage1_evidence.get("signals", []) if isinstance(stage1_evidence, dict) else []:
        scope = str(sig.get("signal_scope", "")).strip()
        if scope == "sample_context":
            obs = sig.get("observation")
            if isinstance(obs, dict):
                sample_context_evidence.append({
                    "evidence": obs.get("evidence")
                })
        elif scope == "item":
            item_obs = sig.get("item_observations")
            if isinstance(item_obs, dict):
                for key, obs in item_obs.items():
                    if not isinstance(obs, dict):
                        continue
                    item_evidence_by_key.setdefault(str(key), []).append({
                        "role": obs.get("role"),
                        "item_id": obs.get("item_id"),
                        "label": obs.get("label"),
                        "evidence": obs.get("evidence"),
                    })
        elif scope == "partial_bundle":
            # Backward compatibility for older Stage 1 outputs.
            obs = sig.get("observation")
            if isinstance(obs, dict):
                sample_context_evidence.append({
                    "evidence": obs.get("evidence")
                })
        elif scope == "candidate":
            # Backward compatibility for older Stage 1 outputs.
            c_obs = sig.get("candidate_observations")
            if isinstance(c_obs, dict):
                for lbl, obs in c_obs.items():
                    if not isinstance(obs, dict):
                        continue
                    item_evidence_by_key.setdefault(f"candidate_{lbl}", []).append({
                        "role": "candidate",
                        "label": lbl,
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
            item_key = f"partial_{int(item.get('item_id'))}"
            unified_case["partial_items"].append({
                "item_key": item_key,
                "item_id": item.get("item_id"),
                "text": item.get("text", ""),
                "stage1_sample_context": sample_context_evidence,
                "stage1_item_evidence": item_evidence_by_key.get(item_key, []),
            })
        for cand in semantic_case.get("candidates", []):
            lbl = str(cand.get("label", ""))
            item_key = f"candidate_{lbl}"
            unified_case["candidates"].append({
                "item_key": item_key,
                "label": lbl,
                "item_id": cand.get("item_id"),
                "text": cand.get("text", ""),
                "stage1_sample_context": sample_context_evidence,
                "stage1_item_evidence": item_evidence_by_key.get(item_key, [])
            })
    else:
        unified_case["partial_items"] = [
            {
                "item_key": f"partial_{int(item_id)}",
                "item_id": item_id,
                "stage1_sample_context": sample_context_evidence,
                "stage1_item_evidence": item_evidence_by_key.get(f"partial_{int(item_id)}", []),
            }
            for item_id in case_view.get("partial_item_ids", [])
        ]
        for cand in case_view.get("candidates", []):
            lbl = str(cand.get("label", ""))
            item_key = f"candidate_{lbl}"
            unified_case["candidates"].append({
                "item_key": item_key,
                "label": lbl,
                "item_id": cand.get("item_id"),
                "stage1_sample_context": sample_context_evidence,
                "stage1_item_evidence": item_evidence_by_key.get(item_key, [])
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
        "Stage 1 has retrieved shared sample-context evidence and item-level evidence for "
        "partial and candidate items. In the Unified Case below, each item has its text next "
        "to both the shared Stage 1 sample context and its own item evidence. Your task is to "
        "build bundle-completion-aware summaries, not fit judgments or candidate selections.\n\n"
        "GOAL 1 - Sample Context Summary (required)\n"
        "  Summarize the shared problem-level context implied by Stage 1 sample-context evidence: "
        "partial-bundle category/role context, candidate-set contrasts, source sparsity, common "
        "co-occurrence patterns, and uncertainty. Do not recommend any candidate.\n\n"
        "GOAL 2 - Item Summaries (required)\n"
        "  For each partial item and each candidate item, profile only that item's text and "
        "attached Stage 1 item evidence. Describe the item's likely outfit role, style attributes, "
        "occasion or season cues, bundle-relevant category/role context, and mixed or sparse evidence. "
        "Do not compare candidates to each other. Do not describe whether a candidate fits the partial bundle.\n\n"
        "STRICT NEUTRALITY RULES\n"
        "- Do not rank candidates or choose a winner.\n"
        "- Do not use phrases such as strong fit, weak fit, fits well, best, worse, should be selected, "
        "more aligned, less aligned, likely correct, or final answer.\n"
        "- Do not output `value`, `evidence`, `observation`, `item_observations`, or "
        "`candidate_observations` in Stage 2. Output one sample-context summary and one "
        "summary string per item key.\n\n"
        "STAGE 2 OUTPUT STRUCTURE\n"
        "Produce JSON with exactly the following two signal shapes:\n"
        "```json\n"
        "{\n"
        '  "signals": [\n'
        "    {\n"
        '      "signal_name": "sample_context_summary",\n'
        '      "signal_scope": "sample_context",\n'
        '      "description": "shared problem-level context summarized from Stage 1 sample-context evidence",\n'
        '      "summary": "sample-context summary using shared Stage 1 evidence"\n'
        "    },\n"
        "    {\n"
        '      "signal_name": "item_summaries",\n'
        '      "signal_scope": "item",\n'
        '      "description": "bundle-completion-aware neutral item profiles for partial and candidate items",\n'
        '      "item_summaries": {\n'
        '        "partial_6606": "partial item profile using item text and Stage 1 item evidence",\n'
        '        "candidate_A": "candidate A item profile using item text and Stage 1 item evidence"\n'
        "      }\n"
        "    }\n"
        "  ]\n"
        "}\n"
        "```\n"
        "Include every partial item key and every candidate item key in `item_summaries`.\n"
    )

    return (
        "You are the Stage 2 Bundle-Completion-Aware Item Profiler in a bundle-completion system.\n"
        "Your task is to compress Stage 1 sample-context evidence and item-level evidence "
        "into neutral sample and item summaries.\n\n"
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
    problem_analysis=None,
):
    labels = ", ".join(candidate_labels(case_view))
    partial_item_keys = [f"partial_{int(item_id)}" for item_id in case_view.get("partial_item_ids", [])]
    candidate_item_keys = [f"candidate_{label}" for label in candidate_labels(case_view)]
    item_keys = ", ".join(partial_item_keys + candidate_item_keys)
    relation_path_rule = ""
    if require_relation_path:
        relation_path_rule = (
            ""
        )

    return (
        "You are repairing Python signal-extraction code. Return ONLY complete executable "
        "Python code — no markdown fences, no explanation.\n"
        "Fix execution errors, JSON schema defects, or analysis-contract coverage defects. "
        "Preserve the Problem Analysis retrieval contract when it is provided; do not treat "
        "the previous code as authoritative if it omitted or weakened analysis instructions.\n"
        "If the Problem Analysis contains code-friendly strategy specs with fields such as "
        "`operator_hint`, `anchors`, `views`, `filters_or_grouping`, and `fallback`, preserve "
        "those fields in `SAMPLE_CONTEXT_PLAN` or `ITEM_PLAN` and repair `run_strategy` to dispatch "
        "from them. If a requested view is infeasible, skip that exact strategy/view with a "
        "concrete data/source reason instead of dropping the whole strategy.\n"
        "When repairing long or truncated code, prefer compact helper functions and config-driven "
        "loops over repeated per-candidate blocks. Do not repair by continuing separate long "
        "`try` blocks for candidates A-J; use an `ITEM_PLAN` loop with shared helpers. "
        "Always finish by assembling the final JSON object and writing it to the required "
        "output path.\n"
        "CPU-only: torch.load(..., map_location=\"cpu\").\n\n"
        f"Script must write UTF-8 JSON to exactly: {output_file}\n"
        f"Required item keys for item-scoped evidence: {item_keys}\n"
        f"Required candidate labels: {labels}\n\n"
        "CRITICAL: `signal_scope` must be either `sample_context` or `item`. "
        "For `sample_context`, use one shared `observation` containing only `evidence`. "
        "For `item`, use `item_observations` with every required item key. "
        "Each item object must contain `evidence` and may include only item identity fields "
        "such as `role`, `item_id`, and `label`. Do not output `value`, "
        "`description`, `sources`, `relation_path`, rankings, or explanations. Do not output "
        "ungrounded raw scores. Counts, ratios, or similarities are allowed only when tied to "
        "a named retrieval path requested by the analysis. Evidence "
        f"strings must preserve the sample-adaptive retrieval contract and may describe "
        f"compact evidence views such as base paths, fallback paths, sparsity, source agreement, "
        f"or grouped representative anchors. If a strategy requests multiple views, output "
        f"separate evidence strings for the meaningful views. Group items by the same retrieval path or view, such as "
        f"`base co-bundle profile via bundle 11186: item title 1; item title 2`. For high-fanout groups, "
        f"include up to 5 representative titles and append a count such as "
        f"`(+1432 more items not shown)`.{relation_path_rule}\n\n"
        "Preserve this exact schema. Include `policy_trace` for audit/debug only. "
        "Include `observation` only for sample_context signals, and include "
        "`item_observations` only for item signals:\n"
        "{\n"
        '  "policy_trace": {\n'
        '    "analysis_driven_needs": ["exact strategy instruction name from the Problem Analysis"],\n'
        '    "implemented_retrieval_paths": ["exact instruction name -> source-grounded retrieval path implemented in code"],\n'
        '    "skipped_analysis_needs": ["exact instruction name/subview -> concrete data or source reason skipped"],\n'
        '    "fallbacks": ["..."],\n'
        '    "evidence_view_policy": ["..."]\n'
        "  },\n"
        '  "signals": [\n'
        "    {\n"
        '      "signal_scope": "sample_context | item",\n'
        '      "observation": {"evidence": ["..."]},\n'
        '      "item_observations": {\n'
        '        "partial_6606": {"role": "partial", "item_id": 6606, "evidence": ["..."]},\n'
        '        "candidate_A": {"role": "candidate", "label": "A", "item_id": 43712, "evidence": ["..."]}\n'
        "      }\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        f"Problem Analysis Retrieval Contract, if available:\n{problem_analysis or '(none provided)'}\n\n"
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
        scope = str(signal.get("signal_scope", "")).strip()
        name = str(signal.get("signal_name", "")).strip()
        if scope == "sample_context" or name == "sample_context_summary":
            summary = _summary_text(signal.get("summary"))
            if summary:
                lines.append(f" - Sample context: {summary}")
        elif scope == "item":
            summaries = signal.get("item_summaries", {})
            if isinstance(summaries, dict):
                for key, summary in summaries.items():
                    if str(key).startswith("partial_"):
                        text = _summary_text(summary)
                        if text:
                            lines.append(f" - {key}: {text}")
        elif scope == "partial_bundle":
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
        if scope in {"partial_bundle", "sample_context"}:
            continue
        if scope == "item":
            summaries = signal.get("item_summaries", {})
            summary = summaries.get(f"candidate_{label}") if isinstance(summaries, dict) else ""
            summary = _summary_text(summary)
            if summary:
                lines.append(f"   Stage 2 summary: {summary}")
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

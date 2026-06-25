"""sem_agent pipeline.

Two-stage fixed pipeline:
  Stage 1 (Item Evidence Expansion)      → code retrieves supporting items that
                                           explain each partial/candidate item.
  Stage 2 (Bundle Context & Candidate Fit) → code builds bundle-level context
                                             and candidate fit narratives.
  Decision                               → reasoning LLM picks the best candidate
                                           using narrative evidence (no bare numbers).
"""

import asyncio
import json
import os
import re

from .common import (
    build_case_view,
    candidate_labels,
    compact_json,
    execution_summary,
    extract_python_code,
    parse_json_from_text,
    task_semantics,
)
from .workspace import (
    build_source_manifest,
    execute_generated_code,
    execution_needs_repair,
    prepare_workspace,
)
from .affordance_graph import build_evidence_affordance_graph
from .prompts import (
    decision_prompt,
    repair_prompt,
    stage1_ecosystem_prompt,
    stage2_gap_prompt,
)


CONFIG_PREFIX = "sem"


# ---------------------------------------------------------------------------
# LLM call helper
# ---------------------------------------------------------------------------

async def _call_stage(generate_content_fn, client, conf, prompt, max_tokens_key, default_tokens, step_name):
    return await generate_content_fn(
        client,
        conf["model"],
        prompt,
        conf,
        conf.get(max_tokens_key, default_tokens),
        step_name,
    )


# ---------------------------------------------------------------------------
# Evidence validation
# ---------------------------------------------------------------------------

def _forbidden_keys(value, path=""):
    forbidden = {"prediction", "winner", "best_candidate", "preferred_candidate",
                 "recommendation", "ranking", "final_score"}
    issues = []
    if isinstance(value, dict):
        for k, v in value.items():
            cur = f"{path}.{k}" if path else str(k)
            if str(k).strip().lower() in forbidden:
                issues.append(f"Forbidden decision field: {cur}")
            issues.extend(_forbidden_keys(v, cur))
    elif isinstance(value, list):
        for i, v in enumerate(value):
            issues.extend(_forbidden_keys(v, f"{path}[{i}]"))
    return issues


def _validate_observation_payload(payload, path):
    issues = []
    if not isinstance(payload, dict):
        return [f"{path} must be an object."]
    if set(payload) != {"value", "evidence"}:
        issues.append(f"{path} must contain only value and evidence.")
    val = payload.get("value")
    if not isinstance(val, str) or not val.strip():
        issues.append(
            f"{path}.value must be a non-empty string narrative "
            f"(got {type(val).__name__}: {repr(val)[:60]})."
        )
    ev = payload.get("evidence")
    if not isinstance(ev, list):
        issues.append(f"{path}.evidence must be a list.")
    elif len(ev) > 5:
        issues.append(f"{path}.evidence exceeds 5 entries.")
    return issues


def validate_evidence(evidence, labels, allowed_source_names, max_chars=30000, require_relation_path=False):
    """Validate the generated evidence JSON.

    Extra check vs simple_signal_agent: every signal's `value` must be a
    non-empty string (not a bare number).
    """
    if not isinstance(evidence, dict):
        return ["Evidence must be a JSON object."]
    issues = _forbidden_keys(evidence)
    unexpected = sorted(set(evidence) - {"signals"})
    if unexpected:
        issues.append("Unexpected top-level fields: " + ", ".join(unexpected))
    signals = evidence.get("signals")
    if not isinstance(signals, list) or not signals:
        issues.append("signals must be a non-empty list.")
        return issues

    allowed = set(allowed_source_names)
    seen = set()
    required = set(labels)

    for idx, sig in enumerate(signals):
        pfx = f"signals[{idx}]"
        if not isinstance(sig, dict):
            issues.append(f"{pfx} must be an object.")
            continue

        allowed_fields = {
            "signal_name",
            "signal_scope",
            "description",
            "sources",
            "relation_path",
            "observation",
            "candidate_observations",
        }
        extra = sorted(set(sig) - allowed_fields)
        if extra:
            issues.append(f"{pfx} has unsupported fields: {', '.join(extra)}")

        name = str(sig.get("signal_name", "")).strip()
        if not name:
            issues.append(f"{pfx}.signal_name must be non-empty.")
        elif name in seen:
            issues.append(f"Duplicate signal_name: {name}")
        seen.add(name)

        if not str(sig.get("description", "")).strip():
            issues.append(f"{pfx}.description must be non-empty.")

        scope = str(sig.get("signal_scope", "")).strip()
        if scope not in {"partial_bundle", "candidate"}:
            issues.append(f"{pfx}.signal_scope must be either partial_bundle or candidate.")

        srcs = sig.get("sources")
        if not isinstance(srcs, list) or not srcs:
            issues.append(f"{pfx}.sources must be a non-empty list.")
        else:
            unknown = sorted({str(s) for s in srcs if str(s) not in allowed})
            if unknown:
                issues.append(f"{pfx}.sources contains unavailable sources: {', '.join(unknown)}")

        rpath = sig.get("relation_path")
        if require_relation_path:
            if not isinstance(rpath, list) or len(rpath) < 2:
                issues.append(f"{pfx}.relation_path must have at least two typed hops in Stage 2.")

        if scope == "partial_bundle":
            if "candidate_observations" in sig:
                issues.append(f"{pfx} with partial_bundle scope must not include candidate_observations.")
            issues.extend(_validate_observation_payload(sig.get("observation"), f"{pfx}.observation"))
            continue

        if scope == "candidate":
            if "observation" in sig:
                issues.append(f"{pfx} with candidate scope must not include observation.")
            obs = sig.get("candidate_observations")
            if not isinstance(obs, dict):
                issues.append(f"{pfx}.candidate_observations must be an object.")
                continue
            missing = sorted(required - set(obs))
            extra_labels = sorted(set(obs) - required)
            if missing:
                issues.append(f"{pfx} missing candidates: {', '.join(missing)}")
            if extra_labels:
                issues.append(f"{pfx} unknown candidates: {', '.join(extra_labels)}")

            for lbl in labels:
                issues.extend(
                    _validate_observation_payload(
                        obs.get(lbl),
                        f"{pfx}.candidate_observations.{lbl}",
                    )
                )

    size = len(compact_json(evidence))
    if size > int(max_chars):
        issues.append(f"Evidence JSON too large ({size} chars > {int(max_chars)}).")
    return issues


def merge_evidence(prev, curr):
    """Merge two evidence dicts; newer signal with same name wins."""
    merged = {}
    for ev in (prev, curr):
        if not isinstance(ev, dict):
            continue
        for sig in ev.get("signals", []):
            if isinstance(sig, dict):
                n = str(sig.get("signal_name", "")).strip()
                if n:
                    merged[n] = sig
    return {"signals": list(merged.values())}


# ---------------------------------------------------------------------------
# Generate → Execute → Repair loop
# ---------------------------------------------------------------------------

def _compact_exec_context(result, issues):
    s = execution_summary(result)
    stdout = str(result.get("stdout") or "")
    s["stdout_tail"] = stdout[-1600:] if stdout else ""
    s["validation_issues"] = issues
    return s


async def _generate_execute_repair(
    *,
    bundle_id,
    stage_index,        # 0 = Stage 1, 1 = Stage 2
    case_view,
    source_manifest,
    initial_prompt,
    client,
    conf,
    generate_content_fn,
    workspace,
    output_file,
    labels,
    affordance_graph=None,
):
    max_tokens_key = "sem_code_max_output_tokens"
    default_tokens = 4000
    stage_label = f"sem stage{stage_index + 1}"
    require_rpath = stage_index > 0

    raw = await _call_stage(
        generate_content_fn, client, conf, initial_prompt,
        max_tokens_key, default_tokens, f"{stage_label} code generation",
    )
    code = extract_python_code(raw)
    script_name = f"sem_bundle{bundle_id}_stage{stage_index + 1}.py"
    result = await asyncio.to_thread(
        execute_generated_code, code, conf, workspace, output_file, script_name, CONFIG_PREFIX,
    )

    src_names = [s["name"] for s in source_manifest.get("sources", [])]
    max_chars = int(conf.get("sem_max_evidence_chars", 30000))

    def _issues(r):
        if execution_needs_repair(r):
            return []
        return validate_evidence(r.get("evidence_json"), labels, src_names, max_chars, require_rpath)

    issues = _issues(result)
    repairs = []
    max_repairs = max(0, int(conf.get("sem_code_max_repair_attempts", 2)))

    while (execution_needs_repair(result) or issues) and len(repairs) < max_repairs:
        ctx = _compact_exec_context(result, issues)
        prompt = repair_prompt(
            case_view, source_manifest, code, ctx, output_file,
            affordance_graph=affordance_graph,
            require_relation_path=require_rpath,
        )
        repair_raw = await _call_stage(
            generate_content_fn, client, conf, prompt,
            max_tokens_key, default_tokens,
            f"{stage_label} repair attempt {len(repairs) + 1}",
        )
        repairs.append({"prompt": prompt, "raw_response": repair_raw})
        code = extract_python_code(repair_raw)
        result = await asyncio.to_thread(
            execute_generated_code, code, conf, workspace, output_file, script_name, CONFIG_PREFIX,
        )
        issues = _issues(result)

    accepted = (
        result.get("evidence_json")
        if not execution_needs_repair(result) and not issues
        else None
    )
    return {
        "prompt": initial_prompt,
        "raw_response": raw,
        "repairs": repairs,
        "generated_code": code,
        "execution_result": result,
        "execution_summary": _compact_exec_context(result, issues),
        "validation_issues": issues,
        "accepted_evidence": accepted,
    }


# ---------------------------------------------------------------------------
# Decision helpers
# ---------------------------------------------------------------------------

def _item_profile(item_id, item_info, dataset):
    key = str(int(item_id))
    raw = item_info.get(key, {})
    info = raw if isinstance(raw, dict) else {}
    name = str(dataset or "").lower()
    if "spotify" in name:
        parts = [info.get("track_name"), info.get("artist_name"), info.get("album_name")]
        text = " - ".join(" ".join(str(p).split()) for p in parts if p)
    else:
        text = " ".join(str(info.get("title", "")).split())
    if not text:
        text = f"Item {key}"
    excluded = {"id", "pic", "picture", "image", "image_url", "url",
                "title", "track_name", "artist_name", "album_name"}
    metadata = {}
    for k, v in info.items():
        if str(k).lower() in excluded:
            continue
        if isinstance(v, (str, bool, int, float)) or v is None:
            metadata[str(k)] = str(v)[:800]
        elif isinstance(v, list):
            metadata[str(k)] = [str(e)[:200] for e in v[:10]]
    return {"item_id": int(item_id), "text": text[:1200], "metadata": metadata}


def build_decision_case(sample, conf):
    info_path = os.path.join(conf["data_path"], conf["dataset"], "item_info.json")
    with open(info_path, "r", encoding="utf-8") as f:
        item_info = json.load(f)
    labels = [chr(ord("A") + i) for i in range(len(sample["candidate_indices"]))]
    return {
        "case_id": f"bundle_{sample['bundle_id']}",
        "dataset": conf["dataset"],
        "bundle_id": int(sample["bundle_id"]),
        "partial_items": [
            _item_profile(iid, item_info, conf["dataset"])
            for iid in sample["input_indices"]
        ],
        "candidates": [
            {"label": lbl, **_item_profile(iid, item_info, conf["dataset"])}
            for lbl, iid in zip(labels, sample["candidate_indices"])
        ],
    }


def _parse_prediction(raw_text, parser, labels):
    parsed = parse_json_from_text(raw_text)
    if isinstance(parsed, dict):
        pred = parser(str(parsed.get("prediction", "")))
        if pred in labels:
            return pred, {"prediction": pred}
    match = re.search(r'"prediction"\s*:\s*"?([A-Z])', str(raw_text or ""), flags=re.IGNORECASE)
    pred = parser(match.group(1) if match else raw_text)
    if pred in labels:
        return pred, {"prediction": pred}
    return pred, parsed


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------

async def run_sem_agent(
    sample,
    conf,
    clients,
    generate_content_fn,
    prediction_parser,
    debug_callback=None,
    is_first_sample=False,
):
    """Run the full 2-stage semantic evidence pipeline."""
    case_view = build_case_view(sample, conf["dataset"])
    labels = candidate_labels(case_view)
    workspace = prepare_workspace(conf, config_prefix=CONFIG_PREFIX)
    source_manifest = build_source_manifest(
        workspace,
        str(conf.get("sem_current_bundle_train_context_policy", "allow")),
    )
    affordance_graph = build_evidence_affordance_graph(source_manifest, conf["dataset"])
    max_chars = int(conf.get("sem_max_evidence_chars", 30000))
    decision_case = build_decision_case(sample, conf)

    # ------------------------------------------------------------------
    # Stage 1: Item Evidence Expansion
    # ------------------------------------------------------------------
    s1_output_file = f"output/sem_evidence_bundle{sample['bundle_id']}_stage1.json"
    s1_prompt = stage1_ecosystem_prompt(
        case_view,
        source_manifest,
        affordance_graph,
        s1_output_file,
        max_chars,
        semantic_case=decision_case,
    )
    if is_first_sample and debug_callback:
        debug_callback("Sem Agent Stage 1 Item Evidence Expansion Prompt", s1_prompt)

    s1_result = await _generate_execute_repair(
        bundle_id=sample["bundle_id"],
        stage_index=0,
        case_view=case_view,
        source_manifest=source_manifest,
        initial_prompt=s1_prompt,
        client=clients["code"],
        conf=conf,
        generate_content_fn=generate_content_fn,
        workspace=workspace,
        output_file=s1_output_file,
        labels=labels,
        affordance_graph=affordance_graph,
    )

    stage1_evidence = s1_result["accepted_evidence"] or {"signals": []}

    # ------------------------------------------------------------------
    # Stage 2: Bundle Context & Candidate Fit
    # ------------------------------------------------------------------
    s2_output_file = f"output/sem_evidence_bundle{sample['bundle_id']}_stage2.json"
    s2_prompt = stage2_gap_prompt(
        case_view, source_manifest, affordance_graph, s2_output_file, max_chars,
        stage1_evidence=stage1_evidence,
        semantic_case=decision_case,
    )
    if is_first_sample and debug_callback:
        debug_callback("Sem Agent Stage 2 Bundle Context & Candidate Fit Prompt", s2_prompt)

    s2_raw = await _call_stage(
        generate_content_fn,
        clients["code"],
        conf,
        s2_prompt,
        "sem_code_max_output_tokens",
        4000,
        "sem stage2 reasoning",
    )
    
    stage2_evidence = parse_json_from_text(s2_raw) or {"signals": []}
    
    s2_result = {
        "raw_response": s2_raw,
        "generated_code": "",
        "repairs": [],
        "execution_summary": {"msg": "pure reasoning, skipped execution"},
        "validation_issues": [],
        "accepted_evidence": stage2_evidence if "signals" in stage2_evidence else None,
    }

    # Merge Stage 1 + Stage 2 evidence; Stage 2 wins on name conflicts
    final_evidence = merge_evidence(stage1_evidence, stage2_evidence)

    # ------------------------------------------------------------------
    # Decision
    # ------------------------------------------------------------------
    d_prompt = decision_prompt(decision_case, final_evidence)
    if is_first_sample and debug_callback:
        debug_callback("Sem Agent Decision Prompt", d_prompt)

    d_raw = await _call_stage(
        generate_content_fn,
        clients["prediction"],
        conf,
        d_prompt,
        "sem_prediction_max_output_tokens",
        200,
        "sem final decision",
    )
    prediction, decision_json = _parse_prediction(d_raw, prediction_parser, labels)

    # ------------------------------------------------------------------
    # Build output row
    # ------------------------------------------------------------------
    row = {
        "sem_workspace_dir": workspace["workspace_dir"],
        "sem_workspace_files": compact_json(workspace["files"]),
        "sem_source_manifest": compact_json(source_manifest),
        "sem_affordance_graph": compact_json(affordance_graph),
        "sem_case_view": compact_json(case_view),
        "sem_decision_case": compact_json(decision_case),
        # Stage 1
        "sem_s1_prompt": s1_prompt,
        "sem_s1_raw_response": s1_result["raw_response"],
        "sem_s1_generated_code": s1_result["generated_code"],
        "sem_s1_repairs": compact_json(s1_result["repairs"]),
        "sem_s1_execution_summary": compact_json(s1_result["execution_summary"]),
        "sem_s1_validation_issues": compact_json(s1_result["validation_issues"]),
        "sem_s1_accepted": s1_result["accepted_evidence"] is not None,
        "sem_s1_evidence_json": compact_json(stage1_evidence),
        # Stage 2
        "sem_s2_prompt": s2_prompt,
        "sem_s2_raw_response": s2_result["raw_response"],
        "sem_s2_generated_code": s2_result["generated_code"],
        "sem_s2_repairs": compact_json(s2_result["repairs"]),
        "sem_s2_execution_summary": compact_json(s2_result["execution_summary"]),
        "sem_s2_validation_issues": compact_json(s2_result["validation_issues"]),
        "sem_s2_accepted": s2_result["accepted_evidence"] is not None,
        "sem_s2_evidence_json": compact_json(stage2_evidence),
        # Final
        "sem_final_evidence_json": compact_json(final_evidence),
        "sem_decision_prompt": d_prompt,
        "sem_decision_raw_response": d_raw,
        "sem_decision_json": compact_json(decision_json),
    }
    return row, prediction, d_raw

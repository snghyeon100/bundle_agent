"""Two-step code-generation bundle-completion pipeline.

Pipeline:
  Code generation/execution -> Prediction
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
)
from .prompts import code_generation_prompt, decision_prompt
from .workspace import (
    build_source_manifest,
    execute_generated_code,
    execution_failed,
    prepare_workspace,
)


CONFIG_PREFIX = "code"


async def _call_stage(generate_content_fn, client, conf, prompt, max_tokens_key, default_tokens, step_name):
    model = client.get("model") if isinstance(client, dict) else conf["model"]
    return await generate_content_fn(
        client,
        model,
        prompt,
        conf,
        conf.get(max_tokens_key, default_tokens),
        step_name,
    )


def _item_profile(item_id, item_info, dataset):
    key = str(int(item_id))
    raw = item_info.get(key, {})
    info = raw if isinstance(raw, dict) else {}
    name = str(dataset or "").lower()
    if "spotify" in name:
        parts = [info.get("track_name"), info.get("artist_name"), info.get("album_name")]
        text = " - ".join(" ".join(str(part).split()) for part in parts if part)
    else:
        text = " ".join(str(info.get("title", "")).split())
    if not text:
        text = f"Item {key}"
    excluded = {
        "id",
        "pic",
        "pic_url",
        "picture",
        "image",
        "image_url",
        "url",
        "title",
        "track_name",
        "artist_name",
        "album_name",
    }
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
    with open(info_path, "r", encoding="utf-8") as handle:
        item_info = json.load(handle)
    labels = [chr(ord("A") + i) for i in range(len(sample["candidate_indices"]))]
    return {
        "case_id": f"bundle_{sample['bundle_id']}",
        "dataset": conf["dataset"],
        "bundle_id": int(sample["bundle_id"]),
        "partial_items": [
            _item_profile(item_id, item_info, conf["dataset"])
            for item_id in sample["input_indices"]
        ],
        "candidates": [
            {"label": label, **_item_profile(item_id, item_info, conf["dataset"])}
            for label, item_id in zip(labels, sample["candidate_indices"])
        ],
    }


def _compact_exec_context(result):
    summary = execution_summary(result)
    stdout = str(result.get("stdout") or "")
    summary["stdout_tail"] = stdout[-1600:] if stdout else ""
    return summary


def validate_adaptive_item_evidence(evidence, case_view):
    """Return deterministic schema/identity issues for Stage 1 evidence."""
    if not isinstance(evidence, dict):
        return ["evidence must be a JSON object"]

    issues = []
    allowed_fields = {
        "schema_version",
        "strategy",
        "partial_evidence",
        "candidate_evidence",
    }
    unexpected = sorted(set(evidence) - allowed_fields)
    missing = sorted(allowed_fields - set(evidence))
    if unexpected:
        issues.append("unexpected top-level fields: " + ", ".join(unexpected))
    if missing:
        issues.append("missing top-level fields: " + ", ".join(missing))
    if evidence.get("schema_version") != "adaptive_item_evidence_v1":
        issues.append("schema_version must be adaptive_item_evidence_v1")

    strategy = evidence.get("strategy")
    if not isinstance(strategy, dict):
        issues.append("strategy must be an object")
    else:
        if set(strategy) != {"name", "description"}:
            issues.append("strategy must contain exactly name and description")
        for key in ("name", "description"):
            if not isinstance(strategy.get(key), str) or not strategy.get(key, "").strip():
                issues.append(f"strategy.{key} must be non-empty")

    expected_partials = {
        f"partial_{int(item_id)}": int(item_id)
        for item_id in case_view.get("partial_item_ids", [])
    }
    partial_payloads = evidence.get("partial_evidence")
    if not isinstance(partial_payloads, dict):
        issues.append("partial_evidence must be an object")
        partial_payloads = {}
    actual_partial_keys = set(partial_payloads)
    expected_partial_keys = set(expected_partials)
    if actual_partial_keys != expected_partial_keys:
        missing_keys = sorted(expected_partial_keys - actual_partial_keys)
        extra_keys = sorted(actual_partial_keys - expected_partial_keys)
        if missing_keys:
            issues.append("missing partial evidence keys: " + ", ".join(missing_keys))
        if extra_keys:
            issues.append("unexpected partial evidence keys: " + ", ".join(extra_keys))

    for key, item_id in expected_partials.items():
        payload = partial_payloads.get(key)
        if not isinstance(payload, dict):
            if key in partial_payloads:
                issues.append(f"partial {key} evidence must be an object")
            continue
        if set(payload) != {"item_id", "evidence"}:
            issues.append(f"partial {key} must contain exactly item_id and evidence")
        try:
            actual_item_id = int(payload.get("item_id"))
        except (TypeError, ValueError):
            actual_item_id = None
        if actual_item_id != item_id:
            issues.append(f"partial {key} item ID mismatch")
        lines = payload.get("evidence")
        if not isinstance(lines, list) or not lines or not all(
            isinstance(line, str) and line.strip() for line in lines
        ):
            issues.append(f"partial {key} evidence must be a non-empty string list")

    expected_candidates = {
        str(candidate.get("label")): int(candidate.get("item_id"))
        for candidate in case_view.get("candidates", [])
    }
    candidate_payloads = evidence.get("candidate_evidence")
    if not isinstance(candidate_payloads, dict):
        issues.append("candidate_evidence must be an object")
        candidate_payloads = {}
    actual_labels = set(candidate_payloads)
    expected_labels = set(expected_candidates)
    if actual_labels != expected_labels:
        missing_labels = sorted(expected_labels - actual_labels)
        extra_labels = sorted(actual_labels - expected_labels)
        if missing_labels:
            issues.append("missing candidate labels: " + ", ".join(missing_labels))
        if extra_labels:
            issues.append("unexpected candidate labels: " + ", ".join(extra_labels))

    for label, candidate_id in expected_candidates.items():
        payload = candidate_payloads.get(label)
        if not isinstance(payload, dict):
            if label in candidate_payloads:
                issues.append(f"candidate {label} evidence must be an object")
            continue
        if set(payload) != {"item_id", "evidence"}:
            issues.append(f"candidate {label} must contain exactly item_id and evidence")
        try:
            actual_candidate_id = int(payload.get("item_id"))
        except (TypeError, ValueError):
            actual_candidate_id = None
        if actual_candidate_id != candidate_id:
            issues.append(f"candidate {label} item ID mismatch")
        lines = payload.get("evidence")
        if not isinstance(lines, list) or not lines or not all(
            isinstance(line, str) and line.strip() for line in lines
        ):
            issues.append(f"candidate {label} evidence must be a non-empty string list")
    return issues


async def generate_code_evidence_once(
    *,
    bundle_id,
    case_view,
    source_manifest,
    initial_prompt,
    client,
    conf,
    generate_content_fn,
    workspace,
    output_file,
    semantic_case,
):
    raw = await _call_stage(
        generate_content_fn,
        client,
        conf,
        initial_prompt,
        "code_generation_max_output_tokens",
        int(conf.get("code_max_output_tokens", conf.get("sem_stage1_max_output_tokens", 16000))),
        "code evidence generation",
    )
    code = extract_python_code(raw)
    script_name = f"code_bundle{bundle_id}_evidence.py"
    result = await asyncio.to_thread(
        execute_generated_code,
        code,
        conf,
        workspace,
        output_file,
        script_name,
        CONFIG_PREFIX,
    )
    accepted = (
        result.get("evidence_json")
        if not execution_failed(result) and isinstance(result.get("evidence_json"), dict)
        else None
    )
    validation_issues = []
    if accepted is None:
        validation_issues.append("execution failed or evidence JSON was missing")
    else:
        validation_issues = validate_adaptive_item_evidence(accepted, case_view)
    if validation_issues:
        accepted = None
    summary = _compact_exec_context(result)
    summary["validation_issues"] = validation_issues
    if accepted is None:
        print(f"  [Bundle {bundle_id}] Code evidence generation FAILED.")
        print(f"  [Bundle {bundle_id}] Validation issues: {' | '.join(validation_issues)}")
        print(f"  [Bundle {bundle_id}] Execution summary: {compact_json(summary)}")
    return {
        "prompt": initial_prompt,
        "raw_response": raw,
        "generated_code": code,
        "execution_result": result,
        "execution_summary": summary,
        "validation_issues": validation_issues,
        "accepted_evidence": accepted,
    }


def build_code_generation_inputs(sample, conf):
    case_view = build_case_view(sample, conf["dataset"])
    workspace = prepare_workspace(conf, config_prefix=CONFIG_PREFIX)
    source_manifest = build_source_manifest(
        workspace,
        str(conf.get("code_current_bundle_train_context_policy", conf.get("sem_current_bundle_train_context_policy", "allow"))),
    )
    decision_case = build_decision_case(sample, conf)
    evidence_output_file = f"output/adaptive_item_evidence_bundle{sample['bundle_id']}.json"
    prompt = code_generation_prompt(
        case_view,
        source_manifest,
        evidence_output_file,
        semantic_case=decision_case,
    )
    return {
        "case_view": case_view,
        "workspace": workspace,
        "source_manifest": source_manifest,
        "decision_case": decision_case,
        "evidence_output_file": evidence_output_file,
        "prompt": prompt,
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


async def run_code_agent(
    sample,
    conf,
    clients,
    generate_content_fn,
    prediction_parser,
    debug_callback=None,
    is_first_sample=False,
):
    inputs = build_code_generation_inputs(sample, conf)
    case_view = inputs["case_view"]
    labels = candidate_labels(case_view)
    workspace = inputs["workspace"]
    source_manifest = inputs["source_manifest"]
    decision_case = inputs["decision_case"]
    evidence_output_file = inputs["evidence_output_file"]
    c_prompt = inputs["prompt"]
    if is_first_sample and debug_callback:
        debug_callback("Code Evidence Generation Prompt", c_prompt)

    code_result = await generate_code_evidence_once(
        bundle_id=sample["bundle_id"],
        case_view=case_view,
        source_manifest=source_manifest,
        initial_prompt=c_prompt,
        client=clients["code_generation"],
        conf=conf,
        generate_content_fn=generate_content_fn,
        workspace=workspace,
        output_file=evidence_output_file,
        semantic_case=decision_case,
    )
    evidence = code_result["accepted_evidence"]

    row = {
        "code_workspace_dir": workspace["workspace_dir"],
        "code_workspace_files": compact_json(workspace["files"]),
        "code_source_manifest": compact_json(source_manifest),
        "code_case_view": compact_json(case_view),
        "code_decision_case": compact_json(decision_case),
        "code_generation_prompt": c_prompt,
        "code_generation_raw_response": code_result["raw_response"],
        "code_generated_code": code_result["generated_code"],
        "code_execution_summary": compact_json(code_result["execution_summary"]),
        "code_evidence_validation_issues": compact_json(code_result["validation_issues"]),
        "code_evidence_accepted": evidence is not None,
        "code_evidence_json": compact_json(evidence) if evidence is not None else "",
        "code_stage1_status": "accepted" if evidence is not None else "failed",
    }
    if evidence is None:
        failure = "ERR_CODE: Stage 1 did not produce accepted adaptive item evidence"
        print(f"  [Bundle {sample['bundle_id']}] Stage 2 skipped: {failure}")
        return row, "ERR_CODE", failure

    print(f"  [Bundle {sample['bundle_id']}] Adaptive item evidence generation completed.")

    p_prompt = decision_prompt(decision_case, evidence)
    if is_first_sample and debug_callback:
        debug_callback("Code Prediction Prompt", p_prompt)
    p_raw = await _call_stage(
        generate_content_fn,
        clients["prediction"],
        conf,
        p_prompt,
        "code_prediction_max_output_tokens",
        int(conf.get("sem_prediction_max_output_tokens", 300)),
        "code final prediction",
    )
    prediction, prediction_json = _parse_prediction(p_raw, prediction_parser, labels)

    row.update(
        {
            "code_prediction_prompt": p_prompt,
            "code_prediction_raw_response": p_raw,
            "code_prediction_json": compact_json(prediction_json),
        }
    )
    return row, prediction, p_raw

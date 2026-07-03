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
    if accepted is None:
        summary = _compact_exec_context(result)
        print(f"  [Bundle {bundle_id}] Code evidence generation FAILED.")
        print(f"  [Bundle {bundle_id}] Execution summary: {compact_json(summary)}")
    return {
        "prompt": initial_prompt,
        "raw_response": raw,
        "generated_code": code,
        "execution_result": result,
        "execution_summary": _compact_exec_context(result),
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
    evidence_output_file = f"output/code_evidence_bundle{sample['bundle_id']}.json"
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


def _empty_evidence(case_view):
    partial_evidence = {
        f"partial_{int(item_id)}": {
            "item_id": int(item_id),
            "evidence": ["sparse note: code evidence generation did not produce accepted evidence"],
        }
        for item_id in case_view.get("partial_item_ids", [])
    }
    candidate_evidence = {
        str(candidate.get("label")): {
            "item_id": int(candidate.get("item_id")),
            "evidence": ["sparse note: code evidence generation did not produce accepted evidence"],
        }
        for candidate in case_view.get("candidates", [])
    }
    return {
        "schema_version": "code_evidence_v1",
        "strategies": [
            {
                "name": "fallback_sparse_note",
                "relation_signal": "none",
                "data_sources": [],
                "description": "Fallback evidence used when generated code was not accepted.",
            },
            {
                "name": "fallback_missing_relation_signal_1",
                "relation_signal": "none",
                "data_sources": [],
                "description": "Placeholder to preserve schema shape after failed code generation.",
            },
            {
                "name": "fallback_missing_relation_signal_2",
                "relation_signal": "none",
                "data_sources": [],
                "description": "Placeholder to preserve schema shape after failed code generation.",
            },
        ],
        "partial_evidence": partial_evidence,
        "candidate_evidence": candidate_evidence,
        "policy_trace": {
            "implemented_strategies": [],
            "skipped_strategies": ["generated code output was not accepted"],
            "notes": [],
        },
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
    evidence = code_result["accepted_evidence"] or _empty_evidence(case_view)
    print(f"  [Bundle {sample['bundle_id']}] Code evidence generation completed.")

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
        "code_evidence_accepted": code_result["accepted_evidence"] is not None,
        "code_evidence_json": compact_json(evidence),
        "code_prediction_prompt": p_prompt,
        "code_prediction_raw_response": p_raw,
        "code_prediction_json": compact_json(prediction_json),
    }
    return row, prediction, p_raw

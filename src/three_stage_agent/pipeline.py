import asyncio
import re

from agents.common import compact_json, extract_python_code, parse_json_from_text
from agents.workspace import (
    code_execution_needs_repair,
    execute_generated_python_code,
    prepare_agent_workspace,
)
from three_stage_agent.prompts import (
    generate_deep_observation_prompt,
    generate_deep_observation_repair_prompt,
    generate_exploratory_retrieval_prompt,
    generate_exploratory_retrieval_repair_prompt,
    generate_final_prediction_prompt,
    generate_synthesis_prompt,
)


async def call_stage(generate_content_fn, client, conf, prompt, max_tokens_key, default_tokens, step_name):
    return await generate_content_fn(
        client,
        conf["model"],
        prompt,
        conf,
        conf.get(max_tokens_key, default_tokens),
        step_name,
    )


def summarize_execution(execution_result):
    stderr = str(execution_result.get("stderr") or "")
    return {
        "returncode": execution_result.get("returncode"),
        "timed_out": bool(execution_result.get("timed_out")),
        "guard_blocked": bool(execution_result.get("guard_blocked")),
        "guard_violations": execution_result.get("guard_violations", []),
        "evidence_json_present": execution_result.get("evidence_json") is not None,
        "evidence_output_file": execution_result.get("evidence_output_file", ""),
        "stderr_tail": stderr[-1200:] if stderr else "",
    }


def parse_prediction_from_jsonish_text(text, prediction_parser):
    final_json = parse_json_from_text(text)
    if isinstance(final_json, dict) and final_json.get("prediction"):
        return prediction_parser(str(final_json.get("prediction", ""))), final_json
    match = re.search(
        r'["\']prediction["\']\s*:\s*["\']?([A-Z])["\']?',
        str(text or ""),
        flags=re.IGNORECASE,
    )
    if match:
        return prediction_parser(match.group(1)), final_json
    return prediction_parser(text), final_json


def synthesis_view_for_prediction(synthesis_json):
    allowed_fields = [
        "evidence_quality",
        "bundle_interpretation",
        "missing_role_hypothesis",
        "cross_candidate_patterns",
        "view_reliability",
        "candidate_synthesis",
        "conflicts",
        "downweighted_evidence",
        "limitations",
        "sources_used",
    ]
    if not isinstance(synthesis_json, dict):
        return {
            "evidence_quality": "none",
            "limitations": ["The synthesis stage did not return valid JSON."],
        }
    return {field: synthesis_json[field] for field in allowed_fields if field in synthesis_json}


async def run_exploratory_retrieval_with_repairs(
    sample,
    conf,
    code_client,
    generate_content_fn,
    retrieval_prompt,
    retrieval_raw_text,
    workspace,
    evidence_output_file,
):
    generated_code = extract_python_code(retrieval_raw_text)
    script_name = f"three_stage_retrieval_bundle{sample['bundle_id']}.py"
    execution_result = await asyncio.to_thread(
        execute_generated_python_code,
        generated_code,
        conf,
        workspace,
        evidence_output_file,
        script_name,
    )

    repair_attempts_used = 0
    repair_raw_responses = []
    max_repair_attempts = int(
        conf.get(
            "three_stage_code_max_repair_attempts",
            conf.get("agent_code_max_repair_attempts", 1),
        )
    )

    while code_execution_needs_repair(execution_result) and repair_attempts_used < max_repair_attempts:
        repair_attempts_used += 1
        repair_prompt = generate_exploratory_retrieval_repair_prompt(
            sample,
            workspace,
            evidence_output_file,
            generated_code,
            summarize_execution(execution_result),
            conf,
        )
        repair_raw_text = await call_stage(
            generate_content_fn,
            code_client,
            conf,
            repair_prompt,
            "three_stage_code_max_output_tokens",
            3600,
            f"sample three-stage exploratory retrieval repair {repair_attempts_used}",
        )
        repair_raw_responses.append(repair_raw_text)
        generated_code = extract_python_code(repair_raw_text)
        execution_result = await asyncio.to_thread(
            execute_generated_python_code,
            generated_code,
            conf,
            workspace,
            evidence_output_file,
            script_name,
        )

    return {
        "retrieval_prompt": retrieval_prompt,
        "retrieval_raw_response": retrieval_raw_text,
        "retrieval_repair_raw_responses": repair_raw_responses,
        "retrieval_repair_attempts_used": repair_attempts_used,
        "generated_code": generated_code,
        "execution_result": execution_result,
    }


async def run_deep_observation_with_repairs(
    sample,
    conf,
    code_client,
    generate_content_fn,
    deep_prompt,
    deep_raw_text,
    workspace,
    surface_evidence_json,
    surface_execution_summary,
    evidence_output_file,
):
    generated_code = extract_python_code(deep_raw_text)
    script_name = f"three_stage_deep_observation_bundle{sample['bundle_id']}.py"
    execution_result = await asyncio.to_thread(
        execute_generated_python_code,
        generated_code,
        conf,
        workspace,
        evidence_output_file,
        script_name,
    )

    repair_attempts_used = 0
    repair_raw_responses = []
    max_repair_attempts = int(
        conf.get(
            "three_stage_code_max_repair_attempts",
            conf.get("agent_code_max_repair_attempts", 1),
        )
    )

    while code_execution_needs_repair(execution_result) and repair_attempts_used < max_repair_attempts:
        repair_attempts_used += 1
        repair_prompt = generate_deep_observation_repair_prompt(
            sample,
            workspace,
            surface_evidence_json,
            surface_execution_summary,
            evidence_output_file,
            generated_code,
            summarize_execution(execution_result),
            conf,
        )
        repair_raw_text = await call_stage(
            generate_content_fn,
            code_client,
            conf,
            repair_prompt,
            "three_stage_deep_code_max_output_tokens",
            3600,
            f"sample three-stage deep observation repair {repair_attempts_used}",
        )
        repair_raw_responses.append(repair_raw_text)
        generated_code = extract_python_code(repair_raw_text)
        execution_result = await asyncio.to_thread(
            execute_generated_python_code,
            generated_code,
            conf,
            workspace,
            evidence_output_file,
            script_name,
        )

    return {
        "deep_prompt": deep_prompt,
        "deep_raw_response": deep_raw_text,
        "deep_repair_raw_responses": repair_raw_responses,
        "deep_repair_attempts_used": repair_attempts_used,
        "deep_generated_code": generated_code,
        "execution_result": execution_result,
    }


async def run_three_stage_agent(
    sample,
    conf,
    clients,
    generate_content_fn,
    prediction_parser,
    debug_callbacks=None,
    is_first_sample=False,
):
    debug_callbacks = debug_callbacks or {}
    sample = dict(sample)
    sample["dataset"] = conf.get("dataset", "")
    workspace = prepare_agent_workspace(conf)
    evidence_output_file = f"output/three_stage_surface_evidence_bundle{sample['bundle_id']}.json"
    deep_evidence_output_file = f"output/three_stage_deep_evidence_bundle{sample['bundle_id']}.json"

    retrieval_prompt = generate_exploratory_retrieval_prompt(
        sample, workspace, evidence_output_file, conf
    )
    if is_first_sample and debug_callbacks.get("prompt"):
        debug_callbacks["prompt"]("Three-Stage Exploratory Retrieval Prompt", retrieval_prompt)

    retrieval_raw_text = await call_stage(
        generate_content_fn,
        clients["code"],
        conf,
        retrieval_prompt,
        "three_stage_code_max_output_tokens",
        3600,
        "sample three-stage exploratory retrieval",
    )
    retrieval_result = await run_exploratory_retrieval_with_repairs(
        sample,
        conf,
        clients["code"],
        generate_content_fn,
        retrieval_prompt,
        retrieval_raw_text,
        workspace,
        evidence_output_file,
    )
    generated_code = retrieval_result["generated_code"]
    execution_result = retrieval_result["execution_result"]
    evidence_json = execution_result.get("evidence_json")
    execution_summary = summarize_execution(execution_result)

    deep_prompt = generate_deep_observation_prompt(
        sample,
        workspace,
        evidence_json,
        execution_summary,
        deep_evidence_output_file,
        conf,
    )
    if is_first_sample and debug_callbacks.get("prompt"):
        debug_callbacks["prompt"]("Three-Stage Deep Observation Prompt", deep_prompt)
    deep_raw_text = await call_stage(
        generate_content_fn,
        clients.get("deep_code", clients["code"]),
        conf,
        deep_prompt,
        "three_stage_deep_code_max_output_tokens",
        3600,
        "sample three-stage deep observation",
    )
    deep_result = await run_deep_observation_with_repairs(
        sample,
        conf,
        clients.get("deep_code", clients["code"]),
        generate_content_fn,
        deep_prompt,
        deep_raw_text,
        workspace,
        evidence_json,
        execution_summary,
        deep_evidence_output_file,
    )
    deep_execution_result = deep_result["execution_result"]
    deep_evidence_json = deep_execution_result.get("evidence_json")
    deep_execution_summary = summarize_execution(deep_execution_result)

    combined_evidence_json = {
        "surface_stage": {
            "execution_summary": execution_summary,
            "evidence_json": evidence_json,
        },
        "deep_stage": {
            "execution_summary": deep_execution_summary,
            "evidence_json": deep_evidence_json,
        },
    }
    combined_execution_summary = {
        "surface": execution_summary,
        "deep": deep_execution_summary,
    }

    synthesis_prompt = generate_synthesis_prompt(
        sample,
        combined_evidence_json,
        combined_execution_summary,
    )
    if is_first_sample and debug_callbacks.get("prompt"):
        debug_callbacks["prompt"]("Three-Stage Evidence Synthesis Prompt", synthesis_prompt)
    synthesis_raw_text = await call_stage(
        generate_content_fn,
        clients["synthesis"],
        conf,
        synthesis_prompt,
        "three_stage_synthesis_max_output_tokens",
        1800,
        "sample three-stage evidence synthesis",
    )
    synthesis_json = parse_json_from_text(synthesis_raw_text)
    synthesis_for_prediction = synthesis_view_for_prediction(synthesis_json)

    prediction_prompt = generate_final_prediction_prompt(sample, synthesis_for_prediction)
    if is_first_sample and debug_callbacks.get("qa"):
        debug_callbacks["qa"](sample, prediction_prompt)
    prediction_raw_text = await call_stage(
        generate_content_fn,
        clients["prediction"],
        conf,
        prediction_prompt,
        "three_stage_prediction_max_output_tokens",
        900,
        "sample three-stage final prediction",
    )
    prediction, prediction_json = parse_prediction_from_jsonish_text(
        prediction_raw_text, prediction_parser
    )

    row_updates = {
        "three_stage_workspace_dir": workspace["workspace_dir"],
        "three_stage_data_dir": workspace["data_dir"],
        "three_stage_workspace_files": compact_json(workspace["files"]),
        "three_stage_retrieval_raw_response": retrieval_raw_text,
        "three_stage_retrieval_repair_raw_responses": compact_json(
            retrieval_result["retrieval_repair_raw_responses"]
        ),
        "three_stage_retrieval_repair_attempts_used": retrieval_result[
            "retrieval_repair_attempts_used"
        ],
        "three_stage_generated_code": generated_code,
        "three_stage_execution_summary": compact_json(execution_summary),
        "three_stage_surface_evidence_json": compact_json(evidence_json),
        "three_stage_deep_prompt": deep_prompt,
        "three_stage_deep_raw_response": deep_raw_text,
        "three_stage_deep_repair_raw_responses": compact_json(
            deep_result["deep_repair_raw_responses"]
        ),
        "three_stage_deep_repair_attempts_used": deep_result[
            "deep_repair_attempts_used"
        ],
        "three_stage_deep_generated_code": deep_result["deep_generated_code"],
        "three_stage_deep_execution_summary": compact_json(deep_execution_summary),
        "three_stage_deep_evidence_json": compact_json(deep_evidence_json),
        "three_stage_evidence_json": compact_json(combined_evidence_json),
        "three_stage_synthesis_raw_response": synthesis_raw_text,
        "three_stage_synthesis_json": compact_json(synthesis_json),
        "three_stage_prediction_raw_response": prediction_raw_text,
        "three_stage_prediction_json": compact_json(prediction_json),
    }
    if isinstance(synthesis_json, dict):
        row_updates.update(
            {
                "three_stage_synthesis_evidence_quality": synthesis_json.get(
                    "evidence_quality", ""
                ),
                "three_stage_bundle_interpretation": synthesis_json.get(
                    "bundle_interpretation", ""
                ),
                "three_stage_missing_role_hypothesis": synthesis_json.get(
                    "missing_role_hypothesis", ""
                ),
                "three_stage_view_reliability": compact_json(
                    synthesis_json.get("view_reliability", {})
                ),
                "three_stage_synthesis_conflicts": compact_json(
                    synthesis_json.get("conflicts", [])
                ),
                "three_stage_synthesis_limitations": compact_json(
                    synthesis_json.get("limitations", [])
                ),
            }
        )
    if isinstance(prediction_json, dict):
        row_updates.update(
            {
                "three_stage_reasoning": prediction_json.get("reasoning", ""),
                "three_stage_confidence": prediction_json.get("confidence", ""),
                "three_stage_prediction_evidence_quality": prediction_json.get(
                    "evidence_quality_used", ""
                ),
                "three_stage_main_observations_used": compact_json(
                    prediction_json.get("main_observations_used", [])
                ),
                "three_stage_downweighted_or_ignored": compact_json(
                    prediction_json.get("downweighted_or_ignored", [])
                ),
            }
        )

    return row_updates, prediction, prediction_raw_text

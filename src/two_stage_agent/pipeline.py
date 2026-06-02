import asyncio
import re

from agents.common import compact_json, extract_python_code, parse_json_from_text
from agents.workspace import execute_generated_python_code, prepare_agent_workspace
from two_stage_agent.prompts import generate_code_retrieval_prompt, generate_prediction_prompt


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


async def run_two_stage_agent(
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
    evidence_output_file = f"output/two_stage_evidence_bundle{sample['bundle_id']}.json"

    code_prompt = generate_code_retrieval_prompt(sample, workspace, evidence_output_file, conf)
    if is_first_sample and debug_callbacks.get("prompt"):
        debug_callbacks["prompt"]("Two-Stage Code Retrieval Prompt", code_prompt)

    code_raw_text = await call_stage(
        generate_content_fn,
        clients["code"],
        conf,
        code_prompt,
        "agent_code_max_output_tokens",
        2400,
        "sample two-stage code retrieval",
    )
    generated_code = extract_python_code(code_raw_text)
    script_name = f"two_stage_retrieval_bundle{sample['bundle_id']}.py"
    execution_result = await asyncio.to_thread(
        execute_generated_python_code,
        generated_code,
        conf,
        workspace,
        evidence_output_file,
        script_name,
    )
    evidence_json = execution_result.get("evidence_json")

    prediction_prompt = generate_prediction_prompt(sample, evidence_json)
    if is_first_sample and debug_callbacks.get("qa"):
        debug_callbacks["qa"](sample, prediction_prompt)

    prediction_raw_text = await call_stage(
        generate_content_fn,
        clients["prediction"],
        conf,
        prediction_prompt,
        "agent_prediction_max_output_tokens",
        600,
        "sample two-stage final prediction",
    )
    prediction, prediction_json = parse_prediction_from_jsonish_text(prediction_raw_text, prediction_parser)

    row_updates = {
        "two_stage_workspace_dir": workspace["workspace_dir"],
        "two_stage_data_dir": workspace["data_dir"],
        "two_stage_workspace_files": compact_json(workspace["files"]),
        "two_stage_code_raw_response": code_raw_text,
        "two_stage_generated_code": generated_code,
        "two_stage_execution_summary": compact_json(summarize_execution(execution_result)),
        "two_stage_evidence_json": compact_json(evidence_json),
        "two_stage_prediction_raw_response": prediction_raw_text,
        "two_stage_prediction_json": compact_json(prediction_json),
    }
    if isinstance(prediction_json, dict):
        row_updates.update(
            {
                "two_stage_reasoning": prediction_json.get("reasoning", ""),
                "two_stage_confidence": prediction_json.get("confidence", ""),
            }
        )

    return row_updates, prediction, prediction_raw_text


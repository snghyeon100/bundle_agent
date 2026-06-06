import asyncio
import re

from agents.common import compact_json, extract_python_code, parse_json_from_text
from agents.workspace import execute_generated_python_code, prepare_agent_workspace
from three_stage_agent.prompts import (
    generate_exploratory_retrieval_prompt,
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
        "cross_candidate_patterns",
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
    evidence_output_file = f"output/three_stage_evidence_bundle{sample['bundle_id']}.json"

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
    evidence_json = execution_result.get("evidence_json")
    execution_summary = summarize_execution(execution_result)

    synthesis_prompt = generate_synthesis_prompt(sample, evidence_json, execution_summary)
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
        "three_stage_generated_code": generated_code,
        "three_stage_execution_summary": compact_json(execution_summary),
        "three_stage_evidence_json": compact_json(evidence_json),
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

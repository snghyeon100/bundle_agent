import asyncio

from agents.common import compact_json, extract_python_code, parse_json_from_text
from agents.planner import generate_planning_prompt
from agents.predictor import generate_prediction_prompt
from agents.retriever import generate_code_prompt, generate_code_repair_prompt
from agents.verifier import generate_verifier_prompt
from agents.workspace import code_execution_needs_repair, execute_generated_python_code, prepare_agent_workspace


def verifier_requests_replanning(verifier_json):
    if not isinstance(verifier_json, dict):
        return False
    return bool(verifier_json.get("needs_replanning", False)) and not bool(
        verifier_json.get("evidence_sufficient", False)
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


def summarize_execution_for_llm(execution_result):
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


def compact_round_for_llm(round_data):
    return {
        "round": round_data.get("round"),
        "plan": round_data.get("planning_json") or str(round_data.get("planning_raw_response", ""))[:2000],
        "execution": summarize_execution_for_llm(round_data.get("execution_result", {})),
        "evidence": round_data.get("evidence_json"),
        "verifier": round_data.get("verifier_json"),
    }


async def run_code_with_repairs(
    sample,
    conf,
    code_client,
    generate_content_fn,
    planning_raw_text,
    code_raw_text,
    workspace,
    evidence_output_file,
    round_number,
):
    generated_code = extract_python_code(code_raw_text)
    script_name = f"agent_retrieval_bundle{sample['bundle_id']}_round{round_number}.py"
    execution_result = await asyncio.to_thread(
        execute_generated_python_code, generated_code, conf, workspace, evidence_output_file, script_name
    )

    repair_attempts_used = 0
    repair_raw_responses = []
    max_repair_attempts = int(conf.get("agent_code_max_repair_attempts", 1))

    while code_execution_needs_repair(execution_result) and repair_attempts_used < max_repair_attempts:
        repair_attempts_used += 1
        repair_prompt = generate_code_repair_prompt(
            sample, planning_raw_text, generated_code, execution_result, workspace, evidence_output_file
        )
        repair_raw_text = await call_stage(
            generate_content_fn,
            code_client,
            conf,
            repair_prompt,
            "agent_code_max_output_tokens",
            2200,
            f"sample agent code repair round {round_number}.{repair_attempts_used}",
        )
        repair_raw_responses.append(repair_raw_text)
        generated_code = extract_python_code(repair_raw_text)
        execution_result = await asyncio.to_thread(
            execute_generated_python_code, generated_code, conf, workspace, evidence_output_file, script_name
        )

    return {
        "code_raw_response": code_raw_text,
        "code_repair_raw_responses": repair_raw_responses,
        "code_repair_attempts_used": repair_attempts_used,
        "generated_code": generated_code,
        "execution_result": execution_result,
    }


async def run_four_stage_agent(
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
    max_rounds = int(conf.get("agent_max_retrieval_rounds", 2))
    retrieval_rounds = []
    verifier_feedback = None

    for round_idx in range(max_rounds):
        round_number = round_idx + 1
        evidence_output_file = f"output/evidence_bundle{sample['bundle_id']}_round{round_number}.json"

        compact_previous_rounds = [compact_round_for_llm(round_data) for round_data in retrieval_rounds]
        planning_prompt = generate_planning_prompt(
            sample,
            workspace,
            evidence_output_file,
            previous_rounds=compact_previous_rounds,
            verifier_feedback=verifier_feedback,
        )
        if is_first_sample and debug_callbacks.get("prompt"):
            debug_callbacks["prompt"](f"Agent Planner Prompt Round {round_number}", planning_prompt)
        planning_raw_text = await call_stage(
            generate_content_fn,
            clients["planning"],
            conf,
            planning_prompt,
            "agent_planning_max_output_tokens",
            900,
            f"sample agent planning round {round_number}",
        )
        planning_json = parse_json_from_text(planning_raw_text)

        code_prompt = generate_code_prompt(sample, planning_raw_text, workspace, evidence_output_file, conf)
        if is_first_sample and debug_callbacks.get("prompt"):
            debug_callbacks["prompt"](f"Agent Retriever Code Prompt Round {round_number}", code_prompt)
        code_raw_text = await call_stage(
            generate_content_fn,
            clients["code"],
            conf,
            code_prompt,
            "agent_code_max_output_tokens",
            2400,
            f"sample agent code writing round {round_number}",
        )

        code_result = await run_code_with_repairs(
            sample,
            conf,
            clients["code"],
            generate_content_fn,
            planning_raw_text,
            code_raw_text,
            workspace,
            evidence_output_file,
            round_number,
        )

        execution_result = code_result["execution_result"]
        evidence_json = execution_result.get("evidence_json")
        remaining_rounds = max_rounds - round_number

        verifier_prompt = generate_verifier_prompt(
            sample,
            planning_json or planning_raw_text[:2000],
            evidence_json,
            summarize_execution_for_llm(execution_result),
            round_number,
            remaining_rounds,
        )
        if is_first_sample and debug_callbacks.get("prompt"):
            debug_callbacks["prompt"](f"Agent Verifier Prompt Round {round_number}", verifier_prompt)
        verifier_raw_text = await call_stage(
            generate_content_fn,
            clients["verifier"],
            conf,
            verifier_prompt,
            "agent_verifier_max_output_tokens",
            900,
            f"sample agent verifier round {round_number}",
        )
        verifier_json = parse_json_from_text(verifier_raw_text)

        round_record = {
            "round": round_number,
            "planning_raw_response": planning_raw_text,
            "planning_json": planning_json,
            "code_raw_response": code_result["code_raw_response"],
            "code_repair_raw_responses": code_result["code_repair_raw_responses"],
            "code_repair_attempts_used": code_result["code_repair_attempts_used"],
            "generated_code": code_result["generated_code"],
            "execution_result": execution_result,
            "evidence_json": evidence_json,
            "verifier_raw_response": verifier_raw_text,
            "verifier_json": verifier_json,
        }
        retrieval_rounds.append(round_record)

        if remaining_rounds > 0 and verifier_requests_replanning(verifier_json):
            verifier_feedback = verifier_json
            continue
        break

    prediction_prompt = generate_prediction_prompt(
        sample, [compact_round_for_llm(round_data) for round_data in retrieval_rounds]
    )
    if is_first_sample and debug_callbacks.get("qa"):
        debug_callbacks["qa"](sample, prediction_prompt)
    final_raw_text = await call_stage(
        generate_content_fn,
        clients["prediction"],
        conf,
        prediction_prompt,
        "agent_prediction_max_output_tokens",
        900,
        "sample agent final prediction",
    )
    final_json = parse_json_from_text(final_raw_text)
    if isinstance(final_json, dict) and final_json.get("prediction"):
        prediction = prediction_parser(str(final_json.get("prediction", "")))
    else:
        prediction = prediction_parser(final_raw_text)

    row_updates = {
        "agent_workspace_dir": workspace["workspace_dir"],
        "agent_workspace_files": compact_json(workspace["files"]),
        "agent_round_count": len(retrieval_rounds),
        "agent_retrieval_rounds_json": compact_json(retrieval_rounds),
        "agent_all_plans_json": compact_json([round_data.get("planning_json") for round_data in retrieval_rounds]),
        "agent_all_generated_codes_json": compact_json(
            [round_data.get("generated_code", "") for round_data in retrieval_rounds]
        ),
        "agent_all_evidence_json": compact_json([round_data.get("evidence_json") for round_data in retrieval_rounds]),
        "agent_all_verifier_json": compact_json([round_data.get("verifier_json") for round_data in retrieval_rounds]),
        "agent_prediction_raw_response": final_raw_text,
        "agent_prediction_json": compact_json(final_json),
    }
    if isinstance(final_json, dict):
        row_updates.update(
            {
                "agent_reasoning": final_json.get("reasoning", ""),
                "agent_confidence": final_json.get("confidence", ""),
                "agent_evidence_quality": final_json.get("evidence_quality", ""),
                "agent_main_sources_used_for_decision": compact_json(
                    final_json.get("main_sources_used_for_decision", [])
                ),
                "agent_candidate_tradeoff": compact_json(final_json.get("candidate_tradeoff", {})),
                "agent_downweighted_evidence": compact_json(final_json.get("downweighted_evidence", [])),
                "agent_decision_rule": final_json.get("decision_rule", ""),
            }
        )

    return row_updates, prediction, final_raw_text

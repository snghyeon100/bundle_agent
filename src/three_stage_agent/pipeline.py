import asyncio
import copy
import re

from agents.common import candidate_labels, compact_json, extract_python_code, parse_json_from_text
from agents.workspace import (
    code_execution_needs_repair,
    execute_generated_python_code,
    prepare_agent_workspace,
)
from three_stage_agent.prompts import (
    generate_deep_observation_planning_prompt,
    generate_deep_observation_prompt,
    generate_deep_observation_repair_prompt,
    generate_exploratory_retrieval_prompt,
    generate_exploratory_retrieval_repair_prompt,
    generate_final_prediction_prompt,
    generate_synthesis_repair_prompt,
    generate_synthesis_prompt,
)


async def call_stage(
    generate_content_fn,
    client,
    conf,
    prompt,
    max_tokens_key,
    default_tokens,
    step_name,
    model_key=None,
):
    model = str(conf.get(model_key, "")).strip() if model_key else ""
    return await generate_content_fn(
        client,
        model or conf["model"],
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


def deep_evidence_validation_issues(
    evidence_json,
    deep_planning_json,
    expected_candidate_labels=None,
):
    issues = []
    if not isinstance(evidence_json, dict):
        return ["Deep evidence is not a JSON object."]

    investigations = evidence_json.get("deep_investigations")
    fulfillment = evidence_json.get("plan_fulfillment")
    if not isinstance(investigations, list):
        issues.append("deep_investigations must be a list.")
        investigations = []
    if not isinstance(fulfillment, list):
        issues.append("plan_fulfillment must be a list.")
        fulfillment = []
    if not isinstance(evidence_json.get("warnings"), list):
        issues.append("warnings must be a list.")

    required_investigation_fields = [
        "investigation_id",
        "question",
        "why_relevant",
        "novelty_from_surface",
        "sources_used",
        "method_summary",
        "observations",
        "limitations",
    ]
    investigation_by_id = {}
    for index, investigation in enumerate(investigations):
        if not isinstance(investigation, dict):
            issues.append(f"deep_investigations[{index}] must be an object.")
            continue
        missing = [field for field in required_investigation_fields if field not in investigation]
        if missing:
            issues.append(
                f"deep_investigations[{index}] is missing fields: {', '.join(missing)}."
            )
        investigation_id = str(investigation.get("investigation_id", "")).strip()
        if investigation_id:
            investigation_by_id[investigation_id] = investigation
        observations = investigation.get("observations")
        if isinstance(observations, list):
            for observation_index, observation in enumerate(observations):
                if not isinstance(observation, dict):
                    issues.append(
                        f"deep_investigations[{index}].observations[{observation_index}] must be an object."
                    )
                    continue
                observation_missing = [
                    field
                    for field in ("source", "scope", "observation", "related_ids", "basis")
                    if field not in observation
                ]
                if observation_missing:
                    issues.append(
                        f"deep_investigations[{index}].observations[{observation_index}] is missing fields: "
                        f"{', '.join(observation_missing)}."
                    )

    fulfillment_by_id = {}
    for index, entry in enumerate(fulfillment):
        if not isinstance(entry, dict):
            issues.append(f"plan_fulfillment[{index}] must be an object.")
            continue
        missing = [field for field in ("investigation_id", "status", "details") if field not in entry]
        if missing:
            issues.append(f"plan_fulfillment[{index}] is missing fields: {', '.join(missing)}.")
        investigation_id = str(entry.get("investigation_id", "")).strip()
        status = str(entry.get("status", "")).strip().lower()
        if status not in {"completed", "partial", "failed"}:
            issues.append(f"plan_fulfillment[{index}].status must be completed, partial, or failed.")
        if investigation_id:
            fulfillment_by_id[investigation_id] = status

    expected_scopes = {
        f"candidate:{label}" for label in (expected_candidate_labels or [])
    }
    if expected_scopes:
        for investigation_id, investigation in investigation_by_id.items():
            if fulfillment_by_id.get(investigation_id) not in {"completed", "partial"}:
                continue
            observations = investigation.get("observations", [])
            observed_scopes = {
                str(observation.get("scope", "")).strip()
                for observation in observations
                if isinstance(observation, dict)
            }
            invalid_candidate_scopes = sorted(
                scope
                for scope in observed_scopes
                if scope.startswith("candidate:") and scope not in expected_scopes
            )
            if invalid_candidate_scopes:
                issues.append(
                    f"Investigation {investigation_id} has invalid candidate scopes: "
                    f"{', '.join(invalid_candidate_scopes)}."
                )
            missing_scopes = sorted(expected_scopes - observed_scopes)
            if missing_scopes:
                issues.append(
                    f"Investigation {investigation_id} is missing candidate observations for: "
                    f"{', '.join(missing_scopes)}."
                )

    portfolio = (
        deep_planning_json.get("investigation_portfolio", [])
        if isinstance(deep_planning_json, dict)
        else []
    )
    planned_ids = {
        str(entry.get("investigation_id", "")).strip()
        for entry in portfolio
        if isinstance(entry, dict) and str(entry.get("investigation_id", "")).strip()
    }
    for investigation_id in sorted(planned_ids):
        if investigation_id not in fulfillment_by_id:
            issues.append(f"Planned investigation {investigation_id} is missing from plan_fulfillment.")
            continue
        if fulfillment_by_id[investigation_id] in {"completed", "partial"} and investigation_id not in investigation_by_id:
            issues.append(
                f"Planned investigation {investigation_id} is marked {fulfillment_by_id[investigation_id]} "
                "but has no deep_investigations entry."
            )
    return issues


def normalize_deep_evidence_json(evidence_json):
    if not isinstance(evidence_json, dict):
        return evidence_json
    normalized = copy.deepcopy(evidence_json)
    investigations = normalized.get("deep_investigations")
    if not isinstance(investigations, list):
        return normalized
    for investigation in investigations:
        if not isinstance(investigation, dict):
            continue
        observations = investigation.get("observations")
        if not isinstance(observations, list):
            continue
        for observation in observations:
            if isinstance(observation, dict):
                observation.setdefault("examples", [])
    return normalized


def synthesis_validation_issues(synthesis_json, expected_candidate_labels):
    if not isinstance(synthesis_json, dict):
        return ["Synthesis response is not a complete JSON object."]
    issues = []
    required_fields = [
        "evidence_quality",
        "bundle_interpretation",
        "missing_role_hypothesis",
        "candidate_synthesis",
        "conflicts",
        "downweighted_evidence",
        "limitations",
        "sources_used",
    ]
    missing_fields = [field for field in required_fields if field not in synthesis_json]
    if missing_fields:
        issues.append(f"Synthesis is missing fields: {', '.join(missing_fields)}.")
    candidate_synthesis = synthesis_json.get("candidate_synthesis")
    if not isinstance(candidate_synthesis, dict):
        issues.append("candidate_synthesis must be an object.")
    else:
        expected = set(expected_candidate_labels)
        actual = set(candidate_synthesis)
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        if missing:
            issues.append(f"candidate_synthesis is missing labels: {', '.join(missing)}.")
        if extra:
            issues.append(f"candidate_synthesis has unexpected labels: {', '.join(extra)}.")
    return issues


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
    deep_planning_json,
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
    execution_result["evidence_json"] = normalize_deep_evidence_json(
        execution_result.get("evidence_json")
    )

    repair_attempts_used = 0
    repair_raw_responses = []
    max_repair_attempts = int(
        conf.get(
            "three_stage_code_max_repair_attempts",
            conf.get("agent_code_max_repair_attempts", 1),
        )
    )

    validation_issues = deep_evidence_validation_issues(
        execution_result.get("evidence_json"),
        deep_planning_json,
        candidate_labels(sample),
    )

    while (
        code_execution_needs_repair(execution_result) or validation_issues
    ) and repair_attempts_used < max_repair_attempts:
        repair_attempts_used += 1
        repair_execution_summary = summarize_execution(execution_result)
        repair_execution_summary["evidence_validation_issues"] = validation_issues
        repair_prompt = generate_deep_observation_repair_prompt(
            sample,
            workspace,
            surface_evidence_json,
            surface_execution_summary,
            deep_planning_json,
            evidence_output_file,
            generated_code,
            repair_execution_summary,
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
            model_key="three_stage_deep_code_model",
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
        execution_result["evidence_json"] = normalize_deep_evidence_json(
            execution_result.get("evidence_json")
        )
        validation_issues = deep_evidence_validation_issues(
            execution_result.get("evidence_json"),
            deep_planning_json,
            candidate_labels(sample),
        )

    return {
        "deep_prompt": deep_prompt,
        "deep_raw_response": deep_raw_text,
        "deep_repair_raw_responses": repair_raw_responses,
        "deep_repair_attempts_used": repair_attempts_used,
        "deep_generated_code": generated_code,
        "execution_result": execution_result,
        "deep_evidence_validation_issues": validation_issues,
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

    deep_planning_prompt = generate_deep_observation_planning_prompt(
        sample,
        workspace,
        evidence_json,
        execution_summary,
        conf,
    )
    if is_first_sample and debug_callbacks.get("prompt"):
        debug_callbacks["prompt"](
            "Three-Stage Deep Observation Planning Prompt", deep_planning_prompt
        )
    deep_planning_raw_text = await call_stage(
        generate_content_fn,
        clients.get("deep_planning", clients.get("deep_code", clients["code"])),
        conf,
        deep_planning_prompt,
        "three_stage_deep_planning_max_output_tokens",
        1800,
        "sample three-stage deep observation planning",
        model_key="three_stage_deep_planning_model",
    )
    deep_planning_json = parse_json_from_text(deep_planning_raw_text)
    if isinstance(deep_planning_json, dict):
        deep_planning_for_code = deep_planning_json
    else:
        deep_planning_for_code = {
            "planning_parse_error": "The planning response was not valid JSON.",
            "raw_planning_text": str(deep_planning_raw_text or "")[:8000],
        }

    deep_prompt = generate_deep_observation_prompt(
        sample,
        workspace,
        evidence_json,
        execution_summary,
        deep_planning_for_code,
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
        model_key="three_stage_deep_code_model",
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
        deep_planning_for_code,
        deep_evidence_output_file,
    )
    deep_execution_result = deep_result["execution_result"]
    deep_evidence_json = deep_execution_result.get("evidence_json")
    deep_execution_summary = summarize_execution(deep_execution_result)
    deep_evidence_for_synthesis = (
        deep_evidence_json
        if not deep_result["deep_evidence_validation_issues"]
        else None
    )

    combined_evidence_json = {
        "surface_stage": {
            "execution_summary": execution_summary,
            "evidence_json": evidence_json,
        },
        "deep_stage": {
            "planning_json": deep_planning_for_code,
            "execution_summary": deep_execution_summary,
            "evidence_validation_issues": deep_result["deep_evidence_validation_issues"],
            "evidence_json": deep_evidence_for_synthesis,
            "rejected_evidence_json": (
                deep_evidence_json if deep_evidence_for_synthesis is None else None
            ),
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
        3600,
        "sample three-stage evidence synthesis",
    )
    synthesis_json = parse_json_from_text(synthesis_raw_text)
    synthesis_final_raw_text = synthesis_raw_text
    synthesis_repair_raw_responses = []
    synthesis_repair_attempts_used = 0
    max_synthesis_repair_attempts = int(
        conf.get("three_stage_synthesis_max_repair_attempts", 1)
    )
    synthesis_issues = synthesis_validation_issues(
        synthesis_json, candidate_labels(sample)
    )
    while synthesis_issues and synthesis_repair_attempts_used < max_synthesis_repair_attempts:
        synthesis_repair_attempts_used += 1
        synthesis_repair_prompt = generate_synthesis_repair_prompt(
            sample,
            combined_evidence_json,
            combined_execution_summary,
            synthesis_final_raw_text,
            synthesis_issues,
        )
        repaired_synthesis_raw_text = await call_stage(
            generate_content_fn,
            clients["synthesis"],
            conf,
            synthesis_repair_prompt,
            "three_stage_synthesis_max_output_tokens",
            3600,
            f"sample three-stage evidence synthesis repair {synthesis_repair_attempts_used}",
        )
        synthesis_repair_raw_responses.append(repaired_synthesis_raw_text)
        synthesis_final_raw_text = repaired_synthesis_raw_text
        synthesis_json = parse_json_from_text(synthesis_final_raw_text)
        synthesis_issues = synthesis_validation_issues(
            synthesis_json, candidate_labels(sample)
        )
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
        "three_stage_deep_planning_prompt": deep_planning_prompt,
        "three_stage_deep_planning_raw_response": deep_planning_raw_text,
        "three_stage_deep_planning_json": compact_json(deep_planning_json),
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
        "three_stage_deep_evidence_validation_issues": compact_json(
            deep_result["deep_evidence_validation_issues"]
        ),
        "three_stage_deep_evidence_json": compact_json(deep_evidence_json),
        "three_stage_evidence_json": compact_json(combined_evidence_json),
        "three_stage_synthesis_raw_response": synthesis_raw_text,
        "three_stage_synthesis_repair_raw_responses": compact_json(
            synthesis_repair_raw_responses
        ),
        "three_stage_synthesis_repair_attempts_used": synthesis_repair_attempts_used,
        "three_stage_synthesis_final_raw_response": synthesis_final_raw_text,
        "three_stage_synthesis_validation_issues": compact_json(synthesis_issues),
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

import asyncio
import re

from .common import (
    build_case_view,
    candidate_labels,
    compact_json,
    execution_summary,
    extract_python_code,
    parse_json_from_text,
)
from .prompts import (
    broad_code_prompt,
    broad_planning_prompt,
    code_repair_prompt,
    decision_prompt,
    deep_code_prompt,
    deep_planning_prompt,
    diagnosis_prompt,
)
from .workspace import (
    build_source_manifest,
    execute_generated_code,
    execution_needs_repair,
    prepare_workspace,
)


async def _call_stage(
    generate_content_fn,
    client,
    conf,
    prompt,
    max_tokens_key,
    default_tokens,
    step_name,
):
    return await generate_content_fn(
        client,
        conf["model"],
        prompt,
        conf,
        conf.get(max_tokens_key, default_tokens),
        step_name,
    )


def _scope_coverage(observations, labels):
    observed = {
        str(observation.get("scope", "")).strip()
        for observation in observations
        if isinstance(observation, dict)
    }
    return [f"candidate:{label}" for label in labels if f"candidate:{label}" not in observed]


def validate_broad_evidence(evidence, labels):
    if not isinstance(evidence, dict):
        return ["Broad evidence must be a JSON object."]
    issues = []
    for field in ("case_profile", "source_attempts", "observations", "warnings"):
        if field not in evidence:
            issues.append(f"Broad evidence is missing {field}.")
    if not isinstance(evidence.get("case_profile"), dict):
        issues.append("case_profile must be an object.")
    if not isinstance(evidence.get("source_attempts"), list):
        issues.append("source_attempts must be a list.")
    observations = evidence.get("observations")
    if not isinstance(observations, list):
        issues.append("observations must be a list.")
        observations = []
    if not isinstance(evidence.get("warnings"), list):
        issues.append("warnings must be a list.")
    missing = _scope_coverage(observations, labels)
    if missing:
        issues.append(f"Broad evidence is missing candidate scopes: {', '.join(missing)}.")
    return issues


def validate_deep_evidence(evidence, plan, labels):
    if not isinstance(evidence, dict):
        return ["Deep evidence must be a JSON object."]
    issues = []
    investigations = evidence.get("investigations")
    fulfillment = evidence.get("plan_fulfillment")
    if not isinstance(investigations, list):
        issues.append("investigations must be a list.")
        investigations = []
    if not isinstance(fulfillment, list):
        issues.append("plan_fulfillment must be a list.")
        fulfillment = []
    if not isinstance(evidence.get("warnings"), list):
        issues.append("warnings must be a list.")

    status_by_id = {}
    for index, entry in enumerate(fulfillment):
        if not isinstance(entry, dict):
            issues.append(f"plan_fulfillment[{index}] must be an object.")
            continue
        investigation_id = str(entry.get("investigation_id", "")).strip()
        status = str(entry.get("status", "")).strip().lower()
        if not investigation_id:
            issues.append(f"plan_fulfillment[{index}] has no investigation_id.")
        if status not in {"completed", "partial", "failed"}:
            issues.append(f"plan_fulfillment[{index}] has invalid status.")
        status_by_id[investigation_id] = status

    evidence_by_id = {}
    for index, investigation in enumerate(investigations):
        if not isinstance(investigation, dict):
            issues.append(f"investigations[{index}] must be an object.")
            continue
        investigation_id = str(investigation.get("investigation_id", "")).strip()
        evidence_by_id[investigation_id] = investigation
        observations = investigation.get("observations")
        if not isinstance(observations, list):
            issues.append(f"investigation {investigation_id or index} observations must be a list.")
            observations = []
        if status_by_id.get(investigation_id) in {"completed", "partial"}:
            missing = _scope_coverage(observations, labels)
            if missing:
                issues.append(
                    f"Investigation {investigation_id} is missing candidate scopes: {', '.join(missing)}."
                )

    planned = plan.get("investigations", []) if isinstance(plan, dict) else []
    for entry in planned:
        if not isinstance(entry, dict):
            continue
        investigation_id = str(entry.get("investigation_id", "")).strip()
        if investigation_id and investigation_id not in status_by_id:
            issues.append(f"Planned investigation {investigation_id} is missing from plan_fulfillment.")
        if status_by_id.get(investigation_id) in {"completed", "partial"} and investigation_id not in evidence_by_id:
            issues.append(f"Planned investigation {investigation_id} has no evidence object.")
    return issues


def _diagnosis_fallback(raw_text):
    return {
        "status": "STOP_INCONCLUSIVE",
        "evidence_quality": "none",
        "reliable_observations": [],
        "observed_failures": ["Diagnosis response was not valid JSON."],
        "unresolved_questions": [],
        "evidence_gaps": [],
        "conflicts": [],
        "signals_to_downweight": [],
        "candidate_coverage": {},
        "stop_reason": str(raw_text or "")[:1200],
    }


async def _generate_execute_repair(
    *,
    stage,
    sample_bundle_id,
    case_view,
    source_manifest,
    specification,
    initial_prompt,
    client,
    conf,
    generate_content_fn,
    workspace,
    output_file,
    validator,
    token_key="psd_code_max_output_tokens",
):
    raw_response = await _call_stage(
        generate_content_fn,
        client,
        conf,
        initial_prompt,
        token_key,
        4000,
        f"{stage} code generation",
    )
    code = extract_python_code(raw_response)
    script_name = f"psd_{stage}_{sample_bundle_id}.py"
    result = await asyncio.to_thread(
        execute_generated_code,
        code,
        conf,
        workspace,
        output_file,
        script_name,
    )
    issues = validator(result.get("evidence_json")) if not execution_needs_repair(result) else []
    repairs = []
    max_repairs = int(conf.get("psd_code_max_repair_attempts", 1))

    while (execution_needs_repair(result) or issues) and len(repairs) < max_repairs:
        repair_context = execution_summary(result)
        repair_context["evidence_validation_issues"] = issues
        prompt = code_repair_prompt(
            case_view,
            source_manifest,
            specification,
            code,
            repair_context,
            output_file,
            stage,
        )
        repair_raw = await _call_stage(
            generate_content_fn,
            client,
            conf,
            prompt,
            token_key,
            4000,
            f"{stage} code repair {len(repairs) + 1}",
        )
        repairs.append({"prompt": prompt, "raw_response": repair_raw})
        code = extract_python_code(repair_raw)
        result = await asyncio.to_thread(
            execute_generated_code,
            code,
            conf,
            workspace,
            output_file,
            script_name,
        )
        issues = validator(result.get("evidence_json")) if not execution_needs_repair(result) else []

    return {
        "prompt": initial_prompt,
        "raw_response": raw_response,
        "repairs": repairs,
        "generated_code": code,
        "execution_result": result,
        "execution_summary": execution_summary(result),
        "validation_issues": issues,
        "accepted_evidence": (
            result.get("evidence_json")
            if not execution_needs_repair(result) and not issues
            else None
        ),
    }


def _parse_prediction(raw_text, parser, labels):
    parsed = parse_json_from_text(raw_text)
    if isinstance(parsed, dict):
        prediction = parser(str(parsed.get("prediction", "")))
        if prediction in labels:
            return prediction, parsed
    match = re.search(r'"prediction"\s*:\s*"?([A-Z])', str(raw_text or ""), flags=re.IGNORECASE)
    prediction = parser(match.group(1) if match else raw_text)
    return prediction, parsed


async def run_progressive_signal_agent(
    sample,
    conf,
    clients,
    generate_content_fn,
    prediction_parser,
    debug_callback=None,
    is_first_sample=False,
):
    case_view = build_case_view(sample, conf["dataset"])
    labels = candidate_labels(case_view)
    workspace = prepare_workspace(conf)
    source_manifest = build_source_manifest(
        workspace,
        str(conf.get("psd_current_bundle_train_context_policy", "allow")),
    )

    broad_plan_prompt = broad_planning_prompt(case_view, source_manifest)
    if is_first_sample and debug_callback:
        debug_callback("PSD Broad Signal Planning Prompt", broad_plan_prompt)
    broad_plan_raw = await _call_stage(
        generate_content_fn,
        clients["planning"],
        conf,
        broad_plan_prompt,
        "psd_broad_planning_max_output_tokens",
        1600,
        "broad signal planning",
    )
    broad_plan = parse_json_from_text(broad_plan_raw)
    if not isinstance(broad_plan, dict):
        broad_plan = {
            "coverage_goal": "Extract broad candidate-scoped observations from every available source.",
            "planning_parse_error": str(broad_plan_raw or "")[:1600],
        }

    broad_output = f"output/psd_broad_evidence_bundle{sample['bundle_id']}.json"
    broad_prompt = broad_code_prompt(case_view, source_manifest, broad_plan, broad_output)
    if is_first_sample and debug_callback:
        debug_callback("PSD Broad Signal Code Prompt", broad_prompt)
    broad_result = await _generate_execute_repair(
        stage="broad",
        sample_bundle_id=sample["bundle_id"],
        case_view=case_view,
        source_manifest=source_manifest,
        specification=broad_plan,
        initial_prompt=broad_prompt,
        client=clients["code"],
        conf=conf,
        generate_content_fn=generate_content_fn,
        workspace=workspace,
        output_file=broad_output,
        validator=lambda evidence: validate_broad_evidence(evidence, labels),
    )

    accumulated_evidence = {
        "broad": broad_result["accepted_evidence"],
        "deep_rounds": [],
    }
    execution_summaries = {"broad": broad_result["execution_summary"], "deep_rounds": []}
    diagnosis_history = []
    diagnosis_raw_history = []
    deep_trace = []
    previous_plans = []
    max_deep_rounds = int(conf.get("psd_max_deep_rounds", 1))

    for diagnosis_index in range(max_deep_rounds + 1):
        prompt = diagnosis_prompt(case_view, source_manifest, accumulated_evidence, execution_summaries)
        if is_first_sample and debug_callback:
            debug_callback(f"PSD Signal Diagnosis Prompt {diagnosis_index + 1}", prompt)
        raw = await _call_stage(
            generate_content_fn,
            clients["diagnosis"],
            conf,
            prompt,
            "psd_diagnosis_max_output_tokens",
            1800,
            f"signal diagnosis {diagnosis_index + 1}",
        )
        diagnosis = parse_json_from_text(raw)
        if not isinstance(diagnosis, dict):
            diagnosis = _diagnosis_fallback(raw)
        diagnosis_history.append(diagnosis)
        diagnosis_raw_history.append(raw)

        if str(diagnosis.get("status", "")).upper() != "NEEDS_DEEPENING":
            break
        if len(deep_trace) >= max_deep_rounds:
            break

        plan_prompt = deep_planning_prompt(
            case_view,
            source_manifest,
            accumulated_evidence,
            diagnosis,
            previous_plans,
        )
        if is_first_sample and debug_callback:
            debug_callback(f"PSD Deep Planning Prompt {len(deep_trace) + 1}", plan_prompt)
        plan_raw = await _call_stage(
            generate_content_fn,
            clients["planning"],
            conf,
            plan_prompt,
            "psd_deep_planning_max_output_tokens",
            2400,
            f"deep research planning round {len(deep_trace) + 1}",
        )
        plan = parse_json_from_text(plan_raw)
        if not isinstance(plan, dict) or not isinstance(plan.get("investigations"), list):
            deep_trace.append(
                {
                    "planning_prompt": plan_prompt,
                    "planning_raw_response": plan_raw,
                    "planning_json": plan,
                    "planning_error": "Deep planner did not return the required JSON specification.",
                }
            )
            break
        previous_plans.append(plan)

        round_number = len(deep_trace) + 1
        deep_output = f"output/psd_deep_evidence_bundle{sample['bundle_id']}_round{round_number}.json"
        code_prompt = deep_code_prompt(
            case_view,
            source_manifest,
            accumulated_evidence,
            diagnosis,
            plan,
            deep_output,
        )
        if is_first_sample and debug_callback:
            debug_callback(f"PSD Deep Signal Code Prompt {round_number}", code_prompt)
        code_result = await _generate_execute_repair(
            stage=f"deep_round{round_number}",
            sample_bundle_id=sample["bundle_id"],
            case_view=case_view,
            source_manifest=source_manifest,
            specification=plan,
            initial_prompt=code_prompt,
            client=clients["code"],
            conf=conf,
            generate_content_fn=generate_content_fn,
            workspace=workspace,
            output_file=deep_output,
            validator=lambda evidence, current_plan=plan: validate_deep_evidence(
                evidence, current_plan, labels
            ),
        )
        accepted = code_result["accepted_evidence"]
        accumulated_evidence["deep_rounds"].append(accepted)
        execution_summaries["deep_rounds"].append(code_result["execution_summary"])
        deep_trace.append(
            {
                "planning_prompt": plan_prompt,
                "planning_raw_response": plan_raw,
                "planning_json": plan,
                "code_result": code_result,
            }
        )

    final_diagnosis = diagnosis_history[-1] if diagnosis_history else _diagnosis_fallback("")
    final_evidence = {
        "method": "progressive_signal_discovery",
        "case": case_view,
        "source_manifest": source_manifest,
        "broad_evidence": accumulated_evidence["broad"],
        "deep_evidence_rounds": accumulated_evidence["deep_rounds"],
        "diagnosis": final_diagnosis,
        "audit": {
            "broad_validation_issues": broad_result["validation_issues"],
            "deep_validation_issues": [
                trace.get("code_result", {}).get("validation_issues", []) for trace in deep_trace
            ],
        },
    }

    final_prompt = decision_prompt(case_view, final_evidence)
    if is_first_sample and debug_callback:
        debug_callback("PSD Final Decision Prompt", final_prompt)
    decision_raw = await _call_stage(
        generate_content_fn,
        clients["prediction"],
        conf,
        final_prompt,
        "psd_prediction_max_output_tokens",
        1200,
        "final evidence decision",
    )
    prediction, decision_json = _parse_prediction(decision_raw, prediction_parser, labels)

    row = {
        "psd_workspace_dir": workspace["workspace_dir"],
        "psd_workspace_files": compact_json(workspace["files"]),
        "psd_source_manifest": compact_json(source_manifest),
        "psd_case_view": compact_json(case_view),
        "psd_broad_planning_prompt": broad_plan_prompt,
        "psd_broad_planning_raw_response": broad_plan_raw,
        "psd_broad_plan_json": compact_json(broad_plan),
        "psd_broad_code_prompt": broad_prompt,
        "psd_broad_code_raw_response": broad_result["raw_response"],
        "psd_broad_generated_code": broad_result["generated_code"],
        "psd_broad_code_repairs": compact_json(broad_result["repairs"]),
        "psd_broad_execution_summary": compact_json(broad_result["execution_summary"]),
        "psd_broad_validation_issues": compact_json(broad_result["validation_issues"]),
        "psd_broad_evidence_json": compact_json(broad_result["accepted_evidence"]),
        "psd_diagnosis_raw_history": compact_json(diagnosis_raw_history),
        "psd_diagnosis_history": compact_json(diagnosis_history),
        "psd_deep_trace": compact_json(deep_trace),
        "psd_final_evidence_json": compact_json(final_evidence),
        "psd_decision_prompt": final_prompt,
        "psd_decision_raw_response": decision_raw,
        "psd_decision_json": compact_json(decision_json),
    }
    if isinstance(decision_json, dict):
        row.update(
            {
                "psd_reasoning": decision_json.get("reasoning", ""),
                "psd_confidence": decision_json.get("confidence", ""),
                "psd_evidence_quality_used": decision_json.get("evidence_quality_used", ""),
                "psd_observations_used": compact_json(decision_json.get("observations_used", [])),
                "psd_downweighted_or_ignored": compact_json(
                    decision_json.get("downweighted_or_ignored", [])
                ),
            }
        )
    return row, prediction, decision_raw

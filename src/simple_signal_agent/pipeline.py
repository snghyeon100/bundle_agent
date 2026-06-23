import asyncio
import json
import os
import re

from progressive_signal_agent.common import (
    build_case_view,
    candidate_labels,
    compact_json,
    execution_summary,
    extract_python_code,
    parse_json_from_text,
)
from progressive_signal_agent.workspace import (
    build_source_manifest,
    execute_generated_code,
    execution_needs_repair,
    prepare_workspace,
)

from .prompts import (
    code_repair_prompt,
    decision_prompt,
    signal_code_prompt,
    sufficiency_evaluation_prompt,
)


CONFIG_PREFIX = "simple_signal"


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


def _forbidden_decision_keys(value, path=""):
    forbidden = {
        "prediction",
        "winner",
        "best_candidate",
        "preferred_candidate",
        "recommendation",
        "ranking",
        "final_score",
    }
    issues = []
    if isinstance(value, dict):
        for key, nested in value.items():
            current = f"{path}.{key}" if path else str(key)
            if str(key).strip().lower() in forbidden:
                issues.append(f"Evidence contains forbidden decision field: {current}.")
            issues.extend(_forbidden_decision_keys(nested, current))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            issues.extend(_forbidden_decision_keys(nested, f"{path}[{index}]"))
    return issues


def validate_signal_evidence(evidence, labels, allowed_source_names, max_evidence_chars=30000):
    if not isinstance(evidence, dict):
        return ["Evidence must be a JSON object."]

    issues = _forbidden_decision_keys(evidence)
    unexpected_top_level = sorted(set(evidence) - {"signals"})
    if unexpected_top_level:
        issues.append(
            "Evidence has unsupported top-level fields: " + ", ".join(unexpected_top_level) + "."
        )

    signals = evidence.get("signals")
    if not isinstance(signals, list) or not signals:
        issues.append("signals must be a non-empty list.")
        return issues

    allowed_sources = set(allowed_source_names)
    seen_names = set()
    required_labels = set(labels)
    for index, signal in enumerate(signals):
        prefix = f"signals[{index}]"
        if not isinstance(signal, dict):
            issues.append(f"{prefix} must be an object.")
            continue
        expected_fields = {"signal_name", "description", "sources", "candidate_observations"}
        extra_fields = sorted(set(signal) - expected_fields)
        if extra_fields:
            issues.append(f"{prefix} has unsupported fields: {', '.join(extra_fields)}.")

        signal_name = str(signal.get("signal_name", "")).strip()
        if not signal_name:
            issues.append(f"{prefix}.signal_name must be non-empty.")
        elif signal_name in seen_names:
            issues.append(f"Duplicate signal_name: {signal_name}.")
        seen_names.add(signal_name)

        if not str(signal.get("description", "")).strip():
            issues.append(f"{prefix}.description must be non-empty.")

        sources = signal.get("sources")
        if not isinstance(sources, list) or not sources:
            issues.append(f"{prefix}.sources must be a non-empty list.")
        else:
            unknown = sorted(
                {
                    str(source)
                    for source in sources
                    if str(source) not in allowed_sources
                }
            )
            if unknown:
                issues.append(f"{prefix}.sources contains unavailable sources: {', '.join(unknown)}.")

        observations = signal.get("candidate_observations")
        if not isinstance(observations, dict):
            issues.append(f"{prefix}.candidate_observations must be an object.")
            continue
        observed_labels = set(observations)
        missing = sorted(required_labels - observed_labels)
        extra = sorted(observed_labels - required_labels)
        if missing:
            issues.append(f"{prefix} is missing candidates: {', '.join(missing)}.")
        if extra:
            issues.append(f"{prefix} has unknown candidates: {', '.join(extra)}.")

        for label in labels:
            observation = observations.get(label)
            if not isinstance(observation, dict):
                issues.append(f"{prefix}.candidate_observations.{label} must be an object.")
                continue
            if set(observation) != {"value", "evidence"}:
                issues.append(
                    f"{prefix}.candidate_observations.{label} must contain only value and evidence."
                )
            evidence_items = observation.get("evidence")
            if not isinstance(evidence_items, list):
                issues.append(f"{prefix}.candidate_observations.{label}.evidence must be a list.")
            elif len(evidence_items) > 3:
                issues.append(
                    f"{prefix}.candidate_observations.{label}.evidence exceeds the limit of 3 entries."
                )

    serialized_size = len(compact_json(evidence))
    if serialized_size > int(max_evidence_chars):
        issues.append(
            f"Evidence JSON is too large ({serialized_size} chars > {int(max_evidence_chars)})."
        )
    return issues


def _evaluation_fallback(reason, evidence_quality="NONE"):
    return {
        "status": "INCONCLUSIVE",
        "evidence_quality": evidence_quality,
        "reliable_signals": [],
        "weak_or_failed_signals": [],
        "coverage_problems": [],
        "redundancy_problems": [],
        "conflicts": [],
        "evidence_gaps": [],
        "required_improvements": [],
        "expected_new_information": "",
        "reason": str(reason or "")[:2000],
    }


def normalize_evaluation(value, remaining_refinement_rounds):
    if not isinstance(value, dict):
        return _evaluation_fallback("The evaluator did not return a JSON object.")

    status = str(value.get("status", "")).strip().upper()
    if status not in {"SUFFICIENT", "REFINE", "INCONCLUSIVE"}:
        return _evaluation_fallback(f"The evaluator returned an invalid status: {status or '(empty)' }.")

    quality = str(value.get("evidence_quality", "NONE")).strip().upper()
    if quality not in {"NONE", "LOW", "MEDIUM", "HIGH"}:
        quality = "NONE"

    normalized = {
        "status": status,
        "evidence_quality": quality,
        "reliable_signals": value.get("reliable_signals", []),
        "weak_or_failed_signals": value.get("weak_or_failed_signals", []),
        "coverage_problems": value.get("coverage_problems", []),
        "redundancy_problems": value.get("redundancy_problems", []),
        "conflicts": value.get("conflicts", []),
        "evidence_gaps": value.get("evidence_gaps", []),
        "required_improvements": value.get("required_improvements", []),
        "expected_new_information": str(value.get("expected_new_information", "")).strip(),
        "reason": str(value.get("reason", "")).strip(),
    }
    list_fields = (
        "reliable_signals",
        "weak_or_failed_signals",
        "coverage_problems",
        "redundancy_problems",
        "conflicts",
        "evidence_gaps",
        "required_improvements",
    )
    for field in list_fields:
        if not isinstance(normalized[field], list):
            normalized[field] = [str(normalized[field])]

    if status == "REFINE":
        has_actionable_gap = bool(normalized["evidence_gaps"])
        has_requirement = bool(normalized["required_improvements"])
        has_expected_information = bool(normalized["expected_new_information"])
        if int(remaining_refinement_rounds) <= 0:
            normalized["status"] = "INCONCLUSIVE"
            normalized["reason"] = (
                "Refinement budget exhausted. " + normalized["reason"]
            ).strip()
        elif not (has_actionable_gap and has_requirement and has_expected_information):
            normalized["status"] = "INCONCLUSIVE"
            normalized["reason"] = (
                "REFINE lacked a concrete evidence gap, required improvement, or expected new information. "
                + normalized["reason"]
            ).strip()
    return normalized


def _compact_execution_context(result, validation_issues):
    summary = execution_summary(result)
    stdout = str(result.get("stdout") or "")
    summary["stdout_tail"] = stdout[-1600:] if stdout else ""
    summary["evidence_validation_issues"] = validation_issues
    return summary


async def _generate_execute_repair(
    *,
    sample_bundle_id,
    round_index,
    case_view,
    source_manifest,
    initial_prompt,
    client,
    conf,
    generate_content_fn,
    workspace,
    output_file,
    labels,
):
    raw_response = await _call_stage(
        generate_content_fn,
        client,
        conf,
        initial_prompt,
        "simple_signal_code_max_output_tokens",
        4000,
        f"simple signal code generation round {round_index + 1}",
    )
    code = extract_python_code(raw_response)
    script_name = f"simple_signal_bundle{sample_bundle_id}_round{round_index + 1}.py"
    result = await asyncio.to_thread(
        execute_generated_code,
        code,
        conf,
        workspace,
        output_file,
        script_name,
        CONFIG_PREFIX,
    )
    source_names = [source["name"] for source in source_manifest.get("sources", [])]
    max_chars = int(conf.get("simple_signal_max_evidence_chars", 30000))
    issues = (
        validate_signal_evidence(result.get("evidence_json"), labels, source_names, max_chars)
        if not execution_needs_repair(result)
        else []
    )
    repairs = []
    max_repairs = max(0, int(conf.get("simple_signal_code_max_repair_attempts", 1)))

    while (execution_needs_repair(result) or issues) and len(repairs) < max_repairs:
        repair_context = _compact_execution_context(result, issues)
        prompt = code_repair_prompt(
            case_view,
            source_manifest,
            code,
            repair_context,
            output_file,
        )
        repair_raw = await _call_stage(
            generate_content_fn,
            client,
            conf,
            prompt,
            "simple_signal_code_max_output_tokens",
            4000,
            f"simple signal code repair round {round_index + 1} attempt {len(repairs) + 1}",
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
            CONFIG_PREFIX,
        )
        issues = (
            validate_signal_evidence(result.get("evidence_json"), labels, source_names, max_chars)
            if not execution_needs_repair(result)
            else []
        )

    accepted = (
        result.get("evidence_json")
        if not execution_needs_repair(result) and not issues
        else None
    )
    return {
        "prompt": initial_prompt,
        "raw_response": raw_response,
        "repairs": repairs,
        "generated_code": code,
        "execution_result": result,
        "execution_summary": _compact_execution_context(result, issues),
        "validation_issues": issues,
        "accepted_evidence": accepted,
    }


def _trim_scalar(value, limit=800):
    if value is None or isinstance(value, (bool, int, float)):
        return value
    text = " ".join(str(value).split())
    return text[:limit]


def _item_profile(item_id, item_info, dataset):
    item_key = str(int(item_id))
    raw = item_info.get(item_key, {})
    info = raw if isinstance(raw, dict) else {}
    name = str(dataset or "").lower()
    if "spotify" in name:
        text_parts = [
            info.get("track_name"),
            info.get("artist_name"),
            info.get("album_name"),
        ]
        text = " - ".join(" ".join(str(part).split()) for part in text_parts if part)
    else:
        text = " ".join(str(info.get("title", "")).split())
    if not text:
        text = f"Item {item_key}"

    excluded = {
        "id",
        "pic",
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
    for key, value in info.items():
        if str(key).lower() in excluded:
            continue
        if isinstance(value, (str, bool, int, float)) or value is None:
            metadata[str(key)] = _trim_scalar(value)
        elif isinstance(value, list):
            metadata[str(key)] = [_trim_scalar(entry, 200) for entry in value[:10]]
    return {"item_id": int(item_id), "text": text[:1200], "metadata": metadata}


def build_decision_case(sample, conf):
    info_path = os.path.join(conf["data_path"], conf["dataset"], "item_info.json")
    with open(info_path, "r", encoding="utf-8") as handle:
        item_info = json.load(handle)
    labels = [chr(ord("A") + index) for index in range(len(sample["candidate_indices"]))]
    return {
        "case_id": f"bundle_{sample['bundle_id']}",
        "dataset": conf["dataset"],
        "bundle_id": int(sample["bundle_id"]),
        "partial_items": [
            _item_profile(item_id, item_info, conf["dataset"])
            for item_id in sample["input_indices"]
        ],
        "candidates": [
            {
                "label": label,
                **_item_profile(item_id, item_info, conf["dataset"]),
            }
            for label, item_id in zip(labels, sample["candidate_indices"])
        ],
    }


def _parse_prediction(raw_text, parser, labels):
    parsed = parse_json_from_text(raw_text)
    if isinstance(parsed, dict):
        prediction = parser(str(parsed.get("prediction", "")))
        if prediction in labels:
            return prediction, {"prediction": prediction}
    match = re.search(r'"prediction"\s*:\s*"?([A-Z])', str(raw_text or ""), flags=re.IGNORECASE)
    prediction = parser(match.group(1) if match else raw_text)
    return prediction, parsed


async def run_simple_signal_agent(
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
    workspace = prepare_workspace(conf, config_prefix=CONFIG_PREFIX)
    source_manifest = build_source_manifest(
        workspace,
        str(conf.get("simple_signal_current_bundle_train_context_policy", "allow")),
    )
    max_refinements = max(0, int(conf.get("simple_signal_max_refinement_rounds", 1)))
    max_evidence_chars = int(conf.get("simple_signal_max_evidence_chars", 30000))

    round_trace = []
    evaluation_history = []
    refinement_context = None
    final_evidence = None
    final_evaluation = _evaluation_fallback("No signal round completed.")

    for round_index in range(max_refinements + 1):
        output_file = (
            f"output/simple_signal_evidence_bundle{sample['bundle_id']}_round{round_index + 1}.json"
        )
        code_prompt = signal_code_prompt(
            case_view,
            source_manifest,
            output_file,
            max_evidence_chars,
            refinement_context=refinement_context,
        )
        if is_first_sample and debug_callback:
            debug_callback(f"Simple Signal Code Prompt {round_index + 1}", code_prompt)

        code_result = await _generate_execute_repair(
            sample_bundle_id=sample["bundle_id"],
            round_index=round_index,
            case_view=case_view,
            source_manifest=source_manifest,
            initial_prompt=code_prompt,
            client=clients["code"],
            conf=conf,
            generate_content_fn=generate_content_fn,
            workspace=workspace,
            output_file=output_file,
            labels=labels,
        )
        evidence = code_result["accepted_evidence"]
        remaining = max_refinements - round_index
        evaluation_prompt = ""
        evaluation_raw = ""

        if evidence is None:
            final_evaluation = _evaluation_fallback(
                "Signal code did not produce valid evidence after the configured repair attempts. "
                + "; ".join(code_result["validation_issues"])
            )
        else:
            evaluation_prompt = sufficiency_evaluation_prompt(
                case_view,
                source_manifest,
                code_result["generated_code"],
                code_result["execution_summary"],
                evidence,
                round_index,
                remaining,
                evaluation_history,
            )
            if is_first_sample and debug_callback:
                debug_callback(
                    f"Simple Signal Sufficiency Evaluation Prompt {round_index + 1}",
                    evaluation_prompt,
                )
            evaluation_raw = await _call_stage(
                generate_content_fn,
                clients["evaluator"],
                conf,
                evaluation_prompt,
                "simple_signal_evaluation_max_output_tokens",
                1800,
                f"simple signal sufficiency evaluation round {round_index + 1}",
            )
            final_evaluation = normalize_evaluation(
                parse_json_from_text(evaluation_raw),
                remaining,
            )
            final_evidence = evidence

        round_trace.append(
            {
                "round": round_index + 1,
                "code_prompt": code_prompt,
                "code_raw_response": code_result["raw_response"],
                "generated_code": code_result["generated_code"],
                "code_repairs": code_result["repairs"],
                "execution_summary": code_result["execution_summary"],
                "validation_issues": code_result["validation_issues"],
                "accepted_evidence": evidence,
                "evaluation_prompt": evaluation_prompt,
                "evaluation_raw_response": evaluation_raw,
                "evaluation": final_evaluation,
            }
        )

        if evidence is None or final_evaluation["status"] != "REFINE":
            break

        evaluation_history.append(final_evaluation)
        refinement_context = {
            "previous_code": code_result["generated_code"],
            "previous_evidence": evidence,
            "execution_summary": code_result["execution_summary"],
            "evaluator_feedback": final_evaluation,
        }

    if final_evidence is None:
        final_evidence = {"signals": []}

    decision_case = build_decision_case(sample, conf)
    refinement_history = [
        {
            "round": trace["round"],
            "status": trace["evaluation"].get("status", "INCONCLUSIVE"),
            "evidence_quality": trace["evaluation"].get("evidence_quality", "NONE"),
            "evidence_gaps": trace["evaluation"].get("evidence_gaps", []),
            "required_improvements": trace["evaluation"].get("required_improvements", []),
        }
        for trace in round_trace
    ]
    final_prompt = decision_prompt(
        decision_case,
        final_evidence,
        final_evaluation,
        refinement_history,
    )
    if is_first_sample and debug_callback:
        debug_callback("Simple Signal Final Decision Prompt", final_prompt)
    decision_raw = await _call_stage(
        generate_content_fn,
        clients["prediction"],
        conf,
        final_prompt,
        "simple_signal_prediction_max_output_tokens",
        200,
        "simple signal final decision",
    )
    prediction, decision_json = _parse_prediction(decision_raw, prediction_parser, labels)

    row = {
        "simple_signal_workspace_dir": workspace["workspace_dir"],
        "simple_signal_workspace_files": compact_json(workspace["files"]),
        "simple_signal_source_manifest": compact_json(source_manifest),
        "simple_signal_case_view": compact_json(case_view),
        "simple_signal_decision_case": compact_json(decision_case),
        "simple_signal_round_count": len(round_trace),
        "simple_signal_refinement_count": max(0, len(round_trace) - 1),
        "simple_signal_round_trace": compact_json(round_trace),
        "simple_signal_final_evidence_json": compact_json(final_evidence),
        "simple_signal_final_evaluation_json": compact_json(final_evaluation),
        "simple_signal_final_status": final_evaluation.get("status", "INCONCLUSIVE"),
        "simple_signal_final_evidence_quality": final_evaluation.get("evidence_quality", "NONE"),
        "simple_signal_decision_prompt": final_prompt,
        "simple_signal_decision_raw_response": decision_raw,
        "simple_signal_decision_json": compact_json(decision_json),
    }
    return row, prediction, decision_raw

"""Two-call hypothesis-conditioned exemplar retrieval and prediction pipeline."""

from hashlib import sha256
import json
import re

from code.common import parse_json_from_text
from operator_learning.runtime import implementation_hash

from .prompts import prediction_prompt, program_generation_prompt
from .raw_workspace import dataset_workspace_manifest
from .renderer import render_retrieval_evidence
from .runtime import execute_program_subprocess
from .schemas import (
    DISCOVERY_SCHEMA_VERSION,
    normalize_discovery_result,
    validate_discovery_result,
    validate_prediction_result,
)
from .source_api import DatasetSourceAPI


def _parse_discovery_response(raw_text):
    """Parse LLM1 JSON, repairing only escaped standalone list strings."""
    parsed = parse_json_from_text(raw_text)
    if isinstance(parsed, dict):
        return parsed, []

    repaired_lines = []
    repair_count = 0
    pattern = re.compile(r'^(\s*)\\"(.*)\\"(,?)\s*$')
    for line in str(raw_text or "").splitlines():
        match = pattern.match(line)
        if match:
            repaired_lines.append(
                f'{match.group(1)}"{match.group(2)}"{match.group(3)}'
            )
            repair_count += 1
        else:
            repaired_lines.append(line)
    repaired = parse_json_from_text("\n".join(repaired_lines))
    if isinstance(repaired, dict):
        return repaired, [
            {
                "type": "standalone_list_string_quote_normalization",
                "replacement_count": repair_count,
            }
        ]
    return parsed, []


def _admit_valid_discovery_entries(value, *, available_sources):
    """Keep independently valid programs from one LLM1 response."""
    empty = {
        "schema_version": (
            value.get("schema_version") if isinstance(value, dict) else None
        ),
        "programs": [],
    }
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != DISCOVERY_SCHEMA_VERSION
        or not isinstance(value.get("programs"), list)
    ):
        return empty, []

    seen_programs = set()
    rejections = []
    for program in value["programs"]:
        program_id = (
            program.get("id") if isinstance(program, dict) else None
        )
        if program_id in seen_programs:
            rejections.append(
                {
                    "program_id": program_id,
                    "validation_issues": ["program id must be unique"],
                }
            )
            continue
        singleton = {
            "schema_version": value["schema_version"],
            "programs": [program],
        }
        issues = validate_discovery_result(
            singleton,
            available_sources=available_sources,
            min_hypotheses=1,
            max_hypotheses=1,
        )
        if issues:
            rejections.append(
                {
                    "program_id": program_id,
                    "validation_issues": issues,
                }
            )
            continue
        seen_programs.add(program_id)
        empty["programs"].append(program)
    return normalize_discovery_result(empty), rejections


def _normalize_discovery_contract(value):
    """Repair harmless declarations of always-available workspace values."""
    normalized = normalize_discovery_result(value)
    repairs = []
    if not isinstance(normalized, dict):
        return normalized, repairs
    programs = normalized.get("programs")
    if not isinstance(programs, list):
        return normalized, repairs
    for index, program in enumerate(programs):
        if not isinstance(program, dict):
            continue
        required_sources = program.get("required_sources")
        if not isinstance(required_sources, list):
            continue
        filtered = [
            source_id
            for source_id in required_sources
            if source_id != "item_ids"
        ]
        if len(filtered) != len(required_sources):
            program["required_sources"] = filtered
            repairs.append(
                {
                    "type": "remove_always_available_item_ids_from_required_sources",
                    "program_index": index,
                    "program_id": program.get("id"),
                }
            )
    return normalized, repairs


def _profile(source_api, item_id):
    item_id = int(item_id)
    raw = source_api.get_item_metadata([item_id]).get(item_id, {})
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
    metadata = {
        str(key): value
        for key, value in raw.items()
        if str(key).lower() not in excluded
        and isinstance(value, (str, int, float, bool))
    }
    return {
        "item_id": item_id,
        "text": source_api.item_text(item_id)[:1200],
        "metadata": metadata,
    }


def build_online_case(sample, conf, source_api):
    labels = [
        chr(ord("A") + index)
        for index in range(len(sample["candidate_indices"]))
    ]
    return {
        "case_id": f"bundle_{int(sample['bundle_id'])}",
        "dataset": conf["dataset"],
        "bundle_id": int(sample["bundle_id"]),
        "partial_items": [
            _profile(source_api, item_id)
            for item_id in sample["input_indices"]
        ],
        "answer_options": [
            {"label": label, **_profile(source_api, item_id)}
            for label, item_id in zip(labels, sample["candidate_indices"])
        ],
        "source_diagnostics": source_api.diagnostics(sample["input_indices"]),
    }


def _model_answer_options(case):
    return [
        {"label": option["label"], "text": option["text"]}
        for option in case["answer_options"]
    ]


def _program_fixation(
    program,
    retrieved_item_budget,
    supporting_context_budget,
):
    contract = {
        "completion_hypothesis": program.get("hypothesis", ""),
        "program_id": program.get("id"),
        "reference_construction": program.get("strategy", {}).get("reference"),
        "retrieval_strategy": program.get("strategy", {}).get("retrieval"),
        "required_sources": program.get("required_sources", []),
        "parameters": program.get("parameters", {}),
        "retrieved_item_budget_per_hypothesis": int(retrieved_item_budget),
        "max_supporting_contexts_per_item": int(supporting_context_budget),
        "implementation_sha256": implementation_hash(program.get("code", "")),
    }
    encoded = json.dumps(
        contract,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        **contract,
        "fixation_sha256": sha256(encoded.encode("utf-8")).hexdigest(),
    }


def _ranking_metrics(ranking, true_label):
    gt_rank = ranking.index(true_label) + 1 if true_label in ranking else None
    return {
        "gt_rank": gt_rank,
        "reciprocal_rank": 1.0 / gt_rank if gt_rank else 0.0,
        "hit_at_1": bool(gt_rank and gt_rank <= 1),
        "hit_at_3": bool(gt_rank and gt_rank <= 3),
        "hit_at_5": bool(gt_rank and gt_rank <= 5),
    }


def _retrieval_metrics(rendered, case, true_label):
    item_by_label = {
        option["label"]: int(option["item_id"])
        for option in case["answer_options"]
    }
    true_item_id = item_by_label.get(true_label)
    merged = list(rendered.get("merged_retrieved_item_ids", []))
    by_program = rendered.get("retrieved_item_ids_by_program", {})
    gt_retrieved_by = [
        program_id
        for program_id, item_ids in by_program.items()
        if true_item_id in item_ids
    ]
    counts = dict(rendered.get("retrieval_counts", {}))
    return {
        **counts,
        "retrieved_item_count_by_program": {
            program_id: len(item_ids)
            for program_id, item_ids in by_program.items()
        },
        "gt_retrieved": bool(gt_retrieved_by),
        "gt_retrieved_by_programs": gt_retrieved_by,
        "gt_retrieval_rank": (
            merged.index(true_item_id) + 1
            if true_item_id in merged
            else None
        ),
    }


async def run_online_hypothesis_program(
    sample,
    conf,
    call_program_llm,
    call_prediction_llm,
    *,
    program_executor=None,
):
    """Run retrieval synthesis once per hypothesis, then one final prediction."""
    host_reader = DatasetSourceAPI(conf)
    workspace_manifest = dataset_workspace_manifest(conf)
    case = build_online_case(sample, conf, host_reader)
    hypothesis_count = int(conf.get("online_hypothesis_count", 3))
    min_hypotheses = hypothesis_count
    max_hypotheses = hypothesis_count
    retrieved_item_budget = int(
        conf.get("online_retrieved_item_budget_per_hypothesis", 5)
    )
    supporting_context_budget = int(
        conf.get("online_max_supporting_contexts_per_item", 2)
    )
    max_rendered_items = int(
        conf.get("online_max_rendered_items_per_hypothesis", 5)
    )
    max_rendered_supporting_contexts = int(
        conf.get("online_max_rendered_supporting_contexts_per_item", 2)
    )

    llm1_prompt = program_generation_prompt(
        dataset=conf["dataset"],
        partial_items=case["partial_items"],
        workspace_manifest=workspace_manifest,
        min_hypotheses=min_hypotheses,
        max_hypotheses=max_hypotheses,
        retrieved_item_budget=retrieved_item_budget,
        supporting_context_budget=supporting_context_budget,
    )
    llm1_raw = await call_program_llm(
        llm1_prompt,
        "completion retrieval program synthesis",
    )
    llm1_parsed, llm1_parse_repairs = _parse_discovery_response(llm1_raw)
    llm1_effective, llm1_contract_repairs = _normalize_discovery_contract(
        llm1_parsed
    )
    available_sources = [
        component["id"]
        for component in workspace_manifest["components"]
    ]
    llm1_issues = validate_discovery_result(
        llm1_effective,
        available_sources=available_sources,
        min_hypotheses=min_hypotheses,
        max_hypotheses=max_hypotheses,
    )
    discovery, rejected_entries = _admit_valid_discovery_entries(
        llm1_effective,
        available_sources=available_sources,
    )

    fixations = {
        program["id"]: _program_fixation(
            program,
            retrieved_item_budget,
            supporting_context_budget,
        )
        for program in discovery["programs"]
    }

    executor = program_executor or execute_program_subprocess
    partial_ids = [
        int(item["item_id"]) for item in case["partial_items"]
    ]
    executions = {}
    for program in discovery["programs"]:
        program_id = program["id"]
        try:
            execution = executor(
                program=program,
                conf=conf,
                partial_item_ids=partial_ids,
                retrieved_item_budget=retrieved_item_budget,
                supporting_context_budget=supporting_context_budget,
            )
        except Exception as error:
            execution = {
                "status": "execution_error",
                "result": None,
                "validation_issues": [],
                "error": f"{type(error).__name__}: {error}",
            }
        executions[program_id] = execution

    rendered = render_retrieval_evidence(
        programs=discovery["programs"],
        executions=executions,
        source_api=host_reader,
        answer_options=case["answer_options"],
        max_items_per_hypothesis=max_rendered_items,
        max_supporting_contexts_per_item=max_rendered_supporting_contexts,
    )
    llm2_prompt = prediction_prompt(
        dataset=conf["dataset"],
        partial_items=case["partial_items"],
        answer_options=_model_answer_options(case),
        retrieval_evidence=rendered["model_view"],
    )
    llm2_raw = await call_prediction_llm(
        llm2_prompt,
        "retrieval-grounded prediction",
    )
    llm2_parsed = parse_json_from_text(llm2_raw)
    labels = [option["label"] for option in case["answer_options"]]
    llm2_issues = validate_prediction_result(llm2_parsed, labels)
    valid_prediction = isinstance(llm2_parsed, dict) and not llm2_issues
    prediction = llm2_parsed.get("prediction") if valid_prediction else None
    ranking = list(llm2_parsed.get("ranking", [])) if valid_prediction else []
    true_label = str(sample["true_option_char"])

    return {
        "case": case,
        "dataset_workspace_manifest": workspace_manifest,
        "llm1": {
            "prompt": llm1_prompt,
            "raw_response": llm1_raw,
            "parsed_response": llm1_parsed,
            "effective_response": llm1_effective,
            "parse_repairs": llm1_parse_repairs,
            "contract_repairs": llm1_contract_repairs,
            "validation_issues": llm1_issues,
            "rejected_entries": rejected_entries,
            "admitted_program_count": len(discovery["programs"]),
        },
        "program_fixations": fixations,
        "executions": executions,
        "rendered_retrieval_evidence": rendered["model_view"],
        "retrieval_trace": {
            "retrieved_item_ids_by_program": (
                rendered["retrieved_item_ids_by_program"]
            ),
            "merged_retrieved_item_ids": rendered["merged_retrieved_item_ids"],
        },
        "llm2": {
            "prompt": llm2_prompt,
            "raw_response": llm2_raw,
            "parsed_response": llm2_parsed,
            "validation_issues": llm2_issues,
        },
        "evaluation": {
            "prediction": prediction,
            "ranking": ranking,
            "valid_prediction": valid_prediction,
            "true_label": true_label,
            "prediction_hit": int(prediction == true_label),
            **_ranking_metrics(ranking, true_label),
            **_retrieval_metrics(rendered, case, true_label),
            "llm_calls": 2,
            "program_count": len(discovery["programs"]),
            "proposed_program_count": (
                len(llm1_parsed.get("programs", []))
                if isinstance(llm1_parsed, dict)
                and isinstance(llm1_parsed.get("programs"), list)
                else 0
            ),
        },
    }

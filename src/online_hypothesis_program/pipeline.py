"""Two-LLM-call online hypothesis-program search and prediction pipeline."""

from code.common import parse_json_from_text

from .prompts import prediction_prompt, program_generation_prompt
from .renderer import render_search_evidence
from .runtime import execute_program_subprocess
from .schemas import (
    normalize_discovery_result,
    validate_discovery_result,
    validate_prediction_result,
)
from .source_api import DatasetSourceAPI, source_capability_manifest


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
    partial_items = [
        _profile(source_api, item_id)
        for item_id in sample["input_indices"]
    ]
    answer_options = [
        {
            "label": label,
            **_profile(source_api, item_id),
        }
        for label, item_id in zip(labels, sample["candidate_indices"])
    ]
    return {
        "case_id": f"bundle_{int(sample['bundle_id'])}",
        "dataset": conf["dataset"],
        "bundle_id": int(sample["bundle_id"]),
        "partial_items": partial_items,
        "answer_options": answer_options,
        "source_diagnostics": source_api.diagnostics(sample["input_indices"]),
    }


def _model_answer_options(case):
    return [
        {
            "label": option["label"],
            "text": option["text"],
            "metadata": option.get("metadata", {}),
        }
        for option in case["answer_options"]
    ]


def _candidate_retrieval_metrics(candidate_ids, ground_truth_item_id):
    target = int(ground_truth_item_id)
    rank = next(
        (
            index
            for index, item_id in enumerate(candidate_ids, start=1)
            if int(item_id) == target
        ),
        None,
    )
    return {
        "retrieved_candidate_count": len(candidate_ids),
        "ground_truth_retrieved": rank is not None,
        "ground_truth_retrieval_rank": rank,
        "ground_truth_reciprocal_rank": 1.0 / rank if rank else 0.0,
    }


async def run_online_hypothesis_program(
    sample,
    conf,
    call_program_llm,
    call_prediction_llm,
    *,
    program_executor=None,
):
    """Run exactly two LLM calls with deterministic execution between them."""
    source_api = DatasetSourceAPI(conf)
    source_capabilities = source_capability_manifest(conf)
    case = build_online_case(sample, conf, source_api)
    max_hypotheses = int(conf.get("online_hypothesis_max_count", 3))
    candidate_budget = int(
        conf.get("online_candidate_budget_per_hypothesis", 5)
    )
    evidence_budget = int(
        conf.get("online_evidence_budget_per_hypothesis", 8)
    )
    total_candidate_budget = int(
        conf.get("online_total_candidate_budget", 10)
    )

    llm1_prompt = program_generation_prompt(
        dataset=conf["dataset"],
        partial_items=case["partial_items"],
        source_diagnostics=case["source_diagnostics"],
        source_capabilities=source_capabilities,
        max_hypotheses=max_hypotheses,
    )
    llm1_raw = await call_program_llm(
        llm1_prompt,
        "online hypothesis-conditioned program synthesis",
    )
    llm1_parsed = parse_json_from_text(llm1_raw)
    llm1_issues = validate_discovery_result(
        llm1_parsed,
        available_sources=source_api.available_sources,
        max_hypotheses=max_hypotheses,
    )
    discovery = (
        normalize_discovery_result(llm1_parsed)
        if not llm1_issues
        else {
            "schema_version": None,
            "hypotheses": [],
            "programs": [],
        }
    )

    executor = program_executor or execute_program_subprocess
    hypotheses_by_id = {
        hypothesis["id"]: hypothesis
        for hypothesis in discovery["hypotheses"]
    }
    executions = {}
    for program in discovery["programs"]:
        hypothesis_id = program["hypothesis_id"]
        hypothesis = hypotheses_by_id[hypothesis_id]
        try:
            execution = executor(
                program=program,
                hypothesis=hypothesis,
                conf=conf,
                partial_item_ids=[
                    item["item_id"] for item in case["partial_items"]
                ],
                candidate_budget=candidate_budget,
                evidence_budget=evidence_budget,
            )
        except Exception as error:
            execution = {
                "status": "execution_error",
                "result": None,
                "validation_issues": [],
                "error": f"{type(error).__name__}: {error}",
            }
        executions[hypothesis_id] = execution

    rendered = render_search_evidence(
        hypotheses=discovery["hypotheses"],
        programs=discovery["programs"],
        executions=executions,
        source_api=source_api,
        answer_options=case["answer_options"],
        total_candidate_budget=total_candidate_budget,
    )
    llm2_prompt = prediction_prompt(
        dataset=conf["dataset"],
        partial_items=case["partial_items"],
        answer_options=_model_answer_options(case),
        search_evidence=rendered["model_view"],
    )
    llm2_raw = await call_prediction_llm(
        llm2_prompt,
        "online hypothesis-program final prediction",
    )
    llm2_parsed = parse_json_from_text(llm2_raw)
    labels = [option["label"] for option in case["answer_options"]]
    llm2_issues = validate_prediction_result(llm2_parsed, labels)
    prediction = (
        llm2_parsed.get("prediction")
        if isinstance(llm2_parsed, dict) and not llm2_issues
        else None
    )
    true_label = str(sample["true_option_char"])
    retrieval_metrics = _candidate_retrieval_metrics(
        rendered["retained_candidate_item_ids"],
        sample["true_indice"],
    )

    return {
        "case": case,
        "source_capabilities": source_capabilities,
        "llm1": {
            "prompt": llm1_prompt,
            "raw_response": llm1_raw,
            "parsed_response": llm1_parsed,
            "validation_issues": llm1_issues,
        },
        "executions": executions,
        "rendered_search_evidence": rendered["model_view"],
        "retained_candidate_item_ids": rendered["retained_candidate_item_ids"],
        "llm2": {
            "prompt": llm2_prompt,
            "raw_response": llm2_raw,
            "parsed_response": llm2_parsed,
            "validation_issues": llm2_issues,
        },
        "evaluation": {
            **retrieval_metrics,
            "prediction": prediction,
            "valid_prediction": prediction in labels,
            "true_label": true_label,
            "prediction_hit": int(prediction == true_label),
            "llm_calls": 2,
            "successful_program_count": sum(
                execution.get("status") == "success"
                for execution in executions.values()
            ),
            "program_count": len(discovery["programs"]),
        },
    }

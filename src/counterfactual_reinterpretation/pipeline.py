"""Two-call train-free candidate-induced set reinterpretation pipeline."""

from code.common import parse_json_from_text
from code.pipeline import build_decision_case

from .prompts import (
    candidate_reinterpretation_prompt,
    contrastive_adjudication_prompt,
)
from .schemas import validate_adjudication, validate_reinterpretations


def _model_case(decision_case):
    return {
        "partial_items": [
            {
                "label": f"P{index}",
                "text": str(item.get("text") or ""),
            }
            for index, item in enumerate(
                decision_case.get("partial_items", []),
                start=1,
            )
            if isinstance(item, dict)
        ],
        "answer_options": [
            {
                "label": str(option.get("label") or ""),
                "text": str(option.get("text") or ""),
            }
            for option in decision_case.get("candidates", [])
            if isinstance(option, dict)
        ],
    }


def _evaluation(
    sample,
    candidate_labels,
    parsed_decision,
    decision_issues,
    method_issues,
):
    ranking = (
        [str(label) for label in parsed_decision.get("ranking", [])]
        if isinstance(parsed_decision, dict) and not decision_issues
        else []
    )
    prediction = ranking[0] if ranking else None
    true_label = str(sample["true_option_char"])
    gt_rank = ranking.index(true_label) + 1 if true_label in ranking else None
    return {
        "candidate_count": len(candidate_labels),
        "prediction": prediction,
        "ranking": ranking,
        "true_label": true_label,
        "hit": bool(prediction == true_label),
        "gt_rank": gt_rank,
        "reciprocal_rank": 1.0 / gt_rank if gt_rank else 0.0,
        "hit_at_1": bool(gt_rank and gt_rank <= 1),
        "hit_at_3": bool(gt_rank and gt_rank <= 3),
        "hit_at_5": bool(gt_rank and gt_rank <= 5),
        "llm_calls": 2,
        "valid": not method_issues,
    }


async def run_counterfactual_reinterpretation(
    sample,
    conf,
    call_analysis,
    call_decision,
):
    """Reinterpret every candidate-completed set, then adjudicate globally."""
    decision_case = build_decision_case(sample, conf)
    model_case = _model_case(decision_case)
    partial_labels = [
        item["label"] for item in model_case["partial_items"]
    ]
    candidate_labels = [
        option["label"] for option in model_case["answer_options"]
    ]

    analysis_prompt = candidate_reinterpretation_prompt(
        dataset=conf["dataset"],
        partial_items=model_case["partial_items"],
        answer_options=model_case["answer_options"],
    )
    analysis_raw = await call_analysis(
        analysis_prompt,
        "candidate-induced completed-set reinterpretation",
    )
    analysis_parsed = parse_json_from_text(analysis_raw)
    analysis_issues = validate_reinterpretations(
        analysis_parsed,
        candidate_labels,
        partial_labels,
    )
    reinterpretations = (
        analysis_parsed.get("reinterpretations", [])
        if isinstance(analysis_parsed, dict)
        else []
    )

    decision_prompt = contrastive_adjudication_prompt(
        dataset=conf["dataset"],
        partial_items=model_case["partial_items"],
        answer_options=model_case["answer_options"],
        reinterpretations=reinterpretations,
        validation_issues=analysis_issues,
    )
    decision_raw = await call_decision(
        decision_prompt,
        "contrastive reinterpretation adjudication",
    )
    decision_parsed = parse_json_from_text(decision_raw)
    decision_issues = validate_adjudication(
        decision_parsed,
        candidate_labels,
    )
    all_issues = [
        *(f"stage_1: {issue}" for issue in analysis_issues),
        *(f"stage_2: {issue}" for issue in decision_issues),
    ]
    return {
        "case": decision_case,
        "model_case": model_case,
        "llm1": {
            "prompt": analysis_prompt,
            "raw_response": analysis_raw,
            "parsed_response": analysis_parsed,
            "validation_issues": analysis_issues,
        },
        "llm2": {
            "prompt": decision_prompt,
            "raw_response": decision_raw,
            "parsed_response": decision_parsed,
            "validation_issues": decision_issues,
        },
        "validation_issues": all_issues,
        "evaluation": _evaluation(
            sample,
            candidate_labels,
            decision_parsed,
            decision_issues,
            all_issues,
        ),
    }


def aggregate_reinterpretation_evaluations(rows):
    """Aggregate ranking metrics over structurally valid two-step results."""
    valid = [
        row
        for row in (rows or [])
        if isinstance(row, dict)
        and bool(row.get("valid"))
        and not row.get("error")
    ]
    total = len(rows or [])
    valid_count = len(valid)
    ranks = [
        int(row["gt_rank"])
        for row in valid
        if row.get("gt_rank") is not None and int(row["gt_rank"]) > 0
    ]
    rank_count = len(ranks)
    hit_rates = {
        cutoff: (
            sum(rank <= cutoff for rank in ranks) / rank_count
            if rank_count
            else 0.0
        )
        for cutoff in (1, 3, 5)
    }
    rank_distribution = {}
    for rank in ranks:
        key = str(rank)
        rank_distribution[key] = rank_distribution.get(key, 0) + 1
    return {
        "completed_sample_count": total,
        "valid_sample_count": valid_count,
        "invalid_or_error_sample_count": total - valid_count,
        "valid_ranking_sample_count": rank_count,
        "hit_rate_at_1": hit_rates[1],
        "hit_rate_at_3": hit_rates[3],
        "hit_rate_at_5": hit_rates[5],
        "mean_reciprocal_rank": (
            sum(1.0 / rank for rank in ranks) / rank_count
            if rank_count
            else 0.0
        ),
        "mean_gt_rank": (
            sum(ranks) / rank_count if rank_count else 0.0
        ),
        "gt_rank_distribution": rank_distribution,
    }

"""One-call plausible-candidate set and full-ranking analysis."""

from code.common import parse_json_from_text
from code.pipeline import build_decision_case

from .prompts import direct_plausible_set_prompt
from .schemas import validate_plausible_set_result


def _model_case(decision_case):
    return {
        "partial_items": [
            {"text": item.get("text", "")}
            for item in decision_case.get("partial_items", [])
            if isinstance(item, dict)
        ],
        "answer_options": [
            {
                "label": option.get("label"),
                "text": option.get("text", ""),
            }
            for option in decision_case.get("candidates", [])
            if isinstance(option, dict)
        ],
    }


async def run_direct_plausible_set(sample, conf, call_text):
    """Run one candidate-aware call, then evaluate set coverage and GT rank."""
    decision_case = build_decision_case(sample, conf)
    model_case = _model_case(decision_case)
    prompt = direct_plausible_set_prompt(
        dataset=conf["dataset"],
        partial_items=model_case["partial_items"],
        answer_options=model_case["answer_options"],
    )
    raw = await call_text(prompt, "direct plausible-candidate set selection")
    parsed = parse_json_from_text(raw)
    labels = [
        str(option["label"])
        for option in model_case["answer_options"]
    ]
    issues = validate_plausible_set_result(parsed, labels)
    selected = (
        [
            str(candidate["label"])
            for candidate in parsed.get("plausible_candidates", [])
        ]
        if isinstance(parsed, dict) and not issues
        else []
    )
    ranking = (
        [str(label) for label in parsed.get("ranking", [])]
        if isinstance(parsed, dict) and not issues
        else []
    )
    true_label = str(sample["true_option_char"])
    candidate_count = len(labels)
    selected_count = len(selected)
    gt_rank = ranking.index(true_label) + 1 if true_label in ranking else None
    plausible_rank_top_k = ranking[:selected_count]
    plausible_ranking_consistent = (
        set(selected) == set(plausible_rank_top_k) if not issues else None
    )
    return {
        "case": decision_case,
        "model_case": model_case,
        "prompt": prompt,
        "raw_response": raw,
        "parsed_response": parsed,
        "validation_issues": issues,
        "evaluation": {
            "candidate_count": candidate_count,
            "plausible_labels": selected,
            "ranking": ranking,
            "plausible_rank_top_k": plausible_rank_top_k,
            "plausible_ranking_consistent": plausible_ranking_consistent,
            "plausible_set_size": selected_count,
            "selection_fraction": (
                selected_count / candidate_count if candidate_count else 0.0
            ),
            "random_same_size_gt_inclusion_baseline": (
                selected_count / candidate_count if candidate_count else 0.0
            ),
            "true_label": true_label,
            "gt_in_plausible_set": true_label in selected,
            "gt_rank": gt_rank,
            "reciprocal_rank": 1.0 / gt_rank if gt_rank else 0.0,
            "hit_at_1": bool(gt_rank and gt_rank <= 1),
            "hit_at_3": bool(gt_rank and gt_rank <= 3),
            "hit_at_5": bool(gt_rank and gt_rank <= 5),
            "llm_calls": 1,
        },
    }


def aggregate_plausible_set_evaluations(rows):
    """Aggregate coverage jointly with the cost of selecting larger sets."""
    valid = [
        row
        for row in (rows or [])
        if isinstance(row, dict) and not row.get("error")
    ]
    total = len(rows or [])
    valid_count = len(valid)
    coverage_count = sum(bool(row.get("gt_in_plausible_set")) for row in valid)
    average_size = (
        sum(int(row.get("plausible_set_size", 0)) for row in valid) / valid_count
        if valid_count
        else 0.0
    )
    average_fraction = (
        sum(float(row.get("selection_fraction", 0.0)) for row in valid)
        / valid_count
        if valid_count
        else 0.0
    )
    coverage = coverage_count / valid_count if valid_count else 0.0
    consistency_count = sum(
        bool(row.get("plausible_ranking_consistent")) for row in valid
    )
    consistency_rate = (
        consistency_count / valid_count if valid_count else 0.0
    )
    gt_ranks = [
        int(row["gt_rank"])
        for row in valid
        if row.get("gt_rank") is not None and int(row["gt_rank"]) > 0
    ]
    ranking_count = len(gt_ranks)
    mean_gt_rank = (
        sum(gt_ranks) / ranking_count if ranking_count else 0.0
    )
    mean_reciprocal_rank = (
        sum(1.0 / rank for rank in gt_ranks) / ranking_count
        if ranking_count
        else 0.0
    )
    hit_rates = {
        cutoff: (
            sum(rank <= cutoff for rank in gt_ranks) / ranking_count
            if ranking_count
            else 0.0
        )
        for cutoff in (1, 3, 5)
    }
    distribution = {}
    for row in valid:
        size = str(int(row.get("plausible_set_size", 0)))
        distribution[size] = distribution.get(size, 0) + 1
    rank_distribution = {}
    for rank in gt_ranks:
        key = str(rank)
        rank_distribution[key] = rank_distribution.get(key, 0) + 1
    return {
        "completed_sample_count": total,
        "valid_sample_count": valid_count,
        "error_sample_count": total - valid_count,
        "gt_coverage_count": coverage_count,
        "gt_plausible_coverage": coverage,
        "average_plausible_set_size": average_size,
        "average_selection_fraction": average_fraction,
        "random_same_size_expected_coverage": average_fraction,
        "coverage_above_random_same_size": coverage - average_fraction,
        "plausible_set_size_distribution": distribution,
        "plausible_ranking_consistency_count": consistency_count,
        "plausible_ranking_consistency_rate": consistency_rate,
        "valid_ranking_sample_count": ranking_count,
        "mean_gt_rank": mean_gt_rank,
        "mean_reciprocal_rank": mean_reciprocal_rank,
        "hit_rate_at_1": hit_rates[1],
        "hit_rate_at_3": hit_rates[3],
        "hit_rate_at_5": hit_rates[5],
        "gt_rank_distribution": rank_distribution,
    }

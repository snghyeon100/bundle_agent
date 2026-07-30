"""Rendering and evaluation helpers for spec-first strategy evidence."""


def _compact_context(context, max_chars):
    if not isinstance(context, dict):
        return None
    text = str(context.get("text") or "")
    if not text.strip():
        return None
    if max_chars > 0 and len(text) > max_chars:
        text = text[: max_chars - 1].rstrip() + "…"
    sources = context.get("sources")
    if not isinstance(sources, list):
        legacy_source = str(context.get("source") or "").strip()
        sources = [legacy_source] if legacy_source else []
    return {
        "sources": [str(source) for source in sources],
        "text": text,
    }


def build_strategy_evidence(
    *,
    specs,
    execution_reports,
    candidate_labels,
    max_contexts_per_candidate=0,
    max_context_chars=0,
):
    """Join immutable specs to candidate-specific runtime contexts."""
    report_by_id = {
        str(report.get("strategy_id")): report
        for report in execution_reports
        if isinstance(report, dict) and report.get("success")
    }
    evidence = []
    for spec in specs:
        strategy_id = str(spec.get("strategy_id") or "")
        report = report_by_id.get(strategy_id)
        if report is None:
            continue
        result_by_label = {
            str(row.get("label") or ""): row
            for row in report.get("result", [])
            if isinstance(row, dict)
        }
        candidates = []
        for label in candidate_labels:
            row = result_by_label.get(label, {})
            contexts = []
            for context in row.get("contexts", []):
                compact = _compact_context(context, int(max_context_chars))
                if compact is not None:
                    contexts.append(compact)
                if (
                    int(max_contexts_per_candidate) > 0
                    and len(contexts) >= int(max_contexts_per_candidate)
                ):
                    break
            candidates.append({"label": label, "contexts": contexts})
        evidence.append(
            {
                "strategy_id": strategy_id,
                "intent": str(spec.get("intent") or ""),
                "reference_construction": str(
                    spec.get("reference_construction") or ""
                ),
                "candidate_relation": str(spec.get("candidate_relation") or ""),
                "candidate_evidence": candidates,
            }
        )
    return evidence


def evaluate_full_ranking(prediction, true_label):
    """Evaluate one already-validated complete ranking."""
    ranking = list(prediction["ranking"])
    gt_rank = ranking.index(str(true_label)) + 1
    return {
        **prediction,
        "true_label": str(true_label),
        "hit": prediction["prediction"] == str(true_label),
        "gt_rank": gt_rank,
        "reciprocal_rank": 1.0 / gt_rank,
        "hit_at_1": gt_rank <= 1,
        "hit_at_3": gt_rank <= 3,
        "hit_at_5": gt_rank <= 5,
    }


def aggregate_prediction_rows(rows):
    """Aggregate ranking metrics over structurally valid batch rows."""
    rows = [row for row in (rows or []) if isinstance(row, dict)]
    valid = [
        row
        for row in rows
        if bool(row.get("valid"))
        and not row.get("error")
        and row.get("gt_rank") is not None
    ]
    ranks = [int(row["gt_rank"]) for row in valid]
    return {
        "completed_sample_count": len(rows),
        "valid_sample_count": len(valid),
        "invalid_or_error_sample_count": len(rows) - len(valid),
        "hit_rate_at_1": (
            sum(rank <= 1 for rank in ranks) / len(ranks) if ranks else 0.0
        ),
        "hit_rate_at_3": (
            sum(rank <= 3 for rank in ranks) / len(ranks) if ranks else 0.0
        ),
        "hit_rate_at_5": (
            sum(rank <= 5 for rank in ranks) / len(ranks) if ranks else 0.0
        ),
        "mean_reciprocal_rank": (
            sum(1.0 / rank for rank in ranks) / len(ranks) if ranks else 0.0
        ),
        "mean_gt_rank": (
            sum(ranks) / len(ranks) if ranks else 0.0
        ),
    }

"""Candidate-local evidence summarization and prediction prompts."""

from code.common import pretty_json


def _compact_item(item):
    return {
        "item_id": item.get("item_id"),
        "text": str(item.get("text") or ""),
    }


def candidate_evidence_summary_prompt(
    *,
    partial_items,
    candidate_items,
    strategy_evidence,
):
    """Ask one LLM call to summarize evidence separately for every candidate."""
    compact_evidence = []
    for evidence in strategy_evidence:
        if not isinstance(evidence, dict):
            continue
        compact_evidence.append(
            {
                "strategy_id": str(evidence.get("strategy_id") or ""),
                "intent": str(evidence.get("intent") or ""),
                "candidate_relation": str(
                    evidence.get("candidate_relation") or ""
                ),
                "candidate_evidence": evidence.get("candidate_evidence", []),
            }
        )

    summary_labels = [
        {
            "label": str(candidate.get("label") or ""),
            "summary": "concise source-grounded evidence summary",
        }
        for candidate in candidate_items
        if isinstance(candidate, dict)
    ]
    summary_input = {
        "partial_items": [
            _compact_item(item)
            for item in partial_items
            if isinstance(item, dict)
        ],
        "candidate_items": [
            {
                "label": str(item.get("label") or ""),
                **_compact_item(item),
            }
            for item in candidate_items
            if isinstance(item, dict)
        ],
        "strategy_results": compact_evidence,
    }
    return (
        "You are an Evidence Summarizer for bundle completion.\n\n"
        "A partial bundle, candidate items, and the execution results of several "
        "source-grounded strategies are given.\n\n"
        "For each candidate, summarize what the returned evidence reveals about "
        "that candidate's completion relationship to the partial bundle.\n\n"
        "Do not repeat the strategy definitions or computation steps. Focus on "
        "relationships and supporting contexts actually found by execution. Do not "
        "present evidence returned unchanged for several candidates as if it "
        "uniquely supports one candidate.\n\n"
        "If no evidence was returned, state only that no relevant evidence was "
        "found by the strategies. Absence of evidence is not evidence against a "
        "candidate.\n\n"
        "Do not choose the answer or rank the candidates. Write a concise summary "
        "of one to three sentences for every candidate.\n\n"
        "INPUT\n"
        f"{pretty_json(summary_input)}\n\n"
        "Return JSON only. Preserve the input candidate-label order and return "
        "every label exactly once:\n"
        "{\n"
        '  "candidate_summaries": '
        f"{pretty_json(summary_labels)}\n"
        "}"
    )


def validate_candidate_summaries(value, labels):
    """Validate one complete, ordered candidate-summary response."""
    if not isinstance(value, dict):
        return ["summary result must be an object"]
    if set(value) != {"candidate_summaries"}:
        return ["summary result must contain exactly candidate_summaries"]
    summaries = value.get("candidate_summaries")
    if not isinstance(summaries, list):
        return ["candidate_summaries must be a list"]

    issues = []
    if len(summaries) != len(labels):
        issues.append("candidate_summaries must contain every candidate")
    output_labels = []
    for index, summary in enumerate(summaries):
        prefix = f"candidate_summaries[{index}]"
        if not isinstance(summary, dict):
            issues.append(f"{prefix} must be an object")
            continue
        if set(summary) != {"label", "summary"}:
            issues.append(f"{prefix} must contain exactly label and summary")
        label = summary.get("label")
        text = summary.get("summary")
        if not isinstance(label, str) or not label:
            issues.append(f"{prefix}.label must be a non-empty string")
        else:
            output_labels.append(label)
        if not isinstance(text, str) or not text.strip():
            issues.append(f"{prefix}.summary must be a non-empty string")

    if output_labels != list(labels):
        issues.append(
            "candidate_summaries must preserve every input label exactly once "
            "and in order"
        )
    return list(dict.fromkeys(issues))


def candidate_summary_prediction_prompt(
    *,
    dataset,
    partial_items,
    candidate_items,
    candidate_summaries,
    evidence_mode="summary",
):
    """Place each generated evidence summary directly beside its candidate."""
    if evidence_mode == "completion_explanation":
        evidence_plural = (
            "candidate-local evidence-conditioned bundle-completion explanations"
        )
        evidence_heading = "Bundle-completion explanation"
        evidence_section = "CANDIDATES WITH BUNDLE-COMPLETION EXPLANATIONS"
        evidence_statement = (
            "An explanation describes the candidate's possible role using its "
            "item text and any additional source evidence selected by the curator."
        )
    elif evidence_mode == "interpretation":
        evidence_plural = "candidate-local source-grounded evidence interpretations"
        evidence_heading = "Evidence interpretation"
        evidence_section = "CANDIDATES WITH EVIDENCE INTERPRETATIONS"
        evidence_statement = (
            "An interpretation describes only what the curator inferred from "
            "its selected executed-strategy evidence."
        )
    else:
        evidence_plural = "candidate-local source-grounded evidence summaries"
        evidence_heading = "Source-grounded evidence summary"
        evidence_section = "CANDIDATES WITH EVIDENCE SUMMARIES"
        evidence_statement = (
            "A summary reports evidence found by the executed strategies."
        )
    name = str(dataset or "").lower()
    if "spotify" in name:
        task_name, bundle_name, item_name = (
            "playlist continuation",
            "music playlist",
            "song",
        )
    else:
        task_name, bundle_name, item_name = (
            "bundle construction",
            "fashion outfit",
            "fashion item",
        )
    partial_text = "; ".join(
        f"{index + 1}. {item.get('text', '')}"
        for index, item in enumerate(partial_items)
        if isinstance(item, dict)
    )
    summaries_by_label = {
        str(row.get("label") or ""): str(row.get("summary") or "")
        for row in candidate_summaries
        if isinstance(row, dict)
    }
    option_blocks = []
    for candidate in candidate_items:
        if not isinstance(candidate, dict):
            continue
        label = str(candidate.get("label") or "")
        option_blocks.append(
            f"[{label}]\n"
            f"Item: {candidate.get('text', '')}\n"
            f"{evidence_heading}: "
            f"{summaries_by_label.get(label, '')}"
        )

    return (
        f"You are a helpful and honest assistant. The following is a multiple choice "
        f"question about {task_name}. Rank all options from most to least plausible "
        f"using the item text and the {evidence_plural}.\n\n"
        f"Question: Given the partial {bundle_name}: {partial_text}, which candidate "
        f"{item_name} should be included into this {bundle_name}?\n\n"
        f"{evidence_section}\n\n"
        + "\n\n".join(option_blocks)
        + "\n\n"
        f"{evidence_statement} Evidence "
        "absence is not automatic contradiction, and evidence quantity alone does "
        "not determine the answer.\n\n"
        "Return JSON only. Include every option label exactly once in ranking, from "
        "most to least plausible, and make prediction equal ranking[0]. Keep the "
        "rationale to at most two sentences:\n"
        "{\n"
        '  "prediction": "top-ranked label",\n'
        '  "ranking": ["all labels exactly once"],\n'
        '  "rationale": "brief evidence-grounded comparison"\n'
        "}"
    )

"""Semantic evidence curation between program execution and prediction."""

from copy import deepcopy
import re

from code.common import pretty_json


CURATION_FIELDS = {
    "selected_strategies",
    "selection_reasons",
    "candidate_explanations",
}


def _remove_internal_ids(text):
    """Remove implementation-local IDs without discarding their semantics."""
    cleaned = re.sub(
        r"\bS\d+:[A-Za-z0-9_-]+:\d+\b",
        "the selected evidence",
        str(text or ""),
    )
    cleaned = re.sub(
        r"\bS\d+\s+evidence\b",
        "the selected evidence",
        cleaned,
        flags=re.IGNORECASE,
    )
    return re.sub(
        r"\bS\d+\b",
        "the selected program",
        cleaned,
        flags=re.IGNORECASE,
    )


def normalize_evidence_curation(value):
    """Normalize harmless formatting leakage before strict validation."""
    if not isinstance(value, dict):
        return value
    normalized = deepcopy(value)
    explanations = normalized.get("candidate_explanations")
    if isinstance(explanations, dict):
        normalized["candidate_explanations"] = {
            label: (
                _remove_internal_ids(explanation)
                if isinstance(explanation, str)
                else explanation
            )
            for label, explanation in explanations.items()
        }
    return normalized


def _compact_item(item, *, include_label=False):
    compact = {"text": str(item.get("text") or "")}
    if include_label:
        compact = {
            "label": str(item.get("label") or ""),
            **compact,
        }
    return compact


def _compact_strategy_specs(strategy_specs):
    fields = (
        "strategy_id",
        "intent",
        "description",
        "reference_construction",
        "candidate_relation",
        "evidence_route",
    )
    return [
        {
            field: deepcopy(spec.get(field))
            for field in fields
        }
        for spec in strategy_specs or []
        if isinstance(spec, dict)
    ]


def _compact_program_results(strategy_evidence):
    compact = []
    for strategy in strategy_evidence or []:
        if not isinstance(strategy, dict):
            continue
        candidate_results = {}
        for candidate in strategy.get("candidate_evidence", []):
            if not isinstance(candidate, dict):
                continue
            label = str(candidate.get("label") or "")
            candidate_results[label] = [
                {
                    "sources": [
                        str(source)
                        for source in (context.get("sources") or [])
                    ],
                    "text": str(context.get("text") or ""),
                }
                for context in (candidate.get("contexts") or [])
                if isinstance(context, dict)
                and str(context.get("text") or "").strip()
            ]
        compact.append(
            {
                "strategy_id": str(strategy.get("strategy_id") or ""),
                "candidate_results": candidate_results,
            }
        )
    return compact


def evidence_curator_prompt(
    *,
    partial_items,
    candidate_items,
    strategy_specs,
    strategy_evidence,
    max_selected_strategies=3,
):
    """Build a strategy-selection and candidate-interpretation prompt."""
    labels = [
        str(candidate.get("label") or "")
        for candidate in candidate_items
        if isinstance(candidate, dict)
    ]
    output_shape = {
        "selected_strategies": [],
        "selection_reasons": {},
        "candidate_explanations": {
            label: (
                "Explain how this item could complete the partial bundle from "
                "the item texts, and state that no additional source evidence "
                "was found when applicable."
            )
            for label in labels
        },
    }
    curator_input = {
        "partial_bundle": [
            _compact_item(item)
            for item in partial_items
            if isinstance(item, dict)
        ],
        "candidate_items": [
            _compact_item(item, include_label=True)
            for item in candidate_items
            if isinstance(item, dict)
        ],
        "strategy_specifications": _compact_strategy_specs(strategy_specs),
        "program_results": _compact_program_results(strategy_evidence),
    }
    return (
        "You are an Evidence Curator for bundle completion.\n\n"
        "Your task is not to choose the missing item or rank candidates. Review the "
        "strategy specifications and their actual program outputs, then select the "
        "strategies whose executed results are useful for distinguishing the current "
        "candidates from a bundle-completion perspective.\n\n"
        "An evidence context is useful when it is consistent with the strategy's "
        "declared reference construction and candidate relation, is semantically "
        "relevant to the partial bundle and that candidate, and reveals a relation "
        "that is not merely repeated unchanged for every candidate. A plausible "
        "strategy description is not enough when its execution output does not show "
        "the declared relation.\n\n"
        "Do not favor evidence because it appears to support a candidate that you "
        "personally consider likely to be correct. Evidence quantity is not evidence "
        "strength, and missing evidence is not evidence against a candidate. Prefer "
        "complementary strategies that expose meaningfully different relations. Do "
        "not retain semantically duplicate or non-discriminative contexts.\n\n"
        "For each candidate item, independently explain how it relates to the partial "
        "bundle and how adding it could complete the bundle. Use the partial-bundle "
        "text and candidate-item text as the primary semantic context. Incorporate the "
        "selected program evidence when it provides additional support, qualification, "
        "or limitation. Do not merely summarize retrieved records.\n\n"
        "Each explanation must be self-contained and concise, using one to three "
        "sentences. Clearly distinguish whether source evidence concerns the exact "
        "candidate, another item from its category, or only a semantically or visually "
        "similar item. Also distinguish direct historical co-occurrence from indirect "
        "or similarity-only evidence. Never claim that the exact candidate participated "
        "in a source relation unless the executed result establishes that fact. Do not "
        "mention strategy IDs, candidate labels as technical identifiers, evidence IDs, "
        "item IDs, bundle IDs, or other internal identifiers.\n\n"
        "Do not state that a candidate is correct, likely, better, or worse, and do "
        "not compare or rank candidates. If the selected strategies provide no useful "
        "evidence for a candidate, still explain its possible bundle-completion role "
        "from the item texts and explicitly state that no additional source evidence "
        "was available. Absence of evidence is not negative evidence. It is valid to "
        "select fewer strategies than the budget, including none when no execution "
        "result is useful. Do not default to empty selection when useful evidence "
        "exists.\n\n"
        "BUDGET\n"
        f"- Maximum selected strategies: {int(max_selected_strategies)}\n"
        "\n"
        "INPUT\n"
        f"{pretty_json(curator_input)}\n\n"
        "OUTPUT\n"
        "Return JSON only with exactly the following three fields. Preserve every "
        "candidate label exactly once and in input order. selected_strategies must "
        "contain only strategy IDs present in the input. selection_reasons must have "
        "exactly the selected strategy IDs as keys. candidate_explanations must map "
        "each candidate label to one non-empty standalone bundle-completion "
        "explanation string.\n"
        f"{pretty_json(output_shape)}"
    )


def _strategy_catalog(strategy_evidence):
    strategy_ids = []
    context_counts = {}
    for strategy in strategy_evidence or []:
        if not isinstance(strategy, dict):
            continue
        strategy_id = str(strategy.get("strategy_id") or "")
        if not strategy_id:
            continue
        strategy_ids.append(strategy_id)
        context_counts[strategy_id] = 0
        for candidate in strategy.get("candidate_evidence", []):
            if not isinstance(candidate, dict):
                continue
            context_counts[strategy_id] += sum(
                1
                for context in (candidate.get("contexts") or [])
                if isinstance(context, dict)
                and str(context.get("text") or "").strip()
            )
    return strategy_ids, context_counts


def validate_evidence_curation(
    value,
    *,
    strategy_evidence,
    candidate_labels,
    max_selected_strategies=3,
):
    """Validate strategy selection and standalone completion explanations."""
    if not isinstance(value, dict):
        return ["curation result must be an object"]
    if set(value) != CURATION_FIELDS:
        return [
            "curation result must contain exactly selected_strategies, "
            "selection_reasons, and candidate_explanations"
        ]

    available_strategy_ids, context_counts = _strategy_catalog(strategy_evidence)
    available_strategy_set = set(available_strategy_ids)
    labels = [str(label) for label in candidate_labels]
    issues = []

    selected = value.get("selected_strategies")
    if not isinstance(selected, list):
        issues.append("selected_strategies must be a list")
        selected = []
    valid_selected = [
        strategy_id
        for strategy_id in selected
        if isinstance(strategy_id, str)
    ]
    if len(valid_selected) != len(selected):
        issues.append("selected_strategies must contain only strings")
    if len(valid_selected) != len(set(valid_selected)):
        issues.append("selected_strategies must not contain duplicates")
    if len(selected) > int(max_selected_strategies):
        issues.append(
            "selected_strategies exceeds the configured selection budget"
        )
    unavailable = [
        strategy_id
        for strategy_id in valid_selected
        if strategy_id not in available_strategy_set
    ]
    if unavailable:
        issues.append(
            "selected_strategies contains unavailable IDs: "
            + ", ".join(unavailable)
        )
    empty_selected = [
        strategy_id
        for strategy_id in valid_selected
        if context_counts.get(strategy_id, 0) == 0
    ]
    if empty_selected:
        issues.append(
            "selected_strategies contains strategies with no executed "
            "evidence contexts: " + ", ".join(empty_selected)
        )

    reasons = value.get("selection_reasons")
    if not isinstance(reasons, dict):
        issues.append("selection_reasons must be an object")
        reasons = {}
    if set(reasons) != set(valid_selected):
        issues.append(
            "selection_reasons keys must exactly match selected_strategies"
        )
    for strategy_id, reason in reasons.items():
        if not isinstance(reason, str) or not reason.strip():
            issues.append(
                f"selection_reasons[{strategy_id}] must be a non-empty string"
            )

    explanations = value.get("candidate_explanations")
    if not isinstance(explanations, dict):
        issues.append("candidate_explanations must be an object")
        explanations = {}
    if list(explanations) != labels:
        issues.append(
            "candidate_explanations must preserve every input label "
            "exactly once and in order"
        )
    for label in labels:
        explanation = explanations.get(label)
        if not isinstance(explanation, str) or not explanation.strip():
            issues.append(
                f"candidate_explanations[{label}] must be a non-empty string"
            )
            continue
        if re.search(r"\bS\d+\b", explanation):
            issues.append(
                f"candidate_explanations[{label}] must not mention "
                "strategy or evidence IDs"
            )
    return list(dict.fromkeys(issues))


def select_curated_strategy_evidence(
    strategy_evidence,
    curation,
):
    """Retain complete raw results for selected strategies as an audit artifact."""
    selected = list(curation.get("selected_strategies", []))
    reasons = dict(curation.get("selection_reasons", {}))
    strategy_by_id = {
        str(strategy.get("strategy_id") or ""): strategy
        for strategy in strategy_evidence or []
        if isinstance(strategy, dict)
    }
    selected_evidence = []
    for strategy_id in selected:
        source = strategy_by_id[strategy_id]
        strategy = deepcopy(source)
        strategy["selection_reason"] = str(reasons.get(strategy_id) or "")
        selected_evidence.append(strategy)
    return selected_evidence


def candidate_completion_explanations(curation):
    """Convert validated candidate explanations to prediction-prompt rows."""
    return [
        {
            "label": str(label),
            "summary": str(explanation or ""),
        }
        for label, explanation in curation.get(
            "candidate_explanations",
            {},
        ).items()
    ]

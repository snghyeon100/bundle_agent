"""Validation for the two counterfactual reinterpretation stages."""


REINTERPRETATION_FIELDS = {
    "label",
    "completed_set_interpretation",
    "partial_member_contributions",
    "candidate_contribution",
    "role_closure",
    "counterfactual_necessity",
    "conflicts_or_redundancies",
}

PARTIAL_CONTRIBUTION_FIELDS = {"partial_label", "contribution"}

DECISION_FIELDS = {
    "ranking",
    "prediction",
    "decisive_comparison",
    "decision_basis",
}

DECISION_BASIS_FIELDS = {
    "explanatory_coverage",
    "role_closure",
    "counterfactual_necessity",
    "conflict_or_redundancy",
}


def _non_empty_string(value):
    return isinstance(value, str) and bool(value.strip())


def _dedupe(issues):
    return list(dict.fromkeys(issues))


def validate_reinterpretations(value, candidate_labels, partial_labels):
    """Require one structurally complete reinterpretation per candidate."""
    if not isinstance(value, dict):
        return ["result must be an object"]
    if set(value) != {"reinterpretations"}:
        return ["result must contain exactly reinterpretations"]
    entries = value.get("reinterpretations")
    if not isinstance(entries, list):
        return ["reinterpretations must be a list"]

    issues = []
    allowed_candidates = set(candidate_labels or [])
    expected_partials = set(partial_labels or [])
    observed_candidates = []
    for index, entry in enumerate(entries):
        prefix = f"reinterpretations[{index}]"
        if not isinstance(entry, dict):
            issues.append(f"{prefix} must be an object")
            continue
        if set(entry) != REINTERPRETATION_FIELDS:
            issues.append(
                f"{prefix} must contain exactly: "
                + ", ".join(sorted(REINTERPRETATION_FIELDS))
            )
        label = entry.get("label")
        if label not in allowed_candidates:
            issues.append(f"{prefix}.label must be a supplied answer-option label")
        else:
            observed_candidates.append(label)
        for field in (
            "completed_set_interpretation",
            "candidate_contribution",
            "role_closure",
            "counterfactual_necessity",
        ):
            if not _non_empty_string(entry.get(field)):
                issues.append(f"{prefix}.{field} must be a non-empty string")

        contributions = entry.get("partial_member_contributions")
        if not isinstance(contributions, list):
            issues.append(f"{prefix}.partial_member_contributions must be a list")
        else:
            observed_partials = []
            for member_index, contribution in enumerate(contributions):
                member_prefix = (
                    f"{prefix}.partial_member_contributions[{member_index}]"
                )
                if not isinstance(contribution, dict):
                    issues.append(f"{member_prefix} must be an object")
                    continue
                if set(contribution) != PARTIAL_CONTRIBUTION_FIELDS:
                    issues.append(
                        f"{member_prefix} must contain exactly partial_label and "
                        "contribution"
                    )
                partial_label = contribution.get("partial_label")
                if partial_label not in expected_partials:
                    issues.append(
                        f"{member_prefix}.partial_label must be a supplied partial label"
                    )
                else:
                    observed_partials.append(partial_label)
                if not _non_empty_string(contribution.get("contribution")):
                    issues.append(
                        f"{member_prefix}.contribution must be a non-empty string"
                    )
            if len(observed_partials) != len(set(observed_partials)):
                issues.append(f"{prefix} partial labels must be unique")
            if set(observed_partials) != expected_partials:
                issues.append(
                    f"{prefix} must explain every supplied partial member exactly once"
                )

        conflicts = entry.get("conflicts_or_redundancies")
        if not isinstance(conflicts, list) or any(
            not _non_empty_string(issue) for issue in conflicts
        ):
            issues.append(
                f"{prefix}.conflicts_or_redundancies must be a list of non-empty strings"
            )

    if len(observed_candidates) != len(set(observed_candidates)):
        issues.append("reinterpretation candidate labels must be unique")
    if set(observed_candidates) != allowed_candidates:
        issues.append(
            "reinterpretations must contain every supplied candidate label exactly once"
        )
    return _dedupe(issues)


def validate_adjudication(value, candidate_labels):
    """Require a complete ranking and a prediction consistent with rank one."""
    if not isinstance(value, dict):
        return ["result must be an object"]
    if set(value) != DECISION_FIELDS:
        return [
            "result must contain exactly ranking, prediction, decisive_comparison, "
            "and decision_basis"
        ]
    issues = []
    labels = list(candidate_labels or [])
    allowed = set(labels)
    ranking = value.get("ranking")
    if not isinstance(ranking, list):
        issues.append("ranking must be a list")
        ranking = []
    else:
        if any(not isinstance(label, str) for label in ranking):
            issues.append("ranking entries must be strings")
        if len(ranking) != len(set(ranking)):
            issues.append("ranking labels must be unique")
        if len(ranking) != len(labels) or set(ranking) != allowed:
            issues.append(
                "ranking must contain every supplied answer-option label exactly once"
            )

    prediction = value.get("prediction")
    if prediction not in allowed:
        issues.append("prediction must be a supplied answer-option label")
    if ranking and prediction != ranking[0]:
        issues.append("prediction must equal the first ranking label")
    if not _non_empty_string(value.get("decisive_comparison")):
        issues.append("decisive_comparison must be a non-empty string")

    basis = value.get("decision_basis")
    if not isinstance(basis, dict) or set(basis) != DECISION_BASIS_FIELDS:
        issues.append(
            "decision_basis must contain exactly explanatory_coverage, role_closure, "
            "counterfactual_necessity, and conflict_or_redundancy"
        )
    else:
        for field in sorted(DECISION_BASIS_FIELDS):
            if not _non_empty_string(basis.get(field)):
                issues.append(f"decision_basis.{field} must be a non-empty string")
    return _dedupe(issues)

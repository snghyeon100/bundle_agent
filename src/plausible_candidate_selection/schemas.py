"""Output validation for plausible-set selection plus full ranking."""


PLAUSIBLE_CANDIDATE_FIELDS = {
    "label",
    "completion_hypothesis",
    "reason",
}


def _non_empty_string(value):
    return isinstance(value, str) and bool(value.strip())


def validate_plausible_set_result(value, candidate_labels):
    """Validate an unconstrained label subset and a complete candidate ranking."""
    if not isinstance(value, dict):
        return ["result must be an object"]
    if set(value) != {"plausible_candidates", "ranking"}:
        return ["result must contain exactly plausible_candidates and ranking"]
    candidates = value.get("plausible_candidates")
    if not isinstance(candidates, list):
        return ["plausible_candidates must be a list"]

    issues = []
    labels = list(candidate_labels or [])
    allowed = set(labels)
    selected = []
    for index, candidate in enumerate(candidates):
        prefix = f"plausible_candidates[{index}]"
        if not isinstance(candidate, dict):
            issues.append(f"{prefix} must be an object")
            continue
        if set(candidate) != PLAUSIBLE_CANDIDATE_FIELDS:
            issues.append(
                f"{prefix} must contain exactly: "
                + ", ".join(sorted(PLAUSIBLE_CANDIDATE_FIELDS))
            )
        label = candidate.get("label")
        if label not in allowed:
            issues.append(f"{prefix}.label must be a supplied answer-option label")
        else:
            selected.append(label)
        for field in ("completion_hypothesis", "reason"):
            if not _non_empty_string(candidate.get(field)):
                issues.append(f"{prefix}.{field} must be a non-empty string")
    if len(selected) != len(set(selected)):
        issues.append("plausible candidate labels must be unique")

    ranking = value.get("ranking")
    if not isinstance(ranking, list):
        issues.append("ranking must be a list")
    else:
        if any(not isinstance(label, str) for label in ranking):
            issues.append("ranking entries must be answer-option label strings")
        if len(ranking) != len(labels):
            issues.append("ranking must contain every supplied answer-option label")
        if len(ranking) != len(set(ranking)):
            issues.append("ranking labels must be unique")
        if set(ranking) != allowed:
            issues.append(
                "ranking must contain every supplied answer-option label exactly once"
            )
    return list(dict.fromkeys(issues))

"""Deterministic validation for compact, composable semantic operators."""

from copy import deepcopy


OPERATOR_FIELDS = (
    "name",
    "objective",
    "input",
    "operation",
    "output",
    "sources",
)


def _non_empty_string(value):
    return isinstance(value, str) and bool(value.strip())


def _string_list(value, *, allow_empty=False):
    return (
        isinstance(value, list)
        and (allow_empty or bool(value))
        and all(_non_empty_string(item) for item in value)
    )


def _normalized_text(value):
    return "".join(character for character in str(value or "").lower() if character.isalnum())


def validate_operator(
    operator,
    *,
    require_provenance=False,
    allowed_source_names=None,
):
    """Return deterministic contract issues for one generated operator."""
    if not isinstance(operator, dict):
        return ["operator must be an object"]
    issues = []
    required = set(OPERATOR_FIELDS)
    if require_provenance:
        required.add("derived_from")
    actual = set(operator)
    missing = sorted(required - actual)
    extra = sorted(actual - required - {"operator_id", "origin_case_id"})
    if missing:
        issues.append("missing operator fields: " + ", ".join(missing))
    if extra:
        issues.append("unsupported operator fields: " + ", ".join(extra))

    for field in ("name", "objective", "input", "operation", "output"):
        if not _non_empty_string(operator.get(field)):
            issues.append(f"operator.{field} must be a non-empty string")
    sources = operator.get("sources")
    if not _string_list(sources, allow_empty=True):
        issues.append("operator.sources must be a string list")
    else:
        if len(sources) != len(set(sources)):
            issues.append("operator.sources values must be unique")
        allowed = set(allowed_source_names or [])
        unknown = sorted(set(sources) - allowed) if allowed else []
        if unknown:
            issues.append(
                "operator.sources contains unavailable capabilities: "
                + ", ".join(unknown)
            )

    deployable_text = " ".join(
        str(operator.get(field, ""))
        for field in ("name", "objective", "input", "operation", "output")
    )
    normalized_deployable = _normalized_text(deployable_text)
    if any(
        marker in normalized_deployable
        for marker in ("groundtruth", "goldlabel", "correctanswer", "trueanswer")
    ):
        issues.append(
            "operator must be deployable without ground truth or the correct answer"
        )

    normalized_output = _normalized_text(operator.get("output"))
    if any(
        marker in normalized_output
        for marker in ("rank", "prediction", "candidatechoice", "finalanswer")
    ):
        issues.append(
            "operator.output must describe an intermediate artifact, not a rank "
            "or final prediction"
        )
    if any(
        marker in normalized_output
        for marker in ("score", "similarityvalue", "compatibilityvalue")
    ):
        issues.append("operator.output must not be a score-only artifact")

    if require_provenance and "derived_from" in operator and not _string_list(
        operator.get("derived_from")
    ):
        issues.append("operator.derived_from must be a non-empty string list")
    return issues


def operator_connection_diagnostics(operators):
    """Report exact flat-interface links without making them validity constraints."""
    interfaces = []
    for index, operator in enumerate(operators or []):
        if not isinstance(operator, dict):
            continue
        interfaces.append(
            {
                "index": index,
                "name": operator.get("name"),
                "input": operator.get("input"),
                "output": operator.get("output"),
            }
        )

    exact_connections = [
        {
            "producer_index": producer["index"],
            "producer": producer["name"],
            "consumer_index": consumer["index"],
            "consumer": consumer["name"],
            "artifact": producer["output"],
        }
        for producer in interfaces
        for consumer in interfaces
        if producer["index"] != consumer["index"]
        and _non_empty_string(producer["output"])
        and _normalized_text(producer["output"])
        == _normalized_text(consumer["input"])
    ]
    return {
        "exact_interface_connection_count": len(exact_connections),
        "exact_interface_connections": exact_connections,
        "interpretation": (
            "Exact text matches are descriptive only. A later composer or "
            "code-generation agent may connect semantically compatible interfaces."
        ),
    }


def validate_induction_result(
    value,
    expected_count=None,
    *,
    allowed_source_names=None,
):
    if not isinstance(value, dict):
        return ["induction result must be an object"]
    operators = value.get("operators")
    if not isinstance(operators, list):
        return ["operators must be a list"]

    issues = []
    if expected_count is not None and len(operators) != int(expected_count):
        issues.append(f"expected exactly {int(expected_count)} operators")
    for index, operator in enumerate(operators):
        issues.extend(
            f"operators[{index}]: {issue}"
            for issue in validate_operator(
                operator,
                allowed_source_names=allowed_source_names,
            )
        )

    names = [
        operator.get("name")
        for operator in operators
        if isinstance(operator, dict) and _non_empty_string(operator.get("name"))
    ]
    if len(names) != len(set(names)):
        issues.append("operator names must be unique within a case")

    return issues


def validate_operator_library(value, *, allowed_source_names=None):
    if not isinstance(value, dict):
        return ["operator library must be an object"]
    issues = []
    if value.get("schema_version") != "compact_operator_library_v1":
        issues.append("schema_version must be compact_operator_library_v1")
    if not _non_empty_string(value.get("dataset")):
        issues.append("dataset must be a non-empty string")

    clusters = value.get("clusters")
    if not isinstance(clusters, list) or not clusters:
        issues.append("clusters must be a non-empty list")
        clusters = []
    operators = value.get("operators")
    if not isinstance(operators, list) or not operators:
        issues.append("operators must be a non-empty list")
        operators = []

    representatives = []
    all_member_ids = []
    cluster_by_representative = {}
    for index, cluster in enumerate(clusters):
        if not isinstance(cluster, dict):
            issues.append(f"clusters[{index}] must be an object")
            continue
        for field in ("name", "rationale", "representative"):
            if not _non_empty_string(cluster.get(field)):
                issues.append(f"clusters[{index}].{field} must be a non-empty string")
        member_ids = cluster.get("member_ids")
        if not _string_list(member_ids):
            issues.append(f"clusters[{index}].member_ids must be a non-empty string list")
        else:
            all_member_ids.extend(member_ids)
        representative = cluster.get("representative")
        if _non_empty_string(representative):
            representatives.append(representative)
            cluster_by_representative[representative] = set(member_ids or [])

    for index, operator in enumerate(operators):
        issues.extend(
            f"operators[{index}]: {issue}"
            for issue in validate_operator(
                operator,
                require_provenance=True,
                allowed_source_names=allowed_source_names,
            )
        )

    names = [
        operator.get("name")
        for operator in operators
        if isinstance(operator, dict) and _non_empty_string(operator.get("name"))
    ]
    if len(names) != len(set(names)):
        issues.append("refined operator names must be unique")
    if len(representatives) != len(set(representatives)):
        issues.append("cluster representative names must be unique")
    if set(names) != set(representatives):
        missing = sorted(set(names) - set(representatives))
        extra = sorted(set(representatives) - set(names))
        if missing:
            issues.append(
                "operators without a matching cluster representative: "
                + ", ".join(missing)
            )
        if extra:
            issues.append(
                "cluster representatives without a matching operator: "
                + ", ".join(extra)
            )
    if len(all_member_ids) != len(set(all_member_ids)):
        issues.append("cluster member_ids must not appear in multiple clusters")

    for operator in operators:
        if not isinstance(operator, dict):
            continue
        name = operator.get("name")
        derived = operator.get("derived_from")
        if (
            _non_empty_string(name)
            and _string_list(derived)
            and name in cluster_by_representative
            and set(derived) != cluster_by_representative[name]
        ):
            issues.append(
                f"operator {name}.derived_from must match its cluster member_ids"
            )
    return issues


def normalize_library(value, dataset):
    """Attach the local envelope and remove unsupported implementation fields."""
    result = deepcopy(value) if isinstance(value, dict) else {}
    result["schema_version"] = "compact_operator_library_v1"
    result["dataset"] = str(dataset)
    result.setdefault("clusters", [])
    result.setdefault("operators", [])
    if isinstance(result["operators"], list):
        allowed = (*OPERATOR_FIELDS, "derived_from")
        result["operators"] = [
            {field: operator[field] for field in allowed if field in operator}
            if isinstance(operator, dict)
            else operator
            for operator in result["operators"]
        ]
    return result


def operator_names(library):
    return {
        operator.get("name")
        for operator in library.get("operators", [])
        if isinstance(operator, dict) and _non_empty_string(operator.get("name"))
    }


def validate_workflow_result(value, library, expected_count=None):
    if not isinstance(value, dict):
        return ["workflow result must be an object"]
    issues = []
    if value.get("schema_version") != "operator_workflow_candidates_v1":
        issues.append("schema_version must be operator_workflow_candidates_v1")
    if not _non_empty_string(value.get("intent")):
        issues.append("intent must be a non-empty string")
    workflows = value.get("workflows")
    if not isinstance(workflows, list):
        return issues + ["workflows must be a list"]
    if expected_count is not None and len(workflows) != int(expected_count):
        issues.append(f"expected exactly {int(expected_count)} workflows")

    allowed = operator_names(library)
    workflow_names = []
    valid_sequences = []
    for index, workflow in enumerate(workflows):
        prefix = f"workflows[{index}]"
        if not isinstance(workflow, dict):
            issues.append(f"{prefix} must be an object")
            continue
        name = workflow.get("name")
        if not _non_empty_string(name):
            issues.append(f"{prefix}.name must be a non-empty string")
        else:
            workflow_names.append(name)
        for field in ("rationale", "source_plan"):
            if not _non_empty_string(workflow.get(field)):
                issues.append(f"{prefix}.{field} must be a non-empty string")

        sequence = workflow.get("operator_sequence")
        sequence_valid = (
            isinstance(sequence, list)
            and 2 <= len(sequence) <= 4
            and all(_non_empty_string(item) for item in sequence)
        )
        if not isinstance(sequence, list) or not 2 <= len(sequence) <= 4:
            issues.append(f"{prefix}.operator_sequence must contain 2 to 4 names")
            sequence = []
        elif not all(_non_empty_string(item) for item in sequence):
            issues.append(f"{prefix}.operator_sequence must be a string list")
            sequence = []
        if sequence_valid:
            unknown = sorted(set(sequence) - allowed)
            if unknown:
                issues.append(
                    f"{prefix} uses unknown operators: " + ", ".join(unknown)
                )
            valid_sequences.append(tuple(sequence))

        steps = workflow.get("steps")
        if not isinstance(steps, list) or len(steps) != len(sequence):
            issues.append(f"{prefix}.steps must align one-to-one with operator_sequence")
            steps = []
        for step_index, step in enumerate(steps):
            if not isinstance(step, dict):
                issues.append(f"{prefix}.steps[{step_index}] must be an object")
                continue
            if set(step) != {"operator", "objective", "input", "output"}:
                issues.append(
                    f"{prefix}.steps[{step_index}] must contain exactly "
                    "operator, objective, input, output"
                )
            if step_index < len(sequence) and step.get("operator") != sequence[step_index]:
                issues.append(f"{prefix}.steps[{step_index}].operator does not match sequence")
            for field in ("operator", "objective", "input", "output"):
                if not _non_empty_string(step.get(field)):
                    issues.append(f"{prefix}.steps[{step_index}].{field} must be non-empty")

    if len(workflow_names) != len(set(workflow_names)):
        issues.append("workflow names must be unique")
    if len(valid_sequences) != len(set(valid_sequences)):
        issues.append("operator sequences must be structurally distinct")
    recommended = value.get("recommended_workflow")
    if recommended not in workflow_names:
        issues.append("recommended_workflow must name one generated workflow")
    return issues

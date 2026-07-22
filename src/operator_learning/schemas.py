"""Deterministic schemas for the rank-free operator MVP."""

from copy import deepcopy


OPERATOR_FIELDS = (
    "name",
    "purpose",
    "anchor",
    "inputs",
    "sources",
    "relation_path",
    "operation",
    "output",
    "when_useful",
)


def _non_empty_string(value):
    return isinstance(value, str) and bool(value.strip())


def _string_list(value):
    return (
        isinstance(value, list)
        and bool(value)
        and all(_non_empty_string(item) for item in value)
    )


def validate_operator(operator, *, require_provenance=False):
    """Return all deterministic validation issues for one operator."""
    if not isinstance(operator, dict):
        return ["operator must be an object"]
    issues = []
    required = set(OPERATOR_FIELDS)
    if require_provenance:
        required.add("derived_from")
    missing = sorted(required - set(operator))
    if missing:
        issues.append("missing operator fields: " + ", ".join(missing))
    for field in (
        "name",
        "purpose",
        "anchor",
        "relation_path",
        "operation",
        "output",
        "when_useful",
    ):
        if field in operator and not _non_empty_string(operator.get(field)):
            issues.append(f"operator.{field} must be a non-empty string")
    for field in ("inputs", "sources"):
        if field in operator and not _string_list(operator.get(field)):
            issues.append(f"operator.{field} must be a non-empty string list")
    if require_provenance and "derived_from" in operator and not _string_list(
        operator.get("derived_from")
    ):
        issues.append("operator.derived_from must be a non-empty string list")
    return issues


def validate_induction_result(value, expected_count=None):
    if not isinstance(value, dict):
        return ["induction result must be an object"]
    issues = []
    operators = value.get("operators")
    if not isinstance(operators, list):
        return ["operators must be a list"]
    if expected_count is not None and len(operators) != int(expected_count):
        issues.append(f"expected exactly {int(expected_count)} operators")
    for index, operator in enumerate(operators):
        issues.extend(f"operators[{index}]: {issue}" for issue in validate_operator(operator))
    names = [operator.get("name") for operator in operators if isinstance(operator, dict)]
    if len(names) != len(set(names)):
        issues.append("operator names must be unique within a case")
    return issues


def validate_operator_library(value):
    if not isinstance(value, dict):
        return ["operator library must be an object"]
    issues = []
    if value.get("schema_version") != "retrieval_operator_library_v1":
        issues.append("schema_version must be retrieval_operator_library_v1")
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
    for index, cluster in enumerate(clusters):
        if not isinstance(cluster, dict):
            issues.append(f"clusters[{index}] must be an object")
            continue
        for field in ("name", "rationale", "representative"):
            if not _non_empty_string(cluster.get(field)):
                issues.append(f"clusters[{index}].{field} must be a non-empty string")
        if not _string_list(cluster.get("member_ids")):
            issues.append(f"clusters[{index}].member_ids must be a non-empty string list")
    for index, operator in enumerate(operators):
        issues.extend(
            f"operators[{index}]: {issue}"
            for issue in validate_operator(operator, require_provenance=True)
        )
    names = [operator.get("name") for operator in operators if isinstance(operator, dict)]
    if len(names) != len(set(names)):
        issues.append("refined operator names must be unique")
    representatives = {
        cluster.get("representative") for cluster in clusters if isinstance(cluster, dict)
    }
    missing_representatives = sorted(set(names) - representatives)
    if missing_representatives:
        issues.append(
            "operators without a matching cluster representative: "
            + ", ".join(str(name) for name in missing_representatives)
        )
    return issues


def normalize_library(value, dataset):
    """Attach the local envelope while retaining only schema fields from the LLM result."""
    result = deepcopy(value) if isinstance(value, dict) else {}
    result["schema_version"] = "retrieval_operator_library_v1"
    result["dataset"] = str(dataset)
    result.setdefault("clusters", [])
    result.setdefault("operators", [])
    return result


def operator_names(library):
    return {
        str(operator.get("name"))
        for operator in library.get("operators", [])
        if isinstance(operator, dict) and operator.get("name")
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
    sequences = []
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
        if not isinstance(sequence, list) or not 2 <= len(sequence) <= 4:
            issues.append(f"{prefix}.operator_sequence must contain 2 to 4 names")
            sequence = []
        elif not all(_non_empty_string(name) for name in sequence):
            issues.append(f"{prefix}.operator_sequence must be a string list")
        unknown = sorted(set(sequence) - allowed)
        if unknown:
            issues.append(f"{prefix} uses unknown operators: " + ", ".join(unknown))
        sequences.append(tuple(sequence))
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
                    f"{prefix}.steps[{step_index}] must contain exactly operator, objective, input, output"
                )
            if step_index < len(sequence) and step.get("operator") != sequence[step_index]:
                issues.append(f"{prefix}.steps[{step_index}].operator does not match sequence")
            for field in ("operator", "objective", "input", "output"):
                if not _non_empty_string(step.get(field)):
                    issues.append(f"{prefix}.steps[{step_index}].{field} must be non-empty")
    if len(workflow_names) != len(set(workflow_names)):
        issues.append("workflow names must be unique")
    if len(sequences) != len(set(sequences)):
        issues.append("operator sequences must be structurally distinct")
    recommended = value.get("recommended_workflow")
    if recommended not in workflow_names:
        issues.append("recommended_workflow must name one generated workflow")
    return issues

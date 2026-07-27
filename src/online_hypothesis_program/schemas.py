"""Schemas for online semantic hypotheses and case-conditioned Python programs."""

from copy import deepcopy

from operator_learning.runtime import validate_program_source


DISCOVERY_SCHEMA_VERSION = "online_hypothesis_programs_v1"
PROGRAM_RESULT_FIELDS = {
    "candidate_proposals",
    "evidence_records",
    "used_sources",
}
HYPOTHESIS_FIELDS = {
    "id",
    "observed_cues",
    "intent",
    "missing_role",
}
PROGRAM_FIELDS = {
    "hypothesis_id",
    "program_id",
    "name",
    "required_sources",
    "evidence_types",
    "code",
}


def _non_empty_string(value):
    return isinstance(value, str) and bool(value.strip())


def _string_list(value, *, allow_empty=False):
    return (
        isinstance(value, list)
        and (allow_empty or bool(value))
        and all(_non_empty_string(item) for item in value)
    )


def hypothesis_statement(hypothesis):
    """Return the exact hypothesis string stored in execution artifacts."""
    if not isinstance(hypothesis, dict):
        return ""
    intent = str(hypothesis.get("intent") or "").strip()
    missing_role = str(hypothesis.get("missing_role") or "").strip()
    if not intent:
        return ""
    return (
        f"{intent} Plausible missing contribution: {missing_role}"
        if missing_role
        else intent
    )


def validate_online_program_source(code):
    """Apply the shared boundary checks plus online-execution restrictions."""
    issues = list(validate_program_source(code))
    if not isinstance(code, str) or not code.strip():
        return issues

    import ast

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return issues

    forbidden_calls = {
        "breakpoint",
        "delattr",
        "dir",
        "getattr",
        "hasattr",
        "help",
        "memoryview",
        "setattr",
        "type",
        "vars",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            issues.append("generated code must not define classes")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            issues.append("generated code must not use global or nonlocal declarations")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            issues.append("generated code must not access private attributes")
        elif isinstance(node, ast.Name) and node.id.startswith("_"):
            issues.append("generated code must not access private names")
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in forbidden_calls
        ):
            issues.append(f"forbidden call: {node.func.id}")
    return list(dict.fromkeys(issues))


def validate_discovery_result(value, *, available_sources, max_hypotheses):
    """Validate one candidate-blind LLM1 response."""
    if not isinstance(value, dict):
        return ["LLM1 result must be an object"]
    expected = {"schema_version", "hypotheses", "programs"}
    if set(value) != expected:
        return [
            "LLM1 result must contain exactly: " + ", ".join(sorted(expected))
        ]

    issues = []
    if value.get("schema_version") != DISCOVERY_SCHEMA_VERSION:
        issues.append(f"schema_version must be {DISCOVERY_SCHEMA_VERSION}")
    hypotheses = value.get("hypotheses")
    programs = value.get("programs")
    if not isinstance(hypotheses, list):
        issues.append("hypotheses must be a list")
        hypotheses = []
    if not isinstance(programs, list):
        issues.append("programs must be a list")
        programs = []
    if not 1 <= len(hypotheses) <= int(max_hypotheses):
        issues.append(
            f"hypotheses must contain between 1 and {int(max_hypotheses)} entries"
        )
    if len(hypotheses) != len(programs):
        issues.append("hypotheses and programs must correspond one-to-one")

    hypothesis_by_id = {}
    for index, hypothesis in enumerate(hypotheses):
        prefix = f"hypotheses[{index}]"
        if not isinstance(hypothesis, dict):
            issues.append(f"{prefix} must be an object")
            continue
        if set(hypothesis) != HYPOTHESIS_FIELDS:
            issues.append(
                f"{prefix} must contain exactly: "
                + ", ".join(sorted(HYPOTHESIS_FIELDS))
            )
        hypothesis_id = hypothesis.get("id")
        if not _non_empty_string(hypothesis_id):
            issues.append(f"{prefix}.id must be a non-empty string")
        elif hypothesis_id in hypothesis_by_id:
            issues.append(f"{prefix}.id must be unique")
        else:
            hypothesis_by_id[hypothesis_id] = hypothesis
        cues = hypothesis.get("observed_cues")
        if not _string_list(cues):
            issues.append(f"{prefix}.observed_cues must be a non-empty string list")
        elif not 1 <= len(cues) <= 4:
            issues.append(f"{prefix}.observed_cues must contain 1 to 4 cues")
        elif len(cues) != len(set(cues)):
            issues.append(f"{prefix}.observed_cues values must be unique")
        for field in ("intent", "missing_role"):
            if not _non_empty_string(hypothesis.get(field)):
                issues.append(f"{prefix}.{field} must be a non-empty string")

    allowed_sources = set(available_sources or [])
    used_hypotheses = []
    program_ids = []
    names = []
    for index, program in enumerate(programs):
        prefix = f"programs[{index}]"
        if not isinstance(program, dict):
            issues.append(f"{prefix} must be an object")
            continue
        if set(program) != PROGRAM_FIELDS:
            issues.append(
                f"{prefix} must contain exactly: "
                + ", ".join(sorted(PROGRAM_FIELDS))
            )
        for field in ("hypothesis_id", "program_id", "name"):
            if not _non_empty_string(program.get(field)):
                issues.append(f"{prefix}.{field} must be a non-empty string")
        hypothesis_id = program.get("hypothesis_id")
        used_hypotheses.append(hypothesis_id)
        if hypothesis_id not in hypothesis_by_id:
            issues.append(f"{prefix}.hypothesis_id must reference hypotheses")
        program_ids.append(program.get("program_id"))
        names.append(program.get("name"))

        required_sources = program.get("required_sources")
        if not _string_list(required_sources):
            issues.append(f"{prefix}.required_sources must be a non-empty string list")
        else:
            unknown = sorted(set(required_sources) - allowed_sources)
            if unknown:
                issues.append(
                    f"{prefix}.required_sources contains unavailable sources: "
                    + ", ".join(unknown)
                )
            if len(required_sources) != len(set(required_sources)):
                issues.append(f"{prefix}.required_sources values must be unique")
        evidence_types = program.get("evidence_types")
        if not _string_list(evidence_types):
            issues.append(f"{prefix}.evidence_types must be a non-empty string list")
        elif len(evidence_types) != len(set(evidence_types)):
            issues.append(f"{prefix}.evidence_types values must be unique")
        issues.extend(
            f"{prefix}: {issue}"
            for issue in validate_online_program_source(program.get("code"))
        )

    if len(used_hypotheses) != len(set(used_hypotheses)):
        issues.append("each hypothesis may be used by exactly one program")
    if set(used_hypotheses) != set(hypothesis_by_id):
        issues.append("every hypothesis must map to exactly one program")
    if len(program_ids) != len(set(program_ids)):
        issues.append("program_id values must be unique")
    if len(names) != len(set(names)):
        issues.append("program names must be unique")
    return list(dict.fromkeys(issues))


def normalize_discovery_result(value):
    """Return a detached result suitable for tracing."""
    return deepcopy(value) if isinstance(value, dict) else value


def validate_prediction_result(value, labels):
    if not isinstance(value, dict):
        return ["prediction result must be an object"]
    if set(value) != {"prediction", "rationale"}:
        return ["prediction result must contain exactly prediction and rationale"]
    issues = []
    if value.get("prediction") not in set(labels or []):
        issues.append("prediction must be one of the supplied answer-option labels")
    if not _non_empty_string(value.get("rationale")):
        issues.append("rationale must be a non-empty string")
    return issues

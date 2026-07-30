"""Schemas for hypothesis-conditioned completion retrieval programs."""

import ast
from copy import deepcopy

from operator_learning.runtime import (
    BANNED_CALL_NAMES,
    BANNED_NAME_FRAGMENTS,
    SAFE_IMPORT_ROOTS,
)


DISCOVERY_SCHEMA_VERSION = "online_completion_retrieval_programs_v3"
RETRIEVAL_RESULT_SCHEMA_VERSION = "completion_exemplar_retrieval_v2"
ONLINE_RETRIEVE_ARGUMENTS = (
    "partial_item_ids",
    "dataset_workspace",
    "parameters",
    "budget",
)
PROVENANCE_FIELDS = {
    "source",
    "relation",
    "supporting_context",
}
SUPPORTING_CONTEXT_FIELDS = {"item_ids", "bundle_ids", "user_ids"}
RETRIEVED_ITEM_FIELDS = {"item_id", "provenance"}
PROGRAM_FIELDS = {
    "id",
    "hypothesis",
    "strategy",
    "required_sources",
    "parameters",
    "code",
}
STRATEGY_FIELDS = {"reference", "retrieval"}


def _non_empty_string(value):
    return isinstance(value, str) and bool(value.strip())


def _string_list(value, *, allow_empty=False):
    return (
        isinstance(value, list)
        and (allow_empty or bool(value))
        and all(_non_empty_string(item) for item in value)
    )


def hypothesis_statement(program):
    """Return the completion hypothesis stored in one program."""
    if not isinstance(program, dict):
        return ""
    return str(program.get("hypothesis") or "").strip()


def validate_online_program_source(code):
    """Validate the partial-only retrieve function boundary."""
    if not isinstance(code, str) or not code.strip():
        return ["generated program code must be a non-empty string"]
    try:
        tree = ast.parse(code)
    except SyntaxError as error:
        return [f"generated program code is invalid Python: {error}"]

    issues = []
    public_functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")
    ]
    functions_by_name = {node.name: node for node in public_functions}
    if len(public_functions) != 1 or set(functions_by_name) != {"retrieve"}:
        issues.append("code must define exactly one public function: retrieve")

    retrieve = functions_by_name.get("retrieve")
    if retrieve is not None:
        arguments = tuple(argument.arg for argument in retrieve.args.args)
        if arguments != ONLINE_RETRIEVE_ARGUMENTS:
            issues.append(
                "retrieve arguments must be exactly: "
                + ", ".join(ONLINE_RETRIEVE_ARGUMENTS)
            )
        if retrieve.args.vararg or retrieve.args.kwarg or retrieve.args.kwonlyargs:
            issues.append("retrieve must not use variadic or keyword-only arguments")
        loaded_names = {
            node.id
            for node in ast.walk(retrieve)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }
        for required_name in ("partial_item_ids", "dataset_workspace", "budget"):
            if required_name not in loaded_names:
                issues.append(f"retrieve must use {required_name}")

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
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root not in SAFE_IMPORT_ROOTS:
                    issues.append(f"unsupported import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            root = str(node.module or "").split(".", 1)[0]
            if root not in SAFE_IMPORT_ROOTS:
                issues.append(f"unsupported import: {node.module}")
        elif isinstance(node, ast.ClassDef):
            issues.append("generated code must not define classes")
        elif isinstance(node, ast.AsyncFunctionDef):
            issues.append("generated code must not define async functions")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            issues.append("generated code must not use global or nonlocal declarations")
        elif isinstance(node, ast.Call):
            name = node.func.id if isinstance(node.func, ast.Name) else None
            if name in BANNED_CALL_NAMES or name in forbidden_calls:
                issues.append(f"forbidden call: {name}")
        elif isinstance(node, ast.Name):
            lowered = node.id.lower()
            if node.id.startswith("__"):
                issues.append("generated code must not access dunder names")
            if any(fragment in lowered for fragment in BANNED_NAME_FRAGMENTS):
                issues.append(f"forbidden answer-dependent name: {node.id}")
        elif isinstance(node, ast.Attribute):
            lowered = node.attr.lower()
            if node.attr.startswith("_"):
                issues.append("generated code must not access private attributes")
            if node.attr.endswith("_"):
                issues.append(f"in-place mutation method is forbidden: {node.attr}")
            if any(fragment in lowered for fragment in BANNED_NAME_FRAGMENTS):
                issues.append(
                    f"forbidden answer-dependent attribute: {node.attr}"
                )
    return list(dict.fromkeys(issues))


def validate_discovery_result(
    value,
    *,
    available_sources,
    min_hypotheses=2,
    max_hypotheses=3,
):
    """Validate completion hypotheses and their retrieval programs."""
    if not isinstance(value, dict):
        return ["LLM1 result must be an object"]
    expected = {"schema_version", "programs"}
    if set(value) != expected:
        return [
            "LLM1 result must contain exactly: " + ", ".join(sorted(expected))
        ]

    issues = []
    if value.get("schema_version") != DISCOVERY_SCHEMA_VERSION:
        issues.append(f"schema_version must be {DISCOVERY_SCHEMA_VERSION}")
    programs = value.get("programs")
    if not isinstance(programs, list):
        issues.append("programs must be a list")
        programs = []
    if not int(min_hypotheses) <= len(programs) <= int(max_hypotheses):
        issues.append(
            "programs must contain between "
            f"{int(min_hypotheses)} and {int(max_hypotheses)} entries"
        )

    allowed_sources = set(available_sources or [])
    program_ids = []
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
        for field in ("id", "hypothesis"):
            if not _non_empty_string(program.get(field)):
                issues.append(f"{prefix}.{field} must be a non-empty string")
        program_ids.append(program.get("id"))

        strategy = program.get("strategy")
        if not isinstance(strategy, dict) or set(strategy) != STRATEGY_FIELDS:
            issues.append(
                f"{prefix}.strategy must contain exactly: "
                + ", ".join(sorted(STRATEGY_FIELDS))
            )
        else:
            for field in sorted(STRATEGY_FIELDS):
                if not _non_empty_string(strategy.get(field)):
                    issues.append(
                        f"{prefix}.strategy.{field} must be a non-empty string"
                    )

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
        if not isinstance(program.get("parameters"), dict):
            issues.append(f"{prefix}.parameters must be an object")
        issues.extend(
            f"{prefix}: {issue}"
            for issue in validate_online_program_source(program.get("code"))
        )

    if len(program_ids) != len(set(program_ids)):
        issues.append("program id values must be unique")
    return list(dict.fromkeys(issues))


def normalize_discovery_result(value):
    """Return a detached result suitable for fixation and tracing."""
    return deepcopy(value) if isinstance(value, dict) else value


def validate_prediction_result(value, labels):
    """Validate a complete final ranking and its top-1 prediction."""
    if not isinstance(value, dict):
        return ["prediction result must be an object"]
    expected = {"prediction", "ranking", "rationale"}
    if set(value) != expected:
        return [
            "prediction result must contain exactly prediction, ranking, and rationale"
        ]
    issues = []
    allowed = set(labels or [])
    prediction = value.get("prediction")
    ranking = value.get("ranking")
    if prediction not in allowed:
        issues.append("prediction must be one of the supplied answer-option labels")
    if not isinstance(ranking, list):
        issues.append("ranking must be a list")
    else:
        if len(ranking) != len(labels or []):
            issues.append("ranking must contain every answer-option label")
        if len(ranking) != len(set(ranking)):
            issues.append("ranking labels must be unique")
        if set(ranking) != allowed:
            issues.append("ranking must contain every answer-option label exactly once")
        if ranking and prediction != ranking[0]:
            issues.append("prediction must equal the first ranking label")
    if not _non_empty_string(value.get("rationale")):
        issues.append("rationale must be a non-empty string")
    return list(dict.fromkeys(issues))

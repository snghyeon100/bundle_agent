"""Schemas for hypothesis-conditioned candidate-retrieval programs."""

from copy import deepcopy
import json
import re


OPERATOR_LIBRARY_SCHEMA_VERSION = "candidate_program_library_v1"
COMPILED_LIBRARY_SCHEMA_VERSION = "compiled_candidate_program_library_v1"
CANDIDATE_PROPOSAL_OUTPUT_CONTRACT = "candidate_proposals_with_source_provenance"

OPERATOR_FIELDS = (
    "name",
    "hypothesis",
    "required_sources",
    "applicability",
    "evidence_types",
    "pseudocode",
    "output_contract",
)
INDUCTION_STRATEGY_SPEC_FIELDS = (
    "strategy_id",
    "intent",
    "name",
    "description",
    "reference_construction",
    "candidate_relation",
    "evidence_route",
    "required_sources",
    "pseudocode",
)
INDUCTION_PROGRAM_FIELDS = (
    "strategy_id",
    "code",
)
OPERATOR_METADATA_FIELDS = (
    "operator_id",
    "origin_case_id",
    "member_operator_ids",
    "origin_case_ids",
    "generated_code",
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
    return "".join(
        character
        for character in str(value or "").lower()
        if character.isalnum()
    )


def normalize_operator(operator, *, preserve_metadata=False):
    """Normalize transitional field names into the candidate-program schema."""
    if not isinstance(operator, dict):
        return operator
    normalized = deepcopy(operator)
    if "hypothesis" not in normalized and "objective" in normalized:
        normalized["hypothesis"] = normalized["objective"]
    if "pseudocode" not in normalized and "operation" in normalized:
        operation = normalized["operation"]
        normalized["pseudocode"] = (
            operation if isinstance(operation, list) else [operation]
        )
    if "required_sources" not in normalized and "sources" in normalized:
        normalized["required_sources"] = normalized["sources"]

    # Graph-era input/output ID ports are deliberately not migrated. Every new
    # program has one fixed runtime contract instead.
    normalized.setdefault(
        "output_contract",
        CANDIDATE_PROPOSAL_OUTPUT_CONTRACT,
    )
    fields = set(OPERATOR_FIELDS)
    if preserve_metadata:
        fields.update(OPERATOR_METADATA_FIELDS)
    return {
        field: normalized[field]
        for field in normalized
        if field in fields
    }


def validate_operator(operator, *, allowed_source_names=None):
    """Return deterministic issues for one candidate-retrieval program spec."""
    if not isinstance(operator, dict):
        return ["operator must be an object"]

    normalized = normalize_operator(operator, preserve_metadata=True)
    required = set(OPERATOR_FIELDS)
    metadata = set(OPERATOR_METADATA_FIELDS)
    actual = set(normalized)
    raw_actual = set(operator)
    issues = []

    missing = sorted(required - actual)
    extra = sorted(
        raw_actual
        - required
        - metadata
        - {"objective", "operation", "sources", "kind"}
    )
    if missing:
        issues.append("missing operator fields: " + ", ".join(missing))
    if extra:
        issues.append("unsupported operator fields: " + ", ".join(extra))

    for field in ("name", "hypothesis"):
        if not _non_empty_string(normalized.get(field)):
            issues.append(f"operator.{field} must be a non-empty string")
    name = normalized.get("name")
    if _non_empty_string(name) and not re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", name):
        issues.append("operator.name must be a concise alphanumeric PascalCase name")

    for field, allow_empty in (
        ("required_sources", True),
        ("applicability", True),
        ("evidence_types", False),
        ("pseudocode", False),
    ):
        value = normalized.get(field)
        if not _string_list(value, allow_empty=allow_empty):
            suffix = " (empty allowed)" if allow_empty else ""
            issues.append(f"operator.{field} must be a string list{suffix}")
        elif len(value) != len(set(value)):
            issues.append(f"operator.{field} values must be unique")

    pseudocode = normalized.get("pseudocode")
    if isinstance(pseudocode, list) and not 3 <= len(pseudocode) <= 8:
        issues.append("operator.pseudocode must contain 3 to 8 steps")

    sources = normalized.get("required_sources")
    if isinstance(sources, list):
        allowed = set(allowed_source_names or [])
        unknown = sorted(set(sources) - allowed) if allowed else []
        if unknown:
            issues.append(
                "operator.required_sources contains unavailable components: "
                + ", ".join(unknown)
            )

    if normalized.get("output_contract") != CANDIDATE_PROPOSAL_OUTPUT_CONTRACT:
        issues.append(
            "operator.output_contract must be "
            + CANDIDATE_PROPOSAL_OUTPUT_CONTRACT
        )

    deployable_text = json.dumps(
        {field: normalized.get(field) for field in OPERATOR_FIELDS},
        ensure_ascii=False,
        sort_keys=True,
    )
    deployable_key = _normalized_text(deployable_text)
    if any(
        marker in deployable_key
        for marker in (
            "groundtruth",
            "goldlabel",
            "correctanswer",
            "trueanswer",
            "candidateoption",
        )
    ):
        issues.append(
            "operator must be deployable without ground truth or supplied answer options"
        )
    return issues


def validate_induction_result(
    value,
    expected_count=None,
    *,
    allowed_source_names=None,
    min_count=None,
    max_count=None,
):
    """Validate one spec-first, single-call strategy and program response."""
    if not isinstance(value, dict):
        return ["induction result must be an object"]
    if set(value) != {"strategy_specs", "programs"}:
        return [
            "induction result must contain exactly strategy_specs and programs"
        ]
    strategy_specs = value.get("strategy_specs")
    programs = value.get("programs")
    if not isinstance(strategy_specs, list):
        return ["strategy_specs must be a list"]
    if not isinstance(programs, list):
        return ["programs must be a list"]

    issues = []
    if expected_count is not None and len(strategy_specs) != int(expected_count):
        issues.append(f"expected exactly {int(expected_count)} strategy_specs")
    if expected_count is not None and len(programs) != int(expected_count):
        issues.append(f"expected exactly {int(expected_count)} programs")
    if min_count is not None and len(strategy_specs) < int(min_count):
        issues.append(f"expected at least {int(min_count)} strategy_specs")
    if max_count is not None and len(strategy_specs) > int(max_count):
        issues.append(f"expected at most {int(max_count)} strategy_specs")

    names = []
    strategy_ids = []
    allowed = set(allowed_source_names or [])
    for index, strategy in enumerate(strategy_specs):
        prefix = f"strategy_specs[{index}]"
        if not isinstance(strategy, dict):
            issues.append(f"{prefix} must be an object")
            continue
        if set(strategy) != set(INDUCTION_STRATEGY_SPEC_FIELDS):
            issues.append(
                f"{prefix} must contain exactly strategy_id, intent, name, "
                "description, reference_construction, candidate_relation, "
                "evidence_route, required_sources, and pseudocode"
            )
        strategy_id = strategy.get("strategy_id")
        if not _non_empty_string(strategy_id):
            issues.append(f"{prefix}.strategy_id must be a non-empty string")
        else:
            strategy_ids.append(strategy_id)
        name = strategy.get("name")
        if not _non_empty_string(name):
            issues.append(f"{prefix}.name must be a non-empty string")
        elif not re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", name):
            issues.append(f"{prefix}.name must be an alphanumeric PascalCase name")
        else:
            names.append(name)
        for field in (
            "description",
            "intent",
            "reference_construction",
            "candidate_relation",
        ):
            if not _non_empty_string(strategy.get(field)):
                issues.append(f"{prefix}.{field} must be a non-empty string")
        for field in ("evidence_route", "pseudocode"):
            if not _string_list(strategy.get(field)):
                issues.append(f"{prefix}.{field} must be a non-empty string list")
        sources = strategy.get("required_sources")
        if not _string_list(sources):
            issues.append(f"{prefix}.required_sources must be a non-empty string list")
        elif len(sources) != len(set(sources)):
            issues.append(f"{prefix}.required_sources values must be unique")
        elif allowed:
            unknown = sorted(set(sources) - allowed)
            if unknown:
                issues.append(
                    f"{prefix}.required_sources contains unavailable components: "
                    + ", ".join(unknown)
                )
    if len(names) != len(set(names)):
        issues.append("strategy names must be unique within a case")
    if len(strategy_ids) != len(set(strategy_ids)):
        issues.append("strategy_ids must be unique within strategy_specs")

    program_ids = []
    for index, program in enumerate(programs):
        prefix = f"programs[{index}]"
        if not isinstance(program, dict):
            issues.append(f"{prefix} must be an object")
            continue
        if set(program) != set(INDUCTION_PROGRAM_FIELDS):
            issues.append(f"{prefix} must contain exactly strategy_id and code")
        strategy_id = program.get("strategy_id")
        if not _non_empty_string(strategy_id):
            issues.append(f"{prefix}.strategy_id must be a non-empty string")
        else:
            program_ids.append(strategy_id)
        if not _non_empty_string(program.get("code")):
            issues.append(f"{prefix}.code must be a non-empty string")
    if len(program_ids) != len(set(program_ids)):
        issues.append("strategy_ids must be unique within programs")
    if set(strategy_ids) != set(program_ids):
        issues.append(
            "program strategy_ids must match strategy_specs strategy_ids one-to-one"
        )
    return issues


def resolve_induction_strategies(value):
    """Join spec-first strategies and programs by immutable strategy_id."""
    if not isinstance(value, dict):
        return []
    programs = {
        program.get("strategy_id"): program.get("code")
        for program in value.get("programs", [])
        if isinstance(program, dict)
        and _non_empty_string(program.get("strategy_id"))
    }
    resolved = []
    for strategy in value.get("strategy_specs", []):
        if not isinstance(strategy, dict):
            continue
        strategy_id = strategy.get("strategy_id")
        merged = deepcopy(strategy)
        merged["code"] = str(programs.get(strategy_id) or "")
        resolved.append(merged)
    return resolved


def resolve_induction_operators(value):
    """Map spec-first strategies to the existing operator-library representation."""
    resolved = []
    for strategy in resolve_induction_strategies(value):
        intent = str(strategy.get("intent") or "").strip()
        description = str(strategy.get("description") or "").strip()
        reference = str(strategy.get("reference_construction") or "").strip()
        relation = str(strategy.get("candidate_relation") or "").strip()
        evidence_route = " -> ".join(strategy.get("evidence_route", []))
        hypothesis = "\n".join(
            line
            for line in (
                f"Intent: {intent}" if intent else "",
                f"Strategy: {description}" if description else "",
                f"Reference: {reference}" if reference else "",
                f"Candidate relation: {relation}" if relation else "",
                f"Evidence route: {evidence_route}" if evidence_route else "",
            )
            if line
        )
        standalone = {
            "name": strategy.get("name"),
            "hypothesis": hypothesis,
            "required_sources": deepcopy(strategy.get("required_sources", [])),
            "applicability": [],
            "evidence_types": ["textual_evidence_context"],
            "pseudocode": deepcopy(strategy.get("pseudocode", [])),
            "output_contract": CANDIDATE_PROPOSAL_OUTPUT_CONTRACT,
        }
        normalized = normalize_operator(standalone)
        normalized["generated_code"] = str(strategy.get("code") or "")
        resolved.append(normalized)
    return resolved


def normalize_library(value):
    """Normalize one candidate-program library."""
    source = deepcopy(value) if isinstance(value, dict) else {}
    operators = source.get("operators", source.get("programs", []))
    normalized = []
    if isinstance(operators, list):
        for operator in operators:
            candidate = normalize_operator(operator, preserve_metadata=True)
            if isinstance(candidate, dict):
                normalized.append(candidate)
    return {
        "schema_version": OPERATOR_LIBRARY_SCHEMA_VERSION,
        "operators": normalized,
    }


def validate_operator_library(value, *, allowed_source_names=None, allow_empty=False):
    """Validate a candidate-program library without any graph constraints."""
    if not isinstance(value, dict):
        return ["operator library must be an object"]
    issues = []
    if value.get("schema_version", OPERATOR_LIBRARY_SCHEMA_VERSION) != (
        OPERATOR_LIBRARY_SCHEMA_VERSION
    ):
        issues.append(
            f"schema_version must be {OPERATOR_LIBRARY_SCHEMA_VERSION}"
        )
    extra = sorted(set(value) - {"schema_version", "operators"})
    if extra:
        issues.append(
            "operator library contains unsupported fields: " + ", ".join(extra)
        )
    operators = value.get("operators")
    if not isinstance(operators, list) or (not allow_empty and not operators):
        issues.append(
            "operators must be a list"
            + (" (empty allowed)" if allow_empty else " with at least one operator")
        )
        operators = []
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
        issues.append("operator names must be unique")
    ids = [
        operator.get("operator_id")
        for operator in operators
        if isinstance(operator, dict)
        and _non_empty_string(operator.get("operator_id"))
    ]
    if len(ids) != len(set(ids)):
        issues.append("operator_id values must be unique")
    return issues


def operator_names(library):
    return {
        operator.get("name")
        for operator in library.get("operators", [])
        if isinstance(operator, dict) and _non_empty_string(operator.get("name"))
    }

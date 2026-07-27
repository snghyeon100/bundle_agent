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
INDUCTION_HYPOTHESIS_FIELDS = ("id", "observed_cues", "statement")
OPERATOR_METADATA_FIELDS = (
    "operator_id",
    "origin_case_id",
    "member_operator_ids",
    "origin_case_ids",
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
    """Validate one candidate-blind hypothesis/operator induction response."""
    if not isinstance(value, dict):
        return ["induction result must be an object"]
    if set(value) != {"hypotheses", "operators"}:
        return ["induction result must contain exactly hypotheses and operators"]
    hypotheses = value.get("hypotheses")
    operators = value.get("operators")
    if not isinstance(hypotheses, list):
        return ["hypotheses must be a list"]
    if not isinstance(operators, list):
        return ["operators must be a list"]

    issues = []
    if len(hypotheses) != len(operators):
        issues.append("hypotheses and operators must have a one-to-one correspondence")
    if expected_count is not None and len(operators) != int(expected_count):
        issues.append(f"expected exactly {int(expected_count)} operators")
    if min_count is not None and len(operators) < int(min_count):
        issues.append(f"expected at least {int(min_count)} operators")
    if max_count is not None and len(operators) > int(max_count):
        issues.append(f"expected at most {int(max_count)} operators")

    hypothesis_by_id = {}
    for index, hypothesis in enumerate(hypotheses):
        prefix = f"hypotheses[{index}]"
        if not isinstance(hypothesis, dict):
            issues.append(f"{prefix} must be an object")
            continue
        if set(hypothesis) != set(INDUCTION_HYPOTHESIS_FIELDS):
            issues.append(
                f"{prefix} must contain exactly id, observed_cues, and statement"
            )
        hypothesis_id = hypothesis.get("id")
        observed_cues = hypothesis.get("observed_cues")
        statement = hypothesis.get("statement")
        if not _non_empty_string(hypothesis_id):
            issues.append(f"{prefix}.id must be a non-empty string")
        elif hypothesis_id in hypothesis_by_id:
            issues.append(f"{prefix}.id must be unique")
        if not _non_empty_string(statement):
            issues.append(f"{prefix}.statement must be a non-empty string")
        if not _string_list(observed_cues):
            issues.append(f"{prefix}.observed_cues must be a non-empty string list")
        elif not 1 <= len(observed_cues) <= 4:
            issues.append(f"{prefix}.observed_cues must contain 1 to 4 cues")
        elif len(observed_cues) != len(set(observed_cues)):
            issues.append(f"{prefix}.observed_cues values must be unique")
        if _non_empty_string(hypothesis_id):
            hypothesis_by_id[hypothesis_id] = statement

    used_hypothesis_ids = []
    for index, operator in enumerate(operators):
        if not isinstance(operator, dict):
            issues.append(f"operators[{index}]: operator must be an object")
            continue
        hypothesis_id = operator.get("hypothesis_id")
        if not _non_empty_string(hypothesis_id):
            issues.append(
                f"operators[{index}].hypothesis_id must be a non-empty string"
            )
            continue
        used_hypothesis_ids.append(hypothesis_id)
        if hypothesis_id not in hypothesis_by_id:
            issues.append(
                f"operators[{index}].hypothesis_id must reference hypotheses"
            )
            continue
        resolved = {
            key: deepcopy(item)
            for key, item in operator.items()
            if key != "hypothesis_id"
        }
        issues.extend(
            f"operators[{index}]: {issue}"
            for issue in validate_operator(
                resolved,
                allowed_source_names=allowed_source_names,
            )
        )

    if len(used_hypothesis_ids) != len(set(used_hypothesis_ids)):
        issues.append("each hypothesis may be used by exactly one operator")
    unused = sorted(set(hypothesis_by_id) - set(used_hypothesis_ids))
    if unused:
        issues.append(
            "every hypothesis must map to one operator: " + ", ".join(unused)
        )

    names = [
        operator.get("name")
        for operator in operators
        if isinstance(operator, dict) and _non_empty_string(operator.get("name"))
    ]
    if len(names) != len(set(names)):
        issues.append("operator names must be unique within a case")
    return issues


def resolve_induction_operators(value):
    """Attach each case hypothesis to its reusable candidate-program spec."""
    if not isinstance(value, dict):
        return []
    hypothesis_by_id = {
        hypothesis.get("id"): hypothesis.get("statement")
        for hypothesis in value.get("hypotheses", [])
        if isinstance(hypothesis, dict)
    }
    resolved = []
    for operator in value.get("operators", []):
        if not isinstance(operator, dict):
            continue
        hypothesis_id = operator.get("hypothesis_id")
        standalone = {
            key: deepcopy(item)
            for key, item in operator.items()
            if key != "hypothesis_id"
        }
        if not _non_empty_string(standalone.get("hypothesis")):
            standalone["hypothesis"] = hypothesis_by_id.get(hypothesis_id, "")
        resolved.append(normalize_operator(standalone))
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

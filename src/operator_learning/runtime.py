"""Runtime contracts and validation for compiled candidate programs."""

import ast
from hashlib import sha256
import json
from typing import Protocol, runtime_checkable


CANDIDATE_PROPOSAL_SCHEMA_VERSION = "candidate_proposal_set_v1"
EXECUTE_ARGUMENTS = (
    "partial_item_ids",
    "source_api",
    "candidate_budget",
    "evidence_budget",
)
SAFE_IMPORT_ROOTS = {
    "collections",
    "functools",
    "heapq",
    "itertools",
    "math",
    "statistics",
}
BANNED_CALL_NAMES = {
    "__import__",
    "compile",
    "eval",
    "exec",
    "globals",
    "input",
    "locals",
    "open",
}
BANNED_NAME_FRAGMENTS = {
    "candidate_indices",
    "ground_truth",
    "true_indice",
    "true_option",
}


@runtime_checkable
class SourceAPI(Protocol):
    """Shared data-access surface used in validation and online execution."""

    @property
    def available_sources(self):
        ...

    def get_all_item_ids(self):
        ...

    def get_dataset_statistics(self):
        ...

    def get_item_metadata(self, item_ids):
        ...

    def get_bundles_for_items(self, item_ids):
        ...

    def get_items_for_bundles(self, bundle_ids):
        ...

    def get_users_for_items(self, item_ids):
        ...

    def get_items_for_users(self, user_ids):
        ...

    def get_item_embeddings(self, item_ids, source_id):
        ...


def implementation_hash(code):
    return sha256(str(code).encode("utf-8")).hexdigest()


def validate_program_source(code):
    """Statically validate the reusable function boundary before execution."""
    if not isinstance(code, str) or not code.strip():
        return ["compiled program code must be a non-empty string"]
    try:
        tree = ast.parse(code)
    except SyntaxError as error:
        return [f"compiled program code is invalid Python: {error}"]

    issues = []
    public_functions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not node.name.startswith("_")
    ]
    execute_functions = [
        node for node in public_functions if node.name == "execute"
    ]
    if len(execute_functions) != 1:
        issues.append("code must define exactly one public execute function")
    if len(public_functions) != 1:
        issues.append("execute must be the only public function")
    if execute_functions:
        execute = execute_functions[0]
        arguments = [argument.arg for argument in execute.args.args]
        if tuple(arguments) != EXECUTE_ARGUMENTS:
            issues.append(
                "execute arguments must be exactly: " + ", ".join(EXECUTE_ARGUMENTS)
            )
        if execute.args.vararg or execute.args.kwarg or execute.args.kwonlyargs:
            issues.append("execute must not use variadic or keyword-only arguments")

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
        elif isinstance(node, ast.Call):
            name = node.func.id if isinstance(node.func, ast.Name) else None
            if name in BANNED_CALL_NAMES:
                issues.append(f"forbidden call: {name}")
        elif isinstance(node, ast.Name):
            lowered = node.id.lower()
            if any(fragment in lowered for fragment in BANNED_NAME_FRAGMENTS):
                issues.append(f"forbidden answer-dependent name: {node.id}")
        elif isinstance(node, ast.Attribute):
            lowered = node.attr.lower()
            if any(fragment in lowered for fragment in BANNED_NAME_FRAGMENTS):
                issues.append(f"forbidden answer-dependent attribute: {node.attr}")

    return list(dict.fromkeys(issues))


def validate_compilation_result(value, operator):
    """Validate one LLM compilation response and its source."""
    if not isinstance(value, dict):
        return ["compilation result must be an object"]
    if set(value) != {"program_name", "code"}:
        return ["compilation result must contain exactly program_name and code"]
    issues = []
    if value.get("program_name") != operator.get("name"):
        issues.append("program_name must exactly match the operator name")
    issues.extend(validate_program_source(value.get("code")))
    return issues


def _non_empty_id(value):
    return (
        isinstance(value, (str, int))
        and not isinstance(value, bool)
        and bool(str(value).strip())
    )


def _id_list(value):
    return isinstance(value, list) and all(_non_empty_id(item) for item in value)


def validate_candidate_proposal_set(
    value,
    *,
    allowed_sources=None,
    allowed_evidence_types=None,
    candidate_budget=None,
    evidence_budget=None,
    expected_program_id=None,
    expected_hypothesis=None,
    excluded_item_ids=None,
):
    """Validate one program execution result independently of its prediction."""
    if not isinstance(value, dict):
        return ["candidate proposal set must be an object"]
    expected_fields = {
        "schema_version",
        "program_id",
        "hypothesis",
        "candidate_proposals",
        "evidence_records",
        "execution_trace",
    }
    if set(value) != expected_fields:
        return [
            "candidate proposal set must contain exactly: "
            + ", ".join(sorted(expected_fields))
        ]

    issues = []
    if value.get("schema_version") != CANDIDATE_PROPOSAL_SCHEMA_VERSION:
        issues.append(
            f"schema_version must be {CANDIDATE_PROPOSAL_SCHEMA_VERSION}"
        )
    if not isinstance(value.get("program_id"), str) or not value["program_id"].strip():
        issues.append("program_id must be a non-empty string")
    elif expected_program_id is not None and value["program_id"] != str(
        expected_program_id
    ):
        issues.append("program_id does not match the compiled operator")
    if not isinstance(value.get("hypothesis"), str) or not value["hypothesis"].strip():
        issues.append("hypothesis must be a non-empty string")
    elif expected_hypothesis is not None and value["hypothesis"] != str(
        expected_hypothesis
    ):
        issues.append("hypothesis does not match the compiled operator")

    evidence_records = value.get("evidence_records")
    evidence_by_id = {}
    if not isinstance(evidence_records, list):
        issues.append("evidence_records must be a list")
        evidence_records = []
    if evidence_budget is not None and len(evidence_records) > int(evidence_budget):
        issues.append("evidence_records exceeds evidence_budget")
    evidence_fields = {
        "evidence_id",
        "type",
        "source",
        "anchor_item_ids",
        "related_item_ids",
        "related_bundle_ids",
        "attributes",
    }
    allowed = set(allowed_sources or [])
    allowed_types = set(allowed_evidence_types or [])
    for index, record in enumerate(evidence_records):
        prefix = f"evidence_records[{index}]"
        if not isinstance(record, dict):
            issues.append(f"{prefix} must be an object")
            continue
        if set(record) != evidence_fields:
            issues.append(
                f"{prefix} must contain exactly: "
                + ", ".join(sorted(evidence_fields))
            )
        evidence_id = record.get("evidence_id")
        if not isinstance(evidence_id, str) or not evidence_id.strip():
            issues.append(f"{prefix}.evidence_id must be a non-empty string")
        elif evidence_id in evidence_by_id:
            issues.append(f"{prefix}.evidence_id must be unique")
        else:
            evidence_by_id[evidence_id] = record
        for field in ("type", "source"):
            if not isinstance(record.get(field), str) or not record[field].strip():
                issues.append(f"{prefix}.{field} must be a non-empty string")
        if allowed and record.get("source") not in allowed:
            issues.append(f"{prefix}.source is not permitted")
        if allowed_types and record.get("type") not in allowed_types:
            issues.append(f"{prefix}.type is not declared by the operator")
        for field in (
            "anchor_item_ids",
            "related_item_ids",
            "related_bundle_ids",
        ):
            if not _id_list(record.get(field)):
                issues.append(f"{prefix}.{field} must be an ID list")
        if not isinstance(record.get("attributes"), dict):
            issues.append(f"{prefix}.attributes must be an object")

    proposals = value.get("candidate_proposals")
    if not isinstance(proposals, list):
        issues.append("candidate_proposals must be a list")
        proposals = []
    if candidate_budget is not None and len(proposals) > int(candidate_budget):
        issues.append("candidate_proposals exceeds candidate_budget")
    seen_items = set()
    excluded = {str(item_id) for item_id in (excluded_item_ids or [])}
    proposal_fields = {"item_id", "evidence_refs"}
    for index, proposal in enumerate(proposals):
        prefix = f"candidate_proposals[{index}]"
        if not isinstance(proposal, dict):
            issues.append(f"{prefix} must be an object")
            continue
        if set(proposal) != proposal_fields:
            issues.append(
                f"{prefix} must contain exactly item_id and evidence_refs"
            )
        item_id = proposal.get("item_id")
        if not _non_empty_id(item_id):
            issues.append(f"{prefix}.item_id must be a canonical ID")
        elif str(item_id) in seen_items:
            issues.append(f"{prefix}.item_id must be unique")
        elif str(item_id) in excluded:
            issues.append(f"{prefix}.item_id must exclude partial items")
        else:
            seen_items.add(str(item_id))
        refs = proposal.get("evidence_refs")
        if (
            not isinstance(refs, list)
            or not refs
            or not all(isinstance(ref, str) and ref.strip() for ref in refs)
        ):
            issues.append(f"{prefix}.evidence_refs must be a non-empty string list")
            continue
        unknown = sorted(set(refs) - set(evidence_by_id))
        if unknown:
            issues.append(
                f"{prefix}.evidence_refs contains unknown IDs: "
                + ", ".join(unknown)
            )
        if _non_empty_id(item_id):
            supported = any(
                any(str(related) == str(item_id) for related in (
                    evidence_by_id.get(ref, {}).get("related_item_ids", [])
                ))
                for ref in refs
            )
            if not supported:
                issues.append(
                    f"{prefix} must reference evidence containing its item_id"
                )

    trace = value.get("execution_trace")
    trace_fields = {"used_sources", "candidate_budget", "evidence_budget"}
    if not isinstance(trace, dict):
        issues.append("execution_trace must be an object")
    else:
        if set(trace) != trace_fields:
            issues.append(
                "execution_trace must contain exactly used_sources, "
                "candidate_budget, and evidence_budget"
            )
        used_sources = trace.get("used_sources")
        if (
            not isinstance(used_sources, list)
            or not all(
                isinstance(source, str) and source.strip()
                for source in used_sources
            )
        ):
            issues.append("execution_trace.used_sources must be a string list")
        elif allowed and not set(used_sources).issubset(allowed):
            issues.append("execution_trace.used_sources contains unpermitted sources")
        if candidate_budget is not None and trace.get("candidate_budget") != int(
            candidate_budget
        ):
            issues.append("execution_trace.candidate_budget does not match runtime")
        if evidence_budget is not None and trace.get("evidence_budget") != int(
            evidence_budget
        ):
            issues.append("execution_trace.evidence_budget does not match runtime")
    return list(dict.fromkeys(issues))


def evaluate_candidate_proposal_set(value, ground_truth_item_id):
    """Return retrieval metrics without asking an LLM to judge evidence."""
    proposals = (
        value.get("candidate_proposals", [])
        if isinstance(value, dict)
        else []
    )
    candidate_ids = [
        proposal.get("item_id")
        for proposal in proposals
        if isinstance(proposal, dict)
    ]
    target = str(ground_truth_item_id)
    rank = next(
        (
            index
            for index, item_id in enumerate(candidate_ids, start=1)
            if str(item_id) == target
        ),
        None,
    )
    return {
        "ground_truth_item_id": ground_truth_item_id,
        "candidate_count": len(candidate_ids),
        "evidence_record_count": len(
            value.get("evidence_records", [])
            if isinstance(value, dict)
            else []
        ),
        "hit": rank is not None,
        "retrieval_rank": rank,
        "reciprocal_rank": 1.0 / rank if rank else 0.0,
    }


def make_compiled_program(operator, code):
    """Create the immutable artifact that validation and online execution share."""
    return {
        "operator": operator,
        "implementation": {
            "language": "python",
            "entrypoint": "execute",
            "code": code,
            "sha256": implementation_hash(code),
        },
        "admission_status": "unverified",
        "validation_profile": None,
    }


def assert_implementation_unchanged(compiled_program):
    implementation = compiled_program.get("implementation", {})
    code = implementation.get("code", "")
    expected = implementation.get("sha256")
    actual = implementation_hash(code)
    if expected != actual:
        raise ValueError(
            "compiled program code hash changed; validation results are invalid"
        )
    return actual


def serialize_compiled_library(value):
    """Stable JSON representation useful for hashing or artifact storage."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

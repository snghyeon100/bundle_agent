"""Constrained execution of LLM-generated online Python programs."""

import json
import os
import subprocess
import sys

from operator_learning.runtime import (
    CANDIDATE_PROPOSAL_SCHEMA_VERSION,
    SAFE_IMPORT_ROOTS,
    validate_candidate_proposal_set,
)

from .schemas import PROGRAM_RESULT_FIELDS, hypothesis_statement, validate_online_program_source
from .source_api import DatasetSourceAPI


SAFE_BUILTIN_NAMES = (
    "Exception",
    "ValueError",
    "TypeError",
    "abs",
    "all",
    "any",
    "bool",
    "dict",
    "enumerate",
    "filter",
    "float",
    "int",
    "isinstance",
    "len",
    "list",
    "map",
    "max",
    "min",
    "next",
    "range",
    "reversed",
    "round",
    "set",
    "slice",
    "sorted",
    "str",
    "sum",
    "tuple",
    "zip",
)


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = str(name).split(".", 1)[0]
    if root not in SAFE_IMPORT_ROOTS:
        raise ImportError(f"unsupported import: {name}")
    return __import__(name, globals, locals, fromlist, level)


def _safe_builtins():
    import builtins

    values = {
        name: getattr(builtins, name)
        for name in SAFE_BUILTIN_NAMES
    }
    values["__import__"] = _safe_import
    return values


def execute_code_in_process(
    code,
    *,
    source_api,
    partial_item_ids,
    candidate_budget,
    evidence_budget,
):
    """Execute already validated code with a minimal builtin namespace."""
    issues = validate_online_program_source(code)
    if issues:
        raise ValueError("invalid generated program: " + " | ".join(issues))
    namespace = {"__builtins__": _safe_builtins()}
    compiled = compile(str(code), "<online_hypothesis_program>", "exec")
    exec(compiled, namespace, namespace)
    result = namespace["execute"](
        [int(item_id) for item_id in partial_item_ids],
        source_api,
        int(candidate_budget),
        int(evidence_budget),
    )
    return result


def _validate_internal_result(value):
    if not isinstance(value, dict):
        return ["program return value must be an object"]
    if set(value) != PROGRAM_RESULT_FIELDS:
        return [
            "program return value must contain exactly: "
            + ", ".join(sorted(PROGRAM_RESULT_FIELDS))
        ]
    if not isinstance(value.get("used_sources"), list):
        return ["used_sources must be a list"]
    return []


def wrap_and_validate_result(
    raw_result,
    *,
    program,
    hypothesis,
    partial_item_ids,
    candidate_budget,
    evidence_budget,
):
    """Attach trusted runtime fields and validate candidate-linked provenance."""
    internal_issues = _validate_internal_result(raw_result)
    if internal_issues:
        return None, internal_issues
    wrapped = {
        "schema_version": CANDIDATE_PROPOSAL_SCHEMA_VERSION,
        "program_id": str(program["program_id"]),
        "hypothesis": hypothesis_statement(hypothesis),
        "candidate_proposals": raw_result["candidate_proposals"],
        "evidence_records": raw_result["evidence_records"],
        "execution_trace": {
            "used_sources": raw_result["used_sources"],
            "candidate_budget": int(candidate_budget),
            "evidence_budget": int(evidence_budget),
        },
    }
    issues = validate_candidate_proposal_set(
        wrapped,
        allowed_sources=program.get("required_sources", []),
        allowed_evidence_types=program.get("evidence_types", []),
        candidate_budget=candidate_budget,
        evidence_budget=evidence_budget,
        expected_program_id=program["program_id"],
        expected_hypothesis=hypothesis_statement(hypothesis),
        excluded_item_ids=partial_item_ids,
    )
    for index, proposal in enumerate(wrapped.get("candidate_proposals", [])):
        if (
            isinstance(proposal, dict)
            and (
                not isinstance(proposal.get("item_id"), int)
                or isinstance(proposal.get("item_id"), bool)
            )
        ):
            issues.append(
                f"candidate_proposals[{index}].item_id must be an integer "
                "canonical item ID"
            )
    for record_index, record in enumerate(wrapped.get("evidence_records", [])):
        if not isinstance(record, dict):
            continue
        for field in (
            "anchor_item_ids",
            "related_item_ids",
            "related_bundle_ids",
        ):
            values = record.get(field)
            if isinstance(values, list) and any(
                not isinstance(value, int) or isinstance(value, bool)
                for value in values
            ):
                issues.append(
                    f"evidence_records[{record_index}].{field} must contain "
                    "integer canonical IDs"
                )
    return wrapped, issues


def execute_worker_request(request):
    """Worker-side entrypoint; returns a JSON-serializable execution record."""
    program = request["program"]
    hypothesis = request["hypothesis"]
    conf = request["conf"]
    partial_item_ids = request["partial_item_ids"]
    candidate_budget = int(request["candidate_budget"])
    evidence_budget = int(request["evidence_budget"])
    source_api = DatasetSourceAPI(
        conf,
        allowed_sources=program.get("required_sources", []),
    )
    raw_result = execute_code_in_process(
        program["code"],
        source_api=source_api,
        partial_item_ids=partial_item_ids,
        candidate_budget=candidate_budget,
        evidence_budget=evidence_budget,
    )
    wrapped, issues = wrap_and_validate_result(
        raw_result,
        program=program,
        hypothesis=hypothesis,
        partial_item_ids=partial_item_ids,
        candidate_budget=candidate_budget,
        evidence_budget=evidence_budget,
    )
    if wrapped is not None and not issues:
        corpus_item_ids = set(DatasetSourceAPI(conf).get_all_item_ids())
        unknown_items = sorted(
            {
                int(proposal["item_id"])
                for proposal in wrapped.get("candidate_proposals", [])
                if int(proposal["item_id"]) not in corpus_item_ids
            }
        )
        if unknown_items:
            issues.append(
                "candidate proposals contain IDs outside the item corpus: "
                + ", ".join(str(item_id) for item_id in unknown_items[:10])
            )
    return {
        "status": "success" if not issues else "invalid_output",
        "result": wrapped,
        "validation_issues": issues,
    }


def execute_program_subprocess(
    *,
    program,
    hypothesis,
    conf,
    partial_item_ids,
    candidate_budget,
    evidence_budget,
):
    """Execute one generated program in a timeout-bounded child process."""
    data_path = os.path.abspath(conf["data_path"])
    worker_conf = {
        "dataset": conf["dataset"],
        "data_path": data_path,
        "online_source_max_query_ids": int(
            conf.get("online_source_max_query_ids", 5000)
        ),
        "online_source_max_embedding_items": int(
            conf.get("online_source_max_embedding_items", 2048)
        ),
        "online_source_max_neighbor_limit": int(
            conf.get("online_source_max_neighbor_limit", 200)
        ),
    }
    request = {
        "program": program,
        "hypothesis": hypothesis,
        "conf": worker_conf,
        "partial_item_ids": [int(item_id) for item_id in partial_item_ids],
        "candidate_budget": int(candidate_budget),
        "evidence_budget": int(evidence_budget),
    }
    src_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    project_root = os.path.abspath(os.path.join(src_root, ".."))
    env = os.environ.copy()
    existing_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        src_root if not existing_path else src_root + os.pathsep + existing_path
    )
    env["PYTHONIOENCODING"] = "utf-8"
    timeout = int(conf.get("online_program_timeout_seconds", 60))
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "online_hypothesis_program.worker",
            ],
            input=json.dumps(request, ensure_ascii=False),
            cwd=project_root,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "result": None,
            "validation_issues": [],
            "stderr": "",
        }

    stdout = completed.stdout.strip()
    try:
        payload = json.loads(stdout) if stdout else None
    except json.JSONDecodeError:
        payload = None
    if completed.returncode != 0 or not isinstance(payload, dict):
        return {
            "status": "execution_error",
            "result": None,
            "validation_issues": [],
            "stderr": completed.stderr[-2000:],
            "stdout": stdout[-2000:],
        }
    payload["stderr"] = completed.stderr[-2000:]
    return payload

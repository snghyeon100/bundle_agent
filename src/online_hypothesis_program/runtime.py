"""Constrained execution of completion-exemplar retrieval programs."""

import json
import os
import subprocess
import sys

from operator_learning.runtime import SAFE_IMPORT_ROOTS

from .raw_workspace import build_dataset_workspace
from .schemas import (
    PROVENANCE_FIELDS,
    RETRIEVAL_RESULT_SCHEMA_VERSION,
    RETRIEVED_ITEM_FIELDS,
    SUPPORTING_CONTEXT_FIELDS,
    hypothesis_statement,
    validate_online_program_source,
)


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

    values = {name: getattr(builtins, name) for name in SAFE_BUILTIN_NAMES}
    values["__import__"] = _safe_import
    return values


def compile_program_in_process(code):
    """Compile one validated retrieve function."""
    issues = validate_online_program_source(code)
    if issues:
        raise ValueError("invalid generated program: " + " | ".join(issues))
    namespace = {"__builtins__": _safe_builtins()}
    compiled = compile(str(code), "<online_completion_retrieval_program>", "exec")
    exec(compiled, namespace, namespace)
    return namespace


def retrieve_in_process(
    namespace,
    *,
    partial_item_ids,
    dataset_workspace,
    parameters,
    retrieved_item_budget,
    supporting_context_budget,
):
    """Execute one fixed retrieval program for one hypothesis."""
    budget = {
        "max_retrieved_items": int(retrieved_item_budget),
        "max_supporting_contexts_per_item": int(supporting_context_budget),
    }
    return namespace["retrieve"](
        [int(item_id) for item_id in partial_item_ids],
        dataset_workspace,
        dict(parameters or {}),
        budget,
    )


def _integer_id_list(value):
    return (
        isinstance(value, list)
        and all(
            isinstance(item, int) and not isinstance(item, bool)
            for item in value
        )
    )


def _to_json_safe(value):
    """Normalize generated tensor/scalar containers into stable JSON values."""
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and isinstance(value, torch.Tensor):
        detached = value.detach().cpu()
        return (
            float(detached.item())
            if detached.numel() == 1
            else detached.tolist()
        )
    if isinstance(value, dict):
        return {key: _to_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return value


def wrap_and_validate_result(
    raw_result,
    *,
    program,
    partial_item_ids,
    retrieved_item_budget,
    supporting_context_budget,
):
    """Validate retrieved items and attach trusted execution fields."""
    if not isinstance(raw_result, list):
        return None, ["retrieve return value must be a list"]

    issues = []
    if len(raw_result) > int(retrieved_item_budget):
        issues.append("retrieve return value exceeds budget.max_retrieved_items")

    allowed_sources = set(program.get("required_sources", []))
    partial_set = {int(item_id) for item_id in partial_item_ids}
    seen_item_ids = set()
    retrieved_items = []
    for item_index, item in enumerate(raw_result):
        prefix = f"retrieved_items[{item_index}]"
        if not isinstance(item, dict):
            issues.append(f"{prefix} must be an object")
            continue
        if set(item) != RETRIEVED_ITEM_FIELDS:
            issues.append(
                f"{prefix} must contain exactly: "
                + ", ".join(sorted(RETRIEVED_ITEM_FIELDS))
            )
        item_id = item.get("item_id")
        if not isinstance(item_id, int) or isinstance(item_id, bool):
            issues.append(f"{prefix}.item_id must be an integer")
        elif item_id in seen_item_ids:
            issues.append(f"{prefix}.item_id must be unique")
        else:
            seen_item_ids.add(item_id)
        if item_id in partial_set:
            issues.append(f"{prefix}.item_id must exclude partial items")

        provenance = item.get("provenance")
        if not isinstance(provenance, list) or not provenance:
            issues.append(f"{prefix}.provenance must be a non-empty list")
            provenance = []
        elif len(provenance) > int(supporting_context_budget):
            issues.append(
                f"{prefix}.provenance exceeds "
                "budget.max_supporting_contexts_per_item"
            )
        for record_index, record in enumerate(provenance):
            record_prefix = f"{prefix}.provenance[{record_index}]"
            if not isinstance(record, dict):
                issues.append(f"{record_prefix} must be an object")
                continue
            if set(record) != PROVENANCE_FIELDS:
                issues.append(
                    f"{record_prefix} must contain exactly: "
                    + ", ".join(sorted(PROVENANCE_FIELDS))
                )
            if record.get("source") not in allowed_sources:
                issues.append(f"{record_prefix}.source is not permitted")
            relation = record.get("relation")
            if not isinstance(relation, str) or not relation.strip():
                issues.append(
                    f"{record_prefix}.relation must be a non-empty string"
                )
            supporting_context = record.get("supporting_context")
            if (
                not isinstance(supporting_context, dict)
                or set(supporting_context) != SUPPORTING_CONTEXT_FIELDS
            ):
                issues.append(
                    f"{record_prefix}.supporting_context must contain exactly: "
                    + ", ".join(sorted(SUPPORTING_CONTEXT_FIELDS))
                )
                supporting_context = {}
            for field in ("item_ids", "bundle_ids", "user_ids"):
                if not _integer_id_list(supporting_context.get(field)):
                    issues.append(
                        f"{record_prefix}.supporting_context.{field} "
                        "must be an integer ID list"
                    )
        retrieved_items.append(item)

    wrapped = {
        "schema_version": RETRIEVAL_RESULT_SCHEMA_VERSION,
        "program_id": str(program["id"]),
        "completion_hypothesis": hypothesis_statement(program),
        "retrieved_items": retrieved_items,
        "execution_trace": {
            "required_sources": list(program.get("required_sources", [])),
            "max_retrieved_items": int(retrieved_item_budget),
            "max_supporting_contexts_per_item": int(
                supporting_context_budget
            ),
        },
    }
    try:
        json.dumps(wrapped, ensure_ascii=False)
    except (TypeError, ValueError) as error:
        issues.append(f"retrieve result must be JSON-serializable: {error}")
    return wrapped, list(dict.fromkeys(issues))


def _validate_provenance_ids(packet, dataset_workspace):
    """Check retrieved and context IDs against loaded source records."""
    issues = []
    known_items = set(dataset_workspace["item_ids"])
    bundle_history = dataset_workspace.get("bundle_item_history", {})
    user_history = dataset_workspace.get("user_item_history", {})
    known_bundles = set(bundle_history.get("bundles_to_items", {}))
    known_users = set(user_history.get("users_to_items", {}))

    for item_index, item in enumerate(packet.get("retrieved_items", [])):
        item_id = item.get("item_id")
        if isinstance(item_id, int) and item_id not in known_items:
            issues.append(
                f"retrieved_items[{item_index}].item_id is outside the item corpus"
            )
        for record_index, record in enumerate(item.get("provenance", [])):
            prefix = (
                f"retrieved_items[{item_index}].provenance[{record_index}]"
            )
            supporting_context = record.get("supporting_context", {})
            unknown_items = sorted(
                set(supporting_context.get("item_ids", [])) - known_items
            )
            if unknown_items:
                issues.append(
                    f"{prefix}.supporting_context.item_ids contains IDs "
                    "outside the item corpus"
                )
            bundle_ids = set(supporting_context.get("bundle_ids", []))
            if bundle_ids and not known_bundles:
                issues.append(
                    f"{prefix}.supporting_context.bundle_ids requires "
                    "bundle_item_history"
                )
            elif bundle_ids - known_bundles:
                issues.append(
                    f"{prefix}.supporting_context.bundle_ids contains unknown IDs"
                )
            user_ids = set(supporting_context.get("user_ids", []))
            if user_ids and not known_users:
                issues.append(
                    f"{prefix}.supporting_context.user_ids requires "
                    "user_item_history"
                )
            elif user_ids - known_users:
                issues.append(
                    f"{prefix}.supporting_context.user_ids contains unknown IDs"
                )
    return issues


def execute_worker_request(request):
    """Compile and execute one hypothesis-conditioned retrieval program."""
    program = request["program"]
    conf = request["conf"]
    partial_item_ids = [int(value) for value in request["partial_item_ids"]]
    retrieved_item_budget = int(request["retrieved_item_budget"])
    supporting_context_budget = int(request["supporting_context_budget"])

    source_issues = validate_online_program_source(program.get("code"))
    if source_issues:
        return {
            "status": "invalid_program",
            "result": None,
            "validation_issues": source_issues,
        }

    dataset_workspace = build_dataset_workspace(
        conf,
        allowed_sources=program.get("required_sources", []),
    )
    try:
        namespace = compile_program_in_process(program["code"])
        raw_result = _to_json_safe(
            retrieve_in_process(
                namespace,
                partial_item_ids=partial_item_ids,
                dataset_workspace=dataset_workspace,
                parameters=program.get("parameters", {}),
                retrieved_item_budget=retrieved_item_budget,
                supporting_context_budget=supporting_context_budget,
            )
        )
        wrapped, issues = wrap_and_validate_result(
            raw_result,
            program=program,
            partial_item_ids=partial_item_ids,
            retrieved_item_budget=retrieved_item_budget,
            supporting_context_budget=supporting_context_budget,
        )
        if wrapped is not None:
            issues.extend(_validate_provenance_ids(wrapped, dataset_workspace))
    except Exception as error:
        return {
            "status": "execution_error",
            "result": None,
            "validation_issues": [],
            "error": f"{type(error).__name__}: {error}",
        }

    return {
        "status": (
            "success"
            if wrapped is not None and not issues
            else "invalid_output"
        ),
        "result": wrapped,
        "validation_issues": list(dict.fromkeys(issues)),
    }


def execute_program_subprocess(
    *,
    program,
    conf,
    partial_item_ids,
    retrieved_item_budget,
    supporting_context_budget,
):
    """Execute one retrieval program in a constrained child process."""
    worker_conf = {
        "dataset": conf["dataset"],
        "data_path": os.path.abspath(conf["data_path"]),
    }
    request = {
        "program": program,
        "conf": worker_conf,
        "partial_item_ids": [int(item_id) for item_id in partial_item_ids],
        "retrieved_item_budget": int(retrieved_item_budget),
        "supporting_context_budget": int(supporting_context_budget),
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
            [sys.executable, "-m", "online_hypothesis_program.worker"],
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

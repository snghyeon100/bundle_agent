"""Guarded subprocess execution for spec-first generated programs."""

import json
import math
import os
import subprocess
import sys
import time

from code.workspace import guard_generated_code


def source_paths_from_capabilities(workspace, capabilities):
    """Resolve capability IDs to workspace-local absolute source files."""
    resolved = {}
    workspace_dir = os.path.abspath(workspace["workspace_dir"])
    for component in capabilities.get("components", []):
        source_id = str(component.get("id") or "")
        relative = str(component.get("format", {}).get("path") or "")
        if not source_id or not relative:
            continue
        path = os.path.abspath(os.path.join(workspace_dir, relative))
        if os.path.commonpath([workspace_dir, path]) != workspace_dir:
            raise ValueError(f"source path escapes workspace: {source_id}")
        if os.path.isfile(path):
            resolved[source_id] = path
    return resolved


def _finite_json(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, dict):
        return {str(key): _finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(item) for item in value]
    return value


def validate_context_result(result, candidate_items, allowed_sources):
    """Validate candidate identity and the textual-context external contract."""
    issues = []
    if not isinstance(result, list):
        return ["program result must be a list"]
    if len(result) != len(candidate_items):
        issues.append("program result must contain one row per candidate")
    expected = [
        (str(candidate.get("label") or ""), int(candidate["item_id"]))
        for candidate in candidate_items
    ]
    observed = []
    allowed = set(allowed_sources)
    for index, row in enumerate(result):
        prefix = f"result[{index}]"
        if not isinstance(row, dict):
            issues.append(f"{prefix} must be an object")
            continue
        try:
            observed.append((str(row.get("label") or ""), int(row.get("item_id"))))
        except (TypeError, ValueError):
            issues.append(f"{prefix} must contain a valid label and item_id")
        contexts = row.get("contexts")
        if not isinstance(contexts, list):
            issues.append(f"{prefix}.contexts must be a list")
            continue
        for context_index, context in enumerate(contexts):
            context_prefix = f"{prefix}.contexts[{context_index}]"
            if not isinstance(context, dict):
                issues.append(f"{context_prefix} must be an object")
                continue
            if not str(context.get("text") or "").strip():
                issues.append(f"{context_prefix}.text must be non-empty")
            sources = context.get("sources")
            if (
                not isinstance(sources, list)
                or not sources
                or not all(
                    isinstance(source, str) and source.strip()
                    for source in sources
                )
            ):
                issues.append(
                    f"{context_prefix}.sources must be a non-empty string list"
                )
            else:
                if len(sources) != len(set(sources)):
                    issues.append(
                        f"{context_prefix}.sources values must be unique"
                    )
                unknown = sorted(set(sources) - allowed)
                if unknown:
                    issues.append(
                        f"{context_prefix}.sources contains undeclared sources: "
                        + ", ".join(unknown)
                    )
    if observed != expected:
        issues.append("program result labels/item_ids must match candidate order")
    return list(dict.fromkeys(issues))


def execute_strategy_program(
    *,
    code,
    strategy_id,
    required_sources,
    partial_items,
    candidate_items,
    all_source_paths,
    case_dir,
    conf,
):
    """Execute one generated run() in a guarded, timed subprocess."""
    started = time.perf_counter()
    report = {
        "strategy_id": str(strategy_id),
        "success": False,
        "elapsed_seconds": None,
        "guard_violations": [],
        "validation_issues": [],
        "error": "",
        "result": None,
    }
    violations = guard_generated_code(code, conf, config_prefix="code")
    if violations:
        report["guard_violations"] = violations
        report["error"] = "generated code blocked by code guard"
        report["elapsed_seconds"] = time.perf_counter() - started
        return report

    scoped_paths = {
        source_id: all_source_paths[source_id]
        for source_id in required_sources
        if source_id in all_source_paths
    }
    missing = sorted(set(required_sources) - set(scoped_paths))
    if missing:
        report["error"] = "missing required sources: " + ", ".join(missing)
        report["elapsed_seconds"] = time.perf_counter() - started
        return report

    program_dir = os.path.join(case_dir, "programs")
    execution_dir = os.path.join(case_dir, "executions", str(strategy_id))
    os.makedirs(program_dir, exist_ok=True)
    os.makedirs(execution_dir, exist_ok=True)
    program_path = os.path.join(program_dir, f"{strategy_id}.py")
    input_path = os.path.join(execution_dir, "input.json")
    output_path = os.path.join(execution_dir, "output.json")
    with open(program_path, "w", encoding="utf-8") as handle:
        handle.write(str(code).rstrip() + "\n")
    with open(input_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "partial_items": partial_items,
                "candidate_items": candidate_items,
                "source_paths": scoped_paths,
                "max_contexts_per_candidate": int(
                    conf.get("operator_program_max_contexts_per_candidate", 5)
                ),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    if os.path.isfile(output_path):
        os.remove(output_path)

    worker_path = os.path.join(os.path.dirname(__file__), "spec_first_worker.py")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    timeout = int(conf.get("operator_program_timeout_seconds", 60))
    try:
        completed = subprocess.run(
            [
                sys.executable,
                worker_path,
                "--program",
                program_path,
                "--input",
                input_path,
                "--output",
                output_path,
            ],
            cwd=case_dir,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired:
        report["error"] = f"program timed out after {timeout} seconds"
        report["elapsed_seconds"] = time.perf_counter() - started
        return report

    if completed.returncode != 0:
        report["error"] = (completed.stderr or completed.stdout)[-4000:]
        report["elapsed_seconds"] = time.perf_counter() - started
        return report
    try:
        with open(output_path, "r", encoding="utf-8") as handle:
            result = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        report["error"] = f"invalid execution output: {error}"
        report["elapsed_seconds"] = time.perf_counter() - started
        return report

    issues = validate_context_result(result, candidate_items, required_sources)
    report["validation_issues"] = issues
    report["result"] = _finite_json(result)
    report["success"] = not issues
    report["elapsed_seconds"] = time.perf_counter() - started
    return report

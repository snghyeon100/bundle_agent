import json
import os
import re
import shutil
import subprocess
import sys

from agents.common import parse_json_from_text


DEFAULT_ALLOWED_FILES = [
    "count.json",
    "item_info.json",
    "bi_train.txt",
    "ui_full.txt",
    "content_feature.pt",
    "description_feature.pt",
]


DEFAULT_FORBIDDEN_CODE_PATTERNS = [
    r"\.\.",
    r"[A-Za-z]:\\",
    r"bi_full",
    r"bi_test_gt",
    r"bi_valid_gt",
    r"[\"'](?:\.?[\\/])?results(?:[\\/][^\"']*)?[\"']",
    r"true_option",
    r"true_indice",
    r"\bhit\b",
    r"Path\.home",
    r"os\.walk",
    r"\brglob\b",
    r"\bsocket\b",
    r"\brequests\b",
    r"\burllib\b",
]


def list_agent_source_files(conf):
    data_root = os.path.abspath(os.path.join(conf["data_path"], conf["dataset"]))
    preferred = conf.get("agent_allowed_files") or DEFAULT_ALLOWED_FILES
    if bool(conf.get("agent_allow_interaction_embeddings", False)):
        preferred = list(preferred) + ["item_cf_feature.pt"]

    files = []
    for filename in preferred:
        path = os.path.join(data_root, filename)
        if os.path.exists(path):
            files.append({"name": filename, "path": path})

    extra_paths = conf.get("agent_extra_data_paths", []) or []
    for extra_path in extra_paths:
        abs_path = os.path.abspath(extra_path)
        if os.path.exists(abs_path):
            files.append({"name": os.path.basename(abs_path), "path": abs_path})
    return files


def copy_if_needed(src_path, dst_path):
    if os.path.exists(dst_path):
        src_stat = os.stat(src_path)
        dst_stat = os.stat(dst_path)
        if src_stat.st_size == dst_stat.st_size and int(src_stat.st_mtime) <= int(dst_stat.st_mtime):
            return False
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    shutil.copy2(src_path, dst_path)
    return True


def prepare_agent_workspace(conf):
    workspace_root = os.path.abspath(conf.get("agent_workspace_root", "./agent_workspaces"))
    workspace_dir = os.path.join(workspace_root, conf["dataset"])
    data_dir = os.path.join(workspace_dir, "data")
    output_dir = os.path.join(workspace_dir, "output")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    available_files = []
    copied_files = []
    for source in list_agent_source_files(conf):
        dst_path = os.path.join(data_dir, source["name"])
        if copy_if_needed(source["path"], dst_path):
            copied_files.append(source["name"])
        available_files.append({"name": source["name"], "path": f"data/{source['name']}"})

    return {
        "workspace_dir": workspace_dir,
        "data_dir": data_dir,
        "output_dir": output_dir,
        "files": available_files,
        "copied_files": copied_files,
    }


def workspace_view(workspace, evidence_output_file):
    return {
        "workspace": ".",
        "data_dir": "data",
        "output_dir": "output",
        "evidence_output_file": evidence_output_file,
        "files": workspace["files"],
        "file_format_notes": [
            "bi_train.txt and ui_full.txt lines are comma-space separated: context_id, item_id, item_id, ...",
            "CRITICAL: For bi_train.txt, values[0] is bundle_id and values[1:] are the only item_ids in that train bundle.",
            "CRITICAL: For ui_full.txt, values[0] is user_id and values[1:] are the only item_ids for that user.",
            "Context IDs and item IDs are different entity types even when their integer values match. Never search the full row for an item, look up values[0] in item_info, or include values[0] in item counts, co-occurrence, categories, neighborhoods, joins, or graph traversal.",
            "Always parse relational rows as context_id = values[0] and item_ids = values[1:], then perform all item operations only on item_ids.",
            "item_info.json is a JSON object keyed by string item_id. Use item_info[str(item_id)] to access fields such as title, cate/cate_id, pic.",
            "content_feature.pt and description_feature.pt are torch tensors indexed directly by integer item_id when their first dimension equals the item count.",
        ],
    }


def guard_generated_code(code, conf):
    if not bool(conf.get("agent_enable_code_guard", True)):
        return []
    patterns = conf.get("agent_forbidden_code_patterns") or DEFAULT_FORBIDDEN_CODE_PATTERNS
    violations = []
    for pattern in patterns:
        if re.search(pattern, code, flags=re.IGNORECASE):
            violations.append(pattern)
    return violations


def execute_generated_python_code(code, conf, workspace, evidence_output_file, script_name):
    timeout = int(conf.get("agent_code_timeout_seconds", 30))
    max_stdout_chars = int(conf.get("agent_code_max_stdout_chars", 20000))
    max_stderr_chars = int(conf.get("agent_code_max_stderr_chars", 8000))
    violations = guard_generated_code(code, conf)
    if violations:
        return {
            "returncode": None,
            "stdout": "",
            "stderr": f"Generated code blocked by guard patterns: {violations}",
            "timed_out": False,
            "guard_blocked": True,
            "guard_violations": violations,
            "evidence_text": "",
            "evidence_json": None,
            "evidence_output_file": evidence_output_file,
        }

    script_path = os.path.join(workspace["workspace_dir"], script_name)
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(code)
        f.write("\n")

    evidence_abs_path = os.path.join(workspace["workspace_dir"], evidence_output_file.replace("/", os.sep))
    if os.path.exists(evidence_abs_path):
        os.remove(evidence_abs_path)

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    try:
        completed = subprocess.run(
            [sys.executable, script_path],
            cwd=workspace["workspace_dir"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=env,
        )
        stdout = completed.stdout[-max_stdout_chars:]
        stderr = completed.stderr[-max_stderr_chars:]
        timed_out = False
        returncode = completed.returncode
    except subprocess.TimeoutExpired as exc:
        stdout = (exc.stdout or "")[-max_stdout_chars:] if isinstance(exc.stdout, str) else ""
        stderr = (exc.stderr or "")[-max_stderr_chars:] if isinstance(exc.stderr, str) else ""
        timed_out = True
        returncode = None

    evidence_text = ""
    if os.path.exists(evidence_abs_path):
        with open(evidence_abs_path, "r", encoding="utf-8") as f:
            evidence_text = f.read()
    if not evidence_text:
        evidence_text = stdout

    return {
        "returncode": returncode,
        "stdout": stdout,
        "stderr": stderr,
        "timed_out": timed_out,
        "guard_blocked": False,
        "guard_violations": [],
        "evidence_text": evidence_text[-max_stdout_chars:],
        "evidence_json": parse_json_from_text(evidence_text),
        "evidence_output_file": evidence_output_file,
    }


def code_execution_needs_repair(execution_result):
    if execution_result.get("guard_blocked"):
        return True
    if execution_result.get("timed_out"):
        return True
    if execution_result.get("returncode") != 0:
        return True
    return execution_result.get("evidence_json") is None

"""Workspace utilities for generated evidence code."""

import copy

from sem_agent import workspace as _sem_workspace


def _with_sem_fallbacks(conf):
    patched = dict(conf)
    fallback_names = [
        "allowed_files",
        "workspace_root",
        "code_timeout_seconds",
        "code_max_stdout_chars",
        "code_max_stderr_chars",
        "enable_code_guard",
        "forbidden_code_patterns",
    ]
    for name in fallback_names:
        code_key = f"code_{name}"
        sem_key = f"sem_{name}"
        if code_key not in patched and sem_key in patched:
            patched[code_key] = patched[sem_key]
    return patched


def prepare_workspace(conf, config_prefix="code"):
    return _sem_workspace.prepare_workspace(_with_sem_fallbacks(conf), config_prefix=config_prefix)


def build_source_manifest(workspace, current_bundle_policy):
    manifest = copy.deepcopy(_sem_workspace.build_source_manifest(workspace, current_bundle_policy))
    for source in manifest.get("sources", []):
        for container in (source, source.get("contract")):
            if not isinstance(container, dict):
                continue
            fields = container.get("fields")
            if isinstance(fields, dict):
                for key in list(fields):
                    lowered = str(key).lower()
                    if lowered in {"cate", "cate_id", "category", "category_id"}:
                        fields.pop(key, None)
    return manifest


def execute_generated_code(code, conf, workspace, output_file, script_name, config_prefix="code"):
    return _sem_workspace.execute_generated_code(
        code,
        _with_sem_fallbacks(conf),
        workspace,
        output_file,
        script_name,
        config_prefix=config_prefix,
    )


def execution_failed(result):
    return _sem_workspace.execution_needs_repair(result)

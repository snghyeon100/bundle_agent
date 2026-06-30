"""sem_agent/workspace.py — Self-contained workspace and code-execution utilities.

No dependency on progressive_signal_agent or simple_signal_agent.
"""

import os
import re
import shutil
import subprocess
import sys

from .common import parse_json_from_text


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

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
    r"results[\\\/]",
    r"true_option",
    r"true_indice",
    r"ground_truth",
    r"\bhit\b",
    r"Path\.home",
    r"os\.walk",
    r"\brglob\b",
    r"\bsocket\b",
    r"\brequests\b",
    r"\burllib\b",
    r"\bsubprocess\b",
]

# ---------------------------------------------------------------------------
# Source contracts
# ---------------------------------------------------------------------------

_BASE_CONTRACTS = {
    "count.json": {
        "entities": ["bundle", "user", "item"],
        "relations": [],
        "format": "JSON dataset counts",
    },
    "item_info.json": {
        "entities": ["item"],
        "relations": ["item has metadata"],
        "format": "JSON object keyed by string item_id",
    },
    "bi_train.txt": {
        "entities": ["bundle", "item"],
        "relations": ["bundle contains item"],
        "format": "comma-space rows: bundle_id, item_id, item_id, ...",
    },
    "ui_full.txt": {
        "entities": ["user", "item"],
        "relations": ["user interacts with item"],
        "format": "comma-space rows: user_id, item_id, item_id, ...",
    },
    "content_feature.pt": {
        "entities": ["item", "feature vector"],
        "relations": ["item has content representation"],
        "format": "torch tensor indexed by integer item_id when dimensions match item count",
    },
    "description_feature.pt": {
        "entities": ["item", "feature vector"],
        "relations": ["item has description representation"],
        "format": "torch tensor indexed by integer item_id when dimensions match item count",
    },
    "item_cf_feature.pt": {
        "entities": ["item", "feature vector"],
        "relations": ["item has UI-LightGCN representation"],
        "format": "PyTorch item embedding derived from ui_full.txt; validate object type and item-axis alignment",
    },
}

_DATASET_OVERRIDES = {
    "pog": {
        "item_info.json": {
            "format": "JSON object keyed by canonical integer item_id encoded as a string",
            "fields": {
                "id": "external product identifier; not the canonical integer item_id",
                "cate": "category identifier",
                #"pic": "item image URL",
                "title": "item title text",
            },
        },
        "content_feature.pt": {
            "modality": "image",
            "encoder": "BLIP image encoder (exact checkpoint/version not recorded)",
            "expected_shape": ["#I", 768],
            "row_alignment": "tensor row index equals canonical integer item_id",
            "normalization": "not recorded; inspect norms before assuming unit normalization",
            #"cross_file_comparability": (
            #    "Do not assume direct comparability with description_feature.pt unless "
            #    "the aligned BLIP image-text provenance is independently confirmed."
            #),
        },
        "description_feature.pt": {
            "modality": "text",
            "encoder": "text embedding encoder (exact model/checkpoint not recorded)",
            "expected_shape": ["#I", 768],
            "row_alignment": "tensor row index equals canonical integer item_id",
            "normalization": "not recorded; inspect norms before assuming unit normalization",
            #"cross_file_comparability": (
            #    "Do not assume direct comparability with content_feature.pt unless "
            #    "aligned image-text provenance is independently confirmed."
            #),
        },
    },
    "pog_dense": {
        "item_info.json": {
            "format": "JSON object keyed by canonical integer item_id encoded as a string",
            "fields": {
                "id": "external product identifier; not the canonical integer item_id",
                "cate_id": "category identifier",
                "title": "item title text",
            },
        },
        "content_feature.pt": {
            "modality": "image",
            "encoder": "BLIP image encoder (exact checkpoint/version not recorded)",
            "expected_shape": ["#I", 768],
            "row_alignment": "tensor row index equals canonical integer item_id",
            "normalization": "not recorded; inspect norms before assuming unit normalization",
            
        },
        "description_feature.pt": {
            "modality": "text",
            "encoder": "text embedding encoder (exact model/checkpoint not recorded)",
            "expected_shape": ["#I", 768],
            "row_alignment": "tensor row index equals canonical integer item_id",
            "normalization": "not recorded; inspect norms before assuming unit normalization",
            
        },
    },
    "spotify": {
        "item_info.json": {
            "format": "JSON object keyed by canonical integer item_id encoded as a string",
            "fields": {
                "pos": "dataset item position/index metadata",
                "artist_name": "artist display name",
                "track_uri": "external Spotify track URI; not the canonical integer item_id",
                "artist_uri": "external Spotify artist URI",
                "track_name": "track title",
                "album_uri": "external Spotify album URI",
                "duration_ms": "track duration in milliseconds",
                "album_name": "album title",
            },
        },
        "content_feature.pt": {
            "modality": "audio",
            "encoder": "CLAP audio encoder (exact checkpoint/version not recorded)",
            "expected_shape": ["#I", 512],
            "row_alignment": "tensor row index equals canonical integer item_id",
            "normalization": "not recorded; inspect norms before assuming unit normalization",
        },
        "description_feature.pt": {
            "modality": "text",
            "encoder": "text embedding encoder (exact model/checkpoint not recorded)",
            "expected_shape": ["#I", 512],
            "row_alignment": "tensor row index equals canonical integer item_id",
            "normalization": "not recorded; inspect norms before assuming unit normalization",
        },
    },
    "spotify_sparse": {
        "item_info.json": {
            "format": "JSON object keyed by canonical integer item_id encoded as a string",
            "fields": {
                "pos": "dataset item position/index metadata",
                "artist_name": "artist display name",
                "track_uri": "external Spotify track URI; not the canonical integer item_id",
                "artist_uri": "external Spotify artist URI",
                "track_name": "track title",
                "album_uri": "external Spotify album URI",
                "duration_ms": "track duration in milliseconds",
                "album_name": "album title",
            },
        },
        "content_feature.pt": {
            "modality": "audio",
            "encoder": "CLAP audio encoder (exact checkpoint/version not recorded)",
            "expected_shape": ["#I", 512],
            "row_alignment": "tensor row index equals canonical integer item_id",
            "normalization": "not recorded; inspect norms before assuming unit normalization",
        },
        "description_feature.pt": {
            "modality": "text",
            "encoder": "text embedding encoder (exact model/checkpoint not recorded)",
            "expected_shape": ["#I", 512],
            "row_alignment": "tensor row index equals canonical integer item_id",
            "normalization": "not recorded; inspect norms before assuming unit normalization",
        },
    },
}


def _source_contract(filename, dataset=None):
    if filename.endswith("_LightGCN_bi_feature.pt"):
        contract = {
            "entities": ["item", "feature vector"],
            "relations": ["item has BI-LightGCN representation"],
            "format": "PyTorch item embedding derived from bi_train.txt; validate object type and item-axis alignment",
        }
    else:
        contract = dict(_BASE_CONTRACTS.get(filename, {}))
    override = _DATASET_OVERRIDES.get(str(dataset or "").lower(), {}).get(filename, {})
    contract.update(override)
    return contract


# ---------------------------------------------------------------------------
# Config helper
# ---------------------------------------------------------------------------

def _setting(conf, prefix, name, default):
    """Look up conf[f'{prefix}_{name}'], fall back to default."""
    key = f"{prefix}_{name}"
    if key in conf:
        return conf[key]
    return default


# ---------------------------------------------------------------------------
# Workspace preparation
# ---------------------------------------------------------------------------

def _copy_if_needed(source, destination):
    if os.path.exists(destination):
        ss = os.stat(source)
        ds = os.stat(destination)
        if ss.st_size == ds.st_size and int(ss.st_mtime) <= int(ds.st_mtime):
            return False
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    shutil.copy2(source, destination)
    return True


def prepare_workspace(conf, config_prefix="sem"):
    data_root = os.path.abspath(os.path.join(conf["data_path"], conf["dataset"]))
    workspace_root = os.path.abspath(
        _setting(conf, config_prefix, "workspace_root", "./agent_workspaces")
    )
    workspace_dir = os.path.join(workspace_root, conf["dataset"])
    data_dir = os.path.join(workspace_dir, "data")
    output_dir = os.path.join(workspace_dir, "output")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    available = []
    copied = []
    allowed_files = _setting(conf, config_prefix, "allowed_files", DEFAULT_ALLOWED_FILES)
    for raw_name in allowed_files:
        filename = str(raw_name).replace("{dataset}", str(conf["dataset"]))
        source = os.path.join(data_root, filename)
        if not os.path.isfile(source):
            continue
        destination = os.path.join(data_dir, filename)
        if _copy_if_needed(source, destination):
            copied.append(filename)
        available.append({"name": filename, "path": f"data/{filename}"})

    return {
        "dataset": conf["dataset"],
        "workspace_dir": workspace_dir,
        "data_dir": data_dir,
        "output_dir": output_dir,
        "files": available,
        "copied_files": copied,
    }


def build_source_manifest(workspace, current_bundle_policy):
    sources = []
    for entry in workspace["files"]:
        contract = _source_contract(entry["name"], workspace.get("dataset"))
        sources.append({"name": entry["name"], "path": entry["path"], **contract})
    return {
        "sources": sources,
        "generic_transformations": [
            "filter",
            "invert relation",
            "join relations",
            "expand neighborhood",
            "aggregate",
            "compare distributions",
            "retrieve representative examples",
            "test robustness",
        ],
        "typed_id_rules": [
            "In BI rows, values[0] is bundle_id and values[1:] are item_ids.",
            "In UI rows, values[0] is user_id and values[1:] are item_ids.",
            "Bundle, user, and item IDs are distinct entity types even when integers match.",
            "Only item IDs may index item_info or item feature tensors.",
        ],
        "current_bundle_train_context_policy": current_bundle_policy,
        "leakage_rules": [
            "Use only the listed workspace sources.",
            "Never access test ground truth, full bundle files, result files, labels, predictions, or hits.",
            "Do not use network access or paths outside the workspace.",
        ],
    }


# ---------------------------------------------------------------------------
# Code guard & execution
# ---------------------------------------------------------------------------

def guard_generated_code(code, conf, config_prefix="sem"):
    if not bool(_setting(conf, config_prefix, "enable_code_guard", True)):
        return []
    patterns = _setting(
        conf, config_prefix, "forbidden_code_patterns", DEFAULT_FORBIDDEN_CODE_PATTERNS
    )
    return [p for p in patterns if re.search(p, code, flags=re.IGNORECASE)]


def execute_generated_code(code, conf, workspace, output_file, script_name, config_prefix="sem"):
    violations = guard_generated_code(code, conf, config_prefix=config_prefix)
    if violations:
        return {
            "returncode": None,
            "stdout": "",
            "stderr": f"Generated code blocked by guard patterns: {violations}",
            "timed_out": False,
            "guard_blocked": True,
            "guard_violations": violations,
            "evidence_json": None,
            "evidence_output_file": output_file,
        }

    script_path = os.path.join(workspace["workspace_dir"], script_name)
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(code.rstrip() + "\n")

    output_path = os.path.join(workspace["workspace_dir"], output_file.replace("/", os.sep))
    if os.path.exists(output_path):
        os.remove(output_path)

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    timeout = int(_setting(conf, config_prefix, "code_timeout_seconds", 60))
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
        max_stdout = int(_setting(conf, config_prefix, "code_max_stdout_chars", 24000))
        max_stderr = int(_setting(conf, config_prefix, "code_max_stderr_chars", 10000))
        stdout = completed.stdout[-max_stdout:]
        stderr = completed.stderr[-max_stderr:]
        returncode = completed.returncode
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        returncode = None
        timed_out = True

    evidence_text = ""
    if os.path.isfile(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
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
        "evidence_json": parse_json_from_text(evidence_text),
        "evidence_output_file": output_file,
    }


def execution_needs_repair(result):
    return (
        bool(result.get("guard_blocked"))
        or bool(result.get("timed_out"))
        or result.get("returncode") != 0
        or not isinstance(result.get("evidence_json"), dict)
    )

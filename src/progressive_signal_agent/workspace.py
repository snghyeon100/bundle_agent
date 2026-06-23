import os
import re
import shutil
import subprocess
import sys

from .common import parse_json_from_text


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
    r"results[\\/]",
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


SOURCE_CONTRACTS = {
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


DATASET_SOURCE_CONTRACTS = {
    "pog": {
        "item_info.json": {
            "format": "JSON object keyed by canonical integer item_id encoded as a string",
            "fields": {
                "id": "external product identifier; not the canonical integer item_id",
                "cate": "category identifier",
                "pic": "item image URL",
                "title": "item title text",
            },
        },
        "content_feature.pt": {
            "modality": "image",
            "encoder": "BLIP image encoder (exact checkpoint/version not recorded)",
            "expected_shape": ["#I", 768],
            "row_alignment": "tensor row index equals canonical integer item_id",
            "normalization": "not recorded; inspect norms before assuming unit normalization",
            "cross_file_comparability": (
                "Do not assume direct comparability with description_feature.pt unless the aligned BLIP "
                "image-text provenance is independently confirmed."
            ),
        },
        "description_feature.pt": {
            "modality": "text",
            "encoder": "text embedding encoder (exact model/checkpoint not recorded)",
            "expected_shape": ["#I", 768],
            "row_alignment": "tensor row index equals canonical integer item_id",
            "normalization": "not recorded; inspect norms before assuming unit normalization",
            "cross_file_comparability": (
                "Do not assume direct comparability with content_feature.pt unless aligned image-text "
                "provenance is independently confirmed."
            ),
        },
    },
    "pog_dense": {
        "item_info.json": {
            "format": "JSON object keyed by canonical integer item_id encoded as a string",
            "fields": {
                "id": "external product identifier; not the canonical integer item_id",
                "cate_id": "category identifier",
                "pic_url": "item image URL",
                "title": "item title text",
            },
        },
        "content_feature.pt": {
            "modality": "image",
            "encoder": "BLIP image encoder (exact checkpoint/version not recorded)",
            "expected_shape": ["#I", 768],
            "row_alignment": "tensor row index equals canonical integer item_id",
            "normalization": "not recorded; inspect norms before assuming unit normalization",
            "cross_file_comparability": (
                "Do not assume direct comparability with description_feature.pt unless the aligned BLIP "
                "image-text provenance is independently confirmed."
            ),
        },
        "description_feature.pt": {
            "modality": "text",
            "encoder": "text embedding encoder (exact model/checkpoint not recorded)",
            "expected_shape": ["#I", 768],
            "row_alignment": "tensor row index equals canonical integer item_id",
            "normalization": "not recorded; inspect norms before assuming unit normalization",
            "cross_file_comparability": (
                "Do not assume direct comparability with content_feature.pt unless aligned image-text "
                "provenance is independently confirmed."
            ),
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
            "cross_file_comparability": (
                "Compare with description_feature.pt only if that file is confirmed to use the aligned "
                "CLAP text encoder from the same embedding space."
            ),
        },
        "description_feature.pt": {
            "modality": "text",
            "encoder": "text embedding encoder (exact model/checkpoint not recorded)",
            "expected_shape": ["#I", 512],
            "row_alignment": "tensor row index equals canonical integer item_id",
            "normalization": "not recorded; inspect norms before assuming unit normalization",
            "cross_file_comparability": (
                "Direct audio-text comparison is unverified until aligned CLAP text-encoder provenance "
                "is confirmed."
            ),
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
            "cross_file_comparability": (
                "Compare with description_feature.pt only if that file is confirmed to use the aligned "
                "CLAP text encoder from the same embedding space."
            ),
        },
        "description_feature.pt": {
            "modality": "text",
            "encoder": "text embedding encoder (exact model/checkpoint not recorded)",
            "expected_shape": ["#I", 512],
            "row_alignment": "tensor row index equals canonical integer item_id",
            "normalization": "not recorded; inspect norms before assuming unit normalization",
            "cross_file_comparability": (
                "Direct audio-text comparison is unverified until aligned CLAP text-encoder provenance "
                "is confirmed."
            ),
        },
    },
}


def _source_contract(filename, dataset=None):
    if filename.endswith("_LightGCN_bi_feature.pt"):
        contract = {
            "entities": ["item", "feature vector"],
            "relations": ["item has BI-LightGCN representation"],
            "format": (
                "PyTorch item embedding derived from bi_train.txt; validate object type and item-axis alignment"
            ),
        }
    else:
        contract = dict(SOURCE_CONTRACTS.get(filename, {}))
    dataset_contract = DATASET_SOURCE_CONTRACTS.get(str(dataset or "").lower(), {}).get(
        filename,
        {},
    )
    contract.update(dataset_contract)
    return contract


def _method_setting(conf, prefix, name, default):
    key = f"{prefix}_{name}"
    if key in conf:
        return conf[key]
    if prefix != "psd":
        legacy_key = f"psd_{name}"
        if legacy_key in conf:
            return conf[legacy_key]
    return default


def _copy_if_needed(source, destination):
    if os.path.exists(destination):
        source_stat = os.stat(source)
        destination_stat = os.stat(destination)
        if source_stat.st_size == destination_stat.st_size and int(source_stat.st_mtime) <= int(
            destination_stat.st_mtime
        ):
            return False
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    shutil.copy2(source, destination)
    return True


def prepare_workspace(conf, config_prefix="psd"):
    data_root = os.path.abspath(os.path.join(conf["data_path"], conf["dataset"]))
    workspace_root = os.path.abspath(
        _method_setting(conf, config_prefix, "workspace_root", "./agent_workspaces")
    )
    workspace_dir = os.path.join(workspace_root, conf["dataset"])
    data_dir = os.path.join(workspace_dir, "data")
    output_dir = os.path.join(workspace_dir, "output")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    available = []
    copied = []
    configured_files = _method_setting(
        conf,
        config_prefix,
        "allowed_files",
        DEFAULT_ALLOWED_FILES,
    )
    for configured_filename in configured_files:
        filename = str(configured_filename).replace("{dataset}", str(conf["dataset"]))
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


def guard_generated_code(code, conf, config_prefix="psd"):
    if not bool(_method_setting(conf, config_prefix, "enable_code_guard", True)):
        return []
    patterns = _method_setting(
        conf,
        config_prefix,
        "forbidden_code_patterns",
        DEFAULT_FORBIDDEN_CODE_PATTERNS,
    )
    return [pattern for pattern in patterns if re.search(pattern, code, flags=re.IGNORECASE)]


def execute_generated_code(
    code,
    conf,
    workspace,
    output_file,
    script_name,
    config_prefix="psd",
):
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
    with open(script_path, "w", encoding="utf-8") as handle:
        handle.write(code.rstrip() + "\n")

    output_path = os.path.join(workspace["workspace_dir"], output_file.replace("/", os.sep))
    if os.path.exists(output_path):
        os.remove(output_path)

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    timeout = int(_method_setting(conf, config_prefix, "code_timeout_seconds", 45))
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
        stdout = completed.stdout[
            -int(_method_setting(conf, config_prefix, "code_max_stdout_chars", 24000)) :
        ]
        stderr = completed.stderr[
            -int(_method_setting(conf, config_prefix, "code_max_stderr_chars", 10000)) :
        ]
        returncode = completed.returncode
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        returncode = None
        timed_out = True

    evidence_text = ""
    if os.path.isfile(output_path):
        with open(output_path, "r", encoding="utf-8") as handle:
            evidence_text = handle.read()
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

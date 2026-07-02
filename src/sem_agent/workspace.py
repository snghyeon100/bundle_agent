"""sem_agent/workspace.py — Self-contained workspace and code-execution utilities.

No dependency on progressive_signal_agent or simple_signal_agent.
"""

import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter

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
# Lightweight analysis reconnaissance
# ---------------------------------------------------------------------------

def _safe_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _item_metadata(meta):
    if not isinstance(meta, dict):
        return {}
    out = {}
    for key in ("title", "cate", "cate_id", "track_name", "artist_name", "album_name"):
        value = meta.get(key)
        if value is not None and str(value).strip():
            out[key] = str(value).strip()
    if not out.get("cate") and out.get("cate_id"):
        out["cate"] = out["cate_id"]
    return out


def _support_bucket(count):
    if count is None:
        return "unknown"
    if count <= 0:
        return "none"
    if count <= 5:
        return "low"
    if count <= 50:
        return "medium"
    return "high"


def _ratio_bucket(ratio):
    if ratio is None:
        return "unknown"
    if ratio <= 0:
        return "none"
    if ratio < 0.05:
        return "low"
    if ratio < 0.20:
        return "medium"
    return "high"


def _coverage_bucket(count, total):
    if not total:
        return "none"
    ratio = count / total
    if count <= 0:
        return "none"
    if ratio < 0.25:
        return "low"
    if ratio < 0.75:
        return "medium"
    return "high"


def build_analysis_recon(workspace, case_view, max_scan_rows=None):
    """Build small, train-safe data reconnaissance for Stage 0 planning.

    This is deliberately not final evidence. It calibrates the analysis prompt
    with source sparsity, role/category structure, and obvious fallback needs.
    It only reads files already present in the sem workspace.
    """
    data_dir = workspace.get("data_dir", "")
    item_info_path = os.path.join(data_dir, "item_info.json")
    bi_train_path = os.path.join(data_dir, "bi_train.txt")

    partial_ids = [_safe_int(v) for v in case_view.get("partial_item_ids", [])]
    partial_ids = [v for v in partial_ids if v is not None]
    candidates = []
    for candidate in case_view.get("candidates", []):
        item_id = _safe_int(candidate.get("item_id"))
        if item_id is not None:
            candidates.append({"label": str(candidate.get("label", "")), "item_id": item_id})
    target_ids = partial_ids + [c["item_id"] for c in candidates]
    target_set = set(target_ids)

    item_info = {}
    if os.path.isfile(item_info_path):
        with open(item_info_path, "r", encoding="utf-8") as handle:
            item_info = json.load(handle)

    metadata = {}
    cate_by_item = {}
    for item_id in target_ids:
        meta = _item_metadata(item_info.get(str(item_id), {}))
        metadata[str(item_id)] = meta
        if meta.get("cate"):
            cate_by_item[item_id] = meta["cate"]

    item_bundle_counts = {item_id: 0 for item_id in target_ids}
    candidate_cooccurs_with_partial = {c["label"]: 0 for c in candidates}
    exact_partial_set_bundle_count = 0
    partial_cates = {cate_by_item[item_id] for item_id in partial_ids if item_id in cate_by_item}
    partial_category_bundle_counts = {cate: 0 for cate in partial_cates}
    partial_category_context_bundle_rows = 0
    candidate_category_in_partial_context_rows = {c["label"]: 0 for c in candidates}
    bundle_rows_scanned = 0

    if os.path.isfile(bi_train_path):
        with open(bi_train_path, "r", encoding="utf-8") as handle:
            for line in handle:
                if max_scan_rows is not None and bundle_rows_scanned >= max_scan_rows:
                    break
                values = [part.strip() for part in line.strip().split(",") if part.strip()]
                if len(values) < 2:
                    continue
                item_ids = []
                for raw in values[1:]:
                    item_id = _safe_int(raw)
                    if item_id is not None:
                        item_ids.append(item_id)
                if not item_ids:
                    continue
                bundle_rows_scanned += 1
                item_set = set(item_ids)
                for item_id in target_set.intersection(item_set):
                    item_bundle_counts[item_id] += 1
                if partial_ids and all(item_id in item_set for item_id in partial_ids):
                    exact_partial_set_bundle_count += 1
                if any(item_id in item_set for item_id in partial_ids):
                    for candidate in candidates:
                        if candidate["item_id"] in item_set:
                            candidate_cooccurs_with_partial[candidate["label"]] += 1

                if partial_cates:
                    row_cates = [
                        cate_by_item.get(item_id)
                        or _item_metadata(item_info.get(str(item_id), {})).get("cate", "")
                        for item_id in item_ids
                    ]
                    row_cates = [cate for cate in row_cates if cate]
                    row_cate_counts = Counter(row_cates)
                    for cate in partial_cates:
                        if row_cate_counts.get(cate, 0) > 0:
                            partial_category_bundle_counts[cate] += 1
                    has_partial_category_context = all(
                        row_cate_counts.get(cate, 0) > 0 for cate in partial_cates
                    )
                    if has_partial_category_context:
                        partial_category_context_bundle_rows += 1
                        for candidate in candidates:
                            cand_cate = cate_by_item.get(candidate["item_id"])
                            if cand_cate and cand_cate in row_cate_counts:
                                candidate_category_in_partial_context_rows[candidate["label"]] += 1

    partial_item_diagnostic = {
        str(item_id): {
            "item_support": _support_bucket(item_bundle_counts.get(item_id, 0)),
            "category_support": _support_bucket(
                partial_category_bundle_counts.get(cate_by_item.get(item_id, ""), 0)
            ),
        }
        for item_id in partial_ids
    }
    candidate_level_diagnostic = {}
    for candidate in candidates:
        cand_cate = cate_by_item.get(candidate["item_id"], "")
        matching_partials = [
            item_id for item_id in partial_ids
            if cate_by_item.get(item_id, "") and cate_by_item.get(item_id, "") == cand_cate
        ]
        rows = candidate_category_in_partial_context_rows.get(candidate["label"], 0)
        ratio = rows / partial_category_context_bundle_rows if partial_category_context_bundle_rows else None
        candidate_level_diagnostic[candidate["label"]] = {
            "item_support": _support_bucket(item_bundle_counts.get(candidate["item_id"], 0)),
            "direct_partial_cobundle_support": _support_bucket(
                candidate_cooccurs_with_partial.get(candidate["label"], 0)
            ),
            "category_overlap_with_partial": "yes" if matching_partials else "no",
            "candidate_category": cand_cate,
            "category_context_support": _support_bucket(rows),
            "relative_category_context_support": _ratio_bucket(ratio),
        }

    num_candidates = len(candidates)
    num_candidates_with_exact_support = sum(
        1 for candidate in candidates
        if item_bundle_counts.get(candidate["item_id"], 0) > 0
    )
    num_candidates_directly_cobundled = sum(
        1 for candidate in candidates
        if candidate_cooccurs_with_partial.get(candidate["label"], 0) > 0
    )
    num_candidates_sharing_partial_category = sum(
        1 for candidate in candidates
        if candidate_level_diagnostic.get(candidate["label"], {}).get("category_overlap_with_partial") == "yes"
    )
    candidate_categories = {
        cate_by_item.get(candidate["item_id"], "")
        for candidate in candidates
        if cate_by_item.get(candidate["item_id"], "")
    }
    category_diversity = "low"
    if len(candidate_categories) >= 5:
        category_diversity = "high"
    elif len(candidate_categories) >= 3:
        category_diversity = "medium"

    if num_candidates_directly_cobundled <= 0:
        direct_feasibility = "none"
    elif num_candidates_directly_cobundled == num_candidates:
        direct_feasibility = "available"
    elif num_candidates_directly_cobundled <= max(1, num_candidates // 3):
        direct_feasibility = "limited"
    else:
        direct_feasibility = "mixed"

    return {
        "purpose": "retrieval planning only; not candidate scoring",
        "recon_legend": {
            "support_labels": "none, low, medium, high are bucketed diagnostics derived from train bundle/category counts.",
            "partial_item_support": "Bucketed support for exact partial items in train bundles.",
            "exact_partial_set_support": "Bucketed support for train bundles containing the full exact partial set.",
            "partial_category_context_support": "Bucketed support for train bundles containing the partial item category context.",
            "category_overlap_with_partial": "Whether a candidate category matches any partial item category.",
            "relative_category_context_support": "Bucketed ratio of candidate-category rows inside the partial-category context.",
            "direct_cobundle_feasibility": "Whether exact candidate-partial co-bundle lookup is feasible across candidates.",
            "exact_item_support_coverage": "Bucketed fraction of candidates with exact train-bundle support.",
            "category_diversity": "Bucketed diversity of candidate categories.",
            "same_category_candidate_presence": "Whether any candidate shares a partial item category.",
        },
        "sources_read": [
            name for name, path in (
                ("item_info.json", item_info_path),
                ("bi_train.txt", bi_train_path),
            )
            if os.path.isfile(path)
        ],
        "sample_level_diagnostic": {
            "partial_item_support": _support_bucket(
                max((item_bundle_counts.get(item_id, 0) for item_id in partial_ids), default=0)
            ),
            "exact_partial_set_support": _support_bucket(exact_partial_set_bundle_count),
            "partial_category_context_support": _support_bucket(partial_category_context_bundle_rows),
            "partial_category_specificity": (
                "single_category"
                if len(partial_cates) == 1
                else "multi_category"
                if len(partial_cates) > 1
                else "unknown"
            ),
        },
        "partial_item_diagnostic": partial_item_diagnostic,
        "candidate_level_diagnostic": candidate_level_diagnostic,
        "aggregate_candidate_diagnostic": {
            "direct_cobundle_feasibility": direct_feasibility,
            "exact_item_support_coverage": _coverage_bucket(
                num_candidates_with_exact_support, num_candidates
            ),
            "category_diversity": category_diversity,
            "same_category_candidate_presence": (
                "present" if num_candidates_sharing_partial_category else "none"
            ),
        },
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

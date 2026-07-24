"""Core orchestration for compact operator induction and composition."""

import json
import os
import random

from code.common import parse_json_from_text
from code.pipeline import _item_profile, build_decision_case
from code.workspace import build_source_manifest, prepare_workspace
from dataset import BundleZeroShotDataset

from .prompts import clustering_prompt, composition_prompt, induction_prompt
from .schemas import (
    OPERATOR_FIELDS,
    normalize_library,
    operator_connection_diagnostics,
    validate_induction_result,
    validate_operator,
    validate_operator_library,
    validate_workflow_result,
)


def _semantic_operator_view(operator, *, include_identity=False):
    """Drop source/implementation fields before persistence or semantic clustering."""
    if not isinstance(operator, dict):
        return operator
    fields = list(OPERATOR_FIELDS)
    if include_identity:
        fields.extend(("operator_id", "origin_case_id"))
    return {field: operator[field] for field in fields if field in operator}


def sample_validation_cases(conf, count, seed=None):
    """Draw deterministic discovery cases from existing bi_valid files, never test."""
    discovery_conf = dict(conf)
    discovery_conf["toy_eval"] = -1
    dataset = BundleZeroShotDataset(discovery_conf, split="valid")
    samples = dataset.get_eval_samples()
    requested = int(count)
    if requested < 1:
        raise ValueError("operator discovery count must be at least 1")
    if requested > len(samples):
        raise ValueError(
            f"operator discovery count {requested} exceeds {len(samples)} validation samples"
        )
    default_seed = conf.get("seed", 45) if seed is None else seed
    rng = random.Random(int(conf.get("operator_discovery_seed", default_seed)))
    indices = sorted(rng.sample(range(len(samples)), requested))
    return [samples[index] for index in indices]


def build_discovery_case(sample, conf):
    info_path = os.path.join(conf["data_path"], conf["dataset"], "item_info.json")
    with open(info_path, "r", encoding="utf-8") as handle:
        item_info = json.load(handle)
    gt_item_id = int(sample["true_indice"])
    gt_profile = _item_profile(gt_item_id, item_info, conf["dataset"])
    labels = [chr(ord("A") + index) for index in range(len(sample["candidate_indices"]))]
    candidates = [
        {"label": label, **_item_profile(item_id, item_info, conf["dataset"])}
        for label, item_id in zip(labels, sample["candidate_indices"])
    ]
    return {
        "case_id": f"bundle_{sample['bundle_id']}",
        "dataset": conf["dataset"],
        "bundle_id": int(sample["bundle_id"]),
        "partial_items": [
            _item_profile(item_id, item_info, conf["dataset"])
            for item_id in sample["input_indices"]
        ],
        "candidates": candidates,
        "ground_truth": {
            "label": str(sample["true_option_char"]),
            "item_id": gt_item_id,
            "text": gt_profile.get("text", ""),
            "metadata": gt_profile.get("metadata", {}),
        },
    }


def build_operator_source_manifest(conf):
    workspace = prepare_workspace(conf, config_prefix="code")
    manifest = build_source_manifest(
        workspace,
        str(conf.get("code_current_bundle_train_context_policy", "allow")),
    )
    return workspace, manifest


def operator_capability_manifest(source_manifest, dataset):
    """Convert concrete workspace files into path-free semantic capabilities."""
    capabilities = []
    for source in source_manifest.get("sources", []):
        name = str(source.get("name", ""))
        lower = name.lower()
        if lower == "count.json":
            capability_id = "dataset_statistics"
            operations = ["inspect entity counts", "check index ranges"]
        elif lower == "item_info.json":
            capability_id = "item_metadata"
            operations = ["lookup item attributes", "filter attributes", "extract semantic cues"]
        elif lower == "bi_train.txt":
            capability_id = "bundle_item_history"
            operations = [
                "invert bundle-item relations",
                "retrieve co-occurrence",
                "expand item neighborhoods",
                "compute conditional frequency",
                "compute association lift",
            ]
        elif lower == "ui_full.txt":
            capability_id = "user_item_history"
            operations = [
                "invert user-item relations",
                "retrieve shared audiences",
                "compare candidate audiences",
                "aggregate user-group support",
            ]
        elif lower == "content_feature.pt":
            capability_id = "item_content_embedding"
            operations = [
                "lookup item vectors",
                "compute similarity or distance",
                "build bundle centroid",
                "compute candidate margins",
            ]
        elif lower == "description_feature.pt":
            capability_id = "item_description_embedding"
            operations = [
                "lookup item vectors",
                "compute similarity or distance",
                "build bundle centroid",
                "compute candidate margins",
            ]
        elif lower == "item_cf_feature.pt":
            capability_id = "user_collaborative_embedding"
            operations = [
                "lookup item vectors",
                "compare collaborative neighborhoods",
                "compute similarity or candidate margins",
            ]
        elif lower.endswith("_lightgcn_bi_feature.pt"):
            capability_id = "bundle_collaborative_embedding"
            operations = [
                "lookup item vectors",
                "compare bundle-affiliation representations",
                "compute similarity or candidate margins",
            ]
        else:
            continue
        capability = {
            "id": capability_id,
            "entities": source.get("entities", []),
            "relations": source.get("relations", []),
            "format": source.get("format", ""),
            "available_operations": operations,
        }
        for field in ("fields", "modality", "expected_shape", "row_alignment"):
            if field in source:
                capability[field] = source[field]
        capabilities.append(capability)
    return {
        "schema_version": "operator_source_capabilities_v1",
        "dataset": str(dataset),
        "capabilities": capabilities,
        "generic_transformations": source_manifest.get("generic_transformations", []),
        "typed_id_rules": source_manifest.get("typed_id_rules", []),
        "leakage_rules": source_manifest.get("leakage_rules", []),
        "current_bundle_train_context_policy": source_manifest.get(
            "current_bundle_train_context_policy"
        ),
    }


def build_operator_capability_manifest(conf):
    """Prepare sources once and expose a path-free manifest for induction."""
    workspace, source_manifest = build_operator_source_manifest(conf)
    capabilities = operator_capability_manifest(source_manifest, conf["dataset"])
    return workspace, source_manifest, capabilities


def _capability_names(source_capabilities):
    return {
        capability.get("id")
        for capability in source_capabilities.get("capabilities", [])
        if isinstance(capability, dict)
        and isinstance(capability.get("id"), str)
        and capability["id"].strip()
    }


async def _request_json(call_text, prompt, step_name):
    raw = await call_text(prompt, step_name)
    parsed = parse_json_from_text(raw)
    if not isinstance(parsed, dict):
        raise ValueError(f"{step_name} did not return a JSON object")
    return raw, parsed


async def induce_raw_operators(
    samples,
    conf,
    call_text,
    *,
    source_capabilities,
    operators_per_case=None,
    trace_callback=None,
):
    """Induce compact source-aware operators from labeled validation samples."""
    per_case = int(operators_per_case or conf.get("operator_induction_count", 4))
    if per_case < 1:
        raise ValueError("operator_induction_count must be at least 1")
    if not isinstance(source_capabilities, dict) or not source_capabilities.get(
        "capabilities"
    ):
        raise ValueError("source_capabilities must contain available capabilities")
    allowed_source_names = _capability_names(source_capabilities)

    raw_operators = []
    induction_traces = []
    discovery_cases = []
    for sample in samples:
        case = build_discovery_case(sample, conf)
        discovery_cases.append(case)
        prompt = induction_prompt(
            case,
            source_capabilities,
            per_case,
            text_only=bool(conf.get("operator_prompt_text_only", True)),
        )
        step_name = f"operator induction for {case['case_id']}"
        raw = await call_text(prompt, step_name)
        result = parse_json_from_text(raw)
        issues = (
            validate_induction_result(
                result,
                expected_count=per_case,
                allowed_source_names=allowed_source_names,
            )
            if isinstance(result, dict)
            else ["induction result must be a JSON object"]
        )
        trace = {
            "case_id": case["case_id"],
            "prompt": prompt,
            "raw_response": raw,
            "parsed_response": result,
            "validation_issues": issues,
            "connection_diagnostics": operator_connection_diagnostics(
                result.get("operators", []) if isinstance(result, dict) else []
            ),
            "operators": [],
        }
        if issues:
            if trace_callback:
                trace_callback(trace)
            raise ValueError(
                f"invalid operator induction for {case['case_id']}: " + " | ".join(issues)
            )
        case_operators = []
        for index, operator in enumerate(result["operators"], start=1):
            enriched = _semantic_operator_view(operator)
            enriched["operator_id"] = f"{case['case_id']}__op{index}"
            enriched["origin_case_id"] = case["case_id"]
            case_operators.append(enriched)
            raw_operators.append(enriched)
        trace["operators"] = case_operators
        induction_traces.append(trace)
        if trace_callback:
            trace_callback(trace)

    return {
        "source_capabilities": source_capabilities,
        "discovery_cases": discovery_cases,
        "induction_traces": induction_traces,
        "raw_operators": raw_operators,
        "operators_per_case": per_case,
    }


async def cluster_raw_operators(
    raw_operators,
    conf,
    call_text,
    *,
    min_library_size=None,
    max_library_size=None,
    trace_callback=None,
):
    """Cluster a previously saved raw operator pool into a refined library."""
    if not isinstance(raw_operators, list) or not raw_operators:
        raise ValueError("raw operator pool must be a non-empty list")
    pool_issues = []
    operator_ids = []
    for index, operator in enumerate(raw_operators):
        pool_issues.extend(
            f"raw_operators[{index}]: {issue}" for issue in validate_operator(operator)
        )
        operator_id = operator.get("operator_id") if isinstance(operator, dict) else None
        if not isinstance(operator_id, str) or not operator_id.strip():
            pool_issues.append(f"raw_operators[{index}].operator_id must be non-empty")
        else:
            operator_ids.append(operator_id)
    if len(operator_ids) != len(set(operator_ids)):
        pool_issues.append("raw operator_id values must be unique")
    if pool_issues:
        raise ValueError("invalid raw operator pool: " + " | ".join(pool_issues))

    minimum = int(min_library_size or conf.get("operator_library_min_size", 8))
    maximum = int(max_library_size or conf.get("operator_library_max_size", 12))
    if minimum < 1 or maximum < minimum:
        raise ValueError("operator library size bounds are invalid")

    semantic_pool = [
        _semantic_operator_view(operator, include_identity=True)
        for operator in raw_operators
    ]
    allowed_source_names = {
        source
        for operator in raw_operators
        if isinstance(operator, dict)
        for source in operator.get("sources", [])
        if isinstance(source, str) and source.strip()
    }
    c_prompt = clustering_prompt(semantic_pool, conf["dataset"], minimum, maximum)
    clustering_raw = await call_text(c_prompt, "functional operator clustering")
    clustering_result = parse_json_from_text(clustering_raw)
    library = normalize_library(clustering_result, conf["dataset"])
    issues = (
        validate_operator_library(
            library,
            allowed_source_names=allowed_source_names,
        )
        if isinstance(clustering_result, dict)
        else ["clustering result must be a JSON object"]
    )
    if isinstance(library.get("operators"), list) and not minimum <= len(
        library["operators"]
    ) <= maximum:
        issues.append(
            f"clustered library has {len(library['operators'])} operators; expected {minimum}..{maximum}"
        )
    raw_ids = {operator["operator_id"] for operator in raw_operators}
    if isinstance(library.get("clusters"), list) and all(
        isinstance(cluster, dict) and isinstance(cluster.get("member_ids"), list)
        for cluster in library["clusters"]
    ):
        member_ids = [
            member for cluster in library["clusters"] for member in cluster["member_ids"]
        ]
        if set(member_ids) != raw_ids or len(member_ids) != len(set(member_ids)):
            issues.append("cluster member_ids must cover every raw operator exactly once")
    if isinstance(library.get("operators"), list):
        derived_ids = {
            member
            for operator in library["operators"]
            if isinstance(operator, dict)
            for member in operator.get("derived_from", [])
        }
        if not derived_ids <= raw_ids:
            issues.append("operator.derived_from contains an unknown raw operator_id")

    trace = {
        "clustering_prompt": c_prompt,
        "clustering_raw_response": clustering_raw,
        "parsed_response": clustering_result,
        "validation_issues": issues,
        "library": library,
    }
    if trace_callback:
        trace_callback(trace)
    if issues:
        raise ValueError("invalid clustered operator library: " + " | ".join(issues))

    return trace


async def discover_operator_library(
    samples,
    conf,
    call_text,
    *,
    source_capabilities,
    operators_per_case=None,
    min_library_size=None,
    max_library_size=None,
):
    """Backward-compatible combined induction and clustering helper."""
    induction = await induce_raw_operators(
        samples,
        conf,
        call_text,
        source_capabilities=source_capabilities,
        operators_per_case=operators_per_case,
    )
    clustering = await cluster_raw_operators(
        induction["raw_operators"],
        conf,
        call_text,
        min_library_size=min_library_size,
        max_library_size=max_library_size,
    )

    return {
        **induction,
        **clustering,
    }


async def compose_workflows(
    sample,
    conf,
    source_manifest,
    library,
    call_text,
    *,
    workflow_count=None,
):
    """Compose multiple rank-free workflows for one held-out sample."""
    library_issues = validate_operator_library(library)
    if library_issues:
        raise ValueError("invalid operator library: " + " | ".join(library_issues))
    count = int(workflow_count or conf.get("operator_workflow_count", 3))
    if count < 2:
        raise ValueError("operator_workflow_count must be at least 2")
    case = build_decision_case(sample, conf)
    prompt = composition_prompt(case, source_manifest, library, count)
    raw, result = await _request_json(call_text, prompt, f"workflow composition for {case['case_id']}")
    issues = validate_workflow_result(result, library, expected_count=count)
    if issues:
        raise ValueError(f"invalid workflows for {case['case_id']}: " + " | ".join(issues))
    result["dataset"] = conf["dataset"]
    result["case_id"] = case["case_id"]
    return {"case": case, "prompt": prompt, "raw_response": raw, "result": result}


def load_operator_library(path):
    with open(path, "r", encoding="utf-8") as handle:
        library = json.load(handle)
    issues = validate_operator_library(library)
    if issues:
        raise ValueError("invalid operator library: " + " | ".join(issues))
    return library


def _write_json(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=str)


def _write_text(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(str(value or ""))


def save_operator_artifacts(output_dir, discovery=None, composition=None):
    """Persist inspectable prompts, raw responses, and normalized JSON artifacts."""
    os.makedirs(output_dir, exist_ok=True)
    if discovery:
        _write_json(
            os.path.join(output_dir, "source_capabilities.json"),
            discovery["source_capabilities"],
        )
        _write_json(os.path.join(output_dir, "discovery_cases.json"), discovery["discovery_cases"])
        _write_json(os.path.join(output_dir, "raw_operators.json"), discovery["raw_operators"])
        _write_json(os.path.join(output_dir, "operator_library.json"), discovery["library"])
        _write_text(os.path.join(output_dir, "clustering_input.txt"), discovery["clustering_prompt"])
        _write_text(
            os.path.join(output_dir, "clustering_output.txt"),
            discovery["clustering_raw_response"],
        )
        for trace in discovery["induction_traces"]:
            case_dir = os.path.join(output_dir, "induction", trace["case_id"])
            _write_text(os.path.join(case_dir, "input.txt"), trace["prompt"])
            _write_text(os.path.join(case_dir, "output.txt"), trace["raw_response"])
            _write_json(os.path.join(case_dir, "operators.json"), trace["operators"])
    if composition:
        case_id = composition["result"].get("case_id", "unknown")
        case_dir = os.path.join(output_dir, "composition", str(case_id))
        _write_json(os.path.join(case_dir, "case.json"), composition["case"])
        _write_text(os.path.join(case_dir, "input.txt"), composition["prompt"])
        _write_text(os.path.join(case_dir, "output.txt"), composition["raw_response"])
        _write_json(os.path.join(case_dir, "workflows.json"), composition["result"])

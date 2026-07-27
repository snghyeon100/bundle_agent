"""Offline induction, deduplication, compilation, and verification."""

import json
import os
import random
from statistics import mean

from code.common import parse_json_from_text, task_semantics
from code.pipeline import _item_profile
from code.workspace import build_source_manifest, prepare_workspace
from dataset import BundleZeroShotDataset

from .memory import (
    compact_operator_memory,
    count_exact_memory_matches,
    deduplicate_operator_pool,
)
from .prompts import (
    compilation_prompt,
    induction_prompt,
)
from .runtime import (
    assert_implementation_unchanged,
    evaluate_candidate_proposal_set,
    make_compiled_program,
    validate_candidate_proposal_set,
    validate_compilation_result,
)
from .schemas import (
    COMPILED_LIBRARY_SCHEMA_VERSION,
    OPERATOR_LIBRARY_SCHEMA_VERSION,
    OPERATOR_FIELDS,
    normalize_library,
    normalize_operator,
    resolve_induction_operators,
    validate_induction_result,
    validate_operator,
    validate_operator_library,
)


def _semantic_operator_view(operator):
    """Expose only reusable fields, never discovery-case identity."""
    if not isinstance(operator, dict):
        return operator
    normalized = normalize_operator(operator)
    return {
        field: normalized[field]
        for field in OPERATOR_FIELDS
        if field in normalized
    }


def sample_validation_cases(conf, count, seed=None):
    """Draw deterministic discovery cases from validation files, never test."""
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
    """Separate the candidate-blind prompt view from evaluator-only labels."""
    info_path = os.path.join(conf["data_path"], conf["dataset"], "item_info.json")
    with open(info_path, "r", encoding="utf-8") as handle:
        item_info = json.load(handle)
    gt_item_id = int(sample["true_indice"])
    gt_profile = _item_profile(gt_item_id, item_info, conf["dataset"])
    partial_items = [
        _item_profile(item_id, item_info, conf["dataset"])
        for item_id in sample["input_indices"]
    ]
    metadata_covered = sum(
        bool(item.get("text") or item.get("metadata"))
        for item in partial_items
    )
    return {
        "case_id": f"bundle_{sample['bundle_id']}",
        "dataset": conf["dataset"],
        "bundle_id": int(sample["bundle_id"]),
        "partial_items": partial_items,
        "source_diagnostics": {
            "partial_item_count": len(partial_items),
            "partial_metadata_coverage": {
                "covered": metadata_covered,
                "total": len(partial_items),
            },
        },
        "evaluation": {
            "ground_truth_item_id": gt_item_id,
            "ground_truth_profile": gt_profile,
            "benchmark_candidate_item_ids": [
                int(item_id) for item_id in sample.get("candidate_indices", [])
            ],
        },
    }


def build_operator_source_manifest(conf):
    workspace = prepare_workspace(conf, config_prefix="code")
    manifest = build_source_manifest(
        workspace,
        str(conf.get("code_current_bundle_train_context_policy", "allow")),
    )
    return workspace, manifest


def _operator_component_format(source):
    """Build a concrete, code-generation-ready access contract for one source."""
    name = str(source.get("name", ""))
    lower = name.lower()
    path = str(source.get("path") or f"data/{name}")

    if lower == "count.json":
        return {
            "path": path,
            "serialization": "UTF-8 JSON object",
            "load": "json.load(open(path, 'r', encoding='utf-8'))",
            "schema": {
                "#B": "number of bundles",
                "#I": "number of canonical items",
                "#U": "number of users",
                "#B-I": "number of bundle-item relations",
                "#U-I": "number of user-item relations",
                "#Avg. I/B": "average items per bundle",
                "#Avg. I/U": "average items per user",
            },
        }

    if lower == "item_info.json":
        return {
            "path": path,
            "serialization": "UTF-8 JSON object",
            "load": "json.load(open(path, 'r', encoding='utf-8'))",
            "key": (
                "canonical integer item_id encoded as a decimal string; access item i "
                "with records.get(str(int(i)))"
            ),
            "value_fields": dict(source.get("fields", {})),
            "id_rule": (
                "The top-level JSON key is the canonical item_id. Do not use an "
                "external identifier stored inside the value as a tensor row index."
            ),
        }

    if lower in {"bi_train.txt", "ui_full.txt"}:
        first_entity = "bundle_id" if lower == "bi_train.txt" else "user_id"
        return {
            "path": path,
            "serialization": "UTF-8 text; one comma-separated integer-ID row per line",
            "parse_row": (
                "values = [int(token.strip()) for token in line.split(',') "
                "if token.strip()]"
            ),
            "row_schema": (
                f"values[0] is {first_entity}; values[1:] are canonical item_ids"
            ),
            "relation_direction": (
                f"{first_entity} -> item_ids; build the reverse item_id -> "
                f"{first_entity}s index explicitly when needed"
            ),
            "id_rule": (
                "Bundle, user, and item IDs are different entity types even when "
                "their integer values happen to match."
            ),
        }

    if lower.endswith(".pt"):
        expected_shape = source.get("expected_shape")
        return {
            "path": path,
            "serialization": "PyTorch .pt object",
            "load": "torch.load(path, map_location='cpu', weights_only=False)",
            "required_object": "torch.Tensor",
            "expected_shape": expected_shape or [
                "#I",
                "embedding dimension; inspect tensor.shape[1]",
            ],
            "item_lookup": (
                "After verifying tensor.ndim == 2 and tensor.shape[0] == #I, "
                "the vector for canonical item_id i is tensor[int(i)]."
            ),
            "dtype_handling": (
                "Convert selected rows to float with tensor[int(i)].detach().float(); "
                "move no tensor to GPU."
            ),
            "normalization": str(
                source.get(
                    "normalization",
                    "not recorded; inspect norms before assuming unit normalization",
                )
            ),
        }

    return {
        "path": path,
        "serialization": str(source.get("format", "unknown")),
    }


def operator_capability_manifest(source_manifest, dataset):
    """Convert workspace files into one induction/code-generation source manifest."""
    dataset_name = str(dataset or "").lower()
    components = []
    for source in source_manifest.get("sources", []):
        name = str(source.get("name", ""))
        lower = name.lower()
        if lower == "count.json":
            capability_id = "dataset_statistics"
            description = "Dataset-level bundle, user, and item counts."
        elif lower == "item_info.json":
            capability_id = "item_metadata"
            if dataset_name.startswith("pog"):
                description = "Item text and category identifiers from item_info.json."
            elif "spotify" in dataset_name:
                description = (
                    "Track, artist, album, duration, and related item information "
                    "from item_info.json."
                )
            else:
                description = "Dataset-specific item information from item_info.json."
        elif lower == "bi_train.txt":
            capability_id = "bundle_item_history"
            description = "Historical bundle-to-item membership relations."
        elif lower == "ui_full.txt":
            capability_id = "user_item_history"
            description = "User-to-item interaction relations."
        elif lower == "content_feature.pt":
            capability_id = "item_content_embedding"
            modality = str(source.get("modality", "content") or "content")
            description = f"Item-level {modality} embedding vectors."
        elif lower == "description_feature.pt":
            capability_id = "item_description_embedding"
            description = "Item-level text-description embedding vectors."
        elif lower == "item_cf_feature.pt":
            capability_id = "user_collaborative_embedding"
            description = "Item embeddings learned from user-item interactions."
        elif lower.endswith("_lightgcn_bi_feature.pt"):
            capability_id = "bundle_collaborative_embedding"
            description = "Item embeddings learned from bundle-item relations."
        else:
            continue
        components.append(
            {
                "id": capability_id,
                "description": description,
                "format": _operator_component_format(source),
            }
        )
    return {
        "dataset": str(dataset),
        "description": task_semantics(dataset),
        "components": components,
    }


def build_operator_capability_manifest(conf):
    """Prepare sources and expose one manifest shared by induction and code generation."""
    workspace, source_manifest = build_operator_source_manifest(conf)
    capabilities = operator_capability_manifest(source_manifest, conf["dataset"])
    return workspace, source_manifest, capabilities


def _capability_names(source_capabilities):
    return {
        component.get("id")
        for component in source_capabilities.get("components", [])
        if isinstance(component, dict)
        and isinstance(component.get("id"), str)
        and component["id"].strip()
    }


def _read_relation_index(path):
    """Read anchor,item rows once for GT-independent source diagnostics."""
    item_to_anchors = {}
    anchor_to_items = {}
    if not os.path.isfile(path):
        return {
            "item_to_anchors": item_to_anchors,
            "anchor_to_items": anchor_to_items,
        }
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            values = [
                value.strip()
                for value in raw_line.strip().split(",")
                if value.strip()
            ]
            if len(values) < 2:
                continue
            anchor = values[0]
            items = list(dict.fromkeys(values[1:]))
            anchor_to_items[anchor] = items
            for item_id in items:
                item_to_anchors.setdefault(item_id, []).append(anchor)
    return {
        "item_to_anchors": item_to_anchors,
        "anchor_to_items": anchor_to_items,
    }


def _build_diagnostic_indices(conf, available_sources):
    data_dir = os.path.join(
        conf.get("data_path", "."),
        conf.get("dataset", ""),
    )
    indices = {}
    if "bundle_item_history" in available_sources:
        indices["bundle_item_history"] = _read_relation_index(
            os.path.join(data_dir, "bi_train.txt")
        )
    if "user_item_history" in available_sources:
        indices["user_item_history"] = _read_relation_index(
            os.path.join(data_dir, "ui_full.txt")
        )
    return indices


def _relation_diagnostics(partial_item_ids, index):
    item_to_anchors = index.get("item_to_anchors", {})
    anchor_to_items = index.get("anchor_to_items", {})
    anchors = []
    covered = 0
    partial_keys = {str(item_id) for item_id in partial_item_ids}
    per_item_counts = []
    for item_id in partial_keys:
        related = item_to_anchors.get(item_id, [])
        per_item_counts.append(len(related))
        if related:
            covered += 1
            anchors.extend(related)
    unique_anchors = list(dict.fromkeys(anchors))
    retrievable_items = {
        item_id
        for anchor in unique_anchors
        for item_id in anchor_to_items.get(anchor, [])
        if item_id not in partial_keys
    }
    return {
        "covered_partial_items": covered,
        "partial_item_count": len(partial_keys),
        "related_record_count": len(unique_anchors),
        "retrievable_non_partial_item_count": len(retrievable_items),
        "mean_records_per_partial_item": (
            sum(per_item_counts) / len(per_item_counts)
            if per_item_counts
            else 0.0
        ),
    }


def enrich_case_source_diagnostics(case, source_capabilities, diagnostic_indices):
    """Attach only GT-independent, instance-level source statistics."""
    diagnostics = dict(case.get("source_diagnostics", {}))
    available = sorted(_capability_names(source_capabilities))
    diagnostics["available_components"] = available
    partial_item_ids = [
        item.get("item_id")
        for item in case.get("partial_items", [])
        if isinstance(item, dict) and item.get("item_id") is not None
    ]
    diagnostics["component_diagnostics"] = {}
    for source_id in available:
        source_view = {"available": True}
        if source_id in diagnostic_indices:
            source_view.update(
                _relation_diagnostics(
                    partial_item_ids,
                    diagnostic_indices[source_id],
                )
            )
        diagnostics["component_diagnostics"][source_id] = source_view
    case["source_diagnostics"] = diagnostics
    return case


def _max_operators_per_case(conf, operators_per_case=None):
    maximum = int(operators_per_case or conf.get("operator_induction_count", 4))
    if maximum < 1:
        raise ValueError("operator_induction_count must be at least 1")
    return maximum


async def induce_raw_operators(
    samples,
    conf,
    call_text,
    *,
    source_capabilities,
    initial_operator_memory=None,
    operators_per_case=None,
    trace_callback=None,
):
    """Induce candidate-blind program specs in one LLM call per case."""
    maximum = _max_operators_per_case(conf, operators_per_case)
    if not isinstance(source_capabilities, dict) or not source_capabilities.get(
        "components"
    ):
        raise ValueError("source_capabilities must contain available components")
    allowed_source_names = _capability_names(source_capabilities)
    diagnostic_indices = _build_diagnostic_indices(
        conf,
        allowed_source_names,
    )
    memory_limit = int(conf.get("operator_memory_max_size", 24))
    operator_memory = compact_operator_memory(
        initial_operator_memory or [],
        max_size=memory_limit,
    )

    raw_operators = []
    discovery_cases = []
    induction_traces = []
    for sample in samples:
        case = build_discovery_case(sample, conf)
        enrich_case_source_diagnostics(
            case,
            source_capabilities,
            diagnostic_indices,
        )
        discovery_cases.append(case)
        prompt = induction_prompt(
            case,
            source_capabilities,
            operator_memory,
            maximum,
            text_only=bool(conf.get("operator_prompt_text_only", True)),
        )
        step_name = f"candidate-blind program induction for {case['case_id']}"
        raw = await call_text(prompt, step_name)
        result = parse_json_from_text(raw)
        issues = (
            validate_induction_result(
                result,
                min_count=0,
                max_count=maximum,
                allowed_source_names=allowed_source_names,
            )
            if isinstance(result, dict)
            else ["induction result must be a JSON object"]
        )
        resolved_operators = (
            resolve_induction_operators(result)
            if isinstance(result, dict)
            else []
        )
        if isinstance(result, dict):
            exact_memory_matches = count_exact_memory_matches(
                resolved_operators,
                operator_memory,
            )
            if exact_memory_matches:
                issues.append(
                    "operators must not exactly duplicate previous operator memory"
                )
        trace = {
            "case_id": case["case_id"],
            "prompt": prompt,
            "raw_response": raw,
            "parsed_response": result,
            "validation_issues": issues,
            "operator_memory_before": operator_memory,
            "operator_memory_after": operator_memory,
            "prompt_case": {
                "dataset": case["dataset"],
                "partial_items": case["partial_items"],
                "source_diagnostics": case["source_diagnostics"],
            },
            "evaluation": case["evaluation"],
            "hypotheses": (
                result.get("hypotheses", [])
                if isinstance(result, dict)
                else []
            ),
            "operators": [],
        }
        if issues:
            if trace_callback:
                trace_callback(trace)
            raise ValueError(
                f"invalid candidate-program induction for {case['case_id']}: "
                + " | ".join(issues)
            )
        case_operators = []
        for index, operator in enumerate(resolved_operators, start=1):
            enriched = _semantic_operator_view(operator)
            enriched["operator_id"] = f"{case['case_id']}__op{index}"
            enriched["origin_case_id"] = case["case_id"]
            case_operators.append(enriched)
            raw_operators.append(enriched)
        trace["operators"] = case_operators
        operator_memory = compact_operator_memory(
            operator_memory + case_operators,
            max_size=memory_limit,
        )
        trace["operator_memory_after"] = operator_memory
        induction_traces.append(trace)
        if trace_callback:
            trace_callback(trace)

    if not raw_operators:
        raise ValueError(
            "operator induction produced an empty raw operator pool across all cases"
        )

    return {
        "source_capabilities": source_capabilities,
        "discovery_cases": discovery_cases,
        "induction_traces": induction_traces,
        "raw_operators": raw_operators,
        "operator_memory": operator_memory,
        "operator_memory_max_size": memory_limit,
        "max_operators_per_case": maximum,
    }


def deduplicate_raw_operators(
    raw_operators,
    conf,
    *,
    similarity_threshold=None,
):
    """Build a canonical library without an LLM clustering call."""
    if not isinstance(raw_operators, list) or not raw_operators:
        raise ValueError("raw operator pool must be a non-empty list")
    normalized_raw_operators = [
        normalize_operator(operator, preserve_metadata=True)
        for operator in raw_operators
    ]
    pool_issues = []
    for index, operator in enumerate(normalized_raw_operators):
        pool_issues.extend(
            f"raw_operators[{index}]: {issue}" for issue in validate_operator(operator)
        )
    if pool_issues:
        raise ValueError("invalid raw operator pool: " + " | ".join(pool_issues))

    threshold = float(
        similarity_threshold
        if similarity_threshold is not None
        else conf.get("operator_dedup_similarity_threshold", 0.9)
    )
    dedup = deduplicate_operator_pool(
        normalized_raw_operators,
        similarity_threshold=threshold,
    )
    maximum = int(conf.get("operator_library_max_size", 0))
    operators = dedup["operators"]
    if maximum > 0:
        operators = operators[:maximum]
    library = {
        "schema_version": OPERATOR_LIBRARY_SCHEMA_VERSION,
        "operators": operators,
    }
    issues = validate_operator_library(library)
    if issues:
        raise ValueError("invalid deduplicated operator library: " + " | ".join(issues))
    return {
        "library": library,
        "deduplication": {
            **dedup,
            "operators": None,
            "max_library_size": maximum or None,
        },
    }


async def compile_operator_programs(
    library,
    conf,
    call_text,
    *,
    source_capabilities,
    trace_callback=None,
):
    """Compile each canonical spec once, after deterministic deduplication."""
    normalized_library = normalize_library(library)
    allowed_source_names = _capability_names(source_capabilities)
    issues = validate_operator_library(
        normalized_library,
        allowed_source_names=allowed_source_names,
    )
    if issues:
        raise ValueError("invalid operator library: " + " | ".join(issues))

    compiled_programs = []
    compilation_traces = []
    for operator in normalized_library["operators"]:
        prompt = compilation_prompt(operator, source_capabilities)
        step_name = f"offline program compilation for {operator['name']}"
        raw = await call_text(prompt, step_name)
        parsed = parse_json_from_text(raw)
        compile_issues = validate_compilation_result(parsed, operator)
        trace = {
            "operator_id": operator.get("operator_id"),
            "operator_name": operator["name"],
            "prompt": prompt,
            "raw_response": raw,
            "parsed_response": parsed,
            "validation_issues": compile_issues,
        }
        compilation_traces.append(trace)
        if trace_callback:
            trace_callback(trace)
        if compile_issues:
            raise ValueError(
                f"invalid compiled program for {operator['name']}: "
                + " | ".join(compile_issues)
            )
        compiled_programs.append(
            make_compiled_program(operator, parsed["code"])
        )

    return {
        "compiled_library": {
            "schema_version": COMPILED_LIBRARY_SCHEMA_VERSION,
            "programs": compiled_programs,
        },
        "compilation_traces": compilation_traces,
    }


async def discover_operator_library(
    samples,
    conf,
    call_text,
    *,
    source_capabilities,
    operators_per_case=None,
):
    """Run the two-call-type offline MVP: induce, dedup, then compile."""
    induction = await induce_raw_operators(
        samples,
        conf,
        call_text,
        source_capabilities=source_capabilities,
        operators_per_case=operators_per_case,
    )
    deduplication = deduplicate_raw_operators(
        induction["raw_operators"],
        conf,
    )
    compilation = await compile_operator_programs(
        deduplication["library"],
        conf,
        call_text,
        source_capabilities=source_capabilities,
    )
    return {
        **induction,
        **deduplication,
        **compilation,
    }


def verify_compiled_programs(
    compiled_library,
    validation_cases,
    execute_program,
    *,
    candidate_budget,
    evidence_budget,
):
    """Execute immutable code through an injected sandbox runner and score retrieval."""
    if not callable(execute_program):
        raise ValueError("execute_program must be callable")
    validation_cases = list(validation_cases or [])
    if not validation_cases:
        raise ValueError("validation_cases must contain held-out cases")
    programs = compiled_library.get("programs", [])
    verification = []
    for compiled_program in programs:
        assert_implementation_unchanged(compiled_program)
        operator = compiled_program["operator"]
        program_id = operator.get("operator_id") or operator["name"]
        origin_case_ids = set(operator.get("origin_case_ids", []))
        origin_case_id = operator.get("origin_case_id")
        if isinstance(origin_case_id, str) and origin_case_id:
            origin_case_ids.add(origin_case_id)
        validation_case_ids = {
            case.get("case_id")
            for case in validation_cases
            if isinstance(case, dict)
        }
        overlap = sorted(origin_case_ids & validation_case_ids)
        if overlap:
            raise ValueError(
                f"program {program_id} cannot be verified on discovery cases: "
                + ", ".join(overlap)
            )
        case_results = []
        for case in validation_cases:
            try:
                output = execute_program(
                    compiled_program,
                    case,
                    int(candidate_budget),
                    int(evidence_budget),
                )
                output_issues = validate_candidate_proposal_set(
                    output,
                    allowed_sources=operator.get("required_sources", []),
                    allowed_evidence_types=operator.get("evidence_types", []),
                    candidate_budget=candidate_budget,
                    evidence_budget=evidence_budget,
                    expected_program_id=program_id,
                    expected_hypothesis=operator["hypothesis"],
                    excluded_item_ids=[
                        item.get("item_id")
                        for item in case.get("partial_items", [])
                        if isinstance(item, dict)
                    ],
                )
                metrics = (
                    evaluate_candidate_proposal_set(
                        output,
                        case["evaluation"]["ground_truth_item_id"],
                    )
                    if not output_issues
                    else None
                )
                case_results.append(
                    {
                        "case_id": case["case_id"],
                        "execution_success": not output_issues,
                        "validation_issues": output_issues,
                        "metrics": metrics,
                    }
                )
            except Exception as error:
                case_results.append(
                    {
                        "case_id": case["case_id"],
                        "execution_success": False,
                        "validation_issues": [str(error)],
                        "metrics": None,
                    }
                )
        successes = [
            result for result in case_results if result["execution_success"]
        ]
        metrics = [
            result["metrics"] for result in successes if result["metrics"]
        ]
        profile = {
            "program_id": program_id,
            "implementation_sha256": compiled_program["implementation"]["sha256"],
            "case_count": len(case_results),
            "execution_success_rate": (
                len(successes) / len(case_results) if case_results else 0.0
            ),
            "candidate_recall": (
                mean(metric["hit"] for metric in metrics) if metrics else 0.0
            ),
            "mean_reciprocal_rank": (
                mean(metric["reciprocal_rank"] for metric in metrics)
                if metrics
                else 0.0
            ),
            "mean_candidate_count": (
                mean(metric["candidate_count"] for metric in metrics)
                if metrics
                else 0.0
            ),
            "case_results": case_results,
        }
        verification.append(profile)
    return {"program_profiles": verification}


def admit_verified_programs(
    compiled_library,
    verification,
    *,
    min_execution_success_rate=1.0,
    min_candidate_recall=0.0,
):
    """Create verified and rejected registries without changing compiled code."""
    profile_by_id = {
        profile.get("program_id"): profile
        for profile in verification.get("program_profiles", [])
        if isinstance(profile, dict)
    }
    verified = []
    rejected = []
    for compiled_program in compiled_library.get("programs", []):
        assert_implementation_unchanged(compiled_program)
        operator = compiled_program["operator"]
        program_id = operator.get("operator_id") or operator["name"]
        profile = profile_by_id.get(program_id)
        accepted = bool(
            profile
            and profile.get("execution_success_rate", 0.0)
            >= float(min_execution_success_rate)
            and profile.get("candidate_recall", 0.0)
            >= float(min_candidate_recall)
        )
        admitted = {
            **compiled_program,
            "admission_status": "verified" if accepted else "rejected",
            "validation_profile": profile,
        }
        (verified if accepted else rejected).append(admitted)
    return {
        "schema_version": COMPILED_LIBRARY_SCHEMA_VERSION,
        "verified_programs": verified,
        "rejected_programs": rejected,
        "admission_policy": {
            "min_execution_success_rate": float(min_execution_success_rate),
            "min_candidate_recall": float(min_candidate_recall),
        },
    }


def load_operator_library(path):
    with open(path, "r", encoding="utf-8") as handle:
        library = normalize_library(json.load(handle))
    issues = validate_operator_library(
        library,
    )
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


def save_operator_artifacts(
    output_dir,
    discovery=None,
):
    """Persist candidate-blind prompts, specs, dedup groups, and compiled code."""
    os.makedirs(output_dir, exist_ok=True)
    if discovery:
        _write_json(
            os.path.join(output_dir, "source_manifest.json"),
            discovery["source_capabilities"],
        )
        _write_json(os.path.join(output_dir, "discovery_cases.json"), discovery["discovery_cases"])
        _write_json(os.path.join(output_dir, "raw_operators.json"), discovery["raw_operators"])
        _write_json(
            os.path.join(output_dir, "operator_memory.json"),
            discovery["operator_memory"],
        )
        _write_json(os.path.join(output_dir, "operator_library.json"), discovery["library"])
        _write_json(
            os.path.join(output_dir, "deduplication.json"),
            discovery["deduplication"],
        )
        _write_json(
            os.path.join(output_dir, "compiled_program_library.json"),
            discovery["compiled_library"],
        )
        for trace in discovery["induction_traces"]:
            case_dir = os.path.join(output_dir, "induction", trace["case_id"])
            _write_text(os.path.join(case_dir, "input.txt"), trace["prompt"])
            _write_text(os.path.join(case_dir, "output.txt"), trace["raw_response"])
            _write_json(
                os.path.join(case_dir, "parsed_response.json"),
                trace["parsed_response"],
            )
            _write_json(
                os.path.join(case_dir, "hypotheses.json"),
                trace["hypotheses"],
            )
            _write_json(
                os.path.join(case_dir, "validation_issues.json"),
                trace["validation_issues"],
            )
            _write_json(os.path.join(case_dir, "prompt_case.json"), trace["prompt_case"])
            _write_json(os.path.join(case_dir, "evaluation.json"), trace["evaluation"])
            _write_json(
                os.path.join(case_dir, "operator_memory_before.json"),
                trace["operator_memory_before"],
            )
            _write_json(
                os.path.join(case_dir, "operator_memory_after.json"),
                trace["operator_memory_after"],
            )
            _write_json(os.path.join(case_dir, "operators.json"), trace["operators"])
        for trace in discovery["compilation_traces"]:
            name = str(trace["operator_name"])
            program_dir = os.path.join(output_dir, "compilation", name)
            _write_text(os.path.join(program_dir, "input.txt"), trace["prompt"])
            _write_text(os.path.join(program_dir, "output.txt"), trace["raw_response"])
            _write_json(
                os.path.join(program_dir, "parsed_response.json"),
                trace["parsed_response"],
            )
            _write_json(
                os.path.join(program_dir, "validation_issues.json"),
                trace["validation_issues"],
            )
            if isinstance(trace["parsed_response"], dict):
                _write_text(
                    os.path.join(program_dir, "program.py"),
                    trace["parsed_response"].get("code", ""),
                )

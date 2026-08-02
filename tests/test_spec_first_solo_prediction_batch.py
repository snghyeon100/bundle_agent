"""OpenAI Batch ablation: rank once from each saved strategy in isolation.

The source run is never regenerated or re-executed. Only samples with a valid
joint prediction and all requested strategy executions are included, so every
solo result can be compared with the saved joint result on the same sample.

Commands:
    python tests/test_spec_first_solo_prediction_batch.py start \
        --config config_operator.yaml --source_run <completed_run> \
        --output_dir <new_run> --api_key_env DMLAB_KEY --dry_run

    python tests/test_spec_first_solo_prediction_batch.py submit --run <new_run>
    python tests/test_spec_first_solo_prediction_batch.py status --run <new_run>
    python tests/test_spec_first_solo_prediction_batch.py advance --run <new_run>
"""

import argparse
import csv
import json
import os
import shutil
import sys
import time

import yaml
from openai import OpenAI


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from code.common import parse_json_from_text
from main import (
    default_api_key_envs_for_provider,
    openai_model_supports_reasoning,
    resolve_api_key_from_keys,
    stage_model,
    stage_provider,
)
from online_hypothesis_program.schemas import validate_prediction_result
from operator_learning.openai_batch import (
    BATCH_TERMINAL_STATUSES,
    batch_output_by_custom_id,
    batch_request,
    download_file,
    extract_response_text,
    response_request_body,
    retrieve_batch,
    submit_batch,
    write_jsonl,
)
from operator_learning.prompts import strategy_evidence_prediction_prompt
from operator_learning.spec_first_prediction import (
    aggregate_prediction_rows,
    evaluate_full_ranking,
)


DEFAULT_STRATEGY_IDS = ("S1", "S2", "S3")
RESULT_FIELDS = [
    "sample_idx",
    "bundle_id",
    "strategy_id",
    "candidate_count",
    "evidence_context_count",
    "prediction",
    "ranking",
    "true_label",
    "hit",
    "gt_rank",
    "reciprocal_rank",
    "hit_at_1",
    "hit_at_3",
    "hit_at_5",
    "source_joint_prediction",
    "source_joint_gt_rank",
    "gt_rank_improvement_over_joint",
    "top1_agrees_with_joint",
    "valid",
    "error",
]
ORACLE_FIELDS = [
    "sample_idx",
    "bundle_id",
    "true_label",
    "oracle_gt_rank",
    "reciprocal_rank",
    "hit_at_1",
    "hit_at_3",
    "hit_at_5",
    "best_strategy_ids",
    "source_joint_gt_rank",
    "gt_rank_improvement_over_joint",
]


def _read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path, value):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=str)
    os.replace(temporary, path)


def _write_text(path, value):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        handle.write(str(value or ""))
    os.replace(temporary, path)


def _write_yaml(path, value):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        yaml.safe_dump(value, handle, allow_unicode=True, sort_keys=False)
    os.replace(temporary, path)


def _write_csv(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for row in rows:
            serialized = {field: row.get(field, "") for field in RESULT_FIELDS}
            if isinstance(serialized["ranking"], (list, dict)):
                serialized["ranking"] = json.dumps(
                    serialized["ranking"],
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            writer.writerow(serialized)
    os.replace(temporary, path)


def _write_oracle_csv(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ORACLE_FIELDS)
        writer.writeheader()
        for row in rows:
            serialized = {field: row.get(field, "") for field in ORACLE_FIELDS}
            serialized["best_strategy_ids"] = json.dumps(
                serialized["best_strategy_ids"],
                ensure_ascii=False,
                separators=(",", ":"),
            )
            writer.writerow(serialized)
    os.replace(temporary, path)


def _load_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"config must contain an object: {path}")
    return value


def _state_path(run_dir):
    return os.path.join(run_dir, "state.json")


def _load_state(run_dir):
    value = _read_json(_state_path(run_dir))
    if not isinstance(value, dict):
        raise ValueError("state.json must contain an object")
    return value


def _save_state(run_dir, state):
    state["updated_at"] = int(time.time())
    _write_json(_state_path(run_dir), state)


def _source_results(source_run):
    for name in ("results.json", "results_partial.json"):
        path = os.path.join(source_run, name)
        if os.path.isfile(path):
            value = _read_json(path)
            if not isinstance(value, list):
                raise ValueError(f"source results must be a list: {path}")
            return sorted(
                [row for row in value if isinstance(row, dict)],
                key=lambda row: int(row["sample_idx"]),
            )
    raise FileNotFoundError(f"source results not found under: {source_run}")


def _case_dir(run_dir, sample_idx, bundle_id):
    return os.path.join(
        run_dir,
        "samples",
        f"{int(sample_idx):04d}_bundle_{int(bundle_id)}",
    )


def _record_key(record):
    return (
        int(record["sample_idx"]),
        int(record["bundle_id"]),
        str(record["strategy_id"]),
    )


def _custom_id(sample_idx, bundle_id, strategy_id):
    return (
        f"solo-{str(strategy_id).lower()}-{int(sample_idx):04d}"
        f"-bundle-{int(bundle_id)}"
    )


def _context_count(evidence):
    return sum(
        len(candidate.get("contexts", []))
        for candidate in evidence.get("candidate_evidence", [])
        if isinstance(candidate, dict)
        and isinstance(candidate.get("contexts", []), list)
    )


def _prediction_model(conf):
    provider = stage_provider(conf, "prediction")
    if provider != "openai":
        raise ValueError(
            f"OpenAI Batch prediction requires OpenAI; got {provider}"
        )
    return stage_model(conf, "prediction")


def _prediction_client(conf):
    api_key, env = resolve_api_key_from_keys(
        conf,
        [
            "operator_prediction_api_key_env",
            "code_prediction_api_key_env",
            "code_generation_api_key_env",
            "code_api_key_env",
        ],
        default_api_key_envs_for_provider("openai"),
    )
    return OpenAI(api_key=api_key), env


def _request_body(conf, prompt):
    model = _prediction_model(conf)
    effort = str(conf.get("openai_reasoning_effort", "")).strip()
    if not openai_model_supports_reasoning(model):
        effort = ""
    temperature = (
        float(conf.get("temperature", 0.0))
        if bool(conf.get("openai_send_temperature", False))
        else None
    )
    return response_request_body(
        model=model,
        prompt=prompt,
        max_output_tokens=int(
            conf.get("operator_prediction_max_output_tokens", 2000)
        ),
        reasoning_effort=effort,
        temperature=temperature,
    )


def _eligible_source_row(row, strategy_count):
    return (
        bool(row.get("valid"))
        and not row.get("error")
        and row.get("gt_rank") is not None
        and int(row.get("successful_program_count", 0)) == strategy_count
    )


def _initialize(args):
    source_run = os.path.abspath(args.source_run)
    if not os.path.isdir(source_run):
        raise FileNotFoundError(f"source run does not exist: {source_run}")
    output_dir = os.path.abspath(args.output_dir)
    if os.path.exists(output_dir) and os.listdir(output_dir):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    source_meta = _read_json(os.path.join(source_run, "run.json"))
    dataset = str(source_meta.get("dataset") or "").strip()
    if not dataset:
        raise ValueError("source run does not identify its dataset")
    conf = _load_config(args.config)
    conf["dataset"] = dataset
    conf["operator_prediction_api_key_env"] = str(args.api_key_env)
    model = _prediction_model(conf)
    strategy_ids = tuple(
        part.strip()
        for part in str(args.strategy_ids).split(",")
        if part.strip()
    )
    if not strategy_ids or len(strategy_ids) != len(set(strategy_ids)):
        raise ValueError("--strategy_ids must contain unique IDs")

    requests = []
    records = []
    excluded = {
        "source_joint_invalid": 0,
        "incomplete_strategy_execution": 0,
        "missing_or_invalid_evidence": 0,
    }
    for source_row in _source_results(source_run):
        if not bool(source_row.get("valid")) or source_row.get("gt_rank") is None:
            excluded["source_joint_invalid"] += 1
            continue
        if not _eligible_source_row(source_row, len(strategy_ids)):
            excluded["incomplete_strategy_execution"] += 1
            continue
        sample_idx = int(source_row["sample_idx"])
        bundle_id = int(source_row["bundle_id"])
        source_case_dir = _case_dir(source_run, sample_idx, bundle_id)
        try:
            case = _read_json(os.path.join(source_case_dir, "case.json"))
            evidence = _read_json(
                os.path.join(
                    source_case_dir,
                    "rendered_strategy_evidence.json",
                )
            )
            if not isinstance(evidence, list):
                raise ValueError("strategy evidence must be a list")
            evidence_by_id = {
                str(item.get("strategy_id") or ""): item
                for item in evidence
                if isinstance(item, dict)
            }
            if any(strategy_id not in evidence_by_id for strategy_id in strategy_ids):
                raise ValueError("one or more requested strategies are missing")
            candidate_items = case.get("candidate_items", [])
            labels = [
                str(candidate.get("label") or "")
                for candidate in candidate_items
                if isinstance(candidate, dict)
            ]
            if not labels or len(labels) != len(set(labels)):
                raise ValueError("candidate labels are empty or duplicated")
        except Exception:
            excluded["missing_or_invalid_evidence"] += 1
            continue

        output_case_dir = _case_dir(output_dir, sample_idx, bundle_id)
        _write_json(os.path.join(output_case_dir, "case.json"), case)
        for strategy_id in strategy_ids:
            solo_evidence = evidence_by_id[strategy_id]
            prompt = strategy_evidence_prediction_prompt(
                dataset=dataset,
                partial_items=case.get("partial_items", []),
                candidate_items=candidate_items,
                strategy_evidence=[solo_evidence],
            )
            custom_id = _custom_id(sample_idx, bundle_id, strategy_id)
            strategy_dir = os.path.join(output_case_dir, strategy_id)
            _write_json(
                os.path.join(strategy_dir, "strategy_evidence.json"),
                solo_evidence,
            )
            _write_text(os.path.join(strategy_dir, "input.txt"), prompt)
            requests.append(
                batch_request(
                    custom_id=custom_id,
                    body=_request_body(conf, prompt),
                )
            )
            records.append(
                {
                    "sample_idx": sample_idx,
                    "bundle_id": bundle_id,
                    "strategy_id": strategy_id,
                    "candidate_count": len(labels),
                    "evidence_context_count": _context_count(solo_evidence),
                    "true_label": str(source_row.get("true_label") or ""),
                    "source_joint_prediction": str(
                        source_row.get("prediction") or ""
                    ),
                    "source_joint_gt_rank": int(source_row["gt_rank"]),
                    "custom_id": custom_id,
                }
            )

    if not requests:
        raise RuntimeError("source run contains no complete paired samples")
    input_path = os.path.join(output_dir, "batches", "prediction_input.jsonl")
    input_info = write_jsonl(input_path, requests)
    _write_json(os.path.join(output_dir, "records.json"), records)
    _write_yaml(os.path.join(output_dir, "config_snapshot.yaml"), conf)
    for name in ("sample_manifest.json", "source_manifest.json"):
        source_path = os.path.join(source_run, name)
        if os.path.isfile(source_path):
            shutil.copyfile(source_path, os.path.join(output_dir, name))

    now = int(time.time())
    run_info = {
        "phase": "spec_first_solo_prediction_openai_batch",
        "dataset": dataset,
        "split": str(source_meta.get("split") or "test"),
        "source_run": source_run,
        "source_joint_prediction": "reused",
        "strategy_generation": "reused",
        "program_execution": "reused",
        "strategy_ids": list(strategy_ids),
        "eligible_sample_count": len(records) // len(strategy_ids),
        "request_count": len(records),
        "prediction_model": model,
        "prediction_api_key_env": str(args.api_key_env),
        "created_at": now,
    }
    _write_json(os.path.join(output_dir, "run.json"), run_info)
    state = {
        "schema_version": 1,
        "phase": "prediction_prepared",
        "dataset": dataset,
        "source_run": source_run,
        "strategy_ids": list(strategy_ids),
        "eligible_sample_count": len(records) // len(strategy_ids),
        "excluded_samples": excluded,
        "created_at": now,
        "updated_at": now,
        "prediction": {
            "status": "prepared",
            "input_path": os.path.relpath(input_path, output_dir),
            **input_info,
            "model": {
                "provider": "openai",
                "model": model,
                "api_key_env": str(args.api_key_env),
            },
        },
    }
    _save_state(output_dir, state)

    print(f">>> Dataset: {dataset}")
    print(f">>> Source run: {source_run}")
    print(f">>> Strategies: {', '.join(strategy_ids)} (one request each)")
    print(f">>> Eligible paired samples: {state['eligible_sample_count']}")
    print(f">>> Prediction requests: {input_info['request_count']}")
    print(f">>> Input bytes: {input_info['input_bytes']}")
    print(f">>> Excluded samples: {excluded}")
    print(">>> Strategy generation and execution: reused, not rerun")
    print(f">>> Output: {output_dir}")
    if args.dry_run:
        print(">>> Dry run: prediction Batch prepared but not submitted")
        return 0
    return _submit(output_dir)


def _submit(run_dir):
    run_dir = os.path.abspath(run_dir)
    state = _load_state(run_dir)
    stage = state["prediction"]
    if stage.get("batch_id"):
        raise RuntimeError(
            f"prediction Batch was already submitted: {stage['batch_id']}"
        )
    if state.get("phase") != "prediction_prepared":
        raise RuntimeError(
            f"cannot submit from phase: {state.get('phase')}"
        )
    conf = _load_config(os.path.join(run_dir, "config_snapshot.yaml"))
    client, key_env = _prediction_client(conf)
    input_path = os.path.join(run_dir, stage["input_path"])
    submission = submit_batch(
        client,
        input_path=input_path,
        metadata={
            "pipeline": "spec-first-solo-prediction",
            "dataset": state["dataset"],
            "strategies": ",".join(state["strategy_ids"]),
            "run": os.path.basename(run_dir),
        },
    )
    batch = submission["batch"]
    stage.update(
        {
            "status": str(batch.get("status") or "submitted"),
            "input_file_id": submission["input_file_id"],
            "batch_id": str(batch["id"]),
            "batch": batch,
            "submitted_at": int(time.time()),
        }
    )
    state["phase"] = "prediction_submitted"
    state["prediction_api_key_env"] = key_env
    _save_state(run_dir, state)
    print(f">>> Prediction Batch submitted: {stage['batch_id']}")
    print(f">>> Requests: {stage['request_count']}")
    print(f">>> Run: {run_dir}")
    return 0


def _refresh(run_dir):
    state = _load_state(run_dir)
    stage = state["prediction"]
    batch_id = str(stage.get("batch_id") or "")
    if not batch_id:
        raise RuntimeError("prediction Batch has not been submitted")
    conf = _load_config(os.path.join(run_dir, "config_snapshot.yaml"))
    client, _ = _prediction_client(conf)
    batch = retrieve_batch(client, batch_id)
    stage["batch"] = batch
    stage["status"] = str(batch.get("status") or "")
    _save_state(run_dir, state)
    return client, state


def _error_result(record, error):
    return {
        "sample_idx": int(record["sample_idx"]),
        "bundle_id": int(record["bundle_id"]),
        "strategy_id": str(record["strategy_id"]),
        "candidate_count": int(record["candidate_count"]),
        "evidence_context_count": int(record["evidence_context_count"]),
        "prediction": None,
        "ranking": [],
        "true_label": str(record["true_label"]),
        "hit": False,
        "gt_rank": None,
        "reciprocal_rank": 0.0,
        "hit_at_1": False,
        "hit_at_3": False,
        "hit_at_5": False,
        "source_joint_prediction": str(record["source_joint_prediction"]),
        "source_joint_gt_rank": int(record["source_joint_gt_rank"]),
        "gt_rank_improvement_over_joint": None,
        "top1_agrees_with_joint": False,
        "valid": False,
        "error": str(error),
    }


def _strategy_summary(strategy_id, rows):
    rows = [row for row in rows if row["strategy_id"] == strategy_id]
    valid = [row for row in rows if row.get("valid")]
    base = aggregate_prediction_rows(rows)
    deltas = [
        int(row["gt_rank_improvement_over_joint"])
        for row in valid
    ]
    joint_ranks = [int(row["source_joint_gt_rank"]) for row in valid]
    return {
        **base,
        "source_joint_hit_rate_at_1_on_valid_intersection": (
            sum(rank == 1 for rank in joint_ranks) / len(joint_ranks)
            if joint_ranks
            else 0.0
        ),
        "source_joint_mean_gt_rank_on_valid_intersection": (
            sum(joint_ranks) / len(joint_ranks) if joint_ranks else 0.0
        ),
        "mean_gt_rank_improvement_over_joint": (
            sum(deltas) / len(deltas) if deltas else 0.0
        ),
        "solo_wins": sum(delta > 0 for delta in deltas),
        "ties": sum(delta == 0 for delta in deltas),
        "solo_losses": sum(delta < 0 for delta in deltas),
        "top1_agreement_rate_with_joint": (
            sum(bool(row["top1_agrees_with_joint"]) for row in valid)
            / len(valid)
            if valid
            else 0.0
        ),
    }


def _oracle_best_evidence(strategy_ids, rows):
    """Select the lowest GT rank across solo strategies for each sample.

    This is intentionally label-aware and is an analysis upper bound, not a
    deployable strategy selector.
    """
    strategy_ids = tuple(str(strategy_id) for strategy_id in strategy_ids)
    expected = set(strategy_ids)
    grouped = {}
    for row in rows:
        key = (int(row["sample_idx"]), int(row["bundle_id"]))
        grouped.setdefault(key, {})[str(row["strategy_id"])] = row

    oracle_rows = []
    unique_best_counts = {strategy_id: 0 for strategy_id in strategy_ids}
    included_in_best_tie_counts = {
        strategy_id: 0 for strategy_id in strategy_ids
    }
    tie_sample_count = 0
    for (sample_idx, bundle_id), by_strategy in sorted(grouped.items()):
        if set(by_strategy) != expected:
            continue
        selected = [by_strategy[strategy_id] for strategy_id in strategy_ids]
        if any(not row.get("valid") for row in selected):
            continue
        best_rank = min(int(row["gt_rank"]) for row in selected)
        best_strategy_ids = [
            strategy_id
            for strategy_id in strategy_ids
            if int(by_strategy[strategy_id]["gt_rank"]) == best_rank
        ]
        if len(best_strategy_ids) == 1:
            unique_best_counts[best_strategy_ids[0]] += 1
        else:
            tie_sample_count += 1
        for strategy_id in best_strategy_ids:
            included_in_best_tie_counts[strategy_id] += 1
        first = selected[0]
        joint_rank = int(first["source_joint_gt_rank"])
        oracle_rows.append(
            {
                "sample_idx": sample_idx,
                "bundle_id": bundle_id,
                "true_label": str(first["true_label"]),
                "oracle_gt_rank": best_rank,
                "reciprocal_rank": 1.0 / best_rank,
                "hit_at_1": best_rank <= 1,
                "hit_at_3": best_rank <= 3,
                "hit_at_5": best_rank <= 5,
                "best_strategy_ids": best_strategy_ids,
                "source_joint_gt_rank": joint_rank,
                "gt_rank_improvement_over_joint": joint_rank - best_rank,
            }
        )

    ranks = [int(row["oracle_gt_rank"]) for row in oracle_rows]
    improvements = [
        int(row["gt_rank_improvement_over_joint"]) for row in oracle_rows
    ]
    joint_ranks = [
        int(row["source_joint_gt_rank"]) for row in oracle_rows
    ]
    count = len(oracle_rows)
    summary = {
        "definition": (
            "label-aware per-sample minimum ground-truth rank across valid "
            "solo strategy rankings; analysis upper bound only"
        ),
        "common_valid_sample_count": count,
        "excluded_without_all_valid_solo_rankings": len(grouped) - count,
        "hit_rate_at_1": (
            sum(rank <= 1 for rank in ranks) / count if count else 0.0
        ),
        "hit_rate_at_3": (
            sum(rank <= 3 for rank in ranks) / count if count else 0.0
        ),
        "hit_rate_at_5": (
            sum(rank <= 5 for rank in ranks) / count if count else 0.0
        ),
        "mean_reciprocal_rank": (
            sum(1.0 / rank for rank in ranks) / count if count else 0.0
        ),
        "mean_gt_rank": sum(ranks) / count if count else 0.0,
        "source_joint_hit_rate_at_1_on_common_intersection": (
            sum(rank <= 1 for rank in joint_ranks) / count
            if count
            else 0.0
        ),
        "source_joint_mean_gt_rank_on_common_intersection": (
            sum(joint_ranks) / count if count else 0.0
        ),
        "mean_gt_rank_improvement_over_joint": (
            sum(improvements) / count if count else 0.0
        ),
        "oracle_wins_over_joint": sum(value > 0 for value in improvements),
        "ties_with_joint": sum(value == 0 for value in improvements),
        "oracle_losses_to_joint": sum(value < 0 for value in improvements),
        "unique_best_strategy_counts": unique_best_counts,
        "included_in_best_tie_counts": included_in_best_tie_counts,
        "best_strategy_tie_sample_count": tie_sample_count,
    }
    return oracle_rows, summary


def _finalize(run_dir, client, state):
    stage = state["prediction"]
    batch = stage.get("batch") or {}
    output_path = os.path.join(
        run_dir,
        "batches",
        "prediction_output.jsonl",
    )
    error_path = os.path.join(
        run_dir,
        "batches",
        "prediction_error.jsonl",
    )
    output_file_id = str(batch.get("output_file_id") or "")
    error_file_id = str(batch.get("error_file_id") or "")
    if output_file_id:
        download_file(client, output_file_id, output_path)
        stage["output_path"] = os.path.relpath(output_path, run_dir)
    if error_file_id:
        download_file(client, error_file_id, error_path)
        stage["error_path"] = os.path.relpath(error_path, run_dir)
    outputs = batch_output_by_custom_id(output_path)
    errors = batch_output_by_custom_id(error_path)
    records = _read_json(os.path.join(run_dir, "records.json"))
    rows = []

    for record in records:
        strategy_dir = os.path.join(
            _case_dir(
                run_dir,
                record["sample_idx"],
                record["bundle_id"],
            ),
            str(record["strategy_id"]),
        )
        try:
            custom_id = str(record["custom_id"])
            output_row = outputs.get(custom_id) or errors.get(custom_id)
            raw = extract_response_text(output_row)
            parsed = parse_json_from_text(raw)
            case = _read_json(
                os.path.join(
                    _case_dir(
                        run_dir,
                        record["sample_idx"],
                        record["bundle_id"],
                    ),
                    "case.json",
                )
            )
            labels = [
                str(candidate.get("label") or "")
                for candidate in case.get("candidate_items", [])
                if isinstance(candidate, dict)
            ]
            issues = validate_prediction_result(parsed, labels)
            _write_text(os.path.join(strategy_dir, "output.txt"), raw)
            _write_json(
                os.path.join(strategy_dir, "parsed_response.json"),
                parsed,
            )
            _write_json(
                os.path.join(strategy_dir, "validation_issues.json"),
                issues,
            )
            if issues:
                raise ValueError(
                    "invalid prediction: " + " | ".join(issues)
                )
            evaluation = evaluate_full_ranking(parsed, record["true_label"])
            improvement = (
                int(record["source_joint_gt_rank"])
                - int(evaluation["gt_rank"])
            )
            row = {
                "sample_idx": int(record["sample_idx"]),
                "bundle_id": int(record["bundle_id"]),
                "strategy_id": str(record["strategy_id"]),
                "candidate_count": int(record["candidate_count"]),
                "evidence_context_count": int(
                    record["evidence_context_count"]
                ),
                **evaluation,
                "source_joint_prediction": str(
                    record["source_joint_prediction"]
                ),
                "source_joint_gt_rank": int(
                    record["source_joint_gt_rank"]
                ),
                "gt_rank_improvement_over_joint": improvement,
                "top1_agrees_with_joint": (
                    evaluation["prediction"]
                    == str(record["source_joint_prediction"])
                ),
                "valid": True,
                "error": "",
            }
        except Exception as error:
            row = _error_result(record, error)
            _write_json(
                os.path.join(strategy_dir, "error.json"),
                {"error": str(error)},
            )
        rows.append(row)

    rows.sort(key=_record_key)
    _write_json(os.path.join(run_dir, "results.json"), rows)
    _write_csv(os.path.join(run_dir, "results.csv"), rows)
    strategy_summaries = {
        strategy_id: _strategy_summary(strategy_id, rows)
        for strategy_id in state["strategy_ids"]
    }
    oracle_rows, oracle_summary = _oracle_best_evidence(
        state["strategy_ids"],
        rows,
    )
    _write_json(
        os.path.join(run_dir, "oracle_best_evidence_results.json"),
        oracle_rows,
    )
    _write_oracle_csv(
        os.path.join(run_dir, "oracle_best_evidence_results.csv"),
        oracle_rows,
    )
    summary = {
        "dataset": state["dataset"],
        "eligible_sample_count": int(state["eligible_sample_count"]),
        "request_count": len(records),
        "valid_request_count": sum(bool(row.get("valid")) for row in rows),
        "invalid_or_error_request_count": sum(
            not bool(row.get("valid")) for row in rows
        ),
        "strategies": strategy_summaries,
        "oracle_best_evidence": oracle_summary,
        "batch_id": stage["batch_id"],
    }
    _write_json(os.path.join(run_dir, "summary.json"), summary)
    state["phase"] = "completed"
    state["completed_at"] = int(time.time())
    stage["status"] = "completed"
    _save_state(run_dir, state)

    print(f">>> Completed: {run_dir}")
    for strategy_id, metrics in strategy_summaries.items():
        print(
            f">>> {strategy_id}: valid={metrics['valid_sample_count']}/"
            f"{metrics['completed_sample_count']}, "
            f"Hit@1={metrics['hit_rate_at_1']:.4f}, "
            f"MRR={metrics['mean_reciprocal_rank']:.4f}, "
            f"mean-rank={metrics['mean_gt_rank']:.4f}, "
            f"W/T/L={metrics['solo_wins']}/"
            f"{metrics['ties']}/{metrics['solo_losses']}"
        )
    print(
        f">>> Oracle best evidence: "
        f"common-valid={oracle_summary['common_valid_sample_count']}, "
        f"Hit@1={oracle_summary['hit_rate_at_1']:.4f}, "
        f"MRR={oracle_summary['mean_reciprocal_rank']:.4f}, "
        f"mean-rank={oracle_summary['mean_gt_rank']:.4f}, "
        f"W/T/L={oracle_summary['oracle_wins_over_joint']}/"
        f"{oracle_summary['ties_with_joint']}/"
        f"{oracle_summary['oracle_losses_to_joint']}"
    )
    return 0


def _status(run_dir, *, finalize=False):
    run_dir = os.path.abspath(run_dir)
    state = _load_state(run_dir)
    if state.get("phase") == "completed":
        print(f">>> Run already complete: {run_dir}")
        return 0
    if state.get("phase") == "prediction_prepared":
        print(f">>> Prediction Batch prepared: {run_dir}")
        print(f">>> Requests: {state['prediction']['request_count']}")
        return 0
    client, state = _refresh(run_dir)
    stage = state["prediction"]
    batch = stage.get("batch") or {}
    counts = batch.get("request_counts") or {}
    print(f">>> Run: {run_dir}")
    print(f">>> Batch ID: {stage['batch_id']}")
    print(f">>> Status: {stage['status']}")
    print(
        f">>> Requests: completed={counts.get('completed', 0)}, "
        f"failed={counts.get('failed', 0)}, total={counts.get('total', 0)}"
    )
    if finalize:
        if stage["status"] not in BATCH_TERMINAL_STATUSES:
            print(">>> Batch is not terminal; advance again later")
            return 0
        if stage["status"] != "completed":
            raise RuntimeError(
                f"prediction Batch ended with status {stage['status']}"
            )
        return _finalize(run_dir, client, state)
    return 0


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Reuse complete saved strategy evidence and submit one isolated "
            "ranking request per sample and strategy"
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    start = subparsers.add_parser("start")
    start.add_argument("--config", default="config_operator.yaml")
    start.add_argument("--source_run", required=True)
    start.add_argument("--output_dir", required=True)
    start.add_argument("--api_key_env", required=True)
    start.add_argument(
        "--strategy_ids",
        default=",".join(DEFAULT_STRATEGY_IDS),
    )
    start.add_argument("--dry_run", action="store_true")

    submit = subparsers.add_parser("submit")
    submit.add_argument("--run", required=True)

    status = subparsers.add_parser("status")
    status.add_argument("--run", required=True)

    advance = subparsers.add_parser("advance")
    advance.add_argument("--run", required=True)

    args = parser.parse_args()
    if args.command == "start":
        return _initialize(args)
    if args.command == "submit":
        return _submit(args.run)
    if args.command == "status":
        return _status(args.run)
    return _status(args.run, finalize=True)


if __name__ == "__main__":
    raise SystemExit(main())

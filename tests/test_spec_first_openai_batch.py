"""OpenAI Batch API evaluation for spec-first strategies.

The orchestrator supports three dependent asynchronous pipelines:

1. OpenAI Batch: over-generate the configured strategy specs and programs.
2. Local runtime: execute generated programs against the dataset workspace.
3a. Direct mode: OpenAI Batch ranks candidates from raw strategy evidence.
3b. Summary mode: OpenAI Batch summarizes evidence, then another Batch ranks.
3c. Curator mode: OpenAI Batch selects useful executed strategies, writes one
    bundle-completion explanation per candidate, then another Batch ranks from
    those explanations. Selected raw strategy results remain saved for traceability.

Commands:
    python tests/test_spec_first_openai_batch.py start \
        --config config_operator.yaml --split test --sample_count 250

    python tests/test_spec_first_openai_batch.py status --run <output_dir>
    python tests/test_spec_first_openai_batch.py advance --run <output_dir>
"""

import argparse
import csv
import json
import os
import sys
import time

import yaml
from openai import OpenAI


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from code.common import parse_json_from_text
from dataset import BundleZeroShotDataset, set_seed
from main import (
    default_api_key_envs_for_provider,
    openai_model_supports_reasoning,
    resolve_api_key_from_keys,
    stage_model,
    stage_provider,
)
from online_hypothesis_program.schemas import validate_prediction_result
from operator_learning.evidence_curator import (
    candidate_completion_explanations,
    evidence_curator_prompt,
    normalize_evidence_curation,
    select_curated_strategy_evidence,
    validate_evidence_curation,
)
from operator_learning.evidence_summary import (
    candidate_evidence_summary_prompt,
    candidate_summary_prediction_prompt,
    validate_candidate_summaries,
)
from operator_learning.openai_batch import (
    BATCH_TERMINAL_STATUSES,
    batch_output_by_custom_id,
    batch_request,
    download_file,
    extract_response_text,
    plain_object,
    read_jsonl,
    response_request_body,
    retrieve_batch,
    submit_batch,
    write_jsonl,
)
from operator_learning.pipeline import (
    _build_diagnostic_indices,
    _capability_names,
    build_discovery_case,
    build_operator_capability_manifest,
    enrich_case_source_diagnostics,
)
from operator_learning.prompts import induction_prompt
from operator_learning.prompts import strategy_evidence_prediction_prompt
from operator_learning.schemas import (
    resolve_induction_strategies,
    validate_induction_result,
)
from operator_learning.spec_first_prediction import (
    aggregate_prediction_rows,
    build_strategy_evidence,
    evaluate_full_ranking,
)
from operator_learning.spec_first_runtime import (
    execute_strategy_program,
    source_paths_from_capabilities,
)


RESULT_FIELDS = [
    "sample_idx",
    "bundle_id",
    "candidate_count",
    "strategy_count",
    "successful_program_count",
    "selected_strategy_count",
    "raw_evidence_context_count",
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
    "batch_requests",
    "sync_prediction_requests",
    "valid",
    "error",
]

PIPELINE_CODE_PREDICTION = "code_prediction"
PIPELINE_CODE_SUMMARY_PREDICTION = "code_summary_prediction"
PIPELINE_CODE_CURATOR_PREDICTION = "code_curator_prediction"


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


def _load_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"config must contain an object: {path}")
    return value


def _run_config(run_dir):
    return _load_config(os.path.join(run_dir, "config_snapshot.yaml"))


def _state_path(run_dir):
    return os.path.join(run_dir, "state.json")


def _load_state(run_dir):
    path = _state_path(run_dir)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Batch run state not found: {path}")
    value = _read_json(path)
    if not isinstance(value, dict):
        raise ValueError("state.json must contain an object")
    return value


def _save_state(run_dir, state):
    state["updated_at"] = int(time.time())
    _write_json(_state_path(run_dir), state)


def _sample_dir(run_dir, record):
    return os.path.join(
        run_dir,
        "samples",
        f"{int(record['sample_idx']):04d}_bundle_{int(record['bundle_id'])}",
    )


def _batch_path(run_dir, stage, kind):
    return os.path.join(run_dir, "batches", f"{stage}_{kind}.jsonl")


def _custom_id(stage, sample_idx, bundle_id):
    return f"{stage}-{int(sample_idx):04d}-bundle-{int(bundle_id)}"


def _pipeline_mode(state):
    value = str(state.get("pipeline_mode") or "").strip()
    if value in {
        PIPELINE_CODE_PREDICTION,
        PIPELINE_CODE_SUMMARY_PREDICTION,
        PIPELINE_CODE_CURATOR_PREDICTION,
    }:
        return value
    return PIPELINE_CODE_SUMMARY_PREDICTION


def _uses_summary(state):
    return _pipeline_mode(state) == PIPELINE_CODE_SUMMARY_PREDICTION


def _uses_curator(state):
    return _pipeline_mode(state) == PIPELINE_CODE_CURATOR_PREDICTION


def _batch_requests_per_success(state):
    return 3 if (_uses_summary(state) or _uses_curator(state)) else 2


def _select_samples(args, conf, samples):
    if args.sample_idx is not None:
        index = int(args.sample_idx)
        if index < 0 or index >= len(samples):
            raise IndexError(f"sample_idx {index} out of range for {len(samples)}")
        return [(index, samples[index])]
    start = max(0, int(args.start_idx))
    count = int(
        args.sample_count
        if args.sample_count is not None
        else conf.get("operator_prediction_sample_count", 250)
    )
    if count <= 0:
        raise ValueError("--sample_count must be positive")
    end = min(len(samples), start + count)
    return list(enumerate(samples[start:end], start=start))


def _default_output_dir(conf):
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return os.path.abspath(
        os.path.join(
            ROOT,
            "tests",
            "outputs",
            "spec_first_openai_batch",
            f"{conf['dataset']}_{stamp}",
        )
    )


def _default_summary_output_dir(source_run):
    return os.path.abspath(f"{source_run}_summary")


def _role_model(conf, stage):
    role = (
        "code_generation"
        if stage == "code"
        else ("curator" if stage == "curator" else "prediction")
    )
    provider = stage_provider(conf, role)
    if provider != "openai":
        raise ValueError(
            f"OpenAI Batch mode requires OpenAI for {stage}; got {provider}"
        )
    model = stage_model(conf, role)
    if stage == "code":
        key_names = [
            "code_generation_api_key_env",
            "operator_api_key_env",
            "code_api_key_env",
        ]
    elif stage == "summary":
        model = str(conf.get("operator_summary_model") or model).strip()
        key_names = [
            "operator_summary_api_key_env",
            "operator_prediction_api_key_env",
            "code_prediction_api_key_env",
            "code_generation_api_key_env",
            "code_api_key_env",
        ]
    elif stage == "curator":
        key_names = [
            "operator_curator_api_key_env",
            "operator_prediction_api_key_env",
            "code_prediction_api_key_env",
            "code_generation_api_key_env",
            "code_api_key_env",
        ]
    else:
        key_names = [
            "operator_prediction_api_key_env",
            "code_prediction_api_key_env",
            "code_generation_api_key_env",
            "code_api_key_env",
        ]
    return {
        "provider": provider,
        "model": model,
        "api_key_config_names": key_names,
    }


def _client_for_stage(conf, stage):
    model = _role_model(conf, stage)
    api_key, env = resolve_api_key_from_keys(
        conf,
        model["api_key_config_names"],
        default_api_key_envs_for_provider("openai"),
    )
    return OpenAI(api_key=api_key), {
        "provider": "openai",
        "model": model["model"],
        "api_key_env": env,
    }


def _request_body(conf, stage, prompt):
    model = _role_model(conf, stage)["model"]
    if stage == "code":
        max_tokens = int(conf.get("operator_max_output_tokens", 25000))
    elif stage == "summary":
        max_tokens = int(conf.get("operator_summary_max_output_tokens", 5000))
    elif stage == "curator":
        max_tokens = int(conf.get("operator_curator_max_output_tokens", 4000))
    else:
        max_tokens = int(
            conf.get("operator_prediction_max_output_tokens", 2000)
        )
    effort = ""
    configured_effort = str(
        conf.get("openai_reasoning_effort", "")
    ).strip()
    if configured_effort and openai_model_supports_reasoning(model):
        effort = configured_effort
    temperature = (
        float(conf.get("temperature", 0.0))
        if bool(conf.get("openai_send_temperature", False))
        else None
    )
    return response_request_body(
        model=model,
        prompt=prompt,
        max_output_tokens=max_tokens,
        reasoning_effort=effort,
        temperature=temperature,
    )


def _record_map(run_dir):
    records = _read_json(os.path.join(run_dir, "records.json"))
    return {
        int(record["sample_idx"]): record
        for record in records
        if isinstance(record, dict)
    }


def _save_records(run_dir, records_by_index):
    _write_json(
        os.path.join(run_dir, "records.json"),
        [
            records_by_index[index]
            for index in sorted(records_by_index)
        ],
    )


def _stage_batch(state, stage):
    value = state.get("stages", {}).get(stage)
    if not isinstance(value, dict):
        raise ValueError(f"state contains no {stage} stage")
    return value


def _submit_prepared_stage(run_dir, conf, state, stage):
    stage_state = _stage_batch(state, stage)
    input_path = os.path.join(run_dir, stage_state["input_path"])
    client, model_info = _client_for_stage(conf, stage)
    submitted = submit_batch(
        client,
        input_path=input_path,
        metadata={
            "pipeline": f"spec-first-{_pipeline_mode(state).replace('_', '-')}",
            "stage": stage,
            "dataset": state["dataset"],
            "run": os.path.basename(run_dir),
        },
    )
    batch = submitted["batch"]
    stage_state.update(
        {
            "input_file_id": submitted["input_file_id"],
            "batch_id": str(batch["id"]),
            "status": str(batch.get("status") or "validating"),
            "batch": batch,
            "model": model_info,
            "submitted_at": int(time.time()),
        }
    )
    state["phase"] = f"{stage}_submitted"
    _save_state(run_dir, state)
    print(
        f">>> Submitted {stage} batch: {stage_state['batch_id']} "
        f"({stage_state['request_count']} requests)"
    )
    return stage_state


def _refresh_stage(run_dir, conf, state, stage):
    stage_state = _stage_batch(state, stage)
    client, _ = _client_for_stage(conf, stage)
    batch = retrieve_batch(client, stage_state["batch_id"])
    stage_state["batch"] = batch
    stage_state["status"] = str(batch.get("status") or "")
    _save_state(run_dir, state)
    return client, stage_state


def _download_stage_files(run_dir, client, stage_state, stage):
    batch = stage_state.get("batch", {})
    if batch.get("output_file_id"):
        output_path = _batch_path(run_dir, stage, "output")
        download_file(client, batch["output_file_id"], output_path)
        stage_state["output_path"] = os.path.relpath(output_path, run_dir)
    if batch.get("error_file_id"):
        error_path = _batch_path(run_dir, stage, "errors")
        download_file(client, batch["error_file_id"], error_path)
        stage_state["error_path"] = os.path.relpath(error_path, run_dir)


def _stage_outputs(run_dir, stage_state):
    indexed = {}
    output_path = stage_state.get("output_path")
    if output_path:
        indexed.update(
            batch_output_by_custom_id(os.path.join(run_dir, output_path))
        )
    error_path = stage_state.get("error_path")
    if error_path:
        for custom_id, row in batch_output_by_custom_id(
            os.path.join(run_dir, error_path)
        ).items():
            indexed.setdefault(custom_id, row)
    return indexed


def _initialize(args):
    conf = _load_config(args.config)
    strategy_count = int(conf.get("operator_induction_count", 5))
    if strategy_count <= 0:
        raise ValueError("operator_induction_count must be positive")
    dataset = str(getattr(args, "dataset", "") or "").strip()
    if dataset:
        conf["dataset"] = dataset
    api_key_env = str(getattr(args, "api_key_env", "") or "").strip()
    if api_key_env:
        conf["code_generation_api_key_env"] = api_key_env
        conf["operator_summary_api_key_env"] = api_key_env
        conf["operator_curator_api_key_env"] = api_key_env
        conf["operator_prediction_api_key_env"] = api_key_env
    if args.skip_summary and args.with_curator:
        raise ValueError("--skip_summary and --with_curator are mutually exclusive")
    if args.with_curator:
        pipeline_mode = PIPELINE_CODE_CURATOR_PREDICTION
    elif args.skip_summary:
        pipeline_mode = PIPELINE_CODE_PREDICTION
    else:
        pipeline_mode = PIPELINE_CODE_SUMMARY_PREDICTION
    for path_key in ("data_path", "code_workspace_root"):
        configured_path = str(conf.get(path_key) or "").strip()
        if configured_path and not os.path.isabs(configured_path):
            conf[path_key] = os.path.abspath(configured_path)
    set_seed(int(conf.get("seed", 45)))
    split = str(args.split).strip().lower()
    if split not in {"valid", "test"}:
        raise ValueError("--split must be valid or test")

    eval_conf = dict(conf)
    eval_conf["toy_eval"] = -1
    samples = BundleZeroShotDataset(eval_conf, split=split).get_eval_samples()
    selected = _select_samples(args, conf, samples)
    if not selected:
        raise ValueError("no evaluation samples selected")

    run_dir = os.path.abspath(args.output_dir or _default_output_dir(conf))
    if os.path.exists(run_dir) and os.listdir(run_dir):
        raise FileExistsError(f"output directory is not empty: {run_dir}")
    os.makedirs(run_dir, exist_ok=True)
    _write_yaml(os.path.join(run_dir, "config_snapshot.yaml"), conf)

    workspace, source_manifest, capabilities = build_operator_capability_manifest(
        conf
    )
    del workspace
    allowed_sources = _capability_names(capabilities)
    diagnostic_indices = _build_diagnostic_indices(conf, allowed_sources)

    requests = []
    records = {}
    manifest = []
    for sample_idx, sample in selected:
        case = build_discovery_case(sample, conf)
        enrich_case_source_diagnostics(
            case,
            capabilities,
            diagnostic_indices,
        )
        case_dir = os.path.join(
            run_dir,
            "samples",
            f"{sample_idx:04d}_bundle_{int(sample['bundle_id'])}",
        )
        prompt = induction_prompt(
            case,
            capabilities,
            [],
            strategy_count,
            text_only=False,
        )
        compact_case = {
            "dataset": case["dataset"],
            "partial_items": case["partial_items"],
            "candidate_items": case["candidate_items"],
            "source_diagnostics": case["source_diagnostics"],
        }
        _write_json(os.path.join(case_dir, "case.json"), compact_case)
        _write_text(os.path.join(case_dir, "code", "input.txt"), prompt)
        custom_id = _custom_id("code", sample_idx, sample["bundle_id"])
        requests.append(
            batch_request(
                custom_id=custom_id,
                body=_request_body(conf, "code", prompt),
            )
        )
        record = {
            "sample_idx": int(sample_idx),
            "bundle_id": int(sample["bundle_id"]),
            "true_label": str(sample["true_option_char"]),
            "candidate_count": len(case["candidate_items"]),
            "code_custom_id": custom_id,
            "code_status": "pending",
            "summary_status": (
                "pending"
                if pipeline_mode == PIPELINE_CODE_SUMMARY_PREDICTION
                else "skipped"
            ),
            "curator_status": (
                "pending"
                if pipeline_mode == PIPELINE_CODE_CURATOR_PREDICTION
                else "skipped"
            ),
            "prediction_status": "pending",
            "strategy_count": 0,
            "successful_program_count": 0,
            "selected_strategy_count": 0,
            "raw_evidence_context_count": 0,
            "evidence_context_count": 0,
            "batch_requests": 0,
            "error": "",
        }
        records[int(sample_idx)] = record
        manifest.append(
            {
                "sample_idx": int(sample_idx),
                "bundle_id": int(sample["bundle_id"]),
                "candidate_item_ids": [
                    int(item["item_id"])
                    for item in case["candidate_items"]
                ],
                "code_custom_id": custom_id,
            }
        )

    input_path = _batch_path(run_dir, "code", "input")
    input_info = write_jsonl(input_path, requests)
    _save_records(run_dir, records)
    _write_json(os.path.join(run_dir, "sample_manifest.json"), manifest)
    _write_json(os.path.join(run_dir, "source_manifest.json"), source_manifest)
    code_model = _role_model(conf, "code")
    prediction_model = _role_model(conf, "prediction")
    summary_model = (
        _role_model(conf, "summary")
        if pipeline_mode == PIPELINE_CODE_SUMMARY_PREDICTION
        else None
    )
    curator_model = (
        _role_model(conf, "curator")
        if pipeline_mode == PIPELINE_CODE_CURATOR_PREDICTION
        else None
    )
    if pipeline_mode == PIPELINE_CODE_SUMMARY_PREDICTION:
        batch_stages = ["code", "summary", "prediction"]
    elif pipeline_mode == PIPELINE_CODE_CURATOR_PREDICTION:
        batch_stages = ["code", "curator", "prediction"]
    else:
        batch_stages = ["code", "prediction"]
    _write_json(
        os.path.join(run_dir, "run.json"),
        {
            "phase": f"openai_batch_spec_first_{pipeline_mode}",
            "dataset": conf["dataset"],
            "split": split,
            "start_idx": selected[0][0],
            "requested_sample_count": len(selected),
            "strategies_per_sample": strategy_count,
            "pipeline_mode": pipeline_mode,
            "batch_stages": batch_stages,
            "local_stage": "guarded generated-program execution",
            "code_model": code_model["model"],
            "summary_model": (
                summary_model["model"] if summary_model is not None else None
            ),
            "curator_model": (
                curator_model["model"] if curator_model is not None else None
            ),
            "prediction_model": prediction_model["model"],
            "created_at": int(time.time()),
        },
    )
    state = {
        "schema_version": 1,
        "phase": "code_prepared",
        "dataset": str(conf["dataset"]),
        "split": split,
        "pipeline_mode": pipeline_mode,
        "requested_sample_count": len(selected),
        "created_at": int(time.time()),
        "updated_at": int(time.time()),
        "stages": {
            "code": {
                "status": "prepared",
                "input_path": os.path.relpath(input_path, run_dir),
                **input_info,
            }
        },
    }
    _save_state(run_dir, state)
    print(f">>> Dataset: {conf['dataset']}")
    print(f">>> Samples: {len(selected)}")
    print(f">>> Pipeline: {pipeline_mode}")
    if api_key_env:
        print(f">>> API key environment: {api_key_env}")
    print(f">>> Code Batch JSONL: {input_path}")
    print(f">>> Requests: {input_info['request_count']}")
    print(f">>> Input bytes: {input_info['input_bytes']}")
    print(f">>> Output: {run_dir}")
    if args.dry_run:
        print(">>> Dry run: code Batch prepared but not submitted")
    else:
        _submit_prepared_stage(run_dir, conf, state, "code")
    return 0


def _initialize_summary_from_run(args):
    source_run = os.path.abspath(args.source_run)
    source_state = _load_state(source_run)
    if source_state.get("phase") != "completed":
        raise RuntimeError(
            f"source run must be completed; got {source_state.get('phase')}"
        )
    conf = _run_config(source_run)
    api_key_env = str(args.api_key_env or "").strip()
    if not api_key_env:
        raise ValueError("--api_key_env must name an environment variable")
    conf["operator_summary_api_key_env"] = api_key_env
    conf["operator_prediction_api_key_env"] = api_key_env

    run_dir = os.path.abspath(
        args.output_dir or _default_summary_output_dir(source_run)
    )
    if os.path.exists(run_dir) and os.listdir(run_dir):
        raise FileExistsError(f"output directory is not empty: {run_dir}")
    os.makedirs(run_dir, exist_ok=True)
    _write_yaml(os.path.join(run_dir, "config_snapshot.yaml"), conf)

    source_records = _record_map(source_run)
    records = {}
    requests = []
    for sample_idx in sorted(source_records):
        source_record = source_records[sample_idx]
        record = dict(source_record)
        record.pop("summary_custom_id", None)
        record.pop("prediction_custom_id", None)
        record["sync_prediction_requests"] = 0
        if record.get("code_status") == "success":
            source_case_dir = _sample_dir(source_run, source_record)
            case = _read_json(os.path.join(source_case_dir, "case.json"))
            strategy_evidence = _read_json(
                os.path.join(
                    source_case_dir,
                    "rendered_strategy_evidence.json",
                )
            )
            case_dir = _sample_dir(run_dir, record)
            _write_json(os.path.join(case_dir, "case.json"), case)
            _write_json(
                os.path.join(case_dir, "rendered_strategy_evidence.json"),
                strategy_evidence,
            )
            summary_prompt = candidate_evidence_summary_prompt(
                partial_items=case["partial_items"],
                candidate_items=case["candidate_items"],
                strategy_evidence=strategy_evidence,
            )
            custom_id = _custom_id(
                "summary",
                record["sample_idx"],
                record["bundle_id"],
            )
            _write_text(
                os.path.join(case_dir, "summary", "input.txt"),
                summary_prompt,
            )
            requests.append(
                batch_request(
                    custom_id=custom_id,
                    body=_request_body(conf, "summary", summary_prompt),
                )
            )
            record.update(
                {
                    "summary_custom_id": custom_id,
                    "summary_status": "pending",
                    "prediction_status": "pending",
                    "batch_requests": 1,
                    "error": "",
                }
            )
        else:
            record.update(
                {
                    "summary_status": "skipped",
                    "prediction_status": "skipped",
                    "batch_requests": 1,
                }
            )
        records[sample_idx] = record

    if not requests:
        raise RuntimeError("source run contains no valid evidence to summarize")
    input_path = _batch_path(run_dir, "summary", "input")
    input_info = write_jsonl(input_path, requests)
    _save_records(run_dir, records)

    source_manifest_path = os.path.join(source_run, "source_manifest.json")
    if os.path.isfile(source_manifest_path):
        _write_json(
            os.path.join(run_dir, "source_manifest.json"),
            _read_json(source_manifest_path),
        )
    source_code_stage = source_state.get("stages", {}).get("code") or {}
    summary_model = _role_model(conf, "summary")
    prediction_model = _role_model(conf, "prediction")
    _write_json(
        os.path.join(run_dir, "run.json"),
        {
            "phase": "openai_batch_reused_code_summary_prediction",
            "dataset": conf["dataset"],
            "split": source_state.get("split"),
            "requested_sample_count": len(records),
            "valid_source_evidence_count": len(requests),
            "pipeline_mode": PIPELINE_CODE_SUMMARY_PREDICTION,
            "batch_stages": ["summary", "prediction"],
            "reused_stage": "code generation and guarded program execution",
            "source_run": source_run,
            "source_code_batch_id": source_code_stage.get("batch_id"),
            "summary_model": summary_model["model"],
            "prediction_model": prediction_model["model"],
            "api_key_env": api_key_env,
            "created_at": int(time.time()),
        },
    )
    state = {
        "schema_version": 1,
        "phase": "summary_prepared",
        "dataset": str(conf["dataset"]),
        "split": source_state.get("split"),
        "pipeline_mode": PIPELINE_CODE_SUMMARY_PREDICTION,
        "source_run": source_run,
        "requested_sample_count": len(records),
        "created_at": int(time.time()),
        "updated_at": int(time.time()),
        "stages": {
            "code": {
                "status": "reused",
                "batch_id": source_code_stage.get("batch_id"),
                "source_run": source_run,
            },
            "summary": {
                "status": "prepared",
                "input_path": os.path.relpath(input_path, run_dir),
                **input_info,
            },
        },
    }
    _save_state(run_dir, state)
    print(f">>> Dataset: {conf['dataset']}")
    print(f">>> Source run: {source_run}")
    print(">>> Code generation and execution: reused, not rerun")
    print(f">>> Summary requests: {input_info['request_count']}")
    print(f">>> Summary input bytes: {input_info['input_bytes']}")
    print(f">>> API key environment: {api_key_env}")
    print(f">>> Output: {run_dir}")
    if args.dry_run:
        print(">>> Dry run: summary Batch prepared but not submitted")
    else:
        _submit_prepared_stage(run_dir, conf, state, "summary")
    return 0


def _request_after_code(
    conf,
    state,
    record,
    case,
    strategy_specs,
    strategy_evidence,
):
    if _uses_curator(state):
        stage = "curator"
        prompt = evidence_curator_prompt(
            partial_items=case["partial_items"],
            candidate_items=case["candidate_items"],
            strategy_specs=strategy_specs,
            strategy_evidence=strategy_evidence,
            max_selected_strategies=int(
                conf.get("operator_curator_max_selected_strategies", 3)
            ),
        )
    elif _uses_summary(state):
        stage = "summary"
        prompt = candidate_evidence_summary_prompt(
            partial_items=case["partial_items"],
            candidate_items=case["candidate_items"],
            strategy_evidence=strategy_evidence,
        )
    else:
        stage = "prediction"
        prompt = strategy_evidence_prediction_prompt(
            dataset=case["dataset"],
            partial_items=case["partial_items"],
            candidate_items=case["candidate_items"],
            strategy_evidence=strategy_evidence,
        )
    custom_id = _custom_id(
        stage,
        record["sample_idx"],
        record["bundle_id"],
    )
    return stage, prompt, custom_id, batch_request(
        custom_id=custom_id,
        body=_request_body(conf, stage, prompt),
    )


def _prepare_after_code_stage(run_dir, conf, state, client, stage_state):
    strategy_count = int(conf.get("operator_induction_count", 5))
    _download_stage_files(run_dir, client, stage_state, "code")
    outputs = _stage_outputs(run_dir, stage_state)
    records = _record_map(run_dir)
    workspace, _, capabilities = build_operator_capability_manifest(conf)
    allowed_sources = _capability_names(capabilities)
    all_source_paths = source_paths_from_capabilities(
        workspace,
        capabilities,
    )
    requests = []

    for sample_idx in sorted(records):
        record = records[sample_idx]
        case_dir = _sample_dir(run_dir, record)
        if record.get("code_status") == "success":
            case = _read_json(os.path.join(case_dir, "case.json"))
            parsed = _read_json(
                os.path.join(case_dir, "code", "parsed_response.json")
            )
            strategy_evidence = _read_json(
                os.path.join(case_dir, "rendered_strategy_evidence.json")
            )
            next_stage, next_prompt, custom_id, request = _request_after_code(
                conf,
                state,
                record,
                case,
                parsed["strategy_specs"],
                strategy_evidence,
            )
            _write_text(
                os.path.join(case_dir, next_stage, "input.txt"),
                next_prompt,
            )
            if next_stage == "summary":
                record["summary_custom_id"] = custom_id
            elif next_stage == "curator":
                record["curator_custom_id"] = custom_id
            else:
                record.update(
                    {
                        "summary_status": "skipped",
                        "prediction_custom_id": custom_id,
                    }
                )
            requests.append(request)
            continue
        if record.get("code_status") == "failed":
            continue
        try:
            output_row = outputs.get(record["code_custom_id"])
            raw = extract_response_text(output_row)
            parsed = normalize_evidence_curation(
                parse_json_from_text(raw)
            )
            issues = validate_induction_result(
                parsed,
                expected_count=strategy_count,
                allowed_source_names=allowed_sources,
            )
            _write_text(os.path.join(case_dir, "code", "output.txt"), raw)
            _write_json(
                os.path.join(case_dir, "code", "parsed_response.json"),
                parsed,
            )
            _write_json(
                os.path.join(case_dir, "code", "validation_issues.json"),
                issues,
            )
            if issues:
                raise ValueError(
                    "invalid spec-first generation: " + " | ".join(issues)
                )

            case = _read_json(os.path.join(case_dir, "case.json"))
            strategies = resolve_induction_strategies(parsed)
            execution_reports = []
            for strategy in strategies:
                report = execute_strategy_program(
                    code=strategy["code"],
                    strategy_id=strategy["strategy_id"],
                    required_sources=strategy["required_sources"],
                    partial_items=case["partial_items"],
                    candidate_items=case["candidate_items"],
                    all_source_paths=all_source_paths,
                    case_dir=case_dir,
                    conf=conf,
                )
                execution_reports.append(report)
            _write_json(
                os.path.join(case_dir, "execution_reports.json"),
                execution_reports,
            )
            successful = [
                report
                for report in execution_reports
                if report.get("success")
            ]
            if not successful:
                raise RuntimeError(
                    f"all {strategy_count} generated programs failed"
                )

            labels = [
                str(candidate["label"])
                for candidate in case["candidate_items"]
            ]
            strategy_evidence = build_strategy_evidence(
                specs=parsed["strategy_specs"],
                execution_reports=execution_reports,
                candidate_labels=labels,
                max_contexts_per_candidate=int(
                    conf.get(
                        "operator_prediction_max_contexts_per_candidate",
                        0,
                    )
                ),
                max_context_chars=int(
                    conf.get("operator_prediction_max_context_chars", 0)
                ),
            )
            _write_json(
                os.path.join(case_dir, "rendered_strategy_evidence.json"),
                strategy_evidence,
            )
            next_stage, next_prompt, custom_id, request = _request_after_code(
                conf,
                state,
                record,
                case,
                parsed["strategy_specs"],
                strategy_evidence,
            )
            _write_text(
                os.path.join(case_dir, next_stage, "input.txt"),
                next_prompt,
            )
            requests.append(request)
            updates = {
                "code_status": "success",
                "strategy_count": len(strategies),
                "successful_program_count": len(successful),
                "raw_evidence_context_count": sum(
                    len(candidate.get("contexts", []))
                    for evidence in strategy_evidence
                    for candidate in evidence.get(
                        "candidate_evidence",
                        [],
                    )
                ),
                "evidence_context_count": sum(
                    len(candidate.get("contexts", []))
                    for evidence in strategy_evidence
                    for candidate in evidence.get(
                        "candidate_evidence",
                        [],
                    )
                ),
                "batch_requests": 1,
            }
            if next_stage == "summary":
                updates["summary_custom_id"] = custom_id
            elif next_stage == "curator":
                updates["curator_custom_id"] = custom_id
            else:
                updates.update(
                    {
                        "summary_status": "skipped",
                        "prediction_custom_id": custom_id,
                    }
                )
            record.update(updates)
            print(
                f"[code {sample_idx}] bundle_{record['bundle_id']} | "
                f"programs={len(successful)}/{strategy_count}"
            )
        except Exception as error:
            record.update(
                {
                    "code_status": "failed",
                    "summary_status": "skipped",
                    "curator_status": "skipped",
                    "prediction_status": "skipped",
                    "batch_requests": 1,
                    "error": str(error),
                }
            )
            _write_json(
                os.path.join(case_dir, "code", "error.json"),
                {"error": str(error)},
            )
            print(
                f"[code {sample_idx}] bundle_{record['bundle_id']} | "
                f"error={error}"
            )
        _save_records(run_dir, records)

    _save_records(run_dir, records)
    if _uses_curator(state):
        next_stage = "curator"
    elif _uses_summary(state):
        next_stage = "summary"
    else:
        next_stage = "prediction"
    if not requests:
        state["phase"] = "failed"
        state["error"] = (
            "no valid code-generation results to summarize"
            if next_stage == "summary"
            else (
                "no valid code-generation results to curate"
                if next_stage == "curator"
                else "no valid code-generation results to predict"
            )
        )
        _save_state(run_dir, state)
        raise RuntimeError(state["error"])
    input_path = _batch_path(run_dir, next_stage, "input")
    input_info = write_jsonl(input_path, requests)
    state["stages"][next_stage] = {
        "status": "prepared",
        "input_path": os.path.relpath(input_path, run_dir),
        **input_info,
    }
    state["phase"] = f"{next_stage}_prepared"
    _save_state(run_dir, state)
    _submit_prepared_stage(run_dir, conf, state, next_stage)
    return 0


def _prepare_after_curator_stage(run_dir, conf, state, client, stage_state):
    _download_stage_files(run_dir, client, stage_state, "curator")
    outputs = _stage_outputs(run_dir, stage_state)
    records = _record_map(run_dir)
    requests = []
    max_selected_strategies = int(
        conf.get("operator_curator_max_selected_strategies", 3)
    )

    for sample_idx in sorted(records):
        record = records[sample_idx]
        if record.get("code_status") != "success":
            continue
        case_dir = _sample_dir(run_dir, record)
        if record.get("curator_status") == "success":
            case = _read_json(os.path.join(case_dir, "case.json"))
            explanations = _read_json(
                os.path.join(
                    case_dir,
                    "candidate_completion_explanations.json",
                )
            )
            prediction_prompt = candidate_summary_prediction_prompt(
                dataset=case["dataset"],
                partial_items=case["partial_items"],
                candidate_items=case["candidate_items"],
                candidate_summaries=explanations,
                evidence_mode="completion_explanation",
            )
            requests.append(
                batch_request(
                    custom_id=record["prediction_custom_id"],
                    body=_request_body(
                        conf,
                        "prediction",
                        prediction_prompt,
                    ),
                )
            )
            continue
        if record.get("curator_status") == "failed":
            continue
        try:
            raw = extract_response_text(
                outputs.get(record["curator_custom_id"])
            )
            parsed = parse_json_from_text(raw)
            case = _read_json(os.path.join(case_dir, "case.json"))
            labels = [
                str(candidate["label"])
                for candidate in case["candidate_items"]
            ]
            strategy_evidence = _read_json(
                os.path.join(case_dir, "rendered_strategy_evidence.json")
            )
            issues = validate_evidence_curation(
                parsed,
                strategy_evidence=strategy_evidence,
                candidate_labels=labels,
                max_selected_strategies=max_selected_strategies,
            )
            _write_text(
                os.path.join(case_dir, "curator", "output.txt"),
                raw,
            )
            _write_json(
                os.path.join(case_dir, "curator", "parsed_response.json"),
                parsed,
            )
            _write_json(
                os.path.join(case_dir, "curator", "validation_issues.json"),
                issues,
            )
            if issues:
                raise ValueError(
                    "invalid evidence curation: " + " | ".join(issues)
                )

            curated_strategy_evidence = select_curated_strategy_evidence(
                strategy_evidence,
                parsed,
            )
            _write_json(
                os.path.join(case_dir, "curated_strategy_evidence.json"),
                curated_strategy_evidence,
            )
            explanations = candidate_completion_explanations(parsed)
            _write_json(
                os.path.join(
                    case_dir,
                    "candidate_completion_explanations.json",
                ),
                explanations,
            )
            prediction_prompt = candidate_summary_prediction_prompt(
                dataset=case["dataset"],
                partial_items=case["partial_items"],
                candidate_items=case["candidate_items"],
                candidate_summaries=explanations,
                evidence_mode="completion_explanation",
            )
            _write_text(
                os.path.join(case_dir, "prediction", "input.txt"),
                prediction_prompt,
            )
            custom_id = _custom_id(
                "prediction",
                record["sample_idx"],
                record["bundle_id"],
            )
            requests.append(
                batch_request(
                    custom_id=custom_id,
                    body=_request_body(
                        conf,
                        "prediction",
                        prediction_prompt,
                    ),
                )
            )
            selected_context_count = sum(
                len(candidate.get("contexts", []))
                for evidence in curated_strategy_evidence
                for candidate in evidence.get("candidate_evidence", [])
            )
            record.update(
                {
                    "curator_status": "success",
                    "selected_strategy_count": len(
                        parsed["selected_strategies"]
                    ),
                    "evidence_context_count": selected_context_count,
                    "prediction_custom_id": custom_id,
                    "batch_requests": 2,
                }
            )
            print(
                f"[curator {sample_idx}] bundle_{record['bundle_id']} | "
                f"strategies={record['selected_strategy_count']} "
                f"evidence={selected_context_count}"
            )
        except Exception as error:
            record.update(
                {
                    "curator_status": "failed",
                    "prediction_status": "skipped",
                    "batch_requests": 2,
                    "error": str(error),
                }
            )
            _write_json(
                os.path.join(case_dir, "curator", "error.json"),
                {"error": str(error)},
            )
            print(
                f"[curator {sample_idx}] bundle_{record['bundle_id']} | "
                f"error={error}"
            )
        _save_records(run_dir, records)

    _save_records(run_dir, records)
    if not requests:
        state["phase"] = "failed"
        state["error"] = "no valid curator results to predict"
        _save_state(run_dir, state)
        raise RuntimeError(state["error"])
    input_path = _batch_path(run_dir, "prediction", "input")
    input_info = write_jsonl(input_path, requests)
    state["stages"]["prediction"] = {
        "status": "prepared",
        "input_path": os.path.relpath(input_path, run_dir),
        **input_info,
    }
    state["phase"] = "prediction_prepared"
    _save_state(run_dir, state)
    _submit_prepared_stage(run_dir, conf, state, "prediction")
    return 0


def _prepare_prediction_stage(run_dir, conf, state, client, stage_state):
    _download_stage_files(run_dir, client, stage_state, "summary")
    outputs = _stage_outputs(run_dir, stage_state)
    records = _record_map(run_dir)
    requests = []

    for sample_idx in sorted(records):
        record = records[sample_idx]
        if record.get("code_status") != "success":
            continue
        case_dir = _sample_dir(run_dir, record)
        if record.get("summary_status") == "success":
            case = _read_json(os.path.join(case_dir, "case.json"))
            candidate_summaries = _read_json(
                os.path.join(case_dir, "candidate_summaries.json")
            )
            prediction_prompt = candidate_summary_prediction_prompt(
                dataset=case["dataset"],
                partial_items=case["partial_items"],
                candidate_items=case["candidate_items"],
                candidate_summaries=candidate_summaries,
            )
            requests.append(
                batch_request(
                    custom_id=record["prediction_custom_id"],
                    body=_request_body(
                        conf,
                        "prediction",
                        prediction_prompt,
                    ),
                )
            )
            continue
        if record.get("summary_status") == "failed":
            continue
        try:
            raw = extract_response_text(
                outputs.get(record["summary_custom_id"])
            )
            parsed = parse_json_from_text(raw)
            case = _read_json(os.path.join(case_dir, "case.json"))
            labels = [
                str(candidate["label"])
                for candidate in case["candidate_items"]
            ]
            issues = validate_candidate_summaries(parsed, labels)
            _write_text(os.path.join(case_dir, "summary", "output.txt"), raw)
            _write_json(
                os.path.join(case_dir, "summary", "parsed_response.json"),
                parsed,
            )
            _write_json(
                os.path.join(case_dir, "summary", "validation_issues.json"),
                issues,
            )
            if issues:
                raise ValueError(
                    "invalid candidate summaries: " + " | ".join(issues)
                )

            candidate_summaries = parsed["candidate_summaries"]
            _write_json(
                os.path.join(case_dir, "candidate_summaries.json"),
                candidate_summaries,
            )
            prediction_prompt = candidate_summary_prediction_prompt(
                dataset=case["dataset"],
                partial_items=case["partial_items"],
                candidate_items=case["candidate_items"],
                candidate_summaries=candidate_summaries,
            )
            _write_text(
                os.path.join(case_dir, "prediction", "input.txt"),
                prediction_prompt,
            )
            custom_id = _custom_id(
                "prediction",
                record["sample_idx"],
                record["bundle_id"],
            )
            requests.append(
                batch_request(
                    custom_id=custom_id,
                    body=_request_body(
                        conf,
                        "prediction",
                        prediction_prompt,
                    ),
                )
            )
            record.update(
                {
                    "summary_status": "success",
                    "prediction_custom_id": custom_id,
                    "batch_requests": 2,
                }
            )
            print(
                f"[summary {sample_idx}] bundle_{record['bundle_id']} | success"
            )
        except Exception as error:
            record.update(
                {
                    "summary_status": "failed",
                    "prediction_status": "skipped",
                    "batch_requests": 2,
                    "error": str(error),
                }
            )
            _write_json(
                os.path.join(case_dir, "summary", "error.json"),
                {"error": str(error)},
            )
            print(
                f"[summary {sample_idx}] bundle_{record['bundle_id']} | "
                f"error={error}"
            )
        _save_records(run_dir, records)

    _save_records(run_dir, records)
    if not requests:
        state["phase"] = "failed"
        state["error"] = "no valid summary results to predict"
        _save_state(run_dir, state)
        raise RuntimeError(state["error"])
    input_path = _batch_path(run_dir, "prediction", "input")
    input_info = write_jsonl(input_path, requests)
    state["stages"]["prediction"] = {
        "status": "prepared",
        "input_path": os.path.relpath(input_path, run_dir),
        **input_info,
    }
    state["phase"] = "prediction_prepared"
    _save_state(run_dir, state)
    _submit_prepared_stage(run_dir, conf, state, "prediction")
    return 0


def _error_result(record):
    return {
        "sample_idx": int(record["sample_idx"]),
        "bundle_id": int(record["bundle_id"]),
        "candidate_count": int(record.get("candidate_count", 0)),
        "strategy_count": int(record.get("strategy_count", 0)),
        "successful_program_count": int(
            record.get("successful_program_count", 0)
        ),
        "selected_strategy_count": int(
            record.get("selected_strategy_count", 0)
        ),
        "raw_evidence_context_count": int(
            record.get("raw_evidence_context_count", 0)
        ),
        "evidence_context_count": int(
            record.get("evidence_context_count", 0)
        ),
        "prediction": None,
        "ranking": [],
        "true_label": str(record["true_label"]),
        "hit": False,
        "gt_rank": None,
        "reciprocal_rank": 0.0,
        "hit_at_1": False,
        "hit_at_3": False,
        "hit_at_5": False,
        "batch_requests": int(record.get("batch_requests", 0)),
        "sync_prediction_requests": int(
            record.get("sync_prediction_requests", 0)
        ),
        "valid": False,
        "error": str(record.get("error") or "upstream stage failed"),
    }


def _finalize_predictions(run_dir, conf, state, client, stage_state):
    del conf
    _download_stage_files(run_dir, client, stage_state, "prediction")
    outputs = _stage_outputs(run_dir, stage_state)
    records = _record_map(run_dir)
    rows = []

    for sample_idx in sorted(records):
        record = records[sample_idx]
        if (
            record.get("code_status") != "success"
            or not record.get("prediction_custom_id")
        ):
            rows.append(_error_result(record))
            continue
        case_dir = _sample_dir(run_dir, record)
        try:
            raw = extract_response_text(
                outputs.get(record["prediction_custom_id"])
            )
            parsed = parse_json_from_text(raw)
            case = _read_json(os.path.join(case_dir, "case.json"))
            labels = [
                str(candidate["label"])
                for candidate in case["candidate_items"]
            ]
            issues = validate_prediction_result(parsed, labels)
            _write_text(
                os.path.join(case_dir, "prediction", "output.txt"),
                raw,
            )
            _write_json(
                os.path.join(case_dir, "prediction", "parsed_response.json"),
                parsed,
            )
            _write_json(
                os.path.join(
                    case_dir,
                    "prediction",
                    "validation_issues.json",
                ),
                issues,
            )
            if issues:
                raise ValueError(
                    "invalid final prediction: " + " | ".join(issues)
                )
            evaluation = evaluate_full_ranking(
                parsed,
                record["true_label"],
            )
            request_count = _batch_requests_per_success(state)
            row = {
                "sample_idx": int(record["sample_idx"]),
                "bundle_id": int(record["bundle_id"]),
                "candidate_count": int(record["candidate_count"]),
                "strategy_count": int(record["strategy_count"]),
                "successful_program_count": int(
                    record["successful_program_count"]
                ),
                "selected_strategy_count": int(
                    record.get("selected_strategy_count", 0)
                ),
                "raw_evidence_context_count": int(
                    record.get("raw_evidence_context_count", 0)
                ),
                "evidence_context_count": int(
                    record["evidence_context_count"]
                ),
                **evaluation,
                "batch_requests": request_count,
                "sync_prediction_requests": 0,
                "valid": True,
                "error": "",
            }
            record.update(
                {
                    "prediction_status": "success",
                    "batch_requests": request_count,
                    "error": "",
                }
            )
            _write_json(os.path.join(case_dir, "evaluation.json"), row)
            print(
                f"[prediction {sample_idx}] bundle_{record['bundle_id']} | "
                f"pred={row['prediction']} true={row['true_label']} "
                f"GT-rank={row['gt_rank']}"
            )
        except Exception as error:
            request_count = _batch_requests_per_success(state)
            record.update(
                {
                    "prediction_status": "failed",
                    "batch_requests": request_count,
                    "error": str(error),
                }
            )
            row = _error_result(record)
            _write_json(
                os.path.join(case_dir, "prediction", "error.json"),
                {"error": str(error)},
            )
            print(
                f"[prediction {sample_idx}] bundle_{record['bundle_id']} | "
                f"error={error}"
            )
        rows.append(row)

    _save_records(run_dir, records)
    _write_json(os.path.join(run_dir, "results.json"), rows)
    _write_csv(os.path.join(run_dir, "results.csv"), rows)
    summary = {
        "requested_sample_count": int(state["requested_sample_count"]),
        **aggregate_prediction_rows(rows),
        "mean_successful_program_count": (
            sum(
                int(row.get("successful_program_count", 0))
                for row in rows
            )
            / len(rows)
            if rows
            else 0.0
        ),
        "total_batch_requests": sum(
            int(row.get("batch_requests", 0)) for row in rows
        ),
        "batch_ids": {
            stage: value.get("batch_id")
            for stage, value in state.get("stages", {}).items()
        },
    }
    _write_json(os.path.join(run_dir, "summary.json"), summary)
    state["phase"] = "completed"
    state["completed_at"] = int(time.time())
    _save_state(run_dir, state)
    print("-" * 56)
    print(
        f">>> Valid samples: {summary['valid_sample_count']} / "
        f"{summary['requested_sample_count']}"
    )
    print(f">>> Hit@1: {summary['hit_rate_at_1']:.4f}")
    print(f">>> Hit@3: {summary['hit_rate_at_3']:.4f}")
    print(f">>> Hit@5: {summary['hit_rate_at_5']:.4f}")
    print(f">>> Mean reciprocal rank: {summary['mean_reciprocal_rank']:.4f}")
    print(f">>> Mean GT rank: {summary['mean_gt_rank']:.4f}")
    print(f">>> Total Batch requests: {summary['total_batch_requests']}")
    print(f">>> Output: {run_dir}")
    return 0


def _sync_output_text(response):
    direct = getattr(response, "output_text", None)
    if isinstance(direct, str) and direct:
        return direct
    return extract_response_text(
        {
            "response": {
                "status_code": 200,
                "body": plain_object(response),
            },
            "error": None,
        }
    )


def _save_sync_results(run_dir, state, records):
    rows = []
    for sample_idx in sorted(records):
        record = records[sample_idx]
        case_dir = _sample_dir(run_dir, record)
        evaluation_path = os.path.join(case_dir, "evaluation.json")
        if (
            record.get("prediction_status") == "success"
            and os.path.isfile(evaluation_path)
        ):
            rows.append(_read_json(evaluation_path))
        else:
            rows.append(_error_result(record))

    _write_json(os.path.join(run_dir, "results.json"), rows)
    _write_csv(os.path.join(run_dir, "results.csv"), rows)
    summary = {
        "requested_sample_count": int(state["requested_sample_count"]),
        **aggregate_prediction_rows(rows),
        "mean_successful_program_count": (
            sum(
                int(row.get("successful_program_count", 0))
                for row in rows
            )
            / len(rows)
            if rows
            else 0.0
        ),
        "total_batch_requests": sum(
            int(row.get("batch_requests", 0)) for row in rows
        ),
        "total_sync_prediction_requests": sum(
            int(row.get("sync_prediction_requests", 0)) for row in rows
        ),
        "prediction_delivery": "synchronous",
        "batch_ids": {
            stage: value.get("batch_id")
            for stage, value in state.get("stages", {}).items()
            if value.get("batch_id")
        },
    }
    _write_json(os.path.join(run_dir, "summary_partial.json"), summary)
    return rows, summary


def _predict_sync(args):
    run_dir = os.path.abspath(args.run)
    conf = _run_config(run_dir)
    state = _load_state(run_dir)
    phase = str(state.get("phase") or "")
    if phase == "completed":
        print(f">>> Run already complete: {run_dir}")
        return 0
    if phase not in {"prediction_prepared", "prediction_sync"}:
        raise RuntimeError(
            "synchronous prediction requires a prepared prediction stage; "
            f"current phase is {phase}"
        )
    prediction_stage = _stage_batch(state, "prediction")
    if prediction_stage.get("batch_id"):
        raise RuntimeError(
            "prediction Batch was already submitted; refusing duplicate "
            "synchronous requests"
        )

    input_path = os.path.join(run_dir, prediction_stage["input_path"])
    requests = {
        str(row.get("custom_id") or ""): row
        for row in read_jsonl(input_path)
        if isinstance(row, dict) and row.get("custom_id")
    }
    records = _record_map(run_dir)
    client, model_info = _client_for_stage(conf, "prediction")
    state["phase"] = "prediction_sync"
    state["prediction_delivery"] = "synchronous"
    state.setdefault(
        "sync_prediction",
        {
            "status": "in_progress",
            "model": model_info,
            "completed": 0,
            "failed": 0,
            "total": len(requests),
        },
    )
    _save_state(run_dir, state)

    completed = sum(
        1
        for record in records.values()
        if record.get("prediction_status") == "success"
    )
    print(f">>> Run: {run_dir}")
    print(f">>> Prediction delivery: synchronous, sequential")
    print(f">>> Requests: {len(requests)} ({completed} already completed)")

    for sample_idx in sorted(records):
        record = records[sample_idx]
        if record.get("code_status") != "success":
            continue
        if record.get("prediction_status") == "success":
            continue
        custom_id = str(record.get("prediction_custom_id") or "")
        request = requests.get(custom_id)
        if not request:
            record.update(
                {
                    "prediction_status": "failed",
                    "error": f"prediction request not found: {custom_id}",
                }
            )
            _save_records(run_dir, records)
            continue

        case_dir = _sample_dir(run_dir, record)
        body = request.get("body")
        if not isinstance(body, dict):
            record.update(
                {
                    "prediction_status": "failed",
                    "error": "prediction request body must be an object",
                }
            )
            _save_records(run_dir, records)
            continue

        record["sync_prediction_requests"] = (
            int(record.get("sync_prediction_requests", 0)) + 1
        )
        try:
            response = client.responses.create(**body)
            raw = _sync_output_text(response)
            parsed = parse_json_from_text(raw)
            case = _read_json(os.path.join(case_dir, "case.json"))
            labels = [
                str(candidate["label"])
                for candidate in case["candidate_items"]
            ]
            issues = validate_prediction_result(parsed, labels)
            _write_text(
                os.path.join(case_dir, "prediction", "output.txt"),
                raw,
            )
            _write_json(
                os.path.join(case_dir, "prediction", "parsed_response.json"),
                parsed,
            )
            _write_json(
                os.path.join(
                    case_dir,
                    "prediction",
                    "validation_issues.json",
                ),
                issues,
            )
            if issues:
                raise ValueError(
                    "invalid final prediction: " + " | ".join(issues)
                )

            evaluation = evaluate_full_ranking(
                parsed,
                record["true_label"],
            )
            row = {
                "sample_idx": int(record["sample_idx"]),
                "bundle_id": int(record["bundle_id"]),
                "candidate_count": int(record["candidate_count"]),
                "strategy_count": int(record["strategy_count"]),
                "successful_program_count": int(
                    record["successful_program_count"]
                ),
                "evidence_context_count": int(
                    record["evidence_context_count"]
                ),
                **evaluation,
                "batch_requests": int(record.get("batch_requests", 1)),
                "sync_prediction_requests": int(
                    record["sync_prediction_requests"]
                ),
                "valid": True,
                "error": "",
            }
            record.update(
                {
                    "prediction_status": "success",
                    "error": "",
                }
            )
            _write_json(os.path.join(case_dir, "evaluation.json"), row)
            print(
                f"[{completed + 1}/{len(requests)}] test[{sample_idx}] "
                f"bundle_{record['bundle_id']} | pred={row['prediction']} "
                f"true={row['true_label']} GT-rank={row['gt_rank']}"
            )
            completed += 1
        except Exception as error:
            message = str(error)
            record.update(
                {
                    "prediction_status": "failed",
                    "error": message,
                }
            )
            _write_json(
                os.path.join(case_dir, "prediction", "error.json"),
                {"error": message},
            )
            print(
                f"[prediction {sample_idx}] bundle_{record['bundle_id']} | "
                f"error={message}"
            )
            lowered = message.lower()
            if (
                "billing_hard_limit_reached" in lowered
                or "billing hard limit" in lowered
                or "insufficient_quota" in lowered
            ):
                state["sync_prediction"].update(
                    {
                        "status": "blocked_by_billing",
                        "completed": completed,
                        "failed": sum(
                            1
                            for value in records.values()
                            if value.get("prediction_status") == "failed"
                            and value.get("code_status") == "success"
                        ),
                    }
                )
                _save_records(run_dir, records)
                _save_sync_results(run_dir, state, records)
                _save_state(run_dir, state)
                print(
                    ">>> Stopped after the first billing-limit error; "
                    "rerun predict-sync after billing is restored"
                )
                return 2

        _save_records(run_dir, records)
        state["sync_prediction"].update(
            {
                "status": "in_progress",
                "completed": completed,
                "failed": sum(
                    1
                    for value in records.values()
                    if value.get("prediction_status") == "failed"
                    and value.get("code_status") == "success"
                ),
            }
        )
        _save_sync_results(run_dir, state, records)
        _save_state(run_dir, state)

    rows, summary = _save_sync_results(run_dir, state, records)
    _write_json(os.path.join(run_dir, "summary.json"), summary)
    state["phase"] = "completed"
    state["completed_at"] = int(time.time())
    state["sync_prediction"].update(
        {
            "status": "completed",
            "completed": summary["valid_sample_count"],
            "failed": len(rows) - summary["valid_sample_count"],
        }
    )
    _save_state(run_dir, state)
    print("-" * 56)
    print(
        f">>> Valid samples: {summary['valid_sample_count']} / "
        f"{summary['requested_sample_count']}"
    )
    print(f">>> Hit@1: {summary['hit_rate_at_1']:.4f}")
    print(f">>> Hit@3: {summary['hit_rate_at_3']:.4f}")
    print(f">>> Hit@5: {summary['hit_rate_at_5']:.4f}")
    print(f">>> Mean reciprocal rank: {summary['mean_reciprocal_rank']:.4f}")
    print(f">>> Mean GT rank: {summary['mean_gt_rank']:.4f}")
    print(
        f">>> Synchronous prediction calls: "
        f"{summary['total_sync_prediction_requests']}"
    )
    print(f">>> Output: {run_dir}")
    return 0


def _submit_prediction_batch(args):
    run_dir = os.path.abspath(args.run)
    conf = _run_config(run_dir)
    state = _load_state(run_dir)
    phase = str(state.get("phase") or "")
    if phase not in {"prediction_prepared", "prediction_sync"}:
        raise RuntimeError(
            "prediction Batch submission requires prepared prediction inputs; "
            f"current phase is {phase}"
        )
    prediction_stage = _stage_batch(state, "prediction")
    if prediction_stage.get("batch_id"):
        raise RuntimeError(
            "prediction Batch was already submitted: "
            f"{prediction_stage['batch_id']}"
        )

    api_key_env = str(args.api_key_env or "").strip()
    if not api_key_env:
        raise ValueError("--api_key_env must name an environment variable")
    conf["operator_prediction_api_key_env"] = api_key_env
    _write_yaml(os.path.join(run_dir, "config_snapshot.yaml"), conf)

    state["phase"] = "prediction_prepared"
    state["prediction_delivery"] = "batch"
    state["prediction_api_key_env"] = api_key_env
    prediction_stage["status"] = "prepared"
    _save_state(run_dir, state)

    run_info_path = os.path.join(run_dir, "run.json")
    if os.path.isfile(run_info_path):
        run_info = _read_json(run_info_path)
        run_info.update(
            {
                "prediction_delivery": "batch",
                "prediction_api_key_env": api_key_env,
            }
        )
        _write_json(run_info_path, run_info)

    print(f">>> Prediction API key environment: {api_key_env}")
    print(
        f">>> Reusing prepared prediction requests: "
        f"{prediction_stage['request_count']}"
    )
    _submit_prepared_stage(
        run_dir,
        conf,
        state,
        "prediction",
    )
    return 0


def _active_stage(phase):
    for stage in ("code", "summary", "curator", "prediction"):
        if phase in {f"{stage}_prepared", f"{stage}_submitted"}:
            return stage
    return None


def _status(args):
    run_dir = os.path.abspath(args.run)
    conf = _run_config(run_dir)
    state = _load_state(run_dir)
    phase = str(state.get("phase") or "")
    stage = _active_stage(phase)
    print(f">>> Run: {run_dir}")
    print(f">>> Phase: {phase}")
    print(f">>> Pipeline: {_pipeline_mode(state)}")
    if phase == "completed":
        summary = _read_json(os.path.join(run_dir, "summary.json"))
        print(
            f">>> Complete: Hit@1={summary['hit_rate_at_1']:.4f}, "
            f"valid={summary['valid_sample_count']}/"
            f"{summary['requested_sample_count']}"
        )
        return 0
    if phase == "prediction_sync":
        progress = state.get("sync_prediction") or {}
        print(
            f">>> Synchronous prediction: "
            f"{progress.get('status', 'in_progress')}"
        )
        print(
            f">>> Requests: completed={progress.get('completed', 0)}, "
            f"failed={progress.get('failed', 0)}, "
            f"total={progress.get('total', 0)}"
        )
        return 0
    if stage is None:
        if state.get("error"):
            print(f">>> Error: {state['error']}")
        return 0
    stage_state = _stage_batch(state, stage)
    if phase.endswith("_prepared"):
        print(f">>> {stage} Batch is prepared but not submitted")
        print(
            f">>> Input: {os.path.join(run_dir, stage_state['input_path'])}"
        )
        return 0
    _, stage_state = _refresh_stage(run_dir, conf, state, stage)
    batch = stage_state["batch"]
    counts = batch.get("request_counts") or {}
    print(f">>> Stage: {stage}")
    print(f">>> Batch ID: {stage_state['batch_id']}")
    print(f">>> Status: {stage_state['status']}")
    print(
        f">>> Requests: completed={counts.get('completed', 0)}, "
        f"failed={counts.get('failed', 0)}, total={counts.get('total', 0)}"
    )
    return 0


def _advance(args):
    run_dir = os.path.abspath(args.run)
    conf = _run_config(run_dir)
    state = _load_state(run_dir)
    phase = str(state.get("phase") or "")
    if args.skip_summary:
        if phase not in {"code_prepared", "code_submitted"}:
            raise RuntimeError(
                "--skip_summary can only be selected while the code stage is active"
            )
        state["pipeline_mode"] = PIPELINE_CODE_PREDICTION
        run_info_path = os.path.join(run_dir, "run.json")
        if os.path.isfile(run_info_path):
            run_info = _read_json(run_info_path)
            run_info.update(
                {
                    "phase": "openai_batch_spec_first_code_prediction",
                    "pipeline_mode": PIPELINE_CODE_PREDICTION,
                    "batch_stages": ["code", "prediction"],
                    "summary_model": None,
                }
            )
            _write_json(run_info_path, run_info)
        _save_state(run_dir, state)
        print(">>> Pipeline switched to code_prediction; summary will be skipped")
    if phase == "completed":
        print(f">>> Run already complete: {run_dir}")
        return 0
    stage = _active_stage(phase)
    if stage is None:
        raise RuntimeError(
            f"run cannot advance from phase {phase}: "
            f"{state.get('error', '')}"
        )
    if phase.endswith("_prepared"):
        _submit_prepared_stage(run_dir, conf, state, stage)
        return 0

    client, stage_state = _refresh_stage(
        run_dir,
        conf,
        state,
        stage,
    )
    status = stage_state["status"]
    counts = stage_state.get("batch", {}).get("request_counts") or {}
    print(
        f">>> {stage} Batch {stage_state['batch_id']}: {status} "
        f"({counts.get('completed', 0)}/{counts.get('total', 0)} completed)"
    )
    if status not in BATCH_TERMINAL_STATUSES:
        print(">>> Not terminal yet; run status or advance again later")
        return 0
    if stage == "code":
        _prepare_after_code_stage(
            run_dir,
            conf,
            state,
            client,
            stage_state,
        )
        return 0
    if stage == "summary":
        _prepare_prediction_stage(
            run_dir,
            conf,
            state,
            client,
            stage_state,
        )
        return 0
    if stage == "curator":
        _prepare_after_curator_stage(
            run_dir,
            conf,
            state,
            client,
            stage_state,
        )
        return 0
    return _finalize_predictions(
        run_dir,
        conf,
        state,
        client,
        stage_state,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "OpenAI Batch orchestrator for generated strategy code and final "
            "ranking, with optional evidence-summary or evidence-curator stages"
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    start = subparsers.add_parser(
        "start",
        help="Prepare and submit the code-generation Batch",
    )
    start.add_argument("--config", default="config_operator.yaml")
    start.add_argument(
        "--dataset",
        choices=["pog", "pog_dense", "spotify", "spotify_sparse"],
        default="",
        help="Override the dataset in the config for this run",
    )
    start.add_argument(
        "--api_key_env",
        default="",
        help=(
            "Override code, summary, curator, and prediction API-key environment "
            "variable names for this run; never pass the key value"
        ),
    )
    start.add_argument("--split", default="test")
    start.add_argument("--start_idx", type=int, default=0)
    start.add_argument("--sample_count", type=int, default=250)
    start.add_argument("--sample_idx", type=int, default=None)
    start.add_argument("--output_dir", default="")
    start.add_argument(
        "--dry_run",
        action="store_true",
        help="Prepare the code Batch JSONL without uploading or submitting it",
    )
    start.add_argument(
        "--skip_summary",
        action="store_true",
        help=(
            "Use code -> local execution -> prediction instead of inserting "
            "a summary Batch"
        ),
    )
    start.add_argument(
        "--with_curator",
        action="store_true",
        help=(
            "Use code -> local execution -> curator Batch -> prediction Batch"
        ),
    )

    start_summary = subparsers.add_parser(
        "start-summary-from-run",
        help=(
            "Reuse completed code evidence and start a separate summary -> "
            "prediction Batch run"
        ),
    )
    start_summary.add_argument("--source_run", required=True)
    start_summary.add_argument("--output_dir", default="")
    start_summary.add_argument(
        "--api_key_env",
        required=True,
        help="Environment variable name loaded from .env, never the key value",
    )
    start_summary.add_argument(
        "--dry_run",
        action="store_true",
        help="Prepare summary Batch JSONL without uploading or submitting it",
    )

    status = subparsers.add_parser(
        "status",
        help="Refresh and print the active Batch status",
    )
    status.add_argument("--run", required=True)

    advance = subparsers.add_parser(
        "advance",
        help="Submit a prepared stage or process a completed stage",
    )
    advance.add_argument("--run", required=True)
    advance.add_argument(
        "--skip_summary",
        action="store_true",
        help=(
            "Persistently switch an active code stage to direct raw-evidence "
            "prediction"
        ),
    )

    predict_sync = subparsers.add_parser(
        "predict-sync",
        help=(
            "Call prepared prediction requests sequentially without submitting "
            "a prediction Batch"
        ),
    )
    predict_sync.add_argument("--run", required=True)

    submit_prediction_batch = subparsers.add_parser(
        "submit-prediction-batch",
        help=(
            "Submit already-prepared prediction requests as a Batch using a "
            "selected API-key environment variable"
        ),
    )
    submit_prediction_batch.add_argument("--run", required=True)
    submit_prediction_batch.add_argument(
        "--api_key_env",
        required=True,
        help="Environment variable name loaded from .env, never the key value",
    )

    args = parser.parse_args()
    if args.command == "start":
        return _initialize(args)
    if args.command == "start-summary-from-run":
        return _initialize_summary_from_run(args)
    if args.command == "status":
        return _status(args)
    if args.command == "predict-sync":
        return _predict_sync(args)
    if args.command == "submit-prediction-batch":
        return _submit_prediction_batch(args)
    return _advance(args)


if __name__ == "__main__":
    raise SystemExit(main())

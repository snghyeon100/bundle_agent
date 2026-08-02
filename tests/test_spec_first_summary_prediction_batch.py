"""Re-predict from saved strategy executions through candidate-local summaries.

This test does not regenerate strategies or execute Python programs. For every
reusable sample in a prior spec-first run it makes exactly two LLM calls:
1. summarize the saved strategy evidence separately for every candidate;
2. rank candidates with each summary placed beside its candidate item.

Example:
    python tests/test_spec_first_summary_prediction_batch.py \
        --config config_operator.yaml \
        --source_run tests/outputs/spec_first_operator_batch/pog_dense_20260730_114851
"""

import argparse
import asyncio
import csv
import json
import os
import sys
import time

import yaml


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from code.common import parse_json_from_text
from main import (
    QuotaExceededError,
    create_llm_client,
    default_api_key_envs_for_provider,
    generate_content_with_retry,
    resolve_api_key_from_keys,
    stage_model,
    stage_provider,
)
from online_hypothesis_program.schemas import validate_prediction_result
from operator_learning.evidence_summary import (
    candidate_evidence_summary_prompt,
    candidate_summary_prediction_prompt,
    validate_candidate_summaries,
)
from operator_learning.spec_first_prediction import (
    aggregate_prediction_rows,
    evaluate_full_ranking,
)


RESULT_FIELDS = [
    "sample_idx",
    "bundle_id",
    "candidate_count",
    "source_strategy_count",
    "source_successful_program_count",
    "source_evidence_context_count",
    "source_gt_rank",
    "prediction",
    "ranking",
    "true_label",
    "hit",
    "gt_rank",
    "gt_rank_improvement",
    "reciprocal_rank",
    "hit_at_1",
    "hit_at_3",
    "hit_at_5",
    "llm_calls",
    "valid",
    "error",
]


def _read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=str)
    os.replace(temporary, path)


def _write_text(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(str(value or ""))


def _write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
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


def _model_client(conf, role):
    prediction_provider = stage_provider(conf, "prediction")
    prediction_model = stage_model(conf, "prediction")
    if role == "summary":
        provider = str(
            conf.get("operator_summary_provider") or prediction_provider
        ).strip().lower()
        model = str(
            conf.get("operator_summary_model") or prediction_model
        ).strip()
        key_names = [
            "operator_summary_api_key_env",
            "operator_prediction_api_key_env",
            "online_prediction_api_key_env",
            "code_prediction_api_key_env",
            "code_generation_api_key_env",
            "code_api_key_env",
        ]
    else:
        provider = prediction_provider
        model = prediction_model
        key_names = [
            "operator_prediction_api_key_env",
            "online_prediction_api_key_env",
            "code_prediction_api_key_env",
            "code_generation_api_key_env",
            "code_api_key_env",
        ]
    api_key, env = resolve_api_key_from_keys(
        conf,
        key_names,
        default_api_key_envs_for_provider(provider),
    )
    return {
        "client": create_llm_client(provider, api_key),
        "provider": provider,
        "model": model,
    }, {
        "provider": provider,
        "model": model,
        "api_key_env": env,
    }


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


def _select_rows(args, rows):
    if args.sample_idx is not None:
        selected = [
            row for row in rows if int(row["sample_idx"]) == int(args.sample_idx)
        ]
        if not selected:
            raise IndexError(f"sample_idx {args.sample_idx} not found in source run")
        return selected
    start = max(0, int(args.start_idx))
    eligible = [row for row in rows if int(row["sample_idx"]) >= start]
    if args.sample_count is None:
        return eligible
    count = int(args.sample_count)
    if count <= 0:
        raise ValueError("--sample_count must be positive")
    return eligible[:count]


def _source_case_dir(source_run, row):
    return os.path.join(
        source_run,
        "samples",
        f"{int(row['sample_idx']):04d}_bundle_{int(row['bundle_id'])}",
    )


def _output_case_dir(output_dir, row):
    return os.path.join(
        output_dir,
        "samples",
        f"{int(row['sample_idx']):04d}_bundle_{int(row['bundle_id'])}",
    )


def _resolve_output_dir(args, dataset):
    if args.resume:
        path = os.path.abspath(args.resume)
        if not os.path.exists(path):
            raise FileNotFoundError(f"resume path does not exist: {path}")
        return path if os.path.isdir(path) else os.path.dirname(path)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return os.path.abspath(
        args.output_dir
        or os.path.join(
            ROOT,
            "tests",
            "outputs",
            "spec_first_summary_prediction",
            f"{dataset}_{stamp}",
        )
    )


def _load_previous_rows(args, output_dir):
    if not args.resume:
        return []
    path = os.path.abspath(args.resume)
    candidates = (
        [path]
        if os.path.isfile(path)
        else [
            os.path.join(output_dir, "results_partial.json"),
            os.path.join(output_dir, "results.json"),
        ]
    )
    for candidate in candidates:
        if os.path.isfile(candidate) and candidate.lower().endswith(".json"):
            value = _read_json(candidate)
            if not isinstance(value, list):
                raise ValueError(f"resume JSON must contain a list: {candidate}")
            return [row for row in value if isinstance(row, dict)]
    raise FileNotFoundError("resume results JSON was not found")


def _error_row(source_row, error, llm_calls=0):
    return {
        "sample_idx": int(source_row["sample_idx"]),
        "bundle_id": int(source_row["bundle_id"]),
        "candidate_count": int(source_row.get("candidate_count", 0)),
        "source_strategy_count": int(source_row.get("strategy_count", 0)),
        "source_successful_program_count": int(
            source_row.get("successful_program_count", 0)
        ),
        "source_evidence_context_count": int(
            source_row.get("evidence_context_count", 0)
        ),
        "source_gt_rank": source_row.get("gt_rank"),
        "prediction": None,
        "ranking": [],
        "true_label": str(source_row.get("true_label") or ""),
        "hit": False,
        "gt_rank": None,
        "gt_rank_improvement": None,
        "reciprocal_rank": 0.0,
        "hit_at_1": False,
        "hit_at_3": False,
        "hit_at_5": False,
        "llm_calls": int(llm_calls),
        "valid": False,
        "error": str(error),
    }


def _save_state(output_dir, rows, requested_count, partial):
    ordered = sorted(rows, key=lambda row: int(row["sample_idx"]))
    stem = "results_partial" if partial else "results"
    _write_json(os.path.join(output_dir, f"{stem}.json"), ordered)
    _write_csv(os.path.join(output_dir, f"{stem}.csv"), ordered)
    summary = {
        "requested_sample_count": int(requested_count),
        **aggregate_prediction_rows(ordered),
        "mean_source_gt_rank": (
            sum(
                float(row["source_gt_rank"])
                for row in ordered
                if row.get("valid") and row.get("source_gt_rank") is not None
            )
            / sum(
                1
                for row in ordered
                if row.get("valid") and row.get("source_gt_rank") is not None
            )
            if any(
                row.get("valid") and row.get("source_gt_rank") is not None
                for row in ordered
            )
            else 0.0
        ),
        "mean_gt_rank_improvement": (
            sum(
                float(row["gt_rank_improvement"])
                for row in ordered
                if row.get("valid")
                and row.get("gt_rank_improvement") is not None
            )
            / sum(
                1
                for row in ordered
                if row.get("valid")
                and row.get("gt_rank_improvement") is not None
            )
            if any(
                row.get("valid")
                and row.get("gt_rank_improvement") is not None
                for row in ordered
            )
            else 0.0
        ),
        "total_llm_calls": sum(int(row.get("llm_calls", 0)) for row in ordered),
    }
    name = "summary_partial.json" if partial else "summary.json"
    _write_json(os.path.join(output_dir, name), summary)
    return summary


async def _run(args):
    with open(args.config, "r", encoding="utf-8") as handle:
        conf = yaml.safe_load(handle)

    source_run = os.path.abspath(args.source_run)
    if not os.path.isdir(source_run):
        raise FileNotFoundError(f"source run does not exist: {source_run}")
    source_meta = _read_json(os.path.join(source_run, "run.json"))
    dataset = str(source_meta.get("dataset") or conf.get("dataset") or "dataset")
    split = str(source_meta.get("split") or "test")
    selected = _select_rows(args, _source_results(source_run))
    selected_indices = {int(row["sample_idx"]) for row in selected}

    output_dir = _resolve_output_dir(args, dataset)
    os.makedirs(output_dir, exist_ok=True)
    previous = _load_previous_rows(args, output_dir)
    rows_by_index = {
        int(row["sample_idx"]): row
        for row in previous
        if int(row.get("sample_idx", -1)) in selected_indices
    }
    pending = [
        row for row in selected if int(row["sample_idx"]) not in rows_by_index
    ]

    summary_client, summary_model = _model_client(conf, "summary")
    prediction_client, prediction_model = _model_client(conf, "prediction")
    concurrency = max(
        1,
        int(
            args.max_concurrent
            if args.max_concurrent is not None
            else conf.get("operator_prediction_max_concurrent", 1)
        ),
    )
    semaphore = asyncio.Semaphore(concurrency)

    _write_json(
        os.path.join(output_dir, "run.json"),
        {
            "phase": "candidate_local_summary_prediction",
            "dataset": dataset,
            "split": split,
            "source_run": source_run,
            "requested_sample_count": len(selected),
            "llm_calls_per_reusable_sample": 2,
            "strategy_generation": "reused",
            "program_execution": "reused",
            "prediction_input": "candidate text with candidate-local summary",
            "max_concurrent_samples": concurrency,
            "resumed": bool(args.resume),
            "summary_model": summary_model,
            "prediction_model": prediction_model,
        },
    )

    print(f">>> Dataset: {dataset}")
    print(f">>> Source run: {source_run}")
    print(
        f">>> Samples: {len(selected)} "
        f"({len(rows_by_index)} resumed, {len(pending)} pending)"
    )
    print(">>> Strategy generation and program execution: reused, not rerun")
    print(
        f">>> Summary: {summary_model['provider']} / "
        f"{summary_model['model']} (10 candidate-local summaries in one call)"
    )
    print(
        f">>> Prediction: {prediction_model['provider']} / "
        f"{prediction_model['model']} (candidate text + adjacent summary)"
    )
    print(f">>> Maximum concurrent samples: {concurrency}")

    async def call_model(client, model, prompt, max_tokens, step_name):
        async with semaphore:
            return await generate_content_with_retry(
                client,
                model["model"],
                prompt,
                conf,
                int(max_tokens),
                step_name,
            )

    progress = len(rows_by_index)

    async def run_one(source_row):
        nonlocal progress
        case_dir = _output_case_dir(output_dir, source_row)
        os.makedirs(case_dir, exist_ok=True)
        llm_calls = 0
        try:
            if not source_row.get("valid"):
                raise ValueError(
                    "source sample is invalid and has no reusable prediction evidence"
                )
            source_case_dir = _source_case_dir(source_run, source_row)
            case = _read_json(os.path.join(source_case_dir, "case.json"))
            strategy_evidence = _read_json(
                os.path.join(source_case_dir, "rendered_strategy_evidence.json")
            )
            partial_items = case.get("partial_items", [])
            candidate_items = case.get("candidate_items", [])
            labels = [
                str(candidate.get("label") or "")
                for candidate in candidate_items
                if isinstance(candidate, dict)
            ]
            if not labels:
                raise ValueError("source case contains no candidate labels")

            summary_prompt = candidate_evidence_summary_prompt(
                partial_items=partial_items,
                candidate_items=candidate_items,
                strategy_evidence=strategy_evidence,
            )
            _write_text(
                os.path.join(case_dir, "summary", "input.txt"),
                summary_prompt,
            )
            summary_raw = await call_model(
                summary_client,
                summary_model,
                summary_prompt,
                conf.get("operator_summary_max_output_tokens", 5000),
                f"candidate evidence summary for bundle_{source_row['bundle_id']}",
            )
            llm_calls += 1
            summary_parsed = parse_json_from_text(summary_raw)
            summary_issues = validate_candidate_summaries(
                summary_parsed,
                labels,
            )
            _write_text(
                os.path.join(case_dir, "summary", "output.txt"),
                summary_raw,
            )
            _write_json(
                os.path.join(case_dir, "summary", "parsed_response.json"),
                summary_parsed,
            )
            _write_json(
                os.path.join(case_dir, "summary", "validation_issues.json"),
                summary_issues,
            )
            if summary_issues:
                raise ValueError(
                    "invalid candidate summaries: " + " | ".join(summary_issues)
                )

            candidate_summaries = summary_parsed["candidate_summaries"]
            _write_json(
                os.path.join(case_dir, "candidate_summaries.json"),
                candidate_summaries,
            )
            prediction_prompt = candidate_summary_prediction_prompt(
                dataset=dataset,
                partial_items=partial_items,
                candidate_items=candidate_items,
                candidate_summaries=candidate_summaries,
            )
            _write_text(
                os.path.join(case_dir, "prediction", "input.txt"),
                prediction_prompt,
            )
            prediction_raw = await call_model(
                prediction_client,
                prediction_model,
                prediction_prompt,
                conf.get("operator_prediction_max_output_tokens", 2000),
                f"summary-grounded prediction for bundle_{source_row['bundle_id']}",
            )
            llm_calls += 1
            prediction_parsed = parse_json_from_text(prediction_raw)
            prediction_issues = validate_prediction_result(
                prediction_parsed,
                labels,
            )
            _write_text(
                os.path.join(case_dir, "prediction", "output.txt"),
                prediction_raw,
            )
            _write_json(
                os.path.join(case_dir, "prediction", "parsed_response.json"),
                prediction_parsed,
            )
            _write_json(
                os.path.join(case_dir, "prediction", "validation_issues.json"),
                prediction_issues,
            )
            if prediction_issues:
                raise ValueError(
                    "invalid final prediction: " + " | ".join(prediction_issues)
                )

            evaluation = evaluate_full_ranking(
                prediction_parsed,
                str(source_row["true_label"]),
            )
            source_gt_rank = source_row.get("gt_rank")
            row = {
                "sample_idx": int(source_row["sample_idx"]),
                "bundle_id": int(source_row["bundle_id"]),
                "candidate_count": len(labels),
                "source_strategy_count": int(
                    source_row.get("strategy_count", 0)
                ),
                "source_successful_program_count": int(
                    source_row.get("successful_program_count", 0)
                ),
                "source_evidence_context_count": int(
                    source_row.get("evidence_context_count", 0)
                ),
                "source_gt_rank": source_gt_rank,
                **evaluation,
                "gt_rank_improvement": (
                    int(source_gt_rank) - int(evaluation["gt_rank"])
                    if source_gt_rank is not None
                    else None
                ),
                "llm_calls": llm_calls,
                "valid": True,
                "error": "",
            }
            _write_json(os.path.join(case_dir, "evaluation.json"), row)
        except QuotaExceededError:
            raise
        except Exception as error:
            _write_json(
                os.path.join(case_dir, "error.json"),
                {
                    "sample_idx": int(source_row["sample_idx"]),
                    "bundle_id": int(source_row["bundle_id"]),
                    "llm_calls": llm_calls,
                    "error": str(error),
                },
            )
            row = _error_row(source_row, error, llm_calls)

        progress += 1
        status = (
            f"error={row['error']}"
            if row["error"]
            else (
                f"pred={row['prediction']} true={row['true_label']} "
                f"GT-rank={row['gt_rank']} raw-rank={row['source_gt_rank']}"
            )
        )
        print(
            f"[{progress}/{len(selected)}] "
            f"{split}[{row['sample_idx']}] bundle_{row['bundle_id']} | {status}"
        )
        return row

    rows = list(rows_by_index.values())
    try:
        for offset in range(0, len(pending), concurrency):
            chunk = pending[offset : offset + concurrency]
            completed = await asyncio.gather(*(run_one(row) for row in chunk))
            rows.extend(completed)
            _save_state(output_dir, rows, len(selected), partial=True)
    except QuotaExceededError as error:
        _save_state(output_dir, rows, len(selected), partial=True)
        print(f"[Stopped] {error}")
        print(
            "[Resume] python tests/test_spec_first_summary_prediction_batch.py "
            f'--config "{args.config}" --source_run "{source_run}" '
            f'--resume "{output_dir}"'
        )
        return 1

    summary = _save_state(
        output_dir,
        rows,
        len(selected),
        partial=False,
    )
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
    print(f">>> Source mean GT rank: {summary['mean_source_gt_rank']:.4f}")
    print(
        ">>> Mean GT-rank improvement over raw evidence: "
        f"{summary['mean_gt_rank_improvement']:.4f}"
    )
    print(f">>> Total new LLM calls: {summary['total_llm_calls']}")
    print(f">>> Output: {output_dir}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Reuse saved strategy executions, summarize evidence per candidate, "
            "and rerun full-ranking prediction"
        )
    )
    parser.add_argument("--config", default="config_operator.yaml")
    parser.add_argument("--source_run", required=True)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--sample_count", type=int, default=None)
    parser.add_argument("--sample_idx", type=int, default=None)
    parser.add_argument("--max_concurrent", type=int, default=None)
    parser.add_argument("--output_dir", default="")
    parser.add_argument(
        "--resume",
        default="",
        help="Existing output directory or results_partial.json",
    )
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())

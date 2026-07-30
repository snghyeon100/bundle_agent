"""End-to-end batch evaluation for spec-first generated strategies.

Each sample uses exactly two LLM calls:
1. generate three immutable intent/strategy specs and their Python programs;
2. rank every answer option from the executed candidate-specific contexts.

The default test[0:250] selection matches the direct plausible-set baseline.

Usage:
    python tests/test_spec_first_operator_batch.py \
        --config config_operator.yaml \
        --split test \
        --start_idx 0 \
        --sample_count 250
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
from dataset import BundleZeroShotDataset, set_seed
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
from operator_learning.pipeline import (
    _build_diagnostic_indices,
    _capability_names,
    build_discovery_case,
    build_operator_capability_manifest,
    enrich_case_source_diagnostics,
)
from operator_learning.prompts import (
    induction_prompt,
    strategy_evidence_prediction_prompt,
)
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
    "llm_calls",
    "valid",
    "error",
]


def _write_json(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=str)
    os.replace(temporary, path)


def _read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


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


def _build_client(conf, role):
    provider = stage_provider(conf, role)
    model = stage_model(conf, role)
    api_key, env = resolve_api_key_from_keys(
        conf,
        [
            (
                "operator_api_key_env"
                if role == "code_generation"
                else "operator_prediction_api_key_env"
            ),
            "online_prediction_api_key_env",
            "code_prediction_api_key_env",
            "code_generation_api_key_env",
            "code_api_key_env",
        ],
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


def _resolve_output_dir(args, conf):
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
            "spec_first_operator_batch",
            f"{conf['dataset']}_{stamp}",
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


def _sample_dir(output_dir, sample_idx, sample):
    return os.path.join(
        output_dir,
        "samples",
        f"{sample_idx:04d}_bundle_{int(sample['bundle_id'])}",
    )


def _error_row(sample_idx, sample, error, llm_calls=0):
    return {
        "sample_idx": int(sample_idx),
        "bundle_id": int(sample["bundle_id"]),
        "candidate_count": len(sample.get("candidate_indices", [])),
        "strategy_count": 0,
        "successful_program_count": 0,
        "evidence_context_count": 0,
        "prediction": None,
        "ranking": [],
        "true_label": str(sample["true_option_char"]),
        "hit": False,
        "gt_rank": None,
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
        "mean_successful_program_count": (
            sum(int(row.get("successful_program_count", 0)) for row in ordered)
            / len(ordered)
            if ordered
            else 0.0
        ),
        "total_llm_calls": sum(int(row.get("llm_calls", 0)) for row in ordered),
    }
    summary_stem = "summary_partial" if partial else "summary"
    _write_json(os.path.join(output_dir, f"{summary_stem}.json"), summary)
    return summary


async def _run(args):
    with open(args.config, "r", encoding="utf-8") as handle:
        conf = yaml.safe_load(handle)
    set_seed(int(conf.get("seed", 45)))
    split = str(args.split).strip().lower()
    if split not in {"valid", "test"}:
        raise ValueError("--split must be valid or test")

    eval_conf = dict(conf)
    eval_conf["toy_eval"] = -1
    samples = BundleZeroShotDataset(eval_conf, split=split).get_eval_samples()
    indexed_samples = _select_samples(args, conf, samples)
    desired_indices = {index for index, _ in indexed_samples}

    output_dir = _resolve_output_dir(args, conf)
    os.makedirs(output_dir, exist_ok=True)
    previous = _load_previous_rows(args, output_dir)
    rows_by_index = {
        int(row["sample_idx"]): row
        for row in previous
        if "sample_idx" in row and int(row["sample_idx"]) in desired_indices
    }
    pending = [
        (index, sample)
        for index, sample in indexed_samples
        if index not in rows_by_index
    ]

    workspace, source_manifest, capabilities = build_operator_capability_manifest(
        conf
    )
    allowed_sources = _capability_names(capabilities)
    diagnostic_indices = _build_diagnostic_indices(conf, allowed_sources)
    all_source_paths = source_paths_from_capabilities(workspace, capabilities)
    generation_client, generation_model = _build_client(conf, "code_generation")
    prediction_client, prediction_model = _build_client(conf, "prediction")

    async def call_generation(prompt, step_name):
        return await generate_content_with_retry(
            generation_client,
            generation_model["model"],
            prompt,
            conf,
            int(conf.get("operator_max_output_tokens", 15000)),
            step_name,
        )

    async def call_prediction(prompt, step_name):
        return await generate_content_with_retry(
            prediction_client,
            prediction_model["model"],
            prompt,
            conf,
            int(conf.get("operator_prediction_max_output_tokens", 2000)),
            step_name,
        )

    concurrency = max(
        1,
        int(
            args.max_concurrent
            if args.max_concurrent is not None
            else conf.get("operator_prediction_max_concurrent", 1)
        ),
    )
    manifest = [
        {
            "sample_idx": int(index),
            "bundle_id": int(sample["bundle_id"]),
            "candidate_item_ids": [
                int(item_id) for item_id in sample.get("candidate_indices", [])
            ],
        }
        for index, sample in indexed_samples
    ]
    manifest_path = os.path.join(output_dir, "sample_manifest.json")
    if args.resume and os.path.isfile(manifest_path):
        previous_manifest = _read_json(manifest_path)
        if previous_manifest != manifest:
            raise ValueError(
                "resume sample manifest does not match the requested split/range "
                "or candidate ordering"
            )
    _write_json(manifest_path, manifest)
    _write_json(os.path.join(output_dir, "source_manifest.json"), source_manifest)
    _write_json(
        os.path.join(output_dir, "run.json"),
        {
            "phase": "spec_first_strategy_code_prediction_batch",
            "dataset": conf["dataset"],
            "split": split,
            "start_idx": indexed_samples[0][0] if indexed_samples else None,
            "requested_sample_count": len(indexed_samples),
            "selection": "contiguous dataset order; identical to direct plausible-set baseline",
            "seed": int(conf.get("seed", 45)),
            "shuffle_seed": int(conf.get("shuffle_seed", 41)),
            "strategies_per_sample": 3,
            "llm_calls_per_sample": 2,
            "max_concurrent_samples": concurrency,
            "context_rendering": "all generated sources/text contexts unchanged",
            "resumed": bool(args.resume),
            "generation_model": generation_model,
            "prediction_model": prediction_model,
        },
    )

    print(f">>> Dataset: {conf['dataset']}")
    print(f">>> Split: {split}")
    print(
        f">>> Samples: {len(indexed_samples)} "
        f"({len(rows_by_index)} resumed, {len(pending)} pending)"
    )
    print(
        f">>> LLM1: {generation_model['provider']} / "
        f"{generation_model['model']} (3 intent/spec/code programs)"
    )
    print(">>> Runtime: guarded subprocess execution of three programs")
    print(
        f">>> LLM2: {prediction_model['provider']} / "
        f"{prediction_model['model']} (evidence-grounded full ranking)"
    )
    print(f">>> Maximum concurrent samples: {concurrency}")

    semaphore = asyncio.Semaphore(concurrency)
    progress = len(rows_by_index)

    async def run_one(sample_idx, sample):
        nonlocal progress
        case_dir = _sample_dir(output_dir, sample_idx, sample)
        os.makedirs(case_dir, exist_ok=True)
        llm_calls = 0
        try:
            case = build_discovery_case(sample, conf)
            enrich_case_source_diagnostics(
                case,
                capabilities,
                diagnostic_indices,
            )
            prompt1 = induction_prompt(
                case,
                capabilities,
                [],
                3,
                text_only=False,
            )
            _write_json(
                os.path.join(case_dir, "case.json"),
                {
                    "dataset": case["dataset"],
                    "partial_items": case["partial_items"],
                    "candidate_items": case["candidate_items"],
                    "source_diagnostics": case["source_diagnostics"],
                },
            )
            _write_text(os.path.join(case_dir, "llm1", "input.txt"), prompt1)
            async with semaphore:
                raw1 = await call_generation(
                    prompt1,
                    f"spec-first generation for {case['case_id']}",
                )
            llm_calls += 1
            parsed1 = parse_json_from_text(raw1)
            induction_issues = validate_induction_result(
                parsed1,
                expected_count=3,
                allowed_source_names=allowed_sources,
            )
            _write_text(os.path.join(case_dir, "llm1", "output.txt"), raw1)
            _write_json(
                os.path.join(case_dir, "llm1", "parsed_response.json"),
                parsed1,
            )
            _write_json(
                os.path.join(case_dir, "llm1", "validation_issues.json"),
                induction_issues,
            )
            if induction_issues:
                raise ValueError(
                    "invalid spec-first generation: " + " | ".join(induction_issues)
                )
            strategies = resolve_induction_strategies(parsed1)
            execution_reports = []
            for strategy in strategies:
                report = await asyncio.to_thread(
                    execute_strategy_program,
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
                report for report in execution_reports if report.get("success")
            ]
            if not successful:
                raise RuntimeError("all three generated programs failed")

            labels = [
                str(candidate["label"]) for candidate in case["candidate_items"]
            ]
            strategy_evidence = build_strategy_evidence(
                specs=parsed1["strategy_specs"],
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
            prompt2 = strategy_evidence_prediction_prompt(
                dataset=case["dataset"],
                partial_items=case["partial_items"],
                candidate_items=case["candidate_items"],
                strategy_evidence=strategy_evidence,
            )
            _write_text(os.path.join(case_dir, "llm2", "input.txt"), prompt2)
            async with semaphore:
                raw2 = await call_prediction(
                    prompt2,
                    f"spec-first prediction for {case['case_id']}",
                )
            llm_calls += 1
            parsed2 = parse_json_from_text(raw2)
            prediction_issues = validate_prediction_result(parsed2, labels)
            _write_text(os.path.join(case_dir, "llm2", "output.txt"), raw2)
            _write_json(
                os.path.join(case_dir, "llm2", "parsed_response.json"),
                parsed2,
            )
            _write_json(
                os.path.join(case_dir, "llm2", "validation_issues.json"),
                prediction_issues,
            )
            if prediction_issues:
                raise ValueError(
                    "invalid final prediction: " + " | ".join(prediction_issues)
                )

            evaluation = evaluate_full_ranking(
                parsed2,
                str(sample["true_option_char"]),
            )
            context_count = sum(
                len(candidate.get("contexts", []))
                for evidence in strategy_evidence
                for candidate in evidence.get("candidate_evidence", [])
            )
            row = {
                "sample_idx": int(sample_idx),
                "bundle_id": int(sample["bundle_id"]),
                "candidate_count": len(labels),
                "strategy_count": len(strategies),
                "successful_program_count": len(successful),
                "evidence_context_count": context_count,
                **evaluation,
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
                    "sample_idx": sample_idx,
                    "bundle_id": int(sample["bundle_id"]),
                    "llm_calls": llm_calls,
                    "error": str(error),
                },
            )
            row = _error_row(sample_idx, sample, error, llm_calls)

        progress += 1
        status = (
            f"error={row['error']}"
            if row["error"]
            else (
                f"pred={row['prediction']} true={row['true_label']} "
                f"GT-rank={row['gt_rank']} programs="
                f"{row['successful_program_count']}/3"
            )
        )
        print(
            f"[{progress}/{len(indexed_samples)}] "
            f"{split}[{sample_idx}] bundle_{sample['bundle_id']} | {status}"
        )
        return row

    rows = list(rows_by_index.values())
    try:
        for offset in range(0, len(pending), concurrency):
            chunk = pending[offset : offset + concurrency]
            completed = await asyncio.gather(
                *(run_one(index, sample) for index, sample in chunk)
            )
            rows.extend(completed)
            _save_state(output_dir, rows, len(indexed_samples), partial=True)
    except QuotaExceededError as error:
        _save_state(output_dir, rows, len(indexed_samples), partial=True)
        print(f"[Stopped] {error}")
        print(
            "[Resume] python tests/test_spec_first_operator_batch.py "
            f'--resume "{output_dir}" --split {split} '
            f"--start_idx {indexed_samples[0][0]} "
            f"--sample_count {len(indexed_samples)}"
        )
        return 1

    summary = _save_state(
        output_dir,
        rows,
        len(indexed_samples),
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
    print(
        ">>> Mean successful programs: "
        f"{summary['mean_successful_program_count']:.4f} / 3"
    )
    print(f">>> Total LLM calls: {summary['total_llm_calls']}")
    print(f">>> Output: {output_dir}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Two-call spec-first strategy/code/prediction batch evaluation"
    )
    parser.add_argument("--config", default="config_operator.yaml")
    parser.add_argument("--split", default="test")
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

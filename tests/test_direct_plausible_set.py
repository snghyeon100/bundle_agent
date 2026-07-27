"""Run direct plausible-set selection over a batch of evaluation samples.

Usage:
    python tests/test_direct_plausible_set.py \
        --config config_operator.yaml \
        --split test \
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
from plausible_candidate_selection.pipeline import (
    aggregate_plausible_set_evaluations,
    run_direct_plausible_set,
)


RESULT_FIELDS = [
    "sample_idx",
    "bundle_id",
    "candidate_count",
    "plausible_labels",
    "ranking",
    "plausible_rank_top_k",
    "plausible_ranking_consistent",
    "plausible_set_size",
    "selection_fraction",
    "random_same_size_gt_inclusion_baseline",
    "true_label",
    "gt_in_plausible_set",
    "gt_rank",
    "reciprocal_rank",
    "hit_at_1",
    "hit_at_3",
    "hit_at_5",
    "llm_calls",
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
            serialized = {key: row.get(key, "") for key in RESULT_FIELDS}
            for field in (
                "plausible_labels",
                "ranking",
                "plausible_rank_top_k",
            ):
                if isinstance(serialized[field], (list, dict)):
                    serialized[field] = json.dumps(
                        serialized[field],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
            writer.writerow(serialized)
    os.replace(temporary, path)


def _build_client(conf):
    provider = stage_provider(conf, "prediction")
    model = stage_model(conf, "prediction")
    api_key, env = resolve_api_key_from_keys(
        conf,
        [
            "plausible_set_api_key_env",
            "online_prediction_api_key_env",
            "code_prediction_api_key_env",
            "operator_api_key_env",
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


def _resolve_output_dir(args, conf):
    if args.resume:
        resume_path = os.path.abspath(args.resume)
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"resume path does not exist: {resume_path}")
        return resume_path if os.path.isdir(resume_path) else os.path.dirname(resume_path)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return os.path.abspath(
        args.output_dir
        or os.path.join(
            ROOT,
            "tests",
            "outputs",
            "direct_plausible_set",
            f"{conf['dataset']}_{timestamp}",
        )
    )


def _load_previous_rows(args, output_dir):
    if not args.resume:
        return []
    resume_path = os.path.abspath(args.resume)
    candidates = (
        [resume_path]
        if os.path.isfile(resume_path)
        else [
            os.path.join(output_dir, "results_partial.json"),
            os.path.join(output_dir, "results.json"),
        ]
    )
    for path in candidates:
        if os.path.isfile(path) and path.lower().endswith(".json"):
            loaded = _read_json(path)
            if not isinstance(loaded, list):
                raise ValueError(f"resume JSON must contain a result list: {path}")
            return [row for row in loaded if isinstance(row, dict)]
    raise FileNotFoundError(
        "resume requires results_partial.json, results.json, or a JSON result file"
    )


def _sample_output_dir(output_dir, sample_idx, sample):
    return os.path.join(
        output_dir,
        "samples",
        f"{sample_idx:04d}_bundle_{int(sample['bundle_id'])}",
    )


def _save_success_artifacts(output_dir, sample_idx, sample, result):
    sample_dir = _sample_output_dir(output_dir, sample_idx, sample)
    _write_json(os.path.join(sample_dir, "case.json"), result["case"])
    _write_json(os.path.join(sample_dir, "model_case.json"), result["model_case"])
    _write_text(os.path.join(sample_dir, "input.txt"), result["prompt"])
    _write_text(os.path.join(sample_dir, "output.txt"), result["raw_response"])
    _write_json(
        os.path.join(sample_dir, "parsed_response.json"),
        result["parsed_response"],
    )
    _write_json(
        os.path.join(sample_dir, "validation_issues.json"),
        result["validation_issues"],
    )
    _write_json(os.path.join(sample_dir, "evaluation.json"), result["evaluation"])


def _save_error_artifact(output_dir, sample_idx, sample, error):
    sample_dir = _sample_output_dir(output_dir, sample_idx, sample)
    _write_json(
        os.path.join(sample_dir, "error.json"),
        {
            "sample_idx": sample_idx,
            "bundle_id": int(sample["bundle_id"]),
            "true_label": str(sample["true_option_char"]),
            "error": str(error),
        },
    )


def _result_row(sample_idx, sample, result):
    evaluation = dict(result["evaluation"])
    issues = [str(issue) for issue in result.get("validation_issues", [])]
    return {
        "sample_idx": sample_idx,
        "bundle_id": int(sample["bundle_id"]),
        **evaluation,
        "error": " | ".join(issues),
    }


def _error_row(sample_idx, sample, error):
    return {
        "sample_idx": sample_idx,
        "bundle_id": int(sample["bundle_id"]),
        "candidate_count": len(sample.get("candidate_indices", [])),
        "plausible_labels": [],
        "ranking": [],
        "plausible_rank_top_k": [],
        "plausible_ranking_consistent": None,
        "plausible_set_size": 0,
        "selection_fraction": 0.0,
        "random_same_size_gt_inclusion_baseline": 0.0,
        "true_label": str(sample["true_option_char"]),
        "gt_in_plausible_set": False,
        "gt_rank": None,
        "reciprocal_rank": 0.0,
        "hit_at_1": False,
        "hit_at_3": False,
        "hit_at_5": False,
        "llm_calls": 1,
        "error": str(error),
    }


def _save_batch_state(output_dir, rows, requested_count, partial):
    ordered = sorted(rows, key=lambda row: int(row["sample_idx"]))
    stem = "results_partial" if partial else "results"
    _write_json(os.path.join(output_dir, f"{stem}.json"), ordered)
    _write_csv(os.path.join(output_dir, f"{stem}.csv"), ordered)
    summary = {
        "requested_sample_count": requested_count,
        **aggregate_plausible_set_evaluations(ordered),
    }
    summary_stem = "summary_partial" if partial else "summary"
    _write_json(os.path.join(output_dir, f"{summary_stem}.json"), summary)
    return summary


def _select_samples(args, conf, samples):
    if args.sample_idx is not None:
        sample_idx = int(args.sample_idx)
        if sample_idx < 0 or sample_idx >= len(samples):
            raise IndexError(
                f"sample_idx {sample_idx} out of range for {len(samples)} samples"
            )
        return [(sample_idx, samples[sample_idx])]

    start_idx = max(0, int(args.start_idx))
    configured_count = int(conf.get("plausible_set_sample_count", 250))
    sample_count = (
        int(args.sample_count)
        if args.sample_count is not None
        else configured_count
    )
    if sample_count <= 0:
        raise ValueError("--sample_count must be positive")
    end_idx = min(len(samples), start_idx + sample_count)
    return list(enumerate(samples[start_idx:end_idx], start=start_idx))


async def _run(args):
    with open(args.config, "r", encoding="utf-8") as handle:
        conf = yaml.safe_load(handle)
    set_seed(int(conf.get("seed", 45)))
    split = str(args.split).strip().lower()
    if split not in {"valid", "test"}:
        raise ValueError("--split must be valid or test")

    eval_conf = dict(conf)
    eval_conf["toy_eval"] = -1
    all_samples = BundleZeroShotDataset(eval_conf, split=split).get_eval_samples()
    indexed_samples = _select_samples(args, conf, all_samples)
    desired_indices = {sample_idx for sample_idx, _ in indexed_samples}

    output_dir = _resolve_output_dir(args, conf)
    os.makedirs(output_dir, exist_ok=True)
    previous_rows = _load_previous_rows(args, output_dir)
    rows_by_idx = {
        int(row["sample_idx"]): row
        for row in previous_rows
        if "sample_idx" in row and int(row["sample_idx"]) in desired_indices
    }
    pending = [
        (sample_idx, sample)
        for sample_idx, sample in indexed_samples
        if sample_idx not in rows_by_idx
    ]

    client, resolved = _build_client(conf)
    concurrency = (
        int(args.max_concurrent)
        if args.max_concurrent is not None
        else int(conf.get("plausible_set_max_concurrent", 1))
    )
    concurrency = max(1, concurrency)
    _write_json(
        os.path.join(output_dir, "run.json"),
        {
            "phase": "direct_plausible_candidate_set",
            "dataset": conf["dataset"],
            "split": split,
            "requested_sample_count": len(indexed_samples),
            "start_idx": indexed_samples[0][0] if indexed_samples else None,
            "sample_idx": args.sample_idx,
            "max_concurrent": concurrency,
            "llm_calls_per_sample": 1,
            "resumed": bool(args.resume),
            **resolved,
        },
    )

    async def call_text(prompt, step_name):
        return await generate_content_with_retry(
            client,
            resolved["model"],
            prompt,
            conf,
            int(conf.get("plausible_set_max_output_tokens", 2500)),
            step_name,
        )

    semaphore = asyncio.Semaphore(concurrency)
    progress = len(rows_by_idx)

    async def run_one(sample_idx, sample):
        nonlocal progress
        try:
            async with semaphore:
                result = await run_direct_plausible_set(sample, conf, call_text)
            _save_success_artifacts(output_dir, sample_idx, sample, result)
            row = _result_row(sample_idx, sample, result)
        except QuotaExceededError:
            raise
        except Exception as exc:
            _save_error_artifact(output_dir, sample_idx, sample, exc)
            row = _error_row(sample_idx, sample, exc)
        progress += 1
        selected = row["plausible_set_size"]
        gt_in_set = row["gt_in_plausible_set"]
        status = f"error={row['error']}" if row["error"] else f"GT-in-set={gt_in_set}"
        print(
            f"[{progress}/{len(indexed_samples)}] "
            f"{split}[{sample_idx}] bundle_{sample['bundle_id']} | "
            f"selected={selected} | {status}"
        )
        return row

    print(f">>> Dataset: {conf['dataset']}")
    print(f">>> Split: {split}")
    print(
        f">>> Samples: {len(indexed_samples)} "
        f"({len(rows_by_idx)} resumed, {len(pending)} pending)"
    )
    print(f">>> Model: {resolved['provider']} / {resolved['model']}")
    print(f">>> Maximum concurrent calls: {concurrency}")
    print(">>> Selecting every defensible completion; no single-answer prediction")

    rows = list(rows_by_idx.values())
    try:
        for offset in range(0, len(pending), concurrency):
            chunk = pending[offset : offset + concurrency]
            completed = await asyncio.gather(
                *(run_one(sample_idx, sample) for sample_idx, sample in chunk)
            )
            rows.extend(completed)
            _save_batch_state(
                output_dir,
                rows,
                requested_count=len(indexed_samples),
                partial=True,
            )
    except QuotaExceededError as exc:
        _save_batch_state(
            output_dir,
            rows,
            requested_count=len(indexed_samples),
            partial=True,
        )
        print(f"[Stopped] {exc}")
        print(f"[Resume] python tests/test_direct_plausible_set.py --resume \"{output_dir}\"")
        return 1

    summary = _save_batch_state(
        output_dir,
        rows,
        requested_count=len(indexed_samples),
        partial=False,
    )
    print("-" * 56)
    print(
        f">>> Valid samples: {summary['valid_sample_count']} / "
        f"{summary['requested_sample_count']}"
    )
    print(f">>> GT plausible coverage: {summary['gt_plausible_coverage']:.4f}")
    print(
        f">>> Average plausible-set size: "
        f"{summary['average_plausible_set_size']:.4f}"
    )
    print(
        f">>> Average selection fraction: "
        f"{summary['average_selection_fraction']:.4f}"
    )
    print(
        f">>> Coverage above random same-size selection: "
        f"{summary['coverage_above_random_same_size']:.4f}"
    )
    print(
        f">>> Plausible-ranking consistency: "
        f"{summary['plausible_ranking_consistency_rate']:.4f}"
    )
    print(f">>> Hit@1: {summary['hit_rate_at_1']:.4f}")
    print(f">>> Hit@3: {summary['hit_rate_at_3']:.4f}")
    print(f">>> Hit@5: {summary['hit_rate_at_5']:.4f}")
    print(f">>> Mean reciprocal rank: {summary['mean_reciprocal_rank']:.4f}")
    print(f">>> Mean GT rank: {summary['mean_gt_rank']:.4f}")
    print(f">>> Output: {output_dir}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Direct one-call plausible candidate-set batch analysis"
    )
    parser.add_argument("--config", default="config_operator.yaml")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--sample_count",
        type=int,
        default=None,
        help="Batch size; defaults to plausible_set_sample_count (250)",
    )
    parser.add_argument(
        "--sample_idx",
        type=int,
        default=None,
        help="Run only one sample instead of the batch",
    )
    parser.add_argument("--start_idx", type=int, default=0)
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

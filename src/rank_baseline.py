"""Matched text-only full-ranking baseline for bundle completion.

This keeps the question and option rendering from ``main_baseline.py`` and
changes only the answer instruction: instead of returning one letter, the
model must rank every supplied option.

Example:
    python src/rank_baseline.py \
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
import time

import yaml

from code.common import parse_json_from_text
from code.pipeline import build_decision_case
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
from operator_learning.spec_first_prediction import (
    aggregate_prediction_rows,
    evaluate_full_ranking,
)


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METHOD_NAME = "text_only_rank_baseline"
RESULT_FIELDS = [
    "sample_idx",
    "bundle_id",
    "candidate_count",
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


def _task_names(dataset):
    if "spotify" in str(dataset or "").lower():
        return "playlist continuation", "music playlist", "song"
    return "bundle construction", "fashion outfit", "fashion item"


def rank_baseline_prompt(decision_case):
    """Render the top-1 baseline question with only the output task changed."""
    task_name, bundle_name, item_name = _task_names(
        decision_case.get("dataset")
    )
    partial_items = "; ".join(
        f"{index + 1}. {item.get('text', '')}"
        for index, item in enumerate(
            decision_case.get("partial_items", [])
        )
    )
    options = "; ".join(
        f"{candidate.get('label', '')}. {candidate.get('text', '')}"
        for candidate in decision_case.get("candidates", [])
    )
    return (
        f"You are a helpful and honest assistant. The following are multiple "
        f"choice questions about {task_name}. "
        "You should directly answer the question by ranking every option from "
        "most to least likely to be the correct option. Do not provide any "
        "explanation or mention the option content. "
        f"Question: Given the partial {bundle_name}: {partial_items}, which "
        f"candidate {item_name} should be included into this {bundle_name}?\n"
        f"Options: {options}\n"
        "Your answer should rank every supplied option label exactly once from "
        "most to least likely to be correct.\n"
        'Return JSON only. The object must contain exactly one field named '
        '"ranking", whose value is the complete ordered label array.\n'
        "Ranking:"
    )


def validate_ranking_result(value, labels):
    """Require exactly one complete ranking over the supplied labels."""
    if not isinstance(value, dict):
        return ["result must be a JSON object"]
    if set(value) != {"ranking"}:
        return ["result must contain exactly the ranking field"]
    ranking = value.get("ranking")
    if not isinstance(ranking, list):
        return ["ranking must be a list"]

    issues = []
    if any(not isinstance(label, str) for label in ranking):
        issues.append("ranking entries must be strings")
    if len(ranking) != len(labels):
        issues.append("ranking must contain every supplied option label")
    if len(ranking) != len(set(ranking)):
        issues.append("ranking labels must be unique")
    if set(ranking) != set(labels):
        issues.append(
            "ranking must contain every supplied option label exactly once"
        )
    return list(dict.fromkeys(issues))


def _build_client(conf):
    provider = stage_provider(conf, "prediction")
    model = stage_model(conf, "prediction")
    api_key, env = resolve_api_key_from_keys(
        conf,
        [
            "rank_baseline_api_key_env",
            "operator_prediction_api_key_env",
            "baseline_prediction_api_key_env",
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
    with open(
        temporary,
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for row in rows:
            serialized = {
                field: row.get(field, "") for field in RESULT_FIELDS
            }
            if isinstance(serialized["ranking"], (list, dict)):
                serialized["ranking"] = json.dumps(
                    serialized["ranking"],
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            writer.writerow(serialized)
    os.replace(temporary, path)


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
            "rank_baseline",
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
                raise ValueError(
                    f"resume JSON must contain a list: {candidate}"
                )
            return [row for row in value if isinstance(row, dict)]
    raise FileNotFoundError("resume results JSON was not found")


def _select_samples(args, conf, samples):
    if args.sample_idx is not None:
        index = int(args.sample_idx)
        if index < 0 or index >= len(samples):
            raise IndexError(
                f"sample_idx {index} out of range for {len(samples)}"
            )
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
        "total_llm_calls": sum(
            int(row.get("llm_calls", 0)) for row in ordered
        ),
    }
    summary_stem = "summary_partial" if partial else "summary"
    _write_json(
        os.path.join(output_dir, f"{summary_stem}.json"),
        summary,
    )
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
    samples = BundleZeroShotDataset(
        eval_conf,
        split=split,
    ).get_eval_samples()
    indexed_samples = _select_samples(args, conf, samples)
    desired_indices = {index for index, _ in indexed_samples}

    output_dir = _resolve_output_dir(args, conf)
    os.makedirs(output_dir, exist_ok=True)
    previous = _load_previous_rows(args, output_dir)
    rows_by_index = {
        int(row["sample_idx"]): row
        for row in previous
        if "sample_idx" in row
        and int(row["sample_idx"]) in desired_indices
    }
    pending = [
        (index, sample)
        for index, sample in indexed_samples
        if index not in rows_by_index
    ]

    client, resolved = _build_client(conf)
    concurrency = max(
        1,
        int(
            args.max_concurrent
            if args.max_concurrent is not None
            else conf.get("operator_prediction_max_concurrent", 1)
        ),
    )
    max_output_tokens = int(
        conf.get(
            "rank_baseline_max_output_tokens",
            conf.get("operator_prediction_max_output_tokens", 2000),
        )
    )

    manifest = [
        {
            "sample_idx": int(index),
            "bundle_id": int(sample["bundle_id"]),
            "partial_item_ids": [
                int(item_id)
                for item_id in sample.get("input_indices", [])
            ],
            "candidate_item_ids": [
                int(item_id)
                for item_id in sample.get("candidate_indices", [])
            ],
            "true_item_id": int(sample["true_indice"]),
            "true_label": str(sample["true_option_char"]),
        }
        for index, sample in indexed_samples
    ]
    manifest_path = os.path.join(output_dir, "sample_manifest.json")
    if args.resume and os.path.isfile(manifest_path):
        if _read_json(manifest_path) != manifest:
            raise ValueError(
                "resume sample manifest does not match the requested "
                "split/range or candidate ordering"
            )
    _write_json(manifest_path, manifest)
    _write_json(
        os.path.join(output_dir, "run.json"),
        {
            "phase": "text_only_full_ranking_baseline",
            "dataset": conf["dataset"],
            "split": split,
            "start_idx": (
                indexed_samples[0][0] if indexed_samples else None
            ),
            "requested_sample_count": len(indexed_samples),
            "selection": (
                "contiguous dataset order; identical to "
                "spec-first operator batch"
            ),
            "seed": int(conf.get("seed", 45)),
            "shuffle_seed": int(conf.get("shuffle_seed", 41)),
            "num_cans": int(conf.get("num_cans", 10)),
            "num_token": int(conf.get("num_token", 5)),
            "llm_calls_per_sample": 1,
            "max_concurrent_samples": concurrency,
            "max_output_tokens": max_output_tokens,
            "resumed": bool(args.resume),
            "prediction_model": resolved,
        },
    )

    print(f">>> Config: {args.config}")
    print(f">>> Method: {METHOD_NAME}")
    print(f">>> Dataset: {conf['dataset']}")
    print(f">>> Split: {split}")
    print(
        f">>> Samples: {len(indexed_samples)} "
        f"({len(rows_by_index)} resumed, {len(pending)} pending)"
    )
    print(
        f">>> Prediction: {resolved['provider']} / "
        f"{resolved['model']}"
    )
    print(">>> Input: main_baseline question/options; no evidence")
    print(">>> Output: complete text-only candidate ranking")
    print(f">>> Maximum concurrent samples: {concurrency}")

    semaphore = asyncio.Semaphore(concurrency)
    progress = len(rows_by_index)

    async def run_one(sample_idx, sample):
        nonlocal progress
        sample_dir = _sample_dir(output_dir, sample_idx, sample)
        os.makedirs(sample_dir, exist_ok=True)
        llm_calls = 0
        try:
            decision_case = build_decision_case(sample, conf)
            labels = [
                str(candidate["label"])
                for candidate in decision_case.get("candidates", [])
            ]
            prompt = rank_baseline_prompt(decision_case)
            _write_json(
                os.path.join(sample_dir, "case.json"),
                decision_case,
            )
            _write_text(os.path.join(sample_dir, "input.txt"), prompt)

            async with semaphore:
                raw = await generate_content_with_retry(
                    client,
                    resolved["model"],
                    prompt,
                    conf,
                    max_output_tokens,
                    "text-only full-ranking baseline",
                )
            llm_calls += 1
            _write_text(os.path.join(sample_dir, "output.txt"), raw)

            parsed = parse_json_from_text(raw)
            _write_json(
                os.path.join(sample_dir, "parsed_response.json"),
                parsed,
            )
            issues = validate_ranking_result(parsed, labels)
            _write_json(
                os.path.join(sample_dir, "validation_issues.json"),
                issues,
            )
            if issues:
                raise ValueError(" | ".join(issues))

            ranking = [str(label) for label in parsed["ranking"]]
            evaluation = evaluate_full_ranking(
                {
                    "prediction": ranking[0],
                    "ranking": ranking,
                },
                sample["true_option_char"],
            )
            row = {
                "sample_idx": int(sample_idx),
                "bundle_id": int(sample["bundle_id"]),
                "candidate_count": len(labels),
                **evaluation,
                "llm_calls": llm_calls,
                "valid": True,
                "error": None,
            }
            _write_json(
                os.path.join(sample_dir, "evaluation.json"),
                row,
            )
        except QuotaExceededError:
            raise
        except Exception as exc:
            row = _error_row(
                sample_idx,
                sample,
                exc,
                llm_calls=llm_calls,
            )
            _write_json(
                os.path.join(sample_dir, "error.json"),
                {
                    "sample_idx": int(sample_idx),
                    "bundle_id": int(sample["bundle_id"]),
                    "error": str(exc),
                    "llm_calls": llm_calls,
                },
            )

        progress += 1
        rank = row.get("gt_rank")
        print(
            f"[{progress}/{len(indexed_samples)}] "
            f"{split}[{sample_idx}] bundle_{sample['bundle_id']} | "
            f"pred={row.get('prediction')} true={row['true_label']} "
            f"GT-rank={rank} valid={row['valid']}"
        )
        return row

    try:
        for offset in range(0, len(pending), concurrency):
            chunk = pending[offset : offset + concurrency]
            completed = await asyncio.gather(
                *(run_one(sample_idx, sample) for sample_idx, sample in chunk)
            )
            for row in completed:
                rows_by_index[int(row["sample_idx"])] = row
            _save_state(
                output_dir,
                list(rows_by_index.values()),
                requested_count=len(indexed_samples),
                partial=True,
            )
    except QuotaExceededError:
        _save_state(
            output_dir,
            list(rows_by_index.values()),
            requested_count=len(indexed_samples),
            partial=True,
        )
        raise

    summary = _save_state(
        output_dir,
        list(rows_by_index.values()),
        requested_count=len(indexed_samples),
        partial=False,
    )
    print("-" * 60)
    print(f"Saved to: {output_dir}")
    print(
        f">>> Valid samples: {summary['valid_sample_count']} / "
        f"{summary['requested_sample_count']}"
    )
    print(f">>> Hit@1: {summary['hit_rate_at_1']:.4f}")
    print(f">>> Hit@3: {summary['hit_rate_at_3']:.4f}")
    print(f">>> Hit@5: {summary['hit_rate_at_5']:.4f}")
    print(
        f">>> Mean reciprocal rank: "
        f"{summary['mean_reciprocal_rank']:.4f}"
    )
    print(f">>> Mean GT rank: {summary['mean_gt_rank']:.4f}")
    print(f">>> Total LLM calls: {summary['total_llm_calls']}")
    print("-" * 60)
    return 0


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run the main_baseline question as a matched text-only "
            "full-ranking baseline"
        )
    )
    parser.add_argument("--config", default="config_operator.yaml")
    parser.add_argument("--split", default="test")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--sample_count", type=int, default=None)
    parser.add_argument("--sample_idx", type=int, default=None)
    parser.add_argument("--max_concurrent", type=int, default=None)
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--resume", default="")
    args = parser.parse_args()

    try:
        return asyncio.run(_run(args))
    except QuotaExceededError as exc:
        print(f"[Stopped] {exc}")
        if args.resume:
            print(f"[Resume] python src/rank_baseline.py --resume {args.resume}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Text-only baseline entry point for bundle completion.

Usage:
    python src/main_baseline.py --config config_baseline.yaml
"""

import argparse
import asyncio
import json
import os
import re
import time

import pandas as pd
import yaml

from code.pipeline import build_decision_case
from dataset import BundleZeroShotDataset, set_seed
from main import (
    QuotaExceededError,
    compact_result_row,
    create_llm_client,
    default_api_key_envs_for_provider,
    generate_content_with_retry,
    llm_provider,
    parse_model_response,
    resolve_api_key_from_keys,
)


METHOD_NAME = "text_baseline"


def _first_config_value(conf, keys, default=""):
    for key in keys:
        value = str(conf.get(key, "")).strip()
        if value:
            return value
    return default


def baseline_provider(conf):
    return _first_config_value(
        conf,
        ["baseline_prediction_provider", "code_prediction_provider", "sem_prediction_provider"],
        llm_provider(conf),
    ).lower()


def baseline_model(conf):
    model = _first_config_value(
        conf,
        ["baseline_prediction_model", "code_prediction_model", "sem_prediction_model"],
        str(conf.get("model", "")).strip(),
    )
    if not model:
        raise ValueError("No model configured for baseline prediction")
    return model


def _build_client(conf):
    provider = baseline_provider(conf)
    model = baseline_model(conf)
    api_key, env = resolve_api_key_from_keys(
        conf,
        [
            "baseline_prediction_api_key_env",
            "code_prediction_api_key_env",
            "sem_prediction_api_key_env",
        ],
        default_api_key_envs_for_provider(provider),
    )
    return {
        "client": create_llm_client(provider, api_key),
        "provider": provider,
        "model": model,
    }, {"api_key_env": env, "provider": provider, "model": model}


def _task_names(dataset):
    if "spotify" in str(dataset or "").lower():
        return "playlist continuation", "music playlist", "song"
    return "bundle construction", "fashion outfit", "fashion item"


def baseline_prompt(decision_case):
    task_name, bundle_name, item_name = _task_names(decision_case.get("dataset"))
    partial_items = "; ".join(
        f"{index + 1}. {item.get('text', '')}"
        for index, item in enumerate(decision_case.get("partial_items", []))
    )
    options = "; ".join(
        f"{candidate.get('label', '')}. {candidate.get('text', '')}"
        for candidate in decision_case.get("candidates", [])
    )
    return (
        f"You are a helpful and honest assistant. The following are multiple choice questions about {task_name}. "
        "You should directly answer the question by choosing the letter of the correct option. Only provide the letter "
        "of your answer, without any explanation or mentioning the option content. "
        f"Question: Given the partial {bundle_name}: {partial_items}, which candidate {item_name} should be included "
        f"into this {bundle_name}?\n"
        f"Options: {options}\n"
        'Your answer should indicate your choice with a single letter (e.g., "A," "B," "C," etc.).\n'
        "Choice:"
    )


def run_output_dir(conf, timestamp):
    directory = os.path.join(conf["output_dir"], conf["dataset"], timestamp)
    os.makedirs(directory, exist_ok=True)
    return directory


def result_path(conf, timestamp, partial=False):
    filename = "results_partial.csv" if partial else "results.csv"
    return os.path.join(run_output_dir(conf, timestamp), filename)


def _write_text(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(str(text or ""))


def _write_json(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=str)


def save_artifacts(row, conf, timestamp):
    bundle_id = row.get("bundle_id", "unknown")
    stage_dir = os.path.join(
        run_output_dir(conf, timestamp),
        f"bundle_{bundle_id}",
        "baseline_prediction",
    )
    _write_text(os.path.join(stage_dir, "input.txt"), row.get("baseline_prompt", ""))
    _write_text(os.path.join(stage_dir, "output.txt"), row.get("baseline_raw_response", ""))
    _write_json(
        os.path.join(stage_dir, "prediction.json"),
        {
            "bundle_id": row.get("bundle_id"),
            "prediction": row.get("prediction", ""),
            "gt_label": row.get("true_option_char", ""),
            "gt_item_id": row.get("true_indice", ""),
            "hit": int(row.get("hit", 0)),
        },
    )
    if row.get("baseline_decision_case"):
        try:
            decision_case = json.loads(row.get("baseline_decision_case"))
        except json.JSONDecodeError:
            decision_case = row.get("baseline_decision_case")
        _write_json(os.path.join(stage_dir, "decision_case.json"), decision_case)


def save_error_artifact(row, conf, timestamp, error_text):
    bundle_id = row.get("bundle_id", "unknown")
    path = os.path.join(run_output_dir(conf, timestamp), f"bundle_{bundle_id}", "errors", "output.txt")
    _write_text(path, error_text)


def save_results(results, conf, timestamp, partial=False):
    frame = pd.DataFrame(results)
    hit_rate = frame["hit"].mean() if not frame.empty else 0.0
    valid_labels = [chr(ord("A") + i) for i in range(int(conf.get("num_cans", 10)))]
    valid_mask = frame["prediction"].isin(valid_labels) if not frame.empty else pd.Series(dtype=bool)
    valid_ratio = valid_mask.mean() if not frame.empty else 0.0
    valid_hit_rate = frame.loc[valid_mask, "hit"].mean() if valid_mask.sum() else 0.0

    if not frame.empty:
        frame["accuracy"] = hit_rate
        columns = [
            "bundle_id",
            "partial_item_ids",
            "candidate_item_ids",
            "prediction",
            "gt_label",
            "gt_item_id",
            "hit",
            "accuracy",
        ]
        frame = frame[[col for col in columns if col in frame.columns]]

    path = result_path(conf, timestamp, partial)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return path, frame, hit_rate, valid_ratio, valid_hit_rate, int(valid_mask.sum()) if not frame.empty else 0


async def process_samples(client, samples, conf, timestamp, initial_results=None, start_idx=0):
    results = list(initial_results or [])
    concurrency = int(conf.get("max_concurrent", 1))
    semaphore = asyncio.Semaphore(concurrency)
    total = start_idx + len(samples)

    async def run_one(sample, current_idx):
        row = dict(sample)
        async with semaphore:
            try:
                decision_case = build_decision_case(sample, conf)
                prompt = baseline_prompt(decision_case)
                raw = await generate_content_with_retry(
                    client,
                    client["model"],
                    prompt,
                    conf,
                    int(conf.get("baseline_prediction_max_output_tokens", conf.get("code_prediction_max_output_tokens", 200))),
                    "baseline prediction",
                )
                prediction = parse_model_response(raw)
                row["prediction"] = prediction
                row["raw_response"] = raw
                row["baseline_prompt"] = prompt
                row["baseline_raw_response"] = raw
                row["baseline_decision_case"] = json.dumps(decision_case, ensure_ascii=False, separators=(",", ":"))
                row["hit"] = int(prediction == sample["true_option_char"])
                save_artifacts(row, conf, timestamp)
            except QuotaExceededError:
                raise
            except Exception as exc:
                prediction = "ERR_EX"
                raw = str(exc)
                row["prediction"] = prediction
                row["raw_response"] = raw
                row["hit"] = int(prediction == sample["true_option_char"])
                save_error_artifact(row, conf, timestamp, raw)
        print(f"[{current_idx + 1}/{total}] True: {sample['true_option_char']} | Pred: {prediction}")
        return compact_result_row(row)

    for offset in range(0, len(samples), concurrency):
        chunk = samples[offset: offset + concurrency]
        tasks = [run_one(sample, start_idx + offset + idx) for idx, sample in enumerate(chunk)]
        try:
            completed = await asyncio.gather(*tasks)
        except QuotaExceededError:
            save_results(results, conf, timestamp, partial=True)
            raise
        results.extend(completed)
        save_results(results, conf, timestamp, partial=True)
    return results


def load_resume(path):
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp949")


def main():
    parser = argparse.ArgumentParser(description="Run text-only baseline bundle evaluation")
    parser.add_argument("--config", default="config_baseline.yaml")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--resume", default="")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        conf = yaml.safe_load(handle)

    os.makedirs(conf["output_dir"], exist_ok=True)
    set_seed(int(conf.get("seed", 45)))

    samples = BundleZeroShotDataset(conf).get_eval_samples()
    initial_results = None
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    if args.resume and os.path.exists(args.resume):
        previous = load_resume(args.resume)
        initial_results = previous.to_dict("records")
        args.start_idx = len(initial_results)
        match = re.search(r"_(\d{8}_\d{6})(?:_partial)?\.csv$", args.resume)
        if match:
            timestamp = match.group(1)
        else:
            parent = os.path.basename(os.path.dirname(os.path.abspath(args.resume)))
            if parent:
                timestamp = parent
    if args.start_idx > 0:
        samples = samples[args.start_idx:]

    print(f">>> Config:  {args.config}")
    print(f">>> Method:  {METHOD_NAME}")
    print(f">>> Dataset: {conf['dataset']}")
    print(f">>> Samples: {len(samples)} (start_idx={args.start_idx})")

    try:
        client, resolved = _build_client(conf)
    except ValueError as exc:
        print(f"[Error] {exc}")
        return 1
    print(
        f">>> Prediction: provider={resolved['provider']} "
        f"model={resolved['model']} api_key_env={resolved['api_key_env']}"
    )

    try:
        results = asyncio.run(
            process_samples(
                client,
                samples,
                conf,
                timestamp,
                initial_results=initial_results,
                start_idx=args.start_idx,
            )
        )
    except QuotaExceededError as exc:
        print(f"[Stopped] {exc}")
        print(f"[Resume]  Partial results: {result_path(conf, timestamp, partial=True)}")
        return 1

    path, _, hit_rate, valid_ratio, valid_hit_rate, valid_count = save_results(
        results, conf, timestamp, partial=False
    )
    partial = result_path(conf, timestamp, partial=True)
    if os.path.exists(partial):
        os.remove(partial)
    print("-" * 40)
    print(f"Saved to:           {path}")
    print(f"Overall Hit Rate:   {hit_rate:.4f}")
    print(f"Valid-Only Hit Rate:{valid_hit_rate:.4f}  ({valid_count} valid samples)")
    print(f"Valid Ratio:        {valid_ratio:.4f}")
    print("-" * 40)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

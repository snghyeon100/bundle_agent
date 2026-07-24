"""Predict one bundle completion using evidence from operator-generated code.

Usage:
    python tests/test_operator_prediction.py \
        --config config_operator.yaml \
        --operator_code_output tests/outputs/operator_code/pog_dense_<date>

If --operator_code_output is omitted, the latest accepted operator-code output
for the configured dataset is used.
"""

import argparse
import asyncio
import json
import os
import sys
import time

import yaml


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from code.common import build_case_view, candidate_labels
from code.pipeline import build_decision_case, validate_adaptive_bundle_evidence
from code.prompts import decision_prompt
from dataset import BundleZeroShotDataset, set_seed
from main import (
    create_llm_client,
    default_api_key_envs_for_provider,
    generate_content_with_retry,
    parse_model_response,
    resolve_api_key_from_keys,
    stage_model,
    stage_provider,
)


def _read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=str)


def _write_text(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(str(value or ""))


def _accepted_code_output(path):
    run_path = os.path.join(path, "run.json")
    evidence_path = os.path.join(path, "evidence.json")
    summary_path = os.path.join(path, "execution_summary.json")
    if not all(os.path.isfile(item) for item in (run_path, evidence_path, summary_path)):
        return False
    try:
        summary = _read_json(summary_path)
        evidence = _read_json(evidence_path)
    except (OSError, json.JSONDecodeError):
        return False
    return bool(summary.get("accepted")) and isinstance(evidence, dict)


def _latest_operator_code_output(dataset):
    output_root = os.path.join(ROOT, "tests", "outputs", "operator_code")
    if not os.path.isdir(output_root):
        raise FileNotFoundError(f"operator-code output root not found: {output_root}")
    candidates = []
    for name in os.listdir(output_root):
        path = os.path.join(output_root, name)
        if (
            name.startswith(f"{dataset}_")
            and os.path.isdir(path)
            and _accepted_code_output(path)
        ):
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(
            f"no accepted operator-code output found for {dataset}; "
            "run test_operator_code_generation.py without --skip_execute first"
        )
    return max(candidates, key=os.path.getmtime)


def _build_client(conf):
    provider = stage_provider(conf, "prediction")
    model = stage_model(conf, "prediction")
    api_key, env = resolve_api_key_from_keys(
        conf,
        [
            "code_prediction_api_key_env",
            "code_decision_api_key_env",
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
    }, {"provider": provider, "model": model, "api_key_env": env}


async def _run(args):
    with open(args.config, "r", encoding="utf-8") as handle:
        conf = yaml.safe_load(handle)
    set_seed(int(conf.get("seed", 45)))

    code_output_dir = os.path.abspath(
        args.operator_code_output
        or _latest_operator_code_output(conf["dataset"])
    )
    run = _read_json(os.path.join(code_output_dir, "run.json"))
    evidence = _read_json(os.path.join(code_output_dir, "evidence.json"))
    execution_summary = _read_json(
        os.path.join(code_output_dir, "execution_summary.json")
    )
    if not execution_summary.get("accepted"):
        raise ValueError(
            "operator-code execution was not accepted; prediction requires valid evidence"
        )
    if run.get("dataset") != conf["dataset"]:
        raise ValueError(
            f"dataset mismatch: config={conf['dataset']} code_output={run.get('dataset')}"
        )

    split = str(run.get("split", "test"))
    sample_idx = int(run.get("sample_idx", 0))
    if split not in {"valid", "test"}:
        raise ValueError(f"unsupported split recorded in code output: {split}")
    eval_conf = dict(conf)
    eval_conf["toy_eval"] = -1
    samples = BundleZeroShotDataset(eval_conf, split=split).get_eval_samples()
    if sample_idx < 0 or sample_idx >= len(samples):
        raise IndexError(
            f"sample_idx {sample_idx} out of range for {len(samples)} {split} samples"
        )
    sample = samples[sample_idx]
    if int(run.get("bundle_id")) != int(sample["bundle_id"]):
        raise ValueError(
            "bundle identity mismatch between operator-code output and dataset sample"
        )

    case_view = build_case_view(sample, conf["dataset"])
    evidence_issues = validate_adaptive_bundle_evidence(evidence, case_view)
    if evidence_issues:
        raise ValueError("invalid operator evidence: " + " | ".join(evidence_issues))

    decision_case = build_decision_case(sample, conf)
    prompt = decision_prompt(decision_case, evidence)
    client, resolved = _build_client(conf)

    print(f">>> Dataset: {conf['dataset']}")
    print(f">>> Operator-code output: {code_output_dir}")
    print(f">>> Sample: {split}[{sample_idx}] / bundle_{sample['bundle_id']}")
    print(f">>> Model: {resolved['provider']} / {resolved['model']}")
    print(">>> Predicting with operator-generated evidence")

    raw = await generate_content_with_retry(
        client,
        resolved["model"],
        prompt,
        conf,
        int(conf.get("code_prediction_max_output_tokens", 300)),
        "operator-evidence final prediction",
    )
    prediction = parse_model_response(raw)
    labels = candidate_labels(case_view)
    valid_prediction = prediction in labels
    true_label = str(sample["true_option_char"])
    hit = int(valid_prediction and prediction == true_label)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.abspath(
        args.output_dir
        or os.path.join(
            ROOT,
            "tests",
            "outputs",
            "operator_prediction",
            f"{conf['dataset']}_{stamp}",
        )
    )
    os.makedirs(output_dir, exist_ok=True)
    result = {
        "prediction": prediction,
        "valid_prediction": valid_prediction,
        "candidate_labels": labels,
        "true_label": true_label,
        "hit": hit,
    }
    _write_json(
        os.path.join(output_dir, "run.json"),
        {
            "phase": "operator_evidence_prediction",
            "dataset": conf["dataset"],
            "operator_code_output": code_output_dir,
            "split": split,
            "sample_idx": sample_idx,
            "bundle_id": int(sample["bundle_id"]),
            **resolved,
        },
    )
    _write_json(os.path.join(output_dir, "case.json"), decision_case)
    _write_json(os.path.join(output_dir, "evidence.json"), evidence)
    _write_text(os.path.join(output_dir, "input.txt"), prompt)
    _write_text(os.path.join(output_dir, "output.txt"), raw)
    _write_json(os.path.join(output_dir, "prediction.json"), result)

    print(f">>> Prediction: {prediction}")
    print(f">>> True label: {true_label}")
    print(f">>> Hit: {hit}")
    print(f">>> Output: {output_dir}")
    return 0 if valid_prediction else 1


def main():
    parser = argparse.ArgumentParser(
        description="Predict one sample using evidence from operator-generated code"
    )
    parser.add_argument("--config", default="config_operator.yaml")
    parser.add_argument("--operator_code_output", default="")
    parser.add_argument("--output_dir", default="")
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())

"""Run Stage 2 prediction from an existing Stage 1 evidence artifact.

Usage:
    .\\.venv\\Scripts\\python.exe tests\\stage_2_prediction\\run.py --config config_code.yaml --stage1_dir analysis\\stage_1_code_generation\\pog_dense\\gpt-4.1-mini\\bundle_9388_20260706_101943
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time

import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from code.pipeline import build_decision_case
from code.prompts import decision_prompt
from dataset import BundleZeroShotDataset, set_seed
from main import (
    create_llm_client,
    default_api_key_envs_for_provider,
    generate_content_with_retry,
    parse_model_response,
    print_debug,
    resolve_api_key_from_keys,
    stage_model,
    stage_provider,
)


def _write_text(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(str(text or ""))


def _write_json(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=str)


def _safe_path_part(value):
    text = str(value or "unknown").strip()
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in text) or "unknown"


def _build_prediction_client(conf):
    provider = stage_provider(conf, "prediction")
    model = stage_model(conf, "prediction")
    api_key, env = resolve_api_key_from_keys(
        conf,
        [
            "code_prediction_api_key_env",
            "code_decision_api_key_env",
        ],
        default_api_key_envs_for_provider(provider),
    )
    return {
        "client": create_llm_client(provider, api_key),
        "provider": provider,
        "model": model,
    }, {"api_key_env": env, "provider": provider, "model": model}


def _bundle_id_from_stage1_dir(stage1_dir):
    name = os.path.basename(os.path.abspath(stage1_dir))
    match = re.search(r"bundle_(\d+)", name)
    if not match:
        raise ValueError(f"Could not infer bundle_id from stage1_dir name: {stage1_dir}")
    return int(match.group(1))


def _find_sample_by_bundle_id(samples, bundle_id):
    matches = [sample for sample in samples if int(sample.get("bundle_id")) == int(bundle_id)]
    if not matches:
        raise ValueError(f"No eval sample found for bundle_id={bundle_id}")
    if len(matches) > 1:
        print(f">>> Warning: found {len(matches)} samples for bundle_id={bundle_id}; using the first.")
    return matches[0]


async def _run(args):
    with open(args.config, "r", encoding="utf-8") as handle:
        conf = yaml.safe_load(handle)

    stage1_dir = os.path.abspath(args.stage1_dir)
    evidence_path = os.path.join(stage1_dir, "evidence.json")
    if not os.path.isfile(evidence_path):
        raise FileNotFoundError(f"Stage 1 evidence not found: {evidence_path}")
    with open(evidence_path, "r", encoding="utf-8") as handle:
        evidence = json.load(handle)

    bundle_id = args.bundle_id if args.bundle_id is not None else _bundle_id_from_stage1_dir(stage1_dir)
    set_seed(int(conf.get("seed", 45)))
    samples = BundleZeroShotDataset(conf).get_eval_samples()
    sample = _find_sample_by_bundle_id(samples, bundle_id)

    client, resolved = _build_prediction_client(conf)
    decision_case = build_decision_case(sample, conf)
    prompt = decision_prompt(decision_case, evidence)
    if args.debug_prompt:
        print_debug("Stage 2 Prediction Prompt", prompt)

    raw = await generate_content_with_retry(
        client,
        resolved["model"],
        prompt,
        conf,
        int(conf.get("code_prediction_max_output_tokens", 300)),
        "stage 2 prediction test",
    )
    prediction = parse_model_response(raw)
    gt_label = sample.get("true_option_char", "")
    hit = int(prediction == gt_label)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(
        ROOT,
        "analysis",
        "stage_2_prediction",
        _safe_path_part(conf.get("dataset")),
        _safe_path_part(resolved["model"]),
        f"bundle_{bundle_id}_{stamp}",
    )
    _write_text(os.path.join(out_dir, "input.txt"), prompt)
    _write_text(os.path.join(out_dir, "output.txt"), raw)
    _write_json(os.path.join(out_dir, "prediction.json"), {
        "bundle_id": int(bundle_id),
        "prediction": prediction,
        "gt_label": gt_label,
        "gt_item_id": sample.get("true_indice", ""),
        "hit": hit,
        "stage1_dir": stage1_dir,
        "evidence_path": evidence_path,
        "provider": resolved["provider"],
        "model": resolved["model"],
        "api_key_env": resolved["api_key_env"],
    })
    _write_json(os.path.join(out_dir, "decision_case.json"), decision_case)
    _write_json(os.path.join(out_dir, "evidence.json"), evidence)

    print(f">>> Config: {args.config}")
    print(f">>> Dataset: {conf['dataset']}")
    print(f">>> Bundle: {bundle_id}")
    print(f">>> Prediction model: {resolved['provider']} / {resolved['model']}")
    print(f">>> Stage 1 source: {stage1_dir}")
    print(f">>> Prediction: {prediction} | GT: {gt_label} | Hit: {hit}")
    print(f">>> Artifacts: {out_dir}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Run Stage 2 prediction from Stage 1 evidence")
    parser.add_argument("--config", default="config_code.yaml")
    parser.add_argument("--stage1_dir", required=True)
    parser.add_argument("--bundle_id", type=int, default=None)
    parser.add_argument("--debug_prompt", action="store_true")
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())

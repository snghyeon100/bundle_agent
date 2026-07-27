"""Run the two-call online hypothesis-program pipeline on one evaluation case.

Usage:
    python tests/test_online_hypothesis_program.py \
        --config config_operator.yaml \
        --split test \
        --sample_idx 1
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

from dataset import BundleZeroShotDataset, set_seed
from main import (
    create_llm_client,
    default_api_key_envs_for_provider,
    generate_content_with_retry,
    resolve_api_key_from_keys,
    stage_model,
    stage_provider,
)
from online_hypothesis_program.pipeline import run_online_hypothesis_program


def _write_json(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=str)


def _write_text(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(str(value or ""))


def _build_client(conf, role):
    provider = stage_provider(conf, role)
    model = stage_model(conf, role)
    api_key, env = resolve_api_key_from_keys(
        conf,
        [
            "online_program_api_key_env"
            if role == "code_generation"
            else "online_prediction_api_key_env",
            "operator_api_key_env",
            "code_generation_api_key_env",
            "code_prediction_api_key_env",
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
    sample_idx = int(args.sample_idx)
    if sample_idx < 0 or sample_idx >= len(samples):
        raise IndexError(
            f"sample_idx {sample_idx} out of range for {len(samples)} {split} samples"
        )
    sample = samples[sample_idx]

    program_client, program_model = _build_client(conf, "code_generation")
    prediction_client, prediction_model = _build_client(conf, "prediction")

    async def call_program(prompt, step_name):
        return await generate_content_with_retry(
            program_client,
            program_model["model"],
            prompt,
            conf,
            int(conf.get("online_program_max_output_tokens", 18000)),
            step_name,
        )

    async def call_prediction(prompt, step_name):
        return await generate_content_with_retry(
            prediction_client,
            prediction_model["model"],
            prompt,
            conf,
            int(conf.get("online_prediction_max_output_tokens", 800)),
            step_name,
        )

    print(f">>> Dataset: {conf['dataset']}")
    print(f">>> Sample: {split}[{sample_idx}] / bundle_{sample['bundle_id']}")
    print(
        f">>> LLM1: {program_model['provider']} / {program_model['model']} "
        "(candidate-blind hypotheses + Python programs)"
    )
    print(">>> Runtime: executing each generated program with scoped SourceAPI")
    print(
        f">>> LLM2: {prediction_model['provider']} / "
        f"{prediction_model['model']} (evidence-aware prediction)"
    )
    result = await run_online_hypothesis_program(
        sample,
        conf,
        call_program,
        call_prediction,
    )

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.abspath(
        args.output_dir
        or os.path.join(
            ROOT,
            "tests",
            "outputs",
            "online_hypothesis_program",
            f"{conf['dataset']}_{timestamp}",
        )
    )
    os.makedirs(output_dir, exist_ok=True)
    _write_json(
        os.path.join(output_dir, "run.json"),
        {
            "phase": "online_hypothesis_program_prediction",
            "dataset": conf["dataset"],
            "split": split,
            "sample_idx": sample_idx,
            "bundle_id": int(sample["bundle_id"]),
            "llm_calls": 2,
            "program_model": program_model,
            "prediction_model": prediction_model,
        },
    )
    _write_json(os.path.join(output_dir, "case.json"), result["case"])
    _write_json(
        os.path.join(output_dir, "source_capabilities.json"),
        result["source_capabilities"],
    )
    _write_text(os.path.join(output_dir, "llm1", "input.txt"), result["llm1"]["prompt"])
    _write_text(
        os.path.join(output_dir, "llm1", "output.txt"),
        result["llm1"]["raw_response"],
    )
    _write_json(
        os.path.join(output_dir, "llm1", "parsed_response.json"),
        result["llm1"]["parsed_response"],
    )
    _write_json(
        os.path.join(output_dir, "llm1", "validation_issues.json"),
        result["llm1"]["validation_issues"],
    )
    parsed = result["llm1"]["parsed_response"]
    if isinstance(parsed, dict):
        for program in parsed.get("programs", []):
            if not isinstance(program, dict):
                continue
            program_id = str(program.get("program_id") or "unknown")
            _write_text(
                os.path.join(output_dir, "programs", program_id, "program.py"),
                program.get("code", ""),
            )
            _write_json(
                os.path.join(output_dir, "programs", program_id, "execution.json"),
                result["executions"].get(program.get("hypothesis_id"), {}),
            )
    _write_json(
        os.path.join(output_dir, "rendered_search_evidence.json"),
        result["rendered_search_evidence"],
    )
    _write_text(os.path.join(output_dir, "llm2", "input.txt"), result["llm2"]["prompt"])
    _write_text(
        os.path.join(output_dir, "llm2", "output.txt"),
        result["llm2"]["raw_response"],
    )
    _write_json(
        os.path.join(output_dir, "llm2", "parsed_response.json"),
        result["llm2"]["parsed_response"],
    )
    _write_json(
        os.path.join(output_dir, "llm2", "validation_issues.json"),
        result["llm2"]["validation_issues"],
    )
    _write_json(os.path.join(output_dir, "evaluation.json"), result["evaluation"])

    evaluation = result["evaluation"]
    print(
        f">>> Programs: {evaluation['successful_program_count']} successful / "
        f"{evaluation['program_count']} generated"
    )
    print(
        f">>> Retrieved exemplars: {evaluation['retrieved_candidate_count']} "
        f"(GT retrieved: {evaluation['ground_truth_retrieved']})"
    )
    print(f">>> Prediction: {evaluation['prediction']}")
    print(f">>> True label: {evaluation['true_label']}")
    print(f">>> Hit: {evaluation['prediction_hit']}")
    print(f">>> Output: {output_dir}")
    return 0 if evaluation["valid_prediction"] else 1


def main():
    parser = argparse.ArgumentParser(
        description="Online multi-hypothesis Python search and prediction"
    )
    parser.add_argument("--config", default="config_operator.yaml")
    parser.add_argument("--split", default="test")
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--output_dir", default="")
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())

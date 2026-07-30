"""Run one code-free macro-strategy induction call for a single sample."""

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
from online_hypothesis_program.pipeline import build_online_case
from online_hypothesis_program.raw_workspace import dataset_workspace_manifest
from online_hypothesis_program.source_api import DatasetSourceAPI
from online_hypothesis_program.strategy_diagnostic import (
    parse_strategy_response,
    strategy_generation_prompt,
    validate_strategy_result,
)


def _write_json(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=str)


def _write_text(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(str(value or ""))


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
    if not 0 <= sample_idx < len(samples):
        raise IndexError(
            f"sample_idx {sample_idx} out of range for {len(samples)} {split} samples"
        )
    sample = samples[sample_idx]

    provider = stage_provider(conf, "code_generation")
    model = stage_model(conf, "code_generation")
    api_key, _ = resolve_api_key_from_keys(
        conf,
        [
            "online_program_api_key_env",
            "operator_api_key_env",
            "code_generation_api_key_env",
            "code_api_key_env",
        ],
        default_api_key_envs_for_provider(provider),
    )
    client = create_llm_client(provider, api_key)
    source_reader = DatasetSourceAPI(conf)
    case = build_online_case(sample, conf, source_reader)
    manifest = dataset_workspace_manifest(conf)
    strategy_count = int(conf.get("online_hypothesis_count", 3))
    prompt = strategy_generation_prompt(
        dataset=conf["dataset"],
        partial_items=case["partial_items"],
        workspace_manifest=manifest,
        strategy_count=strategy_count,
    )

    print(f">>> Dataset: {conf['dataset']}")
    print(f">>> Sample: {split}[{sample_idx}] / bundle_{sample['bundle_id']}")
    print(f">>> Model: {provider} / {model}")
    print(f">>> Strategy-only induction: exactly {strategy_count} hypotheses")
    raw_response = await generate_content_with_retry(
        client,
        model,
        prompt,
        conf,
        int(conf.get("online_strategy_max_output_tokens", 8000)),
        "macro strategy pseudocode induction",
    )
    parsed = parse_strategy_response(raw_response)
    issues = validate_strategy_result(
        parsed,
        available_sources=[
            component["id"] for component in manifest["components"]
        ],
        strategy_count=strategy_count,
        partial_item_ids=[
            item["item_id"] for item in case["partial_items"]
        ],
    )

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.abspath(
        args.output_dir
        or os.path.join(
            ROOT,
            "tests",
            "outputs",
            "online_hypothesis_strategy",
            f"{conf['dataset']}_{timestamp}",
        )
    )
    _write_json(
        os.path.join(output_dir, "run.json"),
        {
            "phase": "online_hypothesis_strategy_only",
            "dataset": conf["dataset"],
            "split": split,
            "sample_idx": sample_idx,
            "bundle_id": int(sample["bundle_id"]),
            "strategy_count": strategy_count,
            "provider": provider,
            "model": model,
            "llm_calls": 1,
        },
    )
    _write_json(os.path.join(output_dir, "case.json"), case)
    _write_text(os.path.join(output_dir, "input.txt"), prompt)
    _write_text(os.path.join(output_dir, "output.txt"), raw_response)
    _write_json(os.path.join(output_dir, "parsed_response.json"), parsed)
    _write_json(os.path.join(output_dir, "validation_issues.json"), issues)

    strategies = (
        parsed.get("strategies", []) if isinstance(parsed, dict) else []
    )
    print(f">>> Strategies: {len(strategies)} generated")
    for strategy in strategies:
        if not isinstance(strategy, dict):
            continue
        print(
            f">>> {strategy.get('id')}: "
            f"{len(strategy.get('pseudocode', []))} stages / "
            f"{', '.join(strategy.get('required_sources', []))}"
        )
        print(f"    Hypothesis: {strategy.get('hypothesis', '')}")
        for step in strategy.get("pseudocode", []):
            print(f"    {step}")
    print(f">>> Valid: {not issues}")
    if issues:
        print(">>> Validation: " + " | ".join(issues[:5]))
    print(f">>> Output: {output_dir}")
    return 0 if not issues else 1


def main():
    parser = argparse.ArgumentParser(
        description="Code-free online macro-strategy induction diagnostic"
    )
    parser.add_argument("--config", default="config_operator.yaml")
    parser.add_argument("--split", default="test")
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--output_dir", default="")
    return asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())

"""Manual live-API runner for Stage 1 of Simple Generate-Evaluate-Decide.

This is intentionally named ``run_...`` rather than ``test_...`` so unittest
discovery never calls a paid external API by accident.
"""

import argparse
import asyncio
import json
import os
import sys
import time


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

import yaml

from dataset import BundleZeroShotDataset, set_seed
from main import (
    console_safe_text,
    create_llm_client,
    default_api_key_envs,
    generate_content_with_retry,
    llm_provider,
    resolve_api_key,
)
from simple_signal_agent import run_simple_signal_stage1


def print_prompt_debug(title, prompt):
    print(f"\n[DEBUG] {title}:")
    print(console_safe_text(prompt))
    print("-" * 60)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Call the real configured LLM for only Stage 1: signal code generation, "
            "execution, optional code repair, and evidence validation."
        )
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--output", default="")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


async def run(args):
    os.chdir(REPO_ROOT)
    with open(args.config, "r", encoding="utf-8") as handle:
        conf = yaml.safe_load(handle)

    set_seed(int(conf.get("seed", 45)))
    samples = BundleZeroShotDataset(conf).get_eval_samples()
    start = int(args.sample_idx)
    limit = int(args.limit)
    if start < 0 or start >= len(samples):
        raise ValueError(f"sample_idx must be between 0 and {len(samples) - 1}.")
    if limit < 1:
        raise ValueError("limit must be at least 1.")
    selected = samples[start : start + limit]

    api_key, api_key_env = resolve_api_key(
        conf,
        "simple_signal_code_api_key_env",
        default_api_key_envs(conf),
    )
    client = create_llm_client(conf, api_key)
    print(f">>> Provider: {llm_provider(conf)}")
    print(f">>> Model: {conf['model']}")
    print(f">>> Code API key env: {api_key_env}")
    print(f">>> Dataset: {conf['dataset']}")
    print(f">>> Stage-1 samples: {len(selected)} (start={start})")

    records = []
    for offset, sample in enumerate(selected):
        sample_index = start + offset
        print(f"\n[{offset + 1}/{len(selected)}] sample_idx={sample_index}, bundle_id={sample['bundle_id']}")
        result = await run_simple_signal_stage1(
            sample,
            conf,
            client,
            generate_content_with_retry,
            debug_callback=print_prompt_debug if args.debug else None,
        )
        record = {"sample_idx": sample_index, **result}
        records.append(record)
        accepted = isinstance(result.get("accepted_evidence"), dict)
        print(f">>> Evidence accepted: {accepted}")
        print(f">>> Code repairs: {len(result.get('code_repairs', []))}")
        if result.get("validation_issues"):
            print(">>> Validation issues:")
            for issue in result["validation_issues"]:
                print(f"    - {issue}")
        if accepted:
            print(">>> Accepted evidence:")
            print(console_safe_text(json.dumps(result["accepted_evidence"], ensure_ascii=False, indent=2)))

    if args.output:
        output_path = os.path.abspath(args.output)
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(
            REPO_ROOT,
            "analysis",
            "simple_signal_stage1",
            f"stage1_{conf['dataset']}_{timestamp}.json",
        )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    payload = {
        "method": "simple_generate_evaluate_decide_stage1_only",
        "dataset": conf["dataset"],
        "model": conf["model"],
        "llm_provider": llm_provider(conf),
        "api_key_env": api_key_env,
        "records": records,
    }
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
    print(f"\nSaved Stage-1 trace to: {output_path}")


def main():
    args = parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()

"""Extract a raw operator pool from existing validation samples.

Usage:
    python tests/test_operator_induction.py --config config_operator.yaml
    python tests/test_operator_induction.py --config config_operator.yaml --sample_count 3
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

from dataset import set_seed
from main import (
    create_llm_client,
    default_api_key_envs_for_provider,
    generate_content_with_retry,
    resolve_api_key_from_keys,
    stage_model,
    stage_provider,
)
from operator_learning.pipeline import (
    induce_raw_operators,
    sample_validation_cases,
)


def _write_json(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=str)


def _write_text(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(str(value or ""))


def _build_client(conf):
    provider = stage_provider(conf, "code_generation")
    model = stage_model(conf, "code_generation")
    api_key, env = resolve_api_key_from_keys(
        conf,
        ["operator_api_key_env", "code_generation_api_key_env", "code_api_key_env"],
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
    sample_count = int(args.sample_count or conf.get("operator_discovery_count", 5))
    operators_per_sample = int(conf.get("operator_induction_count", 4))
    samples = sample_validation_cases(conf, sample_count)
    client, resolved = _build_client(conf)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.abspath(
        args.output_dir
        or os.path.join(
            ROOT,
            "tests",
            "outputs",
            "operators",
            f"{conf['dataset']}_{timestamp}",
        )
    )
    os.makedirs(output_dir, exist_ok=True)
    _write_json(
        os.path.join(output_dir, "run.json"),
        {
            "phase": "operator_induction",
            "dataset": conf["dataset"],
            "validation_source": ["bi_valid_input.txt", "bi_valid_gt.txt"],
            "sample_count": sample_count,
            "operators_per_sample": operators_per_sample,
            "source_manifest_used": False,
            "seed": int(conf.get("operator_discovery_seed", conf.get("seed", 45))),
            **resolved,
        },
    )

    async def call_text(prompt, step_name):
        return await generate_content_with_retry(
            client,
            resolved["model"],
            prompt,
            conf,
            int(conf.get("operator_max_output_tokens", 12000)),
            step_name,
        )

    def save_trace(trace):
        case_dir = os.path.join(output_dir, "cases", trace["case_id"])
        _write_text(os.path.join(case_dir, "input.txt"), trace["prompt"])
        _write_text(os.path.join(case_dir, "output.txt"), trace["raw_response"])
        _write_json(os.path.join(case_dir, "parsed_response.json"), trace["parsed_response"])
        _write_json(os.path.join(case_dir, "validation_issues.json"), trace["validation_issues"])
        _write_json(os.path.join(case_dir, "operators.json"), trace["operators"])

    print(f">>> Dataset: {conf['dataset']}")
    print(">>> Validation source: bi_valid_input.txt + bi_valid_gt.txt")
    print(f">>> Samples: {sample_count}")
    print(f">>> Operators per sample: {operators_per_sample}")
    result = await induce_raw_operators(
        samples,
        conf,
        call_text,
        trace_callback=save_trace,
    )
    _write_json(os.path.join(output_dir, "validation_samples.json"), result["discovery_cases"])
    _write_json(os.path.join(output_dir, "operator_pool.json"), result["raw_operators"])
    operators_by_case = {}
    for operator in result["raw_operators"]:
        operators_by_case.setdefault(operator["origin_case_id"], []).append(operator)
    sample_views = []
    for case in result["discovery_cases"]:
        case_id = case["case_id"]
        sample_view = {
            "sample_id": case_id,
            "bundle_id": case["bundle_id"],
            "input_items": case["partial_items"],
            "candidate_items": case["candidates"],
            "gt_item": case["ground_truth"],
            "operators": operators_by_case.get(case_id, []),
        }
        sample_views.append(sample_view)
        _write_json(os.path.join(output_dir, "samples", f"{case_id}.json"), sample_view)
    _write_json(os.path.join(output_dir, "operators_by_sample.json"), sample_views)
    _write_json(
        os.path.join(output_dir, "summary.json"),
        {
            "validation_sample_count": len(result["discovery_cases"]),
            "operators_per_sample": result["operators_per_case"],
            "raw_operator_count": len(result["raw_operators"]),
        },
    )
    print(f">>> Raw operator count: {len(result['raw_operators'])}")
    print(f">>> Output: {output_dir}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Induce raw operators from validation samples")
    parser.add_argument("--config", default="config_operator.yaml")
    parser.add_argument("--sample_count", type=int, default=None)
    parser.add_argument("--output_dir", default="")
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())

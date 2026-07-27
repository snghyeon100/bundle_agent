"""Extract candidate-blind program specs with one LLM call per discovery case.

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
    build_operator_capability_manifest,
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
    _, _, source_manifest = build_operator_capability_manifest(conf)
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
            "discovery_source": "bi_valid_input.txt",
            "evaluator_only_source": "bi_valid_gt.txt",
            "sample_count": sample_count,
            "max_operators_per_sample": operators_per_sample,
            "llm_calls_per_sample": 1,
            "running_operator_memory_used": True,
            "operator_memory_max_size": int(
                conf.get("operator_memory_max_size", 24)
            ),
            "ground_truth_used_in_prompt": False,
            "candidate_options_used_in_prompt": False,
            "source_manifest_used": True,
            "source_component_ids": [
                component["id"]
                for component in source_manifest.get("components", [])
            ],
            "seed": int(conf.get("operator_discovery_seed", conf.get("seed", 45))),
            **resolved,
        },
    )
    _write_json(
        os.path.join(output_dir, "source_manifest.json"),
        source_manifest,
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
        _write_json(os.path.join(case_dir, "prompt_case.json"), trace["prompt_case"])
        _write_json(os.path.join(case_dir, "evaluation.json"), trace["evaluation"])
        _write_json(
            os.path.join(case_dir, "operator_memory_before.json"),
            trace["operator_memory_before"],
        )
        _write_json(
            os.path.join(case_dir, "operator_memory_after.json"),
            trace["operator_memory_after"],
        )
        _write_json(
            os.path.join(case_dir, "hypotheses.json"),
            trace["hypotheses"],
        )
        _write_json(os.path.join(case_dir, "operators.json"), trace["operators"])

    print(f">>> Dataset: {conf['dataset']}")
    print(">>> Discovery prompt: partial bundle only; candidate options and GT hidden")
    print(f">>> Samples: {sample_count}")
    print(f">>> Maximum operators per sample: {operators_per_sample}")
    print(">>> Induction: one candidate-blind LLM call per sample")
    print(
        ">>> Source components: "
        + ", ".join(
            component["id"]
            for component in source_manifest.get("components", [])
        )
    )
    result = await induce_raw_operators(
        samples,
        conf,
        call_text,
        source_capabilities=source_manifest,
        trace_callback=save_trace,
    )
    _write_json(os.path.join(output_dir, "validation_samples.json"), result["discovery_cases"])
    _write_json(os.path.join(output_dir, "operator_pool.json"), result["raw_operators"])
    _write_json(
        os.path.join(output_dir, "operator_memory.json"),
        result["operator_memory"],
    )
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
            "evaluation": case["evaluation"],
            "operators": operators_by_case.get(case_id, []),
        }
        sample_views.append(sample_view)
        _write_json(os.path.join(output_dir, "samples", f"{case_id}.json"), sample_view)
    _write_json(os.path.join(output_dir, "operators_by_sample.json"), sample_views)
    _write_json(
        os.path.join(output_dir, "summary.json"),
        {
            "validation_sample_count": len(result["discovery_cases"]),
            "max_operators_per_sample": result["max_operators_per_case"],
            "llm_calls_per_sample": 1,
            "raw_operator_count": len(result["raw_operators"]),
            "final_operator_memory_count": len(result["operator_memory"]),
            "operator_memory_max_size": result["operator_memory_max_size"],
        },
    )
    print(f">>> Raw operator count: {len(result['raw_operators'])}")
    print(f">>> Output: {output_dir}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Induce compact source-aware operators from validation samples"
    )
    parser.add_argument("--config", default="config_operator.yaml")
    parser.add_argument("--sample_count", type=int, default=None)
    parser.add_argument("--output_dir", default="")
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())

"""Generate three hyperedge-only spec-first strategies per validation case.

Usage:
    python tests/test_hyperedge_strategy_induction.py \
        --config config_operator.yaml \
        --sample_count 5 \
        --output_dir tests/outputs/hyperedge_strategies/prompt_test
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time

import yaml


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from dataset import set_seed
from hyperedge_strategy.prompts import induction_prompt as hyperedge_induction_prompt
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
        [
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


async def _run(args):
    with open(args.config, "r", encoding="utf-8") as handle:
        conf = yaml.safe_load(handle)
    set_seed(int(conf.get("seed", 45)))
    sample_count = int(args.sample_count or conf.get("operator_discovery_count", 5))
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
            "hyperedge_strategies",
            f"{conf['dataset']}_{timestamp}",
        )
    )
    os.makedirs(output_dir, exist_ok=True)
    _write_json(
        os.path.join(output_dir, "run.json"),
        {
            "phase": "hyperedge_strategy_induction",
            "dataset": conf["dataset"],
            "sample_count": sample_count,
            "strategies_per_sample": 3,
            "llm_calls_per_sample": 1,
            "code_generation_in_same_call": True,
            "ground_truth_used_in_prompt": False,
            "candidate_options_used_in_prompt": True,
            "all_strategies_hyperedge_conditioned": True,
            "source_component_ids": [
                component["id"]
                for component in source_manifest.get("components", [])
            ],
            "seed": int(conf.get("operator_discovery_seed", conf.get("seed", 45))),
            **resolved,
        },
    )
    _write_json(os.path.join(output_dir, "source_manifest.json"), source_manifest)

    async def call_text(prompt, step_name):
        return await generate_content_with_retry(
            client,
            resolved["model"],
            prompt,
            conf,
            int(conf.get("operator_max_output_tokens", 15000)),
            step_name,
        )

    def save_trace(trace):
        case_dir = os.path.join(output_dir, "cases", trace["case_id"])
        _write_text(os.path.join(case_dir, "input.txt"), trace["prompt"])
        _write_text(os.path.join(case_dir, "output.txt"), trace["raw_response"])
        _write_json(
            os.path.join(case_dir, "parsed_response.json"),
            trace["parsed_response"],
        )
        _write_json(
            os.path.join(case_dir, "validation_issues.json"),
            trace["validation_issues"],
        )
        _write_json(
            os.path.join(case_dir, "prompt_case.json"),
            trace["prompt_case"],
        )
        _write_json(
            os.path.join(case_dir, "evaluation.json"),
            trace["evaluation"],
        )
        _write_json(
            os.path.join(case_dir, "strategy_specs.json"),
            trace["strategy_specs"],
        )
        _write_json(
            os.path.join(case_dir, "programs.json"),
            trace["programs"],
        )
        _write_json(
            os.path.join(case_dir, "strategies.json"),
            trace["strategies"],
        )
        for index, strategy in enumerate(trace["strategies"], start=1):
            code = strategy.get("code") if isinstance(strategy, dict) else None
            if not isinstance(code, str):
                continue
            safe_name = re.sub(
                r"[^A-Za-z0-9_-]+",
                "_",
                str(strategy.get("name") or f"strategy_{index}"),
            ).strip("_")
            _write_text(
                os.path.join(
                    case_dir,
                    "programs",
                    f"{index:02d}_{safe_name or f'strategy_{index}'}.py",
                ),
                code,
            )

    print(f">>> Dataset: {conf['dataset']}")
    print(">>> Prompt variant: three higher-order hyperedge strategies")
    print(">>> Case: partial bundle + candidate items; GT identity hidden")
    print(f">>> Samples: {sample_count}")
    print(">>> Strategies per sample: 3")
    print(">>> Induction: one spec-first strategy + Python code LLM call per sample")
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
        operators_per_case=3,
        trace_callback=save_trace,
        prompt_builder=hyperedge_induction_prompt,
    )
    _write_json(
        os.path.join(output_dir, "validation_samples.json"),
        result["discovery_cases"],
    )
    _write_json(
        os.path.join(output_dir, "operator_pool.json"),
        result["raw_operators"],
    )
    generated_code_count = sum(
        isinstance(operator.get("generated_code"), str)
        and bool(operator["generated_code"].strip())
        for operator in result["raw_operators"]
    )
    _write_json(
        os.path.join(output_dir, "summary.json"),
        {
            "validation_sample_count": len(result["discovery_cases"]),
            "strategies_per_sample": 3,
            "llm_calls_per_sample": 1,
            "raw_operator_count": len(result["raw_operators"]),
            "generated_code_count": generated_code_count,
        },
    )
    print(f">>> Raw operator count: {len(result['raw_operators'])}")
    print(f">>> Generated Python programs: {generated_code_count}")
    print(f">>> Output: {output_dir}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Generate three hyperedge-only spec-first strategies"
    )
    parser.add_argument("--config", default="config_operator.yaml")
    parser.add_argument("--sample_count", type=int, default=None)
    parser.add_argument("--output_dir", default="")
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())

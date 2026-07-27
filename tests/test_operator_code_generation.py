"""Compile a deduplicated candidate-program library offline.

Usage:
    python tests/test_operator_code_generation.py \
        --config config_operator.yaml \
        --library tests/outputs/dedup/pog_dense_<date>/operator_library.json

This stage does not receive a test sample, answer options, or ground truth. It
compiles each unique program once so validation and online inference can reuse
the exact same code hash.
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
    compile_operator_programs,
    load_operator_library,
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
    }, {"provider": provider, "model": model, "api_key_env": env}


async def _run(args):
    with open(args.config, "r", encoding="utf-8") as handle:
        conf = yaml.safe_load(handle)
    library = load_operator_library(os.path.abspath(args.library))
    _, _, source_capabilities = build_operator_capability_manifest(conf)
    client, resolved = _build_client(conf)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.abspath(
        args.output_dir
        or os.path.join(
            ROOT,
            "tests",
            "outputs",
            "compiled_programs",
            f"{conf['dataset']}_{timestamp}",
        )
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
        program_dir = os.path.join(output_dir, trace["operator_name"])
        _write_text(os.path.join(program_dir, "input.txt"), trace["prompt"])
        _write_text(os.path.join(program_dir, "output.txt"), trace["raw_response"])
        _write_json(
            os.path.join(program_dir, "parsed_response.json"),
            trace["parsed_response"],
        )
        _write_json(
            os.path.join(program_dir, "validation_issues.json"),
            trace["validation_issues"],
        )
        if isinstance(trace["parsed_response"], dict):
            _write_text(
                os.path.join(program_dir, "program.py"),
                trace["parsed_response"].get("code", ""),
            )

    print(f">>> Dataset: {conf['dataset']}")
    print(f">>> Library: {os.path.abspath(args.library)}")
    print(f">>> Model: {resolved['provider']} / {resolved['model']}")
    print(f">>> Unique programs: {len(library['operators'])}")
    print(">>> Compilation: one LLM call per unique program; no sample or GT")
    result = await compile_operator_programs(
        library,
        conf,
        call_text,
        source_capabilities=source_capabilities,
        trace_callback=save_trace,
    )
    _write_json(
        os.path.join(output_dir, "compiled_program_library.json"),
        result["compiled_library"],
    )
    _write_json(
        os.path.join(output_dir, "run.json"),
        {
            "phase": "offline_candidate_program_compilation",
            "dataset": conf["dataset"],
            "library": os.path.abspath(args.library),
            "program_count": len(result["compiled_library"]["programs"]),
            "llm_calls": len(result["compiled_library"]["programs"]),
            "sample_used": False,
            "ground_truth_used": False,
            **resolved,
        },
    )
    print(f">>> Compiled programs: {len(result['compiled_library']['programs'])}")
    print(f">>> Output: {output_dir}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Compile reusable candidate-retrieval programs offline"
    )
    parser.add_argument("--config", default="config_operator.yaml")
    parser.add_argument("--library", required=True)
    parser.add_argument("--output_dir", default="")
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())

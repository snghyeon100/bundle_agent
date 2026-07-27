"""Offline candidate-program induction, deduplication, and compilation CLI."""

import argparse
import asyncio
import os
import time

import yaml

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
    discover_operator_library,
    sample_validation_cases,
    save_operator_artifacts,
)


def _safe_path_part(value):
    text = str(value or "unknown").strip()
    return "".join(
        character
        if character.isalnum() or character in ("-", "_", ".")
        else "_"
        for character in text
    )


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
    }, {"api_key_env": env, "provider": provider, "model": model}


async def _run(args):
    with open(args.config, "r", encoding="utf-8") as handle:
        conf = yaml.safe_load(handle)
    set_seed(int(conf.get("seed", 45)))
    client, resolved = _build_client(conf)
    _, _, source_capabilities = build_operator_capability_manifest(conf)

    async def call_text(prompt, step_name):
        return await generate_content_with_retry(
            client,
            resolved["model"],
            prompt,
            conf,
            int(conf.get("operator_max_output_tokens", 12000)),
            step_name,
        )

    count = args.discovery_count or int(
        conf.get("operator_discovery_count", 5)
    )
    samples = sample_validation_cases(conf, count)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or os.path.join(
        "analysis",
        "operator_mvp",
        _safe_path_part(conf["dataset"]),
        stamp,
    )

    print(f">>> Dataset: {conf['dataset']}")
    print(f">>> Model: {resolved['provider']} / {resolved['model']}")
    print(f">>> Candidate-blind discovery cases: {len(samples)}")
    print(">>> Calls: one induction call per case, then one compilation call per unique program")
    discovery = await discover_operator_library(
        samples,
        conf,
        call_text,
        source_capabilities=source_capabilities,
    )
    save_operator_artifacts(output_dir, discovery=discovery)
    print(f">>> Raw program specs: {len(discovery['raw_operators'])}")
    print(f">>> Unique program specs: {len(discovery['library']['operators'])}")
    print(f">>> Compiled programs: {len(discovery['compiled_library']['programs'])}")
    print(f">>> Artifacts: {os.path.abspath(output_dir)}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Induce and compile reusable candidate-retrieval programs"
    )
    parser.add_argument("--config", default="config_operator.yaml")
    parser.add_argument("--discovery_count", type=int, default=None)
    parser.add_argument("--output_dir", default="")
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())

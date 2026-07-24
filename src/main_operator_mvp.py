"""Source-aware compact operator discovery and workflow composition CLI.

Examples:
    python src/main_operator_mvp.py --config config_operator.yaml --phase all
    python src/main_operator_mvp.py --config config_operator.yaml --phase discover
    python src/main_operator_mvp.py --config config_operator.yaml --phase compose \
        --library analysis/operator_mvp/pog_dense/<run>/operator_library.json
"""

import argparse
import asyncio
import os
import time

import yaml

from dataset import BundleZeroShotDataset, set_seed
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
    compose_workflows,
    discover_operator_library,
    load_operator_library,
    sample_validation_cases,
    save_operator_artifacts,
)


def _safe_path_part(value):
    text = str(value or "unknown").strip()
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in text)


def _build_client(conf):
    provider = stage_provider(conf, "code_generation")
    model = stage_model(conf, "code_generation")
    api_key, env = resolve_api_key_from_keys(
        conf,
        ["operator_api_key_env", "code_generation_api_key_env", "code_api_key_env"],
        default_api_key_envs_for_provider(provider),
    )
    client = {
        "client": create_llm_client(provider, api_key),
        "provider": provider,
        "model": model,
    }
    return client, {"api_key_env": env, "provider": provider, "model": model}


async def _run(args):
    with open(args.config, "r", encoding="utf-8") as handle:
        conf = yaml.safe_load(handle)
    set_seed(int(conf.get("seed", 45)))
    client, resolved = _build_client(conf)
    _, source_manifest, source_capabilities = build_operator_capability_manifest(conf)

    async def call_text(prompt, step_name):
        return await generate_content_with_retry(
            client,
            resolved["model"],
            prompt,
            conf,
            int(conf.get("operator_max_output_tokens", 12000)),
            step_name,
        )

    stamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or os.path.join(
        "analysis",
        "operator_mvp",
        _safe_path_part(conf["dataset"]),
        stamp,
    )
    discovery = None
    composition = None
    library = None

    print(f">>> Dataset: {conf['dataset']}")
    print(f">>> Model: {resolved['provider']} / {resolved['model']}")
    print(">>> Discovery source: existing bi_valid_input.txt + bi_valid_gt.txt")

    if args.phase in {"discover", "all"}:
        count = args.discovery_count or int(conf.get("operator_discovery_count", 5))
        samples = sample_validation_cases(conf, count)
        print(f">>> Inducing operators from {len(samples)} validation samples")
        discovery = await discover_operator_library(
            samples,
            conf,
            call_text,
            source_capabilities=source_capabilities,
        )
        library = discovery["library"]
        print(
            f">>> Raw operators: {len(discovery['raw_operators'])} | "
            f"Refined library: {len(library['operators'])}"
        )

    if args.phase == "compose":
        if not args.library:
            raise ValueError("--library is required for --phase compose")
        library = load_operator_library(args.library)

    if args.phase in {"compose", "all"}:
        eval_conf = dict(conf)
        eval_conf["toy_eval"] = -1
        samples = BundleZeroShotDataset(eval_conf, split=args.compose_split).get_eval_samples()
        if args.sample_idx < 0 or args.sample_idx >= len(samples):
            raise IndexError(f"sample_idx {args.sample_idx} out of range for {len(samples)} samples")
        print(
            f">>> Composing {int(conf.get('operator_workflow_count', 3))} workflows for "
            f"{args.compose_split} sample {args.sample_idx}"
        )
        composition = await compose_workflows(
            samples[args.sample_idx],
            conf,
            source_manifest,
            library,
            call_text,
        )
        print(f">>> Recommended workflow: {composition['result']['recommended_workflow']}")

    save_operator_artifacts(output_dir, discovery=discovery, composition=composition)
    print(f">>> Artifacts: {os.path.abspath(output_dir)}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Run compact operator learning MVP")
    parser.add_argument("--config", default="config_operator.yaml")
    parser.add_argument("--phase", choices=("discover", "compose", "all"), default="all")
    parser.add_argument("--library", default="")
    parser.add_argument("--discovery_count", type=int, default=None)
    parser.add_argument("--compose_split", choices=("valid", "test"), default="test")
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--output_dir", default="")
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())

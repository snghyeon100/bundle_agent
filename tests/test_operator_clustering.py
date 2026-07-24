"""Cluster operator_pool.json produced by test_operator_induction.py.

Usage:
    python tests/test_operator_clustering.py --config config_operator.yaml
    python tests/test_operator_clustering.py --config config_operator.yaml \
        --operator_pool tests/outputs/operators/pog_dense_<date>/operator_pool.json
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
from operator_learning.pipeline import cluster_raw_operators


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


def _latest_operator_pool(dataset):
    root = os.path.join(ROOT, "tests", "outputs", "operators")
    if not os.path.isdir(root):
        raise FileNotFoundError(f"operator output root not found: {root}")
    paths = []
    for name in os.listdir(root):
        path = os.path.join(root, name, "operator_pool.json")
        if name.startswith(f"{dataset}_") and os.path.isfile(path):
            paths.append(path)
    if not paths:
        raise FileNotFoundError(
            f"no operator_pool.json found for {dataset}; run test_operator_induction.py first"
        )
    return max(paths, key=os.path.getmtime)


async def _run(args):
    with open(args.config, "r", encoding="utf-8") as handle:
        conf = yaml.safe_load(handle)
    operator_pool_path = os.path.abspath(
        args.operator_pool or _latest_operator_pool(conf["dataset"])
    )
    with open(operator_pool_path, "r", encoding="utf-8") as handle:
        operator_pool = json.load(handle)
    client, resolved = _build_client(conf)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.abspath(
        args.output_dir
        or os.path.join(
            ROOT,
            "tests",
            "outputs",
            "cluster",
            f"{conf['dataset']}_{timestamp}",
        )
    )
    os.makedirs(output_dir, exist_ok=True)
    _write_json(
        os.path.join(output_dir, "run.json"),
        {
            "phase": "operator_clustering",
            "dataset": conf["dataset"],
            "operator_pool_path": operator_pool_path,
            "raw_operator_count": len(operator_pool) if isinstance(operator_pool, list) else None,
            "library_min_size": int(conf.get("operator_library_min_size", 8)),
            "library_max_size": int(conf.get("operator_library_max_size", 12)),
            "source_manifest_used": False,
            "source_lists_preserved_from_operator_pool": True,
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
        _write_text(os.path.join(output_dir, "input.txt"), trace["clustering_prompt"])
        _write_text(os.path.join(output_dir, "output.txt"), trace["clustering_raw_response"])
        _write_json(os.path.join(output_dir, "parsed_response.json"), trace["parsed_response"])
        _write_json(os.path.join(output_dir, "validation_issues.json"), trace["validation_issues"])
        _write_json(os.path.join(output_dir, "operator_library.json"), trace["library"])

    print(f">>> Dataset: {conf['dataset']}")
    print(f">>> Operator pool: {operator_pool_path}")
    print(f">>> Raw operator count: {len(operator_pool)}")
    result = await cluster_raw_operators(
        operator_pool,
        conf,
        call_text,
        trace_callback=save_trace,
    )
    _write_json(
        os.path.join(output_dir, "summary.json"),
        {
            "raw_operator_count": len(operator_pool),
            "cluster_count": len(result["library"]["clusters"]),
            "refined_operator_count": len(result["library"]["operators"]),
        },
    )
    print(f">>> Clusters: {len(result['library']['clusters'])}")
    print(f">>> Refined operators: {len(result['library']['operators'])}")
    print(f">>> Output: {output_dir}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Cluster a saved raw operator pool")
    parser.add_argument("--config", default="config_operator.yaml")
    parser.add_argument("--operator_pool", default="")
    parser.add_argument("--output_dir", default="")
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())

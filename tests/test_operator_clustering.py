"""Deterministically deduplicate an induced candidate-program pool.

Usage:
    python tests/test_operator_clustering.py --config config_operator.yaml
    python tests/test_operator_clustering.py --config config_operator.yaml \
        --operator_pool tests/outputs/operators/pog_dense_<date>/operator_pool.json
"""

import argparse
import json
import os
import sys
import time

import yaml


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from operator_learning.pipeline import deduplicate_raw_operators


def _write_json(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=str)


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
            f"no operator_pool.json found for {dataset}; "
            "run test_operator_induction.py first"
        )
    return max(paths, key=os.path.getmtime)


def _run(args):
    with open(args.config, "r", encoding="utf-8") as handle:
        conf = yaml.safe_load(handle)
    operator_pool_path = os.path.abspath(
        args.operator_pool or _latest_operator_pool(conf["dataset"])
    )
    with open(operator_pool_path, "r", encoding="utf-8") as handle:
        operator_pool = json.load(handle)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.abspath(
        args.output_dir
        or os.path.join(
            ROOT,
            "tests",
            "outputs",
            "dedup",
            f"{conf['dataset']}_{timestamp}",
        )
    )
    result = deduplicate_raw_operators(operator_pool, conf)
    _write_json(
        os.path.join(output_dir, "run.json"),
        {
            "phase": "deterministic_operator_deduplication",
            "dataset": conf["dataset"],
            "operator_pool_path": operator_pool_path,
            "raw_operator_count": len(operator_pool),
            "llm_calls": 0,
            "similarity_threshold": result["deduplication"][
                "similarity_threshold"
            ],
        },
    )
    _write_json(
        os.path.join(output_dir, "deduplication.json"),
        result["deduplication"],
    )
    _write_json(
        os.path.join(output_dir, "operator_library.json"),
        result["library"],
    )

    print(f">>> Dataset: {conf['dataset']}")
    print(f">>> Operator pool: {operator_pool_path}")
    print(f">>> Raw program specs: {len(operator_pool)}")
    print(">>> Deduplication LLM calls: 0")
    print(f">>> Unique program specs: {len(result['library']['operators'])}")
    print(f">>> Output: {output_dir}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Deterministically deduplicate a candidate-program pool"
    )
    parser.add_argument("--config", default="config_operator.yaml")
    parser.add_argument("--operator_pool", default="")
    parser.add_argument("--output_dir", default="")
    args = parser.parse_args()
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())

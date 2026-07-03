"""Run only Stage 1 code generation/execution for quick inspection.

Usage:
    .\\.venv\\Scripts\\python.exe tests\\stage_1_code_generation\\run.py --config config_code.yaml
"""

import argparse
import asyncio
import json
import os
import sys
import time

import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from code.pipeline import build_code_generation_inputs, generate_code_evidence_once
from dataset import BundleZeroShotDataset, set_seed
from main import (
    create_llm_client,
    default_api_key_envs_for_provider,
    generate_content_with_retry,
    print_debug,
    resolve_api_key_from_keys,
    stage_model,
    stage_provider,
)


def _write_text(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(str(text or ""))


def _write_json(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=str)


def _build_code_generation_client(conf):
    provider = stage_provider(conf, "code_generation")
    model = stage_model(conf, "code_generation")
    api_key, env = resolve_api_key_from_keys(
        conf,
        [
            "code_generation_api_key_env",
            "code_api_key_env",
            "sem_evidence_api_key_env",
            "sem_stage1_api_key_env",
            "sem_code_api_key_env",
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
    samples = BundleZeroShotDataset(conf).get_eval_samples()
    if args.sample_idx < 0 or args.sample_idx >= len(samples):
        raise IndexError(f"sample_idx {args.sample_idx} out of range for {len(samples)} samples")
    sample = samples[args.sample_idx]

    client, resolved = _build_code_generation_client(conf)
    print(f">>> Config: {args.config}")
    print(f">>> Dataset: {conf['dataset']}")
    print(f">>> Bundle: {sample['bundle_id']}")
    print(f">>> Code generation model: {resolved['provider']} / {resolved['model']}")

    inputs = build_code_generation_inputs(sample, conf)
    if args.debug_prompt:
        print_debug("Stage 1 Code Generation Prompt", inputs["prompt"])

    result = await generate_code_evidence_once(
        bundle_id=sample["bundle_id"],
        case_view=inputs["case_view"],
        source_manifest=inputs["source_manifest"],
        initial_prompt=inputs["prompt"],
        client=client,
        conf=conf,
        generate_content_fn=generate_content_with_retry,
        workspace=inputs["workspace"],
        output_file=inputs["evidence_output_file"],
        semantic_case=inputs["decision_case"],
    )

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(
        ROOT,
        "analysis",
        "stage_1_code_generation",
        f"bundle_{sample['bundle_id']}_{stamp}",
    )
    _write_text(os.path.join(out_dir, "input.txt"), result["prompt"])
    _write_text(os.path.join(out_dir, "output.txt"), result["raw_response"])
    _write_text(os.path.join(out_dir, "code.py"), result["generated_code"])
    _write_json(os.path.join(out_dir, "execution_summary.json"), result["execution_summary"])
    if result["accepted_evidence"] is not None:
        _write_json(os.path.join(out_dir, "evidence.json"), result["execution_result"].get("evidence_json"))

    print(f">>> Artifacts: {out_dir}")
    if result["accepted_evidence"] is None:
        print(">>> Stage 1 FAILED: generated code did not execute into parseable JSON.")
        return 1

    print(">>> Stage 1 OK: generated code executed and produced parseable JSON.")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Run Stage 1 code generation only")
    parser.add_argument("--config", default="config_code.yaml")
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--debug_prompt", action="store_true")
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())

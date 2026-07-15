"""Run only Stage 1 code generation/execution for quick inspection.

Usage:
    .\\.venv\\Scripts\\python.exe tests\\stage_1_code_generation\\run.py --config config_code.yaml
    .\\.venv\\Scripts\\python.exe tests\\stage_1_code_generation\\run.py --sample_idx 0 --count 5
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
        ],
        default_api_key_envs_for_provider(provider),
    )
    return {
        "client": create_llm_client(provider, api_key),
        "provider": provider,
        "model": model,
    }, {"api_key_env": env, "provider": provider, "model": model}


def _safe_path_part(value):
    text = str(value or "unknown").strip()
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in text) or "unknown"


async def _run(args):
    with open(args.config, "r", encoding="utf-8") as handle:
        conf = yaml.safe_load(handle)

    set_seed(int(conf.get("seed", 45)))
    samples = BundleZeroShotDataset(conf).get_eval_samples()
    if args.sample_idx < 0 or args.sample_idx >= len(samples):
        raise IndexError(f"sample_idx {args.sample_idx} out of range for {len(samples)} samples")
    if args.count < 1:
        raise ValueError(f"count must be at least 1, got {args.count}")

    end_idx = min(args.sample_idx + args.count, len(samples))
    selected = list(enumerate(samples[args.sample_idx:end_idx], start=args.sample_idx))
    if len(selected) < args.count:
        print(
            f">>> Warning: requested {args.count} samples from index {args.sample_idx}, "
            f"but only {len(selected)} are available."
        )

    client, resolved = _build_code_generation_client(conf)

    dataset_group = _safe_path_part(conf.get("dataset"))
    model_group = _safe_path_part(resolved["model"])

    print(f">>> Config: {args.config}")
    print(f">>> Dataset: {conf['dataset']}")
    print(f">>> Code generation model: {resolved['provider']} / {resolved['model']}")
    print(
        f">>> Samples: indices {args.sample_idx}..{end_idx - 1} "
        f"({len(selected)} total)"
    )

    succeeded = 0
    failed = 0
    for position, (sample_idx, sample) in enumerate(selected, start=1):
        sample_conf = dict(conf)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(
            ROOT,
            "analysis",
            "stage_1_code_generation",
            dataset_group,
            model_group,
            f"bundle_{sample['bundle_id']}_{stamp}",
        )
        if os.path.exists(out_dir):
            out_dir += f"_sample{sample_idx}"
        sample_conf["code_workspace_root"] = os.path.join(out_dir, "workspaces")
        sample_conf["output_dir"] = os.path.join(out_dir, "results")

        print(f">>> [{position}/{len(selected)}] Sample index: {sample_idx}")
        print(f">>> Bundle: {sample['bundle_id']}")
        print(f">>> Test output root: {out_dir}")

        inputs = build_code_generation_inputs(sample, sample_conf)
        if args.debug_prompt:
            print_debug("Stage 1 Code Generation Prompt", inputs["prompt"])

        result = await generate_code_evidence_once(
            bundle_id=sample["bundle_id"],
            case_view=inputs["case_view"],
            source_manifest=inputs["source_manifest"],
            initial_prompt=inputs["prompt"],
            client=client,
            conf=sample_conf,
            generate_content_fn=generate_content_with_retry,
            workspace=inputs["workspace"],
            output_file=inputs["evidence_output_file"],
            semantic_case=inputs["decision_case"],
        )

        _write_text(os.path.join(out_dir, "input.txt"), result["prompt"])
        _write_text(os.path.join(out_dir, "output.txt"), result["raw_response"])
        _write_text(os.path.join(out_dir, "code.py"), result["generated_code"])
        _write_json(os.path.join(out_dir, "execution_summary.json"), result["execution_summary"])
        if result["accepted_evidence"] is not None:
            _write_json(
                os.path.join(out_dir, "evidence.json"),
                result["execution_result"].get("evidence_json"),
            )

        print(f">>> Artifacts: {out_dir}")
        if result["accepted_evidence"] is None:
            failed += 1
            issues = result.get("validation_issues", [])
            if issues:
                print(">>> Validation issues: " + " | ".join(str(issue) for issue in issues))
            print(">>> Stage 1 FAILED: generated code did not produce valid adaptive bundle evidence.")
        else:
            succeeded += 1
            print(">>> Stage 1 OK: generated code executed and produced valid adaptive bundle evidence.")

    print(f">>> Summary: {succeeded} succeeded, {failed} failed, {len(selected)} total")
    return 1 if failed else 0


def main():
    parser = argparse.ArgumentParser(description="Run Stage 1 code generation only")
    parser.add_argument("--config", default="config_code.yaml")
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of consecutive samples to run, starting at --sample_idx (default: 1)",
    )
    parser.add_argument("--debug_prompt", action="store_true")
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())

import argparse
import asyncio
import json
import os
import time

import yaml

from agents.common import build_agent_sample_view, compact_json
from agents.workspace import prepare_agent_workspace
from dataset import BundleZeroShotDataset, set_seed
from main import (
    create_llm_client,
    default_api_key_envs,
    generate_content_with_retry,
    llm_provider,
    resolve_api_key,
)
from three_stage_agent.pipeline import (
    call_stage,
    run_deep_observation_with_repairs,
    run_exploratory_retrieval_with_repairs,
    summarize_execution,
)
from three_stage_agent.prompts import (
    generate_deep_observation_prompt,
    generate_exploratory_retrieval_prompt,
)


def stage1_api_key_fallback_envs(conf):
    envs = []
    for config_key in ("three_stage_code_api_key_env", "prediction_api_key_env"):
        env_name = str(conf.get(config_key, "")).strip()
        if env_name and env_name not in envs:
            envs.append(env_name)
    for env_name in default_api_key_envs(conf):
        if env_name not in envs:
            envs.append(env_name)
    return envs


def stage1_output_path(output_dir, dataset_name, bundle_id, sample_idx, timestamp):
    filename = f"stage1_{dataset_name}_idx{sample_idx}_bundle{bundle_id}_{timestamp}.json"
    return os.path.join(output_dir, filename)


async def run_stage1_for_sample(
    sample,
    sample_idx,
    conf,
    client,
    output_dir,
    timestamp,
    show_prompt,
    skip_deep,
):
    sample = dict(sample)
    sample["dataset"] = conf.get("dataset", "")
    workspace = prepare_agent_workspace(conf)
    evidence_output_file = f"output/stage1_surface_evidence_bundle{sample['bundle_id']}.json"
    deep_evidence_output_file = f"output/stage1_deep_evidence_bundle{sample['bundle_id']}.json"

    retrieval_prompt = generate_exploratory_retrieval_prompt(
        sample,
        workspace,
        evidence_output_file,
        conf,
    )
    if show_prompt:
        print("\n" + "=" * 80)
        print(f"Stage 1 prompt for sample_idx={sample_idx}, bundle_id={sample['bundle_id']}")
        print("=" * 80)
        print(retrieval_prompt)

    retrieval_raw_text = await call_stage(
        generate_content_with_retry,
        client,
        conf,
        retrieval_prompt,
        "three_stage_code_max_output_tokens",
        3600,
        f"sample {sample_idx} stage1 exploratory retrieval",
    )
    retrieval_result = await run_exploratory_retrieval_with_repairs(
        sample,
        conf,
        client,
        generate_content_with_retry,
        retrieval_prompt,
        retrieval_raw_text,
        workspace,
        evidence_output_file,
    )

    execution_result = retrieval_result["execution_result"]
    execution_summary = summarize_execution(execution_result)
    deep_result = None
    deep_execution_result = None
    deep_execution_summary = None
    if not skip_deep:
        deep_prompt = generate_deep_observation_prompt(
            sample,
            workspace,
            execution_result.get("evidence_json"),
            execution_summary,
            deep_evidence_output_file,
            conf,
        )
        if show_prompt:
            print("\n" + "=" * 80)
            print(f"Stage 1B deep prompt for sample_idx={sample_idx}, bundle_id={sample['bundle_id']}")
            print("=" * 80)
            print(deep_prompt)
        deep_raw_text = await call_stage(
            generate_content_with_retry,
            client,
            conf,
            deep_prompt,
            "three_stage_deep_code_max_output_tokens",
            3600,
            f"sample {sample_idx} stage1 deep observation",
        )
        deep_result = await run_deep_observation_with_repairs(
            sample,
            conf,
            client,
            generate_content_with_retry,
            deep_prompt,
            deep_raw_text,
            workspace,
            execution_result.get("evidence_json"),
            execution_summary,
            deep_evidence_output_file,
        )
        deep_execution_result = deep_result["execution_result"]
        deep_execution_summary = summarize_execution(deep_execution_result)

    record = {
        "sample_idx": sample_idx,
        "bundle_id": sample["bundle_id"],
        "sample": build_agent_sample_view(sample),
        "workspace_dir": workspace["workspace_dir"],
        "data_dir": workspace["data_dir"],
        "workspace_files": workspace["files"],
        "evidence_output_file": evidence_output_file,
        "retrieval_raw_response": retrieval_raw_text,
        "retrieval_repair_raw_responses": retrieval_result["retrieval_repair_raw_responses"],
        "retrieval_repair_attempts_used": retrieval_result["retrieval_repair_attempts_used"],
        "generated_code": retrieval_result["generated_code"],
        "execution_summary": execution_summary,
        "evidence_json": execution_result.get("evidence_json"),
        "evidence_text": execution_result.get("evidence_text", ""),
        "deep_evidence_output_file": deep_evidence_output_file if not skip_deep else "",
        "deep_raw_response": deep_result["deep_raw_response"] if deep_result else "",
        "deep_repair_raw_responses": deep_result["deep_repair_raw_responses"] if deep_result else [],
        "deep_repair_attempts_used": deep_result["deep_repair_attempts_used"] if deep_result else 0,
        "deep_generated_code": deep_result["deep_generated_code"] if deep_result else "",
        "deep_execution_summary": deep_execution_summary,
        "deep_evidence_json": deep_execution_result.get("evidence_json") if deep_execution_result else None,
        "deep_evidence_text": deep_execution_result.get("evidence_text", "") if deep_execution_result else "",
    }

    os.makedirs(output_dir, exist_ok=True)
    output_path = stage1_output_path(
        output_dir,
        conf.get("dataset", "dataset"),
        sample["bundle_id"],
        sample_idx,
        timestamp,
    )
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2)

    print(
        f"[Stage1] idx={sample_idx} bundle={sample['bundle_id']} "
        f"returncode={record['execution_summary']['returncode']} "
        f"evidence_json={record['execution_summary']['evidence_json_present']} "
        f"repairs={record['retrieval_repair_attempts_used']} "
        f"deep_json={bool(record['deep_evidence_json'])} "
        f"deep_repairs={record['deep_repair_attempts_used']} "
        f"saved={output_path}"
    )
    return record


async def async_main(args):
    with open(args.config, "r", encoding="utf-8") as handle:
        conf = yaml.safe_load(handle)

    set_seed(int(conf.get("seed", 45)))
    dataset = BundleZeroShotDataset(conf)
    samples = dataset.get_eval_samples()
    if args.sample_idx < 0 or args.sample_idx >= len(samples):
        raise ValueError(f"sample_idx out of range: {args.sample_idx}; available 0..{len(samples) - 1}")

    output_dir = args.output_dir or conf.get("stage1_output_dir", "./analysis/stage1")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    api_key, api_key_env = resolve_api_key(
        conf,
        "stage1_api_key_env",
        stage1_api_key_fallback_envs(conf),
    )
    client = create_llm_client(conf, api_key)

    print(f">>> Loaded config: {args.config}")
    print(f">>> Dataset: {conf['dataset']}")
    print(f">>> LLM provider: {llm_provider(conf)}")
    print(f">>> Stage 1 API key env: {api_key_env}")
    print(f">>> Output dir: {output_dir}")

    selected = samples[args.sample_idx : args.sample_idx + args.limit]
    records = []
    for offset, sample in enumerate(selected):
        records.append(
            await run_stage1_for_sample(
                sample,
                args.sample_idx + offset,
                conf,
                client,
                output_dir,
                timestamp,
                args.show_prompt,
                args.skip_deep,
            )
        )

    summary_path = os.path.join(output_dir, f"stage1_summary_{conf['dataset']}_{timestamp}.jsonl")
    with open(summary_path, "w", encoding="utf-8") as handle:
        for record in records:
            compact_record = {
                "sample_idx": record["sample_idx"],
                "bundle_id": record["bundle_id"],
                "output_file": stage1_output_path(
                    output_dir,
                    conf.get("dataset", "dataset"),
                    record["bundle_id"],
                    record["sample_idx"],
                    timestamp,
                ),
                "execution_summary": record["execution_summary"],
                "repair_attempts_used": record["retrieval_repair_attempts_used"],
                "evidence_json": record["evidence_json"],
                "deep_execution_summary": record["deep_execution_summary"],
                "deep_repair_attempts_used": record["deep_repair_attempts_used"],
                "deep_evidence_json": record["deep_evidence_json"],
            }
            handle.write(compact_json(compact_record) + "\n")
    print(f">>> Summary saved: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Run only stage 1 of the three-stage agent")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--show_prompt", action="store_true")
    parser.add_argument("--skip_deep", action="store_true")
    args = parser.parse_args()
    asyncio.run(async_main(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

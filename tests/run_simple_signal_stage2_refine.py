"""Manual live-API runner for Simple Signal Stages 1 and 2.

The runner performs Stage 1 broad signal generation and configured Stage 2
graph-guided multi-hop rounds with rule-based validation. It intentionally
stops before the final Decision Agent. The ``run_...`` filename keeps paid API
calls out of unittest discovery.
"""

import argparse
import asyncio
import json
import os
import sys
import time


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

import yaml

from dataset import BundleZeroShotDataset, set_seed
from main import (
    console_safe_text,
    create_llm_client,
    default_api_key_envs,
    generate_content_with_retry,
    llm_provider,
    resolve_api_key,
)
from progressive_signal_agent.common import (
    build_case_view,
    candidate_labels,
)
from progressive_signal_agent.workspace import build_source_manifest, prepare_workspace
from simple_signal_agent.pipeline import (
    CONFIG_PREFIX,
    _generate_execute_repair,
    merge_signal_evidence,
)
from simple_signal_agent.affordance_graph import build_evidence_affordance_graph
from simple_signal_agent.prompts import signal_code_prompt


def print_prompt_debug(title, prompt):
    print(f"\n[DEBUG] {title}:")
    print(console_safe_text(prompt))
    print("-" * 60)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the Simple Signal Stage 1 broad scan and graph-guided Stage 2 "
            "multi-hop rounds without calling the Decision Agent."
        )
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--output", default="")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


async def run_stages_1_and_2(sample, conf, clients, debug_callback=None):
    case_view = build_case_view(sample, conf["dataset"])
    labels = candidate_labels(case_view)
    workspace = prepare_workspace(conf, config_prefix=CONFIG_PREFIX)
    source_manifest = build_source_manifest(
        workspace,
        str(conf.get("simple_signal_current_bundle_train_context_policy", "allow")),
    )
    affordance_graph = build_evidence_affordance_graph(source_manifest, conf["dataset"])
    max_refinements = max(0, int(conf.get("simple_signal_max_refinement_rounds", 1)))
    max_evidence_chars = int(conf.get("simple_signal_max_evidence_chars", 30000))

    round_trace = []
    refinement_context = None
    final_evidence = {"signals": []}

    for round_index in range(max_refinements + 1):
        round_number = round_index + 1
        output_file = (
            f"output/simple_signal_evidence_bundle{sample['bundle_id']}_round{round_number}.json"
        )
        code_prompt = signal_code_prompt(
            case_view,
            source_manifest,
            affordance_graph,
            output_file,
            max_evidence_chars,
            refinement_context=refinement_context,
        )
        if debug_callback:
            debug_callback(f"Simple Signal Code Prompt {round_number}", code_prompt)

        code_result = await _generate_execute_repair(
            sample_bundle_id=sample["bundle_id"],
            round_index=round_index,
            case_view=case_view,
            source_manifest=source_manifest,
            initial_prompt=code_prompt,
            client=clients["code"],
            conf=conf,
            generate_content_fn=generate_content_with_retry,
            workspace=workspace,
            output_file=output_file,
            labels=labels,
            affordance_graph=affordance_graph,
        )
        evidence = code_result["accepted_evidence"]
        if evidence is not None:
            final_evidence = merge_signal_evidence(final_evidence, evidence)

        round_trace.append(
            {
                "round": round_number,
                "is_refinement": round_index > 0,
                "round_role": "primitive_signal_scan" if round_index == 0 else "multi_hop_refinement",
                "code_prompt": code_prompt,
                "code_raw_response": code_result["raw_response"],
                "generated_code": code_result["generated_code"],
                "code_repairs": code_result["repairs"],
                "execution_summary": code_result["execution_summary"],
                "validation_issues": code_result["validation_issues"],
                "rule_validation_passed": evidence is not None,
                "accepted_evidence": evidence,
                "merged_evidence_after_round": final_evidence,
            }
        )

        if evidence is None:
            break

        refinement_context = final_evidence

    return {
        "method": "simple_generate_evaluate_decide_stages1_2_refine",
        "case": case_view,
        "workspace_dir": workspace["workspace_dir"],
        "workspace_files": workspace["files"],
        "copied_files": workspace["copied_files"],
        "source_manifest": source_manifest,
        "evidence_affordance_graph": affordance_graph,
        "configured_max_refinement_rounds": max_refinements,
        "round_count": len(round_trace),
        "refinement_count": max(0, len(round_trace) - 1),
        "round_trace": round_trace,
        "final_evidence": final_evidence,
        "rule_validation_passed": bool(final_evidence.get("signals")),
    }


async def run(args):
    os.chdir(REPO_ROOT)
    with open(args.config, "r", encoding="utf-8") as handle:
        conf = yaml.safe_load(handle)

    set_seed(int(conf.get("seed", 45)))
    samples = BundleZeroShotDataset(conf).get_eval_samples()
    start = int(args.sample_idx)
    limit = int(args.limit)
    if start < 0 or start >= len(samples):
        raise ValueError(f"sample_idx must be between 0 and {len(samples) - 1}.")
    if limit < 1:
        raise ValueError("limit must be at least 1.")
    selected = samples[start : start + limit]

    fallback_envs = default_api_key_envs(conf)
    code_api_key, code_api_key_env = resolve_api_key(
        conf,
        "simple_signal_code_api_key_env",
        fallback_envs,
    )
    clients = {
        "code": create_llm_client(conf, code_api_key),
    }

    print(f">>> Provider: {llm_provider(conf)}")
    print(f">>> Model: {conf['model']}")
    print(f">>> Code API key env: {code_api_key_env}")
    print(f">>> Dataset: {conf['dataset']}")
    print(f">>> Samples: {len(selected)} (start={start})")
    print(
        ">>> Max refinement rounds: "
        f"{max(0, int(conf.get('simple_signal_max_refinement_rounds', 1)))}"
    )

    records = []
    for offset, sample in enumerate(selected):
        sample_index = start + offset
        print(
            f"\n[{offset + 1}/{len(selected)}] "
            f"sample_idx={sample_index}, bundle_id={sample['bundle_id']}"
        )
        result = await run_stages_1_and_2(
            sample,
            conf,
            clients,
            debug_callback=print_prompt_debug if args.debug else None,
        )
        records.append({"sample_idx": sample_index, **result})

        for trace in result["round_trace"]:
            print(
                f">>> Round {trace['round']}: "
                f"evidence_accepted={isinstance(trace['accepted_evidence'], dict)}, "
                f"repairs={len(trace['code_repairs'])}, "
                f"rule_validation_passed={trace['rule_validation_passed']}"
            )
        print(">>> Final merged evidence signals: " + str(len(result["final_evidence"]["signals"])))

    if args.output:
        output_path = os.path.abspath(args.output)
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(
            REPO_ROOT,
            "analysis",
            "simple_signal_stage2_refine",
            f"stage2_refine_{conf['dataset']}_{timestamp}.json",
        )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    payload = {
        "method": "simple_generate_evaluate_decide_stages1_2_refine",
        "dataset": conf["dataset"],
        "model": conf["model"],
        "llm_provider": llm_provider(conf),
        "api_key_envs": {
            "code": code_api_key_env,
        },
        "records": records,
    }
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
    print(f"\nSaved Stage-1/2 refinement trace to: {output_path}")


def main():
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()

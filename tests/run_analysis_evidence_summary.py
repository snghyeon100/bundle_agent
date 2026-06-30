import argparse
import asyncio
import os
import sys
import yaml
from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from dataset import BundleZeroShotDataset, set_seed
from main import _build_clients, generate_content_with_retry
from sem_agent.common import build_case_view, candidate_labels, compact_json, parse_json_from_text
from sem_agent.workspace import prepare_workspace, build_source_manifest
from sem_agent.prompts import problem_analysis_prompt, stage1_ecosystem_prompt, stage2_gap_prompt
from sem_agent.pipeline import (
    _call_stage,
    _generate_execute_repair,
    build_decision_case,
    validate_stage2_summary_evidence,
)

load_dotenv(
    dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"),
    encoding="utf-8-sig",
)


async def run_analysis_evidence_summary(conf, sample, clients):
    print(f"--- Running Test for Bundle ID: {sample['bundle_id']} ---")
    
    # Setup
    case_view = build_case_view(sample, conf["dataset"])
    semantic_case = build_decision_case(sample, conf)
    labels = candidate_labels(case_view)
    workspace = prepare_workspace(conf, config_prefix="sem")
    source_manifest = build_source_manifest(
        workspace,
        str(conf.get("sem_current_bundle_train_context_policy", "allow")),
    )
    max_chars = int(conf.get("sem_max_evidence_chars", 30000))
    
    print("\n[Problem Analysis] Executing...")
    analysis_prompt = problem_analysis_prompt(case_view, source_manifest, semantic_case=semantic_case)
    analysis_raw = await _call_stage(
        generate_content_with_retry,
        clients["analysis"],
        conf=conf,
        prompt=analysis_prompt,
        max_tokens_key="sem_analysis_max_output_tokens",
        default_tokens=1200,
        step_name="sem problem analysis",
    )
    print("Problem Analysis completed.")
    print(analysis_raw)

    print("\n[Evidence Retrieval] Executing...")
    s1_output_file = f"output/test_sem_bundle{sample['bundle_id']}_stage1.json"
    s1_prompt = stage1_ecosystem_prompt(
        case_view,
        source_manifest,
        s1_output_file,
        max_chars,
        semantic_case=semantic_case,
        problem_analysis=analysis_raw,
    )
    
    s1_result = await _generate_execute_repair(
        bundle_id=sample["bundle_id"],
        stage_index=0,
        case_view=case_view,
        source_manifest=source_manifest,
        initial_prompt=s1_prompt,
        client=clients["stage1"],
        conf=conf,
        generate_content_fn=generate_content_with_retry,
        workspace=workspace,
        output_file=s1_output_file,
        labels=labels,
    )
    
    stage1_full_output = s1_result["accepted_evidence"] or {"signals": []}
    stage1_evidence = (
        {"signals": stage1_full_output["signals"]}
        if isinstance(stage1_full_output, dict) and isinstance(stage1_full_output.get("signals"), list)
        else {"signals": []}
    )
    print("Evidence Retrieval completed.")
    print("Evidence Validation Issues:", s1_result["validation_issues"])
    print("Evidence Policy Trace:")
    print(compact_json(stage1_full_output.get("policy_trace", {}) if isinstance(stage1_full_output, dict) else {}))
    print("Evidence Signals Output:")
    print(compact_json(stage1_evidence))
    
    print("\n[Summary/Profile] Executing...")
    s2_output_file = f"output/test_sem_bundle{sample['bundle_id']}_stage2.json"
    s2_prompt = stage2_gap_prompt(
        case_view, source_manifest, s2_output_file, max_chars,
        stage1_evidence=stage1_evidence,
        semantic_case=semantic_case,
    )
    
    s2_raw = await _call_stage(
        generate_content_with_retry,
        clients["stage2"],
        conf=conf,
        prompt=s2_prompt,
        max_tokens_key="sem_stage2_max_output_tokens",
        default_tokens=4000,
        step_name="sem summary/profile",
    )
    parsed_stage2 = parse_json_from_text(s2_raw)
    stage2_issues = []
    if not isinstance(parsed_stage2, dict):
        stage2_issues.append("Summary response was not parseable JSON.")
        stage2_evidence = {"signals": []}
    else:
        stage2_evidence = parsed_stage2
        stage2_issues.extend(validate_stage2_summary_evidence(stage2_evidence, labels))
    s2_result = {
        "raw_response": s2_raw,
        "generated_code": "",
        "repairs": [],
        "execution_summary": {"msg": "pure reasoning, skipped execution"},
        "validation_issues": stage2_issues,
        "accepted_evidence": stage2_evidence if not stage2_issues else None,
    }
    
    stage2_evidence = s2_result["accepted_evidence"] or {"signals": []}
    print("Summary/Profile completed.")
    print("Summary Validation Issues:", s2_result["validation_issues"])
    print("Summary Output:")
    print(compact_json(stage2_evidence))
    
    print("\n--- Test Completed ---")


def main():
    parser = argparse.ArgumentParser(
        description="Test sem_agent Problem Analysis, Evidence Retrieval, and Summary/Profile without decision"
    )
    parser.add_argument("--config", default="config_sem.yaml")
    parser.add_argument("--bundle_index", type=int, default=0, help="Index of the bundle in the dataset to test")
    args = parser.parse_args()

    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", args.config))
    with open(config_path, "r", encoding="utf-8") as f:
        conf = yaml.safe_load(f)

    # Need data_path correctly resolved from root
    conf["data_path"] = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", conf.get("data_path", "datasets")))
    
    set_seed(int(conf.get("seed", 45)))
    samples = BundleZeroShotDataset(conf).get_eval_samples()
    
    if args.bundle_index >= len(samples):
        print(f"Error: bundle_index {args.bundle_index} is out of bounds (max {len(samples)-1})")
        return

    sample = samples[args.bundle_index]
    clients, _ = _build_clients(conf)
    
    asyncio.run(run_analysis_evidence_summary(conf, sample, clients))


if __name__ == "__main__":
    main()

import argparse
import asyncio
import os
import sys
import yaml
from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from dataset import BundleZeroShotDataset, set_seed
from main_sem import _build_clients, generate_content_with_retry
from sem_agent.common import build_case_view, candidate_labels, compact_json
from sem_agent.workspace import prepare_workspace, build_source_manifest
from sem_agent.affordance_graph import build_evidence_affordance_graph
from sem_agent.prompts import stage1_ecosystem_prompt, stage2_gap_prompt
from sem_agent.pipeline import _generate_execute_repair, build_decision_case

load_dotenv(
    dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"),
    encoding="utf-8-sig",
)


async def run_stage1_and_stage2_only(conf, sample, clients):
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
    affordance_graph = build_evidence_affordance_graph(source_manifest, conf["dataset"])
    max_chars = int(conf.get("sem_max_evidence_chars", 30000))
    
    print("\n[Stage 1: Item Evidence Expansion] Executing...")
    s1_output_file = f"output/test_sem_bundle{sample['bundle_id']}_stage1.json"
    s1_prompt = stage1_ecosystem_prompt(
        case_view,
        source_manifest,
        affordance_graph,
        s1_output_file,
        max_chars,
        semantic_case=semantic_case,
    )
    
    s1_result = await _generate_execute_repair(
        bundle_id=sample["bundle_id"],
        stage_index=0,
        case_view=case_view,
        source_manifest=source_manifest,
        initial_prompt=s1_prompt,
        client=clients["code"],
        conf=conf,
        generate_content_fn=generate_content_with_retry,
        workspace=workspace,
        output_file=s1_output_file,
        labels=labels,
        affordance_graph=affordance_graph,
    )
    
    stage1_evidence = s1_result["accepted_evidence"] or {"signals": []}
    print("Stage 1 execution completed.")
    print("Stage 1 Validation Issues:", s1_result["validation_issues"])
    print("Stage 1 Evidence Output:")
    print(compact_json(stage1_evidence))
    
    print("\n[Stage 2: Bundle Context & Candidate Fit] Executing...")
    s2_output_file = f"output/test_sem_bundle{sample['bundle_id']}_stage2.json"
    s2_prompt = stage2_gap_prompt(
        case_view, source_manifest, affordance_graph, s2_output_file, max_chars,
        stage1_evidence=stage1_evidence,
    )
    
    s2_result = await _generate_execute_repair(
        bundle_id=sample["bundle_id"],
        stage_index=1,
        case_view=case_view,
        source_manifest=source_manifest,
        initial_prompt=s2_prompt,
        client=clients["code"],
        conf=conf,
        generate_content_fn=generate_content_with_retry,
        workspace=workspace,
        output_file=s2_output_file,
        labels=labels,
        affordance_graph=affordance_graph,
    )
    
    stage2_evidence = s2_result["accepted_evidence"] or {"signals": []}
    print("Stage 2 execution completed.")
    print("Stage 2 Validation Issues:", s2_result["validation_issues"])
    print("Stage 2 Evidence Output:")
    print(compact_json(stage2_evidence))
    
    print("\n--- Test Completed ---")


def main():
    parser = argparse.ArgumentParser(
        description="Test sem_agent Stage 1 Item Evidence Expansion and Stage 2 Bundle Context & Candidate Fit without decision"
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
    
    asyncio.run(run_stage1_and_stage2_only(conf, sample, clients))


if __name__ == "__main__":
    main()

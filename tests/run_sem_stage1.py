import argparse
import asyncio
import json
import os
import shutil
import sys

import yaml
from dotenv import load_dotenv


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
sys.path.insert(0, SRC_DIR)

from dataset import BundleZeroShotDataset, set_seed
from main import _build_clients, generate_content_with_retry
from sem_agent.common import build_case_view, candidate_labels, compact_json
from sem_agent.pipeline import _generate_execute_repair, build_decision_case
from sem_agent.prompts import stage1_ecosystem_prompt
from sem_agent.workspace import build_source_manifest, prepare_workspace


load_dotenv(
    dotenv_path=os.path.join(REPO_ROOT, ".env"),
    encoding="utf-8-sig",
)


def analysis_run_dir(sample, analysis_dir):
    bundle_id = sample["bundle_id"]
    run_dir = os.path.join(analysis_dir, f"sem_stage1_bundle{bundle_id}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_analysis_outputs(result, prompt, evidence, workspace, output_file, sample, run_dir):
    bundle_id = sample["bundle_id"]
    os.makedirs(run_dir, exist_ok=True)

    workspace_output_path = os.path.join(workspace["workspace_dir"], output_file.replace("/", os.sep))
    workspace_script_path = os.path.join(
        workspace["workspace_dir"],
        f"sem_bundle{bundle_id}_stage1.py",
    )

    evidence_path = os.path.join(run_dir, "stage1_evidence.json")
    prompt_path = os.path.join(run_dir, "stage1_prompt.txt")
    summary_path = os.path.join(run_dir, "stage1_summary.json")
    script_path = os.path.join(run_dir, "stage1_generated_code.py")

    with open(evidence_path, "w", encoding="utf-8") as handle:
        json.dump(evidence, handle, ensure_ascii=False, indent=2)
    with open(prompt_path, "w", encoding="utf-8") as handle:
        handle.write(prompt)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "bundle_id": bundle_id,
                "accepted": result["accepted_evidence"] is not None,
                "validation_issues": result["validation_issues"],
                "execution_summary": result["execution_summary"],
                "workspace_dir": workspace["workspace_dir"],
                "workspace_output_path": workspace_output_path,
                "workspace_script_path": workspace_script_path,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    if os.path.isfile(workspace_script_path):
        shutil.copy2(workspace_script_path, script_path)

    return {
        "run_dir": run_dir,
        "evidence_path": evidence_path,
        "prompt_path": prompt_path,
        "summary_path": summary_path,
        "script_path": script_path if os.path.isfile(script_path) else "",
    }


async def run_stage1_only(conf, sample, clients, print_prompt=False, analysis_dir=None):
    analysis_dir = analysis_dir or os.path.join(REPO_ROOT, "analysis")
    run_dir = analysis_run_dir(sample, analysis_dir)
    run_conf = dict(conf)
    run_conf["sem_workspace_root"] = os.path.join(run_dir, "workspace")

    case_view = build_case_view(sample, conf["dataset"])
    semantic_case = build_decision_case(sample, run_conf)
    labels = candidate_labels(case_view)
    workspace = prepare_workspace(run_conf, config_prefix="sem")
    source_manifest = build_source_manifest(
        workspace,
        str(conf.get("sem_current_bundle_train_context_policy", "allow")),
    )
    max_chars = int(conf.get("sem_max_evidence_chars", 30000))

    output_file = f"output/test_sem_bundle{sample['bundle_id']}_stage1.json"
    prompt = stage1_ecosystem_prompt(
        case_view,
        source_manifest,
        output_file,
        max_chars,
        semantic_case=semantic_case,
    )

    if print_prompt:
        print("\n[Stage 1 Prompt]")
        print(prompt)
        print("\n" + "-" * 80)

    result = await _generate_execute_repair(
        bundle_id=sample["bundle_id"],
        stage_index=0,
        case_view=case_view,
        source_manifest=source_manifest,
        initial_prompt=prompt,
        client=clients["stage1"],
        conf=run_conf,
        generate_content_fn=generate_content_with_retry,
        workspace=workspace,
        output_file=output_file,
        labels=labels,
    )

    evidence = result["accepted_evidence"] or {"signals": []}
    output_path = os.path.join(workspace["workspace_dir"], output_file.replace("/", os.sep))
    script_path = os.path.join(
        workspace["workspace_dir"],
        f"sem_bundle{sample['bundle_id']}_stage1.py",
    )
    analysis_paths = save_analysis_outputs(
        result,
        prompt,
        evidence,
        workspace,
        output_file,
        sample,
        run_dir,
    )

    print(f"Bundle ID: {sample['bundle_id']}")
    print(f"Workspace: {workspace['workspace_dir']}")
    print(f"Generated script: {script_path}")
    print(f"Evidence JSON: {output_path}")
    print(f"Analysis output: {analysis_paths['run_dir']}")
    print(f"Accepted: {result['accepted_evidence'] is not None}")
    print(f"Validation issues: {result['validation_issues']}")
    summary = result.get("execution_summary", {})
    if not result["accepted_evidence"]:
        print(f"Execution summary: {compact_json(summary)}")
    print("\n[Stage 1 Evidence]")
    print(compact_json(evidence))

    return result


def load_config(path):
    config_path = path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)
    with open(config_path, "r", encoding="utf-8") as handle:
        conf = yaml.safe_load(handle)
    conf["data_path"] = os.path.abspath(os.path.join(REPO_ROOT, conf.get("data_path", "datasets")))
    conf["sem_workspace_root"] = os.path.abspath(
        os.path.join(REPO_ROOT, conf.get("sem_workspace_root", "agent_workspaces"))
    )
    return conf


def main():
    parser = argparse.ArgumentParser(
        description="Run only Stage 1 Item Evidence Expansion of sem_agent for one sample"
    )
    parser.add_argument("--config", default="config_sem.yaml")
    parser.add_argument(
        "--bundle_index",
        type=int,
        default=0,
        help="Index of the evaluation sample to run",
    )
    parser.add_argument(
        "--print_prompt",
        action="store_true",
        help="Print the Stage 1 code-generation prompt before calling the model",
    )
    parser.add_argument(
        "--analysis_dir",
        default=os.path.join(REPO_ROOT, "analysis"),
        help="Directory where Stage 1 test outputs are saved",
    )
    args = parser.parse_args()

    os.chdir(REPO_ROOT)
    conf = load_config(args.config)
    set_seed(int(conf.get("seed", 45)))

    samples = BundleZeroShotDataset(conf).get_eval_samples()
    if not 0 <= args.bundle_index < len(samples):
        raise IndexError(
            f"bundle_index {args.bundle_index} is out of range; "
            f"available indices are 0..{len(samples) - 1}"
        )

    sample = samples[args.bundle_index]
    clients, resolved_envs = _build_clients(conf)
    print(f"Stage 1 API key env: {resolved_envs['stage1']}")

    analysis_dir = args.analysis_dir
    if not os.path.isabs(analysis_dir):
        analysis_dir = os.path.join(REPO_ROOT, analysis_dir)
    asyncio.run(
        run_stage1_only(
            conf,
            sample,
            clients,
            print_prompt=args.print_prompt,
            analysis_dir=analysis_dir,
        )
    )


if __name__ == "__main__":
    main()

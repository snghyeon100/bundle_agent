import argparse
import asyncio
import json
import os
import sys

import yaml
from dotenv import load_dotenv


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
sys.path.insert(0, SRC_DIR)

from dataset import BundleZeroShotDataset, set_seed
from main import _build_clients, generate_content_with_retry
from sem_agent.common import build_case_view
from sem_agent.pipeline import _call_stage, build_decision_case
from sem_agent.prompts import problem_analysis_prompt
from sem_agent.workspace import build_source_manifest, prepare_workspace


load_dotenv(
    dotenv_path=os.path.join(REPO_ROOT, ".env"),
    encoding="utf-8-sig",
)


def analysis_run_dir(sample, analysis_dir):
    run_dir = os.path.join(analysis_dir, f"sem_analysis_bundle{sample['bundle_id']}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


async def run_analysis_only(conf, sample, clients, print_prompt=False, analysis_dir=None):
    analysis_dir = analysis_dir or os.path.join(REPO_ROOT, "analysis")
    run_dir = analysis_run_dir(sample, analysis_dir)
    run_conf = dict(conf)
    run_conf["sem_workspace_root"] = os.path.join(run_dir, "workspace")

    case_view = build_case_view(sample, conf["dataset"])
    semantic_case = build_decision_case(sample, run_conf)
    workspace = prepare_workspace(run_conf, config_prefix="sem")
    source_manifest = build_source_manifest(
        workspace,
        str(conf.get("sem_current_bundle_train_context_policy", "allow")),
    )
    prompt = problem_analysis_prompt(case_view, source_manifest, semantic_case=semantic_case)

    if print_prompt:
        print("\n[Problem Analysis Prompt]")
        print(prompt)
        print("\n" + "-" * 80)

    raw = await _call_stage(
        generate_content_with_retry,
        clients["analysis"],
        run_conf,
        prompt,
        "sem_analysis_max_output_tokens",
        1200,
        "sem problem analysis",
    )

    prompt_path = os.path.join(run_dir, "analysis_prompt.txt")
    analysis_path = os.path.join(run_dir, "problem_analysis.txt")
    summary_path = os.path.join(run_dir, "analysis_summary.json")
    with open(prompt_path, "w", encoding="utf-8") as handle:
        handle.write(prompt)
    with open(analysis_path, "w", encoding="utf-8") as handle:
        handle.write(raw)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "bundle_id": sample["bundle_id"],
                "analysis_chars": len(raw),
                "prompt_path": prompt_path,
                "analysis_path": analysis_path,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Bundle ID: {sample['bundle_id']}")
    print(f"Analysis output: {run_dir}")
    print(f"Prompt: {prompt_path}")
    print(f"Problem Analysis: {analysis_path}")
    print("\n[Problem Analysis]")
    print(raw)
    return raw


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
    parser = argparse.ArgumentParser(description="Run only Problem Analysis for one sem_agent sample")
    parser.add_argument("--config", default="config_sem.yaml")
    parser.add_argument("--bundle_index", type=int, default=0)
    parser.add_argument("--print_prompt", action="store_true")
    parser.add_argument("--analysis_dir", default=os.path.join(REPO_ROOT, "analysis"))
    args = parser.parse_args()

    os.chdir(REPO_ROOT)
    conf = load_config(args.config)
    set_seed(int(conf.get("seed", 45)))
    samples = BundleZeroShotDataset(conf).get_eval_samples()
    sample = samples[args.bundle_index]
    clients, resolved_envs = _build_clients(conf)
    print(f"Problem Analysis API key env: {resolved_envs['analysis']}")

    analysis_dir = args.analysis_dir
    if not os.path.isabs(analysis_dir):
        analysis_dir = os.path.join(REPO_ROOT, analysis_dir)
    asyncio.run(
        run_analysis_only(
            conf,
            sample,
            clients,
            print_prompt=args.print_prompt,
            analysis_dir=analysis_dir,
        )
    )


if __name__ == "__main__":
    main()

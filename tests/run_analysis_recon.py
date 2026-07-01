import argparse
import json
import os
import sys

import yaml
from dotenv import load_dotenv


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
sys.path.insert(0, SRC_DIR)

from dataset import BundleZeroShotDataset, set_seed
from sem_agent.common import build_case_view
from sem_agent.workspace import build_analysis_recon, prepare_workspace


load_dotenv(
    dotenv_path=os.path.join(REPO_ROOT, ".env"),
    encoding="utf-8-sig",
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def load_config(path):
    config_path = path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)
    with open(config_path, "r", encoding="utf-8") as handle:
        conf = yaml.safe_load(handle)
    conf["data_path"] = os.path.abspath(os.path.join(REPO_ROOT, conf.get("data_path", "datasets")))
    conf["sem_workspace_root"] = os.path.abspath(
        os.path.join(REPO_ROOT, conf.get("sem_workspace_root", "agent_workspaces"))
    )
    return conf


def recon_run_dir(sample, analysis_dir):
    run_dir = os.path.join(analysis_dir, f"sem_recon_bundle{sample['bundle_id']}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def run_recon_only(conf, sample, analysis_dir=None):
    analysis_dir = analysis_dir or os.path.join(REPO_ROOT, "analysis")
    run_dir = recon_run_dir(sample, analysis_dir)
    run_conf = dict(conf)
    run_conf["sem_workspace_root"] = os.path.join(run_dir, "workspace")

    case_view = build_case_view(sample, conf["dataset"])
    workspace = prepare_workspace(run_conf, config_prefix="sem")
    recon = build_analysis_recon(workspace, case_view)

    out = {
        "case_view": case_view,
        "analysis_recon": recon,
    }

    output_path = os.path.join(run_dir, "analysis_recon.json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(out, handle, ensure_ascii=False, indent=2)

    print(f"Bundle ID: {sample['bundle_id']}")
    print(f"Recon output: {output_path}")
    print("\n[Analysis Recon]")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return out


def main():
    parser = argparse.ArgumentParser(description="Run only deterministic analysis recon for one sem_agent sample")
    parser.add_argument("--config", default="config_sem.yaml")
    parser.add_argument("--bundle_index", type=int, default=0)
    parser.add_argument("--analysis_dir", default=os.path.join(REPO_ROOT, "analysis"))
    args = parser.parse_args()

    os.chdir(REPO_ROOT)
    conf = load_config(args.config)
    set_seed(int(conf.get("seed", 45)))
    samples = BundleZeroShotDataset(conf).get_eval_samples()
    sample = samples[args.bundle_index]

    analysis_dir = args.analysis_dir
    if not os.path.isabs(analysis_dir):
        analysis_dir = os.path.join(REPO_ROOT, analysis_dir)
    run_recon_only(conf, sample, analysis_dir=analysis_dir)


if __name__ == "__main__":
    main()

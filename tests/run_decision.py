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
from main import _build_clients, generate_content_with_retry, parse_model_response
from sem_agent.common import build_case_view, candidate_labels, compact_json
from sem_agent.pipeline import _call_stage, _parse_prediction, build_decision_case, validate_stage2_summary_evidence
from sem_agent.prompts import decision_prompt


load_dotenv(
    dotenv_path=os.path.join(REPO_ROOT, ".env"),
    encoding="utf-8-sig",
)


def decision_run_dir(sample, analysis_dir):
    run_dir = os.path.join(analysis_dir, f"sem_decision_bundle{sample['bundle_id']}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def default_summary_json_path(sample, analysis_dir):
    return os.path.join(
        analysis_dir,
        f"sem_summary_bundle{sample['bundle_id']}",
        "summary_output.json",
    )


def load_summary_evidence(path):
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict) and isinstance(payload.get("signals"), list):
        return payload
    raise ValueError(f"Summary JSON must be an object with a signals list: {path}")


def save_decision_outputs(
    run_dir,
    sample,
    summary_json_path,
    summary_evidence,
    prompt,
    raw,
    prediction,
    decision_json,
    validation_issues,
):
    summary_copy_path = os.path.join(run_dir, "summary_input.json")
    prompt_path = os.path.join(run_dir, "decision_prompt.txt")
    raw_path = os.path.join(run_dir, "decision_raw_response.txt")
    decision_json_path = os.path.join(run_dir, "decision_output.json")
    run_summary_path = os.path.join(run_dir, "decision_run_summary.json")

    with open(summary_copy_path, "w", encoding="utf-8") as handle:
        json.dump(summary_evidence, handle, ensure_ascii=False, indent=2)
    with open(prompt_path, "w", encoding="utf-8") as handle:
        handle.write(prompt)
    with open(raw_path, "w", encoding="utf-8") as handle:
        handle.write(str(raw))
    with open(decision_json_path, "w", encoding="utf-8") as handle:
        json.dump(decision_json, handle, ensure_ascii=False, indent=2)
    with open(run_summary_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "bundle_id": sample["bundle_id"],
                "true_option_char": sample.get("true_option_char"),
                "prediction": prediction,
                "hit": int(prediction == sample.get("true_option_char")),
                "summary_json_path": summary_json_path,
                "summary_input_copy_path": summary_copy_path,
                "decision_prompt_path": prompt_path,
                "decision_raw_response_path": raw_path,
                "decision_output_path": decision_json_path,
                "validation_issues": validation_issues,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    return {
        "run_dir": run_dir,
        "summary_copy_path": summary_copy_path,
        "prompt_path": prompt_path,
        "raw_path": raw_path,
        "decision_json_path": decision_json_path,
        "run_summary_path": run_summary_path,
    }


async def run_decision_only(
    conf,
    sample,
    clients,
    summary_json_path,
    print_prompt=False,
    analysis_dir=None,
):
    analysis_dir = analysis_dir or os.path.join(REPO_ROOT, "analysis")
    run_dir = decision_run_dir(sample, analysis_dir)

    summary_evidence = load_summary_evidence(summary_json_path)
    case_view = build_case_view(sample, conf["dataset"])
    labels = candidate_labels(case_view)
    validation_issues = validate_stage2_summary_evidence(summary_evidence, labels)
    decision_case = build_decision_case(sample, conf)
    prompt = decision_prompt(decision_case, summary_evidence)

    if print_prompt:
        print("\n[Decision Prompt]")
        print(prompt)
        print("\n" + "-" * 80)

    raw = await _call_stage(
        generate_content_with_retry,
        clients["prediction"],
        conf,
        prompt,
        "sem_prediction_max_output_tokens",
        200,
        "sem final decision",
    )
    prediction, decision_json = _parse_prediction(raw, parse_model_response, labels)

    paths = save_decision_outputs(
        run_dir,
        sample,
        summary_json_path,
        summary_evidence,
        prompt,
        raw,
        prediction,
        decision_json,
        validation_issues,
    )

    print(f"Bundle ID: {sample['bundle_id']}")
    print(f"Summary JSON: {summary_json_path}")
    print(f"Decision output: {paths['run_dir']}")
    print(f"Prompt: {paths['prompt_path']}")
    print(f"Raw response: {paths['raw_path']}")
    print(f"Decision JSON: {paths['decision_json_path']}")
    print(f"Validation issues: {validation_issues}")
    print(f"True: {sample.get('true_option_char')} | Pred: {prediction} | Hit: {int(prediction == sample.get('true_option_char'))}")
    print("\n[Decision Raw Response]")
    print(raw)

    return prediction, raw


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
        description="Run only final Decision of sem_agent using saved Summary/Profile JSON"
    )
    parser.add_argument("--config", default="config_sem.yaml")
    parser.add_argument("--bundle_index", type=int, default=0, help="Index of the evaluation sample to run")
    parser.add_argument(
        "--summary_json",
        default="",
        help=(
            "Path to Summary/Profile JSON. Defaults to "
            "analysis/sem_summary_bundle{bundle_id}/summary_output.json"
        ),
    )
    parser.add_argument("--print_prompt", action="store_true", help="Print the final decision prompt")
    parser.add_argument(
        "--analysis_dir",
        default=os.path.join(REPO_ROOT, "analysis"),
        help="Directory where Decision test outputs are saved",
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
    analysis_dir = args.analysis_dir
    if not os.path.isabs(analysis_dir):
        analysis_dir = os.path.join(REPO_ROOT, analysis_dir)

    summary_json_path = args.summary_json.strip() or default_summary_json_path(sample, analysis_dir)
    if not os.path.isabs(summary_json_path):
        summary_json_path = os.path.join(REPO_ROOT, summary_json_path)
    if not os.path.isfile(summary_json_path):
        raise FileNotFoundError(
            "Summary/Profile JSON not found. Run tests/run_summary.py first "
            f"or pass --summary_json. Missing: {summary_json_path}"
        )

    clients, resolved_envs = _build_clients(conf)
    print(f"Decision API key env: {resolved_envs['prediction']}")

    asyncio.run(
        run_decision_only(
            conf,
            sample,
            clients,
            summary_json_path=summary_json_path,
            print_prompt=args.print_prompt,
            analysis_dir=analysis_dir,
        )
    )


if __name__ == "__main__":
    main()

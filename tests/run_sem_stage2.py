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
from main_sem import _build_clients, generate_content_with_retry
from sem_agent.affordance_graph import build_evidence_affordance_graph
from sem_agent.common import build_case_view, candidate_labels, compact_json
from sem_agent.pipeline import _call_stage, build_decision_case, parse_json_from_text
from sem_agent.prompts import stage2_gap_prompt
from sem_agent.workspace import build_source_manifest, prepare_workspace


load_dotenv(
    dotenv_path=os.path.join(REPO_ROOT, ".env"),
    encoding="utf-8-sig",
)


def log(message):
    print(message, flush=True)


def signal_names(evidence):
    signals = evidence.get("signals", []) if isinstance(evidence, dict) else []
    names = [
        str(signal.get("signal_name", "")).strip()
        for signal in signals
        if isinstance(signal, dict) and str(signal.get("signal_name", "")).strip()
    ]
    return names


def summarize_case(sample, labels):
    return {
        "bundle_id": sample["bundle_id"],
        "partial_items": sample.get("input_indices", []),
        "candidate_labels": labels,
        "candidate_items": sample.get("candidate_indices", []),
    }


async def generate_content_with_progress(client, model, contents, conf, max_output_tokens, step_name):
    log(f"[LLM] {step_name}: request sent (max_output_tokens={max_output_tokens})")
    try:
        raw = await generate_content_with_retry(
            client,
            model,
            contents,
            conf,
            max_output_tokens,
            step_name,
        )
    except Exception as exc:
        log(f"[LLM] {step_name}: failed ({type(exc).__name__}: {exc})")
        raise
    log(f"[LLM] {step_name}: response received ({len(raw)} chars)")
    return raw


def analysis_run_dir(sample, analysis_dir):
    bundle_id = sample["bundle_id"]
    run_dir = os.path.join(analysis_dir, f"sem_stage2_bundle{bundle_id}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def default_stage1_json_path(sample, analysis_dir):
    bundle_id = sample["bundle_id"]
    return os.path.join(
        analysis_dir,
        f"sem_stage1_bundle{bundle_id}",
        "stage1_evidence.json",
    )


def load_stage1_evidence(path):
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict) and isinstance(payload.get("signals"), list):
        return payload

    raise ValueError(
        f"Stage 1 JSON must be an evidence object with a signals list: {path}"
    )


def save_analysis_outputs(
    result,
    prompt,
    stage1_evidence,
    workspace,
    output_file,
    sample,
    run_dir,
    stage1_json_path,
):
    bundle_id = sample["bundle_id"]
    os.makedirs(run_dir, exist_ok=True)

    workspace_output_path = os.path.join(
        workspace["workspace_dir"],
        output_file.replace("/", os.sep),
    )
    workspace_script_path = os.path.join(
        workspace["workspace_dir"],
        f"sem_bundle{bundle_id}_stage2.py",
    )

    stage1_copy_path = os.path.join(run_dir, "stage1_evidence_input.json")
    evidence_path = os.path.join(run_dir, "stage2_evidence.json")
    prompt_path = os.path.join(run_dir, "stage2_prompt.txt")
    summary_path = os.path.join(run_dir, "stage2_summary.json")
    script_path = os.path.join(run_dir, "stage2_generated_code.py")

    with open(stage1_copy_path, "w", encoding="utf-8") as handle:
        json.dump(stage1_evidence, handle, ensure_ascii=False, indent=2)

    stage2_evidence = result["accepted_evidence"] or {"signals": []}
    with open(evidence_path, "w", encoding="utf-8") as handle:
        json.dump(stage2_evidence, handle, ensure_ascii=False, indent=2)
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
                "stage1_json_path": stage1_json_path,
                "stage1_input_copy_path": stage1_copy_path,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    if os.path.isfile(workspace_script_path):
        shutil.copy2(workspace_script_path, script_path)

    return {
        "run_dir": run_dir,
        "stage1_copy_path": stage1_copy_path,
        "evidence_path": evidence_path,
        "prompt_path": prompt_path,
        "summary_path": summary_path,
        "script_path": script_path if os.path.isfile(script_path) else "",
    }


async def run_stage2_only(
    conf,
    sample,
    clients,
    stage1_json_path,
    print_prompt=False,
    analysis_dir=None,
):
    analysis_dir = analysis_dir or os.path.join(REPO_ROOT, "analysis")
    run_dir = analysis_run_dir(sample, analysis_dir)
    run_conf = dict(conf)
    run_conf["sem_workspace_root"] = os.path.join(run_dir, "workspace")

    log("[1/8] Loading saved Stage 1 evidence JSON")
    log(f"      path: {stage1_json_path}")
    stage1_evidence = load_stage1_evidence(stage1_json_path)
    stage1_names = signal_names(stage1_evidence)
    log(f"      loaded signals: {len(stage1_names)}")
    if stage1_names:
        log(f"      signal names: {', '.join(stage1_names)}")

    log("[2/8] Building case view")
    case_view = build_case_view(sample, conf["dataset"])
    labels = candidate_labels(case_view)
    decision_case = build_decision_case(sample, conf)
    log(f"      case: {compact_json(summarize_case(sample, labels))}")

    log("[3/8] Preparing isolated Stage 2 workspace")
    workspace = prepare_workspace(run_conf, config_prefix="sem")
    workspace_files = [entry["name"] for entry in workspace.get("files", [])]
    copied_files = workspace.get("copied_files", [])
    log(f"      workspace: {workspace['workspace_dir']}")
    log(f"      available files: {', '.join(workspace_files) if workspace_files else '(none)'}")
    log(f"      copied files this run: {len(copied_files)}")

    log("[4/8] Building source manifest and affordance graph")
    source_manifest = build_source_manifest(
        workspace,
        str(conf.get("sem_current_bundle_train_context_policy", "allow")),
    )
    affordance_graph = build_evidence_affordance_graph(source_manifest, conf["dataset"])
    max_chars = int(conf.get("sem_max_evidence_chars", 30000))
    log(f"      manifest sources: {len(source_manifest.get('sources', []))}")
    log(f"      max evidence chars: {max_chars}")

    log("[5/8] Building Stage 2 prompt")
    output_file = f"output/test_sem_bundle{sample['bundle_id']}_stage2.json"
    prompt = stage2_gap_prompt(
        case_view,
        source_manifest,
        affordance_graph,
        output_file,
        max_chars,
        stage1_evidence=stage1_evidence,
        semantic_case=decision_case,
    )
    log(f"      output file requested from generated code: {output_file}")
    log(f"      prompt chars: {len(prompt)}")

    if print_prompt:
        print("\n[Stage 2 Prompt]")
        print(prompt)
        print("\n" + "-" * 80)

    log("[6/8] Generating Stage 2 pure reasoning response")
    s2_raw = await _call_stage(
        generate_content_with_progress,
        clients["code"],
        run_conf,
        prompt,
        "sem_code_max_output_tokens",
        4000,
        "sem stage2 reasoning",
    )
    
    stage2_evidence = parse_json_from_text(s2_raw) or {"signals": []}
    
    result = {
        "raw_response": s2_raw,
        "generated_code": "",
        "repairs": [],
        "execution_summary": {"msg": "pure reasoning, skipped execution"},
        "validation_issues": [],
        "accepted_evidence": stage2_evidence if "signals" in stage2_evidence else None,
    }
    log("      reasoning finished")

    log("[7/8] Saving debug artifacts")
    evidence = result["accepted_evidence"] or {"signals": []}
    output_path = os.path.join(workspace["workspace_dir"], output_file.replace("/", os.sep))
    script_path = os.path.join(
        workspace["workspace_dir"],
        f"sem_bundle{sample['bundle_id']}_stage2.py",
    )
    analysis_paths = save_analysis_outputs(
        result,
        prompt,
        stage1_evidence,
        workspace,
        output_file,
        sample,
        run_dir,
        stage1_json_path,
    )
    log(f"      analysis dir: {analysis_paths['run_dir']}")
    log(f"      prompt copy: {analysis_paths['prompt_path']}")
    log(f"      summary copy: {analysis_paths['summary_path']}")
    log(f"      script copy: {analysis_paths['script_path'] or '(not created)'}")
    log(f"      evidence copy: {analysis_paths['evidence_path']}")

    log("[8/8] Stage 2 result")
    print(f"Bundle ID: {sample['bundle_id']}")
    print(f"Stage 1 Evidence JSON: {stage1_json_path}")
    print(f"Workspace: {workspace['workspace_dir']}")
    print(f"Generated script: {script_path}")
    print(f"Evidence JSON: {output_path}")
    print(f"Analysis output: {analysis_paths['run_dir']}")
    print(f"Accepted: {result['accepted_evidence'] is not None}")
    print(f"Validation issues: {result['validation_issues']}")
    print("\n[Stage 2 Evidence]")
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
        description="Run only Stage 2 Bundle Context & Candidate Fit of sem_agent using saved Stage 1 JSON"
    )
    parser.add_argument("--config", default="config_sem.yaml")
    parser.add_argument(
        "--bundle_index",
        type=int,
        default=0,
        help="Index of the evaluation sample to run",
    )
    parser.add_argument(
        "--stage1_json",
        default="",
        help=(
            "Path to Stage 1 evidence JSON. Defaults to "
            "analysis/sem_stage1_bundle{bundle_id}/stage1_evidence.json"
        ),
    )
    parser.add_argument(
        "--print_prompt",
        action="store_true",
        help="Print the Stage 2 code-generation prompt before calling the model",
    )
    parser.add_argument(
        "--analysis_dir",
        default=os.path.join(REPO_ROOT, "analysis"),
        help="Directory where Stage 2 test outputs are saved",
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

    stage1_json_path = args.stage1_json.strip() or default_stage1_json_path(sample, analysis_dir)
    if not os.path.isabs(stage1_json_path):
        stage1_json_path = os.path.join(REPO_ROOT, stage1_json_path)
    if not os.path.isfile(stage1_json_path):
        raise FileNotFoundError(
            "Stage 1 evidence JSON not found. Run tests/run_sem_stage1.py first "
            f"or pass --stage1_json. Missing: {stage1_json_path}"
        )

    clients, resolved_envs = _build_clients(conf)
    print(f"Code API key env: {resolved_envs['code']}")

    asyncio.run(
        run_stage2_only(
            conf,
            sample,
            clients,
            stage1_json_path=stage1_json_path,
            print_prompt=args.print_prompt,
            analysis_dir=analysis_dir,
        )
    )


if __name__ == "__main__":
    main()

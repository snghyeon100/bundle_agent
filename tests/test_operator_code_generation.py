"""Let one code-generation agent select, connect, and implement library operators.

Usage:
    python tests/test_operator_code_generation.py \
        --config config_operator.yaml \
        --library tests/outputs/cluster/pog_dense_<date>/operator_library.json \
        --split test \
        --sample_idx 0

If --library is omitted, the latest clustered library for the configured
dataset under tests/outputs/cluster is used.
"""

import argparse
import ast
import asyncio
import json
import os
import sys
import time

import yaml


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from code.common import (
    build_case_view,
    candidate_labels,
    execution_summary,
    extract_python_code,
    pretty_json,
    task_semantics,
)
from code.pipeline import build_decision_case, validate_adaptive_bundle_evidence
from code.prompts import (
    MAX_EVIDENCE_ITEMS,
    _code_skeleton_text,
    _schema_text,
    _unified_case_context,
)
from code.workspace import execute_generated_code, execution_failed
from dataset import BundleZeroShotDataset, set_seed
from main import (
    create_llm_client,
    default_api_key_envs_for_provider,
    generate_content_with_retry,
    resolve_api_key_from_keys,
    stage_model,
    stage_provider,
)
from operator_learning.pipeline import (
    build_operator_capability_manifest,
    load_operator_library,
)
from operator_learning.schemas import OPERATOR_FIELDS


CAPABILITY_FILES = {
    "dataset_statistics": "count.json",
    "item_metadata": "item_info.json",
    "bundle_item_history": "bi_train.txt",
    "user_item_history": "ui_full.txt",
    "item_content_embedding": "content_feature.pt",
    "item_description_embedding": "description_feature.pt",
    "user_collaborative_embedding": "item_cf_feature.pt",
}


def _write_json(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=str)


def _write_text(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(str(value or ""))


def _latest_library(dataset):
    output_root = os.path.join(ROOT, "tests", "outputs", "cluster")
    if not os.path.isdir(output_root):
        raise FileNotFoundError(f"cluster output root not found: {output_root}")
    candidates = []
    for name in os.listdir(output_root):
        path = os.path.join(output_root, name, "operator_library.json")
        if name.startswith(f"{dataset}_") and os.path.isfile(path):
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(
            f"no clustered operator library found for {dataset}; "
            "run test_operator_clustering.py first"
        )
    return max(candidates, key=os.path.getmtime)


def _build_client(conf):
    provider = stage_provider(conf, "code_generation")
    model = stage_model(conf, "code_generation")
    api_key, env = resolve_api_key_from_keys(
        conf,
        ["operator_api_key_env", "code_generation_api_key_env", "code_api_key_env"],
        default_api_key_envs_for_provider(provider),
    )
    return {
        "client": create_llm_client(provider, api_key),
        "provider": provider,
        "model": model,
    }, {"provider": provider, "model": model, "api_key_env": env}


def _capability_bindings(source_manifest, dataset):
    available = {
        str(source.get("name")): source
        for source in source_manifest.get("sources", [])
        if isinstance(source, dict) and source.get("name")
    }
    bindings = []
    for capability, filename in CAPABILITY_FILES.items():
        source = available.get(filename)
        if source:
            bindings.append(
                {
                    "capability": capability,
                    "file": filename,
                    "path": source.get("path"),
                }
            )
    collaborative_name = f"{dataset}_LightGCN_bi_feature.pt"
    collaborative_source = available.get(collaborative_name)
    if collaborative_source:
        bindings.append(
            {
                "capability": "bundle_collaborative_embedding",
                "file": collaborative_name,
                "path": collaborative_source.get("path"),
            }
        )
    return bindings


def _extract_declared_plan(code, library):
    selected = None
    strategy = None
    issues = []
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return {
            "selected_operators": [],
            "strategy": None,
            "validation_issues": [f"generated code is not valid Python: {exc}"],
        }

    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        if target.id == "SELECTED_OPERATORS":
            try:
                selected = ast.literal_eval(node.value)
            except (ValueError, TypeError, SyntaxError):
                selected = None
        elif target.id == "STRATEGY":
            try:
                strategy = ast.literal_eval(node.value)
            except (ValueError, TypeError, SyntaxError):
                strategy = None

    allowed_names = {
        operator.get("name")
        for operator in library.get("operators", [])
        if isinstance(operator, dict) and isinstance(operator.get("name"), str)
    }
    if (
        not isinstance(selected, list)
        or not 2 <= len(selected) <= 4
        or not all(isinstance(name, str) and name.strip() for name in selected)
    ):
        issues.append(
            "SELECTED_OPERATORS must be a literal list of 2 to 4 operator names"
        )
        selected = selected if isinstance(selected, list) else []
    else:
        if len(selected) != len(set(selected)):
            issues.append("SELECTED_OPERATORS must not contain duplicates")
        unknown = sorted(set(selected) - allowed_names)
        if unknown:
            issues.append(
                "SELECTED_OPERATORS contains unknown library names: "
                + ", ".join(unknown)
            )

    if not isinstance(strategy, dict):
        issues.append("STRATEGY must be a literal object")
    else:
        for field in ("name", "description"):
            if not isinstance(strategy.get(field), str) or not strategy[field].strip():
                issues.append(f"STRATEGY.{field} must be a non-empty string")

    return {
        "selected_operators": selected,
        "strategy": strategy,
        "validation_issues": issues,
    }


def _code_generation_operators(library):
    return [
        {
            field: operator[field]
            for field in OPERATOR_FIELDS
            if field in operator
        }
        for operator in library.get("operators", [])
        if isinstance(operator, dict)
    ]


def _operator_code_generation_prompt(
    case_view,
    decision_case,
    operators,
    source_manifest,
    capability_bindings,
    output_file,
):
    unified_case = _unified_case_context(case_view, decision_case)
    labels = ", ".join(candidate_labels(case_view))
    return (
        "You are the Code Generation Agent for a bundle-completion evidence strategy.\n\n"
        "Generate only complete executable Python code, with no markdown or explanation.\n\n"
        f"{task_semantics(case_view.get('dataset'))}\n\n"
        f"Problem instance:\n{pretty_json(unified_case)}\n\n"
        "AVAILABLE OPERATORS\n"
        f"{pretty_json(operators)}\n\n"
        "Select 2 to 4 operators from this list that form one useful evidence strategy for this "
        "problem instance. Decide their execution order and connect their natural-language outputs "
        "to subsequent inputs. Implement any necessary interface adaptation in code. Do not use "
        "operators outside the supplied library.\n\n"
        "At the top of the generated program, declare the exact selected library names in execution "
        "order as a Python literal:\n"
        'SELECTED_OPERATORS = ["ExactLibraryName", "ExactLibraryName"]\n\n'
        "Also declare STRATEGY as a literal object with exactly a concise name and description. Its "
        "description must state how the selected operators connect. These declarations are inspected "
        "after generation, so do not compute them dynamically.\n\n"
        "Infer intent only from partial items. Apply the selected strategy consistently to every "
        "candidate. The program may compute candidate-indexed measurements or diagnostic values "
        "when required by an operator, but it must not rank, select, or predict the final candidate. "
        "Operator outputs such as vectors, similarities, margins, frequencies, and association "
        "statistics are internal strategy artifacts, not final evidence. Use these internal artifacts "
        "only to retrieve, filter, or select concrete item or bundle records from the available "
        "sources.\n\n"
        "FINAL EVIDENCE MATERIALIZATION\n"
        "Final evidence given to the prediction LLM must consist primarily of human-readable, "
        "source-grounded item or bundle records. Each evidence string must identify the source and "
        "record, include available item titles or bundle contents, and state the concrete relation "
        "between that record, the partial bundle, and the candidate. A numeric value alone is not "
        "valid final evidence. Numeric diagnostics may appear only as secondary annotations attached "
        "to a concrete item or bundle record. The candidate's own title alone is not supporting "
        "evidence. If no supporting record is found, return an empty evidence list instead of a score, "
        "missing-data message, interpretation, or synthetic placeholder.\n\n"
        "CAPABILITY-TO-FILE BINDINGS\n"
        f"{pretty_json(capability_bindings)}\n\n"
        "AVAILABLE WORKSPACE SOURCES\n"
        f"{pretty_json(source_manifest)}\n\n"
        "Read only listed relative paths under data/. Use the computation sources required by "
        "SELECTED_OPERATORS. For final evidence materialization, you may additionally use item "
        "metadata and bundle-item history to resolve selected item or bundle IDs into readable "
        "records. Load .pt files on CPU. Do not access network resources, parent directories, "
        "labels, ground truth, validation answers, result files, or files not listed above.\n\n"
        f"Write UTF-8 JSON to exactly: {output_file}\n\n"
        f"Each evidence array must contain zero to {MAX_EVIDENCE_ITEMS} non-empty strings. "
        "Deduplicate evidence and retain the most relevant results. Empty arrays are allowed when "
        "the selected strategy finds no evidence; never invent placeholder evidence.\n\n"
        "Use this high-level program skeleton. Replace every angle-bracket placeholder and do not "
        "leave pass statements, TODOs, ellipses, pseudocode, or undefined functions:\n\n"
        'SELECTED_OPERATORS = ["ExactLibraryName", "ExactLibraryName"]\n'
        f"{_code_skeleton_text()}\n"
        f"Required candidate labels: {labels}\n\n"
        "The written JSON must match this schema exactly:\n"
        f"{_schema_text(case_view)}"
    )


async def _run(args):
    with open(args.config, "r", encoding="utf-8") as handle:
        conf = yaml.safe_load(handle)
    set_seed(int(conf.get("seed", 45)))

    library_path = os.path.abspath(args.library or _latest_library(conf["dataset"]))
    library = load_operator_library(library_path)
    eval_conf = dict(conf)
    eval_conf["toy_eval"] = -1
    samples = BundleZeroShotDataset(eval_conf, split=args.split).get_eval_samples()
    if args.sample_idx < 0 or args.sample_idx >= len(samples):
        raise IndexError(
            f"sample_idx {args.sample_idx} out of range for {len(samples)} {args.split} samples"
        )
    sample = samples[args.sample_idx]

    client, resolved = _build_client(conf)
    workspace, source_manifest, source_capabilities = (
        build_operator_capability_manifest(conf)
    )
    capability_bindings = _capability_bindings(source_manifest, conf["dataset"])

    async def call_text(prompt, step_name):
        return await generate_content_with_retry(
            client,
            resolved["model"],
            prompt,
            conf,
            int(conf.get("operator_max_output_tokens", 15000)),
            step_name,
        )

    print(f">>> Dataset: {conf['dataset']}")
    print(f">>> Library: {library_path}")
    print(f">>> Sample: {args.split}[{args.sample_idx}] / bundle_{sample['bundle_id']}")
    print(f">>> Model: {resolved['provider']} / {resolved['model']}")

    case_view = build_case_view(sample, conf["dataset"])
    decision_case = build_decision_case(sample, conf)
    code_generation_operators = _code_generation_operators(library)
    evidence_output_file = (
        f"output/operator_evidence_bundle{sample['bundle_id']}.json"
    )
    code_prompt = _operator_code_generation_prompt(
        case_view,
        decision_case,
        code_generation_operators,
        source_manifest,
        capability_bindings,
        evidence_output_file,
    )

    print(">>> Generating code and selecting operators in one call")
    raw_code = await call_text(
        code_prompt,
        f"operator code generation for bundle_{sample['bundle_id']}",
    )
    generated_code = extract_python_code(raw_code)
    declared_plan = _extract_declared_plan(generated_code, library)
    selected_names = declared_plan["selected_operators"]
    operator_by_name = {
        operator.get("name"): operator
        for operator in code_generation_operators
        if isinstance(operator, dict)
    }
    selected_operators = [
        operator_by_name[name]
        for name in selected_names
        if isinstance(name, str) and name in operator_by_name
    ]

    stamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.abspath(
        args.output_dir
        or os.path.join(
            ROOT,
            "tests",
            "outputs",
            "operator_code",
            f"{conf['dataset']}_{stamp}",
        )
    )
    os.makedirs(output_dir, exist_ok=True)

    _write_json(
        os.path.join(output_dir, "run.json"),
        {
            "phase": "operator_code_generation",
            "dataset": conf["dataset"],
            "library_path": library_path,
            "split": args.split,
            "sample_idx": args.sample_idx,
            "bundle_id": int(sample["bundle_id"]),
            "execute": not args.skip_execute,
            **resolved,
        },
    )
    _write_json(os.path.join(output_dir, "case.json"), decision_case)
    _write_json(os.path.join(output_dir, "source_manifest.json"), source_manifest)
    _write_json(
        os.path.join(output_dir, "source_capabilities.json"),
        source_capabilities,
    )
    _write_json(
        os.path.join(output_dir, "capability_bindings.json"),
        capability_bindings,
    )
    _write_json(
        os.path.join(output_dir, "code_generation_operators.json"),
        code_generation_operators,
    )
    _write_json(os.path.join(output_dir, "declared_plan.json"), declared_plan)
    _write_json(
        os.path.join(output_dir, "selected_operators.json"),
        selected_operators,
    )
    _write_text(os.path.join(output_dir, "code_generation_input.txt"), code_prompt)
    _write_text(
        os.path.join(output_dir, "code_generation_output.txt"),
        raw_code,
    )
    _write_text(
        os.path.join(output_dir, "generated_code.py"),
        generated_code,
    )

    if selected_names:
        print(">>> Selected operators: " + " -> ".join(selected_names))
    if declared_plan["validation_issues"]:
        print(
            ">>> Plan validation issues: "
            + " | ".join(declared_plan["validation_issues"])
        )

    if args.skip_execute:
        print(">>> Execution skipped")
        print(f">>> Output: {output_dir}")
        return 0 if not declared_plan["validation_issues"] else 1

    print(">>> Executing generated code")
    execution = await asyncio.to_thread(
        execute_generated_code,
        generated_code,
        conf,
        workspace,
        evidence_output_file,
        f"operator_code_bundle{sample['bundle_id']}.py",
        "code",
    )
    evidence = execution.get("evidence_json")
    evidence_validation_issues = (
        validate_adaptive_bundle_evidence(evidence, case_view)
        if isinstance(evidence, dict)
        else ["execution failed or evidence JSON was missing"]
    )
    summary = execution_summary(execution)
    summary["plan_validation_issues"] = declared_plan["validation_issues"]
    summary["evidence_validation_issues"] = evidence_validation_issues
    summary["accepted"] = (
        not execution_failed(execution)
        and not declared_plan["validation_issues"]
        and not evidence_validation_issues
    )

    _write_json(os.path.join(output_dir, "execution_result.json"), execution)
    _write_json(os.path.join(output_dir, "execution_summary.json"), summary)
    _write_json(
        os.path.join(output_dir, "evidence.json"),
        evidence,
    )

    print(f">>> Execution accepted: {summary['accepted']}")
    if evidence_validation_issues:
        print(
            ">>> Evidence validation issues: "
            + " | ".join(evidence_validation_issues)
        )
    print(f">>> Output: {output_dir}")
    return 0 if summary["accepted"] else 1


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Select library operators and generate executable code for one bundle sample"
        )
    )
    parser.add_argument("--config", default="config_operator.yaml")
    parser.add_argument("--library", default="")
    parser.add_argument("--split", choices=("valid", "test"), default="test")
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--skip_execute", action="store_true")
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())

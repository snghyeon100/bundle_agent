"""Run one final prediction from executed spec-first strategy programs.

Usage:
    python tests/test_spec_first_operator_prediction.py \
        --config config_operator.yaml \
        --operator_output tests/outputs/operators/candidate_disambiguation_spec_first_code \
        --case_id bundle_11015

This is an end-to-end wiring test over an induction case. It is not an
out-of-sample evaluation of strategy generalization.
"""

import argparse
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

from code.common import parse_json_from_text
from main import (
    create_llm_client,
    default_api_key_envs_for_provider,
    generate_content_with_retry,
    resolve_api_key_from_keys,
    stage_model,
    stage_provider,
)
from online_hypothesis_program.schemas import validate_prediction_result
from operator_learning.prompts import strategy_evidence_prediction_prompt
from operator_learning.spec_first_prediction import build_strategy_evidence


def _read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=str)


def _write_text(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(str(value or ""))


def _build_client(conf):
    provider = stage_provider(conf, "prediction")
    model = stage_model(conf, "prediction")
    api_key, env = resolve_api_key_from_keys(
        conf,
        [
            "online_prediction_api_key_env",
            "code_prediction_api_key_env",
            "operator_api_key_env",
            "code_generation_api_key_env",
            "code_api_key_env",
        ],
        default_api_key_envs_for_provider(provider),
    )
    return {
        "client": create_llm_client(provider, api_key),
        "provider": provider,
        "model": model,
    }, {"provider": provider, "model": model, "api_key_env": env}


def _true_label(case, evaluation):
    gt_item_id = int(evaluation["ground_truth_item_id"])
    for candidate in case.get("candidate_items", []):
        if int(candidate["item_id"]) == gt_item_id:
            return str(candidate["label"])
    raise ValueError("ground-truth item is absent from the case candidate items")


async def _run(args):
    with open(args.config, "r", encoding="utf-8") as handle:
        conf = yaml.safe_load(handle)

    operator_output = os.path.abspath(args.operator_output)
    executions_path = os.path.join(operator_output, "execution_results.json")
    if not os.path.isfile(executions_path):
        raise FileNotFoundError(
            "execution_results.json is missing; execute the generated programs before "
            "running final prediction"
        )
    executions = _read_json(executions_path)
    reports = [
        report
        for report in executions.get("programs", [])
        if str(report.get("case_id") or "") == args.case_id
    ]
    case_dir = os.path.join(operator_output, "cases", args.case_id)
    case = _read_json(os.path.join(case_dir, "prompt_case.json"))
    specs = _read_json(os.path.join(case_dir, "strategy_specs.json"))
    evaluation = _read_json(os.path.join(case_dir, "evaluation.json"))
    if case.get("dataset") != conf.get("dataset"):
        raise ValueError(
            f"dataset mismatch: config={conf.get('dataset')} "
            f"operator_output={case.get('dataset')}"
        )

    labels = [str(item["label"]) for item in case.get("candidate_items", [])]
    strategy_evidence = build_strategy_evidence(
        specs=specs,
        execution_reports=reports,
        candidate_labels=labels,
        max_contexts_per_candidate=int(
            conf.get("operator_prediction_max_contexts_per_candidate", 0)
        ),
        max_context_chars=int(conf.get("operator_prediction_max_context_chars", 0)),
    )
    if not strategy_evidence:
        raise ValueError(f"no successful executed strategies found for {args.case_id}")

    prompt = strategy_evidence_prediction_prompt(
        dataset=case["dataset"],
        partial_items=case.get("partial_items", []),
        candidate_items=case.get("candidate_items", []),
        strategy_evidence=strategy_evidence,
    )
    client, resolved = _build_client(conf)

    print(f">>> Dataset: {case['dataset']}")
    print(f">>> Case: {args.case_id} (induction-case wiring test)")
    print(f">>> Strategies: {len(strategy_evidence)} executed specs")
    print(f">>> Model: {resolved['provider']} / {resolved['model']}")
    print(">>> Predicting from candidate-specific strategy contexts")

    raw = await generate_content_with_retry(
        client,
        resolved["model"],
        prompt,
        conf,
        int(conf.get("operator_prediction_max_output_tokens", 2000)),
        "spec-first strategy evidence prediction",
    )
    prediction = parse_json_from_text(raw)
    issues = validate_prediction_result(prediction, labels)
    if issues:
        raise ValueError("invalid final prediction: " + " | ".join(issues))

    true_label = _true_label(case, evaluation)
    ranking = prediction["ranking"]
    gt_rank = ranking.index(true_label) + 1
    hit = int(prediction["prediction"] == true_label)
    result = {
        **prediction,
        "true_label": true_label,
        "hit": hit,
        "gt_rank": gt_rank,
        "reciprocal_rank": 1.0 / gt_rank,
    }

    stamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.abspath(
        args.output_dir
        or os.path.join(
            ROOT,
            "tests",
            "outputs",
            "spec_first_operator_prediction",
            f"{case['dataset']}_{stamp}",
        )
    )
    os.makedirs(output_dir, exist_ok=True)
    _write_json(
        os.path.join(output_dir, "run.json"),
        {
            "phase": "spec_first_operator_prediction",
            "evaluation_scope": "induction_case_wiring_test",
            "dataset": case["dataset"],
            "case_id": args.case_id,
            "operator_output": operator_output,
            **resolved,
        },
    )
    _write_json(os.path.join(output_dir, "case.json"), case)
    _write_json(
        os.path.join(output_dir, "rendered_strategy_evidence.json"),
        strategy_evidence,
    )
    _write_text(os.path.join(output_dir, "input.txt"), prompt)
    _write_text(os.path.join(output_dir, "output.txt"), raw)
    _write_json(os.path.join(output_dir, "prediction.json"), result)

    print(f">>> Prediction: {prediction['prediction']}")
    print(f">>> Ranking: {', '.join(ranking)}")
    print(f">>> True label: {true_label}")
    print(f">>> Hit: {hit}")
    print(f">>> GT rank: {gt_rank}")
    print(f">>> Reciprocal rank: {1.0 / gt_rank:.4f}")
    print(f">>> Output: {output_dir}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Predict one induction case from executed spec-first strategies"
    )
    parser.add_argument("--config", default="config_operator.yaml")
    parser.add_argument("--operator_output", required=True)
    parser.add_argument("--case_id", required=True)
    parser.add_argument("--output_dir", default="")
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())

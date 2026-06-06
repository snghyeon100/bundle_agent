#!/usr/bin/env python3
"""Analyze baseline-to-two-stage transitions without requiring pandas."""

import argparse
import ast
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


LABELS = tuple("ABCDEFGHIJ")
SIGNALS = (
    "metadata_fit",
    "bi_signal",
    "ui_signal",
    "embedding_similarity",
    "ui_lightgcn_similarity",
    "bi_lightgcn_similarity",
)


def read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sample_key(row):
    return int(row["bundle_id"]), int(row["true_indice"])


def as_int(value):
    return int(float(value))


def parse_json(value):
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else None
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def numeric_value(value):
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    if isinstance(value, str):
        match = re.fullmatch(r"(?:cate_match=)?(True|False)", value.strip())
        if match:
            return float(match.group(1) == "True")
        try:
            number = float(value)
            return number if math.isfinite(number) else None
        except ValueError:
            return None
    return None


def candidate_values(evidence, signal):
    values = {}
    candidate_evidence = evidence.get("candidate_evidence") or {}
    for label in LABELS:
        candidate = candidate_evidence.get(label) or {}
        retrieved = candidate.get("retrieved_values") or {}
        value = numeric_value(retrieved.get(signal))
        if value is not None:
            values[label] = value
    return values


def evidence_proxy(evidence):
    """Return an evidence-only rank aggregation proxy.

    Stage 1 is forbidden from predicting. This proxy min-max normalizes every
    available non-tied numeric signal and averages them with equal weight.
    """
    if not isinstance(evidence, dict):
        return {
            "prediction": None,
            "signals_used": [],
            "signal_tops": {},
            "scores": {},
        }

    scores = defaultdict(list)
    signals_used = []
    signal_tops = {}
    for signal in SIGNALS:
        values = candidate_values(evidence, signal)
        if len(values) < 2:
            continue
        low, high = min(values.values()), max(values.values())
        if low == high:
            continue
        signals_used.append(signal)
        signal_tops[signal] = sorted(label for label, value in values.items() if value == high)
        for label, value in values.items():
            scores[label].append((value - low) / (high - low))

    averaged = {
        label: sum(values) / len(values)
        for label, values in scores.items()
        if values
    }
    if not averaged:
        prediction = None
    else:
        high = max(averaged.values())
        top = [label for label, value in averaged.items() if math.isclose(value, high)]
        prediction = top[0] if len(top) == 1 else None
    return {
        "prediction": prediction,
        "signals_used": signals_used,
        "signal_tops": signal_tops,
        "scores": averaged,
    }


def transition(before, after):
    labels = {
        (0, 0): "both_fail",
        (0, 1): "fail_to_success",
        (1, 0): "success_to_fail",
        (1, 1): "both_success",
    }
    return labels[(int(before), int(after))]


def exact_mcnemar_p(recovered, damaged):
    discordant = recovered + damaged
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(recovered, damaged) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def mean(values):
    return sum(values) / len(values) if values else None


def analyze_dataset(name, baseline_path, two_stage_path):
    baseline_rows = {sample_key(row): row for row in read_csv(baseline_path)}
    two_stage_rows = {sample_key(row): row for row in read_csv(two_stage_path)}
    keys = sorted(baseline_rows.keys() & two_stage_rows.keys())
    if len(keys) != len(baseline_rows) or len(keys) != len(two_stage_rows):
        raise ValueError(f"{name}: baseline and two-stage sample keys do not match exactly")

    samples = []
    base_transitions = Counter()
    controlled_transitions = Counter()
    proxy_transitions = Counter()
    signal_metrics = {
        signal: Counter(missing=0, tied=0, discriminative=0, true_in_top=0,
                        true_unique_top=0, final_in_top=0, final_hit_true_top=0,
                        final_hit_false_top=0, false_top=0)
        for signal in SIGNALS
    }
    confidence = defaultdict(Counter)
    evidence_present = 0
    execution_ok = 0
    proxy_valid = 0
    proxy_hit = 0
    proxy_final_hit = 0
    proxy_agreement = 0
    no_proxy_final_hit = 0
    candidate_same = 0
    overlaps = []
    input_lengths = []

    for key in keys:
        baseline = baseline_rows[key]
        final = two_stage_rows[key]
        base_hit = as_int(baseline["hit"])
        final_hit = as_int(final["hit"])
        base_transition = transition(base_hit, final_hit)
        base_transitions[base_transition] += 1

        baseline_candidates = ast.literal_eval(baseline["candidate_indices"])
        final_candidates = ast.literal_eval(final["candidate_indices"])
        same_candidates = baseline_candidates == final_candidates
        overlap = len(set(baseline_candidates) & set(final_candidates))
        overlaps.append(overlap)
        input_lengths.append(len(ast.literal_eval(final["input_indices"])))
        if same_candidates:
            candidate_same += 1
            controlled_transitions[base_transition] += 1

        evidence = parse_json(final.get("two_stage_evidence_json"))
        execution = parse_json(final.get("two_stage_execution_summary")) or {}
        if evidence:
            evidence_present += 1
        if execution.get("returncode") == 0 and execution.get("evidence_json_present"):
            execution_ok += 1

        proxy = evidence_proxy(evidence)
        proxy_prediction = proxy["prediction"]
        true_label = final["true_option_char"]
        proxy_is_hit = int(proxy_prediction == true_label) if proxy_prediction else None
        proxy_to_final = ""
        if proxy_prediction:
            proxy_valid += 1
            proxy_hit += proxy_is_hit
            proxy_final_hit += final_hit
            proxy_agreement += int(proxy_prediction == final["prediction"])
            proxy_to_final = transition(proxy_is_hit, final_hit)
            proxy_transitions[proxy_to_final] += 1
        else:
            no_proxy_final_hit += final_hit

        correct_unique_signals = []
        wrong_unique_signals = []
        for signal in SIGNALS:
            values = candidate_values(evidence or {}, signal)
            metrics = signal_metrics[signal]
            if len(values) < 2:
                metrics["missing"] += 1
                continue
            if min(values.values()) == max(values.values()):
                metrics["tied"] += 1
                continue
            metrics["discriminative"] += 1
            high = max(values.values())
            top = [label for label, value in values.items() if value == high]
            if true_label in top:
                metrics["true_in_top"] += 1
                metrics["final_hit_true_top"] += final_hit
            else:
                metrics["false_top"] += 1
                metrics["final_hit_false_top"] += final_hit
            if final["prediction"] in top:
                metrics["final_in_top"] += 1
            if len(top) == 1:
                if top[0] == true_label:
                    metrics["true_unique_top"] += 1
                    correct_unique_signals.append(signal)
                else:
                    wrong_unique_signals.append(signal)

        confidence[final.get("two_stage_confidence", "")]["n"] += 1
        confidence[final.get("two_stage_confidence", "")]["hit"] += final_hit
        samples.append(
            {
                "dataset": name,
                "bundle_id": key[0],
                "true_indice": key[1],
                "true_option_char": true_label,
                "baseline_prediction": baseline["prediction"],
                "two_stage_prediction": final["prediction"],
                "baseline_hit": base_hit,
                "two_stage_hit": final_hit,
                "baseline_to_two_stage": base_transition,
                "candidate_set_same": same_candidates,
                "candidate_overlap_count": overlap,
                "stage1_evidence_present": bool(evidence),
                "stage1_proxy_prediction": proxy_prediction or "",
                "stage1_proxy_hit": "" if proxy_is_hit is None else proxy_is_hit,
                "stage1_to_stage2": proxy_to_final,
                "stage1_signals_used": "|".join(proxy["signals_used"]),
                "stage1_correct_unique_signals": "|".join(correct_unique_signals),
                "stage1_wrong_unique_signals": "|".join(wrong_unique_signals),
                "two_stage_confidence": final.get("two_stage_confidence", ""),
                "input_indices": final["input_indices"],
                "baseline_candidate_indices": baseline["candidate_indices"],
                "two_stage_candidate_indices": final["candidate_indices"],
                "two_stage_reasoning": final.get("two_stage_reasoning", ""),
                "input_str": final["input_str"],
                "two_stage_target_str": final["target_str"],
            }
        )

    n = len(keys)
    base_hits = sum(as_int(baseline_rows[key]["hit"]) for key in keys)
    final_hits = sum(as_int(two_stage_rows[key]["hit"]) for key in keys)
    recovered = base_transitions["fail_to_success"]
    damaged = base_transitions["success_to_fail"]
    controlled_n = sum(controlled_transitions.values())
    controlled_base_hits = controlled_transitions["both_success"] + controlled_transitions["success_to_fail"]
    controlled_final_hits = controlled_transitions["both_success"] + controlled_transitions["fail_to_success"]
    controlled_recovered = controlled_transitions["fail_to_success"]
    controlled_damaged = controlled_transitions["success_to_fail"]

    metrics = {
        "dataset": name,
        "n": n,
        "baseline_accuracy": ratio(base_hits, n),
        "two_stage_accuracy": ratio(final_hits, n),
        "delta_percentage_points": 100 * (final_hits - base_hits) / n,
        "baseline_to_two_stage": dict(base_transitions),
        "recovery_rate_among_baseline_failures": ratio(recovered, n - base_hits),
        "damage_rate_among_baseline_successes": ratio(damaged, base_hits),
        "damage_rate_overall": ratio(damaged, n),
        "mcnemar_exact_p": exact_mcnemar_p(recovered, damaged),
        "candidate_set": {
            "same_n": candidate_same,
            "same_rate": ratio(candidate_same, n),
            "mean_overlap_of_10": mean(overlaps),
        },
        "candidate_controlled": {
            "n": controlled_n,
            "baseline_accuracy": ratio(controlled_base_hits, controlled_n),
            "two_stage_accuracy": ratio(controlled_final_hits, controlled_n),
            "delta_percentage_points": (
                100 * (controlled_final_hits - controlled_base_hits) / controlled_n
                if controlled_n else None
            ),
            "transitions": dict(controlled_transitions),
            "recovery_rate_among_baseline_failures": ratio(
                controlled_recovered, controlled_n - controlled_base_hits
            ),
            "damage_rate_among_baseline_successes": ratio(
                controlled_damaged, controlled_base_hits
            ),
            "mcnemar_exact_p": exact_mcnemar_p(controlled_recovered, controlled_damaged),
        },
        "stage1": {
            "evidence_present_n": evidence_present,
            "execution_ok_n": execution_ok,
            "proxy_definition": (
                "Equal-weight average of min-max-normalized, non-tied numeric evidence signals"
            ),
            "proxy_valid_n": proxy_valid,
            "proxy_accuracy_on_valid": ratio(proxy_hit, proxy_valid),
            "stage2_accuracy_on_proxy_valid": ratio(proxy_final_hit, proxy_valid),
            "stage2_delta_percentage_points_on_proxy_valid": (
                100 * (proxy_final_hit - proxy_hit) / proxy_valid if proxy_valid else None
            ),
            "proxy_to_stage2": dict(proxy_transitions),
            "stage2_correction_rate_among_proxy_failures": ratio(
                proxy_transitions["fail_to_success"], proxy_valid - proxy_hit
            ),
            "stage2_damage_rate_among_proxy_successes": ratio(
                proxy_transitions["success_to_fail"], proxy_hit
            ),
            "stage2_agreement_with_proxy": ratio(proxy_agreement, proxy_valid),
            "stage2_accuracy_without_unique_proxy": ratio(no_proxy_final_hit, n - proxy_valid),
        },
        "signals": {},
        "confidence": {},
        "mean_input_items": mean(input_lengths),
    }
    for signal, counts in signal_metrics.items():
        discriminative = counts["discriminative"]
        true_top = counts["true_in_top"]
        false_top = counts["false_top"]
        metrics["signals"][signal] = {
            **dict(counts),
            "discriminative_rate": ratio(discriminative, n),
            "true_in_top_rate_when_discriminative": ratio(true_top, discriminative),
            "final_follows_top_rate_when_discriminative": ratio(
                counts["final_in_top"], discriminative
            ),
            "final_accuracy_when_true_in_top": ratio(
                counts["final_hit_true_top"], true_top
            ),
            "final_accuracy_when_true_not_in_top": ratio(
                counts["final_hit_false_top"], false_top
            ),
        }
    for label, counts in confidence.items():
        metrics["confidence"][label or "missing"] = {
            "n": counts["n"],
            "accuracy": ratio(counts["hit"], counts["n"]),
        }
    return metrics, samples


def write_samples(path, samples):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(samples[0]))
        writer.writeheader()
        writer.writerows(samples)


def profile_interaction_file(path, num_items):
    rows = 0
    edges = 0
    item_degrees = Counter()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            items = line.rstrip("\n").split(", ")[1:]
            rows += 1
            edges += len(items)
            item_degrees.update(items)
    return {
        "rows": rows,
        "edges": edges,
        "edges_per_item": ratio(edges, num_items),
        "unique_items": len(item_degrees),
        "item_coverage": ratio(len(item_degrees), num_items),
        "mean_degree_among_present_items": ratio(edges, len(item_degrees)),
    }


def profile_data_dir(path):
    data_dir = Path(path)
    with (data_dir / "count.json").open("r", encoding="utf-8") as handle:
        counts = json.load(handle)
    num_items = int(counts["#I"])
    profile = {"count_json": counts}
    for filename in ("bi_train.txt", "ui_full.txt", "ui_full_with_duplicates.txt"):
        interaction_path = data_dir / filename
        if interaction_path.exists():
            profile[filename] = profile_interaction_file(interaction_path, num_items)
    return profile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pog-baseline", required=True)
    parser.add_argument("--pog-two-stage", required=True)
    parser.add_argument("--dense-baseline", required=True)
    parser.add_argument("--dense-two-stage", required=True)
    parser.add_argument("--pog-data-dir")
    parser.add_argument("--dense-data-dir")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_metrics = {}
    for name, baseline, two_stage in (
        ("pog", args.pog_baseline, args.pog_two_stage),
        ("pog_dense", args.dense_baseline, args.dense_two_stage),
    ):
        metrics, samples = analyze_dataset(name, baseline, two_stage)
        data_dir = args.pog_data_dir if name == "pog" else args.dense_data_dir
        if data_dir:
            metrics["data_profile"] = profile_data_dir(data_dir)
        all_metrics[name] = metrics
        write_samples(output_dir / f"sample_transitions_{name}.csv", samples)

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(all_metrics, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

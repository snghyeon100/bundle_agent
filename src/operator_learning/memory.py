"""Deterministic memory and deduplication for candidate-program specs."""

from difflib import SequenceMatcher
import re

from .schemas import normalize_operator


MEMORY_FIELDS = (
    "name",
    "hypothesis",
    "required_sources",
    "evidence_types",
)


def _normalized_text(value):
    return " ".join(
        re.findall(r"[a-z0-9]+", str(value or "").lower())
    )


def _token_set(value):
    return set(_normalized_text(value).split())


def _jaccard(left, right):
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def operator_memory_view(operator):
    """Return a compact forbidden signature, not a generative program template."""
    normalized = normalize_operator(operator)
    if not isinstance(normalized, dict):
        return normalized
    return {
        field: normalized[field]
        for field in MEMORY_FIELDS
        if field in normalized
    }


def operator_program_signature(operator):
    """Build an exact normalized candidate-program signature."""
    view = operator_memory_view(operator)
    if not isinstance(view, dict):
        return ()
    return (
        tuple(sorted(view.get("required_sources", []))),
        tuple(sorted(view.get("evidence_types", []))),
        _normalized_text(view.get("hypothesis")),
        _normalized_text(view.get("pseudocode")),
        str(view.get("output_contract") or ""),
    )


def _family_signature(operator):
    view = operator_memory_view(operator)
    if not isinstance(view, dict):
        return ()
    return (
        tuple(sorted(view.get("required_sources", []))),
        tuple(sorted(view.get("evidence_types", []))),
    )


def operator_spec_similarity(left, right):
    """Return deterministic lexical similarity for specs in the same family."""
    left_view = normalize_operator(left)
    right_view = normalize_operator(right)
    if not isinstance(left_view, dict) or not isinstance(right_view, dict):
        return 0.0
    if _family_signature(left_view) != _family_signature(right_view):
        return 0.0

    left_hypothesis = _normalized_text(left_view.get("hypothesis"))
    right_hypothesis = _normalized_text(right_view.get("hypothesis"))
    left_steps = _normalized_text(left_view.get("pseudocode"))
    right_steps = _normalized_text(right_view.get("pseudocode"))
    hypothesis_similarity = max(
        _jaccard(_token_set(left_hypothesis), _token_set(right_hypothesis)),
        SequenceMatcher(None, left_hypothesis, right_hypothesis).ratio(),
    )
    pseudocode_similarity = max(
        _jaccard(_token_set(left_steps), _token_set(right_steps)),
        SequenceMatcher(None, left_steps, right_steps).ratio(),
    )
    if not left_steps or not right_steps:
        return hypothesis_similarity
    return 0.6 * hypothesis_similarity + 0.4 * pseudocode_similarity


def compact_operator_memory(operators, *, max_size=24):
    """Build a bounded list of compact forbidden program signatures."""
    limit = int(max_size)
    if limit < 1:
        raise ValueError("operator memory max size must be at least 1")

    unique = []
    seen = set()
    for operator in operators or []:
        view = operator_memory_view(operator)
        signature = operator_program_signature(view)
        if not isinstance(view, dict) or not signature or signature in seen:
            continue
        seen.add(signature)
        unique.append(view)
    if len(unique) <= limit:
        return unique

    selected = []
    selected_families = set()
    for operator in reversed(unique):
        family = _family_signature(operator)
        if family in selected_families:
            continue
        selected.append(operator)
        selected_families.add(family)
        if len(selected) >= limit:
            break
    if len(selected) < limit:
        selected_signatures = {
            operator_program_signature(operator)
            for operator in selected
        }
        for operator in reversed(unique):
            signature = operator_program_signature(operator)
            if signature in selected_signatures:
                continue
            selected.append(operator)
            selected_signatures.add(signature)
            if len(selected) >= limit:
                break
    selected.reverse()
    return selected


def count_exact_memory_matches(operators, memory):
    memory_signatures = {
        operator_program_signature(operator)
        for operator in memory or []
    }
    memory_signatures.discard(())
    return sum(
        operator_program_signature(operator) in memory_signatures
        for operator in operators or []
    )


def deduplicate_operator_pool(operators, *, similarity_threshold=0.9):
    """Group exact or high-similarity specs without an LLM call."""
    threshold = float(similarity_threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("similarity_threshold must be between 0 and 1")

    groups = []
    for operator in operators or []:
        normalized = normalize_operator(operator, preserve_metadata=True)
        if not isinstance(normalized, dict):
            continue
        matched_group = None
        matched_similarity = 0.0
        for group in groups:
            representative = group["representative"]
            same_named_family = (
                _normalized_text(normalized.get("name"))
                == _normalized_text(representative.get("name"))
                and _family_signature(normalized)
                == _family_signature(representative)
            )
            exact = (
                operator_program_signature(normalized)
                == operator_program_signature(representative)
            )
            similarity = (
                1.0
                if exact or same_named_family
                else operator_spec_similarity(normalized, representative)
            )
            if similarity >= threshold and similarity > matched_similarity:
                matched_group = group
                matched_similarity = similarity
        if matched_group is None:
            groups.append(
                {
                    "representative": normalized,
                    "members": [normalized],
                    "member_similarities": [1.0],
                }
            )
        else:
            matched_group["members"].append(normalized)
            matched_group["member_similarities"].append(
                round(matched_similarity, 6)
            )

    representatives = []
    group_views = []
    used_names = set()
    for index, group in enumerate(groups, start=1):
        representative = dict(group["representative"])
        member_ids = [
            member.get("operator_id")
            for member in group["members"]
            if isinstance(member.get("operator_id"), str)
            and member["operator_id"].strip()
        ]
        origin_case_ids = list(
            dict.fromkeys(
                member.get("origin_case_id")
                for member in group["members"]
                if isinstance(member.get("origin_case_id"), str)
                and member["origin_case_id"].strip()
            )
        )
        representative["operator_id"] = f"candidate_program_{index:03d}"
        base_name = str(representative.get("name") or f"CandidateProgram{index}")
        unique_name = base_name
        suffix = 2
        while unique_name in used_names:
            unique_name = f"{base_name}Variant{suffix}"
            suffix += 1
        representative["name"] = unique_name
        used_names.add(unique_name)
        representative["member_operator_ids"] = member_ids
        representative["origin_case_ids"] = origin_case_ids
        representatives.append(representative)
        group_views.append(
            {
                "group_id": f"dedup_{index:03d}",
                "representative_operator_id": representative["operator_id"],
                "member_operator_ids": member_ids,
                "origin_case_ids": origin_case_ids,
                "member_similarities": group["member_similarities"],
            }
        )
    return {
        "operators": representatives,
        "groups": group_views,
        "raw_operator_count": len(operators or []),
        "deduplicated_operator_count": len(representatives),
        "similarity_threshold": threshold,
    }

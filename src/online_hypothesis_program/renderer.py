"""Render raw program provenance into compact evidence for the prediction LLM."""


RELATION_TEXT = {
    "historical_bundle_context": (
        "Retrieved from a historical bundle context connected to the partial items."
    ),
    "related_item_context": (
        "Retrieved through a source-grounded item-interaction neighborhood of the "
        "partial items."
    ),
    "semantic_neighbor_context": (
        "Retrieved as a bounded neighbor under the program's semantic representation."
    ),
    "category_profile_context": (
        "Retrieved from an observed category relation in relevant corpus contexts."
    ),
}


def _relation_summary(record):
    evidence_type = str(record.get("type") or "")
    if evidence_type in RELATION_TEXT:
        return RELATION_TEXT[evidence_type]
    readable = evidence_type.replace("_", " ").strip()
    return (
        f"Retrieved through the source-grounded relation: {readable}."
        if readable
        else "Retrieved through a source-grounded corpus relation."
    )


def _representative_contexts(source_api, record, *, max_contexts=2):
    bundle_ids = [
        int(bundle_id)
        for bundle_id in record.get("related_bundle_ids", [])
    ][:max_contexts]
    if not bundle_ids or "bundle_item_history" not in source_api.available_sources:
        return []
    mapping = source_api.get_items_for_bundles(bundle_ids)
    contexts = []
    for bundle_id in bundle_ids:
        item_texts = [
            source_api.item_text(item_id)
            for item_id in mapping.get(bundle_id, [])[:4]
        ]
        if item_texts:
            contexts.append(item_texts)
    return contexts


def render_search_evidence(
    *,
    hypotheses,
    programs,
    executions,
    source_api,
    answer_options,
    total_candidate_budget,
):
    """Return model-facing evidence and the retained raw candidate IDs."""
    hypothesis_by_id = {
        hypothesis["id"]: hypothesis
        for hypothesis in hypotheses
        if isinstance(hypothesis, dict) and hypothesis.get("id")
    }
    program_by_hypothesis = {
        program["hypothesis_id"]: program
        for program in programs
        if isinstance(program, dict) and program.get("hypothesis_id")
    }
    option_labels_by_item = {}
    for option in answer_options:
        option_labels_by_item.setdefault(int(option["item_id"]), []).append(
            str(option["label"])
        )

    accepted_ids = []
    accepted_set = set()
    model_results = []
    for hypothesis_id, hypothesis in hypothesis_by_id.items():
        execution = executions.get(hypothesis_id, {})
        status = str(execution.get("status") or "not_executed")
        entry = {
            "hypothesis_id": hypothesis_id,
            "intent": hypothesis.get("intent", ""),
            "missing_role": hypothesis.get("missing_role", ""),
            "observed_cues": list(hypothesis.get("observed_cues", [])),
            "search_status": status,
            "retrieved_examples": [],
        }
        result = execution.get("result")
        if status != "success" or not isinstance(result, dict):
            model_results.append(entry)
            continue

        evidence_by_id = {
            record.get("evidence_id"): record
            for record in result.get("evidence_records", [])
            if isinstance(record, dict)
        }
        for proposal in result.get("candidate_proposals", []):
            if not isinstance(proposal, dict):
                continue
            try:
                item_id = int(proposal["item_id"])
                item_text = source_api.item_text(item_id)
            except (KeyError, TypeError, ValueError):
                continue
            if (
                item_id not in accepted_set
                and len(accepted_ids) >= int(total_candidate_budget)
            ):
                continue
            if item_id not in accepted_set:
                accepted_set.add(item_id)
                accepted_ids.append(item_id)

            support = []
            for evidence_ref in proposal.get("evidence_refs", [])[:2]:
                record = evidence_by_id.get(evidence_ref)
                if not isinstance(record, dict):
                    continue
                rendered = {
                    "relation": _relation_summary(record),
                    "representative_context_items": _representative_contexts(
                        source_api,
                        record,
                    ),
                }
                if rendered not in support:
                    support.append(rendered)
            entry["retrieved_examples"].append(
                {
                    "example_ref": f"R{len(accepted_ids)}",
                    "item_text": item_text,
                    "matching_answer_options": option_labels_by_item.get(item_id, []),
                    "support": support,
                }
            )
        model_results.append(entry)

    return {
        "model_view": model_results,
        "retained_candidate_item_ids": accepted_ids,
        "program_names": {
            hypothesis_id: program_by_hypothesis.get(hypothesis_id, {}).get("name")
            for hypothesis_id in hypothesis_by_id
        },
    }

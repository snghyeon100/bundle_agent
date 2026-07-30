"""Resolve retrieval IDs into compact readable exemplars for LLM2."""


def _representative_bundle_contexts(source_api, record, *, max_contexts=2):
    supporting_context = record.get("supporting_context", {})
    bundle_ids = [
        int(bundle_id)
        for bundle_id in supporting_context.get("bundle_ids", [])
    ][: int(max_contexts)]
    if not bundle_ids or "bundle_item_history" not in source_api.available_sources:
        return []
    mapping = source_api.get_items_for_bundles(bundle_ids)
    contexts = []
    for bundle_id in bundle_ids:
        texts = [
            source_api.item_text(item_id)
            for item_id in mapping.get(bundle_id, [])[:5]
        ]
        texts = [text for text in texts if text]
        if texts:
            contexts.append(texts)
    return contexts


def _representative_user_contexts(source_api, record, *, max_contexts=2):
    supporting_context = record.get("supporting_context", {})
    user_ids = [
        int(user_id)
        for user_id in supporting_context.get("user_ids", [])
    ][: int(max_contexts)]
    if not user_ids or "user_item_history" not in source_api.available_sources:
        return []
    mapping = source_api.get_items_for_users(user_ids)
    contexts = []
    for user_id in user_ids:
        texts = [
            source_api.item_text(item_id)
            for item_id in mapping.get(user_id, [])[:5]
        ]
        texts = [text for text in texts if text]
        if texts:
            contexts.append(texts)
    return contexts


def _render_provenance(source_api, record, *, retrieved_item_id=None):
    related_item_texts = []
    seen = set()
    supporting_context = record.get("supporting_context", {})
    for item_id in supporting_context.get("item_ids", []):
        if (
            retrieved_item_id is not None
            and int(item_id) == int(retrieved_item_id)
        ):
            continue
        text = source_api.item_text(int(item_id))
        if text and text not in seen:
            seen.add(text)
            related_item_texts.append(text)
        if len(related_item_texts) >= 4:
            break
    return {
        "source": str(record.get("source") or ""),
        "relation": str(record.get("relation") or ""),
        "related_item_texts": related_item_texts,
        "representative_bundle_contexts": _representative_bundle_contexts(
            source_api,
            record,
        ),
        "representative_user_contexts": _representative_user_contexts(
            source_api,
            record,
        ),
    }


def render_retrieval_evidence(
    *,
    programs,
    executions,
    source_api,
    answer_options,
    max_items_per_hypothesis=5,
    max_supporting_contexts_per_item=2,
):
    """Return an ID-free hypothesis/exemplar view plus retrieval diagnostics."""
    valid_programs = [
        program
        for program in programs
        if isinstance(program, dict) and program.get("id")
    ]
    labels_by_item = {}
    for option in answer_options:
        labels_by_item.setdefault(int(option["item_id"]), []).append(option["label"])

    model_view = []
    retrieved_item_ids_by_program = {}
    merged_item_ids = []
    seen_merged = set()
    successful_programs = 0
    total_retrieved = 0

    for program in valid_programs:
        program_id = program["id"]
        execution = executions.get(program_id, {})
        status = execution.get("status", "not_executed")
        packet = execution.get("result") if status == "success" else None
        raw_items = (
            packet.get("retrieved_items", [])
            if isinstance(packet, dict)
            else []
        )
        raw_items = [
            item for item in raw_items if isinstance(item, dict)
        ][: int(max_items_per_hypothesis)]
        successful_programs += int(status == "success")

        item_ids = [int(item["item_id"]) for item in raw_items]
        retrieved_item_ids_by_program[program_id] = item_ids
        total_retrieved += len(item_ids)
        for item_id in item_ids:
            if item_id not in seen_merged:
                seen_merged.add(item_id)
                merged_item_ids.append(item_id)

        entry = {
            "program_id": program_id,
            "completion_hypothesis": program.get("hypothesis", ""),
            "strategy": {
                "reference": program.get("strategy", {}).get("reference", ""),
                "retrieval": program.get("strategy", {}).get("retrieval", ""),
            },
            "program_execution_status": status,
            "retrieved_exemplars": [],
        }
        for item in raw_items:
            item_id = int(item["item_id"])
            entry["retrieved_exemplars"].append(
                {
                    "item_text": source_api.item_text(item_id),
                    "matching_answer_options": labels_by_item.get(item_id, []),
                    "provenance": [
                        _render_provenance(
                            source_api,
                            record,
                            retrieved_item_id=item_id,
                        )
                        for record in item.get("provenance", [])[
                            : int(max_supporting_contexts_per_item)
                        ]
                        if isinstance(record, dict)
                    ],
                }
            )
        model_view.append(entry)

    overlap_item_ids = [
        item_id for item_id in merged_item_ids if item_id in labels_by_item
    ]
    return {
        "model_view": model_view,
        "retrieved_item_ids_by_program": retrieved_item_ids_by_program,
        "merged_retrieved_item_ids": merged_item_ids,
        "retrieval_counts": {
            "successful_program_count": successful_programs,
            "failed_program_count": len(valid_programs) - successful_programs,
            "retrieved_example_count": total_retrieved,
            "unique_retrieved_item_count": len(merged_item_ids),
            "answer_option_overlap_count": len(overlap_item_ids),
            "answer_option_overlap_labels": [
                label
                for item_id in overlap_item_ids
                for label in labels_by_item[item_id]
            ],
        },
    }

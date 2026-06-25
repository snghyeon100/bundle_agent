import json

from progressive_signal_agent.common import candidate_labels, task_semantics

from .affordance_graph import render_affordance_relation_map


def _dump(value):
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def stage1_code_prompt(
    case_view,
    source_manifest,
    affordance_graph,
    output_file,
    max_evidence_chars,
):
    """Stage 1: 초기 넓은 스캔. 다양한 소스에서 직접적이고 빠른 신호를 수집한다."""
    labels = ", ".join(candidate_labels(case_view))

    return (
        "You are the Signal Python Code Generator in a bundle-completion system. Generate only complete "
        "executable Python code without markdown fences or explanation. The script runs with the allowed workspace as "
        "its current directory.\n\n"
        f"{task_semantics(case_view['dataset'])}\n\n"
        "Your role in Stage 1 is a broad initial scan. Collect diverse, direct, factual signals from as many available "
        "sources as possible."
        "Rules:\n"
        "- Signals must be factual outputs of executed code. Do not choose, rank, or recommend.\n"
        "- Use bundle_id only as a bundle entity, user IDs only as user entities, item IDs only as item entities.\n"
        "- CPU-only environment: load every .pt file with torch.load(..., map_location=\"cpu\").\n"
        "- Skip unavailable sources gracefully without crashing.\n\n"
        f"Write UTF-8 JSON to exactly: {output_file}\n"
        f"Required candidate labels: {labels}\n"
        f"Keep the serialized JSON below approximately {int(max_evidence_chars)} characters.\n\n"
        "Output exactly this minimal structure:\n"
        "{\n"
        '  "signals": [\n'
        "    {\n"
        '      "signal_name": "stable descriptive identifier",\n'
        '      "description": "what factual relationship or quantity was measured",\n'
        '      "sources": ["exact available source filename"],\n'
        '      "candidate_observations": {\n'
        '        "A": {"value": null, "evidence": []}\n'
        "      }\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Every signal must apply the same computation to every candidate label. "
        "`value` may be a number, string, boolean, compact object/list, or null. "
        "`evidence` must be a list of short factual strings. "
        "Source names must exactly match names in the manifest. "
        f"ID-only case:\n{_dump(case_view)}\n\n"
        f"Source Capability Manifest:\n{_dump(source_manifest)}"
    )


def stage2_code_prompt(
    case_view,
    source_manifest,
    affordance_graph,
    output_file,
    max_evidence_chars,
    refinement_context,
):
    """Stage 2: 다중 홉 추론. Stage 1 신호를 앵커로 삼아 2개 이상의 릴레이션을 거친 심층 증거를 생성한다."""
    labels = ", ".join(candidate_labels(case_view))
    relation_map = render_affordance_relation_map(affordance_graph)

    return (
        "You are the Stage 2 Signal Python Code Generator in a training-free bundle-completion system. Generate only "
        "complete executable Python code without markdown fences or explanation. The script runs with the allowed "
        "workspace as its current directory.\n\n"
        f"{task_semantics(case_view['dataset'])}\n\n"
        "Investigate how items relate through bundles, users, categories, or "
        "other grounded multi-hop paths.\n\n"
        "Your role in Stage 2 is deep multi-hop refinement. You must produce only NEW or materially REVISED signals whose computation traverses at least TWO typed relation transitions"
        "Rules:\n"
        "- Every signal must include `relation_path`: a list of at least two non-empty typed relation transitions "
        "actually implemented by the generated code.\n"
        "- The runner preserves all valid Stage 1 signals and merges them. If you materially improve a Stage 1 "
        "signal, reuse its exact `signal_name`; otherwise use a new descriptive name.\n"
        "- Signals must be factual outputs of executed code. Do not choose, rank, or recommend.\n"
        "- CPU-only environment: load every .pt file with torch.load(..., map_location=\"cpu\").\n"
        "- Skip unavailable sources gracefully without crashing.\n\n"
        f"Write UTF-8 JSON to exactly: {output_file}\n"
        f"Required candidate labels: {labels}\n"
        f"Keep the serialized JSON below approximately {int(max_evidence_chars)} characters. "
        "Output exactly this minimal structure:\n"
        "{\n"
        '  "signals": [\n'
        "    {\n"
        '      "signal_name": "stable descriptive identifier",\n'
        '      "description": "what factual relationship or quantity was measured",\n'
        '      "sources": ["exact available source filename"],\n'
        '      "relation_path": ["typed transition 1", "typed transition 2"],\n'
        '      "candidate_observations": {\n'
        '        "A": {"value": null, "evidence": []}\n'
        "      }\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Every signal must apply the same computation to every candidate label. "
        "`value` may be a number, string, boolean, compact object/list, or null. "
        "`evidence` must be a list of short factual strings including compact executed path instances. "
        "Source names must exactly match names in the manifest. "
        f"ID-only case:\n{_dump(case_view)}\n\n"
        f"Source Capability Manifest:\n{_dump(source_manifest)}\n\n"
        f"Compact Evidence Relation Map:\n{relation_map}\n\n"
        f"Stage 1 / prior-round evidence (signal/source-first, with all candidates):\n"
        f"{_dump(refinement_context)}"
    )


def code_repair_prompt(
    case_view,
    source_manifest,
    previous_code,
    execution_and_validation,
    output_file,
    affordance_graph=None,
    require_multi_hop=False,
):
    labels = ", ".join(candidate_labels(case_view))
    relation_path_schema = ""
    relation_path_rule = ""
    graph_block = ""
    if require_multi_hop:
        relation_path_schema = (
            '      "relation_path": ["typed transition 1", "typed transition 2"],\n'
        )
        relation_path_rule = (
            " Every signal must retain a `relation_path` list containing at least two non-empty typed transitions that "
            "the repaired code actually executes. Do not repair a multi-hop signal into a raw similarity or direct count."
        )
        graph_block = (
            "\n\nCompact Evidence Relation Map:\n"
            + render_affordance_relation_map(affordance_graph)
        )
    return (
        "You are repairing Python signal-extraction code. Return only complete executable Python code without markdown "
        "fences or explanation. Fix execution, safety-compatible implementation, JSON serialization, output path, or evidence "
        "schema defects. Preserve the intended signal investigation and candidate-symmetric computation. Do not turn this "
        "repair into a new research plan, and do not predict or rank candidates. The execution environment may be CPU-only: "
        "every .pt file must be loaded with torch.load(..., map_location=\"cpu\").\n\n"
        f"The repaired script must write UTF-8 JSON to exactly: {output_file}\n"
        f"Required candidate labels in every signal: {labels}\n"
        "The JSON object may contain only the top-level `signals` field. Preserve or restore this exact minimal schema:\n"
        "{\n"
        '  "signals": [\n'
        "    {\n"
        '      "signal_name": "stable descriptive identifier",\n'
        '      "description": "what factual relationship or quantity was measured",\n'
        '      "sources": ["exact available source filename"],\n'
        f"{relation_path_schema}"
        '      "candidate_observations": {\n'
        '        "A": {"value": null, "evidence": []}\n'
        "      }\n"
        "    }\n"
        "  ]\n"
        "}\n"
        "Each signal object may contain only signal_name, description, sources, optional relation_path, and "
        "candidate_observations. Every candidate "
        "observation may contain only value and evidence. `description` and `sources` are mandatory and must not be removed. "
        f"{relation_path_rule} "
        "Do not add case_id, dataset, bundle_id, partial_item_ids, candidates, prediction, ranking, recommendation, winner, "
        "or final_score to the evidence JSON.\n\n"
        f"ID-only case:\n{_dump(case_view)}\n\n"
        f"Source Capability Manifest:\n{_dump(source_manifest)}\n\n"
        f"Execution and validation defects:\n{_dump(execution_and_validation)}"
        f"{graph_block}\n\n"
        f"Previous code:\n{previous_code}"
    )


def _decision_task_names(dataset):
    if "spotify" in str(dataset or "").lower():
        return "playlist continuation", "music playlist", "song"
    return "bundle construction", "fashion outfit", "fashion item"


def _compact_prompt_value(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def _candidate_evidence_lines(evidence, label):
    lines = []
    signals = evidence.get("signals", []) if isinstance(evidence, dict) else []
    for signal in signals:
        if not isinstance(signal, dict):
            continue
        observations = signal.get("candidate_observations", {})
        observation = observations.get(label) if isinstance(observations, dict) else None
        if not isinstance(observation, dict):
            continue
        signal_name = str(signal.get("signal_name", "unnamed_signal"))
        sources = signal.get("sources", [])
        source_text = ", ".join(str(source) for source in sources) if isinstance(sources, list) else str(sources)
        relation_path = signal.get("relation_path", [])
        path_text = (
            " -> ".join(str(transition) for transition in relation_path)
            if isinstance(relation_path, list) and relation_path
            else ""
        )
        value_text = _compact_prompt_value(observation.get("value"))
        facts = observation.get("evidence", [])
        fact_text = _compact_prompt_value(facts) if isinstance(facts, list) and facts else "[]"
        path_suffix = f"; path={path_text}" if path_text else ""
        lines.append(
            f"   - {signal_name} [sources: {source_text}]: value={value_text}; evidence={fact_text}{path_suffix}"
        )
    return lines


def decision_prompt(decision_case, evidence):
    task_name, bundle_name, item_name = _decision_task_names(decision_case.get("dataset"))
    input_str = "; ".join(
        f"{index + 1}. {item.get('text', '')}"
        for index, item in enumerate(decision_case.get("partial_items", []))
    )

    option_blocks = []
    for candidate in decision_case.get("candidates", []):
        label = str(candidate.get("label", ""))
        block = [f"{label}. {candidate.get('text', '')}"]
        evidence_lines = _candidate_evidence_lines(evidence, label)
        if evidence_lines:
            block.append("   Evidence:")
            block.extend(evidence_lines)
        option_blocks.append("\n".join(block))
    target_str = "\n".join(option_blocks)

    dataset_name = str(decision_case.get("dataset", "")).lower()
    pog_guidance = ""
    if dataset_name in ["pog", "pog_dense"]:
        pog_guidance = (
            "Note: For fashion outfits, similar items are rarely put together. "
            "Therefore, you must prioritize compatibility and complementarity over item resemblance.\n"
        )

    return (
        f"You are a helpful and honest assistant. The following are multiple choice questions about {task_name}. "
        "You should directly answer the question by choosing the letter of the correct option. Only provide the letter "
        "of your answer, without any explanation or mentioning the option content.\n"
        f"{pog_guidance}"
        f"Question: Given the partial {bundle_name}: {input_str}, which candidate {item_name} should be included into this "
        f"{bundle_name}?\n"
        f"Options:\n{target_str}\n"
        'Your answer should indicate your choice with a single letter (e.g., "A," "B," "C," etc.).\n'
        "Choice:"
    )

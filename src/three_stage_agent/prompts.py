import json
import os

from agents.common import build_agent_sample_view, candidate_labels, task_semantics


def dataset_contract(sample, workspace):
    dataset = str(sample.get("dataset", "")).lower()
    count_path = os.path.join(workspace["data_dir"], "count.json")
    count_text = "{}"
    if os.path.exists(count_path):
        with open(count_path, "r", encoding="utf-8") as handle:
            count_text = handle.read().strip()

    if dataset == "pog":
        item_fields = "id, cate, pic, title"
        category_field = "cate"
    elif dataset == "pog_dense":
        item_fields = "id, cate_id, pic_url, title"
        category_field = "cate_id"
    elif "spotify" in dataset:
        item_fields = (
            "pos, artist_name, track_uri, artist_uri, track_name, album_uri, "
            "duration_ms, album_name"
        )
        category_field = "artist_name or album_name when useful"
    else:
        item_fields = "inspect item_info[str(item_id)].keys()"
        category_field = "an available metadata field when useful"

    return (
        f"Observed count.json: {count_text}\n"
        "data/item_info.json is a JSON object keyed by string item_id. "
        f"Expected fields: {item_fields}. The main grouping field is {category_field}.\n"
        "data/bi_train.txt contains comma-space separated train bundle rows: "
        "bundle_id, item_id, item_id, ...\n"
        "data/ui_full.txt contains comma-space separated user/context rows: "
        "user_id, item_id, item_id, ...\n"
        "Optional .pt files listed in the allowed files are torch.Tensor item-feature matrices "
        "indexed by integer item_id when their first dimension matches the item count.\n"
    )


def generate_exploratory_retrieval_prompt(sample, workspace, evidence_output_file, conf):
    sample_view = build_agent_sample_view(sample)
    labels = candidate_labels(sample)
    max_stdout_chars = int(conf.get("agent_code_max_stdout_chars", 20000))
    return (
        "You are the exploratory retrieval researcher for a three-stage bundle completion system.\n"
        "Write executable Python code that searches the allowed local training data and builds a compact, "
        "human-readable evidence pack for a later evidence-synthesis LLM.\n\n"
        "Your role is retrieval, not prediction:\n"
        "- Do not choose, rank, recommend, or imply a preferred candidate.\n"
        "- Do not output a winner, answer, prediction, best label, or true label.\n"
        "- Do not primarily produce candidate-level numeric score tables.\n"
        "- A high similarity or co-occurrence value is not sufficient evidence by itself.\n"
        "- Use numeric or embedding operations internally when useful for discovering examples, but return "
        "representative observations that a later LLM can inspect and interpret.\n\n"
        "Research goal:\n"
        "Retrieve grounded historical observations that help a later reasoner understand the partial bundle, "
        "possible completion relationships, how candidates relate to those relationships, what evidence "
        "supports or challenges them, and how reliable or representative the observations are.\n\n"
        "Method selection:\n"
        "- Choose retrieval and analysis strategies based on this particular sample and the available files.\n"
        "- Possible strategies include, but are not limited to, historical bundle relationships, item or "
        "candidate neighborhoods, user-interaction neighborhoods, metadata relationships, category structures, "
        "embedding-discovered examples, cross-source joins, and contradictory-example retrieval.\n"
        "- These are examples, not required steps. Do not mechanically perform every strategy.\n"
        "- You may devise other useful train-safe strategies.\n\n"
        "Evidence quality requirements:\n"
        "- Relevant: each observation must have a clear relationship to the input or a candidate.\n"
        "- Interpretable: return item titles, metadata, and compact bundle/context compositions when possible.\n"
        "- Comparative: collect observations that can help distinguish candidate relationships.\n"
        "- Diverse: avoid many near-duplicate examples.\n"
        "- Balanced: include counterexamples or limiting observations when available.\n"
        "- Grounded: every observation must identify its source and retrieval basis.\n"
        "- Honest: explicitly report sparse, tied, indirect, missing, or inconclusive evidence.\n\n"
        "Dataset/task semantics:\n"
        f"{task_semantics(sample.get('dataset', ''))}\n\n"
        "Current sample JSON:\n"
        f"{json.dumps(sample_view, ensure_ascii=False, indent=2)}\n\n"
        "Allowed workspace files:\n"
        f"{json.dumps(workspace['files'], ensure_ascii=False, indent=2)}\n\n"
        "File format contract:\n"
        f"{dataset_contract(sample, workspace)}\n"
        "Filesystem and leakage restrictions:\n"
        "- Run from the workspace directory and read only listed files through relative paths under data/.\n"
        "- Write the final JSON only under output/.\n"
        "- Do not inspect or use files that are present in the workspace but absent from the allowed file list.\n"
        "- Do not read bi_full.txt, test or validation ground truth, result CSV files, predictions, hits, or true labels.\n"
        "- Do not access parent directories, absolute paths, home directories, network resources, or URLs.\n"
        "- Do not use os.walk, rglob, requests, urllib, or sockets.\n\n"
        "Output budget and shape:\n"
        "- Search broadly internally, but return only a compact and diverse evidence pack.\n"
        "- Prefer a few representative examples over long raw lists.\n"
        "- Keep each title, observation, rationale, provenance string, and limitation concise.\n"
        f"- Keep printed JSON under about {max_stdout_chars} characters.\n"
        f"- Include every candidate label exactly once in candidate_observations: {', '.join(labels)}.\n"
        "- Empty example lists with honest limitations are better than fabricated evidence.\n\n"
        "Write JSON using this schema:\n"
        "{\n"
        '  "research_focus": ["short question or relationship investigated"],\n'
        '  "global_observations": [\n'
        "    {\n"
        '      "observation": "factual observation from retrieved data",\n'
        '      "representative_examples": [\n'
        "        {\n"
        '          "context_id": 0,\n'
        '          "items": [{"item_id": 0, "title": "short title", "metadata": "short metadata"}],\n'
        '          "why_retrieved": "factual retrieval basis"\n'
        "        }\n"
        "      ],\n"
        '      "provenance": ["bi_train.txt"],\n'
        '      "limitations": ["short limitation"]\n'
        "    }\n"
        "  ],\n"
        '  "candidate_observations": {\n'
        '    "A": {\n'
        '      "candidate": {"item_id": 0, "title": "short title", "metadata": "short metadata"},\n'
        '      "related_examples": [\n'
        "        {\n"
        '          "relationship": "factual relationship, not a recommendation",\n'
        '          "items": [{"item_id": 0, "title": "short title", "metadata": "short metadata"}],\n'
        '          "provenance": ["item_info.json", "bi_train.txt"],\n'
        '          "why_retrieved": "factual retrieval basis"\n'
        "        }\n"
        "      ],\n"
        '      "counter_or_limiting_observations": ["factual caveat or contradictory observation"],\n'
        '      "limitations": ["missing, sparse, indirect, or inconclusive evidence"]\n'
        "    }\n"
        "  },\n"
        '  "source_summary": {\n'
        '    "files_used": ["item_info.json", "bi_train.txt"],\n'
        '    "retrieval_bases_used": ["short retrieval basis"],\n'
        '    "warnings": ["short warning"]\n'
        "  }\n"
        "}\n\n"
        f"Write the JSON to {evidence_output_file} and print the same object to stdout.\n"
        "Generate executable Python code only. Do not wrap it in markdown. Do not include comments, docstrings, "
        "placeholder ellipses, or any line starting with #. Do not include two consecutive periods."
    )


def generate_synthesis_prompt(sample, evidence_json, execution_summary):
    sample_view = build_agent_sample_view(sample)
    labels = candidate_labels(sample)
    return (
        "You are the evidence synthesizer in a three-stage bundle completion system.\n"
        "Read the actual retrieved examples and produce a grounded semantic synthesis for a later final predictor.\n"
        "Do not choose, rank, recommend, or imply a preferred candidate. Do not output a prediction.\n\n"
        "Synthesis goals:\n"
        "- Interpret what kind of bundle or relationship the partial input represents.\n"
        "- Explain how each candidate relates to the retrieved historical observations.\n"
        "- Separate complementary relationships from mere item similarity or category duplication.\n"
        "- Identify supporting observations, counter-evidence, conflicts, and retrieval limitations.\n"
        "- Assess reliability based on provenance, directness, diversity, and consistency of examples.\n"
        "- If retrieval failed or is inconclusive, say so rather than inventing patterns.\n\n"
        "Important reasoning rules:\n"
        "- Ground every synthesis claim in the retrieved evidence pack or the visible item text.\n"
        "- Do not treat similarity-discovered examples as direct historical proof.\n"
        "- Do not infer that a candidate is best merely because it has more retrieved examples.\n"
        "- Do not convert this task into a candidate ranking.\n\n"
        "Dataset/task semantics:\n"
        f"{task_semantics(sample.get('dataset', ''))}\n\n"
        "Current sample JSON:\n"
        f"{json.dumps(sample_view, ensure_ascii=False, indent=2)}\n\n"
        "Retrieval execution summary:\n"
        f"{json.dumps(execution_summary, ensure_ascii=False, indent=2)}\n\n"
        "Retrieved evidence pack:\n"
        f"{json.dumps(evidence_json, ensure_ascii=False, indent=2)}\n\n"
        f"Include every candidate label exactly once: {', '.join(labels)}.\n"
        "Return only valid JSON using this schema:\n"
        "{\n"
        '  "evidence_quality": "none|weak|medium|strong",\n'
        '  "bundle_interpretation": "grounded interpretation of the partial bundle",\n'
        '  "cross_candidate_patterns": ["grounded pattern or relationship"],\n'
        '  "candidate_synthesis": {\n'
        '    "A": {\n'
        '      "role_or_relationship": "how this candidate relates to the partial bundle",\n'
        '      "supporting_observations": ["grounded observation"],\n'
        '      "counter_evidence": ["grounded counter-observation or limitation"],\n'
        '      "evidence_reliability": "none|weak|medium|strong"\n'
        "    }\n"
        "  },\n"
        '  "conflicts": ["important conflict among sources or observations"],\n'
        '  "downweighted_evidence": ["evidence that should not be trusted strongly and why"],\n'
        '  "limitations": ["overall limitation"],\n'
        '  "sources_used": ["item_info.json", "bi_train.txt"]\n'
        "}\n"
    )


def generate_final_prediction_prompt(sample, synthesis_json):
    sample_view = build_agent_sample_view(sample)
    return (
        "You are the final predictor in a three-stage bundle completion system.\n"
        "Choose exactly one candidate using the original item text and the grounded evidence synthesis.\n"
        "The synthesis intentionally does not rank candidates and may be weak or inconclusive.\n\n"
        "Decision rules:\n"
        "- Judge whether a candidate complements the partial bundle, not merely whether it resembles an input item.\n"
        "- Use grounded, direct, diverse, and consistent observations more strongly than indirect similarity-only evidence.\n"
        "- Respect counter-evidence, conflicts, downweighted evidence, and stated limitations.\n"
        "- When evidence quality is weak or none, rely primarily on the original item text and use low confidence.\n"
        "- Do not invent evidence that is absent from the synthesis.\n\n"
        "Dataset/task semantics:\n"
        f"{task_semantics(sample.get('dataset', ''))}\n\n"
        "Current sample JSON:\n"
        f"{json.dumps(sample_view, ensure_ascii=False, indent=2)}\n\n"
        "Evidence synthesis JSON:\n"
        f"{json.dumps(synthesis_json, ensure_ascii=False, indent=2)}\n\n"
        "Return only valid JSON:\n"
        "{\n"
        '  "prediction": "A",\n'
        '  "reasoning": "concise final comparison grounded in text and synthesis",\n'
        '  "confidence": "low|medium|high",\n'
        '  "evidence_quality_used": "none|weak|medium|strong",\n'
        '  "main_observations_used": ["short grounded observation"],\n'
        '  "downweighted_or_ignored": ["weak or conflicting evidence not relied on"]\n'
        "}\n"
    )

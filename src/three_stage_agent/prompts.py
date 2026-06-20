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
    max_stdout_chars = int(conf.get("agent_code_max_stdout_chars", 20000))
    return (
        "You are the factual source-observation collector for a three-stage bundle completion system.\n"
        "Write executable Python code that searches the allowed local training data and builds a compact "
        "observation log for a later evidence critic and final judge.\n\n"
        "Your role is source observation, not interpretation or prediction:\n"
        "- Do not choose, rank, recommend, or imply a preferred candidate.\n"
        "- Do not output a winner, answer, prediction, best label, or true label.\n"
        "- Do not write support, counter, reliability, missing role, complement, redundant, best, or fit judgments.\n"
        "- Do not make semantic claims that are not directly visible in source fields. If text is garbled, "
        "state that the visible text is unclear rather than interpreting it.\n\n"
        "Observation goal:\n"
        "Use source files to produce factual observations that a later LLM can synthesize. Observations may "
        "include any source-grounded facts that help inspect the partial bundle and the candidates.\n\n"
        "Required data extraction:\n"
        "- Build the current input item list and the full candidate label->item_id list from Current sample JSON.\n"
        "- For item_info.json, look up every input item and every candidate item. Extract visible title/text "
        "fields and grouping metadata such as cate/cate_id/artist/album when present. Also record factual "
        "metadata relationships such as whether each candidate has the same grouping value as an input item.\n"
        "- For bi_train.txt, parse train bundle rows as comma-space separated values. For every input item and "
        "every candidate, count rows containing that item. For every candidate, also count rows containing both "
        "that candidate and at least one current input item, and include compact example bundle ids or rows when "
        "available. Output the candidate-only row count and the candidate-with-input row count as separate "
        "candidate-scoped observations. Do not call a candidate-only row count a co-occurrence count.\n"
        "- For ui_full.txt, parse user/context rows as comma-space separated values. For every input item and "
        "every candidate, count rows containing that item. For every candidate, also count rows containing both "
        "that candidate and at least one current input item, and include compact example user/context ids or "
        "rows when available. Output the candidate-only row count and the candidate-with-input row count as "
        "separate candidate-scoped observations. Do not call a candidate-only row count a co-occurrence count.\n"
        "- For content_feature.pt and description_feature.pt, load the tensor. If the first dimension matches "
        "the item count, index vectors by integer item_id for every input item and every candidate. Compute "
        "factual numeric relationships such as cosine similarity or distance between each candidate vector and "
        "the input vector or mean input vector. Emit candidate-scoped numeric observations for every candidate.\n"
        "- Loading a file, reporting existence, row count, column count, or tensor shape is only a diagnostic "
        "check; it does not count as meaningful source use unless followed by sample-specific extraction.\n"
        "- If any required extraction cannot be performed for a source, record the concrete reason in warnings "
        "or source_attempts.\n"
        "- Do not make the judgment yourself. Do not say which candidate is better. Only report factual "
        "information that could support later judgment.\n"
        "- Do not output your reasoning plan.\n\n"
        "Use every allowed source file:\n"
        "- Attempt every file listed in Allowed workspace files.\n"
        "- For each attempted file, add a source_attempt entry with used=true or used=false.\n"
        "- An attempt means actually opening or loading the listed file and trying to extract task-relevant facts, "
        "not merely naming it or checking that it exists.\n"
        "- Do not mark a source as skipped for brevity.\n"
        "- For each source except pure global metadata such as count.json, try both: "
        "one partial bundle/input scoped observation and candidate-scoped observations for every candidate "
        "label in the current sample.\n"
        "- Do not stop after candidate A. Cover all candidate labels exactly as listed in the current sample. "
        "If there are 10 candidates, attempt candidate-scoped observations for all 10 candidates.\n"
        "- If candidate-scoped observation is not meaningful for a source, or cannot be produced after loading "
        "the source, add a factual warning or source_attempt result explaining why.\n"
        "- File access, tensor shape, or row count alone is only a source_status observation; it does not satisfy "
        "the input/candidate scoped observation requirement for that source.\n"
        "- If you mark used=false, the result must describe the concrete failure or unusable condition observed "
        "after attempting to access that source.\n"
        "- If a source cannot be loaded, is unavailable, has an unusable shape, or produces no useful observation, "
        "record that fact in source_attempts and warnings.\n"
        "- Do not stop after one source or one shallow diagnostic check.\n"
        "- Prefer factual observations over summaries. Stage 2 will decide what the facts mean.\n\n"
        "Observation perspectives:\n"
        "- Include factual observations from the partial bundle/input perspective.\n"
        "- Include factual observations from the candidate-specific perspective for every candidate label.\n"
        "- If a candidate has no useful source-grounded fact for a source, record that candidate-specific absence "
        "as a factual observation, warning, or source_attempt result.\n"
        "- If a perspective has little or no useful source-grounded information, record that fact in warnings.\n\n"
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
        "- Keep the output compact; include representative examples and counts instead of long raw lists.\n"
        "- Include observations from both the partial bundle/input perspective and the candidate-specific perspective.\n"
        "- Candidate-specific coverage must include every candidate label in the current sample, not only one "
        "representative candidate.\n"
        "- Include source_attempts for every allowed source file.\n"
        "- Keep each title, observation, basis string, and warning concise.\n"
        f"- Keep printed JSON under about {max_stdout_chars} characters.\n"
        "- For source values, use actual file names from the allowed file list.\n"
        "- For scope values, use labels such as input, candidate:A, candidate:B, input_context, "
        "candidate_context, or source_status.\n\n"
        "Write JSON using this schema:\n"
        "{\n"
        '  "source_observations": [\n'
        "    {\n"
        '      "source": "file name used",\n'
        '      "view": "view_name",\n'
        '      "scope": "input|candidate:A|input_context|candidate_context|source_status",\n'
        '      "observation": "factual observation only",\n'
        '      "related_ids": [0],\n'
        '      "basis": "exact check, field, row type, count, or computation used",\n'
        '      "examples": [\n'
        "        {\n"
        '          "context_id": 0,\n'
        '          "items": [{"item_id": 0, "title": "short title", "metadata": "short metadata"}],\n'
        '          "why_retrieved": "factual retrieval basis"\n'
        "        }\n"
        "      ]\n"
        "    }\n"
        "  ],\n"
        '  "source_attempts": [\n'
        "    {\n"
        '      "source": "file name from allowed list",\n'
        '      "used": true,\n'
        '      "checks": ["short factual check performed"],\n'
        '      "result": "short factual result or reason it produced no useful observation"\n'
        "    }\n"
        "  ],\n"
        '  "warnings": ["short warning about sparse, missing, tied, skipped, or unclear evidence"]\n'
        "}\n\n"
        f"Write the JSON to {evidence_output_file} and print the same object to stdout.\n"
        "Generate executable Python code only. Do not wrap it in markdown. Do not include comments, docstrings, "
        "placeholder ellipses, or any line starting with #. Do not include two consecutive periods."
    )


def generate_exploratory_retrieval_repair_prompt(
    sample,
    workspace,
    evidence_output_file,
    previous_code,
    execution_summary,
    conf,
):
    sample_view = build_agent_sample_view(sample)
    labels = candidate_labels(sample)
    max_stdout_chars = int(conf.get("agent_code_max_stdout_chars", 20000))
    return (
        "You are repairing Python evidence-retrieval code for the first stage of a three-stage bundle "
        "completion system.\n"
        "The previous code failed, was blocked, timed out, or did not produce parseable JSON. Produce a "
        "simpler corrected script that follows the same factual source-observation schema.\n\n"
        "Do not predict, rank, recommend, imply a preferred candidate, or output true labels. Do not write "
        "support/counter/reliability interpretations. Return factual observations only.\n"
        "Prefer robust, simple retrieval over ambitious code. Attempt every allowed source file and record each "
        "attempt in source_attempts. Build the current input item list and the full candidate label->item_id "
        "list from Current sample JSON. For item_info.json, look up every input and every candidate and extract "
        "visible title/text fields, grouping metadata, and factual same-group relationships to input items. For "
        "bi_train.txt, parse comma-space separated train bundle rows; count rows containing every input item "
        "and every candidate, and for each candidate count rows containing both that candidate and at least one "
        "input item. Output candidate-only row counts and candidate-with-input row counts as separate "
        "observations, and do not label candidate-only counts as co-occurrence counts. For ui_full.txt, parse "
        "comma-space separated user/context rows and do the same per-item and candidate-with-input counts as "
        "separate observations. For content_feature.pt and description_feature.pt, load the tensor; "
        "if the first dimension matches item count, index vectors by integer item_id and compute factual numeric "
        "relationships such as cosine similarity or distance between each candidate vector and the input vector "
        "or mean input vector. If there are 10 candidates, attempt candidate-scoped observations for all 10 "
        "candidates. Loading a file, reporting its existence, row count, column count, or tensor shape is only "
        "a diagnostic check and does not count as meaningful source use unless followed by sample-specific "
        "extraction. Do not make the judgment yourself, and do not output your reasoning plan. "
        "Do not mark a listed existing source as skipped for brevity.\n\n"
        "Current sample JSON:\n"
        f"{json.dumps(sample_view, ensure_ascii=False, indent=2)}\n\n"
        "Allowed workspace files:\n"
        f"{json.dumps(workspace['files'], ensure_ascii=False, indent=2)}\n\n"
        "File format contract:\n"
        f"{dataset_contract(sample, workspace)}\n"
        "Previous execution summary:\n"
        f"{json.dumps(execution_summary, ensure_ascii=False, indent=2)}\n\n"
        "Previous code:\n"
        f"{previous_code[:6000]}\n\n"
        "Repair requirements:\n"
        "- Read only listed relative files under data/.\n"
        "- Write valid JSON only under output/ and print the same object to stdout.\n"
        "- Do not access parent directories, absolute paths, network resources, result files, or ground truth.\n"
        "- Include candidate-specific factual observations for every candidate label when available; otherwise "
        "record the candidate-specific absence. Candidate labels: "
        f"{', '.join(labels)}.\n"
        "- Use the schema from the original first-stage prompt: source_observations, source_attempts, and warnings.\n"
        f"- Keep printed JSON under about {max_stdout_chars} characters.\n"
        f"- Write the JSON to {evidence_output_file}.\n\n"
        "Generate executable Python code only. Do not wrap it in markdown. Do not include comments, docstrings, "
        "placeholder ellipses, or any line starting with #. Do not include two consecutive periods."
    )


def generate_deep_observation_prompt(
    sample,
    workspace,
    surface_evidence_json,
    surface_execution_summary,
    evidence_output_file,
    conf,
):
    sample_view = build_agent_sample_view(sample)
    max_stdout_chars = int(conf.get("agent_code_max_stdout_chars", 20000))
    return (
        "You are the deep source-observation agent for a bundle completion system.\n"
        "The surface observation agent has already extracted direct facts such as metadata, direct row counts, "
        "and direct numeric comparisons. Your job is to write executable Python code that designs and executes "
        "additional source-grounded investigations for the current sample.\n\n"
        "Your role is deeper observation, not interpretation or prediction:\n"
        "- Do not choose, rank, recommend, or imply a preferred candidate.\n"
        "- Do not output a winner, answer, prediction, best label, or true label.\n"
        "- Do not repeat surface observations unless they are used as starting points for deeper extraction.\n"
        "- Do not make unsupported semantic claims. Keep observations factual and source-grounded.\n\n"
        "Deep investigation requirements:\n"
        "- Design and execute additional investigations that may reveal non-obvious factual context for the "
        "current completion task.\n"
        "- For each investigation, formulate a factual question, briefly state why the question is relevant to "
        "the completion task without favoring a candidate, and extract concrete facts from allowed sources.\n"
        "- Tie observations to current input items, candidate items, or relationships visible in the sources.\n"
        "- Go beyond direct lookup, file diagnostics, tensor shape, and one-hop item counts.\n"
        "- Use the source structure to discover additional context, comparisons, neighborhoods, repeated "
        "patterns, or indirect relationships that are not already present in the surface observations.\n"
        "- Record compact examples, counts, retrieved ids, or computed values when available.\n"
        "- If an investigation fails or produces no useful facts, record what was tried and what was observed.\n"
        "- Include every candidate label in at least one deep investigation when possible. If a candidate cannot "
        "be covered by deep observations, record that limitation.\n"
        "- Do not output your private planning. Output only the executed investigations and factual results.\n\n"
        "Dataset/task semantics:\n"
        f"{task_semantics(sample.get('dataset', ''))}\n\n"
        "Current sample JSON:\n"
        f"{json.dumps(sample_view, ensure_ascii=False, indent=2)}\n\n"
        "Allowed workspace files:\n"
        f"{json.dumps(workspace['files'], ensure_ascii=False, indent=2)}\n\n"
        "File format contract:\n"
        f"{dataset_contract(sample, workspace)}\n"
        "Surface retrieval execution summary:\n"
        f"{json.dumps(surface_execution_summary, ensure_ascii=False, indent=2)}\n\n"
        "Surface observation JSON:\n"
        f"{json.dumps(surface_evidence_json, ensure_ascii=False, indent=2)}\n\n"
        "Filesystem and leakage restrictions:\n"
        "- Run from the workspace directory and read only listed files through relative paths under data/.\n"
        "- Write the final JSON only under output/.\n"
        "- Do not inspect or use files that are present in the workspace but absent from the allowed file list.\n"
        "- Do not read bi_full.txt, test or validation ground truth, result CSV files, predictions, hits, or true labels.\n"
        "- Do not access parent directories, absolute paths, home directories, network resources, or URLs.\n"
        "- Do not use os.walk, rglob, requests, urllib, or sockets.\n\n"
        "Output budget and shape:\n"
        "- Keep the output compact; include representative examples and counts instead of long raw lists.\n"
        "- Keep each question, relevance note, observation, basis string, and limitation concise.\n"
        f"- Keep printed JSON under about {max_stdout_chars} characters.\n"
        "- For source values, use actual file names from the allowed file list.\n\n"
        "Write JSON using this schema:\n"
        "{\n"
        '  "deep_investigations": [\n'
        "    {\n"
        '      "question": "factual question investigated",\n'
        '      "why_relevant": "task relevance of this investigation type, without favoring a candidate",\n'
        '      "sources_used": ["file name from allowed list"],\n'
        '      "method_summary": "short factual description of what was computed or retrieved",\n'
        '      "observations": [\n'
        "        {\n"
        '          "source": "file name used",\n'
        '          "scope": "input|candidate:A|cross_candidate|context",\n'
        '          "observation": "factual observation",\n'
        '          "related_ids": [0],\n'
        '          "basis": "lookup/count/computation/retrieval basis",\n'
        '          "examples": []\n'
        "        }\n"
        "      ],\n"
        '      "limitations": ["what this investigation could not establish"]\n'
        "    }\n"
        "  ],\n"
        '  "warnings": ["short warning about sparse, failed, or inconclusive deep investigation"]\n'
        "}\n\n"
        f"Write the JSON to {evidence_output_file} and print the same object to stdout.\n"
        "Generate executable Python code only. Do not wrap it in markdown. Do not include comments, docstrings, "
        "placeholder ellipses, or any line starting with #. Do not include two consecutive periods."
    )


def generate_deep_observation_repair_prompt(
    sample,
    workspace,
    surface_evidence_json,
    surface_execution_summary,
    evidence_output_file,
    previous_code,
    execution_summary,
    conf,
):
    sample_view = build_agent_sample_view(sample)
    max_stdout_chars = int(conf.get("agent_code_max_stdout_chars", 20000))
    return (
        "You are repairing Python code for the deep source-observation stage of a bundle completion system.\n"
        "The previous code failed, was blocked, timed out, or did not produce parseable JSON. Produce a simpler "
        "corrected script that follows the same deep investigation schema.\n\n"
        "Do not predict, rank, recommend, imply a preferred candidate, or output true labels. Do not merely "
        "repeat surface observations. Design and execute additional source-grounded investigations that go "
        "beyond direct lookup, file diagnostics, tensor shape, and one-hop item counts. Each investigation must "
        "include a factual question, a brief non-candidate-favoring relevance note, source-grounded observations, "
        "and limitations. Include every candidate label in at least one deep investigation when possible.\n\n"
        "Current sample JSON:\n"
        f"{json.dumps(sample_view, ensure_ascii=False, indent=2)}\n\n"
        "Allowed workspace files:\n"
        f"{json.dumps(workspace['files'], ensure_ascii=False, indent=2)}\n\n"
        "File format contract:\n"
        f"{dataset_contract(sample, workspace)}\n"
        "Surface retrieval execution summary:\n"
        f"{json.dumps(surface_execution_summary, ensure_ascii=False, indent=2)}\n\n"
        "Surface observation JSON:\n"
        f"{json.dumps(surface_evidence_json, ensure_ascii=False, indent=2)}\n\n"
        "Previous execution summary:\n"
        f"{json.dumps(execution_summary, ensure_ascii=False, indent=2)}\n\n"
        "Previous code:\n"
        f"{previous_code[:6000]}\n\n"
        "Repair requirements:\n"
        "- Read only listed relative files under data/.\n"
        "- Write valid JSON only under output/ and print the same object to stdout.\n"
        "- Do not access parent directories, absolute paths, network resources, result files, or ground truth.\n"
        "- Use the schema from the original deep-stage prompt: deep_investigations and warnings.\n"
        f"- Keep printed JSON under about {max_stdout_chars} characters.\n"
        f"- Write the JSON to {evidence_output_file}.\n\n"
        "Generate executable Python code only. Do not wrap it in markdown. Do not include comments, docstrings, "
        "placeholder ellipses, or any line starting with #. Do not include two consecutive periods."
    )


def generate_synthesis_prompt(sample, evidence_json, execution_summary):
    sample_view = build_agent_sample_view(sample)
    labels = candidate_labels(sample)
    return (
        "You are the evidence critic and bundle reasoner in a three-stage bundle completion system.\n"
        "Read the Stage 1 evidence pack and produce a grounded semantic synthesis for a later final predictor. "
        "Stage 1 contains a surface observation stage and a deep observation stage.\n"
        "Do not choose, rank, recommend, or imply a preferred candidate. Do not output a prediction.\n\n"
        "Synthesis goals:\n"
        "- Convert factual surface observations and deep investigations into an evidence summary without inventing facts.\n"
        "- Interpret what kind of bundle or relationship the partial input represents.\n"
        "- State a cautious missing-role hypothesis when the evidence supports one.\n"
        "- Explain how each candidate relates to the source observations and deep investigations across metadata, "
        "bundle history, user context, and numeric/indirect views.\n"
        "- Separate complementary relationships from mere item similarity or category duplication.\n"
        "- Identify support, counter-evidence, conflicts, and retrieval limitations from the observation log.\n"
        "- Assess reliability based on source_attempts, source type, directness, diversity, and consistency.\n"
        "- If Stage 1A or Stage 1B observations are sparse or inconclusive, say so rather than inventing patterns.\n\n"
        "Important reasoning rules:\n"
        "- Ground every synthesis claim in Stage 1 surface source_observations, Stage 1 deep investigations, "
        "source_attempts, or the visible item text.\n"
        "- Treat bundle_history observations as more direct than user_context or embedding_neighbor observations when they are "
        "specific, diverse, and consistent.\n"
        "- Treat user_context, numeric similarity, and deep indirect observations as indirect unless supported by bundle or metadata observations.\n"
        "- Do not treat similarity-discovered examples as direct historical proof.\n"
        "- Do not infer that a candidate is best merely because it has more observations.\n"
        "- Do not convert this task into a candidate ranking.\n\n"
        "Dataset/task semantics:\n"
        f"{task_semantics(sample.get('dataset', ''))}\n\n"
        "Current sample JSON:\n"
        f"{json.dumps(sample_view, ensure_ascii=False, indent=2)}\n\n"
        "Stage 1 execution summary:\n"
        f"{json.dumps(execution_summary, ensure_ascii=False, indent=2)}\n\n"
        "Stage 1 evidence pack:\n"
        f"{json.dumps(evidence_json, ensure_ascii=False, indent=2)}\n\n"
        f"Include every candidate label exactly once: {', '.join(labels)}.\n"
        "Return only valid JSON using this schema:\n"
        "{\n"
        '  "evidence_quality": "none|weak|medium|strong",\n'
        '  "bundle_interpretation": "grounded interpretation of the partial bundle",\n'
        '  "missing_role_hypothesis": "cautious missing role or relationship, or unknown",\n'
        '  "cross_candidate_patterns": ["grounded pattern or relationship"],\n'
        '  "view_reliability": {\n'
        '    "metadata": "none|weak|medium|strong",\n'
        '    "bundle_history": "none|weak|medium|strong",\n'
        '    "user_context": "none|weak|medium|strong",\n'
        '    "embedding_neighbor": "none|weak|medium|strong",\n'
        '    "source_status": "none|weak|medium|strong"\n'
        "  },\n"
        '  "candidate_synthesis": {\n'
        '    "A": {\n'
        '      "role_or_relationship": "how this candidate relates to the partial bundle",\n'
        '      "supporting_observations": ["grounded observation with source/view when possible"],\n'
        '      "counter_evidence": ["grounded counter-observation or limitation"],\n'
        '      "direct_evidence": ["direct bundle-history or metadata evidence"],\n'
        '      "indirect_evidence": ["user/context, embedding, or weak similarity evidence"],\n'
        '      "key_limitations": ["candidate-specific limitation"],\n'
        '      "evidence_reliability": "none|weak|medium|strong"\n'
        "    }\n"
        "  },\n"
        '  "conflicts": ["important conflict among sources or observations"],\n'
        '  "downweighted_evidence": ["evidence that should not be trusted strongly and why"],\n'
        '  "limitations": ["overall limitation"],\n'
        '  "sources_used": ["file name used"]\n'
        "}\n"
    )


def generate_final_prediction_prompt(sample, synthesis_json):
    sample_view = build_agent_sample_view(sample)
    return (
        "You are the final predictor in a three-stage bundle completion system.\n"
        "Choose exactly one candidate using the original item text, the missing-role hypothesis, and the "
        "grounded evidence synthesis.\n"
        "The synthesis intentionally does not rank candidates and may be weak or inconclusive.\n\n"
        "Decision rules:\n"
        "- Judge whether a candidate complements the partial bundle, not merely whether it resembles an input item.\n"
        "- Use grounded, direct, diverse, and consistent bundle or metadata evidence more strongly than indirect "
        "user/context or embedding-neighborhood evidence.\n"
        "- Prefer candidates that plausibly fill the missing role over candidates that duplicate an existing role.\n"
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

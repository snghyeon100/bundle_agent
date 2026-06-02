import json
import os

from agents.common import build_agent_sample_view, candidate_labels, parse_candidate_options, task_semantics


def dataset_format_contract(dataset_name, workspace):
    dataset_key = str(dataset_name or "").lower()
    data_dir = workspace["data_dir"]
    count_path = os.path.join(data_dir, "count.json")
    count_text = "{}"
    if os.path.exists(count_path):
        with open(count_path, "r", encoding="utf-8") as f:
            count_text = f.read().strip()

    shared = (
        "Available raw files are under data/ only. The code must not read any other dataset folder.\n"
        f"Observed count.json content for this workspace: {count_text}\n\n"
        "data/bi_train.txt:\n"
        "- Train bundle-item rows.\n"
        "- Each row is comma-space separated, exactly like: bundle_id, item_id1, item_id2, item_id3\n"
        "- The first value is a train bundle/context id. Remaining values are integer item ids in that train bundle.\n"
        "- Useful retrieval ideas: direct candidate/input co-affiliation, item->bundle inverted index, neighboring items from shared train bundles, and category/artist aggregation over those neighbors.\n\n"
        "data/ui_full.txt:\n"
        "- User-item interaction rows.\n"
        "- Each row is comma-space separated, exactly like: user_id, item_id1, item_id2, item_id3\n"
        "- The first value is a user/context id. Remaining values are integer item ids associated with that user.\n"
        "- Useful retrieval ideas: item->user inverted index, user overlap between candidate and input items, neighboring items from shared users, and metadata aggregation over those neighbors.\n\n"
        "data/content_feature.pt and data/description_feature.pt:\n"
        "- Optional PyTorch tensor feature caches.\n"
        "- In the inspected local datasets these are torch.Tensor matrices with first dimension equal to item count and feature dimension 768.\n"
        "- If a tensor loads and candidate/input ids are valid integer indexes, compute cosine similarity between each candidate vector and the mean input vector.\n"
        "- If torch or a tensor file is unavailable, record null values and a warning instead of failing.\n"
    )

    if dataset_key == "pog":
        item_info = (
            "data/item_info.json for pog:\n"
            "- JSON object keyed by string item_id, with 48676 inspected entries in the local POG data.\n"
            "- Every inspected item has these exact keys: id, cate, pic, title.\n"
            "- id: original hashed item id string.\n"
            "- cate: hashed fashion category id string. Use it for category match/diversity signals.\n"
            "- pic: product image URL string. It is metadata only; do not fetch URLs.\n"
            "- title: product title string, often Chinese fashion text and sometimes ending with a newline. Strip whitespace before comparing text.\n"
        )
    elif dataset_key == "pog_dense":
        item_info = (
            "data/item_info.json for pog_dense:\n"
            "- JSON object keyed by string item_id, with 31217 inspected entries in the local POG-dense data.\n"
            "- Every inspected item has these exact keys: id, cate_id, pic_url, title.\n"
            "- id: original hashed item id string.\n"
            "- cate_id: hashed fashion category id string. Use it for category match/diversity signals.\n"
            "- pic_url: product image URL string. It is metadata only; do not fetch URLs.\n"
            "- title: product title string, often Chinese fashion/product text. Strip whitespace before comparing text.\n"
        )
    elif "spotify" in dataset_key:
        item_info = (
            f"data/item_info.json for {dataset_name}:\n"
            "- JSON object keyed by string item_id.\n"
            "- Inspected local Spotify entries use these exact keys: pos, artist_name, track_uri, artist_uri, track_name, album_uri, duration_ms, album_name.\n"
            "- pos: integer track position from the source playlist data. Treat it as source metadata, not as candidate rank.\n"
            "- artist_name / artist_uri: artist identity. Useful for artist continuity and repeated-artist signals.\n"
            "- track_name / track_uri: song identity. Useful for text comparison and exact identity checks.\n"
            "- album_name / album_uri: album identity. Useful for album continuity and related-track signals.\n"
            "- duration_ms: integer duration in milliseconds. Useful for coarse listening-flow compatibility.\n"
        )
    else:
        item_info = (
            f"data/item_info.json for {dataset_name}:\n"
            "- JSON object keyed by string item_id.\n"
            "- Inspect keys dynamically with item_info[str(item_id)].keys() and record available fields in evidence.\n"
        )

    return item_info + "\n" + shared


def generate_code_retrieval_prompt(sample, workspace, evidence_output_file, conf):
    sample_view = build_agent_sample_view(sample)
    labels = candidate_labels(sample)
    data_dir = workspace["data_dir"]
    workspace_files = workspace["files"]
    return (
        "You are the code retrieval agent for a two-stage bundle completion system.\n\n"
        "Task:\n"
        f"{task_semantics(sample.get('dataset', ''))}\n\n"
        "Your role:\n"
        "- Do not choose the answer.\n"
        "- Write Python code that retrieves useful candidate-level evidence from the allowed local raw dataset files.\n"
        "- The later predictor will receive each candidate text plus only the retrieved values from your code.\n"
        "- Your code should measure signals that may help compare candidates, but it must not output a winner, rank, answer, prediction, best label, or true label.\n\n"
        "Current sample JSON:\n"
        f"{json.dumps(sample_view, ensure_ascii=False, indent=2)}\n\n"
        "Allowed dataset folder:\n"
        f"- The only dataset folder you may inspect is this exact folder: {data_dir}\n"
        "- The code will run with the workspace directory as cwd.\n"
        "- Read files only through relative paths under data/, such as data/item_info.json.\n"
        "- Write the final JSON only under output/.\n"
        f"- Available copied files: {json.dumps(workspace_files, ensure_ascii=False)}\n\n"
        "Dataset-specific file format details:\n"
        f"{dataset_format_contract(sample.get('dataset', ''), workspace)}\n\n"
        "Leakage and filesystem restrictions:\n"
        "- Do not read bi_full.txt, bi_test_gt.txt, validation/test ground-truth files, result CSV files, predictions, hits, or true labels.\n"
        "- Do not access parent directories, absolute paths, home directories, network resources, or URLs.\n"
        "- Do not use os.walk, rglob, requests, urllib, or sockets.\n\n"
        "Code style restrictions:\n"
        "- Generate executable Python code only.\n"
        "- Do not write comments, docstrings, markdown, explanations, or placeholder ellipses.\n"
        "- Do not include any line starting with #.\n"
        "- Do not include two consecutive periods anywhere in the code.\n\n"
        "Evidence expectations:\n"
        "- Include every candidate label exactly once: " + ", ".join(labels) + ".\n"
        "- Prefer compact numeric/text values per candidate: metadata fit, BI train co-affiliation/neighborhood support, UI overlap/neighborhood support, and optional embedding similarity.\n"
        "- For POG datasets, fashion category/title compatibility and train bundle/user neighborhood evidence are useful.\n"
        "- For Spotify datasets, artist/album/track continuity, playlist co-occurrence, user/listener neighborhood evidence, duration compatibility, and optional embedding similarity are useful.\n"
        "- If a source cannot be loaded or a signal is all-zero/tied/unavailable, record that in warnings and put null or zero values honestly.\n"
        "- Keep output compact. Do not dump raw rows, long lists, or large tensors.\n\n"
        f"Output requirements:\n"
        f"- Write JSON to {evidence_output_file}.\n"
        "- Print the same JSON object to stdout.\n"
        "- Use this schema exactly enough that candidate_evidence can be read by the predictor:\n"
        "{\n"
        '  "summary": "short global retrieval summary",\n'
        '  "raw_files_used": ["item_info.json", "bi_train.txt"],\n'
        '  "candidate_evidence": {\n'
        '    "A": {\n'
        '      "retrieved_values": {\n'
        '        "metadata_fit": "short text or numeric value",\n'
        '        "bi_signal": 0,\n'
        '        "ui_signal": 0,\n'
        '        "embedding_similarity": null\n'
        "      },\n"
        '      "short_evidence": "one concise sentence with useful values and caveats"\n'
        "    }\n"
        "  },\n"
        '  "warnings": ["short warning text"]\n'
        "}\n\n"
        "Return only executable Python code. Do not wrap it in markdown. Do not include comments."
    )


def compact_candidate_evidence(evidence_json, label):
    if not isinstance(evidence_json, dict):
        return "No retrieved evidence was produced."
    candidate_evidence = evidence_json.get("candidate_evidence", {})
    evidence = candidate_evidence.get(label, {})
    if not isinstance(evidence, dict):
        return "No candidate-specific retrieved evidence was produced."
    short = str(evidence.get("short_evidence", "")).strip()
    values = evidence.get("retrieved_values", {})
    pieces = []
    if short:
        pieces.append(short)
    if isinstance(values, dict) and values:
        pieces.append("Values: " + json.dumps(values, ensure_ascii=False, sort_keys=True))
    return " ".join(pieces)[:1500] if pieces else "No candidate-specific retrieved evidence was produced."


def generate_prediction_prompt(sample, evidence_json):
    options = parse_candidate_options(sample["target_str"])
    option_lines = []
    for label, candidate_text in options:
        option_lines.append(
            f"{label}. {candidate_text}\n"
            f"Retrieved evidence: {compact_candidate_evidence(evidence_json, label)}"
        )
    warnings = []
    summary = ""
    raw_files_used = []
    if isinstance(evidence_json, dict):
        summary = str(evidence_json.get("summary", "")).strip()
        warnings = evidence_json.get("warnings", []) or []
        raw_files_used = evidence_json.get("raw_files_used", []) or []

    if "spotify" in str(sample.get("dataset", "")).lower():
        task_name = "playlist continuation"
        bundle_name = "music playlist"
        item_name = "song"
    else:
        task_name = "fashion outfit bundle completion"
        bundle_name = "fashion outfit"
        item_name = "item"

    return (
        f"You are a helpful and honest assistant. The following is a multiple choice question about {task_name}.\n"
        f"Choose the candidate {item_name} that should be included in the partial {bundle_name}.\n\n"
        f"Question: Given the partial {bundle_name}: {sample['input_str']}, which candidate {item_name} should be included?\n\n"
        f"Retrieval summary: {summary or 'No global retrieval summary.'}\n"
        f"Raw files used by retrieval: {json.dumps(raw_files_used, ensure_ascii=False)}\n"
        f"Retrieval warnings: {json.dumps(warnings, ensure_ascii=False)}\n\n"
        "Options:\n"
        f"{chr(10).join(option_lines)}\n\n"
        "Decision rules:\n"
        "- Treat candidate text as the primary signal.\n"
        "- Use retrieved evidence only when it is relevant, non-failed, and discriminative.\n"
        "- Downweight retrieved values that are missing, tied, all-zero, contradicted, or listed in warnings.\n"
        "- Do not assume the code retrieval stage chose a winner; it was not allowed to choose.\n\n"
        "Return only valid JSON:\n"
        "{\n"
        '  "prediction": "A",\n'
        '  "reasoning": "concise comparison using candidate text and retrieved values",\n'
        '  "confidence": "low|medium|high"\n'
        "}\n"
    )

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import tempfile
import time

import pandas as pd
import yaml
from dotenv import load_dotenv
from google import genai

from dataset import BundleZeroShotDataset, set_seed


class QuotaExceededError(RuntimeError):
    pass


env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(dotenv_path=env_path, encoding="utf-8-sig")


def parse_model_response(raw_text):
    if not raw_text:
        return "ERR_EM"
    clean_text = re.sub(r"(?i)\b(choice|option|answer)\b[\s]*[:=]*[\s]*", "", raw_text.strip())
    match = re.search(r"([A-Z])", clean_text.upper())
    return match.group(1) if match else clean_text[:1].upper()


def console_safe_text(text):
    encoding = sys.stdout.encoding or "utf-8"
    return str(text).encode(encoding, errors="backslashreplace").decode(encoding)


def use_candidate_reasoning(conf):
    return bool(conf.get("use_candidate_reasoning", False))


def use_three_stage_agent(conf):
    return bool(conf.get("use_three_stage_agent", False))


def method_name(conf):
    if use_three_stage_agent(conf):
        return "three_stage_agent"
    if use_candidate_reasoning(conf):
        return "candidate_reasoning"
    return "baseline"


def resolve_api_key(conf, config_key, fallback_envs):
    env_name = str(conf.get(config_key, "")).strip()
    if env_name:
        api_key = os.getenv(env_name, "").strip()
        if not api_key:
            raise ValueError(f"API key environment variable is not set: {env_name}")
        return api_key, env_name

    for fallback_env in fallback_envs:
        api_key = os.getenv(fallback_env, "").strip()
        if api_key:
            return api_key, fallback_env
    raise ValueError(f"None of these API key environment variables are set: {', '.join(fallback_envs)}")


def parse_candidate_options(target_str):
    pattern = r"(?:^|;\s*)([A-Z])\.\s*(.*?)(?=\s*;\s*[A-Z]\.\s*|$)"
    matches = re.findall(pattern, target_str, flags=re.DOTALL)
    return [(letter, " ".join(text.split())) for letter, text in matches]


def build_agent_sample_view(sample):
    candidate_options = parse_candidate_options(sample["target_str"])
    candidates = []
    for idx, (letter, text) in enumerate(candidate_options):
        item_id = sample["candidate_indices"][idx] if idx < len(sample["candidate_indices"]) else None
        candidates.append({"label": letter, "item_id": item_id, "text": text})
    return {
        "bundle_id": sample["bundle_id"],
        "input_item_ids": sample["input_indices"],
        "input_text": sample["input_str"],
        "candidates": candidates,
    }


def list_agent_available_files(conf):
    data_root = os.path.abspath(os.path.join(conf["data_path"], conf["dataset"]))
    preferred = conf.get("agent_allowed_files") or [
        "count.json",
        "item_info.json",
        "bi_train.txt",
        "ui_full.txt",
        "content_feature.pt",
        "description_feature.pt",
    ]
    if bool(conf.get("agent_allow_interaction_embeddings", False)):
        preferred = list(preferred) + ["item_cf_feature.pt"]

    files = []
    for filename in preferred:
        path = os.path.join(data_root, filename)
        if os.path.exists(path):
            files.append({"name": filename, "path": path})

    extra_paths = conf.get("agent_extra_data_paths", []) or []
    for extra_path in extra_paths:
        abs_path = os.path.abspath(extra_path)
        if os.path.exists(abs_path):
            files.append({"name": os.path.basename(abs_path), "path": abs_path})
    return data_root, files


def parse_json_from_text(text):
    text = str(text or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        try:
            return json.loads(fenced.group(1).strip())
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None


def extract_python_code(text):
    text = str(text or "")
    fenced = re.search(r"```(?:python|py)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    return text.strip()


def compact_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def candidate_labels(sample):
    return [letter for letter, _ in parse_candidate_options(sample["target_str"])]


def generate_agent_planning_prompt(conf, sample):
    data_root, available_files = list_agent_available_files(conf)
    sample_view = build_agent_sample_view(sample)
    return (
        "You are the planning agent for a bundle completion research system.\n"
        "Your job is to decide what evidence should be retrieved for this single test instance. "
        "Do not choose the final answer.\n\n"
        "Current sample JSON:\n"
        f"{json.dumps(sample_view, ensure_ascii=False, indent=2)}\n\n"
        "Available raw data files. The later code-writing agent may inspect these files only:\n"
        f"{json.dumps({'data_root': data_root, 'files': available_files}, ensure_ascii=False, indent=2)}\n\n"
        "Important restrictions:\n"
        "- Do not use or request bi_full.txt, any test/validation ground-truth file such as bi_test_gt.txt, or any true labels.\n"
        "- Do not use prior result CSV files, predictions, hits, or true option labels.\n"
        "- Prefer sample-specific evidence. Different samples may need different data sources.\n\n"
        "Return only valid JSON using this schema:\n"
        "{\n"
        '  "sample_diagnosis": "...",\n'
        '  "planned_sources": ["item_info", "bi_train", "ui_full", "text_embeddings"],\n'
        '  "avoid_sources": ["..."],\n'
        '  "analysis_tasks": ["..."],\n'
        '  "expected_evidence_format": "candidate-level evidence with numeric signals when possible"\n'
        "}\n"
    )


def generate_agent_code_prompt(conf, sample, planning_text):
    data_root, available_files = list_agent_available_files(conf)
    sample_view = build_agent_sample_view(sample)
    max_stdout_chars = int(conf.get("agent_code_max_stdout_chars", 20000))
    labels = candidate_labels(sample)
    return (
        "You are the code-writing retrieval agent for a bundle completion task.\n"
        "Write Python code that retrieves and analyzes evidence for the current sample. "
        "The code must print exactly one JSON object to stdout and must not print anything else.\n\n"
        "Planning agent output:\n"
        f"{planning_text}\n\n"
        "Current sample JSON:\n"
        f"{json.dumps(sample_view, ensure_ascii=False, indent=2)}\n\n"
        "Available raw data files:\n"
        f"{json.dumps({'data_root': data_root, 'files': available_files}, ensure_ascii=False, indent=2)}\n\n"
        "Rules:\n"
        "- You may use Python standard library and installed scientific packages if available.\n"
        "- You may compute co-affiliation from bi_train.txt, co-purchase from ui_full.txt, and cosine similarity from embedding files if you can load them.\n"
        "- Do not read bi_full.txt, bi_test_gt.txt, validation/test ground truth files, result CSV files, or any file containing predictions/hits/true labels.\n"
        "- Do not use item_cf_feature.pt unless it is explicitly listed in the available raw data files.\n"
        "- Be robust: if a source cannot be loaded, record that in warnings and continue.\n"
        f"- Keep the printed JSON under about {max_stdout_chars} characters.\n"
        "- Do not choose the final answer as a separate act; only provide evidence and optional preliminary evidence ranking.\n"
        f"- Include every candidate label exactly once: {', '.join(labels)}.\n\n"
        "The printed JSON must follow this schema:\n"
        "{\n"
        '  "executed_code_summary": "...",\n'
        '  "sources_used": ["item_info", "bi_train"],\n'
        '  "global_findings": ["..."],\n'
        '  "candidate_evidence": {\n'
        '    "A": {\n'
        '      "metadata_summary": "...",\n'
        '      "evidence_for": ["..."],\n'
        '      "evidence_against": ["..."],\n'
        '      "bi_evidence": "...",\n'
        '      "ui_evidence": "...",\n'
        '      "embedding_evidence": "...",\n'
        '      "overall_evidence": "..."\n'
        "    }\n"
        "  },\n"
        '  "numeric_signals": {\n'
        '    "A": {\n'
        '      "bi_coaffiliation_count": null,\n'
        '      "ui_copurchase_count": null,\n'
        '      "embedding_avg_cosine": null,\n'
        '      "embedding_max_cosine": null,\n'
        '      "train_popularity_count": null\n'
        "    }\n"
        "  },\n"
        '  "warnings": ["..."]\n'
        "}\n\n"
        "Return only the Python code. Do not wrap it in explanation."
    )


def generate_agent_code_repair_prompt(conf, sample, planning_text, previous_code, execution_result):
    _, available_files = list_agent_available_files(conf)
    sample_view = build_agent_sample_view(sample)
    labels = candidate_labels(sample)
    return (
        "You are repairing Python retrieval code for a bundle completion task.\n"
        "The previous code failed or did not print valid JSON. Write a corrected complete Python script.\n"
        "Print exactly one JSON object to stdout and nothing else.\n\n"
        "Current sample JSON:\n"
        f"{json.dumps(sample_view, ensure_ascii=False, indent=2)}\n\n"
        "Planning agent output:\n"
        f"{planning_text}\n\n"
        "Available raw data files:\n"
        f"{json.dumps({'files': available_files}, ensure_ascii=False, indent=2)}\n\n"
        "Previous code:\n"
        f"```python\n{previous_code[:12000]}\n```\n\n"
        "Execution result:\n"
        f"{json.dumps(execution_result, ensure_ascii=False, indent=2)}\n\n"
        "Repair requirements:\n"
        "- Do not read bi_full.txt, bi_test_gt.txt, validation/test ground truth files, result CSV files, or true labels.\n"
        "- Use robust fallbacks if optional embedding files cannot be loaded.\n"
        f"- Include every candidate label exactly once: {', '.join(labels)}.\n"
        "- Keep candidate_evidence with metadata_summary, evidence_for, evidence_against, bi_evidence, ui_evidence, embedding_evidence, and overall_evidence.\n"
        "- Keep numeric_signals with bi_coaffiliation_count, ui_copurchase_count, embedding_avg_cosine, embedding_max_cosine, and train_popularity_count.\n\n"
        "Return only the corrected Python code."
    )


def execute_generated_python_code(code, conf):
    timeout = int(conf.get("agent_code_timeout_seconds", 30))
    max_stdout_chars = int(conf.get("agent_code_max_stdout_chars", 20000))
    max_stderr_chars = int(conf.get("agent_code_max_stderr_chars", 8000))
    with tempfile.TemporaryDirectory(prefix="bundle_agent_code_") as tmpdir:
        script_path = os.path.join(tmpdir, "agent_retrieval.py")
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(code)
            f.write("\n")

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        try:
            completed = subprocess.run(
                [sys.executable, script_path],
                cwd=os.getcwd(),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                env=env,
            )
            stdout = completed.stdout[-max_stdout_chars:]
            stderr = completed.stderr[-max_stderr_chars:]
            return {
                "returncode": completed.returncode,
                "stdout": stdout,
                "stderr": stderr,
                "timed_out": False,
            }
        except subprocess.TimeoutExpired as exc:
            return {
                "returncode": None,
                "stdout": (exc.stdout or "")[-max_stdout_chars:] if isinstance(exc.stdout, str) else "",
                "stderr": (exc.stderr or "")[-max_stderr_chars:] if isinstance(exc.stderr, str) else "",
                "timed_out": True,
            }


def code_execution_needs_repair(execution_result):
    if execution_result.get("timed_out"):
        return True
    if execution_result.get("returncode") != 0:
        return True
    return parse_json_from_text(execution_result.get("stdout", "")) is None


def generate_agent_prediction_prompt(conf, sample, planning_text, code_text, execution_result):
    sample_view = build_agent_sample_view(sample)
    labels = candidate_labels(sample)
    evidence_json = parse_json_from_text(execution_result.get("stdout", ""))
    evidence_text = (
        json.dumps(evidence_json, ensure_ascii=False, indent=2)
        if evidence_json is not None
        else execution_result.get("stdout", "")
    )
    return (
        "You are the final prediction agent for a bundle completion task.\n"
        "Use the original sample and the retrieved evidence to choose one candidate letter. "
        "Do not assume the retrieval evidence is always reliable; weigh semantic metadata, interaction evidence, and embedding evidence carefully. "
        "Compare both evidence_for and evidence_against for each candidate, and avoid choosing a candidate only because one numeric signal is high.\n\n"
        "Current sample JSON:\n"
        f"{json.dumps(sample_view, ensure_ascii=False, indent=2)}\n\n"
        "Planning agent output:\n"
        f"{planning_text}\n\n"
        "Executed retrieval code:\n"
        f"```python\n{code_text[:12000]}\n```\n\n"
        "Code execution result:\n"
        f"{json.dumps({'returncode': execution_result.get('returncode'), 'stderr': execution_result.get('stderr'), 'timed_out': execution_result.get('timed_out')}, ensure_ascii=False, indent=2)}\n\n"
        "Retrieved evidence:\n"
        f"{evidence_text[:20000]}\n\n"
        f"In candidate_tradeoff, include every candidate label exactly once: {', '.join(labels)}.\n"
        "Return only valid JSON using this schema:\n"
        "{\n"
        '  "source_reliability_assessment": {\n'
        '    "item_info": "high|medium|low plus short reason",\n'
        '    "bi_train": "high|medium|low plus short reason",\n'
        '    "ui_full": "high|medium|low plus short reason",\n'
        '    "text_embeddings": "high|medium|low plus short reason"\n'
        "  },\n"
        '  "candidate_tradeoff": {\n'
        '    "A": "main evidence for and against this candidate",\n'
        '    "B": "main evidence for and against this candidate"\n'
        "  },\n"
        '  "decision_rule": "how you balanced semantic compatibility, collaborative evidence, embeddings, and counter-evidence",\n'
        '  "reasoning": "concise final comparison across candidates",\n'
        '  "prediction": "A",\n'
        '  "confidence": "low|medium|high",\n'
        '  "main_sources_used_for_decision": ["item_info", "bi_train"]\n'
        "}\n"
    )


def is_quota_error(exc):
    message = str(exc).lower()
    quota_markers = [
        "403",
        "quota",
        "rate limit exceeded",
        "resource exhausted",
        "permission denied",
        "billing",
    ]
    return any(marker in message for marker in quota_markers)


def is_retryable_error(exc):
    message = str(exc).lower()
    retry_markers = [
        "503",
        "high demand",
        "overloaded",
        "service unavailable",
        "temporarily unavailable",
        "try again later",
        "unavailable",
    ]
    return any(marker in message for marker in retry_markers)


async def generate_content_with_retry(client, model, contents, conf, max_output_tokens, step_name):
    max_retries = int(conf.get("max_retries", 5))
    retry_wait_seconds = float(conf.get("retry_wait_seconds", 30))

    for attempt in range(max_retries + 1):
        try:
            response = await client.aio.models.generate_content(
                model=model,
                contents=contents,
                config={
                    "temperature": float(conf.get("temperature", 0.0)),
                    "max_output_tokens": int(max_output_tokens),
                },
            )
            return response.text or ""
        except Exception as exc:
            if is_quota_error(exc):
                raise QuotaExceededError(f"Quota or permission error during {step_name}: {exc}") from exc
            if is_retryable_error(exc) and attempt < max_retries:
                wait_time = retry_wait_seconds * (attempt + 1)
                print(
                    f"[Retry] {step_name} failed with retryable error "
                    f"({attempt + 1}/{max_retries}); waiting {wait_time:.1f}s"
                )
                await asyncio.sleep(wait_time)
                continue
            raise


def generate_prompt(dataset_name, input_str, target_str):
    if "spotify" in dataset_name:
        task_name = "playlist continuation"
        bundle_name = "music playlist"
        item_name = "song"
    else:
        task_name = "bundle construction"
        bundle_name = "fashion outfit"
        item_name = "fashion item"

    return (
        f"You are a helpful and honest assistant. The following are multiple choice questions about {task_name}. "
        f"You should directly answer the question by choosing the letter of the correct option. "
        f"Only provide the letter of your answer, without any explanation or mentioning the option content.\n"
        f"Question: Given the partial {bundle_name}: {input_str}, "
        f"which candidate {item_name} should be included into this {bundle_name}?\n"
        f"Options: {target_str}\n"
        f"Your answer should indicate your choice with a single letter (e.g., \"A,\" \"B,\" \"C,\" etc.).\nChoice: "
    )


def generate_candidate_reasoning_prompt(dataset_name, input_str, candidate_options):
    if "spotify" in dataset_name:
        task_name = "playlist continuation"
        bundle_name = "music playlist"
        criteria = (
            "Discuss whether each playlist feels coherent in theme, mood, genre, artist or album context, "
            "and listening flow."
        )
    else:
        task_name = "bundle construction"
        bundle_name = "fashion outfit"
        criteria = (
            "Discuss whether each outfit feels coherent in concept, seasonality, style, color or material harmony, "
            "and item-category compatibility."
        )

    completed_bundles = []
    for letter, candidate_text in candidate_options:
        completed_bundles.append(f"{letter}: {input_str}; {candidate_text}")
    completed_bundle_block = "\n".join(completed_bundles)

    letters = ", ".join(letter for letter, _ in candidate_options)
    return (
        f"You are a {task_name} analyst.\n"
        f"Review the following completed {bundle_name}s. Each line appends one possible final item to the same input items.\n"
        f"{completed_bundle_block}\n"
        f"For each completed {bundle_name}, provide reasoning about how well the items work together. {criteria}\n"
        f"Write only concise reasoning in English. Use 2-3 sentences for each label. Do not choose an answer.\n"
        f"Return exactly one reasoning paragraph for each label ({letters}) using this format:\n"
        f"A: reasoning text\nB: reasoning text\n...\nReasoning:\n"
    )


def parse_candidate_reasonings(raw_text, candidate_options):
    reasonings = []
    text = str(raw_text or "").strip()
    for idx, (letter, _) in enumerate(candidate_options):
        next_letter = candidate_options[idx + 1][0] if idx + 1 < len(candidate_options) else None
        if next_letter:
            pattern = rf"(?:^|\n)\s*{letter}\s*[:\).\-]\s*(.*?)(?=\n\s*{next_letter}\s*[:\).\-]\s*|$)"
        else:
            pattern = rf"(?:^|\n)\s*{letter}\s*[:\).\-]\s*(.*)$"
        match = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
        reasoning = " ".join(match.group(1).split()) if match else ""
        reasonings.append((letter, reasoning))
    return reasonings

def generate_prediction_from_reasoning_prompt(dataset_name, input_str, target_str, candidate_reasonings):
    if "spotify" in dataset_name:
        task_name = "playlist continuation"
        bundle_name = "music playlist"
        item_name = "song"
    else:
        task_name = "bundle construction"
        bundle_name = "fashion outfit"
        item_name = "fashion item"

    reasoning_by_letter = {letter: reasoning for letter, reasoning in candidate_reasonings}
    option_lines = []
    for letter, candidate_text in parse_candidate_options(target_str):
        reasoning = reasoning_by_letter.get(letter, "")
        option_lines.append(f"{letter}. {candidate_text}\nReasoning: {reasoning}")
    options_with_reasoning = "\n".join(option_lines)

    return (
        f"You are a helpful and honest assistant. The following are multiple choice questions about {task_name}. "
        f"You should directly answer the question by choosing the letter of the correct option. "
        f"Only provide the letter of your answer, without any explanation or mentioning the option content.\n"
        f"Question: Given the partial {bundle_name}: {input_str}, "
        f"which candidate {item_name} should be included into this {bundle_name}?\n"
        f"Options with reasoning:\n{options_with_reasoning}\n"
        f"Your answer should indicate your choice with a single letter (e.g., \"A,\" \"B,\" \"C,\" etc.).\nChoice: "
    )


def print_llm_prompt_debug(title, prompt):
    print(f"\n[DEBUG] {title}:")
    print(console_safe_text(prompt))
    print("-" * 50 + "\n")


def print_first_qa_debug(sample, prompt):
    print("\n[DEBUG] First QA Preview:")
    print(f"  [Bundle ID] {sample.get('bundle_id')}")
    print(f"  [True Option] {sample.get('true_option_char')} | True Item ID: {sample.get('true_indice')}")
    print(f"  [Input Item IDs] {sample.get('input_indices')}")
    print(f"  [Candidate Item IDs] {sample.get('candidate_indices')}")
    print("\n[DEBUG] First Question:")
    print(console_safe_text(sample.get("input_str", "")))
    print("\n[DEBUG] First Options:")
    print(console_safe_text(sample.get("target_str", "")))
    print("\n[DEBUG] First Prompt Sent To Model:")
    print(console_safe_text(prompt))
    print("-" * 50 + "\n")


def result_path(conf, timestamp, partial=False):
    output_dir = os.path.join(conf["output_dir"], conf["dataset"])
    os.makedirs(output_dir, exist_ok=True)
    suffix = "_partial" if partial else ""
    filename = (
        f"results_{conf['dataset']}_{method_name(conf)}_"
        f"C{conf.get('num_cans', '')}_T{conf.get('num_token', '')}_"
        f"{timestamp}{suffix}.csv"
    )
    return os.path.join(output_dir, filename)


def save_results(results, conf, timestamp, partial=False):
    df = pd.DataFrame(results)
    hit_rate = df["hit"].mean() if not df.empty else 0.0
    valid_options = [chr(ord("A") + idx) for idx in range(int(conf.get("num_cans", 10)))]
    valid_mask = df["prediction"].isin(valid_options) if not df.empty else pd.Series(dtype=bool)
    valid_ratio = valid_mask.mean() if not df.empty else 0.0
    valid_only_hit_rate = df.loc[valid_mask, "hit"].mean() if valid_mask.sum() > 0 else 0.0

    if not df.empty:
        df["overall_hit_rate"] = hit_rate
        df["overall_valid_ratio"] = valid_ratio
        df["valid_only_hit_rate"] = valid_only_hit_rate
        df["cfg_dataset"] = conf.get("dataset", "")
        df["cfg_num_cans"] = conf.get("num_cans", "")
        df["cfg_num_token"] = conf.get("num_token", "")
        df["cfg_toy_eval"] = conf.get("toy_eval", "")
        df["cfg_seed"] = conf.get("seed", "")
        df["cfg_shuffle_seed"] = conf.get("shuffle_seed", "")
        df["cfg_model"] = conf.get("model", "")
        df["cfg_temperature"] = conf.get("temperature", "")
        df["cfg_prediction_api_key_env"] = conf.get("prediction_api_key_env", "")
        df["cfg_reasoning_api_key_env"] = conf.get("reasoning_api_key_env", "")
        df["cfg_use_candidate_reasoning"] = use_candidate_reasoning(conf)
        df["cfg_candidate_reasoning_max_output_tokens"] = conf.get("candidate_reasoning_max_output_tokens", "")
        df["cfg_use_three_stage_agent"] = use_three_stage_agent(conf)
        df["cfg_agent_planning_api_key_env"] = conf.get("agent_planning_api_key_env", "")
        df["cfg_agent_code_api_key_env"] = conf.get("agent_code_api_key_env", "")
        df["cfg_agent_prediction_api_key_env"] = conf.get("agent_prediction_api_key_env", "")
        df["cfg_agent_planning_max_output_tokens"] = conf.get("agent_planning_max_output_tokens", "")
        df["cfg_agent_code_max_output_tokens"] = conf.get("agent_code_max_output_tokens", "")
        df["cfg_agent_prediction_max_output_tokens"] = conf.get("agent_prediction_max_output_tokens", "")
        df["cfg_agent_code_timeout_seconds"] = conf.get("agent_code_timeout_seconds", "")
        df["cfg_agent_code_max_repair_attempts"] = conf.get("agent_code_max_repair_attempts", "")
        df["cfg_agent_allowed_files"] = compact_json(conf.get("agent_allowed_files", []))
        df["cfg_agent_allow_interaction_embeddings"] = conf.get("agent_allow_interaction_embeddings", "")
        df["cfg_max_retries"] = conf.get("max_retries", "")
        df["cfg_retry_wait_seconds"] = conf.get("retry_wait_seconds", "")

    path = result_path(conf, timestamp, partial=partial)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path, df, hit_rate, valid_ratio, valid_only_hit_rate, int(valid_mask.sum()) if not df.empty else 0


async def process_samples(
    prediction_client,
    samples,
    conf,
    timestamp,
    initial_results=None,
    start_idx=0,
    reasoning_client=None,
    planning_client=None,
    code_client=None,
    agent_prediction_client=None,
):
    results = list(initial_results or [])
    concurrency = int(conf.get("max_concurrent", 10))
    semaphore = asyncio.Semaphore(concurrency)
    total = start_idx + len(samples)

    async def run_one(sample, current_idx):
        row = dict(sample)
        prompt = generate_prompt(conf["dataset"], sample["input_str"], sample["target_str"])

        if use_three_stage_agent(conf):
            async with semaphore:
                planning_prompt = generate_agent_planning_prompt(conf, sample)
                if current_idx == start_idx:
                    print_llm_prompt_debug("First Agent Planning Prompt Sent To Model", planning_prompt)
                try:
                    planning_raw_text = await generate_content_with_retry(
                        planning_client or prediction_client,
                        conf["model"],
                        planning_prompt,
                        conf,
                        conf.get("agent_planning_max_output_tokens", 800),
                        f"sample {current_idx + 1} agent planning",
                    )
                except QuotaExceededError:
                    raise
                except Exception as exc:
                    planning_raw_text = str(exc)

                code_prompt = generate_agent_code_prompt(conf, sample, planning_raw_text)
                if current_idx == start_idx:
                    print_llm_prompt_debug("First Agent Code Prompt Sent To Model", code_prompt)
                try:
                    code_raw_text = await generate_content_with_retry(
                        code_client or prediction_client,
                        conf["model"],
                        code_prompt,
                        conf,
                        conf.get("agent_code_max_output_tokens", 2200),
                        f"sample {current_idx + 1} agent code writing",
                    )
                except QuotaExceededError:
                    raise
                except Exception as exc:
                    code_raw_text = str(exc)

                generated_code = extract_python_code(code_raw_text)
                execution_result = await asyncio.to_thread(execute_generated_python_code, generated_code, conf)
                repair_attempts_used = 0
                repair_raw_responses = []
                max_repair_attempts = int(conf.get("agent_code_max_repair_attempts", 1))

                while code_execution_needs_repair(execution_result) and repair_attempts_used < max_repair_attempts:
                    repair_attempts_used += 1
                    repair_prompt = generate_agent_code_repair_prompt(
                        conf, sample, planning_raw_text, generated_code, execution_result
                    )
                    if current_idx == start_idx:
                        print_llm_prompt_debug(
                            f"First Agent Code Repair Prompt {repair_attempts_used} Sent To Model",
                            repair_prompt,
                        )
                    try:
                        repair_raw_text = await generate_content_with_retry(
                            code_client or prediction_client,
                            conf["model"],
                            repair_prompt,
                            conf,
                            conf.get("agent_code_max_output_tokens", 2200),
                            f"sample {current_idx + 1} agent code repair {repair_attempts_used}",
                        )
                    except QuotaExceededError:
                        raise
                    except Exception as exc:
                        repair_raw_text = str(exc)

                    repair_raw_responses.append(repair_raw_text)
                    generated_code = extract_python_code(repair_raw_text)
                    execution_result = await asyncio.to_thread(execute_generated_python_code, generated_code, conf)

                evidence_json = parse_json_from_text(execution_result.get("stdout", ""))

                prediction_prompt = generate_agent_prediction_prompt(
                    conf, sample, planning_raw_text, generated_code, execution_result
                )
                if current_idx == start_idx:
                    print_first_qa_debug(sample, prediction_prompt)
                try:
                    final_raw_text = await generate_content_with_retry(
                        agent_prediction_client or prediction_client,
                        conf["model"],
                        prediction_prompt,
                        conf,
                        conf.get("agent_prediction_max_output_tokens", 800),
                        f"sample {current_idx + 1} agent final prediction",
                    )
                    final_json = parse_json_from_text(final_raw_text)
                    if isinstance(final_json, dict) and final_json.get("prediction"):
                        prediction = parse_model_response(str(final_json.get("prediction", "")))
                    else:
                        prediction = parse_model_response(final_raw_text)
                except QuotaExceededError:
                    raise
                except Exception as exc:
                    final_raw_text = str(exc)
                    final_json = None
                    prediction = "ERR_EX"

                row["agent_planning_raw_response"] = planning_raw_text
                row["agent_planning_json"] = compact_json(parse_json_from_text(planning_raw_text))
                row["agent_code_raw_response"] = code_raw_text
                row["agent_code_repair_raw_responses"] = compact_json(repair_raw_responses)
                row["agent_code_repair_attempts_used"] = repair_attempts_used
                row["agent_generated_code"] = generated_code
                row["agent_code_returncode"] = execution_result.get("returncode")
                row["agent_code_stdout"] = execution_result.get("stdout", "")
                row["agent_code_stderr"] = execution_result.get("stderr", "")
                row["agent_code_timed_out"] = execution_result.get("timed_out", False)
                row["agent_evidence_json"] = compact_json(evidence_json)
                row["agent_prediction_raw_response"] = final_raw_text
                row["agent_prediction_json"] = compact_json(final_json)
                if isinstance(final_json, dict):
                    row["agent_reasoning"] = final_json.get("reasoning", "")
                    row["agent_confidence"] = final_json.get("confidence", "")
                    row["agent_main_sources_used_for_decision"] = compact_json(
                        final_json.get("main_sources_used_for_decision", [])
                    )
                    row["agent_source_reliability_assessment"] = compact_json(
                        final_json.get("source_reliability_assessment", {})
                    )
                    row["agent_candidate_tradeoff"] = compact_json(final_json.get("candidate_tradeoff", {}))
                    row["agent_decision_rule"] = final_json.get("decision_rule", "")
                raw_text = final_raw_text

        elif use_candidate_reasoning(conf):
            options = parse_candidate_options(sample["target_str"])
            async with semaphore:
                reasoning_prompt = generate_candidate_reasoning_prompt(
                    conf["dataset"], sample["input_str"], options
                )
                if current_idx == start_idx:
                    print_llm_prompt_debug("First Candidate Reasoning Prompt Sent To Model", reasoning_prompt)
                try:
                    reasoning_raw_text = await generate_content_with_retry(
                        reasoning_client or prediction_client,
                        conf["model"],
                        reasoning_prompt,
                        conf,
                        conf.get("candidate_reasoning_max_output_tokens", 1200),
                        f"sample {current_idx + 1} candidate reasoning",
                    )
                except QuotaExceededError:
                    raise
                except Exception as exc:
                    reasoning_raw_text = str(exc)

                row["reasoning_raw_response"] = reasoning_raw_text
                candidate_reasonings = parse_candidate_reasonings(reasoning_raw_text, options)
                for candidate_letter, reasoning_text in candidate_reasonings:
                    row[f"reasoning_{candidate_letter}"] = reasoning_text

                prompt = generate_prediction_from_reasoning_prompt(
                    conf["dataset"], sample["input_str"], sample["target_str"], candidate_reasonings
                )
                if current_idx == start_idx:
                    print_first_qa_debug(sample, prompt)

                try:
                    raw_text = await generate_content_with_retry(
                        prediction_client,
                        conf["model"],
                        prompt,
                        conf,
                        conf.get("max_output_tokens", 10),
                        f"sample {current_idx + 1} prediction",
                    )
                    prediction = parse_model_response(raw_text)
                except QuotaExceededError:
                    raise
                except Exception as exc:
                    raw_text = str(exc)
                    prediction = "ERR_EX"
        else:
            if current_idx == start_idx:
                print_first_qa_debug(sample, prompt)

            async with semaphore:
                try:
                    raw_text = await generate_content_with_retry(
                        prediction_client,
                        conf["model"],
                        prompt,
                        conf,
                        conf.get("max_output_tokens", 10),
                        f"sample {current_idx + 1} prediction",
                    )
                    prediction = parse_model_response(raw_text)
                except QuotaExceededError:
                    raise
                except Exception as exc:
                    raw_text = str(exc)
                    prediction = "ERR_EX"

        row["prediction"] = prediction
        row["raw_response"] = raw_text
        row["hit"] = int(prediction == sample["true_option_char"])
        print(f"[{current_idx + 1}/{total}] True: {sample['true_option_char']} | Pred: {prediction}")
        return row

    for offset in range(0, len(samples), concurrency):
        chunk = samples[offset : offset + concurrency]
        tasks = [run_one(sample, start_idx + offset + idx) for idx, sample in enumerate(chunk)]
        try:
            completed_rows = await asyncio.gather(*tasks)
        except QuotaExceededError:
            save_results(results, conf, timestamp, partial=True)
            raise
        for row in completed_rows:
            results.append(row)
        save_results(results, conf, timestamp, partial=True)

    return results


def load_resume(path):
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp949")


def main():
    parser = argparse.ArgumentParser(description="Run baseline zero-shot bundle evaluation")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--resume", type=str, default="")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        conf = yaml.safe_load(f)

    os.makedirs(conf["output_dir"], exist_ok=True)
    set_seed(int(conf.get("seed", 45)))

    dataset = BundleZeroShotDataset(conf)
    samples = dataset.get_eval_samples()

    initial_results = None
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if args.resume and os.path.exists(args.resume):
        previous = load_resume(args.resume)
        initial_results = previous.to_dict("records")
        args.start_idx = len(initial_results)
        match = re.search(r"_(\d{8}_\d{6})(_partial)?\.csv$", args.resume)
        if match:
            timestamp = match.group(1)

    if args.start_idx > 0:
        samples = samples[args.start_idx :]

    print(f">>> Loaded config: {args.config}")
    print(f">>> Dataset: {conf['dataset']}")
    print(f">>> Total test samples prepared: {len(samples)} (start_idx={args.start_idx})")

    try:
        prediction_api_key, prediction_api_key_env = resolve_api_key(
            conf,
            "prediction_api_key_env",
            ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
        )
        prediction_client = genai.Client(api_key=prediction_api_key)
        print(f">>> Prediction API key env: {prediction_api_key_env}")

        reasoning_client = None
        planning_client = None
        code_client = None
        agent_prediction_client = None
        if use_candidate_reasoning(conf):
            reasoning_api_key, reasoning_api_key_env = resolve_api_key(
                conf,
                "reasoning_api_key_env",
                [prediction_api_key_env, "GEMINI_API_KEY", "GOOGLE_API_KEY"],
            )
            reasoning_client = genai.Client(api_key=reasoning_api_key)
            print(f">>> Reasoning API key env: {reasoning_api_key_env}")

        if use_three_stage_agent(conf):
            planning_api_key, planning_api_key_env = resolve_api_key(
                conf,
                "agent_planning_api_key_env",
                [prediction_api_key_env, "GEMINI_API_KEY", "GOOGLE_API_KEY"],
            )
            code_api_key, code_api_key_env = resolve_api_key(
                conf,
                "agent_code_api_key_env",
                [planning_api_key_env, prediction_api_key_env, "GEMINI_API_KEY", "GOOGLE_API_KEY"],
            )
            agent_prediction_api_key, agent_prediction_api_key_env = resolve_api_key(
                conf,
                "agent_prediction_api_key_env",
                [prediction_api_key_env, planning_api_key_env, "GEMINI_API_KEY", "GOOGLE_API_KEY"],
            )
            planning_client = genai.Client(api_key=planning_api_key)
            code_client = genai.Client(api_key=code_api_key)
            agent_prediction_client = genai.Client(api_key=agent_prediction_api_key)
            print(f">>> Agent planning API key env: {planning_api_key_env}")
            print(f">>> Agent code API key env: {code_api_key_env}")
            print(f">>> Agent prediction API key env: {agent_prediction_api_key_env}")
    except ValueError as exc:
        print(f"[Error] {exc}")
        return 1

    try:
        results = asyncio.run(
            process_samples(
                prediction_client,
                samples,
                conf,
                timestamp,
                initial_results=initial_results,
                start_idx=args.start_idx,
                reasoning_client=reasoning_client,
                planning_client=planning_client,
                code_client=code_client,
                agent_prediction_client=agent_prediction_client,
            )
        )
    except QuotaExceededError as exc:
        partial_path = result_path(conf, timestamp, partial=True)
        print(f"[Stopped] {exc}")
        print(f"[Resume] Completed samples were saved to: {partial_path}")
        return 1

    save_path, df, hit_rate, valid_ratio, valid_only_hit_rate, valid_sum = save_results(
        results, conf, timestamp, partial=False
    )
    partial_path = result_path(conf, timestamp, partial=True)
    if os.path.exists(partial_path):
        os.remove(partial_path)

    print("-" * 30)
    print(f"Saved to: {save_path}")
    print(f"Overall Hit Rate: {hit_rate:.4f}")
    print(f"Valid-Only Hit Rate: {valid_only_hit_rate:.4f} (from {valid_sum} valid samples)")
    print(f"Valid Ratio: {valid_ratio:.4f}")
    print("-" * 30)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

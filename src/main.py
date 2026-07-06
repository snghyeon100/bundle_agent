"""main.py — Entry point for the code-generation bundle-completion pipeline.

Usage:
    python src/main.py --config config_sem.yaml
    python src/main.py --config config_sem.yaml --resume results/.../partial.csv
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time

import pandas as pd
import yaml
from dotenv import load_dotenv
from google import genai

from dataset import BundleZeroShotDataset, set_seed
from code import run_code_agent
from code.common import compact_json


load_dotenv(
    dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"),
    encoding="utf-8-sig",
)

METHOD_NAME = "code"


# ---------------------------------------------------------------------------
# LLM infrastructure  (reused from main.py patterns)
# ---------------------------------------------------------------------------

class QuotaExceededError(RuntimeError):
    pass


def llm_provider(conf):
    return str(conf.get("llm_provider", "gemini")).strip().lower()


def default_api_key_envs_for_provider(provider):
    return ["OPENAI_API_KEY"] if provider == "openai" else ["GEMINI_API_KEY", "GOOGLE_API_KEY"]


def default_api_key_envs(conf):
    return default_api_key_envs_for_provider(llm_provider(conf))


def create_llm_client(provider, api_key):
    if provider == "gemini":
        return genai.Client(api_key=api_key)
    if provider == "openai":
        from openai import AsyncOpenAI
        return AsyncOpenAI(api_key=api_key)
    raise ValueError(f"Unsupported llm_provider: {provider}")


STAGE_CONFIG = {
    "code_generation": {
        "provider_keys": [
            "code_generation_provider",
            "code_provider",
            "sem_evidence_provider",
            "sem_stage1_provider",
        ],
        "model_keys": [
            "code_generation_model",
            "code_model",
            "sem_evidence_model",
            "sem_stage1_model",
        ],
    },
    "prediction": {
        "provider_keys": [
            "code_prediction_provider",
            "code_decision_provider",
            "sem_decision_provider",
            "sem_prediction_provider",
        ],
        "model_keys": [
            "code_prediction_model",
            "code_decision_model",
            "sem_decision_model",
            "sem_prediction_model",
        ],
    },
}


def _first_config_value(conf, keys, default=""):
    for key in keys:
        value = str(conf.get(key, "")).strip()
        if value:
            return value
    return default


def stage_provider(conf, role):
    keys = STAGE_CONFIG.get(role, {}).get("provider_keys", [])
    return _first_config_value(conf, keys, llm_provider(conf)).lower()


def stage_model(conf, role):
    keys = STAGE_CONFIG.get(role, {}).get("model_keys", [])
    model = _first_config_value(conf, keys, str(conf.get("model", "")).strip())
    if not model:
        raise ValueError(f"No model configured for role {role}")
    return model


def resolve_api_key(conf, config_key, fallback_envs):
    return resolve_api_key_from_keys(conf, [config_key], fallback_envs)


def resolve_api_key_from_keys(conf, config_keys, fallback_envs):
    for config_key in config_keys:
        configured = str(conf.get(config_key, "")).strip()
        if configured:
            value = os.getenv(configured, "").strip()
            if not value:
                raise ValueError(f"API key env var not set: {configured} (configured by {config_key})")
            return value, configured
    for env in fallback_envs:
        value = os.getenv(env, "").strip()
        if value:
            return value, env
    configured_keys = ", ".join(config_keys)
    raise ValueError(
        f"No API key configured via [{configured_keys}], and none of these fallback env vars are set: "
        f"{', '.join(fallback_envs)}"
    )


def is_quota_error(exc):
    msg = str(exc).lower()
    return any(m in msg for m in ("403", "quota", "rate limit exceeded", "resource exhausted",
                                   "permission denied", "billing"))


def is_retryable_error(exc):
    msg = str(exc).lower()
    return any(m in msg for m in ("503", "high demand", "overloaded", "service unavailable",
                                   "temporarily unavailable", "try again later",
                                   "connection error", "api connection error",
                                   "connection reset", "connection aborted",
                                   "timeout", "timed out", "read timeout"))


def _client_provider(client, conf):
    if isinstance(client, dict):
        return str(client.get("provider", llm_provider(conf))).strip().lower()
    return llm_provider(conf)


def _client_obj(client):
    if isinstance(client, dict):
        return client.get("client")
    return client


def openai_model_supports_reasoning(model):
    name = str(model or "").strip().lower()
    return name.startswith("o") or name.startswith("gpt-5")


async def call_llm_once(client, model, contents, conf, max_output_tokens):
    provider = _client_provider(client, conf)
    llm_client = _client_obj(client)
    if provider == "gemini":
        response = await llm_client.aio.models.generate_content(
            model=model,
            contents=contents,
            config={
                "temperature": float(conf.get("temperature", 0.0)),
                "max_output_tokens": int(max_output_tokens),
            },
        )
        return response.text or ""
    request = {
        "model": model,
        "input": contents,
        "max_output_tokens": int(max_output_tokens),
    }
    effort = str(conf.get("openai_reasoning_effort", "")).strip()
    if effort and openai_model_supports_reasoning(model):
        request["reasoning"] = {"effort": effort}
    if bool(conf.get("openai_send_temperature", False)):
        request["temperature"] = float(conf.get("temperature", 0.0))
    response = await llm_client.responses.create(**request)
    return getattr(response, "output_text", "") or ""


async def generate_content_with_retry(client, model, contents, conf, max_output_tokens, step_name):
    max_retries = int(conf.get("max_retries", 5))
    base_wait = float(conf.get("retry_wait_seconds", 30))
    for attempt in range(max_retries + 1):
        try:
            return await call_llm_once(client, model, contents, conf, max_output_tokens)
        except Exception as exc:
            if is_quota_error(exc):
                raise QuotaExceededError(f"Quota error during {step_name}: {exc}") from exc
            if is_retryable_error(exc) and attempt < max_retries:
                wait = base_wait * (attempt + 1)
                print(f"[Retry] {step_name} ({attempt + 1}/{max_retries}); waiting {wait:.1f}s")
                await asyncio.sleep(wait)
                continue
            raise


def parse_model_response(raw_text):
    if not raw_text:
        return "ERR_EM"
    clean = re.sub(r"(?i)\b(choice|option|answer|prediction)\b\s*[:=]*\s*", "", str(raw_text).strip())
    match = re.search(r"([A-Z])", clean.upper())
    return match.group(1) if match else clean[:1].upper()


def console_safe(text):
    enc = sys.stdout.encoding or "utf-8"
    return str(text).encode(enc, errors="backslashreplace").decode(enc)


def print_debug(title, prompt):
    print(f"\n[DEBUG] {title}:")
    print(console_safe(prompt))
    print("-" * 60 + "\n")


# ---------------------------------------------------------------------------
# Results I/O
# ---------------------------------------------------------------------------

def run_output_dir(conf, timestamp):
    directory = os.path.join(conf["output_dir"], conf["dataset"], timestamp)
    os.makedirs(directory, exist_ok=True)
    return directory


def result_path(conf, timestamp, partial=False):
    filename = "results_partial.csv" if partial else "results.csv"
    return os.path.join(run_output_dir(conf, timestamp), filename)


def _format_id_list(values):
    return ";".join(str(int(v)) for v in values)


def _format_candidate_ids(values):
    labels = [chr(ord("A") + i) for i in range(len(values))]
    return ";".join(f"{label}:{int(item_id)}" for label, item_id in zip(labels, values))


def compact_result_row(row):
    return {
        "bundle_id": int(row.get("bundle_id", -1)),
        "partial_item_ids": _format_id_list(row.get("input_indices", [])),
        "candidate_item_ids": _format_candidate_ids(row.get("candidate_indices", [])),
        "prediction": row.get("prediction", ""),
        "gt_label": row.get("true_option_char", ""),
        "gt_item_id": row.get("true_indice", ""),
        "hit": int(row.get("hit", 0)),
    }


def _write_text(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(str(text or ""))


def _write_json_text(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    value = str(text or "").strip()
    if value:
        try:
            value = json.dumps(json.loads(value), ensure_ascii=False, indent=2, default=str)
        except json.JSONDecodeError:
            pass
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(value)


def save_stage_artifacts(row, conf, timestamp):
    bundle_id = row.get("bundle_id", "unknown")
    base = os.path.join(run_output_dir(conf, timestamp), f"bundle_{bundle_id}")
    stage_specs = [
        (
            "stage1_code_generation",
            "code_generation_prompt",
            "code_generation_raw_response",
            "code_generated_code",
        ),
        ("stage2_prediction", "code_prediction_prompt", "code_prediction_raw_response", None),
    ]
    for stage_name, input_key, output_key, code_key in stage_specs:
        stage_dir = os.path.join(base, stage_name)
        if input_key in row:
            _write_text(os.path.join(stage_dir, "input.txt"), row.get(input_key, ""))
        if output_key in row:
            _write_text(os.path.join(stage_dir, "output.txt"), row.get(output_key, ""))
        if code_key and str(row.get(code_key, "")).strip():
            _write_text(os.path.join(stage_dir, "code.py"), row.get(code_key, ""))

    stage1_dir = os.path.join(base, "stage1_code_generation")
    if str(row.get("code_evidence_json", "")).strip():
        _write_json_text(os.path.join(stage1_dir, "evidence.json"), row.get("code_evidence_json", ""))
    if str(row.get("code_execution_summary", "")).strip():
        _write_json_text(
            os.path.join(stage1_dir, "execution_summary.json"),
            row.get("code_execution_summary", ""),
        )

    stage2_dir = os.path.join(base, "stage2_prediction")
    if str(row.get("code_prediction_json", "")).strip():
        _write_json_text(os.path.join(stage2_dir, "prediction.json"), row.get("code_prediction_json", ""))
    if str(row.get("code_decision_case", "")).strip():
        _write_json_text(os.path.join(stage2_dir, "decision_case.json"), row.get("code_decision_case", ""))


def save_error_artifact(row, conf, timestamp, error_text):
    bundle_id = row.get("bundle_id", "unknown")
    path = os.path.join(run_output_dir(conf, timestamp), f"bundle_{bundle_id}", "errors", "output.txt")
    _write_text(path, error_text)


def save_results(results, conf, timestamp, partial=False):
    frame = pd.DataFrame(results)
    hit_rate = frame["hit"].mean() if not frame.empty else 0.0
    valid_labels = [chr(ord("A") + i) for i in range(int(conf.get("num_cans", 10)))]
    valid_mask = frame["prediction"].isin(valid_labels) if not frame.empty else pd.Series(dtype=bool)
    valid_ratio = valid_mask.mean() if not frame.empty else 0.0
    valid_hit_rate = frame.loc[valid_mask, "hit"].mean() if valid_mask.sum() else 0.0

    if not frame.empty:
        frame["accuracy"] = hit_rate
        columns = [
            "bundle_id",
            "partial_item_ids",
            "candidate_item_ids",
            "prediction",
            "gt_label",
            "gt_item_id",
            "hit",
            "accuracy",
        ]
        frame = frame[[col for col in columns if col in frame.columns]]

    path = result_path(conf, timestamp, partial)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return path, frame, hit_rate, valid_ratio, valid_hit_rate, int(valid_mask.sum()) if not frame.empty else 0


# ---------------------------------------------------------------------------
# Sample processing
# ---------------------------------------------------------------------------

async def process_samples(clients, samples, conf, timestamp, initial_results=None, start_idx=0):
    results = list(initial_results or [])
    concurrency = int(conf.get("max_concurrent", 1))
    semaphore = asyncio.Semaphore(concurrency)
    total = start_idx + len(samples)

    async def run_one(sample, current_idx):
        row = dict(sample)
        async with semaphore:
            try:
                updates, prediction, raw_response = await run_code_agent(
                    sample,
                    conf,
                    clients,
                    generate_content_with_retry,
                    parse_model_response,
                    debug_callback=print_debug,
                    is_first_sample=(current_idx == start_idx),
                )
                row.update(updates)
                row["prediction"] = prediction
                row["raw_response"] = raw_response
                row["hit"] = int(prediction == sample["true_option_char"])
                save_stage_artifacts(row, conf, timestamp)
            except QuotaExceededError:
                raise
            except Exception as exc:
                prediction = "ERR_EX"
                raw_response = str(exc)
                row["prediction"] = prediction
                row["raw_response"] = raw_response
                row["hit"] = int(prediction == sample["true_option_char"])
                save_error_artifact(row, conf, timestamp, raw_response)
        print(f"[{current_idx + 1}/{total}] True: {sample['true_option_char']} | Pred: {prediction}")
        return compact_result_row(row)

    for offset in range(0, len(samples), concurrency):
        chunk = samples[offset: offset + concurrency]
        tasks = [run_one(s, start_idx + offset + i) for i, s in enumerate(chunk)]
        try:
            completed = await asyncio.gather(*tasks)
        except QuotaExceededError:
            save_results(results, conf, timestamp, partial=True)
            raise
        results.extend(completed)
        save_results(results, conf, timestamp, partial=True)
    return results


def _build_clients(conf):
    clients = {}
    resolved = {}
    prior_by_provider = {}
    # The code method uses two roles. code_* keys are preferred; sem_* keys remain
    # fallbacks so existing config_sem.yaml files can run this method.
    role_keys = [
        (
            "code_generation",
            [
                "code_generation_api_key_env",
                "code_api_key_env",
                "sem_evidence_api_key_env",
                "sem_stage1_api_key_env",
                "sem_code_api_key_env",
            ],
        ),
        (
            "prediction",
            [
                "code_prediction_api_key_env",
                "code_decision_api_key_env",
                "sem_decision_api_key_env",
                "sem_prediction_api_key_env",
            ],
        ),
    ]
    for role, config_keys in role_keys:
        provider = stage_provider(conf, role)
        model = stage_model(conf, role)
        fallback = default_api_key_envs_for_provider(provider)
        prior = prior_by_provider.get(provider, [])
        api_key, env = resolve_api_key_from_keys(conf, config_keys, prior + fallback)
        clients[role] = {
            "client": create_llm_client(provider, api_key),
            "provider": provider,
            "model": model,
        }
        resolved[role] = {
            "api_key_env": env,
            "provider": provider,
            "model": model,
        }
        prior_by_provider.setdefault(provider, []).append(env)
    return clients, resolved


def load_resume(path):
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp949")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run code-method bundle evaluation")
    parser.add_argument("--config", default="config_code.yaml")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--resume", default="")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        conf = yaml.safe_load(f)

    os.makedirs(conf["output_dir"], exist_ok=True)
    set_seed(int(conf.get("seed", 45)))

    samples = BundleZeroShotDataset(conf).get_eval_samples()
    initial_results = None
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    if args.resume and os.path.exists(args.resume):
        prev = load_resume(args.resume)
        initial_results = prev.to_dict("records")
        args.start_idx = len(initial_results)
        m = re.search(r"_(\d{8}_\d{6})(?:_partial)?\.csv$", args.resume)
        if m:
            timestamp = m.group(1)
        else:
            parent = os.path.basename(os.path.dirname(os.path.abspath(args.resume)))
            if parent:
                timestamp = parent
    if args.start_idx > 0:
        samples = samples[args.start_idx:]

    print(f">>> Config:  {args.config}")
    print(f">>> Method:  {METHOD_NAME}")
    print(f">>> Dataset: {conf['dataset']}")
    print(f">>> Samples: {len(samples)} (start_idx={args.start_idx})")

    try:
        clients, resolved = _build_clients(conf)
    except ValueError as exc:
        print(f"[Error] {exc}")
        return 1
    print(f">>> Default LLM provider: {llm_provider(conf)}")
    for role, info in resolved.items():
        print(
            f">>> {role.title()}: provider={info['provider']} "
            f"model={info['model']} api_key_env={info['api_key_env']}"
        )

    try:
        results = asyncio.run(
            process_samples(
                clients, samples, conf, timestamp,
                initial_results=initial_results,
                start_idx=args.start_idx,
            )
        )
    except QuotaExceededError as exc:
        print(f"[Stopped] {exc}")
        print(f"[Resume]  Partial results: {result_path(conf, timestamp, partial=True)}")
        return 1

    path, _, hit_rate, valid_ratio, valid_hit_rate, valid_count = save_results(
        results, conf, timestamp, partial=False
    )
    partial = result_path(conf, timestamp, partial=True)
    if os.path.exists(partial):
        os.remove(partial)
    print("-" * 40)
    print(f"Saved to:           {path}")
    print(f"Overall Hit Rate:   {hit_rate:.4f}")
    print(f"Valid-Only Hit Rate:{valid_hit_rate:.4f}  ({valid_count} valid samples)")
    print(f"Valid Ratio:        {valid_ratio:.4f}")
    print("-" * 40)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

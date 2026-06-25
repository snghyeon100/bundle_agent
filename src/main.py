"""main_sem.py — Entry point for the sem_agent pipeline.

Usage:
    python main_sem.py --config config_sem.yaml
    python main_sem.py --config config_sem.yaml --resume results/.../partial.csv
"""

import argparse
import asyncio
import os
import re
import sys
import time

import pandas as pd
import yaml
from dotenv import load_dotenv
from google import genai

from dataset import BundleZeroShotDataset, set_seed
from sem_agent import run_sem_agent
from sem_agent.common import compact_json


load_dotenv(
    dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"),
    encoding="utf-8-sig",
)

METHOD_NAME = "sem_agent"


# ---------------------------------------------------------------------------
# LLM infrastructure  (reused from main.py patterns)
# ---------------------------------------------------------------------------

class QuotaExceededError(RuntimeError):
    pass


def llm_provider(conf):
    return str(conf.get("llm_provider", "gemini")).strip().lower()


def default_api_key_envs(conf):
    return ["OPENAI_API_KEY"] if llm_provider(conf) == "openai" else ["GEMINI_API_KEY", "GOOGLE_API_KEY"]


def create_llm_client(conf, api_key):
    provider = llm_provider(conf)
    if provider == "gemini":
        return genai.Client(api_key=api_key)
    if provider == "openai":
        from openai import AsyncOpenAI
        return AsyncOpenAI(api_key=api_key)
    raise ValueError(f"Unsupported llm_provider: {provider}")


def resolve_api_key(conf, config_key, fallback_envs):
    configured = str(conf.get(config_key, "")).strip()
    if configured:
        value = os.getenv(configured, "").strip()
        if not value:
            raise ValueError(f"API key env var not set: {configured}")
        return value, configured
    for env in fallback_envs:
        value = os.getenv(env, "").strip()
        if value:
            return value, env
    raise ValueError(f"None of these API key env vars are set: {', '.join(fallback_envs)}")


def is_quota_error(exc):
    msg = str(exc).lower()
    return any(m in msg for m in ("403", "quota", "rate limit exceeded", "resource exhausted",
                                   "permission denied", "billing"))


def is_retryable_error(exc):
    msg = str(exc).lower()
    return any(m in msg for m in ("503", "high demand", "overloaded", "service unavailable",
                                   "temporarily unavailable", "try again later"))


async def call_llm_once(client, model, contents, conf, max_output_tokens):
    if llm_provider(conf) == "gemini":
        response = await client.aio.models.generate_content(
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
    if effort:
        request["reasoning"] = {"effort": effort}
    if bool(conf.get("openai_send_temperature", False)):
        request["temperature"] = float(conf.get("temperature", 0.0))
    response = await client.responses.create(**request)
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

def result_path(conf, timestamp, partial=False):
    directory = os.path.join(conf["output_dir"], conf["dataset"])
    os.makedirs(directory, exist_ok=True)
    suffix = "_partial" if partial else ""
    filename = (
        f"results_{conf['dataset']}_{METHOD_NAME}_"
        f"C{conf.get('num_cans', '')}_T{conf.get('num_token', '')}_{timestamp}{suffix}.csv"
    )
    return os.path.join(directory, filename)


def save_results(results, conf, timestamp, partial=False):
    frame = pd.DataFrame(results)
    hit_rate = frame["hit"].mean() if not frame.empty else 0.0
    valid_labels = [chr(ord("A") + i) for i in range(int(conf.get("num_cans", 10)))]
    valid_mask = frame["prediction"].isin(valid_labels) if not frame.empty else pd.Series(dtype=bool)
    valid_ratio = valid_mask.mean() if not frame.empty else 0.0
    valid_hit_rate = frame.loc[valid_mask, "hit"].mean() if valid_mask.sum() else 0.0

    if not frame.empty:
        frame["overall_hit_rate"] = hit_rate
        frame["overall_valid_ratio"] = valid_ratio
        frame["valid_only_hit_rate"] = valid_hit_rate
        # Record config fields
        config_fields = [
            "method", "dataset", "num_cans", "num_token", "toy_eval", "seed",
            "shuffle_seed", "llm_provider", "model", "temperature",
            "sem_code_max_output_tokens", "sem_prediction_max_output_tokens",
            "sem_code_max_repair_attempts", "sem_max_evidence_chars",
            "sem_current_bundle_train_context_policy",
            "sem_code_timeout_seconds", "sem_enable_code_guard",
        ]
        for field in config_fields:
            frame[f"cfg_{field}"] = conf.get(field, "")
        frame["cfg_max_retries"] = conf.get("max_retries", "")
        frame["cfg_retry_wait_seconds"] = conf.get("retry_wait_seconds", "")

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
                updates, prediction, raw_response = await run_sem_agent(
                    sample,
                    conf,
                    clients,
                    generate_content_with_retry,
                    parse_model_response,
                    debug_callback=print_debug,
                    is_first_sample=(current_idx == start_idx),
                )
                row.update(updates)
            except QuotaExceededError:
                raise
            except Exception as exc:
                prediction = "ERR_EX"
                raw_response = str(exc)
        row["prediction"] = prediction
        row["raw_response"] = raw_response
        row["hit"] = int(prediction == sample["true_option_char"])
        print(f"[{current_idx + 1}/{total}] True: {sample['true_option_char']} | Pred: {prediction}")
        return row

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
    fallback = default_api_key_envs(conf)
    clients = {}
    resolved = {}
    prior = []
    # sem_agent uses three roles: stage1, stage2, and prediction (Decision)
    role_keys = [
        ("stage1", "sem_stage1_api_key_env"),
        ("stage2", "sem_stage2_api_key_env"),
        ("prediction", "sem_prediction_api_key_env"),
    ]
    for role, key in role_keys:
        api_key, env = resolve_api_key(conf, key, prior + fallback)
        clients[role] = create_llm_client(conf, api_key)
        resolved[role] = env
        prior.append(env)
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
    parser = argparse.ArgumentParser(description="Run sem_agent bundle evaluation")
    parser.add_argument("--config", default="config_sem.yaml")
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
    print(f">>> LLM provider: {llm_provider(conf)}")
    for role, env in resolved.items():
        print(f">>> {role.title()} API key env: {env}")

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

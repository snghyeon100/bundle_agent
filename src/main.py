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
        f"results_{conf['dataset']}_baseline_"
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

    path = result_path(conf, timestamp, partial=partial)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path, df, hit_rate, valid_ratio, valid_only_hit_rate, int(valid_mask.sum()) if not df.empty else 0


async def process_samples(client, samples, conf, timestamp, initial_results=None, start_idx=0):
    results = list(initial_results or [])
    concurrency = int(conf.get("max_concurrent", 10))
    semaphore = asyncio.Semaphore(concurrency)
    total = start_idx + len(samples)

    async def run_one(sample, current_idx):
        prompt = generate_prompt(conf["dataset"], sample["input_str"], sample["target_str"])
        if current_idx == start_idx:
            print_first_qa_debug(sample, prompt)

        async with semaphore:
            try:
                response = await client.aio.models.generate_content(
                    model=conf["model"],
                    contents=prompt,
                    config={
                        "temperature": float(conf.get("temperature", 0.0)),
                        "max_output_tokens": int(conf.get("max_output_tokens", 10)),
                    },
                )
                raw_text = response.text or ""
                prediction = parse_model_response(raw_text)
            except Exception as exc:
                raw_text = str(exc)
                prediction = "ERR_EX"

        row = dict(sample)
        row["prediction"] = prediction
        row["raw_response"] = raw_text
        row["hit"] = int(prediction == sample["true_option_char"])
        print(f"[{current_idx + 1}/{total}] True: {sample['true_option_char']} | Pred: {prediction}")
        return row

    for offset in range(0, len(samples), concurrency):
        chunk = samples[offset : offset + concurrency]
        tasks = [run_one(sample, start_idx + offset + idx) for idx, sample in enumerate(chunk)]
        for row in await asyncio.gather(*tasks):
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

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("[Error] GEMINI_API_KEY or GOOGLE_API_KEY is not set in .env.")
        return 1

    client = genai.Client(api_key=api_key)
    results = asyncio.run(
        process_samples(client, samples, conf, timestamp, initial_results=initial_results, start_idx=args.start_idx)
    )

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


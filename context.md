# Bundle Agent Context

## Purpose

`bundle_agent` is a simplified zero-shot bundle completion runner derived from the larger `LLM-ZeroShot` workspace. The repository keeps the baseline evaluation path as the default and adds one optional two-step method, `candidate_reasoning`.

Large local artifacts are intentionally not committed. The `datasets/`, `results/`, `.env`, and Python cache directories are ignored by Git.

## Repository Structure

- `src/dataset.py`: loads BundleConstruction datasets, builds candidate multiple-choice samples, and formats item text.
- `src/main.py`: runs baseline or candidate-reasoning evaluation, saves partial/final CSV files, supports resume, retry, and separate API keys.
- `config.yaml`: controls dataset, model, evaluation seeds, API key env names, retry policy, and method options.
- `requirements.txt`: minimal Python dependencies.
- `datasets/`: local BundleConstruction dataset checkout plus copied embedding caches. This folder is ignored by Git.
- `results/`: local output CSV files. This folder is ignored by Git except `.gitkeep` if present.

## Baseline Method

The baseline method is active when:

```yaml
use_candidate_reasoning: false
```

For each sample, the runner makes one LLM call. The prompt shows the partial bundle and all candidate options, then asks the model to return a single letter only.

Prompt shape:

```text
You are a helpful and honest assistant. The following are multiple choice questions about {task_name}. You should directly answer the question by choosing the letter of the correct option. Only provide the letter of your answer, without any explanation or mentioning the option content.
Question: Given the partial {bundle_name}: {input_str}, which candidate {item_name} should be included into this {bundle_name}?
Options: {target_str}
Your answer should indicate your choice with a single letter (e.g., "A," "B," "C," etc.).
Choice:
```

The output CSV stores `raw_response`, parsed `prediction`, and `hit`.

## Candidate Reasoning Method

The candidate reasoning method is active when:

```yaml
use_candidate_reasoning: true
```

Each sample uses two API calls total:

1. One reasoning call using `reasoning_api_key_env` if configured.
2. One final prediction call using `prediction_api_key_env` if configured.

### Reasoning Call

The reasoning call receives all candidate-completed bundles in one prompt. Each line appends one candidate item to the same input items. The candidate label is used only so the output can be mapped back into `reasoning_A` through `reasoning_J`.

Fashion prompt shape:

```text
You are a bundle construction analyst.
Review the following completed fashion outfits. Each line appends one possible final item to the same input items.
A: {input_str}; {candidate_A_text}
B: {input_str}; {candidate_B_text}
...
For each completed fashion outfit, provide reasoning about how well the items work together. Discuss whether each outfit feels coherent in concept, seasonality, style, color or material harmony, and item-category compatibility.
Write only concise reasoning in English. Use 2-3 sentences for each label. Do not choose an answer.
Return exactly one reasoning paragraph for each label (A, B, ..., J) using this format:
A: reasoning text
B: reasoning text
...
Reasoning:
```

Spotify prompt shape is the same, but refers to playlist continuation and asks about theme, mood, genre, artist or album context, and listening flow.

The raw reasoning response is stored in `reasoning_raw_response`. Parsed per-candidate reasoning is stored in `reasoning_A`, `reasoning_B`, ..., `reasoning_J`.

### Prediction Call

The final prediction prompt shows each candidate option with its reasoning directly attached below the candidate text.

Prompt shape:

```text
Options with reasoning:
A. {candidate_A_text}
Reasoning: {reasoning_A}
B. {candidate_B_text}
Reasoning: {reasoning_B}
...
Your answer should indicate your choice with a single letter (e.g., "A," "B," "C," etc.).
Choice:
```

The final raw response is stored in `raw_response`, the parsed answer in `prediction`, and correctness in `hit`.

## Result Saving and Resume

The runner saves one CSV row per fully completed sample.

- Baseline mode writes a row after the single prediction call completes.
- Candidate reasoning mode writes a row only after both reasoning and final prediction calls complete.
- If candidate reasoning is interrupted during the reasoning or prediction step, the partial row for that sample is not saved.

Partial files use this form:

```text
results/{dataset}/results_{dataset}_{method}_C{num_cans}_T{num_token}_{timestamp}_partial.csv
```

Final files remove `_partial`.

Resume is supported with:

```powershell
python src/main.py --config config.yaml --resume path\to\partial.csv
```

The resume logic uses the number of completed rows in the partial CSV and starts from the next unfinished sample.

## Retry and Stop Policy

Config:

```yaml
max_retries: 5
retry_wait_seconds: 30
```

Retryable errors include `503`, high-demand, overloaded, service unavailable, temporarily unavailable, and similar messages. Retry wait increases linearly by attempt.

Quota or permission errors such as `403`, quota, resource exhausted, permission denied, or billing stop the run immediately. Completed rows remain saved in the partial CSV for resume.

## API Key Configuration

Config:

```yaml
prediction_api_key_env: ""
reasoning_api_key_env: ""
```

If `prediction_api_key_env` is empty, the runner uses `GEMINI_API_KEY` or `GOOGLE_API_KEY`.

If `reasoning_api_key_env` is empty, the runner falls back to the prediction key env and then `GEMINI_API_KEY` or `GOOGLE_API_KEY`.

Example `.env`:

```env
GEMINI_REASONING_API_KEY=...
GEMINI_PREDICTION_API_KEY=...
```

Example `config.yaml`:

```yaml
prediction_api_key_env: GEMINI_PREDICTION_API_KEY
reasoning_api_key_env: GEMINI_REASONING_API_KEY
```

## Reproducibility

The sampled problems are deterministic when these settings and dataset files are unchanged:

```yaml
num_cans
num_token
toy_eval
seed
shuffle_seed
dataset
data_path
```

The exact LLM responses can still vary slightly despite `temperature: 0.0` because provider-side generation behavior is not guaranteed to be bit-for-bit deterministic.

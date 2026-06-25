# Bundle Agent Context

## Purpose

`bundle_agent` is a simplified zero-shot bundle completion runner derived from the larger `LLM-ZeroShot` workspace. The repository keeps the baseline evaluation path as the default, keeps the optional two-step `candidate_reasoning` method, and includes the `sem_agent` method for raw-data evidence retrieval.

Large local artifacts are intentionally not committed. The `datasets/`, `results/`, `.env`, and Python cache directories are ignored by Git.

## Repository Structure

- `src/dataset.py`: loads BundleConstruction datasets, builds candidate multiple-choice samples, and formats item text.
- `src/main.py`: runs baseline, candidate-reasoning, or sem_agent evaluation, saves partial/final CSV files, supports resume, retry, and separate API keys.
- `src/sem_agent/`: Semantic agent prompts, graph evidence retrieval, reasoning, and final prediction orchestration.
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

## Semantic Agent (sem_agent) Method

The `sem_agent` method is active when configured to use the semantic pipeline.

The methodology is separated into a three-step pipeline to isolate data retrieval from linguistic reasoning:

1. **Stage 1 (Data Retrieval):** A code-generation LLM writes sample-specific Python code to traverse allowed raw files (e.g., `item_info.json`, `bi_train.txt`, `ui_full.txt`). Its only task is to fetch relevant item titles and relations for the partial bundle and candidates. It outputs purely factual, structural data in JSON (e.g., `"value": "Extracted N supporting items."`). It does not write narratives or reason about compatibility.
2. **Stage 2 (Reasoning):** A pure-reasoning LLM receives the Stage 1 JSON evidence alongside the candidate metadata. Without writing any code, it interprets the retrieved item relations to construct the bundle context and assess candidate fit. It returns a concise JSON analysis describing why candidates do or do not fit the established aesthetic and functional context.
3. **Decision (Final Predictor):** A final LLM predictor reads the Stage 2 reasoning and selects the best candidate label (A-J).

This pure-retrieval / pure-reasoning separation prevents the code-generation LLM from struggling with string concatenation and natural language generation within Python, while allowing the Stage 2 LLM to focus entirely on semantic assessment.

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
llm_provider: gemini
model: gemini-3.1-flash-lite
prediction_api_key_env: ""
reasoning_api_key_env: ""
```

`llm_provider` selects the backend used by the shared LLM adapter:

- `gemini`: uses the Google GenAI SDK and defaults to `GEMINI_API_KEY` or `GOOGLE_API_KEY`.
- `openai`: uses the OpenAI SDK and defaults to `OPENAI_API_KEY`.

If `prediction_api_key_env` is empty, the runner uses the provider default envs.

If `reasoning_api_key_env` is empty, the runner falls back to the prediction key env and then the provider default envs.

Example `.env`:

```env
GEMINI_REASONING_API_KEY=...
GEMINI_PREDICTION_API_KEY=...
OPENAI_API_KEY=...
```

Gemini example `config.yaml`:

```yaml
llm_provider: gemini
model: gemini-3.1-flash-lite
prediction_api_key_env: GEMINI_PREDICTION_API_KEY
reasoning_api_key_env: GEMINI_REASONING_API_KEY
```

OpenAI example `config.yaml`:

```yaml
llm_provider: openai
model: gpt-4o
openai_send_temperature: false
openai_reasoning_effort: minimal
prediction_api_key_env: OPENAI_API_KEY
reasoning_api_key_env: OPENAI_API_KEY
agent_planning_api_key_env: OPENAI_API_KEY
agent_code_api_key_env: OPENAI_API_KEY
agent_verifier_api_key_env: OPENAI_API_KEY
agent_prediction_api_key_env: OPENAI_API_KEY
```

For OpenAI, `openai_send_temperature` defaults to false in practice and should stay false for GPT-5 models because some GPT-5 models reject the `temperature` parameter. Set it to true only for OpenAI models that support temperature and when temperature control is needed.

For GPT-5 models, `openai_reasoning_effort: minimal` is recommended for this multi-call agent by default. Higher reasoning effort can consume the configured `max_output_tokens` budget before visible text is produced, especially in code-writing and verifier stages.

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

Candidate negatives are filtered against the full test-GT graph before `toy_eval` truncates the list of evaluated pairs. This matches the original `Bundle_zero` candidate generation and prevents the negative-candidate pool from changing with `toy_eval`. The full test GT is used only inside dataset sample construction to exclude known true items; it is never copied into the agent workspace or exposed to an LLM prompt.

The exact LLM responses can still vary slightly despite `temperature: 0.0` because provider-side generation behavior is not guaranteed to be bit-for-bit deterministic.

## Self-Generated LightGCN Features

`utils/train_lightgcn_features.py` trains LightGCN item features directly from local raw relational data when existing CF feature provenance is unclear.

Default behavior:

- UI graph: trains on `ui_full.txt`, treating users as contexts and items as targets.
- BI graph: trains on `bi_train.txt`, treating bundles as contexts and items as targets.
- Initialization: Xavier uniform.
- Objective: BPR ranking loss with uniform unobserved-item negative sampling.
- Propagation: LightGCN layer averaging over `E^(0)..E^(K)`.
- Outputs: `ui_item_embeddings.pt`, `bi_item_embeddings.pt`, and metadata JSON files under `datasets/<dataset>/lightgcn_self/`.

Example BI-only smoke/debug run:

```powershell
python utils\train_lightgcn_features.py --dataset pog_dense --graphs bi --epochs 1 --max-train-edges 10000
```

Example full run:

```powershell
python utils\train_lightgcn_features.py --dataset pog_dense --graphs ui bi --embedding-dim 64 --num-layers 3 --epochs 100
```\n
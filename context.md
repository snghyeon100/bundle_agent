# Bundle Agent Context

## Purpose

`bundle_agent` is a simplified zero-shot bundle completion runner derived from the larger `LLM-ZeroShot` workspace. The repository keeps the baseline evaluation path as the default, keeps the optional two-step `candidate_reasoning` method, and adds an optional four-stage code-writing agent method for raw-data evidence retrieval.

Large local artifacts are intentionally not committed. The `datasets/`, `results/`, `.env`, and Python cache directories are ignored by Git.

## Repository Structure

- `src/dataset.py`: loads BundleConstruction datasets, builds candidate multiple-choice samples, and formats item text.
- `src/main.py`: runs baseline, candidate-reasoning, or four-stage agent evaluation, saves partial/final CSV files, supports resume, retry, and separate API keys.
- `src/agents/`: modular four-stage agent prompts, allowed-workspace preparation, generated-code execution, verifier-guided replanning, and final prediction orchestration.
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

## Four-Stage Code-Writing Agent Method

The four-stage agent method is active when:

```yaml
use_four_stage_agent: true
```

This mode is mutually prioritized over the baseline and candidate reasoning branches. Each sample uses four LLM stages with up to two investigation rounds by default. The method is designed as agentic code-RAG over raw train-safe data: the model is not handed a precomputed context recipe for every sample. Instead, it receives a task description, raw file contracts, and an allowed local workspace, then writes and executes code to build sample-specific evidence.

The current version keeps the stage contracts intentionally compact. Full generated code, raw stage responses, stdout, stderr, and detailed traces are still saved to CSV for debugging, but downstream LLM stages receive compact round summaries rather than the entire raw trajectory.

All four stage prompts receive a short dataset/task semantics block:

- POG/POG-dense: fashion outfit bundle completion. A bundle is a coordinated set of fashion items, typically combining multiple item roles into one outfit. The goal is to choose the candidate that most naturally completes the outfit as a coherent set.
- Spotify/Spotify-Sparse: playlist continuation. A bundle is a music playlist made of songs intended to be listened to together. The goal is to choose the candidate that most naturally continues the playlist as a coherent listening sequence.

The task semantics are intentionally descriptive rather than prescriptive: they explain what a bundle means for the dataset without hard-coding rules such as which feature must matter most.

The current four-stage structure is:

1. Investigation designer: reads the current input/candidate item ids and texts, task semantics, compact previous rounds if any, verifier feedback if any, and the allowed workspace file contracts. Round 1 is a broad sweep over useful sources and signals. It should be creative and deep within the allowed files: transform, join, invert, aggregate, or re-abstract raw data to expose candidate-level differences that are not obvious from one file. Fused analyses such as `BI x item_info`, `UI x metadata`, `embedding x relational evidence`, `BI x IB`, `IB x BI`, or `IU x UI` are examples of useful patterns when they can produce stronger candidate-level signals. Follow-up rounds are narrower and deeper, guided by verifier feedback.
2. Programmatic evidence builder: writes Python code that implements the plan over allowed workspace files such as `item_info.json`, `bi_train.txt`, `ui_full.txt`, and optional text/content embedding cache files. In round 1 it tries to extract diverse candidate-level analysis values, including cross-source and optional multi-hop graph-style signals when useful; in later rounds it prioritizes deeper or more creative follow-up analyses. It should favor derived evidence such as inverted indexes, neighborhoods, category/style abstractions, and aggregated compatibility signals instead of only reporting direct counts. The generated code acts as a measurement tool: it outputs values, provenance, and warnings, but should not decide winners or write `best_labels`.
3. Evidence critic: evaluates the parsed evidence JSON plus a compact execution summary. It checks candidate coverage, failed analyses, all-zero signals, ties, low numeric margins, weak provenance, and contradictory evidence. The verifier interprets numeric values itself rather than trusting generated-code winner claims. If more retrieval is useful, it suggests deeper follow-up work that remains implementable from allowed files, such as new joins, alternate abstraction levels, candidate subsets, cross-source combinations, or transformations that could make weak broad-sweep signals more discriminative.
4. Reliability-aware predictor: receives only compact round summaries. It returns final JSON with `evidence_quality`, `candidate_tradeoff`, `downweighted_evidence`, `decision_rule`, `prediction`, `reasoning`, `confidence`, and `main_sources_used_for_decision`.

Re-planning uses compact previous rounds with parsed plan, execution summary, parsed evidence, and verifier JSON. The planner is instructed not to simply repeat the previous plan; if signals were all-zero, tie-heavy, failed, or too shallow, it should change the abstraction level or investigation idea.

The runner creates one persistent allowed workspace per dataset under `agent_workspaces/{dataset}/`. It copies allowed files into `data/`, writes generated scripts into the workspace, and runs them with the workspace as `cwd`. The LLM sees only relative workspace paths such as `data/item_info.json` and `output/evidence_*.json`, not original dataset paths.

The runner executes generated code with the same Python executable that launched `src/main.py`, stores stdout/stderr, and continues even if optional evidence sources cannot be loaded. If generated code fails, is blocked by the simple guard, or does not produce valid JSON, the code-writing agent receives stdout/stderr and can repair the script up to `agent_code_max_repair_attempts`.

Stage-specific API key config:

```yaml
agent_planning_api_key_env: ""
agent_code_api_key_env: ""
agent_verifier_api_key_env: ""
agent_prediction_api_key_env: ""
```

Empty stage key settings fall back to the prediction key env and then `GEMINI_API_KEY` or `GOOGLE_API_KEY`.

Default allowed files are train-safe:

```yaml
agent_allowed_files:
  - count.json
  - item_info.json
  - bi_train.txt
  - ui_full.txt
  - content_feature.pt
  - description_feature.pt
agent_allow_interaction_embeddings: false
agent_workspace_root: ./agent_workspaces
agent_max_retrieval_rounds: 2
agent_enable_code_guard: true
```

`bi_full.txt` and `item_cf_feature.pt` are excluded by default. `item_cf_feature.pt` is exposed only when `agent_allow_interaction_embeddings: true`, because its train-only provenance should be checked before use.

Important leakage rule: prompts instruct the agent not to read `bi_full.txt`, `bi_test_gt.txt`, validation/test ground-truth files, result CSVs, predictions, hits, or true labels. The code runs inside the allowed workspace with copied files only, and a simple guard blocks common leakage patterns such as parent-directory traversal, absolute Windows drive paths, `bi_full`, test ground-truth names, and result paths.

Result CSVs store planning/code/verifier/prediction traces including `agent_workspace_dir`, `agent_round_count`, `agent_retrieval_rounds_json`, `agent_all_plans_json`, `agent_all_generated_codes_json`, `agent_all_evidence_json`, `agent_all_verifier_json`, `agent_prediction_raw_response`, `agent_reasoning`, `agent_confidence`, `agent_evidence_quality`, `agent_candidate_tradeoff`, `agent_downweighted_evidence`, and `agent_decision_rule`.

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
```

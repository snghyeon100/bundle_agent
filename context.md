# Bundle Agent Context

## Purpose

`bundle_agent` is a simplified zero-shot bundle completion runner derived from the larger `LLM-ZeroShot` workspace. The repository keeps the baseline evaluation path as the default, keeps the optional two-step `candidate_reasoning` method, and includes optional two-stage, three-stage, and four-stage code-writing agent methods for raw-data evidence retrieval.

Large local artifacts are intentionally not committed. The `datasets/`, `results/`, `.env`, and Python cache directories are ignored by Git.

## Repository Structure

- `src/dataset.py`: loads BundleConstruction datasets, builds candidate multiple-choice samples, and formats item text.
- `src/main.py`: runs baseline, candidate-reasoning, two-stage agent, three-stage agent, or four-stage agent evaluation, saves partial/final CSV files, supports resume, retry, and separate API keys.
- `src/two_stage_agent/`: current two-stage code-retrieval prompt and prediction orchestration.
- `src/three_stage_agent/`: exploratory retrieval, evidence-only synthesis, and final prediction prompts and orchestration.
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

## Current Two-Stage Code-Retrieval Agent

The current two-stage agent is active when:

```yaml
use_two_stage_agent: true
```

The current structure is:

1. Code retrieval: an LLM writes sample-specific Python code over allowed train-safe raw files. The generated code computes candidate-level evidence such as direct BI/UI co-occurrence, metadata/category fit, content similarity, and optional UI/BI LightGCN similarity.
2. Final prediction: a second LLM receives the candidate text plus compact retrieved evidence and returns the final prediction, reasoning, and confidence.

The first-stage prompt explicitly forbids choosing a winner. Its intended role is evidence measurement, but the current output contract strongly favors compact candidate-level numeric values:

```text
metadata_fit
bi_signal
ui_signal
embedding_similarity
ui_lightgcn_similarity
bi_lightgcn_similarity
```

This makes the current method closer to a sample-specific feature extractor followed by an LLM predictor than to a broad exploratory research agent. The final predictor can also become anchored on a large co-occurrence or LightGCN value and then use item text as a post-hoc explanation.

### Fixed-Candidate Evaluation

The most reliable two-stage comparison uses these corrected-candidate result files:

```text
results/pog/results_pog_two_stage_agent_C10_T5_20260605_082843.csv
results/pog_dense/results_pog_dense_two_stage_agent_C10_T5_20260605_082934.csv
```

Baselines:

```text
results/pog/results_pog_20260416_142034.csv
results/pog_dense/results_pog_dense_HN_C10_T5_20260430_172343.csv
```

For both datasets, the baseline and two-stage runs have identical input items, candidate items, candidate order, true-option position, and rendered question text for all 250 samples. This removes the candidate-pool mismatch that affected the earlier two-stage comparison.

Summary:

| Dataset | Baseline | Two-stage | Base fail -> success | Base success -> fail | Both fail | Net gain |
|---|---:|---:|---:|---:|---:|---:|
| POG | 82/250 (32.8%) | 87/250 (34.8%) | 33 | 28 | 135 | +5 (+2.0 pp) |
| POG-dense | 84/250 (33.6%) | 136/250 (54.4%) | 74 | 22 | 92 | +52 (+20.8 pp) |

Interpretation:

- POG improves only slightly. It recovers 19.6% of baseline failures but damages 34.1% of baseline-correct samples. The paired improvement is not significant (`p=0.609`).
- POG-dense improves strongly. It recovers 44.6% of baseline failures while damaging 26.2% of baseline-correct samples. The paired improvement is strongly significant (`p=9.44e-8`).
- POG produced four `ERR_EX` predictions; two of those samples were baseline-correct and therefore count as damage. POG-dense produced one `ERR_EX`.
- The POG baseline CSV does not record its model/config, so identical questions are verified, but identical predictor-model conditions cannot be verified.

Detailed artifacts:

```text
analysis/two_stage_20260605_0828_fixed_candidates/summary.md
analysis/two_stage_20260605_0828_fixed_candidates/metrics.json
analysis/two_stage_20260605_0828_fixed_candidates/sample_transitions_pog.csv
analysis/two_stage_20260605_0828_fixed_candidates/sample_transitions_pog_dense.csv
```

### Stage Contribution Diagnosis

Stage 1 does not output a prediction. For diagnosis only, the analysis builds an `evidence-only proxy`: every available non-tied numeric signal is min-max normalized, the normalized values are equally averaged per candidate, and a unique top candidate is treated as the proxy choice. This is not an actual Stage 1 prediction and can underweight a highly reliable signal by averaging it with weak signals.

| Dataset | Evidence created | Proxy-valid samples | Evidence-only proxy | Stage 2 on same samples |
|---|---:|---:|---:|---:|
| POG | 241/250 | 211/250 | 43/211 (20.4%) | 71/211 (33.6%) |
| POG-dense | 247/250 | 222/250 | 101/222 (45.5%) | 124/222 (55.9%) |

On proxy-valid samples:

- POG Stage 2 corrects 40 proxy failures but damages 12 proxy successes. It agrees with the proxy choice only 43.6% of the time.
- POG-dense Stage 2 corrects 28 proxy failures and damages only 5 proxy successes. It agrees with the proxy choice 69.4% of the time.
- POG reasoning that explicitly mentions retrieved evidence has 30.7% accuracy, versus 41.2% when it does not mention evidence.
- POG-dense reasoning that explicitly mentions retrieved evidence has 56.0% accuracy, versus 40.0% when it does not mention evidence.

The evidence-mention comparison is descriptive rather than causal, but it matches the broader pattern: current evidence is useful in dense data and often distracting in sparse POG.

### Signal-Level Findings

Direct BI evidence is the main source of dense-data improvement:

- POG has a discriminative `bi_signal` in only 12 samples. All 12 point to the true candidate and all 12 final predictions are correct, but coverage is too low to drive overall performance.
- POG-dense has a discriminative `bi_signal` in 160 samples. The true candidate is top in 120; final accuracy is 86.7% when the true candidate is top and 0% when it is not.
- POG-dense final correctness overlaps with true-top direct BI in 104/136 correct samples. Its 74 recoveries include 58 samples with a true-top direct BI signal.

Wrong BI LightGCN evidence is a major damage pattern:

- 19/28 POG damage cases have a wrong BI LightGCN unique top.
- 16/22 POG-dense damage cases have a wrong BI LightGCN unique top.

Simple content similarity and same-category fit are weak completion signals. They often retrieve or favor items that resemble existing items rather than items that complement the partial bundle.

### Why Dense Gains Do Not Transfer to POG

POG and POG-dense are separate dataset evaluations; dense evidence is not passed into POG. The available raw graph density differs dramatically:

| Property | POG | POG-dense |
|---|---:|---:|
| Mean input items | 1.54 | 2.02 |
| BI train edges per item | 1.04 | 2.37 |
| UI edges per item | 1.27 | 203.26 |
| UI item coverage | 25.2% | 77.3% |

In the corrected two-stage runs:

- POG `bi_signal` is all-zero/tied in 229/241 evidence rows, and `ui_signal` is all-zero/tied in 235/241.
- POG-dense `bi_signal` is all-zero/tied in 87/247 evidence rows, and `ui_signal` is all-zero/tied in 137/247.

Dense Stage 1 can frequently distinguish candidates using direct historical relationships. POG Stage 1 usually cannot, so its final predictor falls back to text reasoning or weak similarity evidence; recoveries and damages nearly cancel out.

## Next Direction: Exploratory Evidence Retrieval

The next two-stage direction should move away from:

```text
candidate-level numeric feature extraction -> number-anchored prediction
```

toward:

```text
broad retrieval of grounded related examples -> evidence-based semantic reasoning
```

The goal is not to convert existing numbers into natural-language summaries. The goal is to let retrieval search broadly through related items, historical bundles, user interactions, metadata relationships, and other train-safe structures, then give a later LLM enough grounded examples to reason about bundle roles, complementary relationships, patterns, contradictions, and limitations.

### Prompting Philosophy

Do not make methods such as "find similar bundles" or "analyze category patterns" mandatory checklist steps. A fixed checklist can restrict exploration and force irrelevant analyses on every sample. However, a vague instruction such as "find deep evidence" usually collapses back to simple counts and cosine similarities.

The preferred balance is:

- Constrain the research goal, leakage boundary, output size, provenance, and evidence-quality criteria.
- Let the LLM choose retrieval methods based on the current sample.
- Offer possible strategies only as non-exhaustive examples, not required steps.
- Require evidence to be relevant, interpretable, comparative, diverse, balanced, grounded, and honest about sparse or inconclusive data.
- Explicitly state that a high numeric score is not sufficient evidence.
- Treat embeddings and LightGCN as retrieval mechanisms for discovering examples, not as proof that a candidate is correct.

Useful high-level research questions:

```text
What kind of bundle is the partial bundle?
What roles or relationships might reasonably complete it?
Which historical observations help evaluate the candidates?
Which observations support or challenge those relationships?
How reliable and representative are the retrieved observations?
```

Possible strategies may include, but are not limited to, related historical bundles, candidate/item neighborhoods, user-interaction neighborhoods, metadata relationships, category structures, embedding-discovered examples, cross-source joins, and contradictory-example retrieval. The agent should choose only strategies that are useful for the current sample and may devise other train-safe strategies.

### Desired Stage 1 Contract

The exploratory retrieval stage should:

- Receive current input/candidate IDs, titles, categories, task semantics, raw-file contracts, allowed workspace files, retrieval/output budgets, and a leakage policy.
- Not receive true labels, predictions, hits, result files, or instructions to find the best candidate.
- Search broadly but return a compact, diverse evidence pack.
- Return human-readable representative items and bundle compositions with provenance and factual retrieval rationale.
- Include supporting and contradictory observations when available.
- Distinguish direct historical relationships from similarity-discovered relationships.
- State limitations when evidence is sparse, tied, indirect, or unavailable.
- Avoid candidate rankings, preferred candidates, final conclusions, and raw score tables as the primary output.

The evidence pack can include:

```text
input profile and observed roles/categories
representative related historical bundles
candidate/item neighborhood examples
category or composition observations
supporting and contradictory examples
source provenance and retrieval limitations
```

It should retrieve broadly internally but cap the returned context, for example to a few diverse representative bundles, a few examples per candidate, and a small number of contradictory cases.

### Architectural Constraint and Recommended Form

In the current two-stage implementation, the first LLM writes code before that code is executed. It does not see the retrieved results afterward. Therefore, it cannot itself perform a genuinely semantic natural-language synthesis of the observations it retrieves; executable code can only produce deterministic/template-based descriptions.

There are two implementation options:

1. Minimal two-stage experiment: Stage 1 code retrieves a compact human-readable evidence pack; the final predictor reads the pack, synthesizes it internally, and predicts. This is cheaper but risks conclusion-first reasoning because synthesis and prediction happen in one call.
2. Preferred three-stage form: exploratory retrieval -> evidence synthesizer -> final predictor. The synthesizer reads actual retrieved examples and produces a grounded natural-language account of bundle roles, recurring patterns, candidate support, counter-evidence, conflicts, and reliability before any final decision is requested.

The central design principle is:

> Constrain what qualifies as good evidence, not which retrieval method the LLM must use.

## Implemented Three-Stage Exploratory Agent

The preferred three-stage form is now implemented under `src/three_stage_agent/` and integrated into `src/main.py`.

Activate it with:

```yaml
use_three_stage_agent: true
use_candidate_reasoning: false
use_two_stage_agent: false
use_four_stage_agent: false
```

When multiple method flags are accidentally enabled, the current runner gives the three-stage agent highest priority. The implementation still uses the conceptual three-stage form, but Stage 1 now separates broad observation, adaptive deep research planning, and deep code execution:

1. Stage 1A, surface observation code generator: receives the sample, task semantics, allowed train-safe files, and file contracts. It writes executable Python to extract broad factual observations from every allowed source and every candidate. It is explicitly forbidden from choosing, ranking, recommending, or implying a preferred candidate.
2. Stage 1B-Plan, adaptive research architect: receives the sample, allowed files, Stage 1A execution summary, and Stage 1A evidence JSON. Without receiving a menu of retrieval algorithms, it diagnoses unresolved surface gaps, internally compares multiple possible investigations by novelty, expected information gain, candidate discrimination, grounding, robustness, independence, and feasibility, then returns a compact research specification. It does not write code or predict.
3. Stage 1B-Code, deep observation code generator: receives the fixed research specification and writes executable Python that implements it. It may not replace a planned investigation with an easier direct lookup or one-hop count. Code repair must preserve the research specification and change only implementation defects.
4. Stage 2, evidence synthesizer: receives a combined Stage 1 evidence pack containing the deep research specification plus the surface and executed deep observation outputs. It treats the plan as intent rather than evidence, checks plan fulfillment, and interprets bundle relationships, candidate support, counter-evidence, conflicts, reliability, and limitations without predicting or ranking candidates.
5. Stage 3, final predictor: receives the original sample and the evidence synthesis, then chooses one candidate while respecting evidence quality and downweighted evidence.

Stage 1A is intentionally broad and mostly tabular. It is expected to cover all candidate labels, not only a representative candidate. For POG-style data, the current prompt directly asks for sample-specific extraction from `item_info.json`, `bi_train.txt`, `ui_full.txt`, `content_feature.pt`, and `description_feature.pt`: metadata/group relationships, candidate-only row counts, candidate-with-input row counts, and input-candidate feature similarities. These are observations, not judgments.

The file-format contract explicitly treats the first value of every BI/UI row as a typed context ID, never an item ID: `context_id = values[0]`, `item_ids = values[1:]`. Bundle/user IDs and item IDs remain different entities even when their integer values match. Generated code is instructed to perform every item lookup, count, co-occurrence, category mapping, join, neighborhood operation, and graph traversal only over `item_ids`.

Stage 1B planning is intentionally method-agnostic. It is not given a checklist or menu of retrieval recipes. Instead, it must determine what Stage 1A cannot establish, generate and compare possible research designs internally, and select a small non-redundant portfolio using evidence-quality criteria. The separate code agent then implements that specification and reports concrete facts, examples, counts, retrieved IDs, computed values, provenance, plan fulfillment, and limitations.

Stage-specific configuration:

```yaml
three_stage_code_api_key_env: "GEMINI_API_KEY_2"
three_stage_deep_code_api_key_env: "GEMINI_API_KEY_2"
three_stage_deep_planning_model: ""  # empty = global model; may be set to a stronger compatible model
three_stage_deep_code_model: ""      # empty = global model
three_stage_synthesis_api_key_env: "GEMINI_API_KEY_3"
three_stage_prediction_api_key_env: "GEMINI_API_KEY_4"
three_stage_code_max_output_tokens: 3600
three_stage_deep_planning_max_output_tokens: 1800
three_stage_deep_code_max_output_tokens: 3600
three_stage_synthesis_max_output_tokens: 3600
three_stage_synthesis_max_repair_attempts: 1
three_stage_prediction_max_output_tokens: 900
three_stage_code_max_repair_attempts: 1
```

The result CSV stores the full three-stage trace, including surface generated code, surface execution summary, surface evidence JSON, deep planning prompt/raw/parsed JSON, deep generated code, deep execution summary, deep evidence JSON, combined evidence JSON, synthesis repair traces and validation issues, raw and parsed synthesis, raw and parsed prediction, final reasoning/confidence, observations used, and evidence that was downweighted or ignored. Key columns use the `three_stage_` prefix. Deep evidence validation also requires every completed or partial investigation to emit separate observations for every exact candidate scope (`candidate:A`, `candidate:B`, and so on); item-id scopes and one aggregate candidate-value blob are rejected and sent to code repair. Missing optional `examples` fields are normalized to empty lists and do not invalidate otherwise usable evidence. If validation issues remain after the repair budget is exhausted, the raw deep output is retained for debugging but excluded from the evidence passed to synthesis.

The three-stage agent reuses the existing allowed workspace and generated-code guard from `src/agents/workspace.py`. It does not expose true labels, test ground truth, result files, predictions, or hits in any stage prompt.

Stage 1 can also be run standalone for debugging:

```powershell
.\.venv\Scripts\python.exe src\run_three_stage_stage1.py --config config.yaml --sample_idx 0 --limit 1
```

The standalone runner normally makes three LLM calls per sample: Stage 1A surface code, Stage 1B research planning, and Stage 1B deep code. It writes per-sample JSON and a summary JSONL under `analysis/stage1` by default. Use `--skip_deep` to run only Stage 1A surface observation.

## 2026-06-20 Three-Stage Agent Notes

The main design discussion on 2026-06-20 centered on how to use an LLM when only raw source files are provided. The target behavior is:

```text
source files -> LLM-written retrieval/observation code -> factual observations -> synthesis -> final judgment
```

The key decision was to make Stage 1 an evidence-observation stage rather than a reasoning or prediction stage. Stage 1 should not summarize, rank, or decide. It should extract factual observations from the raw sources and leave interpretation to Stage 2.

Stage 1A went through several prompt iterations:

- Early versions asked for too many interpretive outputs, such as support/counter/reliability/missing role, which belonged in Stage 2.
- The schema was reduced to `source_observations`, `source_attempts`, and `warnings`.
- The prompt was changed to require all allowed sources, not just one source.
- The prompt was changed to require candidate coverage for every candidate label A-J, not only candidate A.
- A weaker prompt asking the LLM to infer meaningful source use helped somewhat but was unstable: it sometimes extracted candidate observations from `item_info.json` and `bi_train.txt`, but often stopped at file existence, row count, or tensor shape for harder sources.
- The final surface prompt became more direct. It tells the LLM exactly what data to extract from each source while still forbidding prediction.

Observed Stage 1A behavior during `--sample_idx 0 --limit 1` experiments:

- `stage1_summary_pog_20260620_170558.jsonl`: valid JSON, but only shallow observations; mostly `item_info.json`, one `bi_train.txt` check, and tensor shape.
- `stage1_summary_pog_20260620_171015.jsonl`: candidate A-J coverage improved for `item_info.json`, but other sources remained shallow.
- `stage1_summary_pog_20260620_171712.jsonl`: `bi_train.txt` began producing candidate-level row observations, but `ui_full.txt` and feature tensors still often stopped at shallow checks.
- `stage1_summary_pog_20260620_172518.jsonl`: direct source-specific extraction produced useful observations for all candidates across `item_info.json`, `bi_train.txt`, `ui_full.txt`, `content_feature.pt`, and `description_feature.pt`.
- `stage1_summary_pog_20260620_172750.jsonl`: `bi_train.txt` and `ui_full.txt` separated candidate-only row counts from candidate-with-input row counts, which is important because candidate-only frequency should not be mislabeled as co-occurrence.

Important Stage 1A lesson: prompt-only autonomy was not enough for stable source extraction. If the goal is reliable broad coverage, direct source-specific extraction instructions work better. If the goal is observing how the LLM chooses deeper investigations, that should be separated into a different agent rather than mixed into the broad surface extraction prompt.

The resulting Stage 1 split is:

```text
Stage 1A Surface Observation:
  Broad, explicit, source-by-source extraction.
  Covers all sources and all candidates.
  Produces factual observations only.

Stage 1B Deep Observation:
  Receives Stage 1A evidence.
  Designs additional source-grounded investigations.
  Avoids explicit hard-coded recipes at first.
  Goes beyond direct lookup, diagnostics, tensor shape, and one-hop counts.
  Produces deep_investigations with question, why_relevant, sources_used, method_summary, observations, limitations.
```

The intended Stage 1B prompt style is deliberately non-recipe-based:

```text
Design and execute additional source-grounded investigations that may reveal non-obvious factual context for the current completion task.
Go beyond direct lookup, file diagnostics, tensor shape, and one-hop item counts.
Use the source structure to discover additional context, comparisons, neighborhoods, repeated patterns, or indirect relationships that are not already present in the surface observations.
Do not choose, rank, recommend, or imply a preferred candidate.
```

This keeps the experiment focused on what the LLM decides to investigate deeply, while Stage 1A guarantees a stable baseline evidence table.

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
```

## 2026-06-22 New Methodology Direction: Progressive Signal Discovery

The next method should keep the source-grounded, code-generating multi-agent pipeline:

```text
Partial bundle + candidates
        -> Source Schema Reader
        -> Signal Planner
        -> Python Signal Code Generator
        -> Code Executor
        -> Signal Diagnosis
        -> Adaptive Re-planning if needed
        -> Evidence JSON
        -> Decision Agent
        -> Final candidate
```

However, the first Signal Planner should not try to predict which signal will be important from item text and categories alone. Before execution, it has no empirical basis for knowing whether direct bundle relations, user interactions, metadata, embeddings, or another derived structure will be informative for the current case. Asking it to choose important signals too early encourages semantic guessing and easy fixed recipes.

The new direction is therefore:

```text
broad source-grounded observation
        -> empirical signal diagnosis
        -> open-ended evidence-gap-driven investigation
        -> verified evidence synthesis and decision
```

### Round 1: Broad Surface Signal Observation

The first planner is a coverage planner rather than an importance planner. It uses the current partial bundle, all candidates, and a Source Capability Manifest to ensure broad factual observation across the allowed sources. It must not rank candidates, choose a winner, or claim that a signal is important before seeing the executed results.

Round 1 should provide stable candidate coverage and basic source observations such as metadata relationships, candidate and input-candidate occurrence facts, source coverage and missingness, and available representation-level comparisons. These are surface observations, not a final scoring model. The exact input format and source contracts should follow the repository's real dataset representation rather than the illustrative input schema in the external implementation specification.

The Source Schema Reader should describe source affordances, not prescribe a scoring recipe. A machine-readable Source Capability Manifest may include:

```json
{
  "source": "bi_train.txt",
  "entities": ["bundle", "item"],
  "relations": ["bundle contains item"],
  "supported_operations": ["filter", "invert", "join", "aggregate", "retrieve examples"],
  "constraints": ["the first value is a typed bundle id", "train data only"]
}
```

### Signal Diagnosis After Execution

Signal importance should be assessed only after code execution. The diagnosis stage examines the actual observations for:

- candidate and source coverage;
- missing, all-zero, or tied outputs;
- candidate discrimination and numeric margins;
- direct versus indirect evidence;
- redundancy between signals;
- single-source or popularity dominance;
- conflicts between sources or views;
- provenance and plan-fulfillment problems;
- possible confusion between similarity, compatibility, and redundancy.

Diagnosis should report facts, reliable observations, and unresolved evidence gaps. It should not prescribe a retrieval method. In particular, it should not tell the next planner to find similar bundles, retrieve similar candidate items, apply category smoothing, or execute a specific multi-hop recipe.

Recommended diagnosis contract:

```json
{
  "reliable_observations": [],
  "observed_failures": [],
  "unresolved_questions": [],
  "evidence_gaps": [],
  "conflicts": [],
  "signals_to_downweight": []
}
```

### Round 2: Open-Ended Deep Signal Discovery

Round 2 is driven by the executed Round 1 evidence and diagnosis. Its purpose is not to generate something merely labeled a "deep signal." It should design executable investigations that can resolve concrete evidence gaps or distinguish competing explanations for the current observations.

Possible outcomes of this process could include discovering related historical bundles, examining historical contexts of items related to a candidate, deriving new graph paths, joining sources, comparing distributions, or devising other train-safe analyses. These are illustrative emergent outcomes only. They should not be included in the Deep Planner prompt as a required checklist or menu of methods. The LLM should invent the investigation based on the current case, observed evidence, source entities and relations, and execution constraints.

The central prompting principle is:

> Constrain the quality, grounding, and boundaries of the investigation, not the investigation method.

The Deep Planner should receive:

- the partial bundle and candidates;
- the Source Capability Manifest and file contracts;
- executed Round 1 factual observations;
- reliable and downweighted signals;
- conflicts, failed signals, and unresolved evidence gaps;
- summaries of previous investigations;
- candidate coverage, leakage, runtime, and output constraints.

It should return a research specification rather than code or a prediction:

```json
{
  "research_objective": "...",
  "investigations": [
    {
      "question": "...",
      "competing_explanations": ["...", "..."],
      "why_needed": "...",
      "sources_used": ["..."],
      "derivation_path": ["..."],
      "method_design": "LLM-designed executable investigation",
      "new_information": "...",
      "possible_outcomes": {
        "positive": "...",
        "negative": "..."
      },
      "distinction_from_surface": "...",
      "expected_candidate_scope": "all candidates",
      "failure_condition": "..."
    }
  ],
  "stop_condition": "..."
}
```

Requiring competing explanations and outcome-dependent interpretation prevents the planner from creating new numbers that cannot change the evidence assessment. If positive and negative results would lead to the same interpretation, the proposed investigation has low information value and should be rejected.

### Investigation Proposal Tournament

A vague instruction to produce deep evidence often collapses to file diagnostics, direct counts, or cosine similarity. To improve autonomous discovery without hard-coding recipes, the Deep Planner should use an investigation proposal tournament:

1. Frame each unresolved evidence gap as a factual question with competing explanations.
2. Internally generate multiple distinct executable investigation proposals.
3. Compare them by novelty, expected information gain, candidate discrimination, grounding, robustness, independence, coverage, and feasibility.
4. Select a small non-redundant portfolio of investigations.
5. Return only the selected research specification.

The planner may be given an investigation grammar consisting of source entities, relations, and generic transformations such as filter, invert, join, expand, aggregate, compare, retrieve examples, and test robustness. These are compositional primitives rather than completed retrieval recipes.

When practical, selected proposals should first run small pilot probes. Probe outputs can report execution success, candidate coverage, nonzero coverage, distinct-value counts, representative observations, and dominance warnings. The planner can then retain, revise, or discard proposals based on real data behavior before generating the full deep investigation code. This changes planning from one-shot speculation into observation-guided research.

### Deepness and Novelty Validation

An investigation should not be accepted as deep merely because it uses a new signal name. The validator or critic should reject investigations limited to:

- file existence checks, row counts, or tensor shapes;
- a repetition of candidate frequency or direct one-hop counts already present in Round 1;
- a single unchanged cosine similarity calculation;
- renaming, renormalizing, or reweighting an existing surface observation;
- aggregate values without candidate-scoped provenance or representative examples;
- a method whose possible outcomes do not change the interpretation of an evidence gap.

This is an anti-redundancy constraint, not a requirement to use a particular graph depth, source combination, or retrieval algorithm. A simple investigation may still be accepted when it produces genuinely new, grounded, discriminative evidence for an unresolved question.

### Planning, Code Generation, Repair, and Re-planning

The research planner and code generator should remain separate:

- The Deep Planner decides what new factual question to investigate and writes the fixed research specification.
- The Deep Code Generator implements that specification without replacing it with an easier lookup.
- Code repair preserves the research specification and changes only implementation defects.
- Research re-planning occurs only when the executed investigation is uninformative, redundant, infeasible, or leaves the evidence gap unresolved.

Execution failure and research failure must therefore be treated differently:

```text
execution error -> repair the implementation
weak or redundant evidence -> redesign the investigation
inconclusive but valid evidence -> record the limitation and either deepen or stop
```

### Evidence JSON and Final Decision

The final Evidence JSON should contain verified candidate-scoped observations, provenance, representative examples when available, diagnosis, conflicts, reliability, and limitations. It should distinguish direct historical evidence from evidence derived through similarity or other indirect transformations. It should not force every observation into one scalar `final_score` before the Decision Agent sees it.

The Decision Agent receives the compact verified evidence rather than raw code, raw execution logs, or unvalidated planner claims. It is the only stage allowed to choose the final candidate. It must account for evidence quality, sparsity, source conflict, downweighted signals, and the difference between compatibility and mere similarity.

### Working Method Name

The current working name is:

```text
Progressive Signal Discovery
```

The intended research framing is a training-free, source-grounded, code-generating multi-agent method that first establishes broad empirical coverage and then lets an LLM autonomously design deeper investigations from observed evidence gaps. Its novelty should come from evidence-driven signal discovery and adaptive executable investigation, not from a fixed library of recommendation scorers.

### 2026-06-22 Initial Code Draft

The first Progressive Signal Discovery code draft is implemented under `src/progressive_signal_agent/`. The earlier `src/agents/`, `src/three_stage_agent/`, and `src/two_stage_agent/` implementations and the three-stage standalone runner were removed. `src/main.py` now runs only the new method, and `config.yaml` contains only options consumed by this pipeline.

The canonical case exposed to the agent is ID-centered:

```json
{
  "case_id": "bundle_418",
  "dataset": "pog",
  "bundle_id": 418,
  "partial_item_ids": [154, 932],
  "candidates": [
    {"label": "A", "item_id": 281},
    {"label": "B", "item_id": 719}
  ]
}
```

The real `bundle_id` is retained because generated investigations may need typed bundle context from allowed train sources. It must never be treated as an item ID. Ground truth, true-option fields, test-GT paths, predictions, hits, and result files are filtered out of the agent view.

The initial implementation performs one broad planning/code/execution round, diagnoses the executed evidence, and runs up to `psd_max_deep_rounds` open-ended planning/code/execution rounds when diagnosis returns `NEEDS_DEEPENING`. Each deep planner prompt requires an internal proposal tournament but does not provide completed retrieval recipes. Only evidence that passes execution and candidate-scope validation is included in the final Evidence JSON.

## 2026-06-23 New Methodology Draft: Simple Generate-Evaluate-Decide

This is an implemented simpler methodology to compare with Progressive Signal Discovery. The working name is `Simple Generate-Evaluate-Decide`; it is not yet a final research name. The method keeps source-grounded code generation but removes the separate broad planner, deep investigation planner, and multi-round progressive discovery structure.

The proposed flow is:

```text
ID-centered case
        -> Signal Code Generator
        -> Python Executor
        -> Minimal candidate-scoped Evidence JSON
        -> Signal Sufficiency Evaluator
             -> REFINE: return diagnostic feedback to the Code Generator
             -> SUFFICIENT: continue to the Decision Agent
             -> INCONCLUSIVE: stop refinement and continue with explicit low-quality evidence
        -> Decision Agent
        -> Prediction only
```

The Signal Sufficiency Evaluator does not rewrite code. It diagnoses the executed evidence and returns actionable requirements to the Signal Code Generator. Keeping generation and evaluation separate makes the reason for each refinement traceable and reduces self-confirming code changes by the evaluator.

### Train-Safe Sources

The intended source set is:

```text
bi_train.txt
ui_full.txt
count.json
item_info.json
content_feature.pt
description_feature.pt
item_cf_feature.pt
{dataset}_LightGCN_bi_feature.pt
```

`item_cf_feature.pt` is an item embedding trained from `ui_full.txt`. `{dataset}_LightGCN_bi_feature.pt` is an item embedding trained from `bi_train.txt`. This provenance is part of the source contract supplied by the user; generated code must still validate the serialized object type and item-index alignment before using either representation. No test ground truth, true label, hit, previous prediction, result file, `bi_full.txt`, or test-GT path may be exposed to any agent.

BI/UI parsing retains the existing typed-ID rule:

```text
context_id = values[0]
item_ids = values[1:]
```

Bundle IDs and user IDs are context IDs, not item IDs, even when their integer values happen to match an item ID.

### Common Case Contract

The evidence-discovery path receives an ID-centered case:

```json
{
  "case_id": "bundle_418",
  "dataset": "pog",
  "task_semantics": "fashion bundle completion",
  "bundle_id": 418,
  "partial_item_ids": [154, 932],
  "candidates": [
    {"label": "A", "item_id": 281},
    {"label": "B", "item_id": 719}
  ]
}
```

The Signal Code Generator and Signal Sufficiency Evaluator do not receive item text directly. Generated code may retrieve text and metadata from the allowed `item_info.json` source with explicit provenance. The final Decision Agent receives deterministic item text and metadata resolved from `item_info.json`, because semantic bundle compatibility cannot be judged reliably from IDs and numeric signals alone.

### Stage 1: Signal Code Generator

The Signal Code Generator receives:

- the ID-centered case;
- task semantics;
- the Source Capability Manifest and real file contracts;
- allowed relative source paths;
- output-path, timeout, leakage, and candidate-coverage constraints;
- on a refinement round only, the previous code, execution summary, evidence, and evaluator feedback.

The agent returns executable Python code only. It must not choose, rank, recommend, or imply a preferred candidate. The code writes a compact Evidence JSON centered on observations for every exact candidate label.

The minimal evidence contract is:

```json
{
  "signals": [
    {
      "signal_name": "bundle_context_relation",
      "description": "What factual relationship or quantity this signal measures",
      "sources": ["bi_train.txt", "item_info.json"],
      "candidate_observations": {
        "A": {
          "value": 0.42,
          "evidence": ["compact representative fact or example"]
        },
        "B": {
          "value": 0.18,
          "evidence": ["compact representative fact or example"]
        }
      }
    }
  ]
}
```

The required evidence elements are deliberately limited to:

- `signal_name`: a stable signal identifier;
- `description`: what was actually measured;
- `sources`: factual provenance;
- `candidate_observations`: values and compact representative evidence for every candidate A-J.

Candidate observations are the central output. `value` may be `null` when an observation is factual or example-based rather than scalar. Representative evidence should be capped to a small number of items per candidate. Coverage, missing labels, execution success, and schema compliance should be computed by the runner rather than redundantly described by generated code. The evaluator can inspect the generated code, so the evidence does not need a verbose derivation narrative.

### System Component: Python Executor and Deterministic Validation

The executor is not an LLM agent. It runs the generated code inside the restricted allowed workspace and records success, runtime, stdout, stderr, output existence, and parse results.

Before the sufficiency LLM is called, deterministic validation checks:

- successful execution and valid JSON;
- the required minimal evidence schema;
- exact candidate-label coverage for all candidates;
- allowed-source and leakage-policy compliance;
- usable provenance fields.

Execution or schema errors follow a code-repair path. They are not semantic sufficiency statuses.

### Stage 2: Signal Sufficiency Evaluator

The evaluator receives:

- the ID-centered case and Source Capability Manifest;
- generated code;
- compact execution summary;
- validated Evidence JSON;
- current iteration and remaining refinement budget;
- previous evaluation summaries when applicable.

It evaluates candidate/source coverage, discrimination, ties and all-zero values, relevance to bundle compatibility, direct versus indirect grounding, redundancy, popularity or single-source dominance, conflicts, missingness, and confusion between similarity, compatibility, and redundancy.

Its structured output is:

```json
{
  "status": "REFINE",
  "evidence_quality": "LOW",
  "reliable_signals": [],
  "weak_or_failed_signals": [],
  "coverage_problems": [],
  "redundancy_problems": [],
  "conflicts": [],
  "evidence_gaps": [],
  "required_improvements": [],
  "expected_new_information": "",
  "reason": ""
}
```

The status space is intentionally limited to:

- `SUFFICIENT`: grounded evidence is adequate to pass to the Decision Agent;
- `REFINE`: evidence is insufficient, but a concrete and feasible additional investigation could provide new decision-relevant information;
- `INCONCLUSIVE`: evidence remains insufficient and another allowed refinement is unlikely to resolve the gap, or the refinement budget has been exhausted.

`REFINE` must not mean merely that the evidence is weak. The evaluator must identify a resolvable evidence gap, state what new information is required, and explain how the possible result could change the evidence assessment. Its `required_improvements` describe information requirements, not replacement Python code or a fixed retrieval recipe.

Conflicts, low evidence quality, and missing coverage remain fields rather than additional statuses. Execution errors remain outside this status space.

### Refinement Loop

On `REFINE`, the original Signal Code Generator receives the previous code, executed evidence, execution summary, and evaluator feedback. It may revise or replace the code while remaining within the same source and leakage constraints.

The configuration now includes:

```yaml
simple_signal_max_refinement_rounds: 1
```

This value counts additional Generate -> Execute -> Evaluate rounds after the initial round. The default therefore allows at most two successful signal-code executions per sample: one initial execution and one refinement execution. If the budget is exhausted while evidence remains insufficient, the method proceeds to the Decision Agent with the unresolved gaps and low-quality or inconclusive status explicitly attached.

### Stage 3: Decision Agent

The Decision Agent receives:

- task semantics and the case identifiers;
- partial items and candidates with deterministic text and metadata resolved from `item_info.json`;
- the final validated Evidence JSON;
- the final sufficiency evaluation;
- a compact refinement-history summary if a refinement occurred.

The Decision Agent is the only component allowed to select a candidate. It should account for evidence quality, unresolved conflicts, sparse data, downweighted signals, and semantic item compatibility. Evidence discovery remains ID-centered, while final decision-making uses IDs, item text, and verified evidence.

The output contract contains only the prediction:

```json
{
  "prediction": "A"
}
```

Reasoning, confidence, candidate trade-offs, and post-hoc explanations are intentionally omitted. The evaluation target is candidate accuracy, and the saved upstream evidence and evaluator trace already provide the method-level diagnostic record. The runner may still store the raw prediction response, parsed label, and system-computed hit for reproducibility and error analysis.

### 2026-06-23 Initial Implementation

The first implementation is under `src/simple_signal_agent/`:

```text
src/simple_signal_agent/prompts.py
src/simple_signal_agent/pipeline.py
src/simple_signal_agent/__init__.py
```

`src/main.py` now selects the runner through:

```yaml
method: simple_generate_evaluate_decide  # progressive_signal_discovery | simple_generate_evaluate_decide
```

The simple method uses separate code, evaluator, and prediction clients and token budgets. Its workspace, timeout, guard, repair, evidence-size, refinement, current-bundle policy, and allowed-file settings use the `simple_signal_` configuration prefix. The default configuration exposes both interaction-derived item embeddings after resolving `{dataset}` in the BI embedding filename.

The generated-code runner reuses the existing restricted workspace and code guard, extended to accept a method-specific configuration prefix. Deterministic validation enforces the minimal evidence schema, exact A-J coverage within every signal, exact available-source provenance, at most three evidence entries per candidate per signal, the configured evidence-size limit, and the absence of decision fields such as prediction, winner, ranking, recommendation, or final score.

Each successful round is evaluated once. `REFINE` is accepted only when the evaluator provides a non-empty evidence gap, required improvement, and expected new information while refinement budget remains. Otherwise it is normalized to `INCONCLUSIVE`. A refinement output is a complete replacement evidence pack. If a refinement execution fails after code repair, the last valid evidence is retained and passed to decision with an inconclusive evaluation.

Result rows use the `simple_signal_` prefix and store the workspace files, manifest, ID-only case, deterministic decision case, complete round trace, final evidence, final evaluation, final status/quality, and raw/parsed prediction response. The public `prediction`, `raw_response`, and `hit` columns remain compatible with existing evaluation and resume behavior. Result filenames include the selected method name.

Offline tests in `tests/test_simple_signal_agent.py` cover minimal evidence validation, refinement-budget normalization, and an end-to-end fake-LLM flow through code generation, execution, evaluation, and prediction. The existing Progressive Signal Discovery smoke test remains passing after the shared workspace changes.

# Bundle Agent Context

## Purpose

`bundle_agent` is a zero-shot bundle completion workspace focused on the `src/code/` method:

```text
Code generation -> Evidence execution -> Prediction
```

For each bundle-completion instance, an LLM first generates executable Python code that retrieves source-grounded evidence from local dataset files. The generated evidence is then attached next to partial items and candidate items, and a prediction LLM chooses the final candidate label.

Large local artifacts are intentionally not committed. The `datasets/`, `results/`, `results_baseline/`, `.env`, workspace caches, and Python cache directories are ignored by Git.

## Repository Structure

- `src/dataset.py`: loads BundleConstruction datasets, builds multiple-choice samples, and formats item text.
- `src/main.py`: main entry point for the `src/code/` method.
- `src/code/`: code-generation evidence pipeline, prompts, shared helpers, and workspace wrapper.
- `src/main_baseline.py`: text-only baseline entry point for separate comparison runs.
- `config_code.yaml`: configuration for the code method.
- `config_baseline.yaml`: configuration for the text-only baseline.
- `tests/stage_1_code_generation/run.py`: isolated Stage 1 code-generation test runner.
- `tests/stage_2_prediction/run.py`: isolated Stage 2 prediction test runner using a saved Stage 1 directory.
- `datasets/`: local BundleConstruction datasets and feature files.
- `results/`: full code-method outputs.
- `analysis/`: stage-specific test outputs.

## Code Method

The active methodology is implemented under `src/code/`.

The pipeline has two conceptual stages:

1. **Stage 1: Code Generation and Evidence Execution**
   - The LLM receives the current bundle problem: partial item IDs/text, candidate item IDs/text, task description, and a source manifest.
   - It generates executable Python code.
   - The code runs in a prepared workspace with allowed local data files copied under `data/`.
   - The code writes an `evidence.json` file with evidence for every partial item and every candidate item.
   - There is no repair loop. If generated code fails or writes invalid evidence, the run prints failure and falls back to sparse placeholder evidence.

2. **Stage 2: Prediction**
   - The prediction LLM receives the original partial/candidate item text with evidence lines attached directly under each item block.
   - It returns only one label, such as `A` through `J`.

The intended separation is:

```text
Stage 1: retrieve compact factual evidence using executable code
Stage 2: choose the final candidate using item text plus retrieved evidence
```

Stage 1 must not choose, rank, score, recommend, or reveal a final prediction.

## Stage 1 Evidence Schema

Generated code must write JSON using the `code_evidence_v1` shape:

```json
{
  "schema_version": "code_evidence_v1",
  "strategies": [
    {
      "name": "strategy_name",
      "relation_signal": "item -> source relation -> retrieved context",
      "data_sources": ["bi_train.txt", "item_info.json"],
      "description": "short strategy description"
    }
  ],
  "partial_evidence": {
    "partial_123": {
      "item_id": 123,
      "evidence": ["short source-grounded evidence string"]
    }
  },
  "candidate_evidence": {
    "A": {
      "item_id": 456,
      "evidence": ["short source-grounded evidence string"]
    }
  },
  "policy_trace": {
    "implemented_strategies": ["strategy name -> concrete relation path implemented"],
    "skipped_strategies": ["strategy/view -> source or sparsity reason"],
    "notes": ["short implementation or fallback note"]
  }
}
```

The prompt gives one example strategy style:

```text
IB x BI co-bundle context:
item -> train bundles -> co-occurring items/context
```

This is only an example. The LLM is asked to inspect the source manifest and design at least two additional sample-adaptive strategies, for at least three strategies total.

## Available Source Signals

Allowed source files are configured in `config_code.yaml`.

Typical sources:

- `count.json`: dataset counts.
- `item_info.json`: item text metadata. Category fields are redacted from the code-method source manifest and should not be used as evidence.
- `bi_train.txt`: bundle-item train relations.
- `ui_full.txt`: user-item interactions.
- `content_feature.pt`: item content/image/audio feature tensor.
- `description_feature.pt`: item text feature tensor.
- `item_cf_feature.pt`: item collaborative feature tensor.
- `{dataset}_LightGCN_bi_feature.pt`: BI LightGCN item feature tensor.

The source manifest is treated as a set of typed relation contracts, not as final answer hints.

## Prediction Prompt Shape

The prediction prompt uses block formatting.

For `pog` and `pog_dense`:

```text
You are a helpful and honest assistant. The following are multiple choice questions about bundle construction.
You should directly answer the question by choosing the letter of the correct option. Only provide the letter of your answer, without any explanation or mentioning the option content.
Question: Given the partial fashion outfit below, which candidate fashion item should be included into this fashion outfit?
Partial fashion outfit:
1. {partial item text}
Evidence: {partial evidence lines}

Options:
A. {candidate item text}
Evidence: {candidate evidence lines}
...
Choice:
```

For `spotify` and `spotify_sparse`, task names change to playlist continuation, music playlist, and song.

Spotify item text is formatted as:

```text
track_name - artist_name - album_name
```

Fashion item text uses the item title.

## Output Layout

`src/main.py` saves artifacts under:

```text
results/{dataset}/{timestamp}/bundle_{bundle_id}/stage1_code_generation/
  input.txt
  output.txt
  code.py
  evidence.json
  execution_summary.json

results/{dataset}/{timestamp}/bundle_{bundle_id}/stage2_prediction/
  input.txt
  output.txt
  prediction.json
  decision_case.json
```

The run-level CSV is saved as:

```text
results/{dataset}/{timestamp}/results.csv
```

During an interrupted run, partial rows are saved as:

```text
results/{dataset}/{timestamp}/results_partial.csv
```

Resume is supported with:

```powershell
python src\main.py --config config_code.yaml --resume results\{dataset}\{timestamp}\results_partial.csv
```

## Stage Test Outputs

Stage-specific tests save outputs under `analysis/`.

Stage 1:

```text
analysis/stage_1_code_generation/{dataset}/{code_generation_model}/bundle_{bundle_id}_{timestamp}/
  input.txt
  output.txt
  code.py
  evidence.json
  execution_summary.json
```

Stage 2:

```text
analysis/stage_2_prediction/{dataset}/{prediction_model}/bundle_{bundle_id}_{timestamp}/
  input.txt
  output.txt
  prediction.json
  decision_case.json
  evidence.json
```

Stage 2 can reuse a saved Stage 1 directory:

```powershell
python tests\stage_2_prediction\run.py --stage1_dir analysis\stage_1_code_generation\pog\gpt-4.1-mini\bundle_722_20260706_105433
```

## Retry and Stop Policy

Config:

```yaml
max_retries: 5
retry_wait_seconds: 30
```

Retryable errors include overloaded/service unavailable messages, temporary provider failures, connection errors, connection resets, and timeouts. Retry wait increases linearly by attempt.

Quota or permission errors such as `403`, quota, resource exhausted, permission denied, or billing stop the run immediately. Completed rows remain saved in `results_partial.csv` for resume.

## API Key Configuration

Code method uses separate clients for Stage 1 and Stage 2:

```yaml
code_generation_provider: openai
code_generation_model: gpt-4.1-mini
code_generation_api_key_env: "DMLAB_KEY"

code_prediction_provider: openai
code_prediction_model: gpt-4.1-mini
code_prediction_api_key_env: "DMLAB_KEY"
```

For OpenAI models, `openai_reasoning_effort` is sent only to models that support reasoning parameters. It should be left empty for models such as `gpt-4.1-mini`.

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

Candidate negatives are filtered against the full test-GT graph before `toy_eval` truncates the evaluated pairs. This matches the original candidate generation protocol and prevents the negative-candidate pool from changing with `toy_eval`.

The exact LLM outputs can still vary slightly despite `temperature: 0.0` because provider-side generation is not guaranteed to be bit-for-bit deterministic.

## Instance Diagnostics and Bundle Clustering

A useful analysis direction is to treat each bundle-completion instance as having its own evidence topology across BI relations, UI relations, and multimodal feature spaces.

Instead of only reporting overall accuracy, compute deterministic diagnostics for each bundle and cluster instances into source-topology types.

Recommended diagnostics:

- `source_coverage`: how many partial/candidate items are observed in each source.
- `direct_relation_strength`: BI co-bundle counts and UI user-overlap counts between partial items and candidates.
- `embedding_contrast`: candidate-to-partial similarity margins in content, description, CF, and LightGCN spaces.
- `partial_bundle_coherence`: how tightly partial items connect to each other in relation or embedding space.
- `source_agreement`: whether multiple sources point to the same top candidate.
- `sparsity_profile`: whether the instance is relation-rich, embedding-driven, source-conflicting, low-contrast, or sparse.

Example diagnostic row:

```text
bundle_id,
bi_partial_coverage,
bi_candidate_coverage,
max_bi_relation,
bi_margin,
ui_partial_coverage,
max_ui_relation,
content_top_score,
content_margin,
description_top_score,
description_margin,
partial_content_cohesion,
partial_description_cohesion,
source_top_consensus,
num_sources_with_signal,
cluster_labels,
prediction,
gt_label,
hit
```

Possible cluster labels:

- `relation_rich`: BI/UI coverage is high and direct relation margins are strong.
- `embedding_driven`: relational signals are weak but embedding contrast is strong.
- `multi_source_agreement`: multiple sources identify the same likely candidate.
- `source_conflict`: different sources point to different candidates.
- `low_contrast_ambiguous`: candidates have similar evidence strength.
- `sparse_hard`: most sources have weak or missing signals.
- `partial_incoherent`: partial items are internally weakly connected.

These labels can be multi-label. For example:

```json
["embedding_driven", "source_conflict"]
```

This enables analysis such as:

- Performance by instance type.
- Whether code-method gains are larger on relation-rich or embedding-driven samples.
- Whether generated strategy families align with deterministic bundle diagnostics.
- Whether strategy-diagnostic alignment correlates with accuracy.

The main research framing is:

```text
Bundle completion instances are heterogeneous. Each instance has its own evidence topology across bundle-item, user-item, and multimodal similarity relations. Code generation acts as an instance-adaptive evidence compiler that selects and materializes relation paths suited to the current bundle.
```

## Strategy Auditing

LLM-generated strategy names are not stable enough for direct statistics. The same relation path may be named in many ways, such as:

```text
ib_x_bi_cobundle_context
bi_train_cooccurrence
bundle_item_neighbor_retrieval
train_bundle_item_expansion
```

Use rule-based canonicalization based on actual source and relation footprints.

Recommended canonical families:

- `bi_cobundle`
- `ui_user_relation`
- `content_embedding`
- `description_embedding`
- `item_cf_embedding`
- `bi_lightgcn_embedding`
- `metadata_text`
- `sparse_fallback`
- `unknown`

Classification should use:

```text
strategies[].name
strategies[].relation_signal
strategies[].data_sources
strategies[].description
policy_trace.implemented_strategies
partial_evidence.*.evidence
candidate_evidence.*.evidence
```

The audit output can be stored as:

```text
analysis/strategy_audit/{dataset}/{run_id}/strategy_audit.csv
```

Useful columns:

```text
bundle_id,
prediction,
gt_label,
hit,
raw_strategy_names,
canonical_families,
uses_bi_cobundle,
uses_ui_user_relation,
uses_content_embedding,
uses_description_embedding,
uses_item_cf_embedding,
uses_bi_lightgcn_embedding,
uses_sparse_fallback,
num_canonical_families
```


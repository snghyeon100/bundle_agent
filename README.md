# Bundle Agent

Training-free, source-grounded bundle completion with two code-generating methods:

- **Simple Generate-Evaluate-Decide**: generate and execute signal code, evaluate evidence sufficiency, refine once by default, and return a prediction-only decision.
- **Progressive Signal Discovery**: establish broad source coverage, diagnose evidence gaps, and plan deeper investigations before decision.

The configured default is Simple Generate-Evaluate-Decide. For each partial bundle and candidate set, it:

1. builds a train-safe Source Capability Manifest;
2. generates and executes Python signal code over ID-only case input;
3. validates compact candidate-scoped Evidence JSON;
4. evaluates whether the evidence is `SUFFICIENT`, requires `REFINE`, or is `INCONCLUSIVE`;
5. passes deterministic item text, verified evidence, and the evaluation to a prediction-only Decision Agent.

The canonical evidence-agent input contains the real `bundle_id`, partial item IDs, and candidate label/item-ID mappings. Item metadata and representative examples are retrieved from allowed sources and recorded with provenance. Ground truth, test GT files, predictions, hits, and result files are never exposed to generated code.

## Run

```powershell
pip install -r requirements.txt
python src/main.py --config config.yaml
```

Resume a partial run with:

```powershell
python src/main.py --config config.yaml --resume path\to\partial.csv
```

The default `data_path` is `./datasets`. Dataset files, generated workspaces, and result files are local artifacts ignored by Git.

## Rank-free operator MVP

The first A2Flow-lite slice is implemented separately from ranking and reflection:

```text
existing bi_valid_input.txt + bi_valid_gt.txt
-> per-sample prompt containing partial texts + candidate texts + GT label
-> source-free semantic operator induction (one LLM call per sample)
-> operator_pool.json
-> one source-free semantic clustering pass
-> dataset-specific semantic operator_library.json
-> source grounding in a later phase
```

Neither induction nor clustering receives a Source Capability Manifest. Sources and concrete
implementation choices are deliberately deferred until after the semantic library is formed.

Step 1 extracts only raw operators from validation samples:

```powershell
python tests/test_operator_induction.py --config config_operator.yaml
```

Results are saved to `tests/outputs/operators/<dataset>_<timestamp>/`. The folder contains the flat `operator_pool.json`, one combined `operators_by_sample.json`, and one compact JSON per sample under `samples/` with `input_items`, `candidate_items`, `gt_item`, and `operators`.

Step 2 clusters the most recent operator pool for the configured dataset:

```powershell
python tests/test_operator_clustering.py --config config_operator.yaml
```

An explicit pool can also be supplied:

```powershell
python tests/test_operator_clustering.py --config config_operator.yaml --operator_pool path\to\operator_pool.json
```

Clustering results are saved to `tests/outputs/cluster/<dataset>_<timestamp>/`.

## Method configuration

Select the method in `config.yaml`:

```yaml
method: simple_generate_evaluate_decide  # progressive_signal_discovery | simple_generate_evaluate_decide
```

Method-specific settings use the `simple_signal_` and `psd_` prefixes. The simple method's default additional refinement budget is:

```yaml
simple_signal_max_refinement_rounds: 1
```

The current bundle's train-side context policy is explicit for both methods:

```yaml
psd_current_bundle_train_context_policy: allow  # allow | exclude
simple_signal_current_bundle_train_context_policy: allow  # allow | exclude
```

The policy is included in every Source Capability Manifest so generated investigations can distinguish same-bundle train context from other historical contexts.

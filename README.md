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

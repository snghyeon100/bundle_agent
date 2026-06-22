# Bundle Agent

Training-free bundle completion with **Progressive Signal Discovery**.

For each partial bundle and candidate set, the runner:

1. builds a train-safe Source Capability Manifest;
2. plans broad source coverage without predicting signal importance;
3. generates and executes Python surface-observation code;
4. diagnoses the observed evidence;
5. autonomously plans and executes deeper investigations when evidence gaps remain;
6. passes compact verified Evidence JSON to the final Decision Agent.

The canonical agent input contains the real `bundle_id`, partial item IDs, and candidate label–item ID mappings. Item metadata and representative examples are retrieved from allowed sources and recorded with provenance. Ground truth, test GT files, predictions, hits, and result files are never exposed to generated code.

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

`config.yaml` contains only the Progressive Signal Discovery runner's dataset, LLM, stage budget, adaptive-loop, workspace, safety, retry, and execution settings.

The current bundle's train-side context policy is explicit:

```yaml
psd_current_bundle_train_context_policy: allow  # allow | exclude
```

The policy is included in every Source Capability Manifest so generated investigations can distinguish same-bundle train context from other historical contexts.

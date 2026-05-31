# Bundle Agent

Minimal zero-shot bundle completion baseline.

This repo intentionally keeps only the baseline method:

- text-only prompt
- multiple-choice prediction
- no ranking mode
- no multimodal input
- no hard negatives
- no co-occurrence features
- no graph, category, ICL, user, or retrieval context

## Run

```powershell
pip install -r requirements.txt
python src/main.py --config config.yaml
```

The default `data_path` points to `../LLM-ZeroShot/datasets` so this repo can reuse the existing local datasets without committing them.

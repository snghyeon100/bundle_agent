# Bundle Agent

Minimal zero-shot bundle completion baseline.

This repo keeps the baseline method as the default:

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

The default `data_path` points to `./datasets`, which is ignored by Git because the dataset and embedding files are large.

## Candidate Reasoning Method

Set `use_candidate_reasoning: true` in `config.yaml` to use the two-step `candidate_reasoning` method. The first step asks the model once to write pure English reasoning for all candidates. The second step passes those candidate reasoning outputs back to the model and asks for the final single-letter prediction.

When disabled, the code runs the original baseline prompt directly.

## Retry Behavior

The runner retries retryable service errors such as `503`, high-demand, overloaded, or temporarily unavailable responses. Quota or permission errors such as `403` stop the run immediately. Only fully completed samples are written to the partial CSV, so resuming starts from the next unfinished sample.

## Separate API Keys

Set `prediction_api_key_env` and `reasoning_api_key_env` in `config.yaml` to use different environment variables for final prediction calls and the single candidate reasoning call. Leave either option empty to fall back to `GEMINI_API_KEY` or `GOOGLE_API_KEY`.

import json

from agents.common import build_agent_sample_view, candidate_labels, task_semantics


def generate_prediction_prompt(sample, retrieval_rounds):
    sample_view = build_agent_sample_view(sample)
    labels = candidate_labels(sample)
    return (
        "You are the final predictor for a four-stage bundle completion agent.\n"
        "Use the compact retrieval trajectory and verifier assessments to choose one candidate letter. "
        "Do not assume retrieved evidence is always reliable. Compare numeric diagnostics, evidence_for, and evidence_against. "
        "Downweight weak/noisy/tied evidence, and avoid choosing a candidate only because one numeric signal is high. "
        "However, if you choose against the strongest discriminative numeric or programmatic evidence, you must identify concrete counter-evidence from the trajectory. "
        "If verifier outputs say the evidence is insufficient or failed, do not invent unsupported evidence; make a low-confidence choice from the original item text and clearly downweight failed retrieval sources.\n\n"
        "Dataset/task semantics:\n"
        f"{task_semantics(sample.get('dataset', ''))}\n\n"
        "Current sample JSON:\n"
        f"{json.dumps(sample_view, ensure_ascii=False, indent=2)}\n\n"
        "Compact retrieval trajectory JSON:\n"
        f"{json.dumps(retrieval_rounds, ensure_ascii=False, indent=2)}\n\n"
        f"In candidate_tradeoff, include every candidate label exactly once: {', '.join(labels)}.\n"
        "Return only valid JSON using this schema:\n"
        "{\n"
        '  "evidence_quality": "none|weak|medium|strong",\n'
        '  "candidate_tradeoff": {\n'
        '    "A": "main evidence for and against this candidate",\n'
        '    "B": "main evidence for and against this candidate"\n'
        "  },\n"
        '  "downweighted_evidence": ["..."],\n'
        '  "decision_rule": "how you balanced task fit, numeric evidence, provenance, and counter-evidence",\n'
        '  "reasoning": "concise final comparison across candidates",\n'
        '  "prediction": "A",\n'
        '  "confidence": "low|medium|high",\n'
        '  "main_sources_used_for_decision": ["item_info.json", "bi_train.txt"]\n'
        "}\n"
    )

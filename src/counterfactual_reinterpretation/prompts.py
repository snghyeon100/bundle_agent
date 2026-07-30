"""Prompts for two-step counterfactual set reinterpretation."""

from code.common import pretty_json, task_semantics


def _case_payload(partial_items, answer_options):
    return {
        "partial_items": [
            {
                "label": str(item.get("label") or ""),
                "text": str(item.get("text") or ""),
            }
            for item in partial_items
            if isinstance(item, dict)
        ],
        "answer_options": [
            {
                "label": str(option.get("label") or ""),
                "text": str(option.get("text") or ""),
            }
            for option in answer_options
            if isinstance(option, dict)
        ],
    }


def candidate_reinterpretation_prompt(
    *,
    dataset,
    partial_items,
    answer_options,
):
    """Analyze the completed set induced by every candidate without ranking."""
    payload = _case_payload(partial_items, answer_options)
    return (
        "You are a Counterfactual Set Interpreter.\n\n"
        f"{task_semantics(dataset)}\n\n"
        "TASK\n"
        "Analyze every answer option independently. For each option, insert it into "
        "the same partial set and reinterpret the resulting completed set as a whole. "
        "Do not choose, rank, compare, or eliminate answer options in this step.\n\n"
        "Apply the same test to every option:\n"
        "1. State the most coherent interpretation of the completed set.\n"
        "2. Explain the contribution of every observed partial member under that "
        "interpretation.\n"
        "3. Explain what the inserted candidate contributes.\n"
        "4. Determine whether it closes a genuine gap in the partial set.\n"
        "5. Counterfactually remove the candidate and state what identifiable gap "
        "would reappear.\n"
        "6. Record contradictions or redundancy. Use an empty list when none is "
        "grounded in the supplied texts.\n\n"
        "Judge composition, complementarity, and whole-set coherence rather than "
        "mere textual similarity. Do not invent unavailable item attributes or "
        "historical facts. Acknowledge ambiguity concisely instead of forcing an "
        "overly specific story.\n\n"
        "CASE\n"
        f"{pretty_json(payload)}\n\n"
        "Return JSON only. Include exactly one reinterpretation for every supplied "
        "answer option, in the supplied option order:\n"
        "{\n"
        '  "reinterpretations": [\n'
        "    {\n"
        '      "label": "one supplied answer-option label",\n'
        '      "completed_set_interpretation": "the most coherent whole-set reading",\n'
        '      "partial_member_contributions": [\n'
        "        {\n"
        '          "partial_label": "one supplied partial-item label",\n'
        '          "contribution": "its contribution under this completed-set reading"\n'
        "        }\n"
        "      ],\n"
        '      "candidate_contribution": "what the inserted candidate contributes",\n'
        '      "role_closure": "the gap it closes, or why it does not close one",\n'
        '      "counterfactual_necessity": "what gap reappears if it is removed",\n'
        '      "conflicts_or_redundancies": ["concise grounded issue"]\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Return no explanation outside the JSON object."
    )


def contrastive_adjudication_prompt(
    *,
    dataset,
    partial_items,
    answer_options,
    reinterpretations,
    validation_issues=None,
):
    """Contrast candidate-induced interpretations and return a full ranking."""
    payload = _case_payload(partial_items, answer_options)
    analysis_payload = {
        "case": payload,
        "candidate_reinterpretations": reinterpretations,
    }
    if validation_issues:
        analysis_payload["stage_1_validation_notes"] = [
            str(issue) for issue in validation_issues
        ]
    return (
        "You are a Contrastive Set-Completion Adjudicator.\n\n"
        f"{task_semantics(dataset)}\n\n"
        "TASK\n"
        "Compare the candidate-induced completed-set interpretations from Stage 1 "
        "and rank every supplied answer option. The Stage 1 prose is analysis, not "
        "evidence of correctness: do not reward a candidate merely because its "
        "interpretation is longer, more confident, or more eloquent.\n\n"
        "Prefer the candidate whose insertion:\n"
        "- explains all observed partial members with the fewest arbitrary assumptions;\n"
        "- closes a genuine compositional gap rather than adding a merely related item;\n"
        "- has a meaningful counterfactual necessity when removed; and\n"
        "- introduces the least contradiction or redundant role duplication.\n\n"
        "Use the original item texts to audit Stage 1. Do not invent attributes, "
        "source evidence, or facts not present in the case. Resolve close candidates "
        "by stating their decisive relational difference. Rank every option exactly "
        "once, even when several are plausible.\n\n"
        "ADJUDICATION INPUT\n"
        f"{pretty_json(analysis_payload)}\n\n"
        "Return JSON only:\n"
        "{\n"
        '  "ranking": ["every supplied label exactly once, best to worst"],\n'
        '  "prediction": "the first label in ranking",\n'
        '  "decisive_comparison": "why the winner completes the set better than its closest rival",\n'
        '  "decision_basis": {\n'
        '    "explanatory_coverage": "comparative judgment",\n'
        '    "role_closure": "comparative judgment",\n'
        '    "counterfactual_necessity": "comparative judgment",\n'
        '    "conflict_or_redundancy": "comparative judgment"\n'
        "  }\n"
        "}\n\n"
        "Return no explanation outside the JSON object."
    )

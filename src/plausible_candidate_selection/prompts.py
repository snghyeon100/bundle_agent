"""Prompt for direct plausible-set selection and complete candidate ranking."""

from code.common import pretty_json, task_semantics


def direct_plausible_set_prompt(*, dataset, partial_items, answer_options):
    """Ask for an unconstrained plausible set and a separate full ranking."""
    payload = {
        "dataset": dataset,
        "partial_items": [
            {"text": str(item.get("text") or "")}
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
    fashion_semantics = (
        "A fashion outfit is a complementary composition of item roles rather than "
        "a set of merely similar or interchangeable products.\n\n"
        if str(dataset or "").lower() in {"pog", "pog_dense"}
        else ""
    )
    return (
        "You are a Plausible Completion Set Analyst for bundle completion.\n\n"
        f"{task_semantics(dataset)}\n\n"
        f"{fashion_semantics}"
        "TASK\n"
        "For the plausible-set judgment, do not reduce the answer to only one best option. "
        "Select every supplied answer option that can be defended as a plausible missing "
        "item under at least one coherent interpretation of the observed partial bundle. "
        "Different selected options may correspond to different bundle intents or "
        "missing-item roles; bundle completion is underdetermined.\n\n"
        "Include an option only when you can state a concrete completion hypothesis that "
        "connects it compositionally to the actual partial items. Mere topical similarity, "
        "generic popularity, or the fact that almost any fashion item could coexist is not "
        "enough. Do not force a fixed number of selections, do not select all options for "
        "coverage, and do not use a top-k quota to define the plausible set. It is valid to "
        "select one, several, all, or none when that is genuinely supported.\n\n"
        "First assess every option independently and form plausible_candidates. Then, as a "
        "separate relative judgment, rank every supplied answer option from most to least "
        "plausible as the missing item. The ranking must contain every supplied label "
        "exactly once. A label may appear in the full ranking even when it is not in the "
        "plausible set; its rank only expresses relative preference among the supplied "
        "options. Do not change the independently determined plausible set merely to make "
        "it agree with a top-ranked subset.\n\n"
        "The option labels and texts are the only candidate information available. Ground "
        "truth, correctness labels, source evidence, item IDs, and prior predictions are "
        "not provided.\n\n"
        "DIRECT PLAUSIBLE-SET CASE\n"
        f"{pretty_json(payload)}\n\n"
        "Return JSON only:\n"
        "{\n"
        '  "plausible_candidates": [\n'
        "    {\n"
        '      "label": "one supplied option label",\n'
        '      "completion_hypothesis": "a concise bundle intent or missing-role hypothesis",\n'
        '      "reason": "why this option plausibly completes the actual partial bundle"\n'
        "    }\n"
        "  ],\n"
        '  "ranking": ["every supplied label exactly once, ordered most to least plausible"]\n'
        "}\n\n"
        "Return no explanation outside the JSON object."
    )

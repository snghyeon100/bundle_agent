"""Prompt variant that frames bundle completion as higher-order hyperedge reasoning."""

from operator_learning.prompts import induction_prompt as base_induction_prompt


_STRATEGY_DESIGN_HEADING = "STRATEGY DESIGN\n"
_STRATEGY_DEFINITION_HEADING = "STRATEGY DEFINITION\n"

_HYPEREDGE_VIEW = (
    "Treat a bundle as one hyperedge whose meaning can emerge from the joint composition "
    "of all its items. Seek higher-order information about the completed set rather than "
    "viewing the bundle only as a collection of independent item pairs.\n\n"
)

_HYPEREDGE_STRATEGY_DEFINITION = (
    "Under each completion intent, design one strategy that examines every "
    "candidate-augmented hyperedge Hi as a whole and retrieves source-grounded evidence "
    "about its joint bundle-completion structure. Choose the sources, reference "
    "construction, candidate relation, and computation that best operationalize that "
    "intent. Pairwise relations may be used as ingredients, but the strategy's evidence "
    "must depend on the composition of Hi rather than only on independent candidate-to-item "
    "comparisons. The three strategies must embody three meaningfully different views of "
    "the completed hyperedge.\n\n"
)

_BASE_HYPOTHETICAL_BUNDLE = (
    "Let P be the partial bundle and let c1, ..., cn be the candidate items. For each "
    "candidate ci, construct Bi = P union {ci} as a hypothetical completed bundle. "
    "The Bi are competing completion hypotheses, not bundles that are all true at "
    "the same time. Inspect {B1, ..., Bn} together and interpret each Bi as a whole "
    "completed bundle."
)

_HYPEREDGE_HYPOTHETICAL_BUNDLE = (
    "Let P be the partial bundle and let c1, ..., cn be the candidate items. For each "
    "candidate ci, construct Hi = P union {ci} as a candidate-augmented hyperedge. "
    "The Hi are competing completion hypotheses, not bundles that are all true at "
    "the same time. Inspect {H1, ..., Hn} together and interpret each Hi as a whole "
    "completed bundle."
)


def induction_prompt(
    case,
    source_capabilities,
    operator_memory,
    max_operator_count,
    *,
    text_only=True,
):
    """Build the current spec-first prompt plus hyperedge-only requirements."""
    prompt = base_induction_prompt(
        case,
        source_capabilities,
        operator_memory,
        max_operator_count,
        text_only=text_only,
    )
    if _STRATEGY_DESIGN_HEADING not in prompt:
        raise RuntimeError("base strategy-design heading was not found")
    if _STRATEGY_DEFINITION_HEADING not in prompt:
        raise RuntimeError("base strategy-definition heading was not found")
    if _BASE_HYPOTHETICAL_BUNDLE not in prompt:
        raise RuntimeError("base hypothetical-bundle definition was not found")
    prompt = prompt.replace(
        _STRATEGY_DESIGN_HEADING,
        _STRATEGY_DESIGN_HEADING + _HYPEREDGE_VIEW,
        1,
    )
    prompt = prompt.replace(
        _BASE_HYPOTHETICAL_BUNDLE,
        _HYPEREDGE_HYPOTHETICAL_BUNDLE,
        1,
    )
    prompt = prompt.replace(
        _STRATEGY_DEFINITION_HEADING,
        _STRATEGY_DEFINITION_HEADING + _HYPEREDGE_STRATEGY_DEFINITION,
        1,
    )
    return prompt.replace("reusable ", "")

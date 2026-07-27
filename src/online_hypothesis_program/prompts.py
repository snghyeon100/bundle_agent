"""Prompts for the two-call online hypothesis-program pipeline."""

from code.common import pretty_json, task_semantics

from .schemas import DISCOVERY_SCHEMA_VERSION


def _fashion_semantics(dataset):
    return (
        "A fashion outfit is a complementary composition of item roles rather than "
        "a set of merely similar or interchangeable products.\n\n"
        if str(dataset or "").lower() in {"pog", "pog_dense"}
        else ""
    )


def program_generation_prompt(
    *,
    dataset,
    partial_items,
    source_diagnostics,
    source_capabilities,
    max_hypotheses,
):
    """Build candidate-blind LLM1 input."""
    case = {
        "dataset": dataset,
        "partial_items": [
            {
                key: item[key]
                for key in ("text", "metadata")
                if key in item
            }
            for item in partial_items
            if isinstance(item, dict)
        ],
        "source_diagnostics": source_diagnostics,
    }
    return (
        "You are the online Hypothesis-Conditioned Search Program Agent for bundle "
        "completion.\n\n"
        f"{task_semantics(dataset)}\n\n"
        f"{_fashion_semantics(dataset)}"
        "GOAL\n"
        "From the observed partial bundle, infer several plausible semantic intents and "
        "missing-item roles. For each hypothesis, synthesize one case-conditioned Python "
        "search program that retrieves a small set of plausible missing-item candidates "
        "from the corpus. This is an online program for the current case; it does not need "
        "to be a reusable library operator.\n\n"
        "A semantic hypothesis is a plausible account of the bundle being assembled: the "
        "coherent style, occasion, function, theme, or composition that could connect the "
        "observed items, plus the complementary contribution a missing item could make. "
        "Ground it in concrete cues visible in the partial-item descriptions. Produce "
        "meaningfully different interpretations, not different search procedures for the "
        "same interpretation.\n\n"
        "CANDIDATE BLINDNESS\n"
        "No benchmark answer options or ground truth are available in this call. Never "
        "assume, reconstruct, or refer to them. The generated code must retrieve canonical "
        "corpus item IDs through SourceAPI and must exclude the partial item IDs.\n\n"
        "PROGRAM EXECUTION BOUNDARY\n"
        "Each code string must define exactly one public function:\n\n"
        "def execute(partial_item_ids, source_api, candidate_budget, evidence_budget):\n"
        "    ...\n\n"
        "The code may use only the supplied arguments, safe Python builtins, and permitted "
        "standard-library imports. It must not read files, access paths, use the network, "
        "spawn processes, inspect the runtime, or import third-party libraries. It must "
        "access data only through the documented SourceAPI methods. Use candidate_budget "
        "and evidence_budget directly; do not hard-code alternative K, M, or top-N output "
        "budgets.\n\n"
        "Every program must return exactly this internal object:\n"
        "{\n"
        '  "candidate_proposals": [\n'
        '    {"item_id": "<integer item ID returned by SourceAPI>", '
        '"evidence_refs": ["E1"]}\n'
        "  ],\n"
        '  "evidence_records": [\n'
        "    {\n"
        '      "evidence_id": "E1",\n'
        '      "type": "one declared evidence type",\n'
        '      "source": "one required source ID",\n'
        '      "anchor_item_ids": ["<integer partial item ID>"],\n'
        '      "related_item_ids": ["<integer proposed item ID>"],\n'
        '      "related_bundle_ids": [],\n'
        '      "attributes": {}\n'
        "    }\n"
        "  ],\n"
        '  "used_sources": ["one required source ID"]\n'
        "}\n\n"
        "Every proposed item must reference at least one evidence record whose "
        "related_item_ids contains that item. Empty candidate and evidence arrays are valid "
        "when the program finds nothing. Evidence records are raw audit provenance; a "
        "deterministic renderer will later resolve useful item/bundle context and remove "
        "raw IDs and numeric scores before prediction.\n\n"
        "SOURCE SEMANTICS\n"
        "Source diagnostics indicate feasibility, not bundle intent. Category IDs are "
        "opaque identifiers: equality, frequency, novelty, and co-occurrence are valid, "
        "but a category ID has no named semantic role unless explicit item text supports "
        "that interpretation. Embeddings are opaque vectors and may be used through the "
        "neighbor API, but they must not be decoded into named attributes. Explicit lexical "
        "information in item text may be used. Choose only sources actually needed by each "
        "program, and keep evidence representative and bounded.\n\n"
        "AVAILABLE SOURCEAPI\n"
        f"{pretty_json(source_capabilities)}\n\n"
        "CANDIDATE-BLIND ONLINE CASE\n"
        f"{pretty_json(case)}\n\n"
        f"Return JSON only with between 1 and {int(max_hypotheses)} hypotheses and "
        "one Python program per hypothesis. Code must be encoded as a JSON string:\n"
        "{\n"
        f'  "schema_version": "{DISCOVERY_SCHEMA_VERSION}",\n'
        '  "hypotheses": [\n'
        "    {\n"
        '      "id": "H1",\n'
        '      "observed_cues": ["concrete cue from a partial-item description"],\n'
        '      "intent": "one-sentence interpretation of the bundle composition",\n'
        '      "missing_role": "the complementary contribution a missing item could make"\n'
        "    }\n"
        "  ],\n"
        '  "programs": [\n'
        "    {\n"
        '      "hypothesis_id": "H1",\n'
        '      "program_id": "P1",\n'
        '      "name": "ConcisePascalCaseName",\n'
        '      "required_sources": ["exact source component ID"],\n'
        '      "evidence_types": ["short structural evidence label"],\n'
        '      "code": "def execute(partial_item_ids, source_api, candidate_budget, evidence_budget):\\n    ..."\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Return no markdown and no explanation outside the JSON object."
    )


def prediction_prompt(*, dataset, partial_items, answer_options, search_evidence):
    """Build LLM2 input without raw source IDs or numeric retrieval scores."""
    payload = {
        "dataset": dataset,
        "partial_items": [
            {
                key: item[key]
                for key in ("text", "metadata")
                if key in item
            }
            for item in partial_items
            if isinstance(item, dict)
        ],
        "hypothesis_search_results": search_evidence,
        "answer_options": answer_options,
    }
    return (
        "You are the final Prediction Agent for bundle completion.\n\n"
        f"{task_semantics(dataset)}\n\n"
        f"{_fashion_semantics(dataset)}"
        "Select exactly one answer-option label. The semantic hypotheses were inferred "
        "without seeing the answer options. Retrieved examples came from the corpus under "
        "hypothesis-conditioned programs; they may or may not be identical to an answer "
        "option. Use them as source-grounded completion exemplars and compare their semantic "
        "and compositional patterns with every answer option. An exact option match is "
        "direct support, but absence of an exact match does not by itself reject an option. "
        "Failed or empty searches provide no evidence and must not be treated as negative "
        "evidence.\n\n"
        "The search evidence intentionally excludes raw item, bundle, and user IDs as well "
        "as opaque numeric scores. Base the decision on the readable item text, relation "
        "summaries, representative contexts, and the partial bundle. Do not output a "
        "retrieved exemplar unless it is one of the labeled answer options.\n\n"
        "ONLINE DECISION CASE\n"
        f"{pretty_json(payload)}\n\n"
        "Return JSON only:\n"
        "{\n"
        '  "prediction": "one supplied answer-option label",\n'
        '  "rationale": "brief comparison grounded in the hypotheses and readable evidence"\n'
        "}\n"
    )

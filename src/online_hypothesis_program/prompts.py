"""Prompts for completion-exemplar retrieval and final prediction."""

from code.common import pretty_json, task_semantics

from .schemas import DISCOVERY_SCHEMA_VERSION


def _fashion_semantics(dataset):
    return (
        "A fashion outfit is a composition of complementary item roles.\n\n"
        if str(dataset or "").lower() in {"pog", "pog_dense"}
        else ""
    )


def program_generation_prompt(
    *,
    dataset,
    partial_items,
    workspace_manifest,
    min_hypotheses,
    max_hypotheses,
    retrieved_item_budget,
    supporting_context_budget,
):
    """Build the LLM1 hypothesis-conditioned retrieval prompt."""
    partial_item_view = [
        {
            "item_id": int(item["item_id"]),
            "text": str(item.get("text") or ""),
        }
        for item in partial_items
        if isinstance(item, dict) and "item_id" in item
    ]
    workspace_view = {
        "item_ids": workspace_manifest.get("always_available", {}).get(
            "item_ids",
            "tuple of every canonical integer corpus item_id",
        ),
        **{
            str(component.get("id")): {
                key: component[key]
                for key in (
                    "meaning",
                    "source_record_format",
                    "runtime_format",
                    "observed_field_schema",
                )
                if key in component
            }
            for component in workspace_manifest.get("components", [])
            if isinstance(component, dict) and component.get("id")
        },
    }
    budget = {
        "max_retrieved_items": int(retrieved_item_budget),
        "max_supporting_contexts_per_item": int(supporting_context_budget),
    }
    interpretation_count_instruction = (
        f"Interpret the partial bundle in exactly {int(min_hypotheses)} "
        "genuinely different ways."
        if int(min_hypotheses) == int(max_hypotheses)
        else (
            f"Interpret the partial bundle in {int(min_hypotheses)} to "
            f"{int(max_hypotheses)} genuinely different ways."
        )
    )
    return (
        "You are the Completion Retrieval Program Synthesis Agent.\n\n"
        f"{task_semantics(dataset)}\n"
        f"{_fashion_semantics(dataset)}"
        "PARTIAL BUNDLE\n"
        f"{pretty_json({'partial_items': partial_item_view})}\n\n"
        "The item IDs above are the same entity IDs received through "
        "partial_item_ids at runtime. Use the function argument to bind the case "
        "instead of writing those literal IDs into code.\n\n"
        f"{interpretation_count_instruction} For each interpretation, "
        "state the latent principle connecting the observed items and the relation "
        "that a plausible additional item should have with the bundle.\n\n"
        "Design each program as a creative, high-level retrieval strategy that goes "
        "beyond an obvious one-hop co-occurrence or nearest-neighbor heuristic by "
        "purposefully connecting non-trivial intermediate references and source "
        "relations; every added step must materially implement the hypothesis rather "
        "than add complexity for its own sake.\n\n"
        "For every interpretation, write one Python program that searches the corpus "
        "for a small set of plausible completion items. The program must first build "
        "a hypothesis-specific reference from the runtime partial_item_ids, then "
        "retrieve items according to their relation to that reference. Different "
        "programs must create materially different references or test materially "
        "different item relations; paraphrasing the same search is not a distinct "
        "strategy.\n\n"
        "The visible partial text guides the interpretation and program design. The "
        "program remains reusable because the actual case is bound at runtime through "
        "partial_item_ids. Semantic role conditions may refine items reached from the "
        "partial-conditioned reference.\n\n"
        "FUNCTION CONTRACT\n\n"
        "def retrieve(partial_item_ids, dataset_workspace, parameters, budget):\n"
        "    partial_set = set(partial_item_ids)\n"
        "    retrieved_items = []\n"
        "    # construct a reference from partial_item_ids and required sources\n"
        "    # retrieve and rank non-partial items against that reference\n"
        "    # attach source records that justify each retrieved item\n"
        "    return retrieved_items[:budget['max_retrieved_items']]\n\n"
        "dataset_workspace has exactly the top-level shape shown below. Access a "
        "source directly, for example dataset_workspace['bundle_item_history']. "
        "item_ids is always available and is not a required_sources source ID. "
        "budget contains max_retrieved_items and "
        "max_supporting_contexts_per_item. Use safe Python builtins and the "
        "standard-library "
        "modules collections, functools, heapq, itertools, math, and statistics as "
        "needed.\n\n"
        "retrieve returns a list of at most budget['max_retrieved_items'] objects. "
        "Each object has this exact shape:\n"
        "{\n"
        '  "item_id": 0,\n'
        '  "provenance": [\n'
        "    {\n"
        '      "source": "one required source ID",\n'
        '      "relation": "why this item was retrieved under the hypothesis",\n'
        '      "supporting_context": {\n'
        '        "item_ids": [],\n'
        '        "bundle_ids": [],\n'
        '        "user_ids": []\n'
        "      }\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Each retrieved item has at least one provenance record and at most "
        "budget['max_supporting_contexts_per_item']. Each provenance record is one "
        "representative supporting context. The returned item IDs and context IDs are "
        "internal references; the runtime resolves them into readable item, bundle, "
        "and user contexts for the prediction agent. Return an empty list when no "
        "grounded completion exemplar is found.\n\n"
        "DATASET_WORKSPACE RUNTIME SHAPE\n"
        f"{pretty_json(workspace_view)}\n\n"
        "FIXED RUNTIME BUDGET\n"
        f"{pretty_json(budget)}\n\n"
        "Return JSON only in this exact structure. Encode Python code as a JSON "
        "string:\n"
        "{\n"
        f'  "schema_version": "{DISCOVERY_SCHEMA_VERSION}",\n'
        '  "programs": [\n'
        "    {\n"
        '      "id": "P1",\n'
        '      "hypothesis": "latent bundle principle and the relation an additional item should have with it",\n'
        '      "strategy": {\n'
        '        "reference": "how the partial-conditioned reference is constructed",\n'
        '        "retrieval": "how plausible completion items are retrieved from that reference"\n'
        "      },\n"
        '      "required_sources": ["exact source component ID"],\n'
        '      "parameters": {},\n'
        '      "code": "def retrieve(partial_item_ids, dataset_workspace, parameters, budget):\\n    ..."\n'
        "    }\n"
        "  ]\n"
        "}\n"
    )


def prediction_prompt(*, dataset, partial_items, answer_options, retrieval_evidence):
    """Build the LLM2 exemplar-aware ranking prompt."""
    payload = {
        "dataset": dataset,
        "partial_item_texts": [
            str(item.get("text") or "")
            for item in partial_items
            if isinstance(item, dict)
        ],
        "hypotheses_and_retrieved_exemplars": retrieval_evidence,
        "answer_options": answer_options,
    }
    return (
        "You are the Retrieval-Grounded Prediction Agent for bundle completion.\n\n"
        f"{task_semantics(dataset)}\n"
        f"{_fashion_semantics(dataset)}"
        "Each completion hypothesis is accompanied by corpus items retrieved from "
        "source data and readable provenance explaining their inclusion. Treat the "
        "retrieved items as hypothesis-specific completion exemplars. An exemplar can "
        "inform the expected relation or item type even when it is not itself an "
        "answer option. Compare the partial bundle, retrieved exemplars, provenance, "
        "and answer-option text, then rank every option from most to least plausible.\n\n"
        "DECISION CASE\n"
        f"{pretty_json(payload)}\n\n"
        "Return JSON only. Include every supplied label exactly once, and make "
        "prediction equal ranking[0]. Keep rationale to at most two sentences:\n"
        "{\n"
        '  "prediction": "top-ranked label",\n'
        '  "ranking": ["all labels exactly once"],\n'
        '  "rationale": "brief retrieval-grounded comparison"\n'
        "}\n"
    )

"""Prompts for candidate-program induction and offline compilation."""

from code.common import pretty_json, task_semantics

from .schemas import CANDIDATE_PROPOSAL_OUTPUT_CONTRACT, OPERATOR_FIELDS


SOURCE_API_CONTRACT = {
    "available_sources": (
        "property returning the exact capability IDs available to this run"
    ),
    "get_all_item_ids()": "return all canonical corpus item IDs",
    "get_dataset_statistics()": "return dataset_statistics as a dictionary",
    "get_item_metadata(item_ids)": (
        "return {item_id: metadata_dict} from item_metadata"
    ),
    "get_bundles_for_items(item_ids)": (
        "return {item_id: [bundle_id, ...]} from bundle_item_history"
    ),
    "get_items_for_bundles(bundle_ids)": (
        "return {bundle_id: [item_id, ...]} from bundle_item_history"
    ),
    "get_users_for_items(item_ids)": (
        "return {item_id: [user_id, ...]} from user_item_history"
    ),
    "get_items_for_users(user_ids)": (
        "return {user_id: [item_id, ...]} from user_item_history"
    ),
    "get_item_embeddings(item_ids, source_id)": (
        "return {item_id: [float, ...]} from one permitted embedding source"
    ),
}


def _compact_item(item, dataset):
    if not isinstance(item, dict):
        return item
    compact = {
        "item_id": item.get("item_id"),
        "text": str(item.get("text", "")),
    }
    metadata = item.get("metadata", {})
    if isinstance(metadata, dict) and metadata:
        compact["metadata"] = metadata
    return compact


def candidate_blind_case_view(case, *, text_only=True):
    """Return the only case fields permitted in the induction prompt."""
    dataset = case.get("dataset")
    items = [
        _compact_item(item, dataset)
        for item in case.get("partial_items", [])
        if isinstance(item, dict)
    ]
    if text_only:
        items = [
            {
                field: item[field]
                for field in ("text", "metadata")
                if field in item
            }
            for item in items
        ]
    return {
        "dataset": dataset,
        "partial_items": items,
        "source_diagnostics": case.get("source_diagnostics", {}),
    }


def _program_definition(case):
    dataset_name = str(case.get("dataset") or "").lower()
    fashion_semantics = (
        "For fashion data, an outfit is a complementary composition of item roles, "
        "not merely a set of similar or interchangeable items.\n\n"
        if dataset_name in {"pog", "pog_dense"}
        else ""
    )
    return (
        "You are the Candidate-Program Discovery Agent for bundle completion.\n\n"
        f"{task_semantics(case.get('dataset'))}\n\n"
        f"{fashion_semantics}"
        "PROGRAM DEFINITION\n"
        "A macro operator is a reusable, hypothesis-conditioned candidate-retrieval "
        "program. Given any partial bundle, it uses only permitted sources to retrieve "
        "a small set of plausible missing-item IDs and source provenance for every "
        "proposed item. It may contain several dependent lookup, traversal, aggregation, "
        "normalization, filtering, and evidence-selection steps when they all implement "
        "one completion hypothesis.\n\n"
        "The program does not select the final answer, fuse unrelated hypotheses, expose "
        "an opaque final relevance score as evidence, or dump an unbounded source context. "
        "Numeric measures may be used internally to select a bounded evidence set. The "
        "fixed runtime output is candidate proposals with source provenance.\n\n"
        "GRANULARITY CALIBRATION\n"
        "- Appropriate: retrieve a bounded candidate set under one stable completion "
        "hypothesis and retain representative source records that explain each inclusion.\n"
        "- Too atomic: load metadata, count a relation, compute one similarity, or sort IDs.\n"
        "- Too broad: combine independent co-occurrence, category, embedding, popularity, "
        "and semantic hypotheses into a final answer-selection strategy.\n"
        "- A category/profile/intent inference is not sufficient by itself. The same macro "
        "must project that inference back to concrete candidate item IDs.\n\n"
    )


def induction_prompt(
    case,
    source_capabilities,
    operator_memory,
    max_operator_count,
    *,
    text_only=True,
):
    prompt_case = candidate_blind_case_view(case, text_only=text_only)
    return (
        _program_definition(case)
        + "DISCOVERY TASK\n"
        "The case is candidate-blind. No answer options or ground truth are available. "
        "First infer several short, sample-conditioned semantic completion hypotheses "
        "from the observed partial items. A semantic completion hypothesis is a plausible "
        "account of the bundle being assembled: the coherent style, occasion, function, "
        "theme, or composition that could connect the observed items, and the complementary "
        "contribution a missing item could make to that composition. Ground it in concrete "
        "cues visible in the partial-item descriptions. It should explain why an unknown "
        "item could belong in this bundle before deciding how any source will be searched. "
        "Make the hypotheses meaningfully different interpretations of the case rather "
        "than different procedures for the same interpretation.\n\n"
        "Then generalize each case hypothesis one-to-one into a case-independent "
        "candidate-retrieval program specification. The reusable program states how the "
        "semantic completion principle can be operationalized with available evidence, "
        "but it must not hard-code current item names, IDs, product types, or a presumed "
        "answer.\n\n"
        "SOURCE GROUNDING\n"
        "Use source diagnostics only to determine which evidence procedure is feasible; "
        "do not confuse source availability with semantic intent. Use only exact component "
        "IDs from the source manifest. Every proposed candidate must be grounded in at "
        "least one evidence record from a required source. The program code generated in "
        "the next stage will receive only partial item IDs and a scoped SourceAPI.\n\n"
        "Category IDs are opaque. They support equality, frequency, novelty, and "
        "co-occurrence, but not named semantic roles unless the manifest supplies such a "
        "taxonomy. Embeddings are opaque vectors. They support similarity and neighborhood "
        "operations, but must not be decoded into named attributes. Item text may use "
        "explicit lexical information but may not assume an unavailable classifier.\n\n"
        "FIXED OUTPUT CONTRACT\n"
        f'Every program has output_contract="{CANDIDATE_PROPOSAL_OUTPUT_CONTRACT}". '
        "At runtime it must return bounded candidate item IDs plus source-grounded evidence "
        "references for each candidate. Do not invent graph ports, intermediate operator "
        "outputs, paths, workflows, or input/output entity contracts.\n\n"
        "FORBIDDEN PREVIOUS PROGRAM SIGNATURES\n"
        "The compact entries below are exclusion records for programs already proposed. "
        "They are not examples, templates, or a verified online library. Do not reproduce, "
        "rename, extend, specialize, or paraphrase a listed program principle. A program "
        "is not new when only its metric, modality, parameter, or wording changes while "
        "its hypothesis and source/evidence family remain the same. Return empty arrays "
        "when this case supports no genuinely new executable program principle.\n\n"
        f"{pretty_json(operator_memory)}\n\n"
        "SOURCE MANIFEST\n"
        f"{pretty_json(source_capabilities)}\n\n"
        "CANDIDATE-BLIND DISCOVERY CASE\n"
        f"{pretty_json(prompt_case)}\n\n"
        "For each case hypothesis, list 1 to 4 concise observed_cues taken from the "
        "partial-item descriptions and keep statement to one sentence. Write 3 to 8 "
        "concrete pseudocode steps. Each program must finish by returning concrete "
        "candidate item IDs and representative provenance records within the runtime "
        "budgets. evidence_types are short structural labels such as "
        "historical_bundle_context, related_item_context, semantic_neighbor_context, or "
        "category_profile_context; choose only types actually needed by the hypothesis.\n\n"
        f"Return JSON only with between 0 and {int(max_operator_count)} hypotheses and "
        "one-to-one programs:\n"
        "{\n"
        '  "hypotheses": [\n'
        "    {\n"
        '      "id": "H1",\n'
        '      "observed_cues": ["a concrete semantic cue visible in the partial items"],\n'
        '      "statement": "one sample-conditioned plausible completion hypothesis"\n'
        "    }\n"
        "  ],\n"
        '  "operators": [\n'
        "    {\n"
        '      "hypothesis_id": "H1",\n'
        '      "name": "ConcisePascalCaseName",\n'
        '      "hypothesis": "case-independent reusable candidate-retrieval hypothesis",\n'
        '      "required_sources": ["exact component ID"],\n'
        '      "applicability": ["GT-independent condition under which this program is useful"],\n'
        '      "evidence_types": ["structural evidence record type"],\n'
        '      "pseudocode": [\n'
        '        "retrieve hypothesis-relevant source records",\n'
        '        "derive concrete candidate item IDs from those records",\n'
        '        "select a bounded candidate set and representative provenance"\n'
        "      ],\n"
        f'      "output_contract": "{CANDIDATE_PROPOSAL_OUTPUT_CONTRACT}"\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Return no explanation outside the JSON object."
    )


def compilation_prompt(operator, source_capabilities):
    """Compile one canonical specification without exposing its discovery case."""
    compile_view = {
        field: operator[field]
        for field in OPERATOR_FIELDS
        if field in operator
    }
    if operator.get("operator_id"):
        compile_view["operator_id"] = operator["operator_id"]
    allowed_sources = set(operator.get("required_sources", []))
    scoped_manifest = {
        **{
            key: value
            for key, value in source_capabilities.items()
            if key != "components"
        },
        "components": [
            component
            for component in source_capabilities.get("components", [])
            if isinstance(component, dict)
            and component.get("id") in allowed_sources
        ],
    }
    result_shape = {
        "schema_version": "candidate_proposal_set_v1",
        "program_id": operator.get("operator_id") or operator.get("name"),
        "hypothesis": operator.get("hypothesis"),
        "candidate_proposals": [
            {
                "item_id": "canonical item ID",
                "evidence_refs": ["E1"],
            }
        ],
        "evidence_records": [
            {
                "evidence_id": "E1",
                "type": "one declared evidence type",
                "source": "one required source ID",
                "anchor_item_ids": ["partial item ID"],
                "related_item_ids": ["candidate item ID"],
                "related_bundle_ids": [],
                "attributes": {},
            }
        ],
        "execution_trace": {
            "used_sources": ["required source ID"],
            "candidate_budget": "runtime integer",
            "evidence_budget": "runtime integer",
        },
    }
    return (
        "You are the offline compiler for one reusable bundle-completion candidate "
        "program. Generate Python source implementing the supplied canonical specification. "
        "The discovery sample, candidate options, validation target, and ground truth are "
        "not available and must never be hard-coded.\n\n"
        f"OPERATOR SPECIFICATION\n{pretty_json(compile_view)}\n\n"
        f"SCOPED SOURCE MANIFEST\n{pretty_json(scoped_manifest)}\n\n"
        f"SOURCE API\n{pretty_json(SOURCE_API_CONTRACT)}\n\n"
        "IMPLEMENTATION CONTRACT\n"
        "Define exactly one public entry point:\n"
        "def execute(partial_item_ids, source_api, candidate_budget, evidence_budget):\n"
        "The function must work for arbitrary partial bundles. It may define private helper "
        "functions and use only the supplied source_api for dataset access. It must exclude "
        "partial items from proposals, respect both positive integer budgets, use only "
        "required_sources, and return plain JSON-serializable Python values. Numeric measures "
        "may guide internal retrieval, but the returned evidence must be source entity IDs, "
        "relations, and provenance rather than an invented natural-language rationale. Every "
        "candidate must reference at least one evidence record that contains that candidate "
        "in related_item_ids. Do not read files, use network access, access environment "
        "variables, call another model, or import non-standard packages.\n\n"
        f"RETURN SHAPE\n{pretty_json(result_shape)}\n\n"
        "Return JSON only:\n"
        "{\n"
        f'  "program_name": "{operator.get("name", "")}",\n'
        '  "code": "complete Python source as one JSON string"\n'
        "}\n"
    )

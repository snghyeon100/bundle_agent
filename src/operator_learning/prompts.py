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


def strategy_case_view(case):
    """Expose partial and candidate identities while keeping the GT hidden."""
    dataset = case.get("dataset")
    return {
        "dataset": dataset,
        "partial_items": [
            _compact_item(item, dataset)
            for item in case.get("partial_items", [])
            if isinstance(item, dict)
        ],
        "candidate_items": [
            {
                "label": str(item.get("label") or ""),
                **_compact_item(item, dataset),
            }
            for item in case.get("candidate_items", [])
            if isinstance(item, dict)
        ],
    }


def strategy_evidence_prediction_prompt(
    *,
    dataset,
    partial_items,
    candidate_items,
    strategy_evidence,
):
    """Build a baseline-shaped final-ranking prompt with strategy evidence."""
    name = str(dataset or "").lower()
    if "spotify" in name:
        task_name, bundle_name, item_name = (
            "playlist continuation",
            "music playlist",
            "song",
        )
    else:
        task_name, bundle_name, item_name = (
            "bundle construction",
            "fashion outfit",
            "fashion item",
        )
    partial_text = "; ".join(
        f"{index + 1}. {item.get('text', '')}"
        for index, item in enumerate(partial_items)
        if isinstance(item, dict)
    )
    option_text = "; ".join(
        f"{item.get('label', '')}. {item.get('text', '')}"
        for item in candidate_items
        if isinstance(item, dict)
    )
    return (
        f"You are a helpful and honest assistant. The following is a multiple choice "
        f"question about {task_name}. Rank all options from most to least plausible "
        "using the item text and the supplied source-grounded strategy evidence.\n\n"
        f"Question: Given the partial {bundle_name}: {partial_text}, which candidate "
        f"{item_name} should be included into this {bundle_name}?\n"
        f"Options: {option_text}\n\n"
        "STRATEGY EVIDENCE\n"
        f"{pretty_json(strategy_evidence)}\n\n"
        "Each strategy describes an intent, how it constructs a shared reference "
        "from the partial bundle, and the relation it applies to every candidate. "
        "Use these descriptions to interpret the returned contexts. A context is "
        "evidence, not a vote: more contexts do not automatically make a candidate "
        "better. Context shared unchanged by all candidates is non-discriminative. "
        "Missing context is not automatic contradiction.\n\n"
        "Return JSON only. Include every option label exactly once in ranking, from "
        "most to least plausible, and make prediction equal ranking[0]. Keep the "
        "rationale to at most two sentences:\n"
        "{\n"
        '  "prediction": "top-ranked label",\n'
        '  "ranking": ["all labels exactly once"],\n'
        '  "rationale": "brief evidence-grounded comparison"\n'
        "}"
    )


def induction_prompt(
    case,
    source_capabilities,
    operator_memory,
    max_operator_count,
    *,
    text_only=True,
):
    del operator_memory, text_only
    strategy_count = int(max_operator_count)
    if strategy_count <= 0:
        raise ValueError("max_operator_count must be positive")
    count_words = {
        1: "one",
        2: "two",
        3: "three",
        4: "four",
        5: "five",
        6: "six",
        7: "seven",
        8: "eight",
        9: "nine",
        10: "ten",
    }
    strategy_count_text = count_words.get(
        strategy_count,
        str(strategy_count),
    )
    strategy_ids = [f"S{index}" for index in range(1, strategy_count + 1)]
    if len(strategy_ids) == 1:
        strategy_ids_text = strategy_ids[0]
    else:
        strategy_ids_text = (
            ", ".join(strategy_ids[:-1]) + f", and {strategy_ids[-1]}"
        )
    prompt_case = strategy_case_view(case)
    return (
        "You are a Strategy Designer for bundle completion.\n\n"
        f"{task_semantics(case.get('dataset'))}\n\n"
        "GOAL\n"
        f"Do not choose the correct answer for the current case. Design exactly "
        f"{strategy_count_text} "
        "reusable computational strategies that distinguish candidates from a "
        "bundle-completion perspective.\n\n"
        "TASK\n"
        "You are given:\n"
        "1. an observed partial bundle,\n"
        "2. candidate items to evaluate,\n"
        "3. available dataset sources and their data types, and\n"
        "4. source availability and partial-coverage diagnostics.\n\n"
        "STRATEGY DESIGN\n"
        "Exactly one candidate item is the actual missing item to add to the given "
        "partial bundle, but its identity is not provided.\n\n"
        "Let P be the partial bundle and let c1, ..., cn be the candidate items. For each "
        "candidate ci, construct Bi = P union {ci} as a hypothetical completed bundle. "
        "The Bi are competing completion hypotheses, not bundles that are all true at "
        "the same time. Inspect {B1, ..., Bn} together and interpret each Bi as a whole "
        "completed bundle. Do not compare ci with each partial item independently. "
        "Consider the purpose, theme, usage, composition, and item relationships that "
        "could make the whole Bi coherent.\n\n"
        f"Infer exactly {strategy_count_text} distinct and plausible completion "
        "intents for how the "
        "observed partial bundle P could be completed. A completion intent is a coherent "
        "explanation of the purpose, theme, usage, composition, or relation that could "
        "make items belong to one bundle. An intent must not be a broad explanation that "
        "fits every hypothetical bundle equally. It must create a meaningful contrast "
        "under which some candidates are more natural completions than others.\n\n"
        "Under each intent, identify the candidates that remain difficult to distinguish "
        "from a bundle-completion perspective. Determine what shared evaluation basis "
        "should be built from the partial bundle and sources, and what source-grounded "
        "relation between each candidate and that basis would distinguish the ambiguous "
        "candidates.\n\n"
        "STRATEGY DEFINITION\n"
        "A strategy is a reusable computation under one completion intent. It:\n"
        "1. builds a shared evaluation basis from the partial bundle and available "
        "sources,\n"
        "2. applies the same computation to every candidate to examine the candidate's "
        "bundle-completion relation to that basis, and\n"
        "3. retrieves source-grounded textual contexts that reveal that relation.\n\n"
        "This shared evaluation basis is called the reference. A reference is not the "
        "correct candidate or any particular candidate. Within one strategy, construct "
        "the reference once from the partial bundle and sources, then use the same "
        "reference to evaluate every candidate. Pair each completion intent one-to-one "
        "with exactly one strategy.\n\n"
        "Keep each strategy narrow enough to be implemented as one coherent mechanism. "
        "Treat reference_construction, candidate_relation, and evidence_route as "
        "executable commitments, not broad conceptual summaries.\n\n"
        "The pseudocode must be an exact operational expansion of these commitments. "
        "It must explicitly show: (1) how the shared reference is constructed once, "
        "(2) where each candidate is bound to that reference, and (3) how "
        "candidate-specific contexts are selected and returned. If every declared "
        "mechanism cannot be implemented in the pseudocode, narrow the strategy before "
        "writing the pseudocode.\n\n"
        "Each strategy returns a small, bounded set of contexts for every candidate. "
        "Every final context must be either a related "
        "item text or a historical bundle item-text composition retrieved from the "
        "available sources. Numerical computations may be used internally to retrieve, "
        "compare, and select representative contexts, but do not output numerical scores, "
        "similarities, distances, counts, or diagnostic messages as contexts.\n\n"
        f"The {strategy_count_text} strategies must use meaningfully different "
        "computations. Distinguish "
        "them through their reference construction, candidate relation, or evidence "
        "route. Each pair must differ in at least two of these aspects. "
        "Changing only wording, a metric, an embedding modality, a threshold, or a source "
        "file is not a different strategy.\n\n"
        f"First complete all {strategy_count_text} strategy specifications. Choose "
        "them according to the "
        "candidate ambiguities that need to be resolved, not according to implementation "
        f"convenience. Do not write any Python until all {strategy_count_text} "
        "specifications are complete. "
        "After emitting the specifications, treat them as immutable: the programs must "
        "implement them without replacing, simplifying, or omitting any declared "
        "reference, relation, or evidence-route step.\n\n"
        "Every returned context must be selected by actually applying the declared "
        "candidate relation and must show a concrete connection between that candidate "
        "and the shared reference. Do not return shared reference contexts selected "
        "independently of the candidate; if no such context exists, return an empty "
        "contexts list for that candidate. Do not invent a fallback or repeat the partial "
        "or candidate text as substitute evidence.\n\n"
        "Use only sources present in the manifest. Item IDs are lookup keys; do not "
        "hard-code the IDs, labels, names, or presumed answer from this case. Write "
        "complete executable Python code for each strategy.\n\n"
        "INPUT CASE\n"
        f"{pretty_json(prompt_case)}\n\n"
        "SOURCE DIAGNOSTICS\n"
        '"availability": "available" means that the source exists and can be read by '
        "the program.\n\n"
        '"partial_coverage" describes whether the current partial item IDs have direct '
        "records or relations in that source:\n"
        '- "full": every partial item has at least one direct source record or relation.\n'
        '- "partial": only some partial items have a direct source record or relation.\n'
        '- "none": no partial item has a direct source record or relation.\n\n'
        f"{pretty_json(case.get('source_diagnostics', {}))}\n\n"
        "SOURCE MANIFEST\n"
        f"{pretty_json(source_capabilities)}\n\n"
        "PROGRAM CONTRACT\n"
        "Each program must define this public entry point:\n"
        "def run(partial_items, candidate_items, source_paths, "
        "max_contexts_per_candidate=5):\n\n"
        "partial_items contains the item_id, text, and metadata objects shown in the input "
        "case. candidate_items additionally contains label. source_paths maps each exact "
        "declared source ID to its local file path. Use source_paths rather than fixed "
        "file paths, and declare every accessed source in required_sources.\n\n"
        "The function must return one JSON-serializable result per candidate:\n"
        "[\n"
        "  {\n"
        '    "label": "candidate label",\n'
        '    "item_id": 0,\n'
        '    "contexts": [\n'
        "      {\n"
        '        "text": "related item text or historical bundle item-text composition",\n'
        '        "sources": ["one or more exact declared source IDs"],\n'
        '        "supporting_item_ids": [],\n'
        '        "supporting_bundle_ids": []\n'
        "      }\n"
        "    ]\n"
        "  }\n"
        "]\n\n"
        "This is only an external contract, not an internal skeleton. The program may "
        "freely choose its helper functions, reference representation, indexing, joint or "
        "individual candidate comparison, multi-hop computation, and aggregation. The "
        "code may use the Python standard library, NumPy, and PyTorch. It "
        "must access only its declared required_sources and provide all imports and helper "
        "functions. Every code path must be implemented; return no placeholders, demos, "
        "or ellipses.\n\n"
        "OUTPUT\n"
        "Return JSON only. The strategy_specs array must appear before programs, and must "
        f"contain all {strategy_count_text} complete specifications before any "
        "Python code appears:\n"
        "{\n"
        '  "strategy_specs": [\n'
        "    {\n"
        '      "strategy_id": "S1",\n'
        '      "intent": "one plausible interpretation of the completed bundle",\n'
        '      "name": "ConcisePascalCaseName",\n'
        '      "description": "candidate ambiguity resolved by this strategy",\n'
        '      "reference_construction": "how the partial bundle and sources build one shared evaluation basis for all candidates",\n'
        '      "candidate_relation": "the same relation evaluated between each candidate and the shared reference",\n'
        '      "evidence_route": ["ordered source-grounded computation stages"],\n'
        '      "required_sources": ["exact source ID from the manifest"],\n'
        '      "pseudocode": ["ordered steps that exactly implement the declared '
        'reference construction, candidate relation, and evidence route"]\n'
        "    }\n"
        "  ],\n"
        '  "programs": [\n'
        "    {\n"
        '      "strategy_id": "S1",\n'
        '      "code": "complete Python source implementing the exact S1 specification"\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        f"Use strategy IDs {strategy_ids_text} exactly once in each array. Both "
        f"arrays must contain exactly {strategy_count_text} entries, and each program "
        "must implement the specification "
        "with the same strategy_id. "
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

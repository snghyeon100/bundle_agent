"""Dataset-level connectivity map for open-ended evidence discovery.

This graph describes source/entity affordances only. It intentionally contains
no sample-specific item, bundle, or user nodes and is not an allowlist of paths.
Generated code decides how to traverse, compose, or derive relations at runtime.
"""

import json


_GRAPH_CACHE = {}


RELATION_COMPOSITION_NOTE = (
    "Relations may be freely inverted when typing permits, joined through shared typed entities, "
    "expanded from retrieved entities, aggregated, compared, or used to derive new source-grounded relations."
)


def render_affordance_relation_map(graph):
    """Render the machine-readable graph as a compact LLM navigation view."""
    if not isinstance(graph, dict):
        return "(no affordance relations available)"

    entity_names = []
    for entry in graph.get("entity_types", []):
        if isinstance(entry, dict) and str(entry.get("entity", "")).strip():
            entity_names.append(str(entry["entity"]).strip())

    lines = ["ENTITY TYPES: " + ", ".join(entity_names)] if entity_names else []
    lines.append("AVAILABLE TYPED RELATIONS AND RETRIEVAL VIEWS:")

    for source in graph.get("source_affordances", []):
        if not isinstance(source, dict):
            continue
        name = str(source.get("source", "unknown_source"))
        connects = " <-> ".join(str(value) for value in source.get("connects", []) if str(value))
        relations = "; ".join(
            str(value) for value in source.get("recorded_relations", []) if str(value)
        ) or "representation or source-defined relation"
        kind = str(source.get("kind", "source_contract"))
        grounding = str(source.get("grounding", "source-defined grounding"))
        line = f"- {name}: {relations} | connects {connects or 'typed entities'} | {kind} | {grounding}"
        derived_from = [str(value) for value in source.get("derived_from", []) if str(value)]
        if derived_from:
            line += " | derived_from=" + ",".join(derived_from)
        semantics = str(source.get("representation_semantics", "")).strip()
        if semantics:
            line += f" | view={semantics}"
        lines.append(line)

    lines.extend(
        [
            "COMPOSITION: invert typed relations when valid; join through shared entity types; use representations to "
            "retrieve anchors, then continue into observed bundle, user, category, or attribute context.",
            "DEPENDENCY: relations derived from the same recorded source are one evidence family, not independent support.",
            "SEMANTICS: representation similarity is a retrieval bridge, not compatibility evidence by itself.",
        ]
    )
    return "\n".join(lines)


def _source_affordance(source):
    name = str(source.get("name", ""))
    base = {
        "source": name,
        "path": source.get("path", ""),
        "connects": list(source.get("entities", [])),
        "recorded_relations": list(source.get("relations", [])),
        "format": source.get("format", ""),
    }

    if name == "count.json":
        base.update(
            {
                "kind": "dataset_statistics",
                "directionality": "dataset summary to typed entity population counts",
                "grounding": "direct dataset-level summary",
                "risks": ["not candidate-specific by itself"],
            }
        )
    elif name == "item_info.json":
        base.update(
            {
                "kind": "item_attributes",
                "directionality": "item to recorded attributes; reversible only through executed grouping or indexing",
                "grounding": "direct item metadata",
                "risks": [
                    "attribute equality does not by itself establish bundle compatibility",
                    "external IDs and URIs are not canonical integer item IDs",
                ],
            }
        )
    elif name == "bi_train.txt":
        base.update(
            {
                "kind": "historical_bipartite_relation",
                "directionality": "recorded bundle to items; invertible through executed indexing",
                "grounding": "direct train bundle-item observations",
                "risks": ["sparse coverage", "raw frequency and popularity bias"],
            }
        )
    elif name == "ui_full.txt":
        base.update(
            {
                "kind": "historical_bipartite_relation",
                "directionality": "recorded user to items; invertible through executed indexing",
                "grounding": "direct user-item observations",
                "risks": ["sparse coverage", "repeated interactions", "popularity bias"],
            }
        )
    elif name == "content_feature.pt":
        modality = str(source.get("modality", "content"))
        base.update(
            {
                "kind": "item_representation",
                "representation_semantics": modality,
                "directionality": "item to validated representation row; any item-item relation must be derived by execution",
                "grounding": f"indirect {modality}-representation relation",
                "risks": [
                    "similarity can indicate redundancy rather than compatibility",
                    "validate tensor type, shape, row alignment, and norms",
                ],
            }
        )
    elif name == "description_feature.pt":
        base.update(
            {
                "kind": "item_representation",
                "representation_semantics": "description text",
                "directionality": "item to validated representation row; any item-item relation must be derived by execution",
                "grounding": "indirect description-representation relation",
                "risks": [
                    "semantic similarity can indicate redundancy rather than compatibility",
                    "validate tensor type, shape, row alignment, and norms",
                ],
            }
        )
    elif name == "item_cf_feature.pt":
        base.update(
            {
                "kind": "item_representation",
                "representation_semantics": "user-interaction context derived from ui_full.txt",
                "derived_from": ["ui_full.txt"],
                "directionality": "item to validated learned representation row; any item-item relation must be derived by execution",
                "grounding": "indirect learned behavioral relation",
                "risks": [
                    "not independent from ui_full.txt",
                    "popularity and graph-density bias",
                    "validate tensor type, shape, and row alignment",
                ],
            }
        )
    elif name.endswith("_LightGCN_bi_feature.pt"):
        base.update(
            {
                "kind": "item_representation",
                "representation_semantics": "bundle-interaction context derived from bi_train.txt",
                "derived_from": ["bi_train.txt"],
                "directionality": "item to validated learned representation row; any item-item relation must be derived by execution",
                "grounding": "indirect learned bundle-context relation",
                "risks": [
                    "not independent from bi_train.txt",
                    "popularity and graph-density bias",
                    "validate tensor type, shape, and row alignment",
                ],
            }
        )
    else:
        base.update(
            {
                "kind": "source_contract",
                "directionality": "infer only from the recorded source contract and typed entities",
                "grounding": "as described by the source contract",
                "risks": ["inspect and validate the source before relying on derived observations"],
            }
        )
    return base


def build_evidence_affordance_graph(source_manifest, dataset):
    """Build a reusable connectivity map from the currently available sources."""
    sources = source_manifest.get("sources", []) if isinstance(source_manifest, dict) else []
    cache_key = (
        str(dataset),
        json.dumps(sources, ensure_ascii=True, sort_keys=True, default=str),
    )
    if cache_key in _GRAPH_CACHE:
        return _GRAPH_CACHE[cache_key]

    graph = {
        "graph_kind": "dataset_level_source_affordance_connectivity",
        "dataset": str(dataset),
        "reuse_scope": "built once per dataset and available-source contract, then reused across samples",
        "scope": (
            "A soft navigation map over entity types and source capabilities. It contains no actual "
            "sample nodes and does not restrict the generator to listed paths or relations."
        ),
        "entity_types": [
            {
                "entity": "item",
                "identity_rule": "canonical integer item_id",
                "sample_roles": ["partial item", "candidate item", "retrieved item"],
            },
            {"entity": "bundle", "identity_rule": "typed bundle context ID"},
            {"entity": "user", "identity_rule": "typed user context ID"},
            {"entity": "item_attribute", "identity_rule": "metadata value attached to an item"},
            {
                "entity": "item_representation",
                "identity_rule": "validated vector row aligned to canonical item_id",
            },
            {"entity": "dataset_statistics", "identity_rule": "dataset-level aggregate"},
        ],
        "source_affordances": [_source_affordance(source) for source in sources],
        "relation_composition_note": RELATION_COMPOSITION_NOTE,
        "navigation_principles": [
            "Treat this graph as a map of reachable evidence, not a checklist or path allowlist.",
            "Infer the investigation method independently; source descriptions are facts, not retrieval recipes.",
            "Relations may support other derived relations when execution preserves typing and provenance.",
            "Do not stop at the first easy score when another reachable relation could materially reduce uncertainty.",
            "Representation similarity is useful as a retrieval bridge but is not proof of compatibility by itself.",
            "Prefer meaningful uncertainty reduction over path length; a grounded short path can be deeper than a long weak path.",
            "Preserve typed IDs, candidate-symmetric computation, provenance, and train/test leakage boundaries.",
        ],
    }
    _GRAPH_CACHE[cache_key] = graph
    return graph

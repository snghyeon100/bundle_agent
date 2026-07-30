"""Strategy-only hypothesis induction for inspecting macro pseudocode quality."""

from copy import deepcopy

from code.common import parse_json_from_text, pretty_json, task_semantics


STRATEGY_SCHEMA_VERSION = "online_completion_strategy_plans_v1"
STRATEGY_FIELDS = {
    "id",
    "hypothesis",
    "partial_roles",
    "required_sources",
    "pseudocode",
    "retrieval_target",
    "provenance_plan",
}
PARTIAL_ROLE_FIELDS = {"item_id", "role"}


def strategy_generation_prompt(
    *,
    dataset,
    partial_items,
    workspace_manifest,
    strategy_count=3,
):
    """Build a code-free prompt for hypothesis-conditioned macro strategies."""
    partial_view = [
        {
            "item_id": int(item["item_id"]),
            "text": str(item.get("text") or ""),
        }
        for item in partial_items
        if isinstance(item, dict) and "item_id" in item
    ]
    source_catalog = {
        component["id"]: {
            "meaning": component.get("meaning", ""),
            "runtime_format": component.get("runtime_format", ""),
            **(
                {
                    "observed_field_schema": component[
                        "observed_field_schema"
                    ]
                }
                if component.get("observed_field_schema")
                else {}
            ),
        }
        for component in workspace_manifest.get("components", [])
        if isinstance(component, dict) and component.get("id")
    }
    return (
        "You are the Macro Completion Strategy Induction Agent.\n\n"
        f"{task_semantics(dataset)}\n\n"
        "PARTIAL BUNDLE\n"
        f"{pretty_json({'partial_items': partial_view})}\n\n"
        f"Produce exactly {int(strategy_count)} genuinely different interpretations "
        "of how this partial bundle could be completed. For each interpretation, "
        "design a creative and high-level retrieval strategy that would be difficult "
        "to obtain from an obvious one-hop co-occurrence or nearest-neighbor "
        "heuristic.\n\n"
        "Express each strategy as 3 to 8 causally dependent pseudocode steps. Every "
        "step must construct an intermediate reference, transform it, or test a "
        "relation that is used by a later step. The stages together must implement "
        "the hypothesis rather than merely rename one simple retrieval operation. "
        "Use only the available sources below, but combine and transform them freely. "
        "Return structured pseudocode, not Python code.\n\n"
        "AVAILABLE SOURCES\n"
        f"{pretty_json(source_catalog)}\n\n"
        "Return JSON only in this exact structure:\n"
        "{\n"
        f'  "schema_version": "{STRATEGY_SCHEMA_VERSION}",\n'
        '  "strategies": [\n'
        "    {\n"
        '      "id": "S1",\n'
        '      "hypothesis": "latent bundle principle and expected completion relation",\n'
        '      "partial_roles": [\n'
        '        {"item_id": 0, "role": "semantic role assigned to this observed item"}\n'
        "      ],\n"
        '      "required_sources": ["exact available source ID"],\n'
        '      "pseudocode": [\n'
        '        "1. construct a hypothesis-specific reference from the partial roles",\n'
        '        "2. use the previous result to derive a new intermediate relation",\n'
        '        "3. retrieve and prioritize plausible completion exemplars"\n'
        "      ],\n"
        '      "retrieval_target": "what kind of corpus items the strategy should return",\n'
        '      "provenance_plan": "which source records would justify each returned item"\n'
        "    }\n"
        "  ]\n"
        "}\n"
    )


def validate_strategy_result(
    value,
    *,
    available_sources,
    strategy_count=3,
    partial_item_ids=None,
):
    """Validate the strategy-only diagnostic response."""
    if not isinstance(value, dict):
        return ["strategy result must be an object"]
    if set(value) != {"schema_version", "strategies"}:
        return [
            "strategy result must contain exactly schema_version and strategies"
        ]
    issues = []
    if value.get("schema_version") != STRATEGY_SCHEMA_VERSION:
        issues.append(f"schema_version must be {STRATEGY_SCHEMA_VERSION}")
    strategies = value.get("strategies")
    if not isinstance(strategies, list):
        return issues + ["strategies must be a list"]
    if len(strategies) != int(strategy_count):
        issues.append(f"strategies must contain exactly {int(strategy_count)} entries")

    allowed_sources = set(available_sources or [])
    strategy_ids = []
    expected_partial_ids = (
        set(int(item_id) for item_id in partial_item_ids)
        if partial_item_ids is not None
        else None
    )
    normalized_hypotheses = []
    normalized_pseudocode = []
    for index, strategy in enumerate(strategies):
        prefix = f"strategies[{index}]"
        if not isinstance(strategy, dict):
            issues.append(f"{prefix} must be an object")
            continue
        if set(strategy) != STRATEGY_FIELDS:
            issues.append(
                f"{prefix} must contain exactly: "
                + ", ".join(sorted(STRATEGY_FIELDS))
            )
        for field in ("id", "hypothesis", "retrieval_target", "provenance_plan"):
            if not isinstance(strategy.get(field), str) or not strategy[field].strip():
                issues.append(f"{prefix}.{field} must be a non-empty string")
        strategy_ids.append(strategy.get("id"))
        normalized_hypotheses.append(
            " ".join(str(strategy.get("hypothesis") or "").lower().split())
        )

        roles = strategy.get("partial_roles")
        if not isinstance(roles, list) or not roles:
            issues.append(f"{prefix}.partial_roles must be a non-empty list")
        else:
            role_item_ids = []
            for role_index, role in enumerate(roles):
                role_prefix = f"{prefix}.partial_roles[{role_index}]"
                if not isinstance(role, dict) or set(role) != PARTIAL_ROLE_FIELDS:
                    issues.append(
                        f"{role_prefix} must contain exactly item_id and role"
                    )
                    continue
                item_id = role.get("item_id")
                if not isinstance(item_id, int) or isinstance(item_id, bool):
                    issues.append(f"{role_prefix}.item_id must be an integer")
                else:
                    role_item_ids.append(item_id)
                if not isinstance(role.get("role"), str) or not role["role"].strip():
                    issues.append(f"{role_prefix}.role must be a non-empty string")
            if len(role_item_ids) != len(set(role_item_ids)):
                issues.append(f"{prefix}.partial_roles item_id values must be unique")
            if (
                expected_partial_ids is not None
                and set(role_item_ids) != expected_partial_ids
            ):
                issues.append(
                    f"{prefix}.partial_roles must cover exactly the partial item IDs"
                )

        sources = strategy.get("required_sources")
        if not isinstance(sources, list) or not sources:
            issues.append(f"{prefix}.required_sources must be a non-empty list")
        elif not all(isinstance(source, str) for source in sources):
            issues.append(f"{prefix}.required_sources must contain strings")
        else:
            unknown = sorted(set(sources) - allowed_sources)
            if unknown:
                issues.append(
                    f"{prefix}.required_sources contains unavailable sources: "
                    + ", ".join(unknown)
                )

        steps = strategy.get("pseudocode")
        if (
            not isinstance(steps, list)
            or not 3 <= len(steps) <= 8
            or not all(isinstance(step, str) and step.strip() for step in steps)
        ):
            issues.append(
                f"{prefix}.pseudocode must contain 3 to 8 non-empty strings"
            )
        else:
            normalized_pseudocode.append(
                tuple(" ".join(step.lower().split()) for step in steps)
            )
            code_markers = ("def retrieve(", "import ", "return {", "return [")
            if any(
                marker in step.lower()
                for step in steps
                for marker in code_markers
            ):
                issues.append(f"{prefix}.pseudocode must not contain Python code")
    if len(strategy_ids) != len(set(strategy_ids)):
        issues.append("strategy id values must be unique")
    if (
        normalized_hypotheses
        and len(normalized_hypotheses) != len(set(normalized_hypotheses))
    ):
        issues.append("strategy hypotheses must be distinct")
    if (
        normalized_pseudocode
        and len(normalized_pseudocode) != len(set(normalized_pseudocode))
    ):
        issues.append("strategy pseudocode plans must be distinct")
    return list(dict.fromkeys(issues))


def parse_strategy_response(raw_text):
    """Parse one strategy-only JSON response."""
    value = parse_json_from_text(raw_text)
    return deepcopy(value) if isinstance(value, dict) else value

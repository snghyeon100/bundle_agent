"""Raw read-only dataset workspace for generated online verifier programs."""

import json
import os

from .source_api import SOURCE_DESCRIPTIONS, SOURCE_FILES, _parse_relation_file


EMBEDDING_SOURCES = {
    "item_content_embedding",
    "item_description_embedding",
    "user_collaborative_embedding",
    "bundle_collaborative_embedding",
}

WORKSPACE_CONTRACTS = {
    "dataset_statistics": {
        "source_record_format": (
            'one JSON object such as {"#U": user_count, "#B": bundle_count, '
            '"#I": item_count}'
        ),
        "runtime_format": (
            "read-only mapping {statistic_name: integer count}"
        ),
    },
    "item_metadata": {
        "source_record_format": (
            'one JSON object {"<item_id as string>": {metadata fields...}}'
        ),
        "runtime_format": (
            "read-only mapping {item_id:int: read-only metadata mapping}; "
            "the manifest also lists observed metadata fields and value types"
        ),
    },
    "bundle_item_history": {
        "source_record_format": (
            "one CSV line: bundle_id, item_id_1, item_id_2, ..."
        ),
        "runtime_format": (
            "read-only mapping with bundles_to_items "
            "{bundle_id:int: tuple[item_id:int, ...]} and the reverse index "
            "items_to_bundles {item_id:int: tuple[bundle_id:int, ...]}"
        ),
    },
    "user_item_history": {
        "source_record_format": (
            "one CSV line: user_id, item_id_1, item_id_2, ..."
        ),
        "runtime_format": (
            "read-only mapping with users_to_items "
            "{user_id:int: tuple[item_id:int, ...]} and the reverse index "
            "items_to_users {item_id:int: tuple[user_id:int, ...]}"
        ),
    },
    "item_content_embedding": {
        "source_record_format": "serialized rank-2 tensor",
        "runtime_format": (
            "read-only-by-contract rank-2 CPU float tensor; row index is item_id"
        ),
    },
    "item_description_embedding": {
        "source_record_format": "serialized rank-2 tensor",
        "runtime_format": (
            "read-only-by-contract rank-2 CPU float tensor; row index is item_id"
        ),
    },
    "user_collaborative_embedding": {
        "source_record_format": "serialized rank-2 tensor",
        "runtime_format": (
            "read-only-by-contract rank-2 CPU float tensor; row index is item_id"
        ),
    },
    "bundle_collaborative_embedding": {
        "source_record_format": "serialized rank-2 tensor",
        "runtime_format": (
            "read-only-by-contract rank-2 CPU float tensor; row index is item_id"
        ),
    },
}


class FrozenDict(dict):
    """A dict-compatible mapping that rejects mutation from generated code."""

    def _immutable(self, *args, **kwargs):
        raise TypeError("raw dataset workspace values are read-only")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable


class ItemMetadataDict(FrozenDict):
    """Read-only metadata mapping with tolerant numeric-string lookup."""

    @staticmethod
    def _normalized_key(key):
        if isinstance(key, str):
            try:
                return int(key)
            except ValueError:
                return key
        return key

    def __getitem__(self, key):
        return dict.__getitem__(self, self._normalized_key(key))

    def get(self, key, default=None):
        return dict.get(self, self._normalized_key(key), default)

    def __contains__(self, key):
        return dict.__contains__(self, self._normalized_key(key))


def available_workspace_sources(conf):
    """Return source component IDs backed by files in the configured dataset."""
    dataset = str(conf["dataset"])
    data_dir = os.path.abspath(os.path.join(conf["data_path"], dataset))
    return tuple(
        source_id
        for source_id, raw_filename in SOURCE_FILES.items()
        if os.path.isfile(
            os.path.join(data_dir, raw_filename.replace("{dataset}", dataset))
        )
    )


def _freeze_mapping(mapping):
    return FrozenDict(
        {
            key: (
                _freeze_mapping(value)
                if isinstance(value, dict)
                else tuple(value)
                if isinstance(value, list)
                else value
            )
            for key, value in mapping.items()
        }
    )


def _load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_embedding(path, source_id):
    import torch

    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, torch.Tensor) or value.ndim != 2:
        raise ValueError(f"{source_id} must contain a rank-2 torch tensor")
    return value.detach().float().cpu()


def _value_type_name(value):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _metadata_field_schema(path):
    """Describe observed metadata keys without exposing item values."""
    raw = _load_json(path)
    field_types = {}
    for metadata in raw.values():
        if not isinstance(metadata, dict):
            continue
        for field, value in metadata.items():
            field_types.setdefault(str(field), set()).add(
                _value_type_name(value)
            )
    return {
        field: sorted(types)
        for field, types in sorted(field_types.items())
    }


def build_dataset_workspace(conf, *, allowed_sources):
    """Materialize selected raw components without prescribing query operations."""
    dataset = str(conf["dataset"])
    data_dir = os.path.abspath(os.path.join(conf["data_path"], dataset))
    available = set(available_workspace_sources(conf))
    requested = set(allowed_sources or [])
    unknown = requested - available
    if unknown:
        raise ValueError(
            "unavailable raw workspace sources: " + ", ".join(sorted(unknown))
        )

    count_path = os.path.join(data_dir, "count.json")
    metadata_path = os.path.join(data_dir, "item_info.json")
    if os.path.isfile(count_path):
        statistics = _load_json(count_path)
        item_ids = tuple(range(int(statistics.get("#I", 0))))
    else:
        statistics = {}
        metadata = _load_json(metadata_path)
        item_ids = tuple(sorted(int(item_id) for item_id in metadata))

    workspace = {"item_ids": item_ids}
    for source_id in SOURCE_FILES:
        if source_id not in requested:
            continue
        filename = SOURCE_FILES[source_id].replace("{dataset}", dataset)
        path = os.path.join(data_dir, filename)
        if source_id == "dataset_statistics":
            workspace[source_id] = _freeze_mapping(_load_json(path))
        elif source_id == "item_metadata":
            raw = _load_json(path)
            workspace[source_id] = ItemMetadataDict(
                {
                    int(item_id): _freeze_mapping(value)
                    for item_id, value in raw.items()
                }
            )
        elif source_id == "bundle_item_history":
            forward, reverse = _parse_relation_file(path)
            workspace[source_id] = _freeze_mapping(
                {
                    "bundles_to_items": forward,
                    "items_to_bundles": reverse,
                }
            )
        elif source_id == "user_item_history":
            forward, reverse = _parse_relation_file(path)
            workspace[source_id] = _freeze_mapping(
                {
                    "users_to_items": forward,
                    "items_to_users": reverse,
                }
            )
        elif source_id in EMBEDDING_SOURCES:
            workspace[source_id] = _load_embedding(path, source_id)
    return FrozenDict(workspace)


def dataset_workspace_manifest(conf):
    """Describe raw values available to LLM-generated programs."""
    available = available_workspace_sources(conf)
    data_dir = os.path.abspath(os.path.join(conf["data_path"], str(conf["dataset"])))
    components = []
    for source_id in available:
        component = {
            "id": source_id,
            "meaning": SOURCE_DESCRIPTIONS[source_id],
            **WORKSPACE_CONTRACTS[source_id],
        }
        if source_id == "item_metadata":
            component["observed_field_schema"] = _metadata_field_schema(
                os.path.join(data_dir, SOURCE_FILES[source_id])
            )
        components.append(component)
    return {
        "dataset": str(conf["dataset"]),
        "binding": (
            "dataset_workspace is a read-only mapping containing item_ids plus only "
            "the components declared in required_sources"
        ),
        "always_available": {
            "item_ids": "tuple of every canonical integer corpus item_id",
        },
        "components": components,
        "freedom": [
            "No retrieval or graph-traversal methods are prescribed.",
            "Programs may build their own indexes, joins, aggregates, neighborhoods, "
            "normalizations, comparisons, and evidence tests from the raw values.",
            "Raw components must be treated as immutable.",
        ],
        "typed_id_rules": [
            "Bundle, user, and item IDs are distinct entity types even when integers match.",
            "Only item IDs index item metadata and item embedding rows.",
        ],
    }

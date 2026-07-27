"""Read-only dataset source API exposed to generated online programs."""

import json
import os

from operator_learning.runtime import SAFE_IMPORT_ROOTS


SOURCE_FILES = {
    "dataset_statistics": "count.json",
    "item_metadata": "item_info.json",
    "bundle_item_history": "bi_train.txt",
    "user_item_history": "ui_full.txt",
    "item_content_embedding": "content_feature.pt",
    "item_description_embedding": "description_feature.pt",
    "user_collaborative_embedding": "item_cf_feature.pt",
    "bundle_collaborative_embedding": "{dataset}_LightGCN_bi_feature.pt",
}

SOURCE_DESCRIPTIONS = {
    "dataset_statistics": "Dataset-level bundle, user, item, and relation counts.",
    "item_metadata": "Canonical item text and opaque category identifiers.",
    "bundle_item_history": "Historical bundle-to-item membership relations.",
    "user_item_history": "Historical user-to-item interaction relations.",
    "item_content_embedding": "Precomputed item-level content embedding vectors.",
    "item_description_embedding": "Precomputed item-description embedding vectors.",
    "user_collaborative_embedding": "Item embeddings learned from user-item relations.",
    "bundle_collaborative_embedding": "Item embeddings learned from bundle-item relations.",
}

SOURCE_API_METHODS = {
    "available_sources": "tuple of source IDs available to this execution",
    "get_all_item_ids()": "return all canonical corpus item IDs",
    "get_dataset_statistics()": "return dataset statistics",
    "get_item_metadata(item_ids)": (
        "return {int item_id: metadata object}; category IDs are opaque"
    ),
    "search_item_text(query_terms, limit)": (
        "return bounded item IDs whose stored text contains explicit query terms"
    ),
    "get_bundles_for_items(item_ids)": (
        "return {int item_id: [bundle_id, ...]} from training bundle history"
    ),
    "get_items_for_bundles(bundle_ids)": (
        "return {int bundle_id: [item_id, ...]} from training bundle history"
    ),
    "get_users_for_items(item_ids)": (
        "return {int item_id: [user_id, ...]} from user history"
    ),
    "get_items_for_users(user_ids)": (
        "return {int user_id: [item_id, ...]} from user history"
    ),
    "get_item_embeddings(item_ids, source_id)": (
        "return a bounded {int item_id: [float, ...]} embedding mapping"
    ),
    "nearest_item_neighbors(anchor_item_ids, source_id, limit, exclude_item_ids=None)": (
        "return bounded [{item_id, similarity}, ...] neighbors of the anchor centroid"
    ),
}


def _parse_relation_file(path):
    forward = {}
    reverse = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            values = [
                int(token.strip())
                for token in line.split(",")
                if token.strip()
            ]
            if not values:
                continue
            owner_id, item_ids = values[0], values[1:]
            forward[owner_id] = item_ids
            for item_id in item_ids:
                reverse.setdefault(item_id, []).append(owner_id)
    return forward, reverse


class DatasetSourceAPI:
    """Lazy read-only implementation used by generated online search programs."""

    def __init__(self, conf, *, allowed_sources=None):
        self.dataset = str(conf["dataset"])
        self.data_dir = os.path.abspath(
            os.path.join(conf["data_path"], self.dataset)
        )
        available = []
        for source_id, raw_filename in SOURCE_FILES.items():
            filename = raw_filename.replace("{dataset}", self.dataset)
            if os.path.isfile(os.path.join(self.data_dir, filename)):
                available.append(source_id)
        requested = set(allowed_sources or available)
        unknown = requested - set(available)
        if unknown:
            raise ValueError(
                "unavailable source IDs: " + ", ".join(sorted(unknown))
            )
        self._available_sources = tuple(
            source_id for source_id in available if source_id in requested
        )
        self.max_query_ids = int(conf.get("online_source_max_query_ids", 5000))
        self.max_embedding_items = int(
            conf.get("online_source_max_embedding_items", 2048)
        )
        self.max_neighbor_limit = int(
            conf.get("online_source_max_neighbor_limit", 200)
        )
        self._statistics = None
        self._metadata = None
        self._bundle_forward = None
        self._bundle_reverse = None
        self._user_forward = None
        self._user_reverse = None
        self._embeddings = {}

    @property
    def available_sources(self):
        return self._available_sources

    def _require(self, source_id):
        if source_id not in self._available_sources:
            raise PermissionError(f"source is not scoped to this program: {source_id}")

    def _bounded_ids(self, values, field):
        if not isinstance(values, (list, tuple, set)):
            raise TypeError(f"{field} must be a list, tuple, or set")
        result = [int(value) for value in values]
        if len(result) > self.max_query_ids:
            raise ValueError(f"{field} exceeds source query limit")
        return result

    def _load_statistics(self):
        self._require("dataset_statistics")
        if self._statistics is None:
            path = os.path.join(self.data_dir, "count.json")
            with open(path, "r", encoding="utf-8") as handle:
                self._statistics = json.load(handle)
        return self._statistics

    def _load_metadata(self):
        self._require("item_metadata")
        if self._metadata is None:
            path = os.path.join(self.data_dir, "item_info.json")
            with open(path, "r", encoding="utf-8") as handle:
                self._metadata = json.load(handle)
        return self._metadata

    def _load_bundle_relations(self):
        self._require("bundle_item_history")
        if self._bundle_forward is None:
            path = os.path.join(self.data_dir, "bi_train.txt")
            self._bundle_forward, self._bundle_reverse = _parse_relation_file(path)

    def _load_user_relations(self):
        self._require("user_item_history")
        if self._user_forward is None:
            path = os.path.join(self.data_dir, "ui_full.txt")
            self._user_forward, self._user_reverse = _parse_relation_file(path)

    def _load_embedding(self, source_id):
        self._require(source_id)
        if source_id not in {
            "item_content_embedding",
            "item_description_embedding",
            "user_collaborative_embedding",
            "bundle_collaborative_embedding",
        }:
            raise ValueError(f"source is not an item embedding: {source_id}")
        if source_id not in self._embeddings:
            import torch

            filename = SOURCE_FILES[source_id].replace("{dataset}", self.dataset)
            path = os.path.join(self.data_dir, filename)
            value = torch.load(path, map_location="cpu", weights_only=False)
            if not isinstance(value, torch.Tensor) or value.ndim != 2:
                raise ValueError(f"{source_id} must contain a rank-2 torch tensor")
            self._embeddings[source_id] = value.detach().float().cpu()
        return self._embeddings[source_id]

    def get_all_item_ids(self):
        # The canonical item universe is part of the execution boundary rather
        # than evidence. It remains available even when a program scopes itself
        # to one relational or embedding source.
        count_path = os.path.join(self.data_dir, "count.json")
        if os.path.isfile(count_path):
            with open(count_path, "r", encoding="utf-8") as handle:
                statistics = json.load(handle)
            if "#I" in statistics:
                return list(range(int(statistics["#I"])))
        metadata_path = os.path.join(self.data_dir, "item_info.json")
        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        return sorted(int(item_id) for item_id in metadata)

    def get_dataset_statistics(self):
        return dict(self._load_statistics())

    def get_item_metadata(self, item_ids):
        ids = self._bounded_ids(item_ids, "item_ids")
        metadata = self._load_metadata()
        return {
            item_id: dict(metadata.get(str(item_id), {}))
            for item_id in ids
            if str(item_id) in metadata
        }

    def item_text(self, item_id):
        metadata = self._load_metadata().get(str(int(item_id)), {})
        if "spotify" in self.dataset.lower():
            parts = [
                metadata.get("track_name"),
                metadata.get("artist_name"),
                metadata.get("album_name"),
            ]
            text = " - ".join(str(value) for value in parts if value)
        else:
            text = str(metadata.get("title") or "")
        return " ".join(text.split()) or f"Item {int(item_id)}"

    def search_item_text(self, query_terms, limit):
        self._require("item_metadata")
        if not _string_terms(query_terms):
            raise ValueError("query_terms must contain non-empty strings")
        bounded_limit = max(0, min(int(limit), self.max_neighbor_limit))
        terms = [str(term).strip().casefold() for term in query_terms]
        matches = []
        for raw_item_id in self._load_metadata():
            item_id = int(raw_item_id)
            text = self.item_text(item_id).casefold()
            matched = sum(term in text for term in terms)
            if matched:
                matches.append((-matched, item_id))
        matches.sort()
        return [item_id for _, item_id in matches[:bounded_limit]]

    def get_bundles_for_items(self, item_ids):
        ids = self._bounded_ids(item_ids, "item_ids")
        self._load_bundle_relations()
        return {
            item_id: list(self._bundle_reverse.get(item_id, []))
            for item_id in ids
        }

    def get_items_for_bundles(self, bundle_ids):
        ids = self._bounded_ids(bundle_ids, "bundle_ids")
        self._load_bundle_relations()
        return {
            bundle_id: list(self._bundle_forward.get(bundle_id, []))
            for bundle_id in ids
        }

    def get_users_for_items(self, item_ids):
        ids = self._bounded_ids(item_ids, "item_ids")
        self._load_user_relations()
        return {
            item_id: list(self._user_reverse.get(item_id, []))
            for item_id in ids
        }

    def get_items_for_users(self, user_ids):
        ids = self._bounded_ids(user_ids, "user_ids")
        self._load_user_relations()
        return {
            user_id: list(self._user_forward.get(user_id, []))
            for user_id in ids
        }

    def get_item_embeddings(self, item_ids, source_id):
        ids = self._bounded_ids(item_ids, "item_ids")
        if len(ids) > self.max_embedding_items:
            raise ValueError("item_ids exceeds embedding materialization limit")
        tensor = self._load_embedding(str(source_id))
        return {
            item_id: tensor[item_id].tolist()
            for item_id in ids
            if 0 <= item_id < tensor.shape[0]
        }

    def nearest_item_neighbors(
        self,
        anchor_item_ids,
        source_id,
        limit,
        exclude_item_ids=None,
    ):
        import torch

        anchors = self._bounded_ids(anchor_item_ids, "anchor_item_ids")
        if not anchors:
            return []
        bounded_limit = max(0, min(int(limit), self.max_neighbor_limit))
        if bounded_limit == 0:
            return []
        tensor = self._load_embedding(str(source_id))
        valid = [item_id for item_id in anchors if 0 <= item_id < tensor.shape[0]]
        if not valid:
            return []
        anchor = tensor[valid].mean(dim=0)
        anchor = anchor / anchor.norm().clamp_min(1e-12)
        normalized = tensor / tensor.norm(dim=1, keepdim=True).clamp_min(1e-12)
        similarities = normalized @ anchor
        excluded = {
            int(item_id)
            for item_id in (exclude_item_ids or [])
            if 0 <= int(item_id) < tensor.shape[0]
        }
        excluded.update(valid)
        if excluded:
            index = torch.tensor(sorted(excluded), dtype=torch.long)
            similarities[index] = float("-inf")
        top_count = min(
            bounded_limit,
            max(0, int(tensor.shape[0]) - len(excluded)),
        )
        if top_count == 0:
            return []
        values, indices = torch.topk(similarities, k=top_count)
        return [
            {"item_id": int(item_id), "similarity": float(similarity)}
            for item_id, similarity in zip(indices.tolist(), values.tolist())
        ]

    def diagnostics(self, partial_item_ids):
        partial_ids = [int(item_id) for item_id in partial_item_ids]
        result = {
            "partial_item_count": len(partial_ids),
            "available_sources": list(self.available_sources),
            "component_diagnostics": {},
        }
        if "item_metadata" in self.available_sources:
            metadata = self.get_item_metadata(partial_ids)
            result["component_diagnostics"]["item_metadata"] = {
                "covered_partial_items": len(metadata),
                "total_partial_items": len(partial_ids),
            }
        if "bundle_item_history" in self.available_sources:
            mapping = self.get_bundles_for_items(partial_ids)
            bundles = sorted(
                {
                    bundle_id
                    for values in mapping.values()
                    for bundle_id in values
                }
            )
            bounded_bundles = bundles[: self.max_query_ids]
            related = {
                item_id
                for values in self.get_items_for_bundles(bounded_bundles).values()
                for item_id in values
            } - set(partial_ids)
            result["component_diagnostics"]["bundle_item_history"] = {
                "covered_partial_items": sum(bool(mapping[item]) for item in partial_ids),
                "related_record_count": len(bundles),
                "retrievable_non_partial_item_count": len(related),
                "diagnostic_record_budget": len(bounded_bundles),
            }
        if "user_item_history" in self.available_sources:
            mapping = self.get_users_for_items(partial_ids)
            users = sorted(
                {
                    user_id
                    for values in mapping.values()
                    for user_id in values
                }
            )
            bounded_users = users[: self.max_query_ids]
            related = {
                item_id
                for values in self.get_items_for_users(bounded_users).values()
                for item_id in values
            } - set(partial_ids)
            result["component_diagnostics"]["user_item_history"] = {
                "covered_partial_items": sum(bool(mapping[item]) for item in partial_ids),
                "related_record_count": len(users),
                "retrievable_non_partial_item_count": len(related),
                "diagnostic_record_budget": len(bounded_users),
            }
        for source_id in (
            "item_content_embedding",
            "item_description_embedding",
            "user_collaborative_embedding",
            "bundle_collaborative_embedding",
        ):
            if source_id in self.available_sources:
                result["component_diagnostics"][source_id] = {
                    "covered_partial_items": len(partial_ids),
                    "total_partial_items": len(partial_ids),
                }
        return result


def _string_terms(value):
    return (
        isinstance(value, (list, tuple))
        and bool(value)
        and all(isinstance(item, str) and item.strip() for item in value)
    )


def source_capability_manifest(conf):
    """Return the exact read-only surface described to LLM1."""
    api = DatasetSourceAPI(conf)
    return {
        "dataset": str(conf["dataset"]),
        "components": [
            {
                "id": source_id,
                "description": SOURCE_DESCRIPTIONS[source_id],
            }
            for source_id in api.available_sources
        ],
        "source_api": dict(SOURCE_API_METHODS),
        "runtime_limits": {
            "safe_import_roots": sorted(SAFE_IMPORT_ROOTS),
            "maximum_IDs_per_relation_query": int(
                conf.get("online_source_max_query_ids", 5000)
            ),
            "maximum_materialized_embedding_items": int(
                conf.get("online_source_max_embedding_items", 2048)
            ),
            "maximum_neighbor_or_text_search_limit": int(
                conf.get("online_source_max_neighbor_limit", 200)
            ),
        },
    }

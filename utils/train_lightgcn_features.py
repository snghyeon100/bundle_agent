import argparse
import json
import math
import os
import random
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device):
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_count_json(dataset_dir):
    with open(os.path.join(dataset_dir, "count.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def load_interactions(path):
    pairs = []
    positives = defaultdict(set)
    max_context_id = -1
    max_item_id = -1

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            values = [int(value) for value in line.split(", ")]
            context_id = values[0]
            max_context_id = max(max_context_id, context_id)
            for item_id in values[1:]:
                pairs.append((context_id, item_id))
                positives[context_id].add(item_id)
                max_item_id = max(max_item_id, item_id)

    if not pairs:
        raise ValueError(f"No interactions found in {path}")

    return {
        "pairs": np.asarray(pairs, dtype=np.int64),
        "positives": positives,
        "max_context_id": max_context_id,
        "max_item_id": max_item_id,
    }


def build_normalized_interaction_matrix(pairs, num_contexts, num_items, device):
    context_ids = pairs[:, 0]
    item_ids = pairs[:, 1]

    context_degree = np.bincount(context_ids, minlength=num_contexts).astype(np.float32)
    item_degree = np.bincount(item_ids, minlength=num_items).astype(np.float32)
    context_degree[context_degree == 0] = 1.0
    item_degree[item_degree == 0] = 1.0

    values = 1.0 / np.sqrt(context_degree[context_ids] * item_degree[item_ids])
    indices = torch.from_numpy(np.vstack([context_ids, item_ids]).astype(np.int64))
    values = torch.from_numpy(values.astype(np.float32))
    matrix = torch.sparse_coo_tensor(
        indices,
        values,
        size=(num_contexts, num_items),
        dtype=torch.float32,
        device=device,
    )
    return matrix.coalesce()


class LightGCN(nn.Module):
    def __init__(self, num_contexts, num_items, embedding_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.context_embedding = nn.Embedding(num_contexts, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        nn.init.xavier_uniform_(self.context_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def propagate(self, norm_matrix):
        context_emb = self.context_embedding.weight
        item_emb = self.item_embedding.weight

        context_sum = context_emb
        item_sum = item_emb

        for _ in range(self.num_layers):
            next_context = torch.sparse.mm(norm_matrix, item_emb)
            next_item = torch.sparse.mm(norm_matrix.transpose(0, 1), context_emb)
            context_emb, item_emb = next_context, next_item
            context_sum = context_sum + context_emb
            item_sum = item_sum + item_emb

        final_context = context_sum / (self.num_layers + 1)
        final_item = item_sum / (self.num_layers + 1)
        return final_context, final_item


def sample_negative_items(context_ids, positives, num_items, rng):
    negatives = np.empty(len(context_ids), dtype=np.int64)
    for idx, context_id in enumerate(context_ids):
        positive_items = positives.get(int(context_id), set())
        negative = int(rng.integers(0, num_items))
        while negative in positive_items:
            negative = int(rng.integers(0, num_items))
        negatives[idx] = negative
    return negatives


def bpr_loss(context_emb, positive_item_emb, negative_item_emb):
    positive_scores = torch.sum(context_emb * positive_item_emb, dim=1)
    negative_scores = torch.sum(context_emb * negative_item_emb, dim=1)
    return -F.logsigmoid(positive_scores - negative_scores).mean()


def train_lightgcn(graph_name, interaction_path, num_contexts, num_items, args, output_dir):
    print(f"[{graph_name}] Loading interactions from {interaction_path}")
    loaded = load_interactions(interaction_path)
    pairs = loaded["pairs"]
    positives = loaded["positives"]

    if loaded["max_context_id"] >= num_contexts:
        raise ValueError(
            f"{graph_name}: max context id {loaded['max_context_id']} >= configured count {num_contexts}"
        )
    if loaded["max_item_id"] >= num_items:
        raise ValueError(f"{graph_name}: max item id {loaded['max_item_id']} >= configured item count {num_items}")

    if args.max_train_edges > 0 and len(pairs) > args.max_train_edges:
        rng = np.random.default_rng(args.seed)
        sampled_idx = rng.choice(len(pairs), size=args.max_train_edges, replace=False)
        pairs = pairs[sampled_idx]
        positives = defaultdict(set)
        for context_id, item_id in pairs:
            positives[int(context_id)].add(int(item_id))
        print(f"[{graph_name}] Sampled {len(pairs)} train edges because --max-train-edges was set")

    device = resolve_device(args.device)
    norm_matrix = build_normalized_interaction_matrix(pairs, num_contexts, num_items, device)
    model = LightGCN(num_contexts, num_items, args.embedding_dim, args.num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed)

    print(
        f"[{graph_name}] contexts={num_contexts} items={num_items} edges={len(pairs)} "
        f"dim={args.embedding_dim} layers={args.num_layers} device={device}"
    )

    start_time = time.time()
    num_batches = math.ceil(len(pairs) / args.batch_size)
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        order = rng.permutation(len(pairs))
        total_loss = 0.0

        for batch_idx in range(num_batches):
            batch_indices = order[batch_idx * args.batch_size : (batch_idx + 1) * args.batch_size]
            batch = pairs[batch_indices]
            context_ids_np = batch[:, 0]
            positive_ids_np = batch[:, 1]
            negative_ids_np = sample_negative_items(context_ids_np, positives, num_items, rng)

            context_ids = torch.from_numpy(context_ids_np).to(device)
            positive_ids = torch.from_numpy(positive_ids_np).to(device)
            negative_ids = torch.from_numpy(negative_ids_np).to(device)

            final_context, final_item = model.propagate(norm_matrix)
            loss = bpr_loss(final_context[context_ids], final_item[positive_ids], final_item[negative_ids])

            reg_loss = (
                model.context_embedding(context_ids).norm(2).pow(2)
                + model.item_embedding(positive_ids).norm(2).pow(2)
                + model.item_embedding(negative_ids).norm(2).pow(2)
            ) / len(context_ids)
            loss = loss + args.weight_decay * reg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())

        avg_loss = total_loss / max(num_batches, 1)
        print(f"[{graph_name}] epoch={epoch:03d} loss={avg_loss:.6f} time={time.time() - epoch_start:.1f}s")

    model.eval()
    with torch.no_grad():
        final_context, final_item = model.propagate(norm_matrix)

    os.makedirs(output_dir, exist_ok=True)
    item_path = os.path.join(output_dir, f"{graph_name}_item_embeddings.pt")
    context_path = os.path.join(output_dir, f"{graph_name}_context_embeddings.pt")
    torch.save(final_item.cpu(), item_path)
    if args.save_context_embeddings:
        torch.save(final_context.cpu(), context_path)

    metadata = {
        "graph": graph_name,
        "interaction_path": os.path.abspath(interaction_path),
        "num_contexts": num_contexts,
        "num_items": num_items,
        "num_edges": int(len(pairs)),
        "embedding_dim": args.embedding_dim,
        "num_layers": args.num_layers,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "init": "xavier_uniform",
        "loss": "bpr",
        "negative_sampling": "uniform_unobserved_item_per_positive_edge",
        "seed": args.seed,
        "device": str(device),
        "item_embedding_path": os.path.abspath(item_path),
        "context_embedding_path": os.path.abspath(context_path) if args.save_context_embeddings else "",
        "elapsed_seconds": round(time.time() - start_time, 3),
    }
    metadata_path = os.path.join(output_dir, f"{graph_name}_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"[{graph_name}] Saved item embeddings to {item_path}")
    if args.save_context_embeddings:
        print(f"[{graph_name}] Saved context embeddings to {context_path}")
    print(f"[{graph_name}] Saved metadata to {metadata_path}")
    return metadata


def parse_args():
    parser = argparse.ArgumentParser(description="Train train-controlled LightGCN item features for UI and BI graphs.")
    parser.add_argument("--dataset", default="pog_dense", help="Dataset folder name under --data-path")
    parser.add_argument("--data-path", default="./datasets", help="Root directory containing dataset folders")
    parser.add_argument("--output-root", default="", help="Output root; default: <dataset_dir>/lightgcn_self")
    parser.add_argument("--graphs", nargs="+", default=["ui", "bi"], choices=["ui", "bi"], help="Graphs to train")
    parser.add_argument("--ui-file", default="ui_full.txt", help="User-item interaction file")
    parser.add_argument("--bi-file", default="bi_train.txt", help="Bundle-item train affiliation file")
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:<id>")
    parser.add_argument("--max-train-edges", type=int, default=-1, help="Debug option: sample at most this many edges")
    parser.add_argument("--save-context-embeddings", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    dataset_dir = os.path.join(args.data_path, args.dataset)
    counts = load_count_json(dataset_dir)
    num_items = int(counts["#I"])
    output_root = args.output_root or os.path.join(dataset_dir, "lightgcn_self")
    os.makedirs(output_root, exist_ok=True)

    all_metadata = {
        "dataset": args.dataset,
        "data_path": os.path.abspath(args.data_path),
        "output_root": os.path.abspath(output_root),
        "leakage_policy": {
            "ui": "uses configured UI interaction file; default ui_full.txt",
            "bi": "uses train split only by default; default bi_train.txt",
            "excluded_by_default": ["bi_full.txt", "bi_test_gt.txt", "bi_valid_gt.txt"],
        },
        "graphs": {},
    }

    if "ui" in args.graphs:
        ui_path = os.path.join(dataset_dir, args.ui_file)
        all_metadata["graphs"]["ui"] = train_lightgcn(
            "ui",
            ui_path,
            int(counts["#U"]),
            num_items,
            args,
            output_root,
        )

    if "bi" in args.graphs:
        bi_path = os.path.join(dataset_dir, args.bi_file)
        all_metadata["graphs"]["bi"] = train_lightgcn(
            "bi",
            bi_path,
            int(counts["#B"]),
            num_items,
            args,
            output_root,
        )

    summary_path = os.path.join(output_root, "metadata.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, ensure_ascii=False, indent=2)
    print(f"Saved combined metadata to {summary_path}")


if __name__ == "__main__":
    main()

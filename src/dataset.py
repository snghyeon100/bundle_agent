import json
import os
import random

import numpy as np
import scipy.sparse as sp


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def list2pairs(path):
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            values = [int(value) for value in line.strip().split(", ")]
            bundle_id = values[0]
            for item_id in values[1:]:
                pairs.append([bundle_id, item_id])
    return np.array(pairs, dtype=np.int32)


def pairs2csr(pairs, shape):
    values = np.ones(len(pairs), dtype=np.float32)
    return sp.csr_matrix((values, (pairs[:, 0], pairs[:, 1])), shape=shape)


class BundleZeroShotDataset:
    def __init__(self, conf):
        self.path = conf["data_path"]
        self.name = conf["dataset"]
        self.num_cans = int(conf["num_cans"])
        self.toy_eval = int(conf["toy_eval"])
        self.num_token = int(conf["num_token"])
        self.seed = int(conf.get("seed", 45))
        self.shuffle_seed = int(conf.get("shuffle_seed", 45))

        count_path = os.path.join(self.path, self.name, "count.json")
        with open(count_path, "r", encoding="utf-8") as f:
            stat = json.load(f)
        self.num_bundles = int(stat["#B"])
        self.num_items = int(stat["#I"])

        info_path = os.path.join(self.path, self.name, "item_info.json")
        with open(info_path, "r", encoding="utf-8") as f:
            self.item_info = json.load(f)

        input_path = os.path.join(self.path, self.name, "bi_test_input.txt")
        gt_path = os.path.join(self.path, self.name, "bi_test_gt.txt")
        self.b_i_pairs_i = list2pairs(input_path)
        self.b_i_pairs_gt = list2pairs(gt_path)
        np.random.shuffle(self.b_i_pairs_gt)

        shape = (self.num_bundles, self.num_items)
        self.b_i_graph_i = pairs2csr(self.b_i_pairs_i, shape)
        # Match Bundle_zero: use every test GT item when excluding false candidates.
        self.b_i_graph_gt = pairs2csr(self.b_i_pairs_gt, shape)

        if self.toy_eval > 0:
            self.b_i_pairs_gt = self.b_i_pairs_gt[: self.toy_eval]

    def get_item_text(self, item_id):
        item_id_str = str(int(item_id))
        if "pog" in self.name:
            return self.item_info[item_id_str].get("title", f"Item {item_id_str}")
        if "spotify" in self.name:
            info = self.item_info[item_id_str]
            parts = []
            if info.get("track_name"):
                parts.append(info["track_name"])
            if info.get("artist_name"):
                parts.append(info["artist_name"])
            if info.get("album_name"):
                parts.append(info["album_name"])
            return " - ".join(parts) if parts else f"Track {item_id_str}"
        return f"Item {item_id_str}"

    def get_eval_samples(self):
        samples = []
        for bundle_id, true_item_id in self.b_i_pairs_gt:
            input_row = self.b_i_graph_i[bundle_id].toarray().squeeze()
            gt_row = self.b_i_graph_gt[bundle_id].toarray().squeeze()

            cand_rng = np.random.default_rng(int(bundle_id) + self.seed)
            false_indices = np.argwhere((input_row + gt_row) == 0).reshape(-1)
            false_indices = cand_rng.choice(false_indices, size=self.num_cans - 1, replace=False)

            candidate_indices = np.concatenate([[true_item_id], false_indices])
            cand_rng.shuffle(candidate_indices)
            true_option_idx = int(np.argwhere(candidate_indices == true_item_id)[0][0])

            input_rng = np.random.default_rng(int(bundle_id) + self.shuffle_seed)
            input_indices = np.argwhere(input_row > 0).reshape(-1)
            input_rng.shuffle(input_indices)
            if self.num_token > 0 and len(input_indices) > self.num_token:
                input_indices = input_indices[: self.num_token]

            input_str = "; ".join(
                f"{idx + 1}. {self.get_item_text(item_id)}"
                for idx, item_id in enumerate(input_indices)
            )
            target_str = "; ".join(
                f"{chr(ord('A') + idx)}. {self.get_item_text(item_id)}"
                for idx, item_id in enumerate(candidate_indices)
            )

            samples.append(
                {
                    "bundle_id": int(bundle_id),
                    "true_indice": int(true_item_id),
                    "true_option_idx": true_option_idx,
                    "true_option_char": chr(ord("A") + true_option_idx),
                    "input_indices": input_indices.tolist(),
                    "candidate_indices": candidate_indices.tolist(),
                    "input_str": input_str,
                    "target_str": target_str,
                }
            )
        return samples

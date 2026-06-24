import os
import sys
import unittest


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from simple_signal_agent.affordance_graph import (  # noqa: E402
    build_evidence_affordance_graph,
    render_affordance_relation_map,
)
from simple_signal_agent.pipeline import merge_signal_evidence, validate_signal_evidence  # noqa: E402
from simple_signal_agent.prompts import signal_code_prompt  # noqa: E402


class SimpleSignalMultiHopTests(unittest.TestCase):
    def setUp(self):
        self.case = {
            "case_id": "bundle_1",
            "dataset": "pog",
            "bundle_id": 1,
            "partial_item_ids": [10],
            "candidates": [
                {"label": "A", "item_id": 20},
                {"label": "B", "item_id": 30},
            ],
        }
        self.manifest = {
            "sources": [
                {
                    "name": "bi_train.txt",
                    "path": "data/bi_train.txt",
                    "entities": ["bundle", "item"],
                    "relations": ["bundle contains item"],
                    "format": "rows",
                },
                {
                    "name": "pog_LightGCN_bi_feature.pt",
                    "path": "data/pog_LightGCN_bi_feature.pt",
                    "entities": ["item", "feature vector"],
                    "relations": ["item has BI-LightGCN representation"],
                    "format": "tensor",
                },
            ]
        }
        self.graph = build_evidence_affordance_graph(self.manifest, "pog")

    def test_stage1_omits_relation_map_and_stage2_receives_prior_evidence(self):
        stage1 = signal_code_prompt(
            self.case,
            self.manifest,
            self.graph,
            "output/stage1.json",
            30000,
        )
        self.assertIn("initial broad signal scan", stage1)
        self.assertNotIn("Compact Evidence Relation Map", stage1)

        prior = {"signals": [{"signal_name": "stage1_similarity"}]}
        stage2 = signal_code_prompt(
            self.case,
            self.manifest,
            self.graph,
            "output/stage2.json",
            30000,
            refinement_context={"stage1_or_prior_evidence": prior},
        )
        self.assertIn("Stage 2 multi-hop refinement", stage2)
        self.assertIn("Compact Evidence Relation Map", stage2)
        self.assertIn("stage1_similarity", stage2)
        self.assertIn('"relation_path"', stage2)

    def test_relation_map_exposes_dependencies_compactly(self):
        relation_map = render_affordance_relation_map(self.graph)
        self.assertIn("bundle contains item", relation_map)
        self.assertIn("derived_from=bi_train.txt", relation_map)
        self.assertIn("retrieval bridge", relation_map)
        self.assertNotIn('"risks"', relation_map)

    def test_stage2_validation_requires_two_relation_transitions(self):
        base_signal = {
            "signal_name": "composed_context",
            "description": "A composed relation",
            "sources": ["bi_train.txt"],
            "candidate_observations": {
                "A": {"value": 1, "evidence": []},
                "B": {"value": 0, "evidence": []},
            },
        }
        evidence = {"signals": [base_signal]}
        issues = validate_signal_evidence(
            evidence,
            ["A", "B"],
            ["bi_train.txt"],
            require_multi_hop=True,
        )
        self.assertTrue(any("at least two transitions" in issue for issue in issues))

        base_signal["relation_path"] = [
            "item -> bundle",
            "bundle -> item",
        ]
        self.assertEqual(
            validate_signal_evidence(
                evidence,
                ["A", "B"],
                ["bi_train.txt"],
                require_multi_hop=True,
            ),
            [],
        )

    def test_all_valid_round_evidence_is_merged_and_newer_names_replace(self):
        stage1 = {
            "signals": [
                {"signal_name": "keep_me", "description": "stage 1"},
                {"signal_name": "replace_me", "description": "old"},
            ]
        }
        stage2 = {
            "signals": [
                {"signal_name": "replace_me", "description": "new"},
                {"signal_name": "multi_hop", "description": "stage 2"},
            ]
        }
        merged = merge_signal_evidence(stage1, stage2)
        by_name = {signal["signal_name"]: signal for signal in merged["signals"]}
        self.assertEqual(set(by_name), {"keep_me", "replace_me", "multi_hop"})
        self.assertEqual(by_name["replace_me"]["description"], "new")


if __name__ == "__main__":
    unittest.main()

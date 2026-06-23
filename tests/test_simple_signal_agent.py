import asyncio
import json
import os
import re
import tempfile
import unittest

from simple_signal_agent.pipeline import (
    normalize_evaluation,
    run_simple_signal_agent,
    validate_signal_evidence,
)


class SimpleSignalAgentTests(unittest.TestCase):
    def test_validate_minimal_evidence(self):
        evidence = {
            "signals": [
                {
                    "signal_name": "metadata_relation",
                    "description": "Candidate metadata observed from item_info.",
                    "sources": ["item_info.json"],
                    "candidate_observations": {
                        "A": {"value": "top", "evidence": ["A fact"]},
                        "B": {"value": "bottom", "evidence": ["B fact"]},
                    },
                }
            ]
        }
        self.assertEqual(
            validate_signal_evidence(evidence, ["A", "B"], ["item_info.json"]),
            [],
        )

    def test_refine_without_budget_becomes_inconclusive(self):
        evaluation = {
            "status": "REFINE",
            "evidence_quality": "LOW",
            "evidence_gaps": ["Compatibility is unresolved."],
            "required_improvements": ["Obtain a candidate-scoped compatibility observation."],
            "expected_new_information": "The observation could separate compatibility from similarity.",
            "reason": "Current evidence is similarity-only.",
        }
        normalized = normalize_evaluation(evaluation, remaining_refinement_rounds=0)
        self.assertEqual(normalized["status"], "INCONCLUSIVE")
        self.assertIn("budget exhausted", normalized["reason"].lower())

    def test_end_to_end_without_live_llm(self):
        with tempfile.TemporaryDirectory(dir=".") as temp_dir:
            dataset_dir = os.path.join(temp_dir, "datasets", "toy")
            os.makedirs(dataset_dir, exist_ok=True)
            with open(os.path.join(dataset_dir, "count.json"), "w", encoding="utf-8") as handle:
                json.dump({"#B": 1, "#U": 1, "#I": 4}, handle)
            with open(os.path.join(dataset_dir, "item_info.json"), "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "0": {"title": "black jacket", "cate": "outer"},
                        "1": {"title": "white shirt", "cate": "top"},
                        "2": {"title": "blue jeans", "cate": "bottom"},
                    },
                    handle,
                )

            conf = {
                "dataset": "toy",
                "data_path": os.path.join(temp_dir, "datasets"),
                "model": "fake-model",
                "simple_signal_workspace_root": os.path.join(temp_dir, "workspaces"),
                "simple_signal_allowed_files": ["count.json", "item_info.json"],
                "simple_signal_enable_code_guard": True,
                "simple_signal_code_timeout_seconds": 10,
                "simple_signal_code_max_repair_attempts": 0,
                "simple_signal_max_refinement_rounds": 1,
                "simple_signal_max_evidence_chars": 10000,
            }
            sample = {
                "bundle_id": 0,
                "input_indices": [0],
                "candidate_indices": [1, 2],
            }

            async def fake_generate(
                client,
                model,
                prompt,
                current_conf,
                max_output_tokens,
                step_name,
            ):
                del client, model, current_conf, max_output_tokens
                if "code generation" in step_name:
                    output_match = re.search(
                        r"Write UTF-8 JSON to exactly: ([^\r\n]+)",
                        prompt,
                    )
                    output_file = output_match.group(1).strip()
                    evidence = {
                        "signals": [
                            {
                                "signal_name": "metadata_relation",
                                "description": "Candidate metadata observed from item_info.",
                                "sources": ["item_info.json"],
                                "candidate_observations": {
                                    "A": {"value": "top", "evidence": ["white shirt"]},
                                    "B": {"value": "bottom", "evidence": ["blue jeans"]},
                                },
                            }
                        ]
                    }
                    return (
                        "import json, os\n"
                        f"output_file = {output_file!r}\n"
                        "os.makedirs(os.path.dirname(output_file), exist_ok=True)\n"
                        f"evidence = {evidence!r}\n"
                        "with open(output_file, 'w', encoding='utf-8') as handle:\n"
                        "    json.dump(evidence, handle, ensure_ascii=False)\n"
                    )
                if "sufficiency evaluation" in step_name:
                    return json.dumps(
                        {
                            "status": "SUFFICIENT",
                            "evidence_quality": "MEDIUM",
                            "reliable_signals": ["metadata_relation"],
                            "weak_or_failed_signals": [],
                            "coverage_problems": [],
                            "redundancy_problems": [],
                            "conflicts": [],
                            "evidence_gaps": [],
                            "required_improvements": [],
                            "expected_new_information": "",
                            "reason": "Both candidates have grounded observations.",
                        }
                    )
                if "final decision" in step_name:
                    return '{"prediction":"A"}'
                raise AssertionError(f"Unexpected step: {step_name}")

            updates, prediction, raw = asyncio.run(
                run_simple_signal_agent(
                    sample,
                    conf,
                    {"code": object(), "evaluator": object(), "prediction": object()},
                    fake_generate,
                    lambda text: str(text).strip().upper()[:1],
                )
            )
            self.assertEqual(prediction, "A")
            self.assertEqual(raw, '{"prediction":"A"}')
            self.assertEqual(updates["simple_signal_round_count"], 1)
            self.assertEqual(updates["simple_signal_final_status"], "SUFFICIENT")


if __name__ == "__main__":
    unittest.main()

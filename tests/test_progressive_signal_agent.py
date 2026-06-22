import asyncio
import json
import os
import sys
import tempfile
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from progressive_signal_agent.pipeline import run_progressive_signal_agent


class ProgressiveSignalAgentSmokeTest(unittest.TestCase):
    def test_broad_deep_decision_flow(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset_dir = os.path.join(temporary, "datasets", "pog")
            os.makedirs(dataset_dir)
            with open(os.path.join(dataset_dir, "item_info.json"), "w", encoding="utf-8") as handle:
                json.dump({}, handle)

            conf = {
                "dataset": "pog",
                "data_path": os.path.join(temporary, "datasets"),
                "model": "mock-model",
                "psd_workspace_root": os.path.join(temporary, "workspaces"),
                "psd_allowed_files": ["item_info.json"],
                "psd_current_bundle_train_context_policy": "allow",
                "psd_max_deep_rounds": 1,
                "psd_code_max_repair_attempts": 0,
                "psd_enable_code_guard": True,
                "psd_code_timeout_seconds": 10,
            }
            sample = {
                "bundle_id": 7,
                "input_indices": [1, 2],
                "candidate_indices": [3, 4],
            }
            diagnosis_calls = 0

            async def fake_generate(client, model, prompt, call_conf, max_tokens, step_name):
                nonlocal diagnosis_calls
                if "Broad Signal Planner" in prompt:
                    return json.dumps(
                        {
                            "coverage_goal": "cover sources",
                            "source_tasks": [],
                            "output_requirements": [],
                        }
                    )
                if "Broad Signal Python Code Generator" in prompt:
                    evidence = {
                        "case_profile": {
                            "bundle_id": 7,
                            "input_items": [{"item_id": 1}, {"item_id": 2}],
                            "candidate_items": [
                                {"label": "A", "item_id": 3},
                                {"label": "B", "item_id": 4},
                            ],
                        },
                        "source_attempts": [
                            {"source": "item_info.json", "status": "used", "details": "mock"}
                        ],
                        "observations": [
                            {
                                "observation_id": "O1",
                                "source": "item_info.json",
                                "scope": "candidate:A",
                                "kind": "mock",
                                "value": 1,
                                "examples": [],
                                "basis": "mock",
                                "limitations": [],
                            },
                            {
                                "observation_id": "O2",
                                "source": "item_info.json",
                                "scope": "candidate:B",
                                "kind": "mock",
                                "value": 0,
                                "examples": [],
                                "basis": "mock",
                                "limitations": [],
                            },
                        ],
                        "warnings": [],
                    }
                    return self._writer_code("output/psd_broad_evidence_bundle7.json", evidence)
                if "Signal Diagnosis Agent" in prompt:
                    diagnosis_calls += 1
                    status = "NEEDS_DEEPENING" if diagnosis_calls == 1 else "USABLE"
                    return json.dumps(
                        {
                            "status": status,
                            "evidence_quality": "medium",
                            "reliable_observations": [],
                            "observed_failures": [],
                            "unresolved_questions": (
                                [
                                    {
                                        "question": "mock gap",
                                        "competing_explanations": ["x", "y"],
                                    }
                                ]
                                if status == "NEEDS_DEEPENING"
                                else []
                            ),
                            "evidence_gaps": ["mock"] if status == "NEEDS_DEEPENING" else [],
                            "conflicts": [],
                            "signals_to_downweight": [],
                            "candidate_coverage": {"A": True, "B": True},
                            "stop_reason": "",
                        }
                    )
                if "Open-Ended Deep Research Planner" in prompt:
                    return json.dumps(
                        {
                            "research_objective": "resolve mock gap",
                            "investigations": [
                                {
                                    "investigation_id": "I1",
                                    "question": "mock question",
                                    "competing_explanations": ["x", "y"],
                                }
                            ],
                            "portfolio_rationale": "mock",
                            "stop_condition": "done",
                        }
                    )
                if "Deep Signal Python Code Generator" in prompt:
                    evidence = {
                        "investigations": [
                            {
                                "investigation_id": "I1",
                                "question": "mock question",
                                "status": "completed",
                                "method_summary": "mock",
                                "sources_used": ["item_info.json"],
                                "observations": [
                                    {"source": "item_info.json", "scope": "candidate:A", "value": 1},
                                    {"source": "item_info.json", "scope": "candidate:B", "value": 0},
                                ],
                                "limitations": [],
                            }
                        ],
                        "plan_fulfillment": [
                            {"investigation_id": "I1", "status": "completed", "details": "mock"}
                        ],
                        "warnings": [],
                    }
                    return self._writer_code("output/psd_deep_evidence_bundle7_round1.json", evidence)
                if "final Decision Agent" in prompt:
                    return json.dumps(
                        {
                            "prediction": "A",
                            "reasoning": "mock decision",
                            "confidence": "medium",
                            "evidence_quality_used": "medium",
                            "observations_used": ["O1"],
                            "downweighted_or_ignored": [],
                        }
                    )
                raise AssertionError(f"Unexpected stage: {step_name}")

            row, prediction, _ = asyncio.run(
                run_progressive_signal_agent(
                    sample,
                    conf,
                    {"planning": None, "code": None, "diagnosis": None, "prediction": None},
                    fake_generate,
                    lambda value: str(value).strip().upper()[:1],
                )
            )
            self.assertEqual(prediction, "A")
            self.assertEqual(diagnosis_calls, 2)
            self.assertEqual(json.loads(row["psd_diagnosis_history"])[-1]["status"], "USABLE")
            self.assertEqual(len(json.loads(row["psd_deep_trace"])), 1)

    @staticmethod
    def _writer_code(relative_path, payload):
        return (
            "import json\n"
            f"payload = {payload!r}\n"
            f"with open({relative_path!r}, 'w', encoding='utf-8') as handle:\n"
            "    json.dump(payload, handle, ensure_ascii=False)\n"
        )


if __name__ == "__main__":
    unittest.main()

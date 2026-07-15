import json
import os
import sys
import tempfile
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from code.workspace import (
    build_source_manifest,
    execute_generated_code,
    execution_failed,
    prepare_workspace,
)


class IndependentCodeWorkspaceTests(unittest.TestCase):
    def test_prepare_workspace_and_manifest(self):
        with tempfile.TemporaryDirectory() as root:
            dataset_dir = os.path.join(root, "datasets", "pog")
            os.makedirs(dataset_dir)
            with open(os.path.join(dataset_dir, "item_info.json"), "w", encoding="utf-8") as handle:
                json.dump({"1": {"title": "shirt"}}, handle)

            conf = {
                "data_path": os.path.join(root, "datasets"),
                "dataset": "pog",
                "code_workspace_root": os.path.join(root, "workspaces"),
                "code_allowed_files": ["item_info.json", "missing.txt"],
            }
            workspace = prepare_workspace(conf)
            self.assertTrue(os.path.isfile(os.path.join(workspace["data_dir"], "item_info.json")))
            self.assertEqual(workspace["files"], [{"name": "item_info.json", "path": "data/item_info.json"}])

            manifest = build_source_manifest(workspace, "allow")
            self.assertEqual(manifest["sources"][0]["name"], "item_info.json")
            self.assertIn("title", manifest["sources"][0]["fields"])
            self.assertEqual(manifest["current_bundle_train_context_policy"], "allow")

    def test_execute_generated_code_and_parse_output(self):
        with tempfile.TemporaryDirectory() as root:
            os.makedirs(os.path.join(root, "output"))
            workspace = {"workspace_dir": root}
            code = (
                "import json, os\n"
                "os.makedirs('output', exist_ok=True)\n"
                "with open('output/evidence.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump({'schema_version': 'test'}, handle)\n"
            )
            result = execute_generated_code(
                code,
                {"code_enable_code_guard": True},
                workspace,
                "output/evidence.json",
                "generated.py",
            )
            self.assertFalse(execution_failed(result))
            self.assertEqual(result["evidence_json"], {"schema_version": "test"})

    def test_guard_blocks_forbidden_generated_code(self):
        with tempfile.TemporaryDirectory() as root:
            result = execute_generated_code(
                "import subprocess",
                {"code_enable_code_guard": True},
                {"workspace_dir": root},
                "output/evidence.json",
                "generated.py",
            )
            self.assertTrue(execution_failed(result))
            self.assertTrue(result["guard_blocked"])
            self.assertIn(r"\bsubprocess\b", result["guard_violations"])


if __name__ == "__main__":
    unittest.main()

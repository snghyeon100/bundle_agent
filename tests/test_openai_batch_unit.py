import json
import os
import sys
import tempfile
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from operator_learning.openai_batch import (
    batch_output_by_custom_id,
    batch_request,
    extract_response_text,
    response_request_body,
    write_jsonl,
)


class OpenAIBatchTest(unittest.TestCase):
    def test_responses_request_matches_sync_shape(self):
        body = response_request_body(
            model="gpt-5-mini",
            prompt="hello",
            max_output_tokens=123,
            reasoning_effort="low",
        )
        self.assertEqual(body["model"], "gpt-5-mini")
        self.assertEqual(body["input"], "hello")
        self.assertEqual(body["max_output_tokens"], 123)
        self.assertEqual(body["reasoning"], {"effort": "low"})

    def test_output_text_is_extracted_from_responses_body(self):
        row = {
            "custom_id": "code-0001",
            "response": {
                "status_code": 200,
                "body": {
                    "output": [
                        {
                            "type": "message",
                            "content": [
                                {"type": "output_text", "text": "first"},
                                {"type": "output_text", "text": " second"},
                            ],
                        }
                    ]
                },
            },
            "error": None,
        }
        self.assertEqual(extract_response_text(row), "first second")

    def test_jsonl_results_are_mapped_by_custom_id_not_line_order(self):
        requests = [
            batch_request(
                custom_id="request-a",
                body={"model": "gpt-5-mini", "input": "a"},
            ),
            batch_request(
                custom_id="request-b",
                body={"model": "gpt-5-mini", "input": "b"},
            ),
        ]
        with tempfile.TemporaryDirectory() as directory:
            input_path = os.path.join(directory, "input.jsonl")
            info = write_jsonl(input_path, requests)
            self.assertEqual(info["request_count"], 2)

            output_path = os.path.join(directory, "output.jsonl")
            with open(output_path, "w", encoding="utf-8") as handle:
                handle.write(json.dumps({"custom_id": "request-b"}) + "\n")
                handle.write(json.dumps({"custom_id": "request-a"}) + "\n")
            indexed = batch_output_by_custom_id(output_path)
            self.assertEqual(set(indexed), {"request-a", "request-b"})
            self.assertEqual(indexed["request-a"]["custom_id"], "request-a")

    def test_duplicate_custom_ids_are_rejected(self):
        request = batch_request(
            custom_id="duplicate",
            body={"model": "gpt-5-mini", "input": "a"},
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "must be unique"):
                write_jsonl(
                    os.path.join(directory, "input.jsonl"),
                    [request, request],
                )


if __name__ == "__main__":
    unittest.main()

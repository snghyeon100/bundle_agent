"""Small OpenAI Batch API helpers for staged offline evaluations."""

import json
import os


BATCH_ENDPOINT = "/v1/responses"
BATCH_COMPLETION_WINDOW = "24h"
BATCH_MAX_REQUESTS = 50_000
BATCH_MAX_INPUT_BYTES = 200 * 1024 * 1024
BATCH_TERMINAL_STATUSES = {
    "completed",
    "failed",
    "expired",
    "cancelled",
}


def response_request_body(
    *,
    model,
    prompt,
    max_output_tokens,
    reasoning_effort="",
    temperature=None,
):
    """Build the same Responses API body used by the synchronous pipeline."""
    body = {
        "model": str(model),
        "input": str(prompt),
        "max_output_tokens": int(max_output_tokens),
    }
    effort = str(reasoning_effort or "").strip()
    if effort:
        body["reasoning"] = {"effort": effort}
    if temperature is not None:
        body["temperature"] = float(temperature)
    return body


def batch_request(*, custom_id, body):
    """Create one JSONL request object for the Responses endpoint."""
    return {
        "custom_id": str(custom_id),
        "method": "POST",
        "url": BATCH_ENDPOINT,
        "body": body,
    }


def write_jsonl(path, rows):
    """Write one compact UTF-8 JSON object per line and enforce Batch limits."""
    rows = list(rows)
    if not rows:
        raise ValueError("batch input must contain at least one request")
    if len(rows) > BATCH_MAX_REQUESTS:
        raise ValueError(
            f"batch input exceeds {BATCH_MAX_REQUESTS} requests: {len(rows)}"
        )
    custom_ids = [str(row.get("custom_id") or "") for row in rows]
    if any(not custom_id for custom_id in custom_ids):
        raise ValueError("every batch request must have a non-empty custom_id")
    if len(custom_ids) != len(set(custom_ids)):
        raise ValueError("batch request custom_id values must be unique")
    for row in rows:
        if row.get("method") != "POST":
            raise ValueError("every batch request method must be POST")
        if row.get("url") != BATCH_ENDPOINT:
            raise ValueError(f"every batch request URL must be {BATCH_ENDPOINT}")

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
            handle.write("\n")
    size = os.path.getsize(temporary)
    if size > BATCH_MAX_INPUT_BYTES:
        os.remove(temporary)
        raise ValueError(
            f"batch input exceeds {BATCH_MAX_INPUT_BYTES} bytes: {size}"
        )
    os.replace(temporary, path)
    return {"request_count": len(rows), "input_bytes": size}


def read_jsonl(path):
    """Read non-empty JSONL lines."""
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSONL at {path}:{line_number}: {error}"
                ) from error
    return rows


def batch_output_by_custom_id(path):
    """Index a downloaded Batch output or error file by custom_id."""
    indexed = {}
    if not path or not os.path.isfile(path):
        return indexed
    for row in read_jsonl(path):
        custom_id = str(row.get("custom_id") or "")
        if not custom_id:
            continue
        indexed[custom_id] = row
    return indexed


def extract_response_text(output_row):
    """Extract concatenated output_text values from one Batch response line."""
    if not isinstance(output_row, dict):
        raise ValueError("batch output row must be an object")
    if output_row.get("error"):
        error = output_row["error"]
        raise ValueError(
            str(error.get("message") if isinstance(error, dict) else error)
        )
    response = output_row.get("response")
    if not isinstance(response, dict):
        raise ValueError("batch output row contains no response")
    status_code = int(response.get("status_code") or 0)
    if status_code < 200 or status_code >= 300:
        raise ValueError(f"batch response returned HTTP {status_code}")
    body = response.get("body")
    if not isinstance(body, dict):
        raise ValueError("batch response contains no body")

    direct = body.get("output_text")
    if isinstance(direct, str) and direct:
        return direct
    texts = []
    for output in body.get("output", []):
        if not isinstance(output, dict):
            continue
        for content in output.get("content", []):
            if not isinstance(content, dict):
                continue
            if content.get("type") == "output_text":
                text = content.get("text")
                if isinstance(text, str):
                    texts.append(text)
    if not texts:
        raise ValueError("batch response contains no output text")
    return "".join(texts)


def plain_object(value):
    """Convert OpenAI SDK response models into JSON-serializable dictionaries."""
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return value
    return json.loads(json.dumps(value, default=str))


def submit_batch(client, *, input_path, metadata=None):
    """Upload a prepared JSONL file and submit one 24-hour Batch job."""
    with open(input_path, "rb") as handle:
        input_file = client.files.create(file=handle, purpose="batch")
    batch = client.batches.create(
        input_file_id=input_file.id,
        endpoint=BATCH_ENDPOINT,
        completion_window=BATCH_COMPLETION_WINDOW,
        metadata={
            str(key): str(value)
            for key, value in (metadata or {}).items()
        },
    )
    return {
        "input_file_id": str(input_file.id),
        "batch": plain_object(batch),
    }


def retrieve_batch(client, batch_id):
    """Retrieve the current server-side Batch object."""
    return plain_object(client.batches.retrieve(str(batch_id)))


def _content_bytes(content):
    if hasattr(content, "content"):
        value = content.content
        if isinstance(value, bytes):
            return value
    if hasattr(content, "read"):
        value = content.read()
        if isinstance(value, bytes):
            return value
        return str(value).encode("utf-8")
    text = getattr(content, "text", None)
    if isinstance(text, str):
        return text.encode("utf-8")
    if isinstance(content, bytes):
        return content
    return str(content).encode("utf-8")


def download_file(client, file_id, path):
    """Download one OpenAI file atomically."""
    response = client.files.content(str(file_id))
    data = _content_bytes(response)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = f"{path}.tmp"
    with open(temporary, "wb") as handle:
        handle.write(data)
    os.replace(temporary, path)
    return {"bytes": len(data), "path": os.path.abspath(path)}

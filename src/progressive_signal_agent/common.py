import json
import re


def parse_candidate_options(target_str):
    pattern = r"(?:^|;\s*)([A-Z])\.\s*(.*?)(?=\s*;\s*[A-Z]\.\s*|$)"
    matches = re.findall(pattern, str(target_str or ""), flags=re.DOTALL)
    return [(letter, " ".join(text.split())) for letter, text in matches]


def build_case_view(sample, dataset):
    """Return the only sample fields exposed to planning and generated code."""
    labels = [chr(ord("A") + index) for index in range(len(sample["candidate_indices"]))]
    return {
        "case_id": f"bundle_{sample['bundle_id']}",
        "dataset": dataset,
        "bundle_id": int(sample["bundle_id"]),
        "partial_item_ids": [int(value) for value in sample["input_indices"]],
        "candidates": [
            {"label": label, "item_id": int(item_id)}
            for label, item_id in zip(labels, sample["candidate_indices"])
        ],
    }


def candidate_labels(case_view):
    return [str(candidate["label"]) for candidate in case_view["candidates"]]


def task_semantics(dataset):
    name = str(dataset or "").lower()
    if "spotify" in name:
        return (
            "This is playlist continuation. A partial bundle is a music playlist, and each candidate is a "
            "possible next song. Historical compatibility may involve theme, mood, genre, artist or album "
            "context, and listening flow."
        )
    return (
        "This is fashion outfit bundle completion. A partial bundle is a coordinated set of fashion items, "
        "and each candidate is a possible missing item whose compatibility with the partial bundle should be evaluated."
    )


def parse_json_from_text(text):
    value = str(text or "").strip()
    if not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        pass
    fenced = re.search(r"```(?:json)?\s*(.*?)```", value, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        try:
            return json.loads(fenced.group(1).strip())
        except json.JSONDecodeError:
            pass
    start = value.find("{")
    end = value.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(value[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None


def extract_python_code(text):
    value = str(text or "").strip()
    fenced = re.search(r"```(?:python)?\s*(.*?)```", value, flags=re.DOTALL | re.IGNORECASE)
    return fenced.group(1).strip() if fenced else value


def compact_json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def execution_summary(result):
    stderr = str(result.get("stderr") or "")
    return {
        "returncode": result.get("returncode"),
        "timed_out": bool(result.get("timed_out")),
        "guard_blocked": bool(result.get("guard_blocked")),
        "guard_violations": result.get("guard_violations", []),
        "evidence_json_present": isinstance(result.get("evidence_json"), dict),
        "evidence_output_file": result.get("evidence_output_file", ""),
        "stderr_tail": stderr[-1600:] if stderr else "",
    }

import json
import re


def parse_candidate_options(target_str):
    pattern = r"(?:^|;\s*)([A-Z])\.\s*(.*?)(?=\s*;\s*[A-Z]\.\s*|$)"
    matches = re.findall(pattern, target_str, flags=re.DOTALL)
    return [(letter, " ".join(text.split())) for letter, text in matches]


def build_agent_sample_view(sample):
    candidate_options = parse_candidate_options(sample["target_str"])
    candidates = []
    for idx, (letter, text) in enumerate(candidate_options):
        item_id = sample["candidate_indices"][idx] if idx < len(sample["candidate_indices"]) else None
        candidates.append({"label": letter, "item_id": item_id, "text": text})
    return {
        "bundle_id": sample["bundle_id"],
        "input_item_ids": sample["input_indices"],
        "input_text": sample["input_str"],
        "candidates": candidates,
    }


def candidate_labels(sample):
    return [letter for letter, _ in parse_candidate_options(sample["target_str"])]


def parse_json_from_text(text):
    text = str(text or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        try:
            return json.loads(fenced.group(1).strip())
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None


def extract_python_code(text):
    text = str(text or "")
    fenced = re.search(r"```(?:python|py)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    return text.strip()


def compact_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def task_semantics(dataset_name):
    if "spotify" in str(dataset_name).lower():
        return (
            "This is a playlist continuation task. Each bundle is a music playlist made of songs intended "
            "to be listened to together. The partial bundle contains songs already in the playlist, and each "
            "candidate is a possible missing or next song. The goal is to choose the candidate that would most "
            "naturally continue the playlist as a coherent listening sequence."
        )
    return (
        "This is a fashion outfit bundle completion task. Each bundle is a coordinated set of fashion items, "
        "typically combining multiple item roles into one outfit. The partial bundle contains items already in "
        "the outfit, and each candidate is a possible missing item. The goal is to choose the candidate that "
        "would most naturally complete the outfit as a coherent set."
    )

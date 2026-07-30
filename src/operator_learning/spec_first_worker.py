"""Subprocess worker for one generated spec-first strategy program."""

import argparse
import importlib.util
import json


def _load_program(path):
    spec = importlib.util.spec_from_file_location("generated_spec_first_program", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load generated program: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--program", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    module = _load_program(args.program)
    run = getattr(module, "run", None)
    if not callable(run):
        raise TypeError("generated program must define callable run")
    result = run(
        payload["partial_items"],
        payload["candidate_items"],
        payload["source_paths"],
        max_contexts_per_candidate=int(payload["max_contexts_per_candidate"]),
    )
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

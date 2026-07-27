"""Child-process entrypoint for generated online program execution."""

import json
import sys

from .runtime import execute_worker_request


def main():
    try:
        request = json.loads(sys.stdin.read())
        result = execute_worker_request(request)
    except Exception as error:
        result = {
            "status": "execution_error",
            "result": None,
            "validation_issues": [],
            "error": f"{type(error).__name__}: {error}",
        }
    sys.stdout.write(json.dumps(result, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

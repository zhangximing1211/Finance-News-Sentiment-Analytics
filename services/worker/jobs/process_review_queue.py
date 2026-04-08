from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json
import sys


BASE_DIR = Path(__file__).resolve().parents[3]
WORKER_SRC = BASE_DIR / "services" / "worker" / "src"

if str(WORKER_SRC) not in sys.path:
    sys.path.insert(0, str(WORKER_SRC))

from worker_service import ReviewQueueWorker


def main() -> None:
    parser = ArgumentParser(description="Process pending review queue items and enrich them with external LLM explanations.")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()

    worker = ReviewQueueWorker()
    result = worker.process_pending(limit=args.limit, retry_failed=args.retry_failed)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

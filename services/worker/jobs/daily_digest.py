from __future__ import annotations

from pathlib import Path
import json
import sys


BASE_DIR = Path(__file__).resolve().parents[3]
WORKER_SRC = BASE_DIR / "services" / "worker" / "src"

if str(WORKER_SRC) not in sys.path:
    sys.path.insert(0, str(WORKER_SRC))

from worker_service import ReviewQueueRepository, build_daily_digest


def main() -> None:
    repository = ReviewQueueRepository()
    digest = build_daily_digest(repository.list_items(limit=500))
    print(json.dumps(digest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

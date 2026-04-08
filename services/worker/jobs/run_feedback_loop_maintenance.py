from __future__ import annotations

from argparse import ArgumentParser
from datetime import date
from pathlib import Path
import json
import sys


BASE_DIR = Path(__file__).resolve().parents[3]
WORKER_SRC = BASE_DIR / "services" / "worker" / "src"

if str(WORKER_SRC) not in sys.path:
    sys.path.insert(0, str(WORKER_SRC))

from worker_service import get_agent_workflow_service


def main() -> None:
    parser = ArgumentParser(description="Run feedback-loop maintenance: auto sampling review and periodic retrain scheduling.")
    parser.add_argument("--report-type", choices=["daily", "weekly"], default="weekly")
    parser.add_argument("--date", default=None, help="Reference date in YYYY-MM-DD format.")
    parser.add_argument("--sample-limit", type=int, default=12)
    args = parser.parse_args()

    reference_date = date.fromisoformat(args.date) if args.date else None
    service = get_agent_workflow_service()
    payload = service.run_feedback_loop_maintenance(
        report_type=args.report_type,
        reference_date=reference_date,
        sample_limit=args.sample_limit,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

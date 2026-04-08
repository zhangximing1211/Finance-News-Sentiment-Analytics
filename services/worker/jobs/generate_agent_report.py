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
    parser = ArgumentParser(description="Generate daily or weekly agent workflow reports.")
    parser.add_argument("--report-type", choices=["daily", "weekly"], required=True)
    parser.add_argument("--date", default=None, help="Reference date in YYYY-MM-DD format. Defaults to today.")
    args = parser.parse_args()

    reference_date = date.fromisoformat(args.date) if args.date else None
    service = get_agent_workflow_service()
    report = service.generate_report(report_type=args.report_type, reference_date=reference_date)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

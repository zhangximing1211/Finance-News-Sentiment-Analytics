from __future__ import annotations

from pathlib import Path
import json
import sys


BASE_DIR = Path(__file__).resolve().parents[3]
TRAINER_SRC = BASE_DIR / "services" / "trainer" / "src"

if str(TRAINER_SRC) not in sys.path:
    sys.path.insert(0, str(TRAINER_SRC))

from trainer_service.data_prep import run_full_data_audit


def main() -> None:
    result = run_full_data_audit()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

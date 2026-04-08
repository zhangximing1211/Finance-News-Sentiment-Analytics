from __future__ import annotations

from pathlib import Path
import json
import sys


BASE_DIR = Path(__file__).resolve().parents[3]
TRAINER_SRC = BASE_DIR / "services" / "trainer" / "src"

if str(TRAINER_SRC) not in sys.path:
    sys.path.insert(0, str(TRAINER_SRC))

from trainer_service import TrainerService


def main() -> None:
    trainer = TrainerService()
    report = {
        "dataset": trainer.dataset_summary(),
        "evaluation": trainer.train_and_evaluate(),
    }
    output_path = BASE_DIR / "data" / "processed" / "training-report.json"
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

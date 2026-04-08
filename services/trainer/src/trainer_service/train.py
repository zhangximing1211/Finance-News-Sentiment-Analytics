from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parents[4]
UTILS_DIR = BASE_DIR / "packages" / "utils" / "python"

if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

from finance_utils import build_sentiment_pipeline, normalize_text


class TrainerService:
    def __init__(self, data_path: str | Path | None = None) -> None:
        self.data_path = Path(data_path) if data_path else BASE_DIR / "data" / "raw" / "all-data.csv"

    def load_dataset(self) -> pd.DataFrame:
        dataset = pd.read_csv(
            self.data_path,
            names=["sentiment", "text"],
            header=None,
            encoding="ISO-8859-1",
        ).dropna()
        dataset["text"] = dataset["text"].astype(str).map(normalize_text)
        return dataset[dataset["text"] != ""]

    def dataset_summary(self) -> dict[str, object]:
        dataset = self.load_dataset()
        label_distribution = dataset["sentiment"].value_counts().sort_index().to_dict()
        return {
            "rows": int(len(dataset)),
            "labels": label_distribution,
            "data_path": str(self.data_path),
        }

    def train_and_evaluate(self, test_size: float = 0.2, random_state: int = 42) -> dict[str, object]:
        dataset = self.load_dataset()
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            dataset["text"],
            dataset["sentiment"],
            test_size=test_size,
            random_state=random_state,
            stratify=dataset["sentiment"],
        )

        pipeline = build_sentiment_pipeline()
        pipeline.fit(train_texts, train_labels)
        predictions = pipeline.predict(test_texts)

        accuracy = float(accuracy_score(test_labels, predictions))
        report = classification_report(test_labels, predictions, output_dict=True, zero_division=0)

        return {
            "accuracy": round(accuracy, 4),
            "train_size": int(len(train_texts)),
            "test_size": int(len(test_texts)),
            "classification_report": report,
        }

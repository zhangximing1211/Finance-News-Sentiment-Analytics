from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
TRAINER_SRC = ROOT / "services" / "trainer" / "src"

if str(TRAINER_SRC) not in sys.path:
    sys.path.insert(0, str(TRAINER_SRC))

from trainer_service.data_prep import LABEL_TO_ID, prepare_dataset


class DataPrepTests(unittest.TestCase):
    def test_prepare_dataset_outputs_expected_sizes_and_splits(self) -> None:
        split_df, summary = prepare_dataset()

        self.assertEqual(len(split_df), 4836)
        self.assertEqual(summary["deduplication"]["exact_duplicates_removed"], 6)
        self.assertEqual(summary["deduplication"]["conflicting_duplicate_rows_removed"], 4)

        split_counts = split_df["split"].value_counts().to_dict()
        self.assertEqual(split_counts, {"train": 3868, "val": 484, "test": 484})

        self.assertEqual(set(split_df["label_id"].unique()), set(LABEL_TO_ID.values()))

        train_texts = set(split_df.loc[split_df["split"] == "train", "text"])
        val_texts = set(split_df.loc[split_df["split"] == "val", "text"])
        test_texts = set(split_df.loc[split_df["split"] == "test", "text"])

        self.assertTrue(train_texts.isdisjoint(val_texts))
        self.assertTrue(train_texts.isdisjoint(test_texts))
        self.assertTrue(val_texts.isdisjoint(test_texts))


if __name__ == "__main__":
    unittest.main()

"""FinBERT fine-tuning module for sentiment classification.

Provides train/evaluate functions that integrate with the existing
``baselines.py`` evaluation framework and ``data_prep`` data splits.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from .baselines import (
    BASELINE_DIR,
    LABEL_ORDER,
    PROCESSED_DIR,
    evaluate_classifier,
    load_processed_split,
    _save_metrics_files,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PRETRAINED_MODEL_NAME = "ProsusAI/finbert"
BERT_MODEL_DIR = PROCESSED_DIR / "bert_models"

LABEL_TO_ID = {label: idx for idx, label in enumerate(LABEL_ORDER)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}

DEFAULT_HYPERPARAMS: dict[str, Any] = {
    "epochs": 4,
    "batch_size": 16,
    "max_seq_len": 512,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "seed": 42,
}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class SentimentDataset(Dataset):
    """PyTorch dataset backed by a processed-split DataFrame."""

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer: AutoTokenizer,
        max_len: int,
    ) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _compute_class_weights(labels: list[int], num_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights, analogous to sklearn balanced."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    total = counts.sum()
    weights = total / (num_classes * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32)


def _build_dataloader(
    df: pd.DataFrame,
    tokenizer: AutoTokenizer,
    max_len: int,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    texts = df["text"].tolist()
    labels = [LABEL_TO_ID[label] for label in df["label"]]
    dataset = SentimentDataset(texts, labels, tokenizer, max_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class _BertModelWrapper:
    """Thin wrapper so ``evaluate_classifier`` can call predict / predict_proba."""

    def __init__(
        self,
        model: AutoModelForSequenceClassification,
        tokenizer: AutoTokenizer,
        max_len: int,
        device: torch.device,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.device = device
        self.classes_ = np.array(LABEL_ORDER)

    # -- sklearn-compatible interface for evaluate_classifier ---------------

    def predict(self, texts: Any) -> np.ndarray:
        proba = self.predict_proba(texts)
        return self.classes_[np.argmax(proba, axis=1)]

    def predict_proba(self, texts: Any) -> np.ndarray:
        text_list = list(texts)
        self.model.eval()
        all_probs: list[np.ndarray] = []
        batch_size = 32

        with torch.no_grad():
            for start in range(0, len(text_list), batch_size):
                batch_texts = text_list[start : start + batch_size]
                encoding = self.tokenizer(
                    batch_texts,
                    max_length=self.max_len,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                input_ids = encoding["input_ids"].to(self.device)
                attention_mask = encoding["attention_mask"].to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
                all_probs.append(probs)

        return np.concatenate(all_probs, axis=0)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_finbert(
    processed_dir: str | Path = PROCESSED_DIR,
    output_dir: str | Path = BERT_MODEL_DIR,
    hyperparams: dict[str, Any] | None = None,
    pretrained_model: str = PRETRAINED_MODEL_NAME,
) -> dict[str, Any]:
    """Fine-tune FinBERT on processed train/val splits.

    Returns a summary dict compatible with the existing training pipeline.
    """
    hp = {**DEFAULT_HYPERPARAMS, **(hyperparams or {})}
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(hp["seed"])
    np.random.seed(hp["seed"])

    device = _resolve_device()
    print(f"[train_finbert] device={device}  hyperparams={hp}")

    # ---- data -------------------------------------------------------------
    train_df = load_processed_split("train", processed_dir)
    val_df = load_processed_split("val", processed_dir)

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    train_loader = _build_dataloader(train_df, tokenizer, hp["max_seq_len"], hp["batch_size"], shuffle=True)
    val_loader = _build_dataloader(val_df, tokenizer, hp["max_seq_len"], hp["batch_size"], shuffle=False)

    # ---- model ------------------------------------------------------------
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model,
        num_labels=len(LABEL_ORDER),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )
    model.to(device)

    # ---- class-weighted loss ----------------------------------------------
    train_labels = [LABEL_TO_ID[label] for label in train_df["label"]]
    class_weights = _compute_class_weights(train_labels, len(LABEL_ORDER)).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    # ---- optimizer & scheduler --------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hp["learning_rate"],
        weight_decay=hp["weight_decay"],
    )
    total_steps = len(train_loader) * hp["epochs"]
    warmup_steps = int(total_steps * hp["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # ---- training loop ----------------------------------------------------
    history: list[dict[str, Any]] = []
    best_val_f1 = -1.0
    best_epoch = -1

    for epoch in range(1, hp["epochs"] + 1):
        # -- train ----------------------------------------------------------
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item() * len(labels)
            preds = outputs.logits.argmax(dim=-1)
            train_correct += (preds == labels).sum().item()
            train_total += len(labels)

        avg_train_loss = running_loss / train_total
        train_accuracy = train_correct / train_total

        # -- validate -------------------------------------------------------
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_val_preds: list[int] = []
        all_val_labels: list[int] = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, labels)
                val_loss += loss.item() * len(labels)
                preds = outputs.logits.argmax(dim=-1)
                val_correct += (preds == labels).sum().item()
                val_total += len(labels)
                all_val_preds.extend(preds.cpu().tolist())
                all_val_labels.extend(labels.cpu().tolist())

        avg_val_loss = val_loss / val_total
        val_accuracy = val_correct / val_total

        # per-class f1 on val
        from sklearn.metrics import f1_score as sklearn_f1
        val_macro_f1 = float(
            sklearn_f1(all_val_labels, all_val_preds, labels=list(range(len(LABEL_ORDER))), average="macro", zero_division=0)
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "train_accuracy": round(train_accuracy, 4),
            "val_loss": round(avg_val_loss, 4),
            "val_accuracy": round(val_accuracy, 4),
            "val_macro_f1": round(val_macro_f1, 4),
            "lr": round(scheduler.get_last_lr()[0], 8),
        }
        history.append(epoch_record)
        print(
            f"  epoch {epoch}/{hp['epochs']}  "
            f"train_loss={avg_train_loss:.4f}  train_acc={train_accuracy:.4f}  "
            f"val_loss={avg_val_loss:.4f}  val_acc={val_accuracy:.4f}  "
            f"val_macro_f1={val_macro_f1:.4f}"
        )

        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            best_epoch = epoch
            model.save_pretrained(output_path / "best_model")
            tokenizer.save_pretrained(output_path / "best_model")

    # ---- save final model -------------------------------------------------
    model.save_pretrained(output_path / "final_model")
    tokenizer.save_pretrained(output_path / "final_model")

    # ---- full evaluation using existing framework -------------------------
    best_model = AutoModelForSequenceClassification.from_pretrained(output_path / "best_model")
    best_model.to(device)
    best_tokenizer = AutoTokenizer.from_pretrained(output_path / "best_model")
    wrapper = _BertModelWrapper(best_model, best_tokenizer, hp["max_seq_len"], device)

    val_metrics = evaluate_classifier(wrapper, val_df, split_name="val")
    val_artifact_paths = _save_metrics_files(output_path, "val", val_metrics)

    # ---- save metadata ----------------------------------------------------
    metadata = {
        "model_name": "finbert_finetuned",
        "model_family": "transformer",
        "pretrained_model": pretrained_model,
        "hyperparameters": hp,
        "device": str(device),
        "training_history": history,
        "best_epoch": best_epoch,
        "best_val_macro_f1": round(best_val_f1, 4),
        "calibration": {
            "method": "softmax",
            "validation_ece": val_metrics["calibration"]["expected_calibration_error"],
        },
        "abstain_policy": {
            "low_confidence_threshold": val_metrics["abstain_policy"]["low_confidence_threshold"],
            "selection_note": val_metrics["abstain_policy"]["threshold_selection"]["selection_note"],
        },
        "review_queue_policy": {
            "primary_trigger": "low_confidence",
            "secondary_trigger": "neutral_boundary",
        },
        "neutral_boundary": {
            "margin_threshold": 0.08,
        },
        "class_imbalance_strategy": {
            "type": "class_weight",
            "value": "inverse_frequency",
            "note": "CrossEntropyLoss with inverse-frequency class weights.",
        },
        "selection_metric": "macro_f1 on validation split",
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "artifact_path": str(output_path / "best_model"),
    }
    metadata_path = output_path / "bert_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "model_name": "finbert_finetuned",
        "model_path": str(output_path / "best_model"),
        "metadata_path": str(metadata_path),
        "best_epoch": best_epoch,
        "best_val_macro_f1": round(best_val_f1, 4),
        "training_history": history,
        "validation_metrics": {
            key: value
            for key, value in val_metrics.items()
            if key not in {"predictions", "probabilities", "confidences", "review_queue"}
        },
        "validation_artifacts": val_artifact_paths,
    }


def evaluate_bert_model(
    split_name: str = "test",
    model_dir: str | Path | None = None,
    processed_dir: str | Path = PROCESSED_DIR,
) -> dict[str, Any]:
    """Evaluate a saved FinBERT model on a data split.

    Compatible with the ``evaluate_saved_model`` interface from baselines.
    """
    resolved_dir = Path(model_dir) if model_dir else BERT_MODEL_DIR / "best_model"
    metadata_path = resolved_dir.parent / "bert_metadata.json" if resolved_dir.name == "best_model" else resolved_dir / "bert_metadata.json"

    device = _resolve_device()
    model = AutoModelForSequenceClassification.from_pretrained(resolved_dir)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(resolved_dir)

    hp = DEFAULT_HYPERPARAMS.copy()
    low_confidence_threshold = None
    if metadata_path.exists():
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        hp.update(meta.get("hyperparameters", {}))
        low_confidence_threshold = meta.get("abstain_policy", {}).get("low_confidence_threshold")

    wrapper = _BertModelWrapper(model, tokenizer, hp["max_seq_len"], device)
    split_df = load_processed_split(split_name, processed_dir)
    metrics = evaluate_classifier(
        wrapper,
        split_df,
        split_name=split_name,
        low_confidence_threshold=low_confidence_threshold,
    )

    output_dir = BERT_MODEL_DIR
    artifact_paths = _save_metrics_files(output_dir, f"bert_{split_name}", metrics)

    return {
        "model_path": str(resolved_dir),
        "metadata_path": str(metadata_path) if metadata_path.exists() else None,
        "split": split_name,
        "artifacts": artifact_paths,
        "metrics": {
            key: value
            for key, value in metrics.items()
            if key not in {"predictions", "probabilities", "confidences", "review_queue"}
        },
    }

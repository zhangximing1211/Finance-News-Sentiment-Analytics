from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from typing import Any, Iterable
from zoneinfo import ZoneInfo


BASE_DIR = Path(__file__).resolve().parents[4]
DEFAULT_DB_PATH = BASE_DIR / "data" / "interim" / "review_queue.sqlite3"
DEFAULT_TIMEZONE = os.getenv("WORKFLOW_TIMEZONE", "Asia/Hong_Kong")
QUEUE_STATUSES = {"pending", "processing", "ready_for_review", "failed"}
PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}


def resolve_review_queue_db_path(db_path: str | Path | None = None) -> Path:
    configured = db_path or os.getenv("REVIEW_QUEUE_DB_PATH")
    return Path(configured) if configured else DEFAULT_DB_PATH


def _utc_now_iso() -> str:
    return datetime.now(ZoneInfo(DEFAULT_TIMEZONE)).isoformat()


def _priority_sort_key(priority: str) -> int:
    return PRIORITY_ORDER.get(priority, len(PRIORITY_ORDER))


class ReviewQueueRepository:
    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = resolve_review_queue_db_path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    @contextmanager
    def _managed_connection(self):
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def initialize(self) -> None:
        with self._managed_connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS review_queue_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fingerprint TEXT NOT NULL UNIQUE,
                    status TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    primary_entity TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    predicted_label TEXT NOT NULL,
                    decision_label TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    review_reasons_json TEXT NOT NULL,
                    recommended_action TEXT NOT NULL,
                    text_excerpt TEXT NOT NULL,
                    input_text TEXT NOT NULL,
                    analysis_payload_json TEXT NOT NULL,
                    secondary_explanation_json TEXT,
                    llm_provider TEXT,
                    last_error TEXT,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_review_queue_status_priority
                ON review_queue_items (status, priority, updated_at)
                """
            )

    def _fingerprint(self, analysis_result: dict[str, Any]) -> str:
        review_item = analysis_result["review_queue_item"]
        payload = {
            "input_text": analysis_result["input_text"],
            "decision_label": review_item["decision_label"],
            "event_type": review_item["event_type"],
            "primary_entity": review_item["primary_entity"],
            "review_reasons": review_item["review_reasons"],
        }
        return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()

    def _row_to_record(self, row: sqlite3.Row) -> dict[str, Any]:
        secondary_explanation = row["secondary_explanation_json"]
        return {
            "id": int(row["id"]),
            "status": row["status"],
            "priority": row["priority"],
            "primary_entity": row["primary_entity"],
            "event_type": row["event_type"],
            "predicted_label": row["predicted_label"],
            "decision_label": row["decision_label"],
            "confidence": round(float(row["confidence"]), 4),
            "review_reasons": json.loads(row["review_reasons_json"]),
            "recommended_action": row["recommended_action"],
            "text_excerpt": row["text_excerpt"],
            "input_text": row["input_text"],
            "analysis_payload": row["analysis_payload_json"],
            "secondary_explanation": json.loads(secondary_explanation) if secondary_explanation else None,
            "llm_provider": row["llm_provider"],
            "last_error": row["last_error"],
            "attempts": int(row["attempts"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def get_item(self, item_id: int) -> dict[str, Any] | None:
        with self._managed_connection() as connection:
            row = connection.execute(
                "SELECT * FROM review_queue_items WHERE id = ?",
                (item_id,),
            ).fetchone()
        return self._row_to_record(row) if row else None

    def enqueue_analysis(self, analysis_result: dict[str, Any]) -> dict[str, Any] | None:
        review_item = analysis_result.get("review_queue_item")
        if not review_item:
            return None

        fingerprint = self._fingerprint(analysis_result)
        secondary_explanation = analysis_result.get("secondary_explanation")
        now = _utc_now_iso()
        status = "ready_for_review" if secondary_explanation and secondary_explanation.get("used_external_llm") else "pending"
        payload_json = json.dumps(analysis_result, ensure_ascii=False)
        secondary_json = json.dumps(secondary_explanation, ensure_ascii=False) if secondary_explanation else None
        llm_provider = secondary_explanation.get("provider") if secondary_explanation else None

        with self._managed_connection() as connection:
            existing = connection.execute(
                "SELECT id, status, attempts, created_at FROM review_queue_items WHERE fingerprint = ?",
                (fingerprint,),
            ).fetchone()

            if existing:
                next_status = status
                if existing["status"] == "ready_for_review" and status != "ready_for_review":
                    next_status = existing["status"]

                connection.execute(
                    """
                    UPDATE review_queue_items
                    SET status = ?,
                        priority = ?,
                        primary_entity = ?,
                        event_type = ?,
                        predicted_label = ?,
                        decision_label = ?,
                        confidence = ?,
                        review_reasons_json = ?,
                        recommended_action = ?,
                        text_excerpt = ?,
                        input_text = ?,
                        analysis_payload_json = ?,
                        secondary_explanation_json = ?,
                        llm_provider = ?,
                        last_error = NULL,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        next_status,
                        review_item["priority"],
                        review_item["primary_entity"],
                        review_item["event_type"],
                        review_item["predicted_label"],
                        review_item["decision_label"],
                        review_item["confidence"],
                        json.dumps(review_item["review_reasons"], ensure_ascii=False),
                        review_item["recommended_action"],
                        review_item["text_excerpt"],
                        analysis_result["input_text"],
                        payload_json,
                        secondary_json,
                        llm_provider,
                        now,
                        existing["id"],
                    ),
                )
                item_id = int(existing["id"])
            else:
                cursor = connection.execute(
                    """
                    INSERT INTO review_queue_items (
                        fingerprint,
                        status,
                        priority,
                        primary_entity,
                        event_type,
                        predicted_label,
                        decision_label,
                        confidence,
                        review_reasons_json,
                        recommended_action,
                        text_excerpt,
                        input_text,
                        analysis_payload_json,
                        secondary_explanation_json,
                        llm_provider,
                        last_error,
                        attempts,
                        created_at,
                        updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        fingerprint,
                        status,
                        review_item["priority"],
                        review_item["primary_entity"],
                        review_item["event_type"],
                        review_item["predicted_label"],
                        review_item["decision_label"],
                        review_item["confidence"],
                        json.dumps(review_item["review_reasons"], ensure_ascii=False),
                        review_item["recommended_action"],
                        review_item["text_excerpt"],
                        analysis_result["input_text"],
                        payload_json,
                        secondary_json,
                        llm_provider,
                        None,
                        0,
                        now,
                        now,
                    ),
                )
                item_id = int(cursor.lastrowid)

        return self.get_item(item_id)

    def list_items(self, status: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        query = "SELECT * FROM review_queue_items"
        params: list[Any] = []
        if status:
            if status not in QUEUE_STATUSES:
                raise ValueError(f"Unsupported review queue status: {status}")
            query += " WHERE status = ?"
            params.append(status)
        query += " ORDER BY CASE priority WHEN 'high' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END, updated_at DESC LIMIT ?"
        params.append(limit)

        with self._managed_connection() as connection:
            rows = connection.execute(query, params).fetchall()
        return [self._row_to_record(row) for row in rows]

    def get_summary(self) -> dict[str, Any]:
        with self._managed_connection() as connection:
            rows = connection.execute(
                "SELECT status, priority, review_reasons_json FROM review_queue_items"
            ).fetchall()

        status_counter = Counter()
        priority_counter = Counter()
        reason_counter = Counter()
        for row in rows:
            status_counter[row["status"]] += 1
            priority_counter[row["priority"]] += 1
            for reason in json.loads(row["review_reasons_json"]):
                reason_counter[reason] += 1

        return {
            "total_items": sum(status_counter.values()),
            "status_breakdown": dict(status_counter),
            "priority_breakdown": dict(priority_counter),
            "review_reason_breakdown": dict(reason_counter),
            "pending_count": status_counter.get("pending", 0),
            "processing_count": status_counter.get("processing", 0),
            "ready_for_review_count": status_counter.get("ready_for_review", 0),
            "failed_count": status_counter.get("failed", 0),
        }

    def claim_items(self, limit: int = 20, retry_failed: bool = False) -> list[dict[str, Any]]:
        eligible_statuses = ["pending"]
        if retry_failed:
            eligible_statuses.append("failed")

        placeholders = ", ".join("?" for _ in eligible_statuses)
        params: list[Any] = [*eligible_statuses, limit]

        with self._managed_connection() as connection:
            rows = connection.execute(
                f"""
                SELECT id
                FROM review_queue_items
                WHERE status IN ({placeholders})
                ORDER BY CASE priority WHEN 'high' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END, updated_at ASC
                LIMIT ?
                """,
                params,
            ).fetchall()

            item_ids = [int(row["id"]) for row in rows]
            if not item_ids:
                return []

            now = _utc_now_iso()
            for item_id in item_ids:
                connection.execute(
                    """
                    UPDATE review_queue_items
                    SET status = 'processing',
                        attempts = attempts + 1,
                        last_error = NULL,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (now, item_id),
                )

        return [item for item in (self.get_item(item_id) for item_id in item_ids) if item]

    def mark_ready_for_review(self, item_id: int, secondary_explanation: dict[str, Any]) -> dict[str, Any] | None:
        now = _utc_now_iso()
        with self._managed_connection() as connection:
            connection.execute(
                """
                UPDATE review_queue_items
                SET status = 'ready_for_review',
                    secondary_explanation_json = ?,
                    llm_provider = ?,
                    last_error = NULL,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    json.dumps(secondary_explanation, ensure_ascii=False),
                    secondary_explanation.get("provider"),
                    now,
                    item_id,
                ),
            )
        return self.get_item(item_id)

    def mark_failed(self, item_id: int, error_message: str) -> dict[str, Any] | None:
        now = _utc_now_iso()
        with self._managed_connection() as connection:
            connection.execute(
                """
                UPDATE review_queue_items
                SET status = 'failed',
                    last_error = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (error_message[:1000], now, item_id),
            )
        return self.get_item(item_id)


def build_review_queue_digest(items: Iterable[dict[str, Any]]) -> dict[str, Any]:
    records = list(items)
    status_counter = Counter(item.get("status", "unknown") for item in records)
    priority_counter = Counter(item.get("priority", "unknown") for item in records)
    reason_counter = Counter(reason for item in records for reason in item.get("review_reasons", []))
    return {
        "total_items": len(records),
        "status_breakdown": dict(status_counter),
        "priority_breakdown": dict(priority_counter),
        "review_reason_breakdown": dict(reason_counter),
        "top_items": sorted(
            records,
            key=lambda item: (_priority_sort_key(item.get("priority", "low")), item.get("updated_at", "")),
        )[:10],
    }

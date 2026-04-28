from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from datetime import date, datetime, timedelta
import json
import os
from pathlib import Path
import sqlite3
from typing import Any
from zoneinfo import ZoneInfo

from .review_queue import DEFAULT_DB_PATH

DEFAULT_TIMEZONE = os.getenv("WORKFLOW_TIMEZONE", "Asia/Hong_Kong")


def resolve_agent_workflow_db_path(db_path: str | Path | None = None) -> Path:
    configured = db_path or os.getenv("AGENT_WORKFLOW_DB_PATH") or os.getenv("REVIEW_QUEUE_DB_PATH")
    return Path(configured) if configured else DEFAULT_DB_PATH


def _utc_now_iso() -> str:
    return datetime.now(ZoneInfo(DEFAULT_TIMEZONE)).isoformat()


def _workflow_today() -> date:
    return datetime.now(ZoneInfo(DEFAULT_TIMEZONE)).date()


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    if not denominator:
        return 0.0
    return round(float(numerator) / float(denominator), 4)


def _distribution(counter: Counter[str], *, allowed_keys: list[str] | None = None) -> dict[str, float]:
    keys = allowed_keys or sorted(counter.keys())
    total = sum(counter.values())
    if total <= 0:
        return {key: 0.0 for key in keys}
    return {key: round(counter.get(key, 0) / total, 4) for key in keys}


def _variation_distance(current: dict[str, float], baseline: dict[str, float]) -> float:
    all_keys = set(current) | set(baseline)
    score = sum(abs(current.get(key, 0.0) - baseline.get(key, 0.0)) for key in all_keys) / 2
    return round(score, 4)


class AgentWorkflowRepository:
    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = resolve_agent_workflow_db_path(db_path)
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

    def _table_columns(self, connection: sqlite3.Connection, table_name: str) -> set[str]:
        rows = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
        return {row["name"] for row in rows}

    def _table_exists(self, connection: sqlite3.Connection, table_name: str) -> bool:
        row = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table_name,),
        ).fetchone()
        return row is not None

    def _ensure_column(
        self,
        connection: sqlite3.Connection,
        *,
        table_name: str,
        column_name: str,
        definition: str,
    ) -> None:
        if column_name in self._table_columns(connection, table_name):
            return
        connection.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {definition}")

    def initialize(self) -> None:
        with self._managed_connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    input_text TEXT NOT NULL,
                    input_context_json TEXT NOT NULL DEFAULT '{}',
                    model_label TEXT NOT NULL,
                    model_confidence REAL NOT NULL,
                    final_label TEXT NOT NULL,
                    final_confidence REAL NOT NULL,
                    final_source TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    primary_entity TEXT NOT NULL,
                    review_queue_record_id INTEGER,
                    needs_human_review INTEGER NOT NULL,
                    llm_review_json TEXT NOT NULL,
                    alert_json TEXT NOT NULL,
                    workflow_steps_json TEXT NOT NULL,
                    result_payload_json TEXT NOT NULL,
                    explanation TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            self._ensure_column(
                connection,
                table_name="agent_runs",
                column_name="input_context_json",
                definition="TEXT NOT NULL DEFAULT '{}'",
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analysis_run_id INTEGER NOT NULL,
                    severity TEXT NOT NULL,
                    status TEXT NOT NULL,
                    primary_entity TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    final_label TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    reasons_json TEXT NOT NULL,
                    message TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (analysis_run_id) REFERENCES agent_runs(id)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS report_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    report_type TEXT NOT NULL,
                    period_start TEXT NOT NULL,
                    period_end TEXT NOT NULL,
                    summary_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE (report_type, period_start, period_end)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS watchlist_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    company_name TEXT NOT NULL,
                    ticker TEXT,
                    industry TEXT,
                    notes TEXT,
                    is_active INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analysis_run_id INTEGER NOT NULL,
                    feedback_label TEXT NOT NULL,
                    feedback_event_type TEXT,
                    reviewer TEXT,
                    notes TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (analysis_run_id) REFERENCES agent_runs(id)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS retrain_jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    status TEXT NOT NULL,
                    trigger_source TEXT NOT NULL,
                    include_feedback_only INTEGER NOT NULL,
                    requested_by TEXT,
                    notes TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS error_samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analysis_run_id INTEGER NOT NULL UNIQUE,
                    primary_entity TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    final_label TEXT NOT NULL,
                    reasons_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    latest_feedback_label TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (analysis_run_id) REFERENCES agent_runs(id)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS golden_test_cases (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT,
                    input_text TEXT NOT NULL,
                    expected_label TEXT NOT NULL,
                    expected_event_type TEXT,
                    source_name TEXT,
                    notes TEXT,
                    context_json TEXT NOT NULL,
                    is_active INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    def create_run(
        self,
        *,
        analysis_result: dict[str, Any],
        model_decision: dict[str, Any],
        final_decision: dict[str, Any],
        llm_review: dict[str, Any],
        alert: dict[str, Any],
        workflow_steps: list[dict[str, Any]],
        review_queue_record_id: int | None,
    ) -> dict[str, Any]:
        created_at = _utc_now_iso()
        context = analysis_result.get("context") or {}
        primary_entity = (
            analysis_result["entities"]["companies"][0]
            if analysis_result["entities"]["companies"]
            else (
                analysis_result["entities"]["tickers"][0]
                if analysis_result["entities"]["tickers"]
                else (
                    (context.get("company_name") or "").strip()
                    or (context.get("ticker") or "").strip()
                    or "unknown"
                )
            )
        )

        with self._managed_connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO agent_runs (
                    input_text,
                    input_context_json,
                    model_label,
                    model_confidence,
                    final_label,
                    final_confidence,
                    final_source,
                    event_type,
                    primary_entity,
                    review_queue_record_id,
                    needs_human_review,
                    llm_review_json,
                    alert_json,
                    workflow_steps_json,
                    result_payload_json,
                    explanation,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    analysis_result["input_text"],
                    json.dumps(analysis_result.get("context", {}), ensure_ascii=False),
                    model_decision["label"],
                    model_decision["confidence"],
                    final_decision["label"],
                    final_decision["confidence"],
                    final_decision["source"],
                    analysis_result["event"]["type"],
                    primary_entity,
                    review_queue_record_id,
                    1 if analysis_result["risk_alert"]["needs_human_review"] else 0,
                    json.dumps(llm_review, ensure_ascii=False),
                    json.dumps(alert, ensure_ascii=False),
                    json.dumps(workflow_steps, ensure_ascii=False),
                    json.dumps(analysis_result, ensure_ascii=False),
                    analysis_result["explanation"],
                    created_at,
                ),
            )
            run_id = int(cursor.lastrowid)

            alert_record: dict[str, Any] | None = None
            if alert["triggered"]:
                alert_cursor = connection.execute(
                    """
                    INSERT INTO alerts (
                        analysis_run_id,
                        severity,
                        status,
                        primary_entity,
                        event_type,
                        final_label,
                        confidence,
                        reasons_json,
                        message,
                        created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        alert["severity"],
                        alert["status"],
                        primary_entity,
                        analysis_result["event"]["type"],
                        final_decision["label"],
                        final_decision["confidence"],
                        json.dumps(alert["reasons"], ensure_ascii=False),
                        alert["message"],
                        created_at,
                    ),
                )
                alert_record = {
                    "id": int(alert_cursor.lastrowid),
                    "triggered": True,
                    "analysis_run_id": run_id,
                    "severity": alert["severity"],
                    "status": alert["status"],
                    "primary_entity": primary_entity,
                    "event_type": analysis_result["event"]["type"],
                    "final_label": final_decision["label"],
                    "confidence": round(float(final_decision["confidence"]), 4),
                    "reasons": alert["reasons"],
                    "message": alert["message"],
                    "created_at": created_at,
                    "watchlist_match": False,
                }

            error_reasons: list[str] = []
            if final_decision["source"] == "llm_rejudge":
                error_reasons.append("llm_override")
            if analysis_result["risk_alert"]["needs_human_review"]:
                error_reasons.append("manual_review")
            if alert["triggered"]:
                error_reasons.append("alert_triggered")
            error_reasons.extend(analysis_result.get("risk_alert", {}).get("reasons", [])[:3])
            self._upsert_error_sample(
                connection,
                analysis_run_id=run_id,
                primary_entity=primary_entity,
                event_type=analysis_result["event"]["type"],
                final_label=final_decision["label"],
                reasons=error_reasons,
            )

        return {
            "id": run_id,
            "created_at": created_at,
            "primary_entity": primary_entity,
            "event_type": analysis_result["event"]["type"],
            "final_label": final_decision["label"],
            "final_confidence": round(float(final_decision["confidence"]), 4),
            "final_source": final_decision["source"],
            "review_queue_record_id": review_queue_record_id,
            "alert": alert_record,
        }

    def _active_watchlist_index(self, connection: sqlite3.Connection) -> tuple[set[str], set[str]]:
        rows = connection.execute(
            """
            SELECT company_name, ticker
            FROM watchlist_items
            WHERE is_active = 1
            """
        ).fetchall()
        company_names = {row["company_name"].strip().lower() for row in rows if row["company_name"]}
        tickers = {row["ticker"].strip().upper() for row in rows if row["ticker"]}
        return company_names, tickers

    def _normalize_context_payload(self, context: dict[str, Any] | None) -> dict[str, Any]:
        payload = context or {}
        return {
            "news_source": payload.get("news_source"),
            "source_name": payload.get("source_name"),
            "source_url": payload.get("source_url"),
            "published_at": payload.get("published_at"),
            "company_name": payload.get("company_name"),
            "ticker": payload.get("ticker"),
            "industry": payload.get("industry"),
            "event_type": payload.get("event_type"),
            "historical_announcements": payload.get("historical_announcements", []),
        }

    def _derive_source_name(self, context: dict[str, Any]) -> str:
        return context.get("source_name") or context.get("news_source") or "unknown"

    def _upsert_error_sample(
        self,
        connection: sqlite3.Connection,
        *,
        analysis_run_id: int,
        primary_entity: str,
        event_type: str,
        final_label: str,
        reasons: list[str],
        latest_feedback_label: str | None = None,
    ) -> None:
        deduped_reasons = list(dict.fromkeys([reason for reason in reasons if reason]))
        if not deduped_reasons:
            return

        now = _utc_now_iso()
        existing = connection.execute(
            "SELECT id, reasons_json, latest_feedback_label FROM error_samples WHERE analysis_run_id = ?",
            (analysis_run_id,),
        ).fetchone()
        if existing:
            merged_reasons = list(dict.fromkeys(json.loads(existing["reasons_json"]) + deduped_reasons))
            connection.execute(
                """
                UPDATE error_samples
                SET primary_entity = ?,
                    event_type = ?,
                    final_label = ?,
                    reasons_json = ?,
                    latest_feedback_label = COALESCE(?, latest_feedback_label),
                    status = 'open',
                    updated_at = ?
                WHERE analysis_run_id = ?
                """,
                (
                    primary_entity,
                    event_type,
                    final_label,
                    json.dumps(merged_reasons, ensure_ascii=False),
                    latest_feedback_label,
                    now,
                    analysis_run_id,
                ),
            )
            return

        connection.execute(
            """
            INSERT INTO error_samples (
                analysis_run_id,
                primary_entity,
                event_type,
                final_label,
                reasons_json,
                status,
                latest_feedback_label,
                created_at,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                analysis_run_id,
                primary_entity,
                event_type,
                final_label,
                json.dumps(deduped_reasons, ensure_ascii=False),
                "open",
                latest_feedback_label,
                now,
                now,
            ),
        )

    def _payload_matches_watchlist(
        self,
        payload: dict[str, Any],
        *,
        company_names: set[str],
        tickers: set[str],
        primary_entity: str,
    ) -> bool:
        if primary_entity and primary_entity.strip().lower() in company_names:
            return True

        entity_block = payload.get("entities", {})
        companies = {company.strip().lower() for company in entity_block.get("companies", []) if company}
        if companies.intersection(company_names):
            return True

        payload_tickers = {ticker.strip().upper() for ticker in entity_block.get("tickers", []) if ticker}
        if payload_tickers.intersection(tickers):
            return True

        context = payload.get("context", {})
        if context:
            ctx_company = context.get("company_name", "")
            if ctx_company and ctx_company.strip().lower() in company_names:
                return True
            ctx_ticker = context.get("ticker", "")
            if ctx_ticker and ctx_ticker.strip().upper() in tickers:
                return True

        return False

    def _row_to_feedback_record(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": int(row["id"]),
            "analysis_run_id": int(row["analysis_run_id"]),
            "feedback_label": row["feedback_label"],
            "feedback_event_type": row["feedback_event_type"],
            "reviewer": row["reviewer"],
            "notes": row["notes"],
            "created_at": row["created_at"],
        }

    def _fetch_feedback_maps(
        self,
        connection: sqlite3.Connection,
        run_ids: list[int],
    ) -> tuple[dict[int, int], dict[int, dict[str, Any]], dict[int, list[str]], dict[int, list[str]]]:
        if not run_ids:
            return {}, {}, {}, {}

        placeholders = ",".join("?" for _ in run_ids)
        rows = connection.execute(
            f"""
            SELECT *
            FROM feedback_items
            WHERE analysis_run_id IN ({placeholders})
            ORDER BY created_at DESC
            """,
            run_ids,
        ).fetchall()

        counts: dict[int, int] = {}
        latest: dict[int, dict[str, Any]] = {}
        labels: dict[int, list[str]] = {}
        event_types: dict[int, list[str]] = {}
        for row in rows:
            analysis_run_id = int(row["analysis_run_id"])
            counts[analysis_run_id] = counts.get(analysis_run_id, 0) + 1
            labels.setdefault(analysis_run_id, [])
            labels[analysis_run_id].append(row["feedback_label"])
            if row["feedback_event_type"]:
                event_types.setdefault(analysis_run_id, [])
                event_types[analysis_run_id].append(row["feedback_event_type"])
            if analysis_run_id not in latest:
                latest[analysis_run_id] = self._row_to_feedback_record(row)

        labels = {run_id: list(dict.fromkeys(values)) for run_id, values in labels.items()}
        event_types = {run_id: list(dict.fromkeys(values)) for run_id, values in event_types.items()}
        return counts, latest, labels, event_types

    def list_results(
        self,
        *,
        limit: int = 50,
        label: str | None = None,
        event_type: str | None = None,
        entity_query: str | None = None,
        source: str | None = None,
        error_only: bool = False,
        watchlist_only: bool = False,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM agent_runs"
        conditions: list[str] = []
        params: list[Any] = []

        if label:
            conditions.append("final_label = ?")
            params.append(label)
        if event_type:
            conditions.append("event_type = ?")
            params.append(event_type)
        if entity_query:
            conditions.append("LOWER(primary_entity) LIKE ?")
            params.append(f"%{entity_query.strip().lower()}%")
        if source:
            conditions.append("final_source = ?")
            params.append(source)
        if error_only:
            conditions.append(
                "(id IN (SELECT analysis_run_id FROM error_samples WHERE status = 'open') OR final_source = 'llm_rejudge' OR needs_human_review = 1 OR id IN (SELECT DISTINCT analysis_run_id FROM feedback_items))"
            )

        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        with self._managed_connection() as connection:
            rows = connection.execute(query, params).fetchall()
            run_ids = [int(row["id"]) for row in rows]
            feedback_count_map, latest_feedback_map, feedback_label_map, feedback_event_type_map = self._fetch_feedback_maps(connection, run_ids)
            error_rows = connection.execute(
                f"""
                SELECT analysis_run_id
                FROM error_samples
                WHERE analysis_run_id IN ({",".join("?" for _ in run_ids)})
                """ if run_ids else "SELECT analysis_run_id FROM error_samples WHERE 1 = 0",
                run_ids,
            ).fetchall()
            error_run_ids = {int(row["analysis_run_id"]) for row in error_rows}
            alert_rows = connection.execute(
                f"""
                SELECT *
                FROM alerts
                WHERE analysis_run_id IN ({",".join("?" for _ in run_ids)}) 
                ORDER BY created_at DESC
                """ if run_ids else "SELECT * FROM alerts WHERE 1 = 0",
                run_ids,
            ).fetchall()
            alert_map: dict[int, dict[str, Any]] = {}
            for row in alert_rows:
                analysis_run_id = int(row["analysis_run_id"])
                if analysis_run_id in alert_map:
                    continue
                alert_map[analysis_run_id] = {
                    "id": int(row["id"]),
                    "analysis_run_id": analysis_run_id,
                    "severity": row["severity"],
                    "status": row["status"],
                    "primary_entity": row["primary_entity"],
                    "event_type": row["event_type"],
                    "final_label": row["final_label"],
                    "confidence": round(float(row["confidence"]), 4),
                    "reasons": json.loads(row["reasons_json"]),
                    "message": row["message"],
                    "created_at": row["created_at"],
                }
            company_names, tickers = self._active_watchlist_index(connection)

        items: list[dict[str, Any]] = []
        for row in rows:
            payload = json.loads(row["result_payload_json"])
            context = json.loads(row["input_context_json"]) if row["input_context_json"] else payload.get("context", {})
            normalized_context = {
                "news_source": context.get("news_source"),
                "source_name": context.get("source_name"),
                "source_url": context.get("source_url"),
                "published_at": context.get("published_at"),
                "company_name": context.get("company_name"),
                "ticker": context.get("ticker"),
                "industry": context.get("industry"),
                "event_type": context.get("event_type"),
                "historical_announcements": context.get("historical_announcements", []),
            }
            watchlist_match = self._payload_matches_watchlist(
                payload,
                company_names=company_names,
                tickers=tickers,
                primary_entity=row["primary_entity"],
            )
            if watchlist_only and not watchlist_match:
                continue

            analysis_run_id = int(row["id"])
            items.append(
                {
                    "id": analysis_run_id,
                    "created_at": row["created_at"],
                    "input_text": row["input_text"],
                    "context": normalized_context,
                    "primary_entity": row["primary_entity"],
                    "event_type": row["event_type"],
                    "model_label": row["model_label"],
                    "model_confidence": round(float(row["model_confidence"]), 4),
                    "final_label": row["final_label"],
                    "final_confidence": round(float(row["final_confidence"]), 4),
                    "final_source": row["final_source"],
                    "needs_human_review": bool(row["needs_human_review"]),
                    "review_queue_record_id": row["review_queue_record_id"],
                    "explanation": row["explanation"],
                    "feedback_count": feedback_count_map.get(analysis_run_id, 0),
                    "feedback_labels": feedback_label_map.get(analysis_run_id, []),
                    "feedback_event_types": feedback_event_type_map.get(analysis_run_id, []),
                    "watchlist_match": watchlist_match,
                    "in_error_pool": analysis_run_id in error_run_ids,
                    "latest_feedback": latest_feedback_map.get(analysis_run_id),
                    "result": {**payload, "context": payload.get("context", normalized_context)},
                    "llm_review": json.loads(row["llm_review_json"]),
                    "alert": alert_map.get(analysis_run_id),
                }
            )

        return items

    def list_alerts(
        self,
        status: str | None = None,
        severity: str | None = None,
        limit: int = 50,
        watchlist_only: bool = False,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM alerts"
        conditions: list[str] = []
        params: list[Any] = []
        if status:
            conditions.append("status = ?")
            params.append(status)
        if severity:
            conditions.append("severity = ?")
            params.append(severity)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        with self._managed_connection() as connection:
            rows = connection.execute(query, params).fetchall()
            company_names, tickers = self._active_watchlist_index(connection)

            run_ids = [int(row["analysis_run_id"]) for row in rows]
            context_map: dict[int, dict[str, Any]] = {}
            if run_ids:
                placeholders = ",".join("?" * len(run_ids))
                ctx_rows = connection.execute(
                    f"SELECT id, input_context_json, result_payload_json FROM agent_runs WHERE id IN ({placeholders})",
                    run_ids,
                ).fetchall()
                for cr in ctx_rows:
                    ctx = json.loads(cr["input_context_json"]) if cr["input_context_json"] else {}
                    if not ctx:
                        payload = json.loads(cr["result_payload_json"]) if cr["result_payload_json"] else {}
                        ctx = payload.get("context", {})
                    context_map[int(cr["id"])] = ctx

        items: list[dict[str, Any]] = []
        for row in rows:
            pe = row["primary_entity"]
            match = pe.strip().lower() in company_names if pe else False
            if not match:
                ctx = context_map.get(int(row["analysis_run_id"]), {})
                ctx_company = ctx.get("company_name", "")
                if ctx_company and ctx_company.strip().lower() in company_names:
                    match = True
                if not match:
                    ctx_ticker = ctx.get("ticker", "")
                    if ctx_ticker and ctx_ticker.strip().upper() in tickers:
                        match = True
            items.append(
                {
                    "id": int(row["id"]),
                    "analysis_run_id": int(row["analysis_run_id"]),
                    "severity": row["severity"],
                    "status": row["status"],
                    "primary_entity": row["primary_entity"],
                    "event_type": row["event_type"],
                    "final_label": row["final_label"],
                    "confidence": round(float(row["confidence"]), 4),
                    "reasons": json.loads(row["reasons_json"]),
                    "message": row["message"],
                    "created_at": row["created_at"],
                    "watchlist_match": match,
                }
            )
        if watchlist_only:
            return [item for item in items if item["watchlist_match"]]
        return items

    def add_watchlist_item(
        self,
        *,
        company_name: str,
        ticker: str | None = None,
        industry: str | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        normalized_company_name = company_name.strip()
        normalized_ticker = ticker.strip().upper() if ticker else None
        if not normalized_company_name:
            raise ValueError("company_name cannot be empty.")

        now = _utc_now_iso()
        with self._managed_connection() as connection:
            existing = connection.execute(
                """
                SELECT id
                FROM watchlist_items
                WHERE LOWER(company_name) = ? AND COALESCE(ticker, '') = COALESCE(?, '')
                """,
                (normalized_company_name.lower(), normalized_ticker),
            ).fetchone()
            if existing:
                connection.execute(
                    """
                    UPDATE watchlist_items
                    SET industry = ?, notes = ?, is_active = 1, updated_at = ?
                    WHERE id = ?
                    """,
                    (industry, notes, now, existing["id"]),
                )
                item_id = int(existing["id"])
            else:
                cursor = connection.execute(
                    """
                    INSERT INTO watchlist_items (
                        company_name,
                        ticker,
                        industry,
                        notes,
                        is_active,
                        created_at,
                        updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (normalized_company_name, normalized_ticker, industry, notes, 1, now, now),
                )
                item_id = int(cursor.lastrowid)

        items = self.list_watchlist(limit=200)
        for item in items:
            if item["id"] == item_id:
                return item
        raise ValueError("Failed to persist watchlist item.")

    def list_watchlist(self, *, limit: int = 100, active_only: bool = True) -> list[dict[str, Any]]:
        query = "SELECT * FROM watchlist_items"
        params: list[Any] = []
        if active_only:
            query += " WHERE is_active = 1"
        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)

        with self._managed_connection() as connection:
            rows = connection.execute(query, params).fetchall()
            result_counts = {
                row["primary_entity"]: int(row["count"])
                for row in connection.execute(
                    """
                    SELECT primary_entity, COUNT(*) AS count
                    FROM agent_runs
                    GROUP BY primary_entity
                    """
                ).fetchall()
            }
            alert_counts = {
                row["primary_entity"]: int(row["count"])
                for row in connection.execute(
                    """
                    SELECT primary_entity, COUNT(*) AS count
                    FROM alerts
                    GROUP BY primary_entity
                    """
                ).fetchall()
            }

        return [
            {
                "id": int(row["id"]),
                "company_name": row["company_name"],
                "ticker": row["ticker"],
                "industry": row["industry"],
                "notes": row["notes"],
                "is_active": bool(row["is_active"]),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "recent_result_count": result_counts.get(row["company_name"], 0),
                "recent_alert_count": alert_counts.get(row["company_name"], 0),
            }
            for row in rows
        ]

    def create_feedback(
        self,
        *,
        analysis_run_id: int,
        feedback_label: str,
        feedback_event_type: str | None = None,
        reviewer: str | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        created_at = _utc_now_iso()
        with self._managed_connection() as connection:
            existing = connection.execute(
                "SELECT id, primary_entity, event_type, final_label FROM agent_runs WHERE id = ?",
                (analysis_run_id,),
            ).fetchone()
            if not existing:
                raise ValueError(f"Unknown analysis_run_id: {analysis_run_id}")

            cursor = connection.execute(
                """
                INSERT INTO feedback_items (
                    analysis_run_id,
                    feedback_label,
                    feedback_event_type,
                    reviewer,
                    notes,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (analysis_run_id, feedback_label, feedback_event_type, reviewer, notes, created_at),
            )
            feedback_id = int(cursor.lastrowid)

            row = connection.execute(
                "SELECT * FROM feedback_items WHERE id = ?",
                (feedback_id,),
            ).fetchone()

            correction_reasons: list[str] = []
            if feedback_label != existing["final_label"]:
                correction_reasons.append("user_correction_label")
            if feedback_event_type and feedback_event_type != existing["event_type"]:
                correction_reasons.append("user_correction_event")
            if correction_reasons:
                self._upsert_error_sample(
                    connection,
                    analysis_run_id=analysis_run_id,
                    primary_entity=existing["primary_entity"],
                    event_type=existing["event_type"],
                    final_label=existing["final_label"],
                    reasons=correction_reasons,
                    latest_feedback_label=feedback_label,
                )

        return self._row_to_feedback_record(row)

    def list_feedback(self, *, limit: int = 100, analysis_run_id: int | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM feedback_items"
        params: list[Any] = []
        if analysis_run_id is not None:
            query += " WHERE analysis_run_id = ?"
            params.append(analysis_run_id)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        with self._managed_connection() as connection:
            rows = connection.execute(query, params).fetchall()

        return [self._row_to_feedback_record(row) for row in rows]

    def list_error_samples(self, *, limit: int = 100, status: str = "open") -> list[dict[str, Any]]:
        with self._managed_connection() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM error_samples
                WHERE status = ?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (status, limit),
            ).fetchall()

        return [
            {
                "id": int(row["id"]),
                "analysis_run_id": int(row["analysis_run_id"]),
                "primary_entity": row["primary_entity"],
                "event_type": row["event_type"],
                "final_label": row["final_label"],
                "reasons": json.loads(row["reasons_json"]),
                "status": row["status"],
                "latest_feedback_label": row["latest_feedback_label"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]

    def add_golden_test_case(
        self,
        *,
        input_text: str,
        expected_label: str,
        expected_event_type: str | None = None,
        title: str | None = None,
        source_name: str | None = None,
        notes: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = _utc_now_iso()
        normalized_context = self._normalize_context_payload(context)
        with self._managed_connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO golden_test_cases (
                    title,
                    input_text,
                    expected_label,
                    expected_event_type,
                    source_name,
                    notes,
                    context_json,
                    is_active,
                    created_at,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    title,
                    input_text,
                    expected_label,
                    expected_event_type,
                    source_name,
                    notes,
                    json.dumps(normalized_context, ensure_ascii=False),
                    1,
                    now,
                    now,
                ),
            )
            case_id = int(cursor.lastrowid)
            row = connection.execute("SELECT * FROM golden_test_cases WHERE id = ?", (case_id,)).fetchone()

        return {
            "id": int(row["id"]),
            "title": row["title"],
            "input_text": row["input_text"],
            "expected_label": row["expected_label"],
            "expected_event_type": row["expected_event_type"],
            "source_name": row["source_name"],
            "notes": row["notes"],
            "context": self._normalize_context_payload(json.loads(row["context_json"])),
            "is_active": bool(row["is_active"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def list_golden_test_cases(self, *, limit: int = 100, active_only: bool = True) -> list[dict[str, Any]]:
        query = "SELECT * FROM golden_test_cases"
        params: list[Any] = []
        if active_only:
            query += " WHERE is_active = 1"
        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)

        with self._managed_connection() as connection:
            rows = connection.execute(query, params).fetchall()

        return [
            {
                "id": int(row["id"]),
                "title": row["title"],
                "input_text": row["input_text"],
                "expected_label": row["expected_label"],
                "expected_event_type": row["expected_event_type"],
                "source_name": row["source_name"],
                "notes": row["notes"],
                "context": self._normalize_context_payload(json.loads(row["context_json"])),
                "is_active": bool(row["is_active"]),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]

    def create_retrain_job(
        self,
        *,
        trigger_source: str,
        include_feedback_only: bool,
        requested_by: str | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        now = _utc_now_iso()
        with self._managed_connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO retrain_jobs (
                    status,
                    trigger_source,
                    include_feedback_only,
                    requested_by,
                    notes,
                    created_at,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                ("queued", trigger_source, 1 if include_feedback_only else 0, requested_by, notes, now, now),
            )
            job_id = int(cursor.lastrowid)
            row = connection.execute(
                "SELECT * FROM retrain_jobs WHERE id = ?",
                (job_id,),
            ).fetchone()

        return {
            "id": int(row["id"]),
            "status": row["status"],
            "trigger_source": row["trigger_source"],
            "include_feedback_only": bool(row["include_feedback_only"]),
            "requested_by": row["requested_by"],
            "notes": row["notes"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def _select_runs_for_period(
        self,
        connection: sqlite3.Connection,
        *,
        period_start_iso: str,
        period_end_iso: str,
    ) -> list[sqlite3.Row]:
        return connection.execute(
            """
            SELECT id, final_label, event_type, primary_entity, created_at, review_queue_record_id, final_source,
                   result_payload_json, input_context_json
            FROM agent_runs
            WHERE substr(created_at, 1, 10) >= ? AND substr(created_at, 1, 10) < ?
            """,
            (period_start_iso, period_end_iso),
        ).fetchall()

    def _select_feedback_for_period(
        self,
        connection: sqlite3.Connection,
        *,
        period_start_iso: str,
        period_end_iso: str,
    ) -> list[sqlite3.Row]:
        return connection.execute(
            """
            SELECT f.feedback_label, f.feedback_event_type, f.created_at,
                   r.final_label, r.event_type
            FROM feedback_items f
            JOIN agent_runs r ON r.id = f.analysis_run_id
            WHERE substr(f.created_at, 1, 10) >= ? AND substr(f.created_at, 1, 10) < ?
            """,
            (period_start_iso, period_end_iso),
        ).fetchall()

    def _build_drift_snapshot(
        self,
        *,
        current_counter: Counter[str],
        baseline_counter: Counter[str],
        allowed_keys: list[str] | None = None,
        baseline_period_start: str | None = None,
        baseline_period_end: str | None = None,
        threshold: float = 0.2,
    ) -> dict[str, Any]:
        keys = allowed_keys or sorted(set(current_counter.keys()) | set(baseline_counter.keys()))
        current_distribution = _distribution(current_counter, allowed_keys=keys)
        baseline_distribution = _distribution(baseline_counter, allowed_keys=keys)
        score = _variation_distance(current_distribution, baseline_distribution)
        return {
            "score": score,
            "changed": score >= threshold,
            "current_distribution": current_distribution,
            "baseline_distribution": baseline_distribution,
            "baseline_period_start": baseline_period_start,
            "baseline_period_end": baseline_period_end,
        }

    def _compute_per_class_metrics(
        self,
        rows: list[sqlite3.Row],
        *,
        baseline_rows: list[sqlite3.Row] | None = None,
    ) -> list[dict[str, Any]]:
        labels = ["negative", "neutral", "positive"]

        def calculate(values: list[sqlite3.Row]) -> dict[str, dict[str, Any]]:
            metrics: dict[str, dict[str, Any]] = {}
            for label in labels:
                tp = sum(1 for row in values if row["feedback_label"] == label and row["final_label"] == label)
                fp = sum(1 for row in values if row["feedback_label"] != label and row["final_label"] == label)
                fn = sum(1 for row in values if row["feedback_label"] == label and row["final_label"] != label)
                precision = _safe_ratio(tp, tp + fp)
                recall = _safe_ratio(tp, tp + fn)
                f1 = _safe_ratio(2 * precision * recall, precision + recall) if (precision + recall) else 0.0
                metrics[label] = {
                    "precision": precision,
                    "recall": recall,
                    "f1": round(f1, 4),
                    "support": sum(1 for row in values if row["feedback_label"] == label),
                }
            return metrics

        current_metrics = calculate(rows)
        baseline_metrics = calculate(baseline_rows or [])
        items: list[dict[str, Any]] = []
        for label in labels:
            current = current_metrics[label]
            baseline = baseline_metrics[label]
            items.append(
                {
                    "label": label,
                    "precision": current["precision"],
                    "recall": current["recall"],
                    "f1": current["f1"],
                    "support": current["support"],
                    "precision_delta": round(current["precision"] - baseline["precision"], 4),
                    "recall_delta": round(current["recall"] - baseline["recall"], 4),
                    "f1_delta": round(current["f1"] - baseline["f1"], 4),
                }
            )
        return items

    def _build_monitoring_snapshot(
        self,
        connection: sqlite3.Connection,
        *,
        period_start_iso: str,
        period_end_iso: str,
        baseline_period_start_iso: str,
        baseline_period_end_iso: str,
        runs: list[sqlite3.Row],
    ) -> dict[str, Any]:
        baseline_runs = self._select_runs_for_period(
            connection,
            period_start_iso=baseline_period_start_iso,
            period_end_iso=baseline_period_end_iso,
        )
        current_feedback = self._select_feedback_for_period(
            connection,
            period_start_iso=period_start_iso,
            period_end_iso=period_end_iso,
        )
        baseline_feedback = self._select_feedback_for_period(
            connection,
            period_start_iso=baseline_period_start_iso,
            period_end_iso=baseline_period_end_iso,
        )

        low_confidence_count = 0
        current_sentiment_counter: Counter[str] = Counter()
        baseline_sentiment_counter: Counter[str] = Counter(row["final_label"] for row in baseline_runs)
        current_source_counter: Counter[str] = Counter()
        baseline_source_counter: Counter[str] = Counter()

        for row in runs:
            payload = json.loads(row["result_payload_json"])
            context = self._normalize_context_payload(json.loads(row["input_context_json"]) if row["input_context_json"] else payload.get("context", {}))
            current_sentiment_counter[row["final_label"]] += 1
            current_source_counter[self._derive_source_name(context)] += 1
            review_queue_item = payload.get("review_queue_item") or {}
            review_reasons = review_queue_item.get("review_reasons", [])
            if payload.get("sentiment", {}).get("abstained") or "low_confidence" in review_reasons:
                low_confidence_count += 1

        for row in baseline_runs:
            payload = json.loads(row["result_payload_json"])
            context = self._normalize_context_payload(json.loads(row["input_context_json"]) if row["input_context_json"] else payload.get("context", {}))
            baseline_source_counter[self._derive_source_name(context)] += 1

        correction_count = sum(
            1
            for row in current_feedback
            if row["feedback_label"] != row["final_label"] or (row["feedback_event_type"] and row["feedback_event_type"] != row["event_type"])
        )
        sampled_review_count = 0
        if self._table_exists(connection, "review_queue_items"):
            sampled_review_count = int(
                connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM review_queue_items
                    WHERE substr(created_at, 1, 10) >= ? AND substr(created_at, 1, 10) < ?
                      AND review_reasons_json LIKE '%audit_sample%'
                    """,
                    (period_start_iso, period_end_iso),
                ).fetchone()[0]
            )

        return {
            "inference_volume": len(runs),
            "low_confidence_ratio": _safe_ratio(low_confidence_count, len(runs)),
            "user_correction_rate": _safe_ratio(correction_count, len(current_feedback)),
            "reviewed_feedback_count": len(current_feedback),
            "sampled_review_count": sampled_review_count,
            "class_drift": self._build_drift_snapshot(
                current_counter=current_sentiment_counter,
                baseline_counter=baseline_sentiment_counter,
                allowed_keys=["negative", "neutral", "positive"],
                baseline_period_start=baseline_period_start_iso,
                baseline_period_end=baseline_period_end_iso,
                threshold=0.15,
            ),
            "source_shift": self._build_drift_snapshot(
                current_counter=current_source_counter,
                baseline_counter=baseline_source_counter,
                baseline_period_start=baseline_period_start_iso,
                baseline_period_end=baseline_period_end_iso,
                threshold=0.25,
            ),
            "per_class_metrics": self._compute_per_class_metrics(current_feedback, baseline_rows=baseline_feedback),
        }

    def _build_feedback_loop_assets(
        self,
        connection: sqlite3.Connection,
        *,
        report_type: str,
        monitoring: dict[str, Any],
    ) -> dict[str, Any]:
        error_sample_pool_size = int(connection.execute("SELECT COUNT(*) FROM error_samples WHERE status = 'open'").fetchone()[0])
        golden_test_set_size = int(connection.execute("SELECT COUNT(*) FROM golden_test_cases WHERE is_active = 1").fetchone()[0])
        open_retrain_jobs = int(
            connection.execute(
                "SELECT COUNT(*) FROM retrain_jobs WHERE status IN ('queued', 'running')"
            ).fetchone()[0]
        )
        periodic_retrain_due = False
        if report_type == "weekly" and open_retrain_jobs == 0:
            periodic_retrain_due = (
                monitoring["user_correction_rate"] >= 0.15
                or monitoring["class_drift"]["changed"]
                or monitoring["source_shift"]["changed"]
                or monitoring["reviewed_feedback_count"] >= 10
            )

        return {
            "error_sample_pool_size": error_sample_pool_size,
            "golden_test_set_size": golden_test_set_size,
            "open_retrain_jobs": open_retrain_jobs,
            "periodic_retrain_due": periodic_retrain_due,
            "scheduled_retrain_created": False,
        }

    def generate_report(self, report_type: str, reference_date: date | None = None) -> dict[str, Any]:
        if report_type not in {"daily", "weekly"}:
            raise ValueError(f"Unsupported report type: {report_type}")

        ref_date = reference_date or _workflow_today()
        if report_type == "daily":
            period_start = ref_date
            period_end = ref_date + timedelta(days=1)
        else:
            period_start = ref_date - timedelta(days=ref_date.weekday())
            period_end = period_start + timedelta(days=7)

        period_start_iso = period_start.isoformat()
        period_end_iso = period_end.isoformat()
        baseline_period_start = period_start - (period_end - period_start)
        baseline_period_end = period_start
        baseline_period_start_iso = baseline_period_start.isoformat()
        baseline_period_end_iso = baseline_period_end.isoformat()

        with self._managed_connection() as connection:
            runs = self._select_runs_for_period(
                connection,
                period_start_iso=period_start_iso,
                period_end_iso=period_end_iso,
            )
            alerts = connection.execute(
                """
                SELECT severity, primary_entity
                FROM alerts
                WHERE substr(created_at, 1, 10) >= ? AND substr(created_at, 1, 10) < ?
                """,
                (period_start_iso, period_end_iso),
            ).fetchall()
            review_queue_count = 0
            if self._table_exists(connection, "review_queue_items"):
                review_queue_count = connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM review_queue_items
                    WHERE substr(created_at, 1, 10) >= ? AND substr(created_at, 1, 10) < ?
                    """,
                    (period_start_iso, period_end_iso),
                ).fetchone()[0]
            monitoring = self._build_monitoring_snapshot(
                connection,
                period_start_iso=period_start_iso,
                period_end_iso=period_end_iso,
                baseline_period_start_iso=baseline_period_start_iso,
                baseline_period_end_iso=baseline_period_end_iso,
                runs=runs,
            )
            feedback_loop_assets = self._build_feedback_loop_assets(
                connection,
                report_type=report_type,
                monitoring=monitoring,
            )

        sentiment_counter = Counter(row["final_label"] for row in runs)
        event_counter = Counter(row["event_type"] for row in runs)
        entity_counter = Counter(
            row["primary_entity"] for row in runs if row["primary_entity"] and row["primary_entity"] != "unknown"
        )
        alert_counter = Counter(row["severity"] for row in alerts)
        alert_entity_counter = Counter(row["primary_entity"] for row in alerts if row["primary_entity"])

        summary = {
            "report_type": report_type,
            "period_start": period_start_iso,
            "period_end": period_end_iso,
            "generated_at": _utc_now_iso(),
            "total_runs": len(runs),
            "alert_count": len(alerts),
            "review_queue_count": int(review_queue_count),
            "sentiment_breakdown": dict(sentiment_counter),
            "event_breakdown": dict(event_counter),
            "alert_breakdown": dict(alert_counter),
            "top_entities": [{"name": name, "count": count} for name, count in entity_counter.most_common(5)],
            "top_alert_entities": [{"name": name, "count": count} for name, count in alert_entity_counter.most_common(5)],
            "monitoring": monitoring,
            "feedback_loop_assets": feedback_loop_assets,
        }

        with self._managed_connection() as connection:
            connection.execute(
                """
                INSERT INTO report_snapshots (
                    report_type,
                    period_start,
                    period_end,
                    summary_json,
                    created_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(report_type, period_start, period_end)
                DO UPDATE SET summary_json = excluded.summary_json, created_at = excluded.created_at
                """,
                (
                    report_type,
                    period_start_iso,
                    period_end_iso,
                    json.dumps(summary, ensure_ascii=False),
                    summary["generated_at"],
                ),
            )

        return summary

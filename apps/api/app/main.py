from __future__ import annotations

from datetime import date
import os
from pathlib import Path
import sys

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware


BASE_DIR = Path(__file__).resolve().parents[3]
MODEL_SERVING_SRC = BASE_DIR / "services" / "model-serving" / "src"
SCHEMAS_PYTHON = BASE_DIR / "packages" / "schemas" / "python"
WORKER_SRC = BASE_DIR / "services" / "worker" / "src"

for path in [MODEL_SERVING_SRC, SCHEMAS_PYTHON, WORKER_SRC]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from model_serving.env import load_local_env

load_local_env()

from finance_schemas import (
    AlertListResponse,
    AnalysisResultListResponse,
    AnalyzeRequest,
    AnalyzeResponse,
    BatchAnalyzeRequest,
    BatchAnalyzeResponse,
    ErrorSampleListResponse,
    FeedbackLoopMaintenanceResponse,
    FeedbackCreateRequest,
    FeedbackListResponse,
    FeedbackRecord,
    GoldenTestCaseCreateRequest,
    GoldenTestCaseListResponse,
    GoldenTestCaseRecord,
    HealthResponse,
    ReportSnapshotResponse,
    RetrainJobRecord,
    RetrainRequest,
    ReviewQueueListResponse,
    ReviewQueueSummaryResponse,
    WatchlistCreateRequest,
    WatchlistListResponse,
    WatchlistRecord,
)
from worker_service import ReviewQueueRepository, get_agent_workflow_service


def _cors_origins() -> list[str]:
    configured = os.getenv("CORS_ALLOW_ORIGINS")
    if configured:
        return [origin.strip() for origin in configured.split(",") if origin.strip()]
    return [
        "http://127.0.0.1:3000",
        "http://localhost:3000",
    ]


app = FastAPI(
    title="Finance Sentiment API",
    version="0.2.0",
    description="FastAPI gateway for finance news sentiment, watchlists, feedback, and retraining.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent_workflow_service = get_agent_workflow_service()
service = agent_workflow_service.analysis_service
review_queue_repository = ReviewQueueRepository()


def _bad_request(exc: ValueError) -> HTTPException:
    return HTTPException(status_code=400, detail=str(exc))


@app.get("/health", response_model=HealthResponse)
@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(**service.health())


@app.post("/analyze", response_model=AnalyzeResponse)
@app.post("/api/analyze", response_model=AnalyzeResponse)
def analyze(payload: AnalyzeRequest) -> AnalyzeResponse:
    try:
        result = agent_workflow_service.run_text(payload.text, context=payload.context.model_dump() if payload.context else None)
    except ValueError as exc:
        raise _bad_request(exc) from exc
    return AnalyzeResponse(**result)


@app.post("/batch-analyze", response_model=BatchAnalyzeResponse)
@app.post("/api/batch-analyze", response_model=BatchAnalyzeResponse)
def batch_analyze(payload: BatchAnalyzeRequest) -> BatchAnalyzeResponse:
    try:
        if payload.items:
            serialized_items = [item.model_dump() for item in payload.items if item.text.strip()]
            if not serialized_items:
                raise HTTPException(status_code=400, detail="items cannot be empty.")
            items = agent_workflow_service.batch_run_requests(serialized_items)
        else:
            texts = [text.strip() for text in (payload.texts or []) if text.strip()]
            if not texts:
                raise HTTPException(status_code=400, detail="texts cannot be empty.")
            items = agent_workflow_service.batch_run_texts(texts)
    except ValueError as exc:
        raise _bad_request(exc) from exc
    return BatchAnalyzeResponse(items=[AnalyzeResponse(**item) for item in items], total=len(items))


@app.get("/results", response_model=AnalysisResultListResponse)
@app.get("/api/results", response_model=AnalysisResultListResponse)
def list_results(
    limit: int = 50,
    label: str | None = None,
    event_type: str | None = None,
    entity_query: str | None = None,
    source: str | None = None,
    error_only: bool = False,
    watchlist_only: bool = False,
) -> AnalysisResultListResponse:
    items = agent_workflow_service.list_results(
        limit=limit,
        label=label,
        event_type=event_type,
        entity_query=entity_query,
        source=source,
        error_only=error_only,
        watchlist_only=watchlist_only,
    )
    return AnalysisResultListResponse(items=items, total=len(items))


@app.post("/watchlist", response_model=WatchlistRecord)
@app.post("/api/watchlist", response_model=WatchlistRecord)
def add_watchlist(payload: WatchlistCreateRequest) -> WatchlistRecord:
    try:
        item = agent_workflow_service.add_watchlist_item(
            company_name=payload.company_name,
            ticker=payload.ticker,
            industry=payload.industry,
            notes=payload.notes,
        )
    except ValueError as exc:
        raise _bad_request(exc) from exc
    return WatchlistRecord(**item)


@app.get("/watchlist", response_model=WatchlistListResponse)
@app.get("/api/watchlist", response_model=WatchlistListResponse)
def list_watchlist(limit: int = 100) -> WatchlistListResponse:
    items = agent_workflow_service.list_watchlist(limit=limit)
    return WatchlistListResponse(items=items, total=len(items))


@app.get("/alerts", response_model=AlertListResponse)
@app.get("/api/alerts", response_model=AlertListResponse)
def list_alerts(
    status: str | None = None,
    severity: str | None = None,
    limit: int = 50,
    watchlist_only: bool = False,
) -> AlertListResponse:
    items = agent_workflow_service.list_alerts(
        status=status,
        severity=severity,
        limit=limit,
        watchlist_only=watchlist_only,
    )
    return AlertListResponse(items=items, total=len(items))


@app.post("/feedback", response_model=FeedbackRecord)
@app.post("/api/feedback", response_model=FeedbackRecord)
def create_feedback(payload: FeedbackCreateRequest) -> FeedbackRecord:
    try:
        item = agent_workflow_service.create_feedback(
            analysis_run_id=payload.analysis_run_id,
            feedback_label=payload.feedback_label,
            feedback_event_type=payload.feedback_event_type,
            reviewer=payload.reviewer,
            notes=payload.notes,
        )
    except ValueError as exc:
        raise _bad_request(exc) from exc
    return FeedbackRecord(**item)


@app.get("/feedback", response_model=FeedbackListResponse)
@app.get("/api/feedback", response_model=FeedbackListResponse)
def list_feedback(limit: int = 100, analysis_run_id: int | None = None) -> FeedbackListResponse:
    items = agent_workflow_service.list_feedback(limit=limit, analysis_run_id=analysis_run_id)
    return FeedbackListResponse(items=items, total=len(items))


@app.post("/retrain", response_model=RetrainJobRecord)
@app.post("/api/retrain", response_model=RetrainJobRecord)
def create_retrain_job(payload: RetrainRequest) -> RetrainJobRecord:
    item = agent_workflow_service.create_retrain_job(
        trigger_source=payload.trigger_source,
        include_feedback_only=payload.include_feedback_only,
        requested_by=payload.requested_by,
        notes=payload.notes,
    )
    return RetrainJobRecord(**item)


@app.get("/error-samples", response_model=ErrorSampleListResponse)
@app.get("/api/error-samples", response_model=ErrorSampleListResponse)
def list_error_samples(limit: int = 100, status: str = "open") -> ErrorSampleListResponse:
    items = agent_workflow_service.list_error_samples(limit=limit, status=status)
    return ErrorSampleListResponse(items=items, total=len(items))


@app.post("/golden-set", response_model=GoldenTestCaseRecord)
@app.post("/api/golden-set", response_model=GoldenTestCaseRecord)
def add_golden_test_case(payload: GoldenTestCaseCreateRequest) -> GoldenTestCaseRecord:
    item = agent_workflow_service.add_golden_test_case(
        input_text=payload.input_text,
        expected_label=payload.expected_label,
        expected_event_type=payload.expected_event_type,
        title=payload.title,
        source_name=payload.source_name,
        notes=payload.notes,
        context=payload.context.model_dump() if payload.context else None,
    )
    return GoldenTestCaseRecord(**item)


@app.get("/golden-set", response_model=GoldenTestCaseListResponse)
@app.get("/api/golden-set", response_model=GoldenTestCaseListResponse)
def list_golden_test_cases(limit: int = 100, active_only: bool = True) -> GoldenTestCaseListResponse:
    items = agent_workflow_service.list_golden_test_cases(limit=limit, active_only=active_only)
    return GoldenTestCaseListResponse(items=items, total=len(items))


@app.post("/feedback-loop/maintenance", response_model=FeedbackLoopMaintenanceResponse)
@app.post("/api/feedback-loop/maintenance", response_model=FeedbackLoopMaintenanceResponse)
def run_feedback_loop_maintenance(
    report_type: str = "weekly",
    date_value: str | None = None,
    sample_limit: int = 12,
) -> FeedbackLoopMaintenanceResponse:
    try:
        reference_date = date.fromisoformat(date_value) if date_value else None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid date_value, expected YYYY-MM-DD.") from exc
    try:
        item = agent_workflow_service.run_feedback_loop_maintenance(
            report_type=report_type,
            reference_date=reference_date,
            sample_limit=sample_limit,
        )
    except ValueError as exc:
        raise _bad_request(exc) from exc
    return FeedbackLoopMaintenanceResponse(**item)


@app.get("/review-queue", response_model=ReviewQueueListResponse)
@app.get("/api/review-queue", response_model=ReviewQueueListResponse)
def list_review_queue(status: str | None = None, limit: int = 50) -> ReviewQueueListResponse:
    try:
        items = review_queue_repository.list_items(status=status, limit=limit)
    except ValueError as exc:
        raise _bad_request(exc) from exc
    return ReviewQueueListResponse(items=items, total=len(items))


@app.get("/review-queue/summary", response_model=ReviewQueueSummaryResponse)
@app.get("/api/review-queue/summary", response_model=ReviewQueueSummaryResponse)
def review_queue_summary() -> ReviewQueueSummaryResponse:
    return ReviewQueueSummaryResponse(**review_queue_repository.get_summary())


@app.get("/reports/daily", response_model=ReportSnapshotResponse)
@app.get("/api/reports/daily", response_model=ReportSnapshotResponse)
def daily_report(date_value: str | None = None) -> ReportSnapshotResponse:
    try:
        reference_date = date.fromisoformat(date_value) if date_value else None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid date_value, expected YYYY-MM-DD.") from exc
    report = agent_workflow_service.generate_report(report_type="daily", reference_date=reference_date)
    return ReportSnapshotResponse(**report)


@app.get("/reports/weekly", response_model=ReportSnapshotResponse)
@app.get("/api/reports/weekly", response_model=ReportSnapshotResponse)
def weekly_report(date_value: str | None = None) -> ReportSnapshotResponse:
    try:
        reference_date = date.fromisoformat(date_value) if date_value else None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid date_value, expected YYYY-MM-DD.") from exc
    report = agent_workflow_service.generate_report(report_type="weekly", reference_date=reference_date)
    return ReportSnapshotResponse(**report)

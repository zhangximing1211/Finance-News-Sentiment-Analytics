from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


SentimentLabel = Literal["positive", "neutral", "negative"]


class HistoricalAnnouncement(BaseModel):
    title: str | None = None
    summary: str = Field(..., min_length=1)
    event_type: str | None = None
    announced_at: str | None = None
    source_name: str | None = None


class AnalysisContext(BaseModel):
    news_source: str | None = None
    source_name: str | None = None
    source_url: str | None = None
    published_at: str | None = None
    company_name: str | None = None
    ticker: str | None = None
    industry: str | None = None
    event_type: str | None = None
    historical_announcements: list[HistoricalAnnouncement] = Field(default_factory=list)


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, description="News article, filing, or announcement content.")
    context: AnalysisContext | None = None


class BatchAnalyzeRequest(BaseModel):
    texts: list[str] | None = Field(default=None, description="Batch of news snippets or announcements.")
    items: list[AnalyzeRequest] | None = Field(default=None, description="Structured batch items with per-item context.")

    @model_validator(mode="after")
    def validate_payload(self) -> "BatchAnalyzeRequest":
        has_texts = bool(self.texts)
        has_items = bool(self.items)
        if not has_texts and not has_items:
            raise ValueError("Either texts or items must be provided.")
        return self


class SentimentResult(BaseModel):
    label: str
    label_zh: str
    confidence: float
    probabilities: dict[str, float]
    decision_label: str
    abstained: bool
    confidence_gap: float
    low_confidence_threshold: float
    source: str | None = None
    overridden_by_llm: bool | None = None
    model_label: str | None = None
    model_label_zh: str | None = None
    model_confidence: float | None = None


class EventResult(BaseModel):
    type: str
    type_zh: str
    matched_signals: list[str]
    secondary_type: str


class Entities(BaseModel):
    companies: list[str]
    tickers: list[str]
    industry: str
    industry_zh: str


class RiskAlert(BaseModel):
    needs_human_review: bool
    message: str
    reasons: list[str]


class ReviewQueueItem(BaseModel):
    record_id: int | None = None
    status: str | None = None
    attempts: int | None = None
    llm_provider: str | None = None
    last_error: str | None = None
    queue: str
    priority: str
    primary_entity: str
    event_type: str
    predicted_label: str
    decision_label: str
    confidence: float
    review_reasons: list[str]
    recommended_action: str
    text_excerpt: str


class SecondaryExplanation(BaseModel):
    provider: str
    template_path: str
    summary: str
    review_note: str
    rationale: str
    llm_ready: bool
    used_external_llm: bool
    prompt_available: bool
    input_excerpt: str
    risk_message: str
    fallback_reason: str | None = None


class AnalyzeResponse(BaseModel):
    input_text: str
    context: AnalysisContext
    sentiment: SentimentResult
    event: EventResult
    entities: Entities
    explanation: str
    risk_alert: RiskAlert
    review_queue_item: ReviewQueueItem | None = None
    secondary_explanation: SecondaryExplanation
    metadata: dict[str, Any]


class BatchAnalyzeResponse(BaseModel):
    items: list[AnalyzeResponse]
    total: int


class ReviewQueueRecord(BaseModel):
    id: int
    status: str
    priority: str
    primary_entity: str
    event_type: str
    predicted_label: str
    decision_label: str
    confidence: float
    review_reasons: list[str]
    recommended_action: str
    text_excerpt: str
    input_text: str
    secondary_explanation: SecondaryExplanation | None = None
    llm_provider: str | None = None
    last_error: str | None = None
    attempts: int
    created_at: str
    updated_at: str


class ReviewQueueListResponse(BaseModel):
    items: list[ReviewQueueRecord]
    total: int


class ReviewQueueSummaryResponse(BaseModel):
    total_items: int
    status_breakdown: dict[str, int]
    priority_breakdown: dict[str, int]
    review_reason_breakdown: dict[str, int]
    pending_count: int
    processing_count: int
    ready_for_review_count: int
    failed_count: int


class AlertRecord(BaseModel):
    id: int
    analysis_run_id: int
    severity: str
    status: str
    primary_entity: str
    event_type: str
    final_label: str
    confidence: float
    reasons: list[str]
    message: str
    created_at: str
    watchlist_match: bool | None = None


class AlertListResponse(BaseModel):
    items: list[AlertRecord]
    total: int


class ReportCountItem(BaseModel):
    name: str
    count: int


class DriftSnapshot(BaseModel):
    score: float
    changed: bool
    current_distribution: dict[str, float]
    baseline_distribution: dict[str, float]
    baseline_period_start: str | None = None
    baseline_period_end: str | None = None


class PerClassMetric(BaseModel):
    label: str
    precision: float
    recall: float
    f1: float
    support: int
    precision_delta: float | None = None
    recall_delta: float | None = None
    f1_delta: float | None = None


class MonitoringSnapshot(BaseModel):
    inference_volume: int
    low_confidence_ratio: float
    user_correction_rate: float
    reviewed_feedback_count: int
    sampled_review_count: int
    class_drift: DriftSnapshot
    source_shift: DriftSnapshot
    per_class_metrics: list[PerClassMetric]


class FeedbackLoopAssets(BaseModel):
    error_sample_pool_size: int
    golden_test_set_size: int
    open_retrain_jobs: int
    periodic_retrain_due: bool
    scheduled_retrain_created: bool = False


class ReportSnapshotResponse(BaseModel):
    report_type: str
    period_start: str
    period_end: str
    generated_at: str
    total_runs: int
    alert_count: int
    review_queue_count: int
    sentiment_breakdown: dict[str, int]
    event_breakdown: dict[str, int]
    alert_breakdown: dict[str, int]
    top_entities: list[ReportCountItem]
    top_alert_entities: list[ReportCountItem]
    monitoring: MonitoringSnapshot
    feedback_loop_assets: FeedbackLoopAssets


class WatchlistCreateRequest(BaseModel):
    company_name: str = Field(..., min_length=1)
    ticker: str | None = None
    industry: str | None = None
    notes: str | None = None


class WatchlistRecord(BaseModel):
    id: int
    company_name: str
    ticker: str | None = None
    industry: str | None = None
    notes: str | None = None
    is_active: bool
    created_at: str
    updated_at: str
    recent_result_count: int = 0
    recent_alert_count: int = 0


class WatchlistListResponse(BaseModel):
    items: list[WatchlistRecord]
    total: int


class FeedbackCreateRequest(BaseModel):
    analysis_run_id: int
    feedback_label: SentimentLabel
    feedback_event_type: str | None = None
    reviewer: str | None = None
    notes: str | None = None


class FeedbackRecord(BaseModel):
    id: int
    analysis_run_id: int
    feedback_label: SentimentLabel
    feedback_event_type: str | None = None
    reviewer: str | None = None
    notes: str | None = None
    created_at: str


class FeedbackListResponse(BaseModel):
    items: list[FeedbackRecord]
    total: int


class RetrainRequest(BaseModel):
    requested_by: str | None = None
    notes: str | None = None
    trigger_source: str = "manual_feedback"
    include_feedback_only: bool = True


class RetrainJobRecord(BaseModel):
    id: int
    status: str
    trigger_source: str
    include_feedback_only: bool
    requested_by: str | None = None
    notes: str | None = None
    created_at: str
    updated_at: str


class GoldenTestCaseCreateRequest(BaseModel):
    input_text: str = Field(..., min_length=1)
    expected_label: SentimentLabel
    expected_event_type: str | None = None
    title: str | None = None
    source_name: str | None = None
    notes: str | None = None
    context: AnalysisContext | None = None


class GoldenTestCaseRecord(BaseModel):
    id: int
    title: str | None = None
    input_text: str
    expected_label: SentimentLabel
    expected_event_type: str | None = None
    source_name: str | None = None
    notes: str | None = None
    context: AnalysisContext
    is_active: bool
    created_at: str
    updated_at: str


class GoldenTestCaseListResponse(BaseModel):
    items: list[GoldenTestCaseRecord]
    total: int


class ErrorSampleRecord(BaseModel):
    id: int
    analysis_run_id: int
    primary_entity: str
    event_type: str
    final_label: str
    reasons: list[str]
    status: str
    latest_feedback_label: str | None = None
    created_at: str
    updated_at: str


class ErrorSampleListResponse(BaseModel):
    items: list[ErrorSampleRecord]
    total: int


class FeedbackLoopMaintenanceResponse(BaseModel):
    auto_sampled_review_count: int
    sampled_run_ids: list[int]
    periodic_retrain_due: bool
    retrain_job: RetrainJobRecord | None = None
    error_sample_pool_size: int
    golden_test_set_size: int


class AnalysisResultRecord(BaseModel):
    id: int
    created_at: str
    input_text: str
    context: AnalysisContext
    primary_entity: str
    event_type: str
    model_label: str
    model_confidence: float
    final_label: str
    final_confidence: float
    final_source: str
    needs_human_review: bool
    review_queue_record_id: int | None = None
    explanation: str
    feedback_count: int = 0
    feedback_labels: list[SentimentLabel] = Field(default_factory=list)
    feedback_event_types: list[str] = Field(default_factory=list)
    watchlist_match: bool = False
    in_error_pool: bool = False
    latest_feedback: FeedbackRecord | None = None
    result: AnalyzeResponse
    llm_review: dict[str, Any]
    alert: dict[str, Any] | None = None


class AnalysisResultListResponse(BaseModel):
    items: list[AnalysisResultRecord]
    total: int


class HealthResponse(BaseModel):
    status: str
    model_ready: bool
    training_error: str | None = None

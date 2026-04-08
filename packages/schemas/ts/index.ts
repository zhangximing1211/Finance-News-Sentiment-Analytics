export type SentimentLabel = "positive" | "neutral" | "negative";

export interface HistoricalAnnouncement {
  title?: string | null;
  summary: string;
  event_type?: string | null;
  announced_at?: string | null;
  source_name?: string | null;
}

export interface AnalysisContext {
  news_source?: string | null;
  source_name?: string | null;
  source_url?: string | null;
  published_at?: string | null;
  company_name?: string | null;
  ticker?: string | null;
  industry?: string | null;
  event_type?: string | null;
  historical_announcements: HistoricalAnnouncement[];
}

export interface AnalyzeRequest {
  text: string;
  context?: AnalysisContext | null;
}

export interface BatchAnalyzeRequest {
  texts?: string[] | null;
  items?: AnalyzeRequest[] | null;
}

export interface SentimentResult {
  label: SentimentLabel;
  label_zh: string;
  confidence: number;
  probabilities: Record<string, number>;
  decision_label: string;
  abstained: boolean;
  confidence_gap: number;
  low_confidence_threshold: number;
  source?: string | null;
  overridden_by_llm?: boolean | null;
  model_label?: string | null;
  model_label_zh?: string | null;
  model_confidence?: number | null;
}

export interface EventResult {
  type: string;
  type_zh: string;
  matched_signals: string[];
  secondary_type: string;
}

export interface Entities {
  companies: string[];
  tickers: string[];
  industry: string;
  industry_zh: string;
}

export interface RiskAlert {
  needs_human_review: boolean;
  message: string;
  reasons: string[];
}

export interface ReviewQueueItem {
  record_id?: number | null;
  status?: string | null;
  attempts?: number | null;
  llm_provider?: string | null;
  last_error?: string | null;
  queue: string;
  priority: string;
  primary_entity: string;
  event_type: string;
  predicted_label: string;
  decision_label: string;
  confidence: number;
  review_reasons: string[];
  recommended_action: string;
  text_excerpt: string;
}

export interface SecondaryExplanation {
  provider: string;
  template_path: string;
  summary: string;
  review_note: string;
  rationale: string;
  llm_ready: boolean;
  used_external_llm: boolean;
  prompt_available: boolean;
  input_excerpt: string;
  risk_message: string;
  fallback_reason?: string | null;
}

export interface AnalyzeResponse {
  input_text: string;
  context: AnalysisContext;
  sentiment: SentimentResult;
  event: EventResult;
  entities: Entities;
  explanation: string;
  risk_alert: RiskAlert;
  review_queue_item?: ReviewQueueItem | null;
  secondary_explanation: SecondaryExplanation;
  metadata: Record<string, unknown>;
}

export interface BatchAnalyzeResponse {
  items: AnalyzeResponse[];
  total: number;
}

export interface ReviewQueueRecord {
  id: number;
  status: string;
  priority: string;
  primary_entity: string;
  event_type: string;
  predicted_label: string;
  decision_label: string;
  confidence: number;
  review_reasons: string[];
  recommended_action: string;
  text_excerpt: string;
  input_text: string;
  secondary_explanation?: SecondaryExplanation | null;
  llm_provider?: string | null;
  last_error?: string | null;
  attempts: number;
  created_at: string;
  updated_at: string;
}

export interface ReviewQueueListResponse {
  items: ReviewQueueRecord[];
  total: number;
}

export interface ReviewQueueSummaryResponse {
  total_items: number;
  status_breakdown: Record<string, number>;
  priority_breakdown: Record<string, number>;
  review_reason_breakdown: Record<string, number>;
  pending_count: number;
  processing_count: number;
  ready_for_review_count: number;
  failed_count: number;
}

export interface AlertRecord {
  id: number;
  analysis_run_id: number;
  severity: string;
  status: string;
  primary_entity: string;
  event_type: string;
  final_label: string;
  confidence: number;
  reasons: string[];
  message: string;
  created_at: string;
  watchlist_match?: boolean | null;
}

export interface AlertListResponse {
  items: AlertRecord[];
  total: number;
}

export interface ReportCountItem {
  name: string;
  count: number;
}

export interface DriftSnapshot {
  score: number;
  changed: boolean;
  current_distribution: Record<string, number>;
  baseline_distribution: Record<string, number>;
  baseline_period_start?: string | null;
  baseline_period_end?: string | null;
}

export interface PerClassMetric {
  label: string;
  precision: number;
  recall: number;
  f1: number;
  support: number;
  precision_delta?: number | null;
  recall_delta?: number | null;
  f1_delta?: number | null;
}

export interface MonitoringSnapshot {
  inference_volume: number;
  low_confidence_ratio: number;
  user_correction_rate: number;
  reviewed_feedback_count: number;
  sampled_review_count: number;
  class_drift: DriftSnapshot;
  source_shift: DriftSnapshot;
  per_class_metrics: PerClassMetric[];
}

export interface FeedbackLoopAssets {
  error_sample_pool_size: number;
  golden_test_set_size: number;
  open_retrain_jobs: number;
  periodic_retrain_due: boolean;
  scheduled_retrain_created: boolean;
}

export interface ReportSnapshotResponse {
  report_type: string;
  period_start: string;
  period_end: string;
  generated_at: string;
  total_runs: number;
  alert_count: number;
  review_queue_count: number;
  sentiment_breakdown: Record<string, number>;
  event_breakdown: Record<string, number>;
  alert_breakdown: Record<string, number>;
  top_entities: ReportCountItem[];
  top_alert_entities: ReportCountItem[];
  monitoring: MonitoringSnapshot;
  feedback_loop_assets: FeedbackLoopAssets;
}

export interface WatchlistCreateRequest {
  company_name: string;
  ticker?: string | null;
  industry?: string | null;
  notes?: string | null;
}

export interface WatchlistRecord {
  id: number;
  company_name: string;
  ticker?: string | null;
  industry?: string | null;
  notes?: string | null;
  is_active: boolean;
  created_at: string;
  updated_at: string;
  recent_result_count: number;
  recent_alert_count: number;
}

export interface WatchlistListResponse {
  items: WatchlistRecord[];
  total: number;
}

export interface FeedbackCreateRequest {
  analysis_run_id: number;
  feedback_label: SentimentLabel;
  feedback_event_type?: string | null;
  reviewer?: string | null;
  notes?: string | null;
}

export interface FeedbackRecord {
  id: number;
  analysis_run_id: number;
  feedback_label: SentimentLabel;
  feedback_event_type?: string | null;
  reviewer?: string | null;
  notes?: string | null;
  created_at: string;
}

export interface FeedbackListResponse {
  items: FeedbackRecord[];
  total: number;
}

export interface RetrainRequest {
  requested_by?: string | null;
  notes?: string | null;
  trigger_source: string;
  include_feedback_only: boolean;
}

export interface RetrainJobRecord {
  id: number;
  status: string;
  trigger_source: string;
  include_feedback_only: boolean;
  requested_by?: string | null;
  notes?: string | null;
  created_at: string;
  updated_at: string;
}

export interface GoldenTestCaseCreateRequest {
  input_text: string;
  expected_label: SentimentLabel;
  expected_event_type?: string | null;
  title?: string | null;
  source_name?: string | null;
  notes?: string | null;
  context?: AnalysisContext | null;
}

export interface GoldenTestCaseRecord {
  id: number;
  title?: string | null;
  input_text: string;
  expected_label: SentimentLabel;
  expected_event_type?: string | null;
  source_name?: string | null;
  notes?: string | null;
  context: AnalysisContext;
  is_active: boolean;
  created_at: string;
  updated_at: string;
}

export interface GoldenTestCaseListResponse {
  items: GoldenTestCaseRecord[];
  total: number;
}

export interface ErrorSampleRecord {
  id: number;
  analysis_run_id: number;
  primary_entity: string;
  event_type: string;
  final_label: string;
  reasons: string[];
  status: string;
  latest_feedback_label?: string | null;
  created_at: string;
  updated_at: string;
}

export interface ErrorSampleListResponse {
  items: ErrorSampleRecord[];
  total: number;
}

export interface FeedbackLoopMaintenanceResponse {
  auto_sampled_review_count: number;
  sampled_run_ids: number[];
  periodic_retrain_due: boolean;
  retrain_job?: RetrainJobRecord | null;
  error_sample_pool_size: number;
  golden_test_set_size: number;
}

export interface AnalysisResultRecord {
  id: number;
  created_at: string;
  input_text: string;
  context: AnalysisContext;
  primary_entity: string;
  event_type: string;
  model_label: string;
  model_confidence: number;
  final_label: string;
  final_confidence: number;
  final_source: string;
  needs_human_review: boolean;
  review_queue_record_id?: number | null;
  explanation: string;
  feedback_count: number;
  feedback_labels: SentimentLabel[];
  feedback_event_types: string[];
  watchlist_match: boolean;
  in_error_pool: boolean;
  latest_feedback?: FeedbackRecord | null;
  result: AnalyzeResponse;
  llm_review: Record<string, unknown>;
  alert?: Record<string, unknown> | null;
}

export interface AnalysisResultListResponse {
  items: AnalysisResultRecord[];
  total: number;
}

export interface HealthResponse {
  status: string;
  model_ready: boolean;
  training_error: string | null;
}

import type {
  AnalysisContext,
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
  ReportSnapshotResponse,
  RetrainJobRecord,
  RetrainRequest,
  WatchlistCreateRequest,
  WatchlistListResponse,
  WatchlistRecord,
} from "../../../packages/schemas/ts";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
    cache: "no-store",
  });

  if (!response.ok) {
    let detail = response.statusText;
    try {
      const payload = (await response.json()) as { detail?: string };
      detail = payload.detail ?? detail;
    } catch {
      detail = (await response.text()) || detail;
    }
    throw new Error(detail || "Request failed.");
  }

  return (await response.json()) as T;
}

function buildQuery(params: Record<string, string | number | boolean | undefined | null>): string {
  const searchParams = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value === undefined || value === null || value === "") {
      return;
    }
    searchParams.set(key, String(value));
  });
  const queryString = searchParams.toString();
  return queryString ? `?${queryString}` : "";
}

export async function analyzeText(text: string, context?: AnalysisContext): Promise<AnalyzeResponse> {
  const payload: AnalyzeRequest = { text, context };
  return request<AnalyzeResponse>("/analyze", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function batchAnalyzeTexts(texts: string[]): Promise<BatchAnalyzeResponse> {
  const payload: BatchAnalyzeRequest = { texts };
  return request<BatchAnalyzeResponse>("/batch-analyze", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function batchAnalyzeItems(items: AnalyzeRequest[]): Promise<BatchAnalyzeResponse> {
  const payload: BatchAnalyzeRequest = { items };
  return request<BatchAnalyzeResponse>("/batch-analyze", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function fetchResults(params: {
  limit?: number;
  label?: string;
  event_type?: string;
  entity_query?: string;
  source?: string;
  error_only?: boolean;
  watchlist_only?: boolean;
} = {}): Promise<AnalysisResultListResponse> {
  return request<AnalysisResultListResponse>(`/results${buildQuery(params)}`);
}

export async function fetchAlerts(params: {
  status?: string;
  severity?: string;
  limit?: number;
  watchlist_only?: boolean;
} = {}): Promise<AlertListResponse> {
  return request<AlertListResponse>(`/alerts${buildQuery(params)}`);
}

export async function fetchWatchlist(limit = 100): Promise<WatchlistListResponse> {
  return request<WatchlistListResponse>(`/watchlist${buildQuery({ limit })}`);
}

export async function createWatchlistItem(payload: WatchlistCreateRequest): Promise<WatchlistRecord> {
  return request<WatchlistRecord>("/watchlist", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function fetchFeedback(params: { limit?: number; analysis_run_id?: number } = {}): Promise<FeedbackListResponse> {
  return request<FeedbackListResponse>(`/feedback${buildQuery(params)}`);
}

export async function createFeedback(payload: FeedbackCreateRequest): Promise<FeedbackRecord> {
  return request<FeedbackRecord>("/feedback", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function createRetrainJob(payload: RetrainRequest): Promise<RetrainJobRecord> {
  return request<RetrainJobRecord>("/retrain", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function fetchErrorSamples(params: { limit?: number; status?: string } = {}): Promise<ErrorSampleListResponse> {
  return request<ErrorSampleListResponse>(`/error-samples${buildQuery(params)}`);
}

export async function fetchGoldenSet(params: { limit?: number; active_only?: boolean } = {}): Promise<GoldenTestCaseListResponse> {
  return request<GoldenTestCaseListResponse>(`/golden-set${buildQuery(params)}`);
}

export async function createGoldenTestCase(payload: GoldenTestCaseCreateRequest): Promise<GoldenTestCaseRecord> {
  return request<GoldenTestCaseRecord>("/golden-set", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function runFeedbackLoopMaintenance(params: {
  report_type?: string;
  date_value?: string;
  sample_limit?: number;
} = {}): Promise<FeedbackLoopMaintenanceResponse> {
  return request<FeedbackLoopMaintenanceResponse>(`/feedback-loop/maintenance${buildQuery(params)}`, {
    method: "POST",
  });
}

export async function fetchDailyReport(dateValue?: string): Promise<ReportSnapshotResponse> {
  return request<ReportSnapshotResponse>(`/reports/daily${buildQuery({ date_value: dateValue })}`);
}

export async function fetchWeeklyReport(dateValue?: string): Promise<ReportSnapshotResponse> {
  return request<ReportSnapshotResponse>(`/reports/weekly${buildQuery({ date_value: dateValue })}`);
}

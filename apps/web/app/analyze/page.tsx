"use client";

import { useState } from "react";
import { formatConfidence } from "../../../../packages/utils/ts";
import { MetricCard } from "../../components/metric-card";
import { PageHeader } from "../../components/page-header";
import { StatusPill } from "../../components/status-pill";
import { analyzeText } from "../../lib/api";
import { sentimentTone } from "../../lib/view";
import type { AnalysisContext, AnalyzeResponse } from "../../../../packages/schemas/ts";

const samples = {
  positive:
    "Apple Inc. (NASDAQ: AAPL) raised its full-year revenue guidance after signing a multiyear supply agreement with a major cloud customer.",
  mixed:
    "The company beat revenue expectations but warned it may cut jobs and lower its outlook.",
  neutral:
    "Tesla (NASDAQ: TSLA) said it will host an investor event next month to discuss factory plans and updated production targets.",
};

type ContextDraft = {
  news_source: string;
  source_name: string;
  source_url: string;
  published_at: string;
  company_name: string;
  ticker: string;
  industry: string;
  event_type: string;
  historical_announcements_text: string;
};

const emptyContext: ContextDraft = {
  news_source: "",
  source_name: "",
  source_url: "",
  published_at: "",
  company_name: "",
  ticker: "",
  industry: "",
  event_type: "",
  historical_announcements_text: "",
};

function parseHistoricalAnnouncements(input: string): AnalysisContext["historical_announcements"] {
  return input
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => {
      const [announced_at, event_type, summary] = line.split("|").map((part) => part.trim());
      return {
        announced_at: announced_at || undefined,
        event_type: event_type || undefined,
        summary: summary || line,
      };
    });
}

function buildContextPayload(draft: ContextDraft): AnalysisContext | undefined {
  const historical_announcements = parseHistoricalAnnouncements(draft.historical_announcements_text);
  const payload: AnalysisContext = {
    news_source: draft.news_source.trim() || undefined,
    source_name: draft.source_name.trim() || undefined,
    source_url: draft.source_url.trim() || undefined,
    published_at: draft.published_at.trim() || undefined,
    company_name: draft.company_name.trim() || undefined,
    ticker: draft.ticker.trim() || undefined,
    industry: draft.industry.trim() || undefined,
    event_type: draft.event_type.trim() || undefined,
    historical_announcements,
  };

  const hasValue = Object.entries(payload).some(([key, value]) => key === "historical_announcements" ? (value as unknown[]).length > 0 : Boolean(value));
  return hasValue ? payload : undefined;
}

export default function AnalyzePage() {
  const [input, setInput] = useState(samples.positive);
  const [contextDraft, setContextDraft] = useState<ContextDraft>(emptyContext);
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [status, setStatus] = useState("等待输入");
  const [loading, setLoading] = useState(false);

  async function handleAnalyze() {
    if (!input.trim()) {
      setStatus("请输入一段新闻或公告。");
      return;
    }

    setLoading(true);
    setStatus("分析中...");
    try {
      const response = await analyzeText(input.trim(), buildContextPayload(contextDraft));
      setResult(response);
      setStatus("分析完成");
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "分析失败");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="page-root">
      <PageHeader
        eyebrow="Single Analyze"
        title="单条文本分析"
        description="给一条新闻、公告或 filing，返回情绪、事件、实体、解释、人工复核和 LLM 二次判断。"
        actions={<StatusPill tone="info">{status}</StatusPill>}
      />

      <section className="two-column-grid">
        <div className="panel">
          <div className="panel-titlebar">
            <div>
              <h2>输入新闻</h2>
              <p className="panel-copy">除了正文，这里还可以补新闻源、发布时间、公司与 ticker 映射、历史公告背景。</p>
            </div>
            <div className="button-row">
              {Object.entries(samples).map(([key, value]) => (
                <button className="ghost-button" key={key} onClick={() => setInput(value)} type="button">
                  {key}
                </button>
              ))}
            </div>
          </div>
          <div className="form-grid">
            <textarea
              className="text-area"
              value={input}
              onChange={(event) => setInput(event.target.value)}
              placeholder="Paste one finance news snippet."
            />

            <div className="compact-form">
              <div>
                <label className="field-label" htmlFor="news_source">
                  新闻源
                </label>
                <input
                  className="text-input"
                  id="news_source"
                  value={contextDraft.news_source}
                  onChange={(event) => setContextDraft((current) => ({ ...current, news_source: event.target.value }))}
                  placeholder="Reuters"
                />
              </div>
              <div>
                <label className="field-label" htmlFor="source_name">
                  文章来源
                </label>
                <input
                  className="text-input"
                  id="source_name"
                  value={contextDraft.source_name}
                  onChange={(event) => setContextDraft((current) => ({ ...current, source_name: event.target.value }))}
                  placeholder="Reuters Breakingviews"
                />
              </div>
              <div>
                <label className="field-label" htmlFor="source_url">
                  来源 URL
                </label>
                <input
                  className="text-input"
                  id="source_url"
                  value={contextDraft.source_url}
                  onChange={(event) => setContextDraft((current) => ({ ...current, source_url: event.target.value }))}
                  placeholder="https://..."
                />
              </div>
              <div>
                <label className="field-label" htmlFor="published_at">
                  日期时间
                </label>
                <input
                  className="text-input"
                  id="published_at"
                  value={contextDraft.published_at}
                  onChange={(event) => setContextDraft((current) => ({ ...current, published_at: event.target.value }))}
                  placeholder="2026-04-08T09:30:00+08:00"
                />
              </div>
              <div>
                <label className="field-label" htmlFor="company_name">
                  公司名
                </label>
                <input
                  className="text-input"
                  id="company_name"
                  value={contextDraft.company_name}
                  onChange={(event) => setContextDraft((current) => ({ ...current, company_name: event.target.value }))}
                  placeholder="Apple Inc."
                />
              </div>
              <div>
                <label className="field-label" htmlFor="ticker">
                  Ticker
                </label>
                <input
                  className="text-input"
                  id="ticker"
                  value={contextDraft.ticker}
                  onChange={(event) => setContextDraft((current) => ({ ...current, ticker: event.target.value }))}
                  placeholder="AAPL"
                />
              </div>
              <div>
                <label className="field-label" htmlFor="industry">
                  行业
                </label>
                <input
                  className="text-input"
                  id="industry"
                  value={contextDraft.industry}
                  onChange={(event) => setContextDraft((current) => ({ ...current, industry: event.target.value }))}
                  placeholder="technology / 科技"
                />
              </div>
              <div>
                <label className="field-label" htmlFor="event_type">
                  事件类型
                </label>
                <input
                  className="text-input"
                  id="event_type"
                  value={contextDraft.event_type}
                  onChange={(event) => setContextDraft((current) => ({ ...current, event_type: event.target.value }))}
                  placeholder="earnings / 财报"
                />
              </div>
            </div>

            <div>
              <label className="field-label" htmlFor="historical_announcements_text">
                历史公告
              </label>
              <textarea
                className="text-area"
                id="historical_announcements_text"
                value={contextDraft.historical_announcements_text}
                onChange={(event) =>
                  setContextDraft((current) => ({ ...current, historical_announcements_text: event.target.value }))
                }
                placeholder={"每行一条，格式：2025-12-01 | guidance | 公司上调全年指引"}
              />
            </div>

            <div className="action-row">
              <button className="primary-button" disabled={loading} onClick={handleAnalyze} type="button">
                {loading ? "分析中..." : "开始分析"}
              </button>
              <span className="status-text">{status}</span>
            </div>
          </div>
        </div>

        <div className="panel">
          <div className="panel-titlebar">
            <div>
              <h2>分析输出</h2>
              <p className="panel-copy">结果里会显式回传新闻源、发布时间、company/ticker 映射和历史公告数量。</p>
            </div>
          </div>

          {!result ? (
            <div className="empty-state">结果会显示在这里。先试一条包含公司名和事件的文本。</div>
          ) : (
            <div className="panel-stack">
              <div className="result-grid">
                <MetricCard
                  emphasis
                  label="最终情绪"
                  value={`${result.sentiment.label_zh} / ${result.sentiment.label}`}
                  subtitle={`source ${result.sentiment.source ?? "sentiment_model"} · ${formatConfidence(result.sentiment.confidence)}`}
                />
                <MetricCard
                  label="模型原判"
                  value={`${result.sentiment.model_label_zh ?? result.sentiment.label_zh} / ${result.sentiment.model_label ?? result.sentiment.label}`}
                  subtitle={`model confidence ${formatConfidence(result.sentiment.model_confidence ?? result.sentiment.confidence)}`}
                />
                <MetricCard
                  label="事件类型"
                  value={`${result.event.type_zh} / ${result.event.type}`}
                  subtitle={result.event.matched_signals.join(" / ") || "No stable signal"}
                />
                <MetricCard
                  label="主体"
                  value={result.entities.companies.join(", ") || "未识别"}
                  subtitle={result.entities.tickers.join(", ") || result.entities.industry_zh}
                />
                <MetricCard
                  label="来源 / 时间"
                  value={result.context.source_name || result.context.news_source || "未填写"}
                  subtitle={result.context.published_at || result.context.source_url || "No publish time"}
                />
                <MetricCard
                  label="历史公告"
                  value={String(result.context.historical_announcements.length)}
                  subtitle={result.context.event_type || "No provided event override"}
                />
                <MetricCard
                  label="人工复核"
                  value={result.risk_alert.needs_human_review ? "需要人工复核" : "可自动通过"}
                  subtitle={result.review_queue_item ? result.review_queue_item.review_reasons.join(" / ") : "No queue item"}
                />
                <MetricCard label="一句解释" value={result.explanation} subtitle={result.secondary_explanation.summary} />
              </div>

              <div className="soft-card">
                <div className="split-line">
                  <StatusPill tone={sentimentTone(result.sentiment.label)}>{result.sentiment.label}</StatusPill>
                  {result.sentiment.overridden_by_llm ? <StatusPill tone="warning">LLM 覆盖</StatusPill> : null}
                  {result.secondary_explanation.used_external_llm ? <StatusPill tone="positive">OpenAI 已调用</StatusPill> : null}
                </div>
                <p className="panel-copy">{result.secondary_explanation.rationale}</p>
              </div>

              <div className="note-block">
                <strong>上下文回显</strong>
                <p className="panel-copy">
                  {result.context.company_name || "未提供公司名"} / {result.context.ticker || "未提供 ticker"} / {result.context.industry || "未提供行业"}
                </p>
              </div>

              <div className="note-block">
                <strong>风险提示</strong>
                <p className="panel-copy">{result.risk_alert.message}</p>
              </div>
            </div>
          )}
        </div>
      </section>
    </main>
  );
}

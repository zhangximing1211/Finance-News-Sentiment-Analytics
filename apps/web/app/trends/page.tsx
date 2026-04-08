"use client";

import { useEffect, useMemo, useState } from "react";
import { MetricCard } from "../../components/metric-card";
import { PageHeader } from "../../components/page-header";
import { StatusPill } from "../../components/status-pill";
import { fetchDailyReport, fetchResults, fetchWeeklyReport, runFeedbackLoopMaintenance } from "../../lib/api";
import { buildTrendSeries, formatDate } from "../../lib/view";
import type { AnalysisResultRecord, ReportSnapshotResponse } from "../../../../packages/schemas/ts";

export default function TrendsPage() {
  const [results, setResults] = useState<AnalysisResultRecord[]>([]);
  const [daily, setDaily] = useState<ReportSnapshotResponse | null>(null);
  const [weekly, setWeekly] = useState<ReportSnapshotResponse | null>(null);
  const [status, setStatus] = useState("加载中...");

  useEffect(() => {
    async function load() {
      try {
        const [resultsResponse, dailyReport, weeklyReport] = await Promise.all([
          fetchResults({ limit: 200 }),
          fetchDailyReport(),
          fetchWeeklyReport(),
        ]);
        setResults(resultsResponse.items);
        setDaily(dailyReport);
        setWeekly(weeklyReport);
        setStatus("趋势数据已更新");
      } catch (error) {
        setStatus(error instanceof Error ? error.message : "加载失败");
      }
    }

    void load();
  }, []);

  const trendSeries = useMemo(() => buildTrendSeries(results), [results]);
  const maxBucket = useMemo(
    () => Math.max(1, ...trendSeries.map((item) => item.positive + item.neutral + item.negative)),
    [trendSeries],
  );

  async function handleMaintenance() {
    setStatus("执行反馈闭环维护...");
    try {
      const summary = await runFeedbackLoopMaintenance({ report_type: "weekly", sample_limit: 12 });
      const [dailyReport, weeklyReport] = await Promise.all([fetchDailyReport(), fetchWeeklyReport()]);
      setDaily(dailyReport);
      setWeekly(weeklyReport);
      setStatus(`已采样 ${summary.auto_sampled_review_count} 条，${summary.retrain_job ? `并创建重训 #${summary.retrain_job.id}` : "未创建重训"}`);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "维护失败");
    }
  }

  return (
    <main className="page-root">
      <PageHeader
        eyebrow="Sentiment Trends"
        title="情绪趋势页"
        description="从结果库聚合近端走势，用来观察情绪分布、事件偏移和 watchlist 变化。"
        actions={
          <div className="button-row">
            <StatusPill tone="info">{status}</StatusPill>
            <button className="secondary-button" onClick={handleMaintenance} type="button">
              运行闭环维护
            </button>
          </div>
        }
      />

      <section className="dashboard-grid">
        <MetricCard emphasis label="今日样本" value={String(daily?.total_runs ?? 0)} subtitle="daily analyzed items" />
        <MetricCard label="低置信度比例" value={`${(((daily?.monitoring.low_confidence_ratio ?? 0) * 100).toFixed(1))}%`} subtitle="daily low confidence" />
        <MetricCard label="类别漂移" value={String(weekly?.monitoring.class_drift.score ?? 0)} subtitle="weekly class drift" />
        <MetricCard label="新闻源偏移" value={String(weekly?.monitoring.source_shift.score ?? 0)} subtitle="weekly source shift" />
      </section>

      <section className="dashboard-grid">
        <MetricCard label="本周样本" value={String(weekly?.total_runs ?? 0)} subtitle="weekly analyzed items" />
        <MetricCard label="本周告警" value={String(weekly?.alert_count ?? 0)} subtitle="weekly alerts" />
        <MetricCard
          label="用户纠错率"
          value={`${(((weekly?.monitoring.user_correction_rate ?? 0) * 100).toFixed(1))}%`}
          subtitle={`reviewed ${weekly?.monitoring.reviewed_feedback_count ?? 0}`}
        />
        <MetricCard
          label="自动采样复核"
          value={String(weekly?.monitoring.sampled_review_count ?? 0)}
          subtitle="weekly sampled review"
        />
        <MetricCard
          label="错误池 / 黄金集"
          value={`${weekly?.feedback_loop_assets.error_sample_pool_size ?? 0} / ${weekly?.feedback_loop_assets.golden_test_set_size ?? 0}`}
          subtitle={weekly?.feedback_loop_assets.periodic_retrain_due ? "retrain due" : "retrain not due"}
        />
      </section>

      <section className="two-column-grid">
        <div className="panel">
          <div className="panel-titlebar">
            <div>
              <h2>近十个自然日趋势</h2>
              <p className="panel-copy">用最近结果做轻量聚合，不依赖额外图表库。</p>
            </div>
          </div>
          {trendSeries.length === 0 ? (
            <div className="empty-state">还没有足够结果可供绘图。</div>
          ) : (
            <div className="trend-chart">
              {trendSeries.map((item) => {
                const total = item.positive + item.neutral + item.negative;
                return (
                  <div className="trend-row" key={item.day}>
                    <strong>{formatDate(item.day)}</strong>
                    <div className="trend-bars">
                      <div className="trend-bar">
                        <div className="trend-fill trend-fill-positive" style={{ width: `${(item.positive / maxBucket) * 100}%` }} />
                      </div>
                      <div className="trend-bar">
                        <div className="trend-fill trend-fill-neutral" style={{ width: `${(item.neutral / maxBucket) * 100}%` }} />
                      </div>
                      <div className="trend-bar">
                        <div className="trend-fill trend-fill-negative" style={{ width: `${(item.negative / maxBucket) * 100}%` }} />
                      </div>
                    </div>
                    <span className="muted monospace">{total}</span>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        <div className="panel">
          <div className="panel-titlebar">
            <div>
              <h2>聚合摘要</h2>
              <p className="panel-copy">日报和周报仍来自后端 report snapshot。</p>
            </div>
          </div>
          <div className="stack-list">
            <div className="soft-card">
              <strong>今日情绪分布</strong>
              <p className="panel-copy monospace">{JSON.stringify(daily?.sentiment_breakdown ?? {}, null, 2)}</p>
            </div>
            <div className="soft-card">
              <strong>本周事件分布</strong>
              <p className="panel-copy monospace">{JSON.stringify(weekly?.event_breakdown ?? {}, null, 2)}</p>
            </div>
            <div className="soft-card">
              <strong>本周高频主体</strong>
              <p className="panel-copy">
                {(weekly?.top_entities ?? []).map((item) => `${item.name} (${item.count})`).join(" / ") || "暂无"}
              </p>
            </div>
            <div className="soft-card">
              <strong>按类 Precision / Recall / F1 变化</strong>
              <p className="panel-copy monospace">
                {(weekly?.monitoring.per_class_metrics ?? [])
                  .map((item) => `${item.label}: P ${item.precision.toFixed(2)} (${item.precision_delta?.toFixed(2)}), R ${item.recall.toFixed(2)} (${item.recall_delta?.toFixed(2)}), F1 ${item.f1.toFixed(2)} (${item.f1_delta?.toFixed(2)})`)
                  .join("\n") || "暂无反馈样本"}
              </p>
            </div>
          </div>
        </div>
      </section>
    </main>
  );
}

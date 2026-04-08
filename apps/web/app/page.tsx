"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { MetricCard } from "../components/metric-card";
import { PageHeader } from "../components/page-header";
import { StatusPill } from "../components/status-pill";
import { fetchAlerts, fetchDailyReport, fetchResults } from "../lib/api";
import { formatDateTime, sentimentTone } from "../lib/view";
import type { AlertListResponse, AnalysisResultListResponse, ReportSnapshotResponse } from "../../../packages/schemas/ts";

const shortcuts = [
  { href: "/analyze", title: "单条文本分析", copy: "适合研究员、客服和运营拿一条新闻立即下判断。" },
  { href: "/batch", title: "批量分析", copy: "把公告流、爬虫结果或日报候选一次跑完并落库。" },
  { href: "/watchlist", title: "公司 Watchlist", copy: "维护重点公司池，盯住相关结果和告警。" },
  { href: "/trends", title: "情绪趋势", copy: "看最近样本分布，判断市场语气和事件重心。" },
  { href: "/errors", title: "错误案例回看", copy: "优先看模型与 LLM 分歧、人工复核和异常输出。" },
  { href: "/feedback", title: "人工反馈标注", copy: "给错例打标签，再发起重训请求。" },
];

export default function HomePage() {
  const [report, setReport] = useState<ReportSnapshotResponse | null>(null);
  const [alerts, setAlerts] = useState<AlertListResponse["items"]>([]);
  const [results, setResults] = useState<AnalysisResultListResponse["items"]>([]);
  const [status, setStatus] = useState("加载中...");

  useEffect(() => {
    async function load() {
      try {
        const [daily, alertsResponse, resultsResponse] = await Promise.all([
          fetchDailyReport(),
          fetchAlerts({ limit: 5 }),
          fetchResults({ limit: 5, error_only: true }),
        ]);
        setReport(daily);
        setAlerts(alertsResponse.items);
        setResults(resultsResponse.items);
        setStatus("已同步最新数据");
      } catch (error) {
        setStatus(error instanceof Error ? error.message : "加载失败");
      }
    }

    void load();
  }, []);

  return (
    <main className="page-root">
      <section className="hero-panel">
        <p className="eyebrow">Product Console</p>
        <h1 className="hero-title">把分析、复判、反馈和重训接成一个产品面。</h1>
        <p className="hero-copy">
          当前版本同时覆盖单条分析、批量分析、watchlist、趋势、错误案例和人工反馈，
          后端直接落在 FastAPI + SQLite + agent workflow 上。
        </p>
      </section>

      <PageHeader
        eyebrow="Overview"
        title="今日总览"
        description="先看系统状态，再进入具体页面处理样本、告警和反馈。"
        actions={<StatusPill tone="info">{status}</StatusPill>}
      />

      <section className="dashboard-grid">
        <MetricCard emphasis label="今日分析量" value={String(report?.total_runs ?? 0)} subtitle="Daily report snapshot" />
        <MetricCard
          label="低置信度比例"
          value={`${(((report?.monitoring.low_confidence_ratio ?? 0) * 100).toFixed(1))}%`}
          subtitle="Low-confidence ratio"
        />
        <MetricCard
          label="用户纠错率"
          value={`${(((report?.monitoring.user_correction_rate ?? 0) * 100).toFixed(1))}%`}
          subtitle="Feedback correction rate"
        />
        <MetricCard
          label="错误样本池 / 黄金集"
          value={`${report?.feedback_loop_assets.error_sample_pool_size ?? 0} / ${report?.feedback_loop_assets.golden_test_set_size ?? 0}`}
          subtitle={`${report?.feedback_loop_assets.open_retrain_jobs ?? 0} open retrain jobs`}
        />
      </section>

      <section className="three-column-grid">
        <div className="panel">
          <div className="panel-titlebar">
            <div>
              <h2>快捷入口</h2>
              <p className="panel-copy">每个页面都接真实 API，不是静态 mock。</p>
            </div>
          </div>
          <div className="shortcut-grid">
            {shortcuts.map((item) => (
              <Link className="shortcut-card" href={item.href} key={item.href}>
                <strong>{item.title}</strong>
                <p>{item.copy}</p>
                <StatusPill tone="info">进入页面</StatusPill>
              </Link>
            ))}
          </div>
        </div>

        <div className="panel">
          <div className="panel-titlebar">
            <div>
              <h2>最新告警</h2>
              <p className="panel-copy">最近触发的高优先级事项。</p>
            </div>
            <Link className="ghost-button" href="/watchlist">
              看 watchlist
            </Link>
          </div>
          <div className="stack-list">
            {alerts.length === 0 ? (
              <div className="empty-state">暂无告警，当前系统没有触发需要升级的样本。</div>
            ) : (
              alerts.map((item) => (
                <article className="list-item" key={item.id}>
                  <div className="list-item-title">
                    <strong>{item.primary_entity || "未识别主体"}</strong>
                    <StatusPill tone={item.severity === "high" ? "negative" : item.severity === "medium" ? "warning" : "info"}>
                      {item.severity}
                    </StatusPill>
                  </div>
                  <div className="meta-line">
                    <span>{item.event_type}</span>
                    <span>{item.final_label}</span>
                    <span>{formatDateTime(item.created_at)}</span>
                  </div>
                  <p className="panel-copy">{item.message}</p>
                </article>
              ))
            )}
          </div>
        </div>

        <div className="panel">
          <div className="panel-titlebar">
            <div>
              <h2>错误案例池</h2>
              <p className="panel-copy">优先处理人工复核和模型分歧样本。</p>
            </div>
            <Link className="ghost-button" href="/errors">
              全量查看
            </Link>
          </div>
          <div className="stack-list">
            {results.length === 0 ? (
              <div className="empty-state">还没有错误案例进入列表。</div>
            ) : (
              results.map((item) => (
                <article className="list-item" key={item.id}>
                  <div className="list-item-title">
                    <strong>{item.primary_entity || "未识别主体"}</strong>
                    <StatusPill tone={sentimentTone(item.final_label)}>{item.final_label}</StatusPill>
                  </div>
                  <div className="meta-line">
                    <span>model: {item.model_label}</span>
                    <span>final: {item.final_label}</span>
                    <span>{item.final_source}</span>
                  </div>
                  <p className="panel-copy">{item.explanation}</p>
                </article>
              ))
            )}
          </div>
        </div>
      </section>
    </main>
  );
}

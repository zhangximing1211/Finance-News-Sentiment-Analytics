"use client";

import { useEffect, useState } from "react";
import { PageHeader } from "../../components/page-header";
import { StatusPill } from "../../components/status-pill";
import { createWatchlistItem, fetchAlerts, fetchResults, fetchWatchlist } from "../../lib/api";
import { formatDateTime, sentimentTone } from "../../lib/view";
import type { AlertRecord, AnalysisResultRecord, WatchlistCreateRequest, WatchlistRecord } from "../../../../packages/schemas/ts";

const initialForm: WatchlistCreateRequest = {
  company_name: "",
  ticker: "",
  industry: "",
  notes: "",
};

export default function WatchlistPage() {
  const [form, setForm] = useState<WatchlistCreateRequest>(initialForm);
  const [items, setItems] = useState<WatchlistRecord[]>([]);
  const [alerts, setAlerts] = useState<AlertRecord[]>([]);
  const [results, setResults] = useState<AnalysisResultRecord[]>([]);
  const [status, setStatus] = useState("加载中...");

  async function load() {
    try {
      const [watchlistResponse, alertResponse, resultResponse] = await Promise.all([
        fetchWatchlist(100),
        fetchAlerts({ limit: 20, watchlist_only: true }),
        fetchResults({ limit: 20, watchlist_only: true }),
      ]);
      setItems(watchlistResponse.items);
      setAlerts(alertResponse.items);
      setResults(resultResponse.items);
      setStatus("Watchlist 已同步");
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "加载失败");
    }
  }

  useEffect(() => {
    void load();
  }, []);

  async function handleSubmit() {
    if (!form.company_name?.trim()) {
      setStatus("company_name 不能为空。");
      return;
    }

    setStatus("写入 watchlist...");
    try {
      await createWatchlistItem({
        company_name: form.company_name.trim(),
        ticker: form.ticker?.trim() || undefined,
        industry: form.industry?.trim() || undefined,
        notes: form.notes?.trim() || undefined,
      });
      setForm(initialForm);
      await load();
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "写入失败");
    }
  }

  return (
    <main className="page-root">
      <PageHeader
        eyebrow="Watchlist"
        title="公司 Watchlist 页"
        description="维护重点公司池，并联动查看相关分析结果和告警。"
        actions={<StatusPill tone="info">{status}</StatusPill>}
      />

      <section className="two-column-grid">
        <div className="panel">
          <div className="panel-titlebar">
            <div>
              <h2>新增公司</h2>
              <p className="panel-copy">公司名是主键，ticker 和行业用于补充识别。</p>
            </div>
          </div>
          <div className="compact-form">
            <div>
              <label className="field-label" htmlFor="company_name">
                公司名
              </label>
              <input
                className="text-input"
                id="company_name"
                value={form.company_name ?? ""}
                onChange={(event) => setForm((current) => ({ ...current, company_name: event.target.value }))}
              />
            </div>
            <div>
              <label className="field-label" htmlFor="ticker">
                Ticker
              </label>
              <input
                className="text-input"
                id="ticker"
                value={form.ticker ?? ""}
                onChange={(event) => setForm((current) => ({ ...current, ticker: event.target.value }))}
              />
            </div>
            <div>
              <label className="field-label" htmlFor="industry">
                行业
              </label>
              <input
                className="text-input"
                id="industry"
                value={form.industry ?? ""}
                onChange={(event) => setForm((current) => ({ ...current, industry: event.target.value }))}
              />
            </div>
            <div>
              <label className="field-label" htmlFor="notes">
                备注
              </label>
              <input
                className="text-input"
                id="notes"
                value={form.notes ?? ""}
                onChange={(event) => setForm((current) => ({ ...current, notes: event.target.value }))}
              />
            </div>
          </div>
          <div className="action-row">
            <button className="primary-button" onClick={handleSubmit} type="button">
              添加到 watchlist
            </button>
          </div>
        </div>

        <div className="panel">
          <div className="panel-titlebar">
            <div>
              <h2>当前 watchlist</h2>
              <p className="panel-copy">结果数和告警数由后端实时汇总。</p>
            </div>
          </div>
          <div className="stack-list">
            {items.length === 0 ? (
              <div className="empty-state">还没有重点公司。</div>
            ) : (
              items.map((item) => (
                <article className="list-item" key={item.id}>
                  <div className="list-item-title">
                    <strong>{item.company_name}</strong>
                    <StatusPill tone="info">{item.ticker || "No ticker"}</StatusPill>
                  </div>
                  <div className="meta-line">
                    <span>{item.industry || "未填写行业"}</span>
                    <span>{item.recent_result_count} 条结果</span>
                    <span>{item.recent_alert_count} 条告警</span>
                  </div>
                  {item.notes ? <p className="panel-copy">{item.notes}</p> : null}
                </article>
              ))
            )}
          </div>
        </div>
      </section>

      <section className="three-column-grid">
        <div className="panel">
          <div className="panel-titlebar">
            <div>
              <h2>Watchlist 告警</h2>
              <p className="panel-copy">已过滤出 watchlist 命中的告警。</p>
            </div>
          </div>
          <div className="stack-list">
            {alerts.length === 0 ? (
              <div className="empty-state">暂无 watchlist 告警。</div>
            ) : (
              alerts.map((alert) => (
                <article className="list-item" key={alert.id}>
                  <div className="list-item-title">
                    <strong>{alert.primary_entity}</strong>
                    <StatusPill tone={alert.severity === "high" ? "negative" : alert.severity === "medium" ? "warning" : "info"}>
                      {alert.severity}
                    </StatusPill>
                  </div>
                  <div className="meta-line">
                    <span>{alert.event_type}</span>
                    <span>{formatDateTime(alert.created_at)}</span>
                  </div>
                  <p className="panel-copy">{alert.message}</p>
                </article>
              ))
            )}
          </div>
        </div>

        <div className="panel">
          <div className="panel-titlebar">
            <div>
              <h2>最近结果</h2>
              <p className="panel-copy">命中 watchlist 的最新分析。</p>
            </div>
          </div>
          <div className="stack-list">
            {results.length === 0 ? (
              <div className="empty-state">暂无 watchlist 结果。</div>
            ) : (
              results.map((item) => (
                <article className="list-item" key={item.id}>
                  <div className="list-item-title">
                    <strong>{item.primary_entity}</strong>
                    <StatusPill tone={sentimentTone(item.final_label)}>{item.final_label}</StatusPill>
                  </div>
                  <div className="meta-line">
                    <span>{item.event_type}</span>
                    <span>{item.final_source}</span>
                    <span>{formatDateTime(item.created_at)}</span>
                  </div>
                  <p className="panel-copy">{item.explanation}</p>
                </article>
              ))
            )}
          </div>
        </div>

        <div className="panel panel-strong">
          <div className="panel-titlebar">
            <div>
              <h2>使用建议</h2>
              <p className="panel-copy">当前 watchlist 是产品层能力，不影响模型训练逻辑。</p>
            </div>
          </div>
          <div className="stack-list">
            <div className="soft-card">
              先加公司名，再补 ticker。当前后端对 `primary_entity` 做精确命中，对 ticker 做辅助匹配。
            </div>
            <div className="soft-card">如果想把 watchlist 直接纳入告警加权，下一步应该把匹配逻辑前移到 alerting 层。</div>
          </div>
        </div>
      </section>
    </main>
  );
}

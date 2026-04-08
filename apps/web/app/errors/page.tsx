"use client";

import { useEffect, useState } from "react";
import { PageHeader } from "../../components/page-header";
import { StatusPill } from "../../components/status-pill";
import { fetchResults } from "../../lib/api";
import { formatDateTime, sentimentTone } from "../../lib/view";
import type { AnalysisResultRecord } from "../../../../packages/schemas/ts";

export default function ErrorsPage() {
  const [items, setItems] = useState<AnalysisResultRecord[]>([]);
  const [status, setStatus] = useState("加载中...");

  useEffect(() => {
    async function load() {
      try {
        const response = await fetchResults({ limit: 60, error_only: true });
        setItems(response.items);
        setStatus(`已加载 ${response.total} 条错误案例`);
      } catch (error) {
        setStatus(error instanceof Error ? error.message : "加载失败");
      }
    }

    void load();
  }, []);

  return (
    <main className="page-root">
      <PageHeader
        eyebrow="Error Cases"
        title="错误案例回看页"
        description="优先看 LLM 覆盖、人工复核、反馈回流和高风险错例。"
        actions={<StatusPill tone="warning">{status}</StatusPill>}
      />

      <section className="panel">
        <div className="panel-titlebar">
          <div>
            <h2>错误池</h2>
            <p className="panel-copy">当前判定条件：LLM 覆盖、人工复核、或已有人工反馈。</p>
          </div>
        </div>

        {items.length === 0 ? (
          <div className="empty-state">还没有进入错误池的样本。</div>
        ) : (
          <div className="table-wrap">
            <table className="data-table">
              <thead>
                <tr>
                  <th>时间</th>
                  <th>主体 / 事件</th>
                  <th>来源 / Ticker</th>
                  <th>模型原判</th>
                  <th>最终结论</th>
                  <th>原因</th>
                  <th>文本</th>
                </tr>
              </thead>
              <tbody>
                {items.map((item) => (
                  <tr key={item.id}>
                    <td className="muted">{formatDateTime(item.created_at)}</td>
                    <td>
                      <div className="inline-grid">
                        <strong>{item.primary_entity || "未识别主体"}</strong>
                        <span className="muted">{item.event_type}</span>
                      </div>
                    </td>
                    <td>
                      <div className="inline-grid">
                        <span>{item.context.source_name || item.context.news_source || "未填写来源"}</span>
                        <span className="muted">{item.context.ticker || item.result.entities.tickers[0] || "No ticker"}</span>
                      </div>
                    </td>
                    <td>
                      <div className="inline-grid">
                        <StatusPill tone={sentimentTone(item.model_label)}>{item.model_label}</StatusPill>
                        <span className="muted">{item.result.sentiment.model_confidence?.toFixed(2) ?? item.model_confidence.toFixed(2)}</span>
                      </div>
                    </td>
                    <td>
                      <div className="inline-grid">
                        <StatusPill tone={sentimentTone(item.final_label)}>{item.final_label}</StatusPill>
                        <span className="muted">{item.final_source}</span>
                      </div>
                    </td>
                    <td>
                      <div className="inline-grid">
                        {item.result.sentiment.overridden_by_llm ? <StatusPill tone="warning">LLM 覆盖</StatusPill> : null}
                        {item.needs_human_review ? <StatusPill tone="info">人工复核</StatusPill> : null}
                        {item.feedback_count > 0 ? <StatusPill tone="positive">已有反馈</StatusPill> : null}
                        {item.feedback_labels.map((label) => (
                          <StatusPill key={`${item.id}-${label}`} tone={sentimentTone(label)}>
                            {`反馈 ${label}`}
                          </StatusPill>
                        ))}
                      </div>
                    </td>
                    <td>{item.input_text}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </main>
  );
}

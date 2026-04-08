"use client";

import { useMemo, useState } from "react";
import { MetricCard } from "../../components/metric-card";
import { PageHeader } from "../../components/page-header";
import { StatusPill } from "../../components/status-pill";
import { batchAnalyzeItems } from "../../lib/api";
import { sentimentTone } from "../../lib/view";
import type { AnalysisContext, AnalyzeRequest, AnalyzeResponse } from "../../../../packages/schemas/ts";

const sampleLines = [
  "Apple raised full-year guidance after signing a cloud supply agreement.",
  "某制造企业公告称，公司计划关闭一条产线并裁员。",
  "Tesla said it will host an investor event next month.",
];

const structuredSampleLines = [
  "2026-04-08T09:30:00+08:00\tReuters\tReuters Breakingviews\thttps://example.com/apple\tApple Inc.\tAAPL\ttechnology\tearnings\tApple raised full-year guidance after signing a cloud supply agreement.",
  "2026-04-08T10:00:00+08:00\tExchange Filing\tHKEX Filing\thttps://example.com/factory\t示例制造集团\t0941.HK\tindustrial\tlayoffs\t某制造企业公告称，公司计划关闭一条产线并裁员。",
];

function parseBatchInput(input: string): AnalyzeRequest[] {
  return input
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => {
      const columns = line.split("\t");
      if (columns.length >= 9) {
        const [published_at, news_source, source_name, source_url, company_name, ticker, industry, event_type, ...textParts] = columns;
        const context: AnalysisContext = {
          news_source: news_source || undefined,
          source_name: source_name || undefined,
          source_url: source_url || undefined,
          published_at: published_at || undefined,
          company_name: company_name || undefined,
          ticker: ticker || undefined,
          industry: industry || undefined,
          event_type: event_type || undefined,
          historical_announcements: [],
        };
        return {
          text: textParts.join("\t").trim(),
          context,
        };
      }

      return {
        text: line,
        context: {
          historical_announcements: [],
        },
      };
    })
    .filter((item) => item.text.trim());
}

export default function BatchPage() {
  const [input, setInput] = useState(sampleLines.join("\n"));
  const [items, setItems] = useState<AnalyzeResponse[]>([]);
  const [status, setStatus] = useState("等待批量输入");
  const [loading, setLoading] = useState(false);

  const summary = useMemo(() => {
    return items.reduce(
      (accumulator, item) => {
        accumulator.total += 1;
        accumulator[item.sentiment.label] += 1;
        return accumulator;
      },
      { total: 0, positive: 0, neutral: 0, negative: 0 },
    );
  }, [items]);

  async function handleBatchAnalyze() {
    const requests = parseBatchInput(input);

    if (requests.length === 0) {
      setStatus("请至少输入一条文本。");
      return;
    }

    setLoading(true);
    setStatus(`正在分析 ${requests.length} 条文本...`);
    try {
      const response = await batchAnalyzeItems(requests);
      setItems(response.items);
      setStatus(`已完成 ${response.total} 条分析`);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "批量分析失败");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="page-root">
      <PageHeader
        eyebrow="Batch Analyze"
        title="批量分析页"
        description="支持纯文本逐行分析，也支持带 metadata 的 TSV 批量输入。"
        actions={<StatusPill tone="info">{status}</StatusPill>}
      />

      <section className="two-column-grid">
        <div className="panel">
          <div className="panel-titlebar">
            <div>
              <h2>批量输入</h2>
              <p className="panel-copy">纯文本模式一行一条；结构化模式用 tab 分隔 9 列：published_at, news_source, source_name, source_url, company_name, ticker, industry, event_type, text。</p>
            </div>
            <div className="button-row">
              <button className="ghost-button" onClick={() => setInput(sampleLines.join("\n"))} type="button">
                纯文本样例
              </button>
              <button className="ghost-button" onClick={() => setInput(structuredSampleLines.join("\n"))} type="button">
                结构化样例
              </button>
            </div>
          </div>
          <textarea className="text-area" value={input} onChange={(event) => setInput(event.target.value)} />
          <div className="action-row">
            <button className="primary-button" disabled={loading} onClick={handleBatchAnalyze} type="button">
              {loading ? "分析中..." : "提交批量任务"}
            </button>
            <span className="status-text">{status}</span>
          </div>
        </div>

        <div className="panel">
          <div className="panel-titlebar">
            <div>
              <h2>批量摘要</h2>
              <p className="panel-copy">先看分布，再进入明细 table。</p>
            </div>
          </div>
          <div className="result-grid">
            <MetricCard emphasis label="总量" value={String(summary.total)} subtitle="analyzed items" />
            <MetricCard label="积极" value={String(summary.positive)} subtitle="positive" />
            <MetricCard label="中性" value={String(summary.neutral)} subtitle="neutral" />
            <MetricCard label="消极" value={String(summary.negative)} subtitle="negative" />
          </div>
        </div>
      </section>

      <section className="panel">
        <div className="panel-titlebar">
          <div>
            <h2>批量结果</h2>
            <p className="panel-copy">每条记录都已走后端 agent workflow，并会保留来源、时间和 company/ticker 映射。</p>
          </div>
        </div>
        {items.length === 0 ? (
          <div className="empty-state">运行批量分析后，这里会展示结果表。</div>
        ) : (
          <div className="table-wrap">
            <table className="data-table">
              <thead>
                <tr>
                  <th>文本</th>
                  <th>情绪</th>
                  <th>事件</th>
                  <th>主体 / Ticker</th>
                  <th>来源 / 时间</th>
                  <th>复核</th>
                </tr>
              </thead>
              <tbody>
                {items.map((item, index) => (
                  <tr key={`${item.input_text}-${index}`}>
                    <td>{item.input_text}</td>
                    <td>
                      <div className="inline-grid">
                        <StatusPill tone={sentimentTone(item.sentiment.label)}>{item.sentiment.label}</StatusPill>
                        <span className="muted">{item.sentiment.source ?? "sentiment_model"}</span>
                      </div>
                    </td>
                    <td>{item.event.type_zh}</td>
                    <td>{item.entities.companies.join(", ") || item.context.company_name || item.context.ticker || "未识别"}</td>
                    <td>{item.context.source_name || item.context.news_source || item.context.published_at || "未填写"}</td>
                    <td>{item.risk_alert.needs_human_review ? "需要" : "否"}</td>
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

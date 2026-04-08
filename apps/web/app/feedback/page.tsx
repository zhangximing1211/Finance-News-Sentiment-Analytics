"use client";

import { useEffect, useMemo, useState } from "react";
import { PageHeader } from "../../components/page-header";
import { StatusPill } from "../../components/status-pill";
import { createFeedback, createRetrainJob, fetchFeedback, fetchResults } from "../../lib/api";
import { formatDateTime, labelToChinese, sentimentTone } from "../../lib/view";
import type { AnalysisResultRecord, FeedbackRecord, SentimentLabel } from "../../../../packages/schemas/ts";

type FeedbackDraft = {
  feedback_label: SentimentLabel;
  feedback_event_type: string;
  reviewer: string;
  notes: string;
};

export default function FeedbackPage() {
  const [results, setResults] = useState<AnalysisResultRecord[]>([]);
  const [feedback, setFeedback] = useState<FeedbackRecord[]>([]);
  const [status, setStatus] = useState("加载中...");
  const [drafts, setDrafts] = useState<Record<number, FeedbackDraft>>({});

  async function load() {
    try {
      const [resultsResponse, feedbackResponse] = await Promise.all([
        fetchResults({ limit: 20, error_only: true }),
        fetchFeedback({ limit: 20 }),
      ]);
      setResults(resultsResponse.items);
      setFeedback(feedbackResponse.items);
      setStatus("反馈数据已同步");
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "加载失败");
    }
  }

  useEffect(() => {
    void load();
  }, []);

  const feedbackCount = useMemo(() => feedback.length, [feedback]);

  function draftFor(item: AnalysisResultRecord): FeedbackDraft {
    return (
      drafts[item.id] ?? {
        feedback_label: item.final_label as SentimentLabel,
        feedback_event_type: item.event_type,
        reviewer: "",
        notes: "",
      }
    );
  }

  async function handleSubmit(item: AnalysisResultRecord) {
    const draft = draftFor(item);
    setStatus(`提交反馈 #${item.id}...`);
    try {
      await createFeedback({
        analysis_run_id: item.id,
        feedback_label: draft.feedback_label,
        feedback_event_type: draft.feedback_event_type || undefined,
        reviewer: draft.reviewer || undefined,
        notes: draft.notes || undefined,
      });
      await load();
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "提交失败");
    }
  }

  async function handleRetrain() {
    setStatus("创建重训任务...");
    try {
      const job = await createRetrainJob({
        trigger_source: "manual_feedback",
        include_feedback_only: true,
        requested_by: "web-console",
        notes: "Triggered from feedback page.",
      });
      setStatus(`已创建重训任务 #${job.id}`);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "重训任务创建失败");
    }
  }

  return (
    <main className="page-root">
      <PageHeader
        eyebrow="Human Feedback"
        title="人工反馈标注页"
        description="对错例重新标注，并将纠偏数据送入 retrain 队列。"
        actions={
          <div className="button-row">
            <StatusPill tone="info">{status}</StatusPill>
            <button className="secondary-button" onClick={handleRetrain} type="button">
              发起重训
            </button>
          </div>
        }
      />

      <section className="dashboard-grid">
        <div className="metric-card metric-card-emphasis">
          <span className="metric-label">待标注错例</span>
          <strong className="metric-value">{results.length}</strong>
          <p className="metric-subtitle">来自错误案例池</p>
        </div>
        <div className="metric-card">
          <span className="metric-label">近期反馈</span>
          <strong className="metric-value">{feedbackCount}</strong>
          <p className="metric-subtitle">最近 20 条反馈记录</p>
        </div>
      </section>

      <section className="two-column-grid">
        <div className="panel">
          <div className="panel-titlebar">
            <div>
              <h2>待标注样本</h2>
              <p className="panel-copy">直接对结果库里的错例提交人工标签。</p>
            </div>
          </div>

          <div className="stack-list">
            {results.length === 0 ? (
              <div className="empty-state">当前没有待标注错例。</div>
            ) : (
              results.map((item) => {
                const draft = draftFor(item);
                return (
                  <article className="feedback-row" key={item.id}>
                    <div className="list-item-title">
                      <strong>{item.primary_entity || "未识别主体"}</strong>
                      <StatusPill tone={sentimentTone(item.final_label)}>{item.final_label}</StatusPill>
                    </div>
                    <div className="meta-line">
                      <span>model {item.model_label}</span>
                      <span>final {item.final_label}</span>
                      <span>{item.context.source_name || item.context.news_source || "未填写来源"}</span>
                      <span>{formatDateTime(item.created_at)}</span>
                    </div>
                    <p className="panel-copy">{item.input_text}</p>

                    <div className="feedback-form">
                      <select
                        className="select-input"
                        value={draft.feedback_label}
                        onChange={(event) =>
                          setDrafts((current) => ({
                            ...current,
                            [item.id]: { ...draft, feedback_label: event.target.value as SentimentLabel },
                          }))
                        }
                      >
                        <option value="positive">积极 / positive</option>
                        <option value="neutral">中性 / neutral</option>
                        <option value="negative">消极 / negative</option>
                      </select>
                      <input
                        className="text-input"
                        placeholder="feedback event type"
                        value={draft.feedback_event_type}
                        onChange={(event) =>
                          setDrafts((current) => ({
                            ...current,
                            [item.id]: { ...draft, feedback_event_type: event.target.value },
                          }))
                        }
                      />
                      <input
                        className="text-input"
                        placeholder="reviewer"
                        value={draft.reviewer}
                        onChange={(event) =>
                          setDrafts((current) => ({
                            ...current,
                            [item.id]: { ...draft, reviewer: event.target.value },
                          }))
                        }
                      />
                      <textarea
                        className="text-area"
                        placeholder="反馈备注"
                        value={draft.notes}
                        onChange={(event) =>
                          setDrafts((current) => ({
                            ...current,
                            [item.id]: { ...draft, notes: event.target.value },
                          }))
                        }
                      />
                      <div className="action-row">
                        <button className="primary-button" onClick={() => void handleSubmit(item)} type="button">
                          提交反馈
                        </button>
                        <span className="status-text">建议标签：{labelToChinese(draft.feedback_label)}</span>
                      </div>
                    </div>
                  </article>
                );
              })
            )}
          </div>
        </div>

        <div className="panel">
          <div className="panel-titlebar">
            <div>
              <h2>最近反馈</h2>
              <p className="panel-copy">这里是已经写入后端的反馈记录。</p>
            </div>
          </div>

          <div className="stack-list">
            {feedback.length === 0 ? (
              <div className="empty-state">还没有历史反馈。</div>
            ) : (
              feedback.map((item) => (
                <article className="list-item" key={item.id}>
                  <div className="list-item-title">
                    <strong>Run #{item.analysis_run_id}</strong>
                    <StatusPill tone={sentimentTone(item.feedback_label)}>{item.feedback_label}</StatusPill>
                  </div>
                  <div className="meta-line">
                    <span>{item.reviewer || "anonymous"}</span>
                    <span>{item.feedback_event_type || "event unchanged"}</span>
                    <span>{formatDateTime(item.created_at)}</span>
                  </div>
                  {item.notes ? <p className="panel-copy">{item.notes}</p> : null}
                </article>
              ))
            )}
          </div>
        </div>
      </section>
    </main>
  );
}

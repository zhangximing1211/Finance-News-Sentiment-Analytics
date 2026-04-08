from __future__ import annotations

from datetime import date
from typing import Any

from .review_queue import ReviewQueueRepository
from .workflow_store import AgentWorkflowRepository


class FeedbackLoopService:
    def __init__(
        self,
        workflow_repository: AgentWorkflowRepository | None = None,
        review_queue_repository: ReviewQueueRepository | None = None,
    ) -> None:
        self.workflow_repository = workflow_repository or AgentWorkflowRepository()
        self.review_queue_repository = review_queue_repository or ReviewQueueRepository()

    def auto_sample_review(self, *, limit: int = 12, candidate_limit: int = 200) -> dict[str, Any]:
        results = self.workflow_repository.list_results(limit=candidate_limit)
        sampled_run_ids: list[int] = []
        sampled_labels: set[str] = set()
        sampled_sources: set[str] = set()

        for item in results:
            if len(sampled_run_ids) >= limit:
                break
            if item["review_queue_record_id"] or item.get("in_error_pool"):
                continue
            if item["final_confidence"] < 0.72:
                continue

            source_name = item["context"].get("source_name") or item["context"].get("news_source") or "unknown"
            should_sample = item["final_label"] not in sampled_labels or source_name not in sampled_sources
            if not should_sample and len(sampled_run_ids) >= max(3, limit // 2):
                continue

            analysis_result = dict(item["result"])
            analysis_result["risk_alert"] = {
                **analysis_result.get("risk_alert", {}),
                "needs_human_review": True,
                "message": "自动采样复核：用于监控高置信度样本和新闻源分布偏移。",
                "reasons": list(
                    dict.fromkeys(list(analysis_result.get("risk_alert", {}).get("reasons", [])) + ["自动采样复核"])
                ),
            }
            analysis_result["review_queue_item"] = {
                "queue": "audit_sample",
                "priority": "medium",
                "primary_entity": item["primary_entity"],
                "event_type": item["event_type"],
                "predicted_label": item["model_label"],
                "decision_label": item["final_label"],
                "confidence": item["final_confidence"],
                "review_reasons": ["audit_sample", "distribution_monitoring"],
                "recommended_action": "Audit sampled production item for drift monitoring and quality control.",
                "text_excerpt": item["input_text"][:240],
            }

            persisted = self.review_queue_repository.enqueue_analysis(analysis_result)
            if not persisted:
                continue

            sampled_run_ids.append(item["id"])
            sampled_labels.add(item["final_label"])
            sampled_sources.add(source_name)

        return {
            "auto_sampled_review_count": len(sampled_run_ids),
            "sampled_run_ids": sampled_run_ids,
        }

    def run_maintenance(
        self,
        *,
        report_type: str = "weekly",
        reference_date: date | None = None,
        sample_limit: int = 12,
    ) -> dict[str, Any]:
        report = self.workflow_repository.generate_report(report_type=report_type, reference_date=reference_date)
        sampled = self.auto_sample_review(limit=sample_limit)

        retrain_job = None
        if report["feedback_loop_assets"]["periodic_retrain_due"]:
            retrain_job = self.workflow_repository.create_retrain_job(
                trigger_source="scheduled_feedback_loop",
                include_feedback_only=False,
                requested_by="feedback-loop-maintenance",
                notes=(
                    f"Auto-scheduled after {report_type} maintenance. "
                    f"correction_rate={report['monitoring']['user_correction_rate']}, "
                    f"class_drift={report['monitoring']['class_drift']['score']}, "
                    f"source_shift={report['monitoring']['source_shift']['score']}"
                ),
            )

        return {
            "auto_sampled_review_count": sampled["auto_sampled_review_count"],
            "sampled_run_ids": sampled["sampled_run_ids"],
            "periodic_retrain_due": report["feedback_loop_assets"]["periodic_retrain_due"],
            "retrain_job": retrain_job,
            "error_sample_pool_size": report["feedback_loop_assets"]["error_sample_pool_size"],
            "golden_test_set_size": report["feedback_loop_assets"]["golden_test_set_size"],
        }

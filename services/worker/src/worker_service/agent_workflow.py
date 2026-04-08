from __future__ import annotations

from pathlib import Path
import sys
from typing import Any


BASE_DIR = Path(__file__).resolve().parents[4]
MODEL_SERVING_SRC = BASE_DIR / "services" / "model-serving" / "src"

if str(MODEL_SERVING_SRC) not in sys.path:
    sys.path.insert(0, str(MODEL_SERVING_SRC))

from model_serving import AnalysisService
from model_serving.llm_reviewer import LLMReviewer

from .alerting import decide_alert
from .feedback_loop import FeedbackLoopService
from .review_queue import ReviewQueueRepository
from .workflow_store import AgentWorkflowRepository


SENTIMENT_LABELS_ZH = {
    "positive": "积极",
    "neutral": "中性",
    "negative": "消极",
}


class AgentWorkflowService:
    def __init__(
        self,
        analysis_service: AnalysisService | None = None,
        review_queue_repository: ReviewQueueRepository | None = None,
        workflow_repository: AgentWorkflowRepository | None = None,
        llm_reviewer: LLMReviewer | None = None,
    ) -> None:
        self.analysis_service = analysis_service or AnalysisService()
        self.review_queue_repository = review_queue_repository or ReviewQueueRepository()
        self.workflow_repository = workflow_repository or AgentWorkflowRepository()
        self.llm_reviewer = llm_reviewer or LLMReviewer()
        self.feedback_loop_service = FeedbackLoopService(
            workflow_repository=self.workflow_repository,
            review_queue_repository=self.review_queue_repository,
        )

    def run_text(self, text: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        workflow_steps = [{"step": "receive_text", "status": "completed"}]
        result = self.analysis_service.analyze_text(text, context=context)
        model_decision = {
            "label": result["sentiment"]["label"],
            "label_zh": result["sentiment"]["label_zh"],
            "confidence": round(float(result["sentiment"]["confidence"]), 4),
            "decision_label": result["sentiment"]["decision_label"],
            "abstained": bool(result["sentiment"]["abstained"]),
            "source": "sentiment_model",
        }

        workflow_steps.append(
            {
                "step": "extract_entities_and_event",
                "status": "completed",
                "primary_entity": result["entities"]["companies"][0] if result["entities"]["companies"] else (result["entities"]["tickers"][0] if result["entities"]["tickers"] else "unknown"),
                "event_type": result["event"]["type"],
            }
        )
        workflow_steps.append(
            {
                "step": "run_sentiment_model",
                "status": "completed",
                "label": result["sentiment"]["label"],
                "decision_label": result["sentiment"]["decision_label"],
                "confidence": result["sentiment"]["confidence"],
            }
        )

        low_confidence = bool(result["sentiment"]["abstained"]) or (
            result.get("review_queue_item") is not None
            and "low_confidence" in result["review_queue_item"].get("review_reasons", [])
        )

        capability_decision = {
            "decision_label": result["sentiment"]["decision_label"],
            "top_label": result["sentiment"]["label"],
            "confidence": result["sentiment"]["confidence"],
            "review_reasons": result["review_queue_item"]["review_reasons"] if result.get("review_queue_item") else [],
            "neutral_boundary": result.get("review_queue_item") is not None
            and "neutral_boundary" in result["review_queue_item"].get("review_reasons", []),
        }

        if low_confidence:
            llm_review = self.llm_reviewer.review(
                input_text=result["input_text"],
                sentiment=result["sentiment"],
                event=result["event"],
                entities=result["entities"],
                risk_alert=result["risk_alert"],
                capability_decision=capability_decision,
            )
            workflow_steps.append(
                {
                    "step": "llm_rejudge",
                    "status": "completed" if llm_review["used_external_llm"] else "fallback",
                    "provider": llm_review["provider"],
                    "should_override": llm_review["should_override"],
                }
            )
        else:
            llm_review = self.llm_reviewer.skipped(sentiment=result["sentiment"])
            workflow_steps.append({"step": "llm_rejudge", "status": "skipped"})

        final_decision = self._resolve_final_decision(result, llm_review)
        self._apply_final_decision(result=result, model_decision=model_decision, final_decision=final_decision)
        if llm_review["triggered"] and llm_review["review_summary"]:
            result["explanation"] = llm_review["review_summary"]

        persisted_queue_item = self.review_queue_repository.enqueue_analysis(result)
        if persisted_queue_item and result.get("review_queue_item"):
            result["review_queue_item"] = {
                **result["review_queue_item"],
                "record_id": persisted_queue_item["id"],
                "status": persisted_queue_item["status"],
                "attempts": persisted_queue_item["attempts"],
                "llm_provider": persisted_queue_item["llm_provider"],
                "last_error": persisted_queue_item["last_error"],
            }
        workflow_steps.append(
            {
                "step": "persist_review_queue",
                "status": "completed" if persisted_queue_item else "skipped",
                "record_id": persisted_queue_item["id"] if persisted_queue_item else None,
            }
        )

        result.setdefault("metadata", {})
        result["metadata"]["agent_workflow"] = {
            "low_confidence_triggered": low_confidence,
            "model_decision": model_decision,
            "llm_review": llm_review,
            "final_decision": final_decision,
            "workflow_steps": workflow_steps,
        }

        alert = decide_alert(result, final_decision)
        workflow_steps.append(
            {
                "step": "decide_alert",
                "status": "completed",
                "triggered": alert["triggered"],
                "severity": alert["severity"],
            }
        )

        persisted_run = self.workflow_repository.create_run(
            analysis_result=result,
            model_decision=model_decision,
            final_decision=final_decision,
            llm_review=llm_review,
            alert=alert,
            workflow_steps=workflow_steps,
            review_queue_record_id=result.get("review_queue_item", {}).get("record_id") if result.get("review_queue_item") else None,
        )
        workflow_steps.append(
            {
                "step": "write_to_database",
                "status": "completed",
                "run_id": persisted_run["id"],
            }
        )
        workflow_steps.append(
            {
                "step": "aggregate_report_targets",
                "status": "completed",
                "targets": ["daily", "weekly"],
            }
        )

        result["metadata"]["agent_workflow"] = {
            "run_id": persisted_run["id"],
            "created_at": persisted_run["created_at"],
            "review_queue_record_id": persisted_run["review_queue_record_id"],
            "low_confidence_triggered": low_confidence,
            "model_decision": model_decision,
            "llm_review": llm_review,
            "final_decision": final_decision,
            "alert": persisted_run["alert"] or alert,
            "workflow_steps": workflow_steps,
            "report_targets": ["daily", "weekly"],
        }
        return result

    def batch_run_texts(self, texts: list[str]) -> list[dict[str, Any]]:
        return [self.run_text(text) for text in texts if text.strip()]

    def batch_run_requests(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for item in items:
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            results.append(self.run_text(text, context=item.get("context")))
        return results

    def list_results(
        self,
        *,
        limit: int = 50,
        label: str | None = None,
        event_type: str | None = None,
        entity_query: str | None = None,
        source: str | None = None,
        error_only: bool = False,
        watchlist_only: bool = False,
    ) -> list[dict[str, Any]]:
        return self.workflow_repository.list_results(
            limit=limit,
            label=label,
            event_type=event_type,
            entity_query=entity_query,
            source=source,
            error_only=error_only,
            watchlist_only=watchlist_only,
        )

    def list_alerts(
        self,
        status: str | None = None,
        severity: str | None = None,
        limit: int = 50,
        watchlist_only: bool = False,
    ) -> list[dict[str, Any]]:
        return self.workflow_repository.list_alerts(
            status=status,
            severity=severity,
            limit=limit,
            watchlist_only=watchlist_only,
        )

    def add_watchlist_item(
        self,
        *,
        company_name: str,
        ticker: str | None = None,
        industry: str | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        return self.workflow_repository.add_watchlist_item(
            company_name=company_name,
            ticker=ticker,
            industry=industry,
            notes=notes,
        )

    def list_watchlist(self, *, limit: int = 100) -> list[dict[str, Any]]:
        return self.workflow_repository.list_watchlist(limit=limit)

    def create_feedback(
        self,
        *,
        analysis_run_id: int,
        feedback_label: str,
        feedback_event_type: str | None = None,
        reviewer: str | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        return self.workflow_repository.create_feedback(
            analysis_run_id=analysis_run_id,
            feedback_label=feedback_label,
            feedback_event_type=feedback_event_type,
            reviewer=reviewer,
            notes=notes,
        )

    def list_feedback(self, *, limit: int = 100, analysis_run_id: int | None = None) -> list[dict[str, Any]]:
        return self.workflow_repository.list_feedback(limit=limit, analysis_run_id=analysis_run_id)

    def create_retrain_job(
        self,
        *,
        trigger_source: str,
        include_feedback_only: bool,
        requested_by: str | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        return self.workflow_repository.create_retrain_job(
            trigger_source=trigger_source,
            include_feedback_only=include_feedback_only,
            requested_by=requested_by,
            notes=notes,
        )

    def list_error_samples(self, *, limit: int = 100, status: str = "open") -> list[dict[str, Any]]:
        return self.workflow_repository.list_error_samples(limit=limit, status=status)

    def add_golden_test_case(
        self,
        *,
        input_text: str,
        expected_label: str,
        expected_event_type: str | None = None,
        title: str | None = None,
        source_name: str | None = None,
        notes: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.workflow_repository.add_golden_test_case(
            input_text=input_text,
            expected_label=expected_label,
            expected_event_type=expected_event_type,
            title=title,
            source_name=source_name,
            notes=notes,
            context=context,
        )

    def list_golden_test_cases(self, *, limit: int = 100, active_only: bool = True) -> list[dict[str, Any]]:
        return self.workflow_repository.list_golden_test_cases(limit=limit, active_only=active_only)

    def run_feedback_loop_maintenance(
        self,
        *,
        report_type: str = "weekly",
        reference_date=None,
        sample_limit: int = 12,
    ) -> dict[str, Any]:
        return self.feedback_loop_service.run_maintenance(
            report_type=report_type,
            reference_date=reference_date,
            sample_limit=sample_limit,
        )

    def generate_report(self, report_type: str, reference_date=None) -> dict[str, Any]:
        return self.workflow_repository.generate_report(report_type=report_type, reference_date=reference_date)

    def _resolve_final_decision(self, result: dict[str, Any], llm_review: dict[str, Any]) -> dict[str, Any]:
        if llm_review["triggered"] and llm_review["used_external_llm"] and llm_review["should_override"]:
            return {
                "label": llm_review["reviewed_label"],
                "label_zh": SENTIMENT_LABELS_ZH[llm_review["reviewed_label"]],
                "confidence": round(float(llm_review["reviewed_confidence"]), 4),
                "source": "llm_rejudge",
            }
        return {
            "label": result["sentiment"]["label"],
            "label_zh": result["sentiment"]["label_zh"],
            "confidence": round(float(result["sentiment"]["confidence"]), 4),
            "source": "sentiment_model",
        }

    def _apply_final_decision(
        self,
        *,
        result: dict[str, Any],
        model_decision: dict[str, Any],
        final_decision: dict[str, Any],
    ) -> None:
        result["sentiment"]["source"] = final_decision["source"]
        result["sentiment"]["overridden_by_llm"] = final_decision["source"] == "llm_rejudge"
        result["sentiment"]["model_label"] = model_decision["label"]
        result["sentiment"]["model_label_zh"] = model_decision["label_zh"]
        result["sentiment"]["model_confidence"] = model_decision["confidence"]

        if final_decision["source"] != "llm_rejudge":
            return

        result["sentiment"]["label"] = final_decision["label"]
        result["sentiment"]["label_zh"] = final_decision["label_zh"]
        result["sentiment"]["confidence"] = round(float(final_decision["confidence"]), 4)
        result["sentiment"]["decision_label"] = final_decision["label"]
        result["sentiment"]["abstained"] = False


_AGENT_WORKFLOW_SERVICE: AgentWorkflowService | None = None


def get_agent_workflow_service() -> AgentWorkflowService:
    global _AGENT_WORKFLOW_SERVICE
    if _AGENT_WORKFLOW_SERVICE is None:
        _AGENT_WORKFLOW_SERVICE = AgentWorkflowService()
    return _AGENT_WORKFLOW_SERVICE

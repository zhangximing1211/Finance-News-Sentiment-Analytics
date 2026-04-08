from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
MODEL_SERVING_SRC = ROOT / "services" / "model-serving" / "src"
WORKER_SRC = ROOT / "services" / "worker" / "src"

for path in [MODEL_SERVING_SRC, WORKER_SRC]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from model_serving import AnalysisService
from worker_service import AgentWorkflowRepository, AgentWorkflowService, ReviewQueueRepository


class FakeAnalysisService:
    def analyze_text(self, text: str, context: dict[str, object] | None = None) -> dict[str, object]:
        return {
            "input_text": text,
            "context": {
                "news_source": context.get("news_source") if context else None,
                "source_name": context.get("source_name") if context else None,
                "source_url": None,
                "published_at": context.get("published_at") if context else None,
                "company_name": "Example Corp",
                "ticker": "EXM",
                "industry": "technology",
                "event_type": "earnings",
                "historical_announcements": context.get("historical_announcements", []) if context else [],
            },
            "sentiment": {
                "label": "neutral",
                "label_zh": "中性",
                "confidence": 0.48,
                "probabilities": {"negative": 0.27, "neutral": 0.48, "positive": 0.25},
                "decision_label": "abstain",
                "abstained": True,
                "confidence_gap": 0.21,
                "low_confidence_threshold": 0.62,
            },
            "event": {
                "type": "earnings",
                "type_zh": "财报",
                "matched_signals": ["财报指标"],
                "secondary_type": "unknown",
            },
            "entities": {
                "companies": ["Example Corp"],
                "tickers": [],
                "industry": "technology",
                "industry_zh": "科技",
            },
            "explanation": "模型初判偏中性。",
            "risk_alert": {
                "needs_human_review": True,
                "message": "建议人工复核：置信度偏低。",
                "reasons": ["置信度偏低"],
            },
            "review_queue_item": {
                "queue": "human_review",
                "priority": "high",
                "primary_entity": "Example Corp",
                "event_type": "earnings",
                "predicted_label": "neutral",
                "decision_label": "abstain",
                "confidence": 0.48,
                "review_reasons": ["low_confidence"],
                "recommended_action": "Escalate to analyst review before auto-publishing this classification.",
                "text_excerpt": text[:240],
            },
            "secondary_explanation": {
                "provider": "template_fallback",
                "template_path": "template.md",
                "summary": "原始二次解释。",
                "review_note": "需要人工复核。",
                "rationale": "原始模型置信度偏低。",
                "llm_ready": True,
                "used_external_llm": False,
                "prompt_available": True,
                "input_excerpt": text[:240],
                "risk_message": "建议人工复核：置信度偏低。",
                "fallback_reason": "openai_not_configured",
            },
            "metadata": {
                "used_ml_model": True,
                "used_rule_engine": True,
                "training_error": None,
                "model_source": "artifact",
                "model_path": "artifact.joblib",
                "model_metadata_path": "artifact.json",
                "capability_module": {
                    "probability_source": "calibrated_ml_plus_rules",
                    "ml_probabilities": {"negative": 0.27, "neutral": 0.48, "positive": 0.25},
                    "rule_probabilities": {"negative": 0.3, "neutral": 0.4, "positive": 0.3},
                    "final_probabilities": {"negative": 0.27, "neutral": 0.48, "positive": 0.25},
                    "signal_confidence_estimate": 0.48,
                    "low_confidence_threshold": 0.62,
                    "neutral_boundary_margin": 0.08,
                    "review_reasons": ["low_confidence"],
                    "ranked_labels": [
                        {"label": "neutral", "probability": 0.48},
                        {"label": "negative", "probability": 0.27},
                        {"label": "positive", "probability": 0.25},
                    ],
                },
            },
        }


class FakeLLMReviewer:
    external_llm_enabled = True

    def review(self, **_: object) -> dict[str, object]:
        return {
            "provider": "openai_responses_api",
            "used_external_llm": True,
            "triggered": True,
            "should_override": True,
            "reviewed_label": "negative",
            "reviewed_confidence": 0.81,
            "review_summary": "LLM 复判认为该文本偏负面。",
            "review_rationale": "文本包含明确亏损表述，模型原判断过于保守。",
            "fallback_reason": None,
        }

    def skipped(self, *, sentiment: dict[str, object], reason: str = "confidence_above_threshold") -> dict[str, object]:
        return {
            "provider": "workflow_skip",
            "used_external_llm": False,
            "triggered": False,
            "should_override": False,
            "reviewed_label": sentiment["label"],
            "reviewed_confidence": sentiment["confidence"],
            "review_summary": "未触发 LLM 复判。",
            "review_rationale": "模型置信度足够高。",
            "fallback_reason": reason,
        }


class AgentWorkflowTests(unittest.TestCase):
    def test_low_confidence_flow_persists_run_and_alert(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "agent_workflow.sqlite3"
            service = AgentWorkflowService(
                analysis_service=FakeAnalysisService(),
                review_queue_repository=ReviewQueueRepository(db_path),
                workflow_repository=AgentWorkflowRepository(db_path),
                llm_reviewer=FakeLLMReviewer(),
            )

            result = service.run_text("The result before taxes was a loss of 25.0 million euros.")
            workflow = result["metadata"]["agent_workflow"]

            self.assertTrue(workflow["low_confidence_triggered"])
            self.assertEqual(workflow["model_decision"]["label"], "neutral")
            self.assertEqual(workflow["final_decision"]["source"], "llm_rejudge")
            self.assertEqual(workflow["final_decision"]["label"], "negative")
            self.assertEqual(result["sentiment"]["label"], "negative")
            self.assertEqual(result["sentiment"]["source"], "llm_rejudge")
            self.assertTrue(result["sentiment"]["overridden_by_llm"])
            self.assertEqual(result["sentiment"]["model_label"], "neutral")
            self.assertIsNotNone(workflow["run_id"])
            self.assertTrue(workflow["alert"]["triggered"])

            alerts = service.list_alerts(limit=10)
            self.assertEqual(len(alerts), 1)
            self.assertEqual(alerts[0]["final_label"], "negative")

    def test_context_round_trips_into_persisted_result(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "agent_workflow.sqlite3"
            service = AgentWorkflowService(
                analysis_service=FakeAnalysisService(),
                review_queue_repository=ReviewQueueRepository(db_path),
                workflow_repository=AgentWorkflowRepository(db_path),
                llm_reviewer=FakeLLMReviewer(),
            )

            service.run_text(
                "The result before taxes was a loss of 25.0 million euros.",
                context={
                    "news_source": "Reuters",
                    "source_name": "Reuters Breakingviews",
                    "published_at": "2026-04-08T09:30:00+08:00",
                    "historical_announcements": [
                        {
                            "announced_at": "2025-12-01",
                            "event_type": "guidance",
                            "summary": "Raised full-year guidance.",
                        }
                    ],
                },
            )

            results = service.list_results(limit=10)
            self.assertEqual(results[0]["context"]["news_source"], "Reuters")
            self.assertEqual(results[0]["context"]["source_name"], "Reuters Breakingviews")
            self.assertEqual(len(results[0]["context"]["historical_announcements"]), 1)

    def test_report_generation_aggregates_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "agent_workflow.sqlite3"
            service = AgentWorkflowService(
                analysis_service=AnalysisService(),
                review_queue_repository=ReviewQueueRepository(db_path),
                workflow_repository=AgentWorkflowRepository(db_path),
                llm_reviewer=FakeLLMReviewer(),
            )

            service.run_text("The result before taxes was a loss of 25.0 million euros.")
            second_result = service.run_text("Apple Inc. (NASDAQ: AAPL) raised its full-year revenue guidance after signing a new supply agreement.")
            reference_date = datetime.fromisoformat(second_result["metadata"]["agent_workflow"]["created_at"]).date()
            report = service.generate_report(report_type="daily", reference_date=reference_date)

            self.assertGreaterEqual(report["total_runs"], 2)
            self.assertIn("negative", report["sentiment_breakdown"])
            self.assertIn("guidance", report["event_breakdown"])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
WORKER_SRC = ROOT / "services" / "worker" / "src"

if str(WORKER_SRC) not in sys.path:
    sys.path.insert(0, str(WORKER_SRC))

from worker_service import AgentWorkflowRepository, FeedbackLoopService, ReviewQueueRepository

from tests.unit.test_product_repository import build_analysis_result


class FeedbackLoopTests(unittest.TestCase):
    def test_generate_report_contains_monitoring_and_assets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repository = AgentWorkflowRepository(Path(temp_dir) / "feedback_loop.sqlite3")
            analysis_result = build_analysis_result()

            run = repository.create_run(
                analysis_result=analysis_result,
                model_decision={"label": "neutral", "confidence": 0.49},
                final_decision={"label": "neutral", "confidence": 0.49, "source": "sentiment_model"},
                llm_review=analysis_result["metadata"]["agent_workflow"]["llm_review"],
                alert={"triggered": False, "severity": "none", "status": "suppressed", "reasons": [], "message": "当前无需触发告警。"},
                workflow_steps=[{"step": "receive_text", "status": "completed"}],
                review_queue_record_id=None,
            )
            repository.create_feedback(
                analysis_run_id=run["id"],
                feedback_label="negative",
                feedback_event_type="guidance",
                reviewer="qa-user",
            )
            repository.add_golden_test_case(
                input_text="Example Corp warned demand may soften next quarter.",
                expected_label="negative",
                source_name="Reuters",
                context=analysis_result["context"],
            )

            report = repository.generate_report(report_type="daily")
            self.assertIn("monitoring", report)
            self.assertIn("feedback_loop_assets", report)
            self.assertIn("class_drift", report["monitoring"])
            self.assertIn("per_class_metrics", report["monitoring"])
            self.assertGreaterEqual(report["feedback_loop_assets"]["error_sample_pool_size"], 1)
            self.assertEqual(report["feedback_loop_assets"]["golden_test_set_size"], 1)

    def test_feedback_loop_maintenance_creates_audit_samples(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "feedback_loop.sqlite3"
            repository = AgentWorkflowRepository(db_path)
            review_repository = ReviewQueueRepository(db_path)
            feedback_loop = FeedbackLoopService(
                workflow_repository=repository,
                review_queue_repository=review_repository,
            )

            analysis_result = build_analysis_result()
            analysis_result["risk_alert"]["needs_human_review"] = False
            analysis_result["risk_alert"]["reasons"] = []
            analysis_result["risk_alert"]["message"] = "可自动通过。"
            analysis_result["review_queue_item"] = None
            analysis_result["sentiment"]["abstained"] = False
            analysis_result["sentiment"]["confidence"] = 0.91
            analysis_result["sentiment"]["decision_label"] = "neutral"

            repository.create_run(
                analysis_result=analysis_result,
                model_decision={"label": "neutral", "confidence": 0.91},
                final_decision={"label": "neutral", "confidence": 0.91, "source": "sentiment_model"},
                llm_review=analysis_result["metadata"]["agent_workflow"]["llm_review"],
                alert={"triggered": False, "severity": "none", "status": "suppressed", "reasons": [], "message": "当前无需触发告警。"},
                workflow_steps=[{"step": "receive_text", "status": "completed"}],
                review_queue_record_id=None,
            )

            summary = feedback_loop.run_maintenance(report_type="weekly", sample_limit=5)
            self.assertGreaterEqual(summary["auto_sampled_review_count"], 1)
            records = review_repository.list_items(limit=10)
            self.assertEqual(records[0]["review_reasons"], ["audit_sample", "distribution_monitoring"])


if __name__ == "__main__":
    unittest.main()

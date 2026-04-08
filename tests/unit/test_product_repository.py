from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
WORKER_SRC = ROOT / "services" / "worker" / "src"

if str(WORKER_SRC) not in sys.path:
    sys.path.insert(0, str(WORKER_SRC))

from worker_service import AgentWorkflowRepository


def build_analysis_result(primary_entity: str = "Example Corp") -> dict[str, object]:
    return {
        "input_text": "Example Corp warned demand may soften next quarter.",
        "context": {
            "news_source": "Reuters",
            "source_name": "Reuters Breakingviews",
            "source_url": "https://example.com/story",
            "published_at": "2026-04-08T09:30:00+08:00",
            "company_name": primary_entity,
            "ticker": "EXM",
            "industry": "technology",
            "event_type": "guidance",
            "historical_announcements": [
                {
                    "announced_at": "2025-12-01",
                    "event_type": "guidance",
                    "summary": "Company previously raised guidance.",
                }
            ],
        },
        "sentiment": {
            "label": "neutral",
            "label_zh": "中性",
            "confidence": 0.49,
            "probabilities": {"negative": 0.29, "neutral": 0.49, "positive": 0.22},
            "decision_label": "abstain",
            "abstained": True,
            "confidence_gap": 0.2,
            "low_confidence_threshold": 0.62,
            "source": "sentiment_model",
            "overridden_by_llm": False,
            "model_label": "neutral",
            "model_label_zh": "中性",
            "model_confidence": 0.49,
        },
        "event": {
            "type": "guidance",
            "type_zh": "指引更新",
            "matched_signals": ["指引展望"],
            "secondary_type": "earnings",
        },
        "entities": {
            "companies": [primary_entity],
            "tickers": ["EXM"],
            "industry": "technology",
            "industry_zh": "科技",
        },
        "explanation": "模型认为文本偏中性。",
        "risk_alert": {
            "needs_human_review": True,
            "message": "建议人工复核：置信度偏低。",
            "reasons": ["置信度偏低"],
        },
        "review_queue_item": None,
        "secondary_explanation": {
            "provider": "openai_responses_api",
            "template_path": "template.md",
            "summary": "需求存在不确定性。",
            "review_note": "建议继续关注。",
            "rationale": "文本包含 may 和 guidance 更新。",
            "llm_ready": True,
            "used_external_llm": True,
            "prompt_available": True,
            "input_excerpt": "Example Corp warned demand may soften next quarter.",
            "risk_message": "建议人工复核：置信度偏低。",
            "fallback_reason": None,
        },
        "metadata": {
            "agent_workflow": {
                "llm_review": {
                    "provider": "openai_responses_api",
                    "used_external_llm": True,
                    "triggered": True,
                    "should_override": False,
                    "reviewed_label": "neutral",
                    "reviewed_confidence": 0.49,
                    "review_summary": "维持中性。",
                    "review_rationale": "文本方向弱。",
                    "fallback_reason": None,
                }
            }
        },
    }


class ProductRepositoryTests(unittest.TestCase):
    def test_watchlist_results_feedback_and_retrain_job(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repository = AgentWorkflowRepository(Path(temp_dir) / "product.sqlite3")
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

            watchlist_item = repository.add_watchlist_item(company_name="Example Corp", ticker="EXM", industry="technology")
            self.assertEqual(watchlist_item["company_name"], "Example Corp")

            results = repository.list_results(limit=10, watchlist_only=True)
            self.assertEqual(len(results), 1)
            self.assertTrue(results[0]["watchlist_match"])
            self.assertEqual(results[0]["id"], run["id"])

            feedback = repository.create_feedback(
                analysis_run_id=run["id"],
                feedback_label="negative",
                reviewer="qa-user",
                notes="This should be reviewed as negative.",
            )
            self.assertEqual(feedback["feedback_label"], "negative")

            refreshed_results = repository.list_results(limit=10)
            self.assertEqual(refreshed_results[0]["feedback_count"], 1)
            self.assertEqual(refreshed_results[0]["latest_feedback"]["feedback_label"], "negative")
            self.assertEqual(refreshed_results[0]["feedback_labels"], ["negative"])
            self.assertEqual(refreshed_results[0]["context"]["ticker"], "EXM")
            self.assertTrue(refreshed_results[0]["in_error_pool"])

            error_samples = repository.list_error_samples(limit=10)
            self.assertGreaterEqual(len(error_samples), 1)

            retrain_job = repository.create_retrain_job(
                trigger_source="manual_feedback",
                include_feedback_only=True,
                requested_by="qa-user",
                notes="Regression sweep",
            )
            self.assertEqual(retrain_job["status"], "queued")
            self.assertTrue(retrain_job["include_feedback_only"])

            golden = repository.add_golden_test_case(
                input_text="Example Corp warned demand may soften next quarter.",
                expected_label="negative",
                expected_event_type="guidance",
                title="Guidance warning",
                source_name="Reuters",
                context=analysis_result["context"],
            )
            self.assertEqual(golden["expected_label"], "negative")
            self.assertEqual(repository.list_golden_test_cases(limit=10)[0]["source_name"], "Reuters")


if __name__ == "__main__":
    unittest.main()

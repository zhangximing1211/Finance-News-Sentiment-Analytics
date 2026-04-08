from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any


BASE_DIR = Path(__file__).resolve().parents[4]
MODEL_SERVING_SRC = BASE_DIR / "services" / "model-serving" / "src"

if str(MODEL_SERVING_SRC) not in sys.path:
    sys.path.insert(0, str(MODEL_SERVING_SRC))

from model_serving.secondary_explainer import SecondaryExplainer

from .review_queue import ReviewQueueRepository


class ReviewQueueWorker:
    def __init__(
        self,
        repository: ReviewQueueRepository | None = None,
        explainer: SecondaryExplainer | None = None,
    ) -> None:
        self.repository = repository or ReviewQueueRepository()
        self.explainer = explainer or SecondaryExplainer()

    def process_pending(self, limit: int = 20, retry_failed: bool = False) -> dict[str, Any]:
        claimed_items = self.repository.claim_items(limit=limit, retry_failed=retry_failed)
        summary = {
            "claimed": len(claimed_items),
            "completed": 0,
            "failed": 0,
            "external_llm_enabled": self.explainer.external_llm_enabled,
            "items": [],
        }

        for item in claimed_items:
            try:
                analysis_result = json.loads(item["analysis_payload"]) if "analysis_payload" in item else None
                if analysis_result is None:
                    persisted = self.repository.get_item(item["id"])
                    if not persisted:
                        raise RuntimeError("Queue item disappeared before processing.")
                    raise RuntimeError("Stored analysis payload is unavailable to the worker.")

                secondary_explanation = self.explainer.generate(
                    input_text=analysis_result["input_text"],
                    sentiment=analysis_result["sentiment"],
                    event=analysis_result["event"],
                    entities=analysis_result["entities"],
                    risk_alert=analysis_result["risk_alert"],
                    capability_decision={
                        "decision_label": analysis_result["sentiment"]["decision_label"],
                        "top_label": analysis_result["sentiment"]["label"],
                        "confidence": analysis_result["sentiment"]["confidence"],
                        "review_reasons": analysis_result["review_queue_item"]["review_reasons"],
                        "neutral_boundary": "neutral_boundary" in analysis_result["review_queue_item"]["review_reasons"],
                    },
                )

                if not secondary_explanation.get("used_external_llm"):
                    fallback_reason = secondary_explanation.get("fallback_reason") or "external_llm_unavailable"
                    updated = self.repository.mark_failed(item["id"], fallback_reason)
                    summary["failed"] += 1
                    summary["items"].append(updated or {"id": item["id"], "status": "failed"})
                    continue

                updated = self.repository.mark_ready_for_review(item["id"], secondary_explanation)
                summary["completed"] += 1
                summary["items"].append(updated or {"id": item["id"], "status": "ready_for_review"})
            except Exception as exc:
                updated = self.repository.mark_failed(item["id"], str(exc))
                summary["failed"] += 1
                summary["items"].append(updated or {"id": item["id"], "status": "failed"})

        return summary

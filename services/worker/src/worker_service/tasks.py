from __future__ import annotations

from collections import Counter
from typing import Iterable


def collect_review_queue(items: Iterable[dict]) -> list[dict]:
    queue: list[dict] = []
    for item in items:
        if "review_queue_item" in item:
            if item.get("risk_alert", {}).get("needs_human_review"):
                queue.append(item)
            continue
        if item.get("status") in {"pending", "processing", "ready_for_review", "failed"}:
            queue.append(item)
    return queue


def build_daily_digest(items: Iterable[dict]) -> dict[str, object]:
    queue = collect_review_queue(items)
    if queue and "status" in queue[0]:
        status_counter = Counter(item.get("status", "unknown") for item in queue)
        priority_counter = Counter(item.get("priority", "unknown") for item in queue)
        review_reason_counter = Counter(reason for item in queue for reason in item.get("review_reasons", []))
        return {
            "review_queue_size": len(queue),
            "status_breakdown": dict(status_counter),
            "priority_breakdown": dict(priority_counter),
            "review_reason_breakdown": dict(review_reason_counter),
        }

    sentiment_counter = Counter(item.get("sentiment", {}).get("label", "unknown") for item in items)
    event_counter = Counter(item.get("event", {}).get("type", "unknown") for item in items)
    return {
        "total_items": sum(sentiment_counter.values()),
        "review_queue_size": len(queue),
        "sentiment_breakdown": dict(sentiment_counter),
        "event_breakdown": dict(event_counter),
        "top_review_reasons": [
            item.get("risk_alert", {}).get("reasons", []) for item in queue[:10]
        ],
    }

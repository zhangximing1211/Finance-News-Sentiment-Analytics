from __future__ import annotations

from typing import Any


HIGH_SEVERITY_EVENTS = {"earnings", "guidance", "layoffs", "capacity"}


def decide_alert(analysis_result: dict[str, Any], final_decision: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    severity: str | None = None

    event_type = analysis_result["event"]["type"]
    final_label = final_decision["label"]
    final_confidence = float(final_decision["confidence"])
    review_item = analysis_result.get("review_queue_item")
    risk_reasons = list(analysis_result.get("risk_alert", {}).get("reasons", []))

    if final_label == "negative" and event_type in HIGH_SEVERITY_EVENTS and final_confidence >= 0.65:
        severity = "high"
        reasons.append("negative_sensitive_event")
    elif final_label == "negative" and final_confidence >= 0.55:
        severity = "medium"
        reasons.append("negative_sentiment")

    if review_item and review_item.get("priority") == "high":
        severity = "high" if severity != "high" else severity
        reasons.append("high_priority_review_queue")
    elif review_item and severity is None:
        severity = "low"
        reasons.append("manual_review_queue")

    if analysis_result.get("metadata", {}).get("agent_workflow", {}).get("llm_review", {}).get("should_override"):
        severity = "high" if severity == "high" else "medium"
        reasons.append("llm_override")

    reasons.extend(risk_reasons[:3])

    deduped_reasons: list[str] = []
    seen: set[str] = set()
    for reason in reasons:
        if reason in seen:
            continue
        seen.add(reason)
        deduped_reasons.append(reason)

    triggered = severity is not None
    message = (
        f"{analysis_result['entities']['companies'][0] if analysis_result['entities']['companies'] else '未识别主体'} "
        f"{analysis_result['event']['type_zh']} 事件需要{severity}级告警。"
        if triggered
        else "当前无需触发告警。"
    )

    return {
        "triggered": triggered,
        "severity": severity or "none",
        "status": "open" if triggered else "suppressed",
        "reasons": deduped_reasons,
        "message": message,
    }

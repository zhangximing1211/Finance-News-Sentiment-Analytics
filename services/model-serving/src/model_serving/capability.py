from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any


DEFAULT_LOW_CONFIDENCE_THRESHOLD = 0.62
DEFAULT_NEUTRAL_BOUNDARY_MARGIN = 0.08
LOW_CONFIDENCE_THRESHOLD_ENV = "LOW_CONFIDENCE_THRESHOLD_OVERRIDE"
NEUTRAL_BOUNDARY_MARGIN_ENV = "NEUTRAL_BOUNDARY_MARGIN_OVERRIDE"


@dataclass
class CapabilityDecision:
    low_confidence_threshold: float
    neutral_boundary_margin: float

    def decide(self, probabilities: dict[str, float]) -> dict[str, Any]:
        ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
        top_label, top_probability = ranked[0]
        second_label, second_probability = ranked[1]
        confidence_gap = round(top_probability - second_probability, 4)
        neutral_in_top_two = "neutral" in {top_label, second_label}
        neutral_boundary = neutral_in_top_two and confidence_gap <= self.neutral_boundary_margin
        abstained = top_probability < self.low_confidence_threshold

        review_reasons: list[str] = []
        if abstained:
            review_reasons.append("low_confidence")
        if neutral_boundary:
            review_reasons.append("neutral_boundary")

        priority = "high" if abstained and top_probability < (self.low_confidence_threshold - 0.05) else "medium"
        if not review_reasons:
            priority = "low"

        return {
            "decision_label": "abstain" if abstained else top_label,
            "top_label": top_label,
            "confidence": round(top_probability, 4),
            "confidence_gap": confidence_gap,
            "abstained": abstained,
            "neutral_boundary": neutral_boundary,
            "neutral_boundary_margin": self.neutral_boundary_margin,
            "review_reasons": review_reasons,
            "priority": priority,
            "ranked_labels": [
                {"label": label, "probability": round(probability, 4)} for label, probability in ranked
            ],
        }


def _read_env_float(name: str) -> float | None:
    value = os.getenv(name)
    if value is None:
        return None

    parsed = float(value)
    if not 0 <= parsed <= 1:
        raise ValueError(f"{name} must be between 0 and 1.")
    return parsed


def load_capability_policy(
    metadata_path: str | Path | None,
) -> CapabilityDecision:
    low_confidence_threshold = DEFAULT_LOW_CONFIDENCE_THRESHOLD
    neutral_boundary_margin = DEFAULT_NEUTRAL_BOUNDARY_MARGIN
    resolved_metadata_path = Path(metadata_path) if metadata_path else None
    if resolved_metadata_path and resolved_metadata_path.exists():
        metadata = json.loads(resolved_metadata_path.read_text(encoding="utf-8"))
        low_confidence_threshold = float(
            metadata.get("abstain_policy", {}).get("low_confidence_threshold", low_confidence_threshold)
        )
        neutral_boundary_margin = float(
            metadata.get("neutral_boundary", {}).get("margin_threshold", neutral_boundary_margin)
        )

    env_low_confidence_threshold = _read_env_float(LOW_CONFIDENCE_THRESHOLD_ENV)
    env_neutral_boundary_margin = _read_env_float(NEUTRAL_BOUNDARY_MARGIN_ENV)
    if env_low_confidence_threshold is not None:
        low_confidence_threshold = env_low_confidence_threshold
    if env_neutral_boundary_margin is not None:
        neutral_boundary_margin = env_neutral_boundary_margin

    return CapabilityDecision(
        low_confidence_threshold=low_confidence_threshold,
        neutral_boundary_margin=neutral_boundary_margin,
    )


def build_review_queue_item(
    *,
    input_text: str,
    entities: dict[str, Any],
    event: dict[str, Any],
    decision: dict[str, Any],
) -> dict[str, Any] | None:
    if not decision["review_reasons"]:
        return None

    primary_entity = entities["companies"][0] if entities["companies"] else (entities["tickers"][0] if entities["tickers"] else "unknown")
    return {
        "queue": "human_review",
        "priority": decision["priority"],
        "primary_entity": primary_entity,
        "event_type": event["type"],
        "predicted_label": decision["top_label"],
        "decision_label": decision["decision_label"],
        "confidence": decision["confidence"],
        "review_reasons": decision["review_reasons"],
        "recommended_action": "Escalate to analyst review before auto-publishing this classification.",
        "text_excerpt": input_text[:240],
    }

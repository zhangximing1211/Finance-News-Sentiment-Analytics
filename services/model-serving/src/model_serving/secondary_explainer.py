from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import requests

from .env import load_local_env


class SecondaryExplainer:
    def __init__(self, template_path: str | Path | None = None) -> None:
        load_local_env()
        default_template_path = Path(__file__).resolve().parents[4] / "packages" / "prompts" / "templates" / "secondary-explanation.md"
        self.template_path = Path(template_path) if template_path else default_template_path
        self.template = self.template_path.read_text(encoding="utf-8") if self.template_path.exists() else ""
        self.provider = "template_fallback"
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.timeout_seconds = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "30"))
        self.api_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1/responses")
        self.organization = os.getenv("OPENAI_ORGANIZATION")
        self.project = os.getenv("OPENAI_PROJECT")

    @property
    def external_llm_enabled(self) -> bool:
        return bool(self.api_key)

    def generate(
        self,
        *,
        input_text: str,
        sentiment: dict[str, Any],
        event: dict[str, Any],
        entities: dict[str, Any],
        risk_alert: dict[str, Any],
        capability_decision: dict[str, Any],
    ) -> dict[str, Any]:
        if self.external_llm_enabled:
            try:
                return self._generate_with_openai(
                    input_text=input_text,
                    sentiment=sentiment,
                    event=event,
                    entities=entities,
                    risk_alert=risk_alert,
                    capability_decision=capability_decision,
                )
            except Exception as exc:
                return self._build_fallback_payload(
                    input_text=input_text,
                    sentiment=sentiment,
                    event=event,
                    entities=entities,
                    risk_alert=risk_alert,
                    capability_decision=capability_decision,
                    fallback_reason=f"openai_request_failed: {exc}",
                )

        return self._build_fallback_payload(
            input_text=input_text,
            sentiment=sentiment,
            event=event,
            entities=entities,
            risk_alert=risk_alert,
            capability_decision=capability_decision,
            fallback_reason="openai_not_configured",
        )

    def _build_fallback_payload(
        self,
        *,
        input_text: str,
        sentiment: dict[str, Any],
        event: dict[str, Any],
        entities: dict[str, Any],
        risk_alert: dict[str, Any],
        capability_decision: dict[str, Any],
        fallback_reason: str | None,
    ) -> dict[str, Any]:
        reason_labels = {
            "low_confidence": "置信度偏低",
            "neutral_boundary": "neutral 边界样本",
        }
        primary_entity = entities["companies"][0] if entities["companies"] else (entities["tickers"][0] if entities["tickers"] else "未识别主体")
        review_text = (
            f"需要人工复核，原因是 {', '.join(reason_labels.get(reason, reason) for reason in capability_decision['review_reasons'])}。"
            if capability_decision["review_reasons"]
            else "当前可以进入自动初筛，无需立即升级人工。"
        )
        neutral_note = (
            "该样本靠近 neutral 边界，说明文本语义可能同时包含方向不强或混合信号。"
            if capability_decision["neutral_boundary"]
            else "该样本不属于明显的 neutral 边界案例。"
        )

        summary = (
            f"主体 {primary_entity} 对应的事件更接近{event['type_zh']}，"
            f"主判断为 {sentiment['label_zh']}，综合置信度约 {round(sentiment['confidence'] * 100)}%。"
        )
        review_note = f"{review_text} {neutral_note}"
        rationale = (
            f"Primary decision: {capability_decision['decision_label']}. "
            f"Confidence summary: top label {capability_decision['top_label']} at {capability_decision['confidence']}. "
            f"Event context: {event['type']} / {event['type_zh']}. "
            f"Review recommendation: {review_note}"
        )

        return {
            "provider": self.provider,
            "template_path": str(self.template_path),
            "summary": summary,
            "review_note": review_note,
            "rationale": rationale,
            "llm_ready": True,
            "used_external_llm": False,
            "prompt_available": bool(self.template),
            "input_excerpt": input_text[:240],
            "risk_message": risk_alert["message"],
            "fallback_reason": fallback_reason,
        }

    def _generate_with_openai(
        self,
        *,
        input_text: str,
        sentiment: dict[str, Any],
        event: dict[str, Any],
        entities: dict[str, Any],
        risk_alert: dict[str, Any],
        capability_decision: dict[str, Any],
    ) -> dict[str, Any]:
        primary_entity = entities["companies"][0] if entities["companies"] else (entities["tickers"][0] if entities["tickers"] else "未识别主体")
        instructions = (
            "You are a financial news secondary explainer. "
            "Produce a short, analyst-safe second-pass explanation for a queued finance sentiment decision. "
            "Do not change the predicted label, entity, or event type. "
            "If the signal is ambiguous, explain the ambiguity instead of inventing certainty."
        )
        if self.template:
            instructions = f"{instructions}\n\nReference template:\n{self.template}"

        context = {
            "input_text": input_text,
            "primary_entity": primary_entity,
            "sentiment": sentiment,
            "event": event,
            "entities": entities,
            "risk_alert": risk_alert,
            "capability_decision": capability_decision,
        }
        user_prompt = (
            "Generate a JSON object with fields summary, review_note, and rationale. "
            "summary must be concise and business-facing in Chinese. "
            "review_note must state whether the item should stay in analyst review. "
            "rationale must explain why the classifier reached this decision, referencing ambiguity or boundary signals when relevant.\n\n"
            f"{json.dumps(context, ensure_ascii=False, indent=2)}"
        )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.organization:
            headers["OpenAI-Organization"] = self.organization
        if self.project:
            headers["OpenAI-Project"] = self.project

        payload = {
            "model": self.model,
            "instructions": instructions,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": user_prompt,
                        }
                    ],
                }
            ],
            "max_output_tokens": 400,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "secondary_explanation",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "summary": {"type": "string"},
                            "review_note": {"type": "string"},
                            "rationale": {"type": "string"},
                        },
                        "required": ["summary", "review_note", "rationale"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                }
            },
        }

        response = requests.post(
            self.api_url,
            headers=headers,
            json=payload,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()

        if payload.get("error"):
            raise RuntimeError(payload["error"].get("message", "Unknown OpenAI API error"))

        parsed = self._extract_structured_output(payload)
        return {
            "provider": "openai_responses_api",
            "template_path": str(self.template_path),
            "summary": parsed["summary"].strip(),
            "review_note": parsed["review_note"].strip(),
            "rationale": parsed["rationale"].strip(),
            "llm_ready": True,
            "used_external_llm": True,
            "prompt_available": bool(self.template),
            "input_excerpt": input_text[:240],
            "risk_message": risk_alert["message"],
            "fallback_reason": None,
        }

    def _extract_structured_output(self, response_payload: dict[str, Any]) -> dict[str, Any]:
        output_text = response_payload.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return json.loads(output_text)

        for item in response_payload.get("output", []):
            for content in item.get("content", []):
                if content.get("type") == "output_text" and content.get("text", "").strip():
                    return json.loads(content["text"])

        raise RuntimeError("No structured output text returned from OpenAI.")

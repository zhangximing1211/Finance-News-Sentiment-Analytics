from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import requests

from .env import load_local_env


LABELS = {"positive", "neutral", "negative"}


class LLMReviewer:
    def __init__(self, template_path: str | Path | None = None) -> None:
        load_local_env()
        default_template_path = Path(__file__).resolve().parents[4] / "packages" / "prompts" / "templates" / "llm-rejudge.md"
        self.template_path = Path(template_path) if template_path else default_template_path
        self.template = self.template_path.read_text(encoding="utf-8") if self.template_path.exists() else ""
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.timeout_seconds = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "30"))
        self.api_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1/responses")
        self.organization = os.getenv("OPENAI_ORGANIZATION")
        self.project = os.getenv("OPENAI_PROJECT")

    @property
    def external_llm_enabled(self) -> bool:
        return bool(self.api_key)

    def review(
        self,
        *,
        input_text: str,
        sentiment: dict[str, Any],
        event: dict[str, Any],
        entities: dict[str, Any],
        risk_alert: dict[str, Any],
        capability_decision: dict[str, Any],
    ) -> dict[str, Any]:
        if not self.external_llm_enabled:
            return self._fallback(
                sentiment=sentiment,
                fallback_reason="openai_not_configured",
                review_summary="未执行 LLM 复判，沿用模型结果。",
                review_rationale="服务端未配置 OpenAI 凭证，当前保留模型原始判断。",
            )

        try:
            return self._review_with_openai(
                input_text=input_text,
                sentiment=sentiment,
                event=event,
                entities=entities,
                risk_alert=risk_alert,
                capability_decision=capability_decision,
            )
        except Exception as exc:
            return self._fallback(
                sentiment=sentiment,
                fallback_reason=f"openai_request_failed: {exc}",
                review_summary="LLM 复判失败，沿用模型结果。",
                review_rationale=f"外部 LLM 请求失败，错误为：{exc}",
            )

    def skipped(self, *, sentiment: dict[str, Any], reason: str = "confidence_above_threshold") -> dict[str, Any]:
        return {
            "provider": "workflow_skip",
            "used_external_llm": False,
            "triggered": False,
            "should_override": False,
            "reviewed_label": sentiment["label"],
            "reviewed_confidence": round(float(sentiment["confidence"]), 4),
            "review_summary": "未触发 LLM 复判。",
            "review_rationale": "模型置信度高于阈值，本次直接沿用模型结果。",
            "fallback_reason": reason,
        }

    def _fallback(
        self,
        *,
        sentiment: dict[str, Any],
        fallback_reason: str,
        review_summary: str,
        review_rationale: str,
    ) -> dict[str, Any]:
        return {
            "provider": "fallback",
            "used_external_llm": False,
            "triggered": True,
            "should_override": False,
            "reviewed_label": sentiment["label"],
            "reviewed_confidence": round(float(sentiment["confidence"]), 4),
            "review_summary": review_summary,
            "review_rationale": review_rationale,
            "fallback_reason": fallback_reason,
        }

    def _review_with_openai(
        self,
        *,
        input_text: str,
        sentiment: dict[str, Any],
        event: dict[str, Any],
        entities: dict[str, Any],
        risk_alert: dict[str, Any],
        capability_decision: dict[str, Any],
    ) -> dict[str, Any]:
        instructions = (
            "You are a financial-news reviewer. Review a low-confidence model classification. "
            "Decide whether to keep or override the model label. "
            "Prefer neutral when the text is ambiguous. "
            "Output only the requested JSON schema."
        )
        if self.template:
            instructions = f"{instructions}\n\nReference review policy:\n{self.template}"

        context = {
            "input_text": input_text,
            "sentiment": sentiment,
            "event": event,
            "entities": entities,
            "risk_alert": risk_alert,
            "capability_decision": capability_decision,
        }
        user_prompt = (
            "Review this low-confidence finance classification. "
            "Return JSON with reviewed_label, reviewed_confidence, should_override, review_summary, and review_rationale. "
            "review_summary must be concise Chinese for downstream users.\n\n"
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
                    "content": [{"type": "input_text", "text": user_prompt}],
                }
            ],
            "max_output_tokens": 400,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "llm_sentiment_review",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "reviewed_label": {
                                "type": "string",
                                "enum": ["positive", "neutral", "negative"],
                            },
                            "reviewed_confidence": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1,
                            },
                            "should_override": {"type": "boolean"},
                            "review_summary": {"type": "string"},
                            "review_rationale": {"type": "string"},
                        },
                        "required": [
                            "reviewed_label",
                            "reviewed_confidence",
                            "should_override",
                            "review_summary",
                            "review_rationale",
                        ],
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
        response_payload = response.json()

        if response_payload.get("error"):
            raise RuntimeError(response_payload["error"].get("message", "Unknown OpenAI API error"))

        parsed = self._extract_structured_output(response_payload)
        reviewed_label = parsed["reviewed_label"].strip().lower()
        if reviewed_label not in LABELS:
            raise RuntimeError(f"Unsupported reviewed label returned by LLM: {reviewed_label}")

        reviewed_confidence = round(min(max(float(parsed["reviewed_confidence"]), 0.0), 1.0), 4)
        should_override = bool(parsed["should_override"]) and reviewed_label != sentiment["label"]

        return {
            "provider": "openai_responses_api",
            "used_external_llm": True,
            "triggered": True,
            "should_override": should_override,
            "reviewed_label": reviewed_label,
            "reviewed_confidence": reviewed_confidence,
            "review_summary": parsed["review_summary"].strip(),
            "review_rationale": parsed["review_rationale"].strip(),
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

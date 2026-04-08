from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import math
import re
import sys
from typing import Any

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline


BASE_DIR = Path(__file__).resolve().parents[4]
UTILS_DIR = BASE_DIR / "packages" / "utils" / "python"

if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

from finance_utils import build_calibrated_sentiment_pipeline, build_sentiment_pipeline, contains_cjk, dedupe_keep_order, normalize_text
from .capability import build_review_queue_item, load_capability_policy
from .secondary_explainer import SecondaryExplainer


SENTIMENT_LABELS_ZH = {
    "positive": "积极",
    "neutral": "中性",
    "negative": "消极",
}

EVENT_LABELS = {
    "earnings": "财报",
    "acquisition": "收购",
    "layoffs": "裁员",
    "contract": "合同",
    "capacity": "产能",
    "price_change": "价格变动",
    "guidance": "指引更新",
    "unknown": "未识别",
}

INDUSTRY_LABELS = {
    "technology": "科技",
    "finance": "金融",
    "healthcare": "医疗健康",
    "energy": "能源",
    "consumer": "消费",
    "industrial": "工业制造",
    "real_estate": "房地产",
    "transportation": "交通物流",
    "materials": "原材料",
    "unknown": "未识别",
}

REVIEW_REASON_LABELS = {
    "low_confidence": "置信度偏低",
    "neutral_boundary": "neutral 边界样本",
}


@dataclass(frozen=True)
class WeightedPattern:
    regex: re.Pattern[str]
    weight: float
    description: str


def _compile_patterns(definitions: list[tuple[str, float, str]]) -> list[WeightedPattern]:
    return [
        WeightedPattern(re.compile(pattern, re.IGNORECASE), weight, description)
        for pattern, weight, description in definitions
    ]


POSITIVE_PATTERNS = _compile_patterns(
    [
        (r"\b(beat|beats|beating)\b", 1.6, "业绩超预期"),
        (r"\b(record|strong|robust)\b", 0.9, "经营表现强劲"),
        (r"\b(surge[sd]?|jumped|soared)\b", 1.8, "快速增长"),
        (r"\b(rose|rises|rising|grew|growth|increased|higher)\b", 1.2, "增长信号"),
        (r"\b(profitability|profitable|profit rose|net profit)\b", 0.9, "盈利改善"),
        (r"\b(win|won|sign(?:ed)?|agreement|contract|order book)\b", 1.0, "签约或拿单"),
        (r"\b(raised guidance|raise outlook|upgraded|dividend)\b", 1.6, "指引或股东回报改善"),
        (r"\b(expand(?:ed)? capacity|ramp(?:ing)? up|new plant)\b", 1.3, "扩产信号"),
        (r"超预期|增长|上调|签署|中标|盈利改善|扭亏为盈|扩产|增持|回购", 1.5, "中文积极信号"),
    ]
)

NEGATIVE_PATTERNS = _compile_patterns(
    [
        (r"\b(miss(?:ed)?|below expectations)\b", 1.8, "业绩不及预期"),
        (r"\b(fell|falling|dropped|slumped|declined|lower)\b", 1.3, "下滑信号"),
        (r"\b(loss|losses|warning|warned|weakness)\b", 1.4, "盈利或经营承压"),
        (r"\b(layoffs?|laid off|lay off|job cuts|workforce reductions?|restructuring)\b", 2.0, "裁员或重组"),
        (r"\b(cut guidance|lowered outlook|downgraded)\b", 1.8, "指引下修"),
        (r"\b(delay(?:ed)?|probe|investigation|lawsuit|default)\b", 1.2, "经营风险或监管风险"),
        (r"\b(shutdown|halted production|capacity cut)\b", 1.4, "停产或减产"),
        (r"裁员|亏损|下滑|下降|下调|减产|停产|违约|处罚|调查|诉讼|不及预期|暴跌", 1.6, "中文消极信号"),
    ]
)

UNCERTAINTY_PATTERNS = _compile_patterns(
    [
        (r"\b(may|might|could|reportedly|rumor|considering)\b", 0.8, "表述存在不确定性"),
        (r"可能|或将|拟|传闻|据悉|预计|计划", 0.8, "中文前瞻或未确认表述"),
    ]
)

EVENT_PATTERNS: dict[str, list[WeightedPattern]] = {
    "earnings": _compile_patterns(
        [
            (r"\b(earnings|results|revenue|sales|ebit|ebitda|eps|net income|net profit)\b", 1.0, "财报指标"),
            (r"\b(q[1-4]|quarter|annual results|guidance)\b", 0.7, "财报语境"),
            (r"财报|业绩|营收|净利润|每股收益|季报|年报", 1.1, "中文财报表述"),
        ]
    ),
    "acquisition": _compile_patterns(
        [
            (r"\b(acquire|acquired|acquisition|merge[rd]?|merger|takeover|buyout)\b", 1.3, "收购并购"),
            (r"\b(purchase agreement|offer for)\b", 0.9, "交易安排"),
            (r"收购|并购|合并|要约收购|买下", 1.4, "中文收购表述"),
        ]
    ),
    "layoffs": _compile_patterns(
        [
            (r"\b(layoffs?|laid off|lay off|job cuts|restructuring|headcount reductions?)\b", 1.5, "裁员重组"),
            (r"\b(cost cutting|streamlining)\b", 0.8, "降本动作"),
            (r"裁员|减员|人员优化|组织调整|缩编", 1.6, "中文裁员表述"),
        ]
    ),
    "contract": _compile_patterns(
        [
            (r"\b(contract|agreement|deal|order|supply agreement|signed)\b", 1.2, "合同订单"),
            (r"\b(customer win|purchase agreement)\b", 1.0, "新增客户或采购"),
            (r"合同|协议|订单|签署|中标|供货", 1.4, "中文合同表述"),
        ]
    ),
    "capacity": _compile_patterns(
        [
            (r"\b(capacity|plant|factory|production line|ramp up|expand production)\b", 1.2, "产能设施"),
            (r"\b(shutdown|halted production|new production plant)\b", 1.0, "产线变化"),
            (r"产能|工厂|产线|投产|扩产|停产|复产", 1.5, "中文产能表述"),
        ]
    ),
    "price_change": _compile_patterns(
        [
            (r"\b(price hike|prices rose|prices fell|raise prices|cut prices|pricing)\b", 1.2, "价格调整"),
            (r"\b(stock fell|share price|shares rose)\b", 0.9, "股价变化"),
            (r"提价|降价|价格上涨|价格下调|股价|跌停|涨停", 1.4, "中文价格表述"),
        ]
    ),
    "guidance": _compile_patterns(
        [
            (r"\b(guidance|outlook|forecast|expects|sees|target)\b", 1.3, "指引展望"),
            (r"\b(raises outlook|cuts outlook|target for)\b", 1.0, "目标更新"),
            (r"指引|展望|预期|目标|预计|预测", 1.4, "中文指引表述"),
        ]
    ),
}

INDUSTRY_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "technology": [
        re.compile(r"\b(software|semiconductor|chip|cloud|ai|cybersecurity|telecom)\b", re.IGNORECASE),
        re.compile(r"科技|芯片|半导体|软件|云计算|人工智能|通信"),
    ],
    "finance": [
        re.compile(r"\b(bank|insurance|asset management|brokerage|fintech)\b", re.IGNORECASE),
        re.compile(r"银行|保险|证券|资管|金融科技"),
    ],
    "healthcare": [
        re.compile(r"\b(pharma|biotech|drug|hospital|medical device)\b", re.IGNORECASE),
        re.compile(r"医药|医疗|生物科技|器械|医院"),
    ],
    "energy": [
        re.compile(r"\b(oil|gas|power|renewable|solar|battery|utility)\b", re.IGNORECASE),
        re.compile(r"石油|天然气|电力|新能源|光伏|电池"),
    ],
    "consumer": [
        re.compile(r"\b(retail|consumer|e-commerce|food|beverage|apparel)\b", re.IGNORECASE),
        re.compile(r"消费|零售|电商|食品|饮料|服饰"),
    ],
    "industrial": [
        re.compile(r"\b(manufacturing|industrial|machinery|construction|automation)\b", re.IGNORECASE),
        re.compile(r"制造|工业|机械|建筑|自动化"),
    ],
    "real_estate": [
        re.compile(r"\b(real estate|property|developer|reit)\b", re.IGNORECASE),
        re.compile(r"房地产|物业|开发商|REIT"),
    ],
    "transportation": [
        re.compile(r"\b(logistics|shipping|airline|rail|transport)\b", re.IGNORECASE),
        re.compile(r"物流|航运|航空|铁路|运输"),
    ],
    "materials": [
        re.compile(r"\b(mining|steel|chemical|paper|forestry|metals)\b", re.IGNORECASE),
        re.compile(r"矿业|钢铁|化工|纸业|林业|金属"),
    ],
}

ENGLISH_COMPANY_PATTERN = re.compile(
    r"\b([A-Z][A-Za-z&.\-']+(?:\s+[A-Z][A-Za-z&.\-']+){0,3}\s+"
    r"(?:Inc\.?|Corp\.?|Corporation|Ltd\.?|Limited|PLC|Group|Holdings|Technologies|Bank|Energy|Systems|Motors|Company))\b"
)
ENGLISH_COMPANY_WITH_TICKER_PATTERN = re.compile(
    r"\b([A-Z][A-Za-z&.\-']+(?:\s+[A-Z][A-Za-z&.\-']+){0,3})\s*"
    r"\((?:NASDAQ|NYSE|HKEX|SSE|SZSE|TSX|LSE|TSE|BSE|NSE)\s*[:：]\s*[A-Z0-9.]+\)",
    re.IGNORECASE,
)
CHINESE_COMPANY_PATTERN = re.compile(
    r"([\u4e00-\u9fffA-Za-z0-9]{2,30}(?:股份有限公司|有限公司|集团|控股|科技|银行|证券|药业|汽车|能源|电子|通信|实业))"
)
TICKER_PATTERNS = [
    re.compile(r"\$([A-Z]{1,6}(?:\.[A-Z])?)"),
    re.compile(r"(?:NASDAQ|NYSE|HKEX|SSE|SZSE|TSX|LSE|TSE|BSE|NSE)\s*[:：]\s*([A-Z0-9.]{1,8})", re.IGNORECASE),
    re.compile(r"\bticker(?:\s+symbol)?\s*(?:is|:)\s*([A-Z]{1,6}(?:\.[A-Z])?)", re.IGNORECASE),
]


def _normalize_probabilities(scores: dict[str, float]) -> dict[str, float]:
    total = sum(max(value, 0.0001) for value in scores.values())
    return {label: max(value, 0.0001) / total for label, value in scores.items()}


def _clean_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    normalized = normalize_text(str(value))
    return normalized or None


class FinanceNewsAnalyzer:
    def __init__(self, data_path: str | Path, model_path: str | Path | None = None) -> None:
        self.data_path = Path(data_path)
        self.model_path = Path(model_path) if model_path else BASE_DIR / "data" / "processed" / "baseline_models" / "best_baseline.joblib"
        self.metadata_path = self.model_path.with_name("best_baseline_metadata.json")
        self.pipeline: Pipeline | None = None
        self.model_classes: list[str] = ["negative", "neutral", "positive"]
        self.training_error: str | None = None
        self.model_source = "uninitialized"
        self.model_metadata: dict[str, Any] = {}
        self.capability_policy = load_capability_policy(self.metadata_path)
        self.secondary_explainer = SecondaryExplainer()
        self._load_or_train_sentiment_model()

    def _load_or_train_sentiment_model(self) -> None:
        if self.metadata_path.exists():
            try:
                self.model_metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
                self.capability_policy = load_capability_policy(self.metadata_path)
            except Exception as exc:  # pragma: no cover - runtime safeguard
                self.training_error = f"Failed to load model metadata: {exc}"

        if self.model_path.exists():
            try:
                loaded_pipeline = joblib.load(self.model_path)
                classifier = loaded_pipeline.named_steps["classifier"]
                self.model_classes = list(classifier.classes_)
                self.pipeline = loaded_pipeline
                self.model_source = "artifact"
                return
            except Exception as exc:  # pragma: no cover - runtime safeguard
                self.training_error = f"Failed to load model artifact: {exc}"

        self._train_sentiment_model()

    def _train_sentiment_model(self) -> None:
        try:
            dataset = pd.read_csv(
                self.data_path,
                names=["sentiment", "text"],
                header=None,
                encoding="ISO-8859-1",
            ).dropna()
            dataset["text"] = dataset["text"].astype(str).map(normalize_text)
            dataset = dataset[dataset["text"] != ""]

            pipeline = build_calibrated_sentiment_pipeline()
            pipeline.fit(dataset["text"], dataset["sentiment"])
            self.model_classes = list(getattr(pipeline, "classes_", self.model_classes))
            self.pipeline = pipeline
            self.model_source = "runtime_training"
        except Exception as exc:  # pragma: no cover - runtime safeguard
            self.training_error = str(exc)
            self.pipeline = None
            self.model_source = "unavailable"

    def analyze(self, text: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        normalized_text = normalize_text(text)
        if not normalized_text:
            raise ValueError("Input text cannot be empty.")

        normalized_context = self._normalize_context(context)
        entities = self._extract_entities(normalized_text)
        event = self._classify_event(normalized_text)
        context_enrichment = self._apply_context_overrides(
            entities=entities,
            event=event,
            context=normalized_context,
        )
        rule_signal = self._score_rule_sentiment(normalized_text, event["type"])
        ml_probabilities = self._predict_sentiment_with_model(normalized_text)
        probabilities = self._blend_sentiment_probabilities(
            ml_probabilities=ml_probabilities,
            rule_probabilities=rule_signal["probabilities"],
            has_cjk=contains_cjk(normalized_text),
        )
        capability_decision = self.capability_policy.decide(probabilities)
        signal_confidence = self._calculate_confidence(
            probabilities=probabilities,
            positive_hits=rule_signal["positive_hits"],
            negative_hits=rule_signal["negative_hits"],
            uncertainty_hits=rule_signal["uncertainty_hits"],
        )

        sorted_probabilities = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
        sentiment_label = sorted_probabilities[0][0]
        confidence = capability_decision["confidence"]
        explanation = self._build_explanation(
            sentiment_label=sentiment_label,
            confidence=confidence,
            event=event,
            entities=entities,
            rule_signal=rule_signal,
            context=normalized_context,
        )
        risk_alert = self._build_risk_alert(
            text=normalized_text,
            confidence=confidence,
            event=event,
            entities=entities,
            rule_signal=rule_signal,
            sentiment_label=sentiment_label,
            low_confidence_threshold=self.capability_policy.low_confidence_threshold,
        )
        review_queue_item = build_review_queue_item(
            input_text=normalized_text,
            entities=entities,
            event=event,
            decision=capability_decision,
        )
        if review_queue_item:
            risk_alert["needs_human_review"] = True
            translated_reasons = [REVIEW_REASON_LABELS.get(reason, reason) for reason in capability_decision["review_reasons"]]
            risk_alert["reasons"] = dedupe_keep_order(risk_alert["reasons"] + translated_reasons)
            risk_alert["message"] = "建议人工复核：" + "；".join(risk_alert["reasons"]) + "。"

        secondary_explanation = self.secondary_explainer.generate(
            input_text=normalized_text,
            sentiment={
                "label": sentiment_label,
                "label_zh": SENTIMENT_LABELS_ZH[sentiment_label],
                "confidence": confidence,
            },
            event=event,
            entities=entities,
            risk_alert=risk_alert,
            capability_decision=capability_decision,
        )

        return {
            "input_text": normalized_text,
            "context": normalized_context,
            "sentiment": {
                "label": sentiment_label,
                "label_zh": SENTIMENT_LABELS_ZH[sentiment_label],
                "confidence": confidence,
                "probabilities": {label: round(score, 4) for label, score in probabilities.items()},
                "decision_label": capability_decision["decision_label"],
                "abstained": capability_decision["abstained"],
                "confidence_gap": capability_decision["confidence_gap"],
                "low_confidence_threshold": self.capability_policy.low_confidence_threshold,
            },
            "event": {
                "type": event["type"],
                "type_zh": event["type_zh"],
                "matched_signals": event["matched_signals"],
                "secondary_type": event["secondary_type"],
            },
            "entities": entities,
            "explanation": explanation,
            "risk_alert": risk_alert,
            "review_queue_item": review_queue_item,
            "secondary_explanation": secondary_explanation,
            "metadata": {
                "used_ml_model": bool(ml_probabilities),
                "used_rule_engine": True,
                "training_error": self.training_error,
                "model_source": self.model_source,
                "model_path": str(self.model_path) if self.model_source == "artifact" else None,
                "model_metadata_path": str(self.metadata_path) if self.metadata_path.exists() else None,
                "capability_module": {
                    "probability_source": "calibrated_ml_plus_rules" if ml_probabilities else "rules_only",
                    "ml_probabilities": {label: round(score, 4) for label, score in ml_probabilities.items()},
                    "rule_probabilities": {label: round(score, 4) for label, score in rule_signal["probabilities"].items()},
                    "final_probabilities": {label: round(score, 4) for label, score in probabilities.items()},
                    "signal_confidence_estimate": round(signal_confidence, 4),
                    "low_confidence_threshold": self.capability_policy.low_confidence_threshold,
                    "neutral_boundary_margin": self.capability_policy.neutral_boundary_margin,
                    "review_reasons": capability_decision["review_reasons"],
                    "ranked_labels": capability_decision["ranked_labels"],
                },
                "context_enrichment": context_enrichment,
            },
        }

    def _normalize_context(self, context: dict[str, Any] | None) -> dict[str, Any]:
        if not context:
            return {
                "news_source": None,
                "source_name": None,
                "source_url": None,
                "published_at": None,
                "company_name": None,
                "ticker": None,
                "industry": None,
                "event_type": None,
                "historical_announcements": [],
            }

        historical_announcements: list[dict[str, Any]] = []
        for item in context.get("historical_announcements", []) or []:
            if not isinstance(item, dict):
                continue
            summary = _clean_optional_text(item.get("summary"))
            if not summary:
                continue
            historical_announcements.append(
                {
                    "title": _clean_optional_text(item.get("title")),
                    "summary": summary,
                    "event_type": self._normalize_event_type_override(item.get("event_type")) or _clean_optional_text(item.get("event_type")),
                    "announced_at": _clean_optional_text(item.get("announced_at")),
                    "source_name": _clean_optional_text(item.get("source_name")),
                }
            )

        return {
            "news_source": _clean_optional_text(context.get("news_source")),
            "source_name": _clean_optional_text(context.get("source_name")),
            "source_url": _clean_optional_text(context.get("source_url")),
            "published_at": _clean_optional_text(context.get("published_at")),
            "company_name": _clean_optional_text(context.get("company_name")),
            "ticker": (_clean_optional_text(context.get("ticker")) or "").upper() or None,
            "industry": _clean_optional_text(context.get("industry")),
            "event_type": self._normalize_event_type_override(context.get("event_type")) or _clean_optional_text(context.get("event_type")),
            "historical_announcements": historical_announcements[:10],
        }

    def _normalize_event_type_override(self, value: Any) -> str | None:
        normalized = _clean_optional_text(value)
        if not normalized:
            return None
        lookup = {
            "earnings": "earnings",
            "财报": "earnings",
            "acquisition": "acquisition",
            "收购": "acquisition",
            "layoffs": "layoffs",
            "裁员": "layoffs",
            "contract": "contract",
            "合同": "contract",
            "capacity": "capacity",
            "产能": "capacity",
            "price_change": "price_change",
            "price change": "price_change",
            "价格变动": "price_change",
            "guidance": "guidance",
            "指引更新": "guidance",
            "unknown": "unknown",
            "未识别": "unknown",
        }
        return lookup.get(normalized.casefold())

    def _normalize_industry_override(self, value: Any) -> str | None:
        normalized = _clean_optional_text(value)
        if not normalized:
            return None
        lookup = {key.casefold(): key for key in INDUSTRY_LABELS}
        lookup.update({label.casefold(): key for key, label in INDUSTRY_LABELS.items()})
        return lookup.get(normalized.casefold())

    def _apply_context_overrides(
        self,
        *,
        entities: dict[str, Any],
        event: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, bool]:
        enrichment = {
            "company_override": False,
            "ticker_override": False,
            "industry_override": False,
            "event_override": False,
            "history_attached": bool(context["historical_announcements"]),
        }

        company_name = context.get("company_name")
        if company_name:
            entities["companies"] = dedupe_keep_order([company_name] + entities["companies"])[:5]
            enrichment["company_override"] = True

        ticker = context.get("ticker")
        if ticker:
            entities["tickers"] = dedupe_keep_order([ticker] + entities["tickers"])[:5]
            enrichment["ticker_override"] = True

        industry_key = self._normalize_industry_override(context.get("industry"))
        if industry_key:
            entities["industry"] = industry_key
            entities["industry_zh"] = INDUSTRY_LABELS[industry_key]
            enrichment["industry_override"] = True

        event_override = self._normalize_event_type_override(context.get("event_type"))
        if event_override:
            event["type"] = event_override
            event["type_zh"] = EVENT_LABELS[event_override]
            event["matched_signals"] = dedupe_keep_order(["provided_context"] + event["matched_signals"])[:4]
            enrichment["event_override"] = True

        return enrichment

    def _predict_sentiment_with_model(self, text: str) -> dict[str, float]:
        if not self.pipeline or contains_cjk(text):
            return {}

        if hasattr(self.pipeline, "predict_proba"):
            probabilities = self.pipeline.predict_proba([text])[0]
            return {
                label: float(probability)
                for label, probability in zip(self.model_classes, probabilities)
            }

        if hasattr(self.pipeline, "decision_function"):
            raw_scores = self.pipeline.decision_function([text])[0]
            if not hasattr(raw_scores, "__len__"):
                raw_scores = [raw_scores]

            max_score = max(raw_scores)
            exps = [math.exp(score - max_score) for score in raw_scores]
            total = sum(exps) or 1.0
            probabilities = [score / total for score in exps]
            return {
                label: float(probability)
                for label, probability in zip(self.model_classes, probabilities)
            }

        return {}

    def _collect_matches(self, text: str, patterns: list[WeightedPattern]) -> tuple[float, list[str]]:
        score = 0.0
        hits: list[str] = []
        for pattern in patterns:
            if pattern.regex.search(text):
                score += pattern.weight
                hits.append(pattern.description)
        return score, hits

    def _score_rule_sentiment(self, text: str, event_type: str) -> dict[str, Any]:
        positive_score, positive_hits = self._collect_matches(text, POSITIVE_PATTERNS)
        negative_score, negative_hits = self._collect_matches(text, NEGATIVE_PATTERNS)
        uncertainty_score, uncertainty_hits = self._collect_matches(text, UNCERTAINTY_PATTERNS)

        base_scores = {
            "positive": 1.0 + positive_score,
            "neutral": 1.0,
            "negative": 1.0 + negative_score,
        }

        if event_type == "layoffs":
            base_scores["negative"] += 0.8
        elif event_type == "contract":
            base_scores["positive"] += 0.5
        elif event_type == "capacity" and ("停产" in text or "shutdown" in text.lower()):
            base_scores["negative"] += 0.6
        elif event_type == "guidance":
            if re.search(r"raise|up|上调|提高", text, re.IGNORECASE):
                base_scores["positive"] += 0.7
            if re.search(r"cut|lower|下调|降低", text, re.IGNORECASE):
                base_scores["negative"] += 0.7

        if positive_hits and negative_hits:
            base_scores["neutral"] += 0.8
        elif not positive_hits and not negative_hits:
            base_scores["neutral"] += 1.2

        if uncertainty_hits:
            base_scores["neutral"] += min(uncertainty_score, 1.2)

        probabilities = _normalize_probabilities(base_scores)
        return {
            "probabilities": probabilities,
            "positive_hits": positive_hits,
            "negative_hits": negative_hits,
            "uncertainty_hits": uncertainty_hits,
        }

    def _blend_sentiment_probabilities(
        self,
        ml_probabilities: dict[str, float],
        rule_probabilities: dict[str, float],
        has_cjk: bool,
    ) -> dict[str, float]:
        if not ml_probabilities:
            return rule_probabilities

        ml_confidence = max(ml_probabilities.values())
        ml_weight = 0.72 if not has_cjk else 0.0
        if ml_confidence < 0.58:
            ml_weight = 0.55
        rule_weight = 1.0 - ml_weight

        blended = {
            label: (ml_probabilities.get(label, 0.0) * ml_weight)
            + (rule_probabilities.get(label, 0.0) * rule_weight)
            for label in {"positive", "neutral", "negative"}
        }
        return _normalize_probabilities(blended)

    def _classify_event(self, text: str) -> dict[str, Any]:
        ranked_events: list[tuple[str, float, list[str]]] = []
        for event_type, patterns in EVENT_PATTERNS.items():
            score, hits = self._collect_matches(text, patterns)
            ranked_events.append((event_type, score, hits))

        ranked_events.sort(key=lambda item: item[1], reverse=True)
        top_event, top_score, top_hits = ranked_events[0]
        second_event = ranked_events[1][0] if len(ranked_events) > 1 else "unknown"
        second_score = ranked_events[1][1] if len(ranked_events) > 1 else 0.0

        score_map = {event_type: score for event_type, score, _ in ranked_events}
        hit_map = {event_type: hits for event_type, _, hits in ranked_events}

        if (
            top_event == "earnings"
            and score_map.get("guidance", 0.0) >= 1.3
            and (top_score - score_map["guidance"]) <= 0.8
        ):
            second_event = top_event
            second_score = top_score
            top_event = "guidance"
            top_score = score_map["guidance"]
            top_hits = hit_map["guidance"]

        if top_score <= 0:
            top_event = "unknown"
            top_hits = []

        return {
            "type": top_event,
            "type_zh": EVENT_LABELS[top_event],
            "matched_signals": top_hits[:4],
            "score": round(top_score, 3),
            "secondary_type": second_event,
            "secondary_score": round(second_score, 3),
        }

    def _extract_entities(self, text: str) -> dict[str, Any]:
        companies: list[str] = []
        companies.extend(match.group(1) for match in ENGLISH_COMPANY_WITH_TICKER_PATTERN.finditer(text))
        companies.extend(match.group(1) for match in ENGLISH_COMPANY_PATTERN.finditer(text))
        companies.extend(match.group(1) for match in CHINESE_COMPANY_PATTERN.finditer(text))

        tickers: list[str] = []
        for pattern in TICKER_PATTERNS:
            tickers.extend(match.group(1).upper() for match in pattern.finditer(text))

        industry_key = "unknown"
        best_score = 0
        for key, patterns in INDUSTRY_PATTERNS.items():
            score = sum(1 for pattern in patterns if pattern.search(text))
            if score > best_score:
                best_score = score
                industry_key = key

        return {
            "companies": dedupe_keep_order(companies)[:5],
            "tickers": dedupe_keep_order(tickers)[:5],
            "industry": industry_key,
            "industry_zh": INDUSTRY_LABELS[industry_key],
        }

    def _calculate_confidence(
        self,
        probabilities: dict[str, float],
        positive_hits: list[str],
        negative_hits: list[str],
        uncertainty_hits: list[str],
    ) -> float:
        ordered = sorted(probabilities.values(), reverse=True)
        top_probability = ordered[0]
        second_probability = ordered[1]
        margin = top_probability - second_probability
        confidence = 0.22 + (top_probability * 0.60) + (margin * 0.35)

        if positive_hits or negative_hits:
            confidence += 0.08
        else:
            confidence += 0.03
        if positive_hits and negative_hits:
            confidence -= 0.07
        else:
            confidence += 0.05
        if uncertainty_hits:
            confidence -= min(0.08, 0.03 * len(uncertainty_hits))

        return round(max(0.35, min(confidence, 0.97)), 3)

    def _build_explanation(
        self,
        sentiment_label: str,
        confidence: float,
        event: dict[str, Any],
        entities: dict[str, Any],
        rule_signal: dict[str, Any],
        context: dict[str, Any],
    ) -> str:
        direction = SENTIMENT_LABELS_ZH[sentiment_label]
        signal_hits = rule_signal["positive_hits"] if sentiment_label == "positive" else rule_signal["negative_hits"]
        signal_text = "、".join(signal_hits[:3]) if signal_hits else "事件语义和整体措辞"
        company_text = ""
        if entities["companies"]:
            company_text = f"，并识别到公司 {', '.join(entities['companies'][:2])}"
        elif entities["tickers"]:
            company_text = f"，并识别到 ticker {', '.join(entities['tickers'][:2])}"
        context_text = ""
        if context["historical_announcements"]:
            context_text += f"，并参考了 {len(context['historical_announcements'])} 条历史公告"
        if context["source_name"]:
            context_text += f"，文章来源为 {context['source_name']}"

        if event["type"] == "unknown":
            return (
                f"判断为{direction}，置信度 {confidence}；文本未形成非常明确的单一事件类别，"
                f"但从 {signal_text} 可以看出当前语气更偏{direction}{company_text}{context_text}。"
            )

        return (
            f"判断为{direction}，置信度 {confidence}；文本包含{event['type_zh']}相关表述，"
            f"且出现 {signal_text} 等信号{company_text}{context_text}。"
        )

    def _build_risk_alert(
        self,
        text: str,
        confidence: float,
        event: dict[str, Any],
        entities: dict[str, Any],
        rule_signal: dict[str, Any],
        sentiment_label: str,
        low_confidence_threshold: float,
    ) -> dict[str, Any]:
        reasons: list[str] = []
        paired_event_combo = {event["type"], event["secondary_type"]}

        if confidence < low_confidence_threshold:
            reasons.append("置信度偏低")
        if rule_signal["positive_hits"] and rule_signal["negative_hits"]:
            reasons.append("文本同时包含正负面信号")
        if rule_signal["uncertainty_hits"]:
            reasons.append("存在前瞻性或未确认表述")
        if (
            event["secondary_score"] >= max(event["score"] * 0.82, 1.0)
            and paired_event_combo != {"earnings", "guidance"}
        ):
            reasons.append("可能包含多个事件主题")
        if not entities["companies"] and not entities["tickers"]:
            reasons.append("未识别到明确主体")
        if len(text) < 40:
            reasons.append("输入文本较短")
        if sentiment_label == "neutral" and event["type"] == "unknown":
            reasons.append("关键信号不足")

        needs_human_review = bool(reasons)
        message = (
            "建议人工复核：" + "；".join(reasons) + "。"
            if reasons
            else "暂不强制要求人工复核，当前结果可用于自动初筛。"
        )
        return {
            "needs_human_review": needs_human_review,
            "message": message,
            "reasons": reasons,
        }

from .capability import CapabilityDecision, build_review_queue_item, load_capability_policy
from .llm_reviewer import LLMReviewer
from .secondary_explainer import SecondaryExplainer

__all__ = [
    "AnalysisService",
    "CapabilityDecision",
    "FinanceNewsAnalyzer",
    "LLMReviewer",
    "SecondaryExplainer",
    "build_review_queue_item",
    "get_analysis_service",
    "load_capability_policy",
]


def __getattr__(name: str):
    if name == "FinanceNewsAnalyzer":
        from .analyzer import FinanceNewsAnalyzer

        return FinanceNewsAnalyzer
    if name in {"AnalysisService", "get_analysis_service"}:
        from .service import AnalysisService, get_analysis_service

        return {
            "AnalysisService": AnalysisService,
            "get_analysis_service": get_analysis_service,
        }[name]
    raise AttributeError(f"module 'model_serving' has no attribute {name!r}")

from __future__ import annotations

from pathlib import Path

from .analyzer import FinanceNewsAnalyzer


class AnalysisService:
    def __init__(self, data_path: str | Path | None = None) -> None:
        base_dir = Path(__file__).resolve().parents[4]
        resolved_data_path = Path(data_path) if data_path else base_dir / "data" / "raw" / "all-data.csv"
        self.analyzer = FinanceNewsAnalyzer(resolved_data_path)

    def analyze_text(self, text: str, context: dict | None = None) -> dict:
        return self.analyzer.analyze(text, context=context)

    def health(self) -> dict[str, object]:
        return {
            "status": "ok",
            "model_ready": self.analyzer.pipeline is not None or self.analyzer._bert_model is not None,
            "model_source": self.analyzer.model_source,
            "training_error": self.analyzer.training_error,
        }


_SERVICE: AnalysisService | None = None


def get_analysis_service() -> AnalysisService:
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = AnalysisService()
    return _SERVICE

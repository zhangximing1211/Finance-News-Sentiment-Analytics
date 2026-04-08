from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def build_tfidf_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        strip_accents="unicode",
    )


def build_logistic_regression_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("vectorizer", build_tfidf_vectorizer()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=500,
                    class_weight="balanced",
                    solver="lbfgs",
                ),
            ),
        ]
    )


def build_linear_svm_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("vectorizer", build_tfidf_vectorizer()),
            (
                "classifier",
                LinearSVC(
                    C=1.0,
                    class_weight="balanced",
                ),
            ),
        ]
    )


MODEL_BUILDERS = {
    "tfidf_logistic_regression": build_logistic_regression_pipeline,
    "tfidf_linear_svm": build_linear_svm_pipeline,
}


def build_sentiment_pipeline(model_name: str = "tfidf_logistic_regression") -> Pipeline:
    if model_name not in MODEL_BUILDERS:
        raise ValueError(f"Unsupported model_name: {model_name}")
    return MODEL_BUILDERS[model_name]()


def build_calibrated_sentiment_pipeline(
    model_name: str = "tfidf_logistic_regression",
    calibration_method: str = "sigmoid",
    cv: int = 5,
) -> CalibratedClassifierCV:
    return CalibratedClassifierCV(
        estimator=build_sentiment_pipeline(model_name),
        method=calibration_method,
        cv=cv,
    )

from dataclasses import dataclass
from math import isfinite
import os
from pathlib import Path
from typing import Mapping

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

FEATURE_COLUMNS = (
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
)

CATEGORICAL_VALUES = {
    "sex": {0, 1},
    "cp": {1, 2, 3, 4},
    "fbs": {0, 1},
    "restecg": {0, 1, 2},
    "exang": {0, 1},
    "slope": {1, 2, 3},
    "ca": {0, 1, 2, 3},
    "thal": {3, 6, 7},
}

DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "heart_disease_final_model_v1.pkl"
)
MODEL_PATH = Path(os.environ.get("MEDICARE_MODEL_PATH", str(DEFAULT_MODEL_PATH)))

FeatureValues = Mapping[str, int | float | None]


@dataclass(frozen=True)
class Prediction:
    flagged: bool
    positive_probability: float | None


def load_model(model_path: Path = MODEL_PATH) -> Pipeline:
    """Load the repository's trusted, version-pinned scikit-learn artifact."""
    if not model_path.is_file():
        raise FileNotFoundError(f"Trained model not found at {model_path}")

    model = joblib.load(model_path)
    model_features = tuple(getattr(model, "feature_names_in_", ()))
    if model_features != FEATURE_COLUMNS:
        raise ValueError(
            "The trained model feature schema does not match the application inputs."
        )
    if set(model.classes_) != {0, 1}:
        raise ValueError("The trained model must contain both binary target classes.")
    return model


def build_input_frame(features: FeatureValues) -> pd.DataFrame:
    supplied_features = set(features)
    expected_features = set(FEATURE_COLUMNS)
    missing = expected_features - supplied_features
    unexpected = supplied_features - expected_features
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing: {', '.join(sorted(missing))}")
        if unexpected:
            details.append(f"unexpected: {', '.join(sorted(unexpected))}")
        raise ValueError(f"Invalid feature fields ({'; '.join(details)}).")

    row = {}
    for feature in FEATURE_COLUMNS:
        value = features[feature]
        if value is None:
            if feature not in {"ca", "thal"}:
                raise ValueError(f"{feature} is required.")
            row[feature] = float("nan")
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{feature} must be numeric.")
        if not isfinite(value):
            raise ValueError(f"{feature} must be finite.")
        if feature in CATEGORICAL_VALUES and value not in CATEGORICAL_VALUES[feature]:
            choices = ", ".join(map(str, sorted(CATEGORICAL_VALUES[feature])))
            raise ValueError(f"{feature} must be one of: {choices}.")
        row[feature] = value

    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def predict_risk(model: Pipeline, features: FeatureValues) -> Prediction:
    input_frame = build_input_frame(features)
    predicted_class = int(model.predict(input_frame)[0])

    positive_probability = None
    if hasattr(model, "predict_proba"):
        classes = list(model.classes_)
        if 1 not in classes:
            raise ValueError("The trained model does not contain the positive class.")
        positive_probability = float(
            model.predict_proba(input_frame)[0][classes.index(1)]
        )

    return Prediction(
        flagged=predicted_class == 1,
        positive_probability=positive_probability,
    )

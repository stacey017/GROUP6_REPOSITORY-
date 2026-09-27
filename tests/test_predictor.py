import math

import pytest

from medicare_app.predictor import (
    FEATURE_COLUMNS,
    MODEL_PATH,
    build_input_frame,
    load_model,
    predict_risk,
)

SAMPLE_FEATURES = {
    "age": 50,
    "sex": 1,
    "cp": 2,
    "trestbps": 120,
    "chol": 200,
    "fbs": 0,
    "restecg": 0,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 1.0,
    "slope": 2,
    "ca": None,
    "thal": None,
}


def test_input_frame_orders_features_and_preserves_missing_values():
    frame = build_input_frame(SAMPLE_FEATURES)

    assert tuple(frame.columns) == FEATURE_COLUMNS
    assert math.isnan(frame.loc[0, "ca"])
    assert math.isnan(frame.loc[0, "thal"])


def test_input_validation_rejects_missing_fields():
    features = SAMPLE_FEATURES.copy()
    del features["age"]

    with pytest.raises(ValueError, match="missing: age"):
        build_input_frame(features)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"age": None}, "age is required"),
        ({"extra": 1}, "unexpected: extra"),
        ({"sex": 2}, "sex must be one of"),
        ({"ca": 4}, "ca must be one of"),
        ({"age": float("inf")}, "age must be finite"),
        ({"age": True}, "age must be numeric"),
    ],
)
def test_input_validation_rejects_invalid_values(changes, message):
    features = SAMPLE_FEATURES | changes

    with pytest.raises(ValueError, match=message):
        build_input_frame(features)


def test_saved_model_predicts_a_probability():
    model = load_model()
    result = predict_risk(model, SAMPLE_FEATURES)

    assert MODEL_PATH.is_file()
    assert isinstance(result.flagged, bool)
    assert result.positive_probability is not None
    assert 0 <= result.positive_probability <= 1

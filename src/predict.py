import logging
import time
import joblib
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("heart_disease_inference")

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
REQUIRED_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

_model_cache = None

def load_model(model_path: str = "model.pkl"):
    global _model_cache
    if _model_cache is None:
        logger.info(f"Loading model from '{model_path}'...")
        _model_cache = joblib.load(model_path)
        logger.info("Model loaded and cached successfully.")
    return _model_cache

def validate_input(input_data: dict) -> None:
    missing = [f for f in REQUIRED_FEATURES if f not in input_data]
    if missing:
        raise ValueError(f"Missing required feature(s): {missing}")
    unexpected = [f for f in input_data if f not in REQUIRED_FEATURES]
    if unexpected:
        logger.warning(f"Ignoring unexpected field(s) not used by the model: {unexpected}")

def predict_heart_disease(input_data: dict) -> dict:
    start_time = time.time()
    validate_input(input_data)
    model = load_model()
    df = pd.DataFrame([{k: input_data[k] for k in REQUIRED_FEATURES}])
    try:
        prediction = int(model.predict(df)[0])
        proba = model.predict_proba(df)[0]
        confidence = float(proba[prediction])
    except Exception:
        logger.exception("Inference failed on input payload.")
        raise
    latency_ms = (time.time() - start_time) * 1000
    result = {
        "prediction": prediction,
        "label": "Disease" if prediction == 1 else "No Disease",
        "probability": round(confidence, 4),
        "latency_ms": round(latency_ms, 2),
    }
    logger.info(f"Prediction={result['prediction']} ({result['label']}) | confidence={result['probability']} | latency={result['latency_ms']}ms")
    return result

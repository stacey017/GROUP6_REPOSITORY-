from predict import predict_heart_disease, REQUIRED_FEATURES

dummy_payload = {
    "age": 55, "sex": 1, "cp": 3, "trestbps": 140, "chol": 250,
    "fbs": 0, "restecg": 1, "thalach": 150, "exang": 0, "oldpeak": 2.0,
    "slope": 1, "ca": 0, "thal": 2,
}

print("Test 1: normal payload")
result = predict_heart_disease(dummy_payload)
assert result["prediction"] in [0, 1]
assert "latency_ms" in result
print("  passed:", result)

print("\nTest 2: missing feature should raise ValueError")
bad_payload = dummy_payload.copy()
del bad_payload["thal"]
try:
    predict_heart_disease(bad_payload)
    raise AssertionError("Expected ValueError, got none")
except ValueError as e:
    print("  passed — correctly rejected:", e)

print(f"\nAll checks passed. Model expects {len(REQUIRED_FEATURES)} features.")

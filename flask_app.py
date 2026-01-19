import os
import pickle
from flask import Flask, request, jsonify

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.bin")

app = Flask(__name__)
with open(MODEL_PATH, "rb") as f:
    model_data = pickle.load(f)

model = model_data["model"]
scaler = model_data.get("scaler", None)
feature_names = model_data["feature_names"]
metrics = model_data.get("metrics", {})
config = model_data.get("config", {})


@app.get("/")
def root():
    return jsonify({
        "message": "Hit Song Classification API (Flask)",
        "required_features": feature_names,
        "model_metrics": metrics,
        "hit_threshold": config.get("hit_threshold", 70),
        "scaling": scaler is not None,
    })


@app.post("/predict")
def predict():
    try:
        payload = request.get_json(force=True)
        features = payload.get("features") if isinstance(payload, dict) else None
        if not isinstance(features, dict):
            return jsonify({"error": "Body must be JSON with 'features' dict"}), 400

        # missing feature check
        missing = [k for k in feature_names if k not in features]
        if missing:
            return jsonify({"error": "Missing features", "missing": missing}), 400

        try:
            x_row = [float(features[k]) for k in feature_names]
        except (TypeError, ValueError):
            return jsonify({"error": "All feature values must be numeric"}), 400

        X = [x_row]
        if scaler is not None:
            X = scaler.transform(X)

        # prediction
        pred = int(model.predict(X)[0])

        resp = {"hit": pred}
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]  # [P0, P1]
            resp["probabilities"] = {"non_hit": float(proba[0]), "hit": float(proba[1])}
            resp["confidence"] = float(proba[pred])

        return jsonify(resp)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)

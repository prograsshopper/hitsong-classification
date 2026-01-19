import os
import pickle
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# --- paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.bin")

app = FastAPI(title="Hit Song Classification API (FastAPI)")
with open(MODEL_PATH, "rb") as f:
    model_data = pickle.load(f)

model = model_data["model"]
scaler = model_data.get("scaler", None)
feature_names = model_data["feature_names"]
metrics = model_data.get("metrics", {})
config = model_data.get("config", {})


class TrackFeatures(BaseModel):
    features: Dict[str, float] = Field(
        ...,
        description="Dictionary of audio features (keys must match required_features)",
        example={
            "acousticness": 0.25,
            "danceability": 0.654,
            "duration_ms": 207857,
            "energy": 0.547,
            "instrumentalness": 0.0,
            "key": 10,
            "liveness": 0.0961,
            "loudness": -6.598,
            "mode": 1,
            "speechiness": 0.127,
            "tempo": 173.981,
            "time_signature": 4,
            "valence": 0.507,
            "popularity": 75
        }
    )


class PredictionResponse(BaseModel):
    hit: int
    confidence: Optional[float] = None
    probabilities: Optional[Dict[str, float]] = None


@app.get("/")
def root():
    return {
        "message": "Hit Song Classification API (FastAPI)",
        "required_features": feature_names,
        "model_metrics": metrics,
        "hit_threshold": config.get("hit_threshold", 70),
        "scaling": scaler is not None,
        "best_model": config.get("best_model"),
        "best_params": config.get("best_params"),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(input_data: TrackFeatures):
    try:
        feats = input_data.features
        missing = [k for k in feature_names if k not in feats]
        if missing:
            raise HTTPException(status_code=400, detail={"message": "Missing features", "missing": missing})

        try:
            x_row = [float(feats[k]) for k in feature_names]
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="All feature values must be numeric.")

        X = [x_row]
        if scaler is not None:
            X = scaler.transform(X)

        pred = int(model.predict(X)[0])
        confidence = None
        probabilities = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            probabilities = {"non_hit": float(proba[0]), "hit": float(proba[1])}
            confidence = float(proba[pred])

        return PredictionResponse(
            hit=pred,
            confidence=confidence,
            probabilities=probabilities
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

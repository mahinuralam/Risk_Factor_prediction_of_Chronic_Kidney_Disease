from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import RootModel

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "ckd_xgb_pipeline.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model pipeline not found at {MODEL_PATH}. Please run the training notebook or script to generate it."
    )

model = joblib.load(MODEL_PATH)
preprocessor = model.named_steps.get("preprocessor")
feature_names: List[str] = []
if preprocessor and preprocessor.transformers_:
    feature_names = list(preprocessor.transformers_[0][2])

if not feature_names:
    raise ValueError(
        "Unable to infer feature names from the trained pipeline. Ensure the preprocessing step exposes feature columns."
    )

default_feature_values: Dict[str, float] = {}
imputer = None
if preprocessor and preprocessor.transformers_:
    transformer_pipeline = preprocessor.transformers_[0][1]
    if hasattr(transformer_pipeline, "named_steps"):
        imputer = transformer_pipeline.named_steps.get("imputer")

if imputer is not None and hasattr(imputer, "statistics_"):
    default_feature_values = {
        name: (float(value) if value is not None and not pd.isna(value) else 0.0)
        for name, value in zip(feature_names, imputer.statistics_)
    }
else:
    default_feature_values = {name: 0.0 for name in feature_names}

classifier = model.named_steps.get("classifier")
if classifier is None or not hasattr(classifier, "feature_importances_"):
    raise ValueError("Classifier does not expose feature importances. Ensure the trained model supports them.")

feature_importance_pairs = list(zip(feature_names, classifier.feature_importances_))
feature_importance_pairs.sort(key=lambda item: item[1], reverse=True)

feature_importance_summary = [
    {"name": name, "importance": float(score)} for name, score in feature_importance_pairs
]

INPUT_FEATURE_ALIASES = {
    "age": {"feature": "age", "label": "Age"},
    "bgr": {"feature": "bgr", "label": "Blood Glucose (Random)"},
    "wbcc": {"feature": "wbcc", "label": "White Blood Cell Count"},
}

for alias, info in INPUT_FEATURE_ALIASES.items():
    if info["feature"] not in feature_names:
        raise ValueError(
            f"Configured feature '{info['feature']}' for alias '{alias}' not found in the trained pipeline.")

REQUIRED_FEATURES = [info["feature"] for info in INPUT_FEATURE_ALIASES.values()]
INPUT_FEATURE_DETAILS = [
    {"alias": alias, "feature": info["feature"], "full_name": info["label"]}
    for alias, info in INPUT_FEATURE_ALIASES.items()
]


class PredictRequest(RootModel[Dict[str, float]]):
    @property
    def payload(self) -> Dict[str, float]:
        return self.root


app = FastAPI(title="CKD Risk API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
@app.get("/")
def root() -> Dict[str, object]:
    return {
        "message": "CKD Risk API is running",
        "endpoints": ["/health", "/schema", "/predict"],
        "docs": "/docs",
        "inputs": INPUT_FEATURE_DETAILS
    }


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/schema")
def schema() -> Dict[str, object]:
    return {
        "features": feature_names,
        "accepted_inputs": INPUT_FEATURE_DETAILS,
        "feature_importances": feature_importance_summary,
        "target": "class",
        "notes": "Provide numeric inputs matching the preprocessing described in the notebook."
    }


@app.post("/predict")
def predict(request: PredictRequest) -> Dict[str, object]:
    raw_payload = request.payload

    normalized_payload: Dict[str, float] = {}
    provided_aliases: Dict[str, float] = {}

    for alias, info in INPUT_FEATURE_ALIASES.items():
        matching_keys = [key for key in raw_payload.keys() if key.lower() == alias or key.lower() == info["feature"]]
        if not matching_keys:
            continue

        key = matching_keys[0]
        value = raw_payload[key]

        if value is None:
            continue

        if isinstance(value, str):
            stripped = value.strip()
            if stripped == "":
                continue
            value = stripped

        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail={"error": f"{info['label']} must be numeric."})

        normalized_payload[info["feature"]] = numeric_value
        provided_aliases[alias] = numeric_value

    filled_payload: Dict[str, float] = {}
    autofilled: Dict[str, float] = {}
    for feature in feature_names:
        if feature in normalized_payload:
            filled_payload[feature] = normalized_payload[feature]
        else:
            default_value = default_feature_values.get(feature, 0.0)
            filled_payload[feature] = float(default_value)
            autofilled[feature] = float(default_value)

    sample_df = pd.DataFrame([filled_payload])
    proba = model.predict_proba(sample_df)[0, 1]
    prediction = int(proba >= 0.5)

    return {
        "probability_ckd": float(proba),
        "prediction": prediction,
        "inputs_used": provided_aliases,
        "autofilled_features": autofilled
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.app:app", host="127.0.0.1", port=5000, reload=True)

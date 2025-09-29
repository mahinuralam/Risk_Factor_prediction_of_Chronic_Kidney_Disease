from __future__ import annotations

import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Dataset" / "ckd-dataset-v2.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

number_regex = re.compile(r"[-+]?[0-9]*\.?[0-9]+")
categorical_lookup = {"yes": 1, "no": 0, "present": 1, "absent": 0, "good": 1, "poor": 0}


def parse_numeric_value(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    if text in ("", "nan"):
        return np.nan
    if text in categorical_lookup:
        return categorical_lookup[text]
    if text.startswith("s") and text[1:].isdigit():
        return float(text[1:])
    matches = number_regex.findall(text)
    if matches:
        numbers = [float(m) for m in matches]
        return float(np.mean(numbers))
    return pd.to_numeric(text, errors="coerce")


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df.columns = [re.sub(r"[^0-9a-zA-Z_]+", "_", col.strip()).strip("_").lower() for col in df.columns]
    df = df[df["class"].isin(["ckd", "notckd"])]
    df = df.replace({"?": np.nan})
    df = df.apply(
        lambda col: col.str.strip() if col.dtype == "object" else col
    )

    feature_cols = [col for col in df.columns if col != "class"]
    for col in feature_cols:
        df[col] = df[col].apply(parse_numeric_value)

    return df


def build_model(feature_cols: list[str]) -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, feature_cols),
        ]
    )

    xgb_clf = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=42,
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", xgb_clf)])


def main() -> None:
    df = load_dataset()
    feature_cols = [col for col in df.columns if col != "class"]
    X = df[feature_cols]
    y = df["class"].map({"ckd": 1, "notckd": 0})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = build_model(feature_cols)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_proba))
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["Not CKD", "CKD"]))

    model_path = MODEL_DIR / "ckd_xgb_pipeline.joblib"
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()

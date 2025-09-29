# Chronic Kidney Disease Risk Prediction

This project provides an end-to-end workflow for predicting the probability that a patient has Chronic Kidney Disease (CKD) using Bangladeshi patient records collected from Enam Medical College, Savar, Dhaka. The workspace includes:

- A Jupyter notebook (`CKD_Pipeline.ipynb`) for exploratory analysis, preprocessing, modeling, and evaluation.
- A reproducible training script (`train_model.py`) that saves the trained XGBoost pipeline to `models/ckd_xgb_pipeline.joblib`.
- A lightweight FastAPI backend (`backend/app.py`) that exposes REST endpoints for predictions.
- A simple HTML frontend (`frontend/index.html`) that interacts with the backend to display CKD probabilities for user-provided inputs.

## Project Structure

```
Risk Factor prediction of Chronic Kidney Disease/
├── Dataset/
│   └── ckd-dataset-v2.csv
├── CKD_Pipeline.ipynb
├── train_model.py
├── backend/
│   └── app.py
├── frontend/
│   └── index.html
├── models/
│   └── ckd_xgb_pipeline.joblib  # generated after training
├── requirements.txt
└── README.md
```

## Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train or Refresh the Model

The training notebook contains the full analysis, but you can also train directly via the script:

```bash
python train_model.py
```

A serialized pipeline is written to `models/ckd_xgb_pipeline.joblib` and is required by the backend API.

### 3. Launch the Backend

```bash
uvicorn backend.app:app --host 127.0.0.1 --port 5000 --reload
```

Endpoints:
- `GET /health` – service health check.
- `GET /schema` – returns the full feature schema, model feature importances, and the accepted input aliases.
- `POST /predict` – accepts only three inputs (`age`, `bgr`, `wbcc`) and back-fills the remaining model features before returning CKD probability.

### 4. Open the Frontend

Serve the static file from the `frontend` directory (for example, with VS Code Live Server or Python's simple server) and open `index.html` in a browser. Ensure the backend is running at `http://127.0.0.1:5000` before submitting inputs.

```bash
python -m http.server 8000 --directory frontend
```

Visit `http://127.0.0.1:8000` and fill in the form to receive a CKD probability and classification.

## Dataset Notes

- The dataset contains 173 patient records with mixed discrete measurements and ranges.
- Custom preprocessing converts textual ranges (e.g., `"1.019 - 1.021"`, `"≥ 1.023"`) into numerical midpoints for modeling.
- Missing values are imputed with median values.
- The target label `class` is mapped to binary: `ckd = 1`, `notckd = 0`.

## Modeling Overview

- **Algorithm**: XGBoost classifier with light regularization.
- **Pipeline**: Median imputation + standardization for numerical features.
- **Evaluation**: Accuracy, ROC AUC, classification report, ROC curve, and confusion matrix.
- **Performance**: Perfect accuracy/ROC AUC on the held-out test split (note: results may vary with different splits due to dataset size).

## Prediction Example

Example JSON payload for `POST /predict`:

```json
{
  "age": 45,
  "bgr": 118.0,
  "wbcc": 8200
}
```

Provide the patient's age (years), blood glucose random (mg/dL), and white blood cell count (cells/µL). The API fills in the remaining model features using the pipeline's training medians and returns the CKD probability and classification along with the interpreted inputs.

## Next Steps

- Gather more data to reduce overfitting and improve generalization.
- Add authentication or rate limiting before deploying publicly.
- Enhance the frontend with input validation and value hints based on original ranges.

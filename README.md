# Chronic Kidney Disease Risk Prediction

FastAPI + XGBoost service that estimates Chronic Kidney Disease (CKD) risk and ships with a lightweight web UI.

## Quick Start

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure the trained model exists**
   A serialized pipeline should live at `models/ckd_xgb_pipeline.joblib`. Regenerate it if needed:
   ```bash
   python train_model.py
   ```

3. **Run the app (development)**
   ```bash
   uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
   ```
   The landing page is available at `http://127.0.0.1:8000/` (redirects to `/app/`).

### Production-style server

```bash
gunicorn backend.app:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

### Docker workflow

```bash
docker build -t ckd-risk .
docker run --rm -p 8000:8000 ckd-risk
```

### Quick API check

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 55, "bgr": 130, "wbcc": 9000}'
```

## Screenshot

<!-- Replace with actual screenshot once available -->
![Main UI Screenshot](docs/screenshot.png)

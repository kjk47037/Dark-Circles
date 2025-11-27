# Models_detection
my models

## Cosmetics Detection API (Acne + Dark Circles)

FastAPI service that loads two YOLO models and exposes endpoints to detect acne and dark circles. Accepts base64 or multipart file uploads and returns detections plus base64-annotated images.

### Endpoints
- POST `/predict/acne` or `/predict/acne/json`
- POST `/predict/dark_circles` or `/predict/dark_circles/json`
- POST `/predict/combined` runs both on the same image and returns both results

### Run locally
```
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```
Docs: `http://localhost:8000`

### Deploy to Railway
- Uses Dockerfile with CPU-only PyTorch to keep image size < 4GB
- Set `CORS_ORIGINS` if needed for your frontend

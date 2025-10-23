# Deployment Guide

## Quick Start

### 1. Install Dependencies

```bash
pip install fastapi uvicorn
pip install git+https://github.com/JoHof/lungmask
```

### 2. Start Server

```bash
python deployment/serve.py
```

Server sẽ chạy tại: `http://localhost:8000`

### 3. Test API

#### Health Check

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "R231"
}
```

#### Segment Lung (JSON Response)

```bash
curl -X POST "http://localhost:8000/segment" \
  -F "file=@sample-data/Task06_Lung/imagesTr/lung_001.nii.gz"
```

**Response:**
```json
{
  "status": "success",
  "lung_volume_ml": 4523.8,
  "inference_time_seconds": 5.2,
  "patient_id": "lung_001",
  "model_used": "R231"
}
```

#### Segment Lung (Download Mask File)

```bash
curl -X POST "http://localhost:8000/segment-with-mask" \
  -F "file=@sample-data/Task06_Lung/imagesTr/lung_001.nii.gz" \
  -o lung_001_segmentation.nii.gz
```

### 4. Interactive API Docs

Mở browser: `http://localhost:8000/docs`

FastAPI tự động tạo Swagger UI để test API.

---

## Configuration

### Environment Variables

```bash
# Model selection
export MODEL_NAME=R231                # R231 or R231CovidWeb

# Force CPU (if no GPU)
export FORCE_CPU=true

# Start server
python deployment/serve.py
```

### Change Port

```bash
# Edit serve.py line:
uvicorn.run(app, host="0.0.0.0", port=9000)  # Change to 9000
```

---

## Production Deployment

### Option 1: Docker

**Dockerfile:**
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install git+https://github.com/JoHof/lungmask
RUN pip install fastapi uvicorn

# Copy code
COPY deployment/serve.py .

# Expose port
EXPOSE 8000

# Run
CMD ["python", "serve.py"]
```

**Build and run:**
```bash
docker build -t lung-segmentation-api .
docker run -p 8000:8000 lung-segmentation-api
```

### Option 2: Kubernetes

**deployment.yaml:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: lung-segmentation
spec:
  replicas: 2
  selector:
    matchLabels:
      app: lung-segmentation
  template:
    metadata:
      labels:
        app: lung-segmentation
    spec:
      containers:
      - name: api
        image: lung-segmentation-api:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
        env:
        - name: MODEL_NAME
          value: "R231"
        - name: FORCE_CPU
          value: "true"
---
apiVersion: v1
kind: Service
metadata:
  name: lung-segmentation
spec:
  selector:
    app: lung-segmentation
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

**Deploy:**
```bash
kubectl apply -f deployment.yaml
```

### Option 3: Systemd Service (Linux Server)

**lung-segmentation.service:**
```ini
[Unit]
Description=Lung Segmentation API
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/hospital-mlops
Environment="PATH=/opt/hospital-mlops/venv/bin"
Environment="MODEL_NAME=R231"
Environment="FORCE_CPU=true"
ExecStart=/opt/hospital-mlops/venv/bin/python deployment/serve.py
Restart=always

[Install]
WantedBy=multi-user.target
```

**Install:**
```bash
sudo cp lung-segmentation.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable lung-segmentation
sudo systemctl start lung-segmentation
```

---

## Performance Tuning

### Batch Processing

Modify `serve.py`:

```python
# Change batch_size for faster processing (requires more RAM)
lung_mask = mask.apply(ct_scan, model=MODEL_NAME, batch_size=4)
```

### GPU Support

```python
# Remove force_cpu to use GPU
lung_mask = mask.apply(ct_scan, model=MODEL_NAME)
```

**Requirements:**
- CUDA-compatible GPU
- PyTorch with CUDA support: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`

---

## Monitoring

### Add Prometheus Metrics

```bash
pip install prometheus-fastapi-instrumentator
```

**In serve.py:**
```python
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI(...)

# Add Prometheus metrics
Instrumentator().instrument(app).expose(app)
```

Access metrics: `http://localhost:8000/metrics`

### Logging

**Add structured logging:**
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

@app.post("/segment")
async def segment_lung(file: UploadFile):
    logger.info(f"Processing file: {file.filename}")
    # ... existing code
    logger.info(f"Completed in {inference_time:.2f}s, Dice: {dice:.4f}")
```

---

## Security

### Add Authentication

```bash
pip install python-jose[cryptography] passlib[bcrypt]
```

**Add JWT auth:**
```python
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

@app.post("/segment")
async def segment_lung(
    file: UploadFile,
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    token = credentials.credentials
    # Verify JWT token
    # ... existing code
```

### Rate Limiting

```bash
pip install slowapi
```

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/segment")
@limiter.limit("10/minute")  # Max 10 requests per minute
async def segment_lung(request: Request, file: UploadFile):
    # ... existing code
```

---

## Troubleshooting

### "lungmask not installed"

```bash
pip install git+https://github.com/JoHof/lungmask
```

### "CUDA out of memory"

```bash
export FORCE_CPU=true
python deployment/serve.py
```

### Port already in use

```bash
# Find process using port 8000
lsof -i :8000

# Kill process
kill -9 <PID>

# Or change port in serve.py
uvicorn.run(app, port=9000)
```

---

## API Reference

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Root endpoint |
| `/health` | GET | Health check |
| `/segment` | POST | Segment lung (JSON response) |
| `/segment-with-mask` | POST | Segment lung (download mask) |
| `/models` | GET | List available models |
| `/docs` | GET | Swagger UI |

### Request Format

**Content-Type:** `multipart/form-data`

**Parameters:**
- `file`: CT scan file (.nii or .nii.gz)

### Response Format

```json
{
  "status": "success",
  "lung_volume_ml": 4523.8,
  "inference_time_seconds": 5.2,
  "patient_id": "lung_001",
  "model_used": "R231"
}
```

---

## Next Steps

1. ✅ Test locally: `python deployment/serve.py`
2. ✅ Test API: `curl http://localhost:8000/health`
3. ✅ Deploy to production (Docker/Kubernetes)
4. ✅ Add monitoring (Prometheus)
5. ✅ Add authentication (JWT)
6. ✅ Set up logging and alerting

**For production checklist, see:** [PRODUCTION_CHECKLIST.md](PRODUCTION_CHECKLIST.md)

# Setup Guide: Kubeflow Spleen Segmentation Pipeline

## Prerequisites

### 1. Start Docker Desktop
- Mở Docker Desktop và chờ nó khởi động hoàn toàn
- Kiểm tra: `docker info` (phải không có lỗi)

### 2. Start Minikube
```bash
minikube start
minikube status  # Phải show "Running"
```

### 3. Verify Kubeflow
```bash
kubectl get pods -n kubeflow
# Phải thấy các pods đang chạy
```

## Setup Pipeline

### Bước 1: Prepare Input Data ✅

**Đã hoàn thành!** Data đã được copy vào:
```
~/kubeflow-data/inputs/week_current/
├── spleen_12/imaging.nii.gz  (46MB)
├── spleen_19/imaging.nii.gz  (15MB)
├── spleen_29/imaging.nii.gz  (30MB)
└── spleen_9/imaging.nii.gz   (12MB)
```

Kiểm tra:
```bash
ls -lh ~/kubeflow-data/inputs/week_current/*/imaging.nii.gz
```

### Bước 2: Mount Data to Minikube

Mở **terminal riêng** và chạy (giữ terminal này mở):
```bash
minikube mount ~/kubeflow-data:/mnt/data
```

**QUAN TRỌNG:** Terminal này phải luôn chạy. Đừng tắt!

### Bước 3: Build Docker Image

```bash
cd E:/monai-kubeflow-demo/monai-demo/kubeflow_pipeline

# Cách 1: Dùng script (khuyến nghị)
bash build.sh

# Cách 2: Thủ công
eval $(minikube docker-env)
cd ../..
docker build -f monai-demo/kubeflow_pipeline/Dockerfile -t spleen-pipeline:v1 .
```

Verify:
```bash
docker images | grep spleen-pipeline
```

### Bước 4: Deploy PersistentVolume

```bash
cd E:/monai-kubeflow-demo/monai-demo/kubeflow_pipeline
kubectl apply -f pv.yaml
```

Verify:
```bash
kubectl get pv
kubectl get pvc -n kubeflow
```

Output mong muốn:
```
NAME              STATUS   VOLUME           CAPACITY   ACCESS MODES
data-pvc          Bound    spleen-data-pv   10Gi       RWX
```

### Bước 5: Compile Pipeline

```bash
# Install KFP SDK (nếu chưa có)
pip install kfp

# Compile
python pipeline.py
```

Output: `spleen_pipeline.yaml`

### Bước 6: Upload to Kubeflow UI

1. Mở Kubeflow UI:
   ```bash
   # Nếu chưa port-forward
   kubectl port-forward -n istio-system service/istio-ingressgateway 8080:80
   ```
   Truy cập: http://localhost:8080

2. **Upload Pipeline:**
   - Sidebar: **Pipelines** → **Upload Pipeline**
   - File: `spleen_pipeline.yaml`
   - Name: `Spleen Segmentation Weekly`
   - Click **Create**

3. **Create Run:**
   - Click vào pipeline vừa tạo
   - **Create run**
   - Run name: `test-run-001`
   - Namespace: `kubeflow`
   - Click **Start**

### Bước 7: Monitor Run

Trong Kubeflow UI:
- Click vào run name
- Xem real-time logs của từng component
- Theo dõi progress: preprocess → inference → visualize

Hoặc dùng kubectl:
```bash
# List pods
kubectl get pods -n kubeflow | grep spleen

# View logs
kubectl logs -f <pod-name> -n kubeflow
```

### Bước 8: View Results

Sau khi pipeline hoàn thành:
```bash
# Check outputs
ls ~/kubeflow-data/outputs/week_current/*/

# View images (Windows)
start ~/kubeflow-data/outputs/week_current/spleen_12/axial.png
```

Expected structure:
```
~/kubeflow-data/outputs/week_current/
├── spleen_12/
│   ├── segmentation.nii.gz
│   ├── axial.png
│   ├── coronal.png
│   └── sagittal.png
├── spleen_19/
│   └── ... (same)
├── spleen_29/
│   └── ... (same)
└── spleen_9/
    └── ... (same)
```

## Setup Recurring Run

Để pipeline tự chạy mỗi tuần:

1. Kubeflow UI → **Recurring Runs**
2. **Create Recurring Run**
3. Config:
   - Pipeline: `Spleen Segmentation Weekly`
   - Run name: `weekly-auto-run`
   - Trigger: **Cron**
   - Cron expression: `0 0 * * 6` (Mỗi T7 lúc 00:00)
   - Max concurrent runs: `1`
4. **Create**

## Weekly Workflow (Sau khi setup xong)

Mỗi tuần, BSI chỉ cần:

1. Copy CT scans mới vào:
   ```bash
   cp new_patient.nii.gz ~/kubeflow-data/inputs/week_current/spleen_XX/imaging.nii.gz
   ```

2. Pipeline sẽ tự chạy vào T7, hoặc:
   ```bash
   # Chạy manual qua UI: Pipelines → Run
   ```

3. Xem kết quả tại:
   ```bash
   ~/kubeflow-data/outputs/week_current/
   ```

## Troubleshooting

### Docker build fails
```bash
# Check Docker is running
docker info

# Check you're in the right directory
pwd  # Should be E:/monai-kubeflow-demo

# Try building with verbose output
docker build --progress=plain -f monai-demo/kubeflow_pipeline/Dockerfile -t spleen-pipeline:v1 .
```

### Minikube mount not working
```bash
# Stop existing mount
pkill -f "minikube mount"

# Restart mount
minikube mount ~/kubeflow-data:/mnt/data
```

### PVC not bound
```bash
# Delete and recreate
kubectl delete -f pv.yaml
kubectl apply -f pv.yaml

# Check status
kubectl get pv,pvc -n kubeflow
```

### Pipeline pods failing
```bash
# Get pod name
kubectl get pods -n kubeflow | grep spleen

# View logs
kubectl logs <pod-name> -n kubeflow

# Describe pod
kubectl describe pod <pod-name> -n kubeflow
```

### Can't access Kubeflow UI
```bash
# Port-forward
kubectl port-forward -n istio-system service/istio-ingressgateway 8080:80

# Or check minikube service
minikube service list
```

## Quick Reference

```bash
# Start services
docker info                           # Check Docker
minikube start                        # Start Minikube
minikube mount ~/kubeflow-data:/mnt/data  # Mount data (keep running)

# Build & deploy
cd E:/monai-kubeflow-demo/monai-demo/kubeflow_pipeline
bash build.sh                         # Build image
kubectl apply -f pv.yaml              # Deploy PV/PVC
python pipeline.py                    # Compile pipeline

# Monitor
kubectl get pv,pvc -n kubeflow        # Check volumes
kubectl get pods -n kubeflow          # Check pods
kubectl logs -f <pod> -n kubeflow     # View logs

# Clean up
kubectl delete -f pv.yaml             # Remove PV/PVC
rm -rf ~/kubeflow-data/outputs/*      # Clear outputs
```

# HƯỚNG DẪN HOÀN TẤT KUBEFLOW PIPELINE

## 📌 TÌNH TRẠNG HIỆN TẠI

✅ **ĐÃ XONG:**
- Cài đặt Kubeflow Pipelines thành công
- Sửa UI pod (không còn crash)
- Tạo PVC (`data-pvc`) cho pipeline data
- Đang build Docker image `spleen-pipeline:v1`

⏳ **ĐANG CHỜ:**
- Docker image build (có thể mất 5-10 phút)

❌ **VẤN ĐỀ CÒN LẠI:**
- Pipeline cần image `spleen-pipeline:v1` để chạy

---

## 🔧 BƯỚC 1: HOÀN TẤT BUILD DOCKER IMAGE

### Kiểm tra build đã xong chưa:
```bash
# Kiểm tra image trong Minikube
minikube image ls | grep spleen
```

**Nếu thấy `spleen-pipeline:v1` → Chuyển sang Bước 2**

**Nếu KHÔNG thấy image → Build lại:**

### Cách 1: Build bằng Minikube (khuyến nghị)
```bash
cd E:\monai-kubeflow-demo
minikube image build -t spleen-pipeline:v1 -f Dockerfile.spleen .
```

### Cách 2: Build bằng Docker rồi load vào Minikube
```bash
# Build image
docker build -t spleen-pipeline:v1 -f Dockerfile.spleen .

# Load vào Minikube
minikube image load spleen-pipeline:v1
```

### Verify image đã có:
```bash
minikube image ls | grep spleen
# Kết quả: spleen-pipeline:v1
```

---

## 🔧 BƯỚC 2: XÓA CÁC WORKFLOW CŨ (FAILED)

```bash
# Xem các workflows
kubectl get workflows -n kubeflow

# Xóa tất cả workflows cũ
kubectl delete workflows --all -n kubeflow
```

---

## 🔧 BƯỚC 3: TẠO LẠI PIPELINE RUN

### Option A: Qua UI (dễ nhất)

1. Mở port-forward:
```bash
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80
```

2. Truy cập: **http://localhost:8080**

3. Vào **Pipelines** → Chọn pipeline → Click **"Create run"**

4. Điền thông tin:
   - Experiment: Default
   - Run name: Tùy chọn
   - Click **"Start"**

### Option B: Qua kubectl (nhanh)

```bash
# Tạo run từ YAML file
kubectl create -f monai-demo/kubeflow_pipeline/spleen_pipeline_v2.yaml -n kubeflow
```

---

## 🔧 BƯỚC 4: MONITOR PIPELINE EXECUTION

### Xem workflow status:
```bash
kubectl get workflows -n kubeflow
```

### Xem pods đang chạy:
```bash
kubectl get pods -n kubeflow | grep spleen
```

### Xem logs của một pod:
```bash
# Lấy tên pod
POD_NAME=$(kubectl get pods -n kubeflow | grep spleen | grep Running | head -1 | awk '{print $1}')

# Xem logs
kubectl logs $POD_NAME -n kubeflow -c main
```

### Xem chi tiết workflow:
```bash
kubectl describe workflow <workflow-name> -n kubeflow
```

---

## 📊 TROUBLESHOOTING

### Vấn đề 1: "ErrImageNeverPull"
**Nguyên nhân:** Image chưa được build

**Giải pháp:** Làm lại Bước 1

### Vấn đề 2: "PVC not found"
**Nguyên nhân:** PVC bị xóa

**Giải pháp:**
```bash
kubectl apply -f monai-demo/kubeflow_pipeline/data-pvc.yaml
```

### Vấn đề 3: Pod bị "Pending"
**Kiểm tra:**
```bash
kubectl describe pod <pod-name> -n kubeflow
```

**Nguyên nhân thường gặp:**
- Thiếu resources (CPU/Memory)
- PVC chưa bound
- Image pull error

### Vấn đề 4: UI không truy cập được
**Giải pháp:**
```bash
# Kiểm tra UI pod
kubectl get pods -n kubeflow | grep ml-pipeline-ui

# Nếu không Running, restart
kubectl delete pod -n kubeflow -l app=ml-pipeline-ui

# Tạo port-forward mới
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80
```

---

## 🚀 SAU KHI PIPELINE CHẠY THÀNH CÔNG

### Xem kết quả:
1. Vào UI → **Runs** → Click vào run name
2. Xem từng step và logs
3. Download artifacts nếu có

### Xem output files:
```bash
# List files trong PVC
kubectl exec -it -n kubeflow <pod-name> -- ls -la /data
```

---

## 📝 CÁC LỆNH HỮU ÍCH

```bash
# Xem tất cả resources trong namespace kubeflow
kubectl get all -n kubeflow

# Xem logs real-time
kubectl logs -f <pod-name> -n kubeflow -c main

# Restart một deployment
kubectl rollout restart deployment/<deployment-name> -n kubeflow

# Xóa tất cả workflows failed
kubectl delete workflows -n kubeflow --field-selector status.phase=Failed

# Port-forward đến Minio (xem artifacts)
kubectl port-forward -n kubeflow svc/minio-service 9000:9000
# Truy cập: http://localhost:9000
# Credentials: minio / minio123
```

---

## 🎯 CHECKLIST HOÀN TẤT

- [ ] Docker image `spleen-pipeline:v1` đã được build và có trong Minikube
- [ ] PVC `data-pvc` đã tạo và Bound
- [ ] UI pod đang Running và accessible qua port 8080
- [ ] Workflows cũ (failed) đã được xóa
- [ ] Pipeline run mới đã được tạo
- [ ] Các pods đang Running (không còn ErrImageNeverPull)
- [ ] Workflow status: Running hoặc Succeeded

---

## ⏰ THỜI GIAN DỰ KIẾN

- Build Docker image: **5-10 phút**
- Pipeline execution: **10-20 phút** (tùy số lượng images và complexity)

---

## 📞 NẾU GẶP VẤN ĐỀ

1. Kiểm tra logs của pods bị lỗi
2. Verify image đã có trong Minikube
3. Kiểm tra PVC status
4. Xem events: `kubectl get events -n kubeflow --sort-by='.lastTimestamp'`

**Chúc may mắn! 🚀**

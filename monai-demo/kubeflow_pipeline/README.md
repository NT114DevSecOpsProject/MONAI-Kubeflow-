# Kubeflow Pipeline: Spleen Segmentation

Automated weekly spleen CT segmentation for 4 patients using MONAI.

## Structure

```
kubeflow_pipeline/
├── components/
│   ├── preprocess.py    # Load NIfTI → MONAI normalize → save tensor
│   ├── inference.py     # Load tensor → MONAI UNet → save mask.nii.gz
│   └── visualize.py     # Load CT+mask → matplotlib overlay → save 3 PNG
├── Dockerfile           # Docker image: python:3.9-slim + MONAI
├── pv.yaml             # PersistentVolume + PVC for /mnt/data
├── pipeline.py         # Kubeflow pipeline definition
└── README.md           # This file
```

## Prerequisites

- Kubeflow + Minikube installed
- Model downloaded: `../../models/spleen_ct_segmentation/models/model.pt`
- Dataset: Task09 Spleen (spleen_12, spleen_19, spleen_29, spleen_9)

## Data Structure

```
~/kubeflow-data/
├── inputs/week_current/
│   ├── spleen_12/imaging.nii.gz
│   ├── spleen_19/imaging.nii.gz
│   ├── spleen_29/imaging.nii.gz
│   └── spleen_9/imaging.nii.gz
└── outputs/week_current/
    ├── spleen_12/
    │   ├── segmentation.nii.gz
    │   ├── axial.png
    │   ├── coronal.png
    │   └── sagittal.png
    └── ... (same for other patients)
```

## Setup Steps

### 1. Prepare Data Directory

```bash
# On Windows (Git Bash or PowerShell)
mkdir -p ~/kubeflow-data/inputs/week_current
mkdir -p ~/kubeflow-data/outputs/week_current

# Copy patient CT scans
# Example:
# cp path/to/spleen_12.nii.gz ~/kubeflow-data/inputs/week_current/spleen_12/imaging.nii.gz
```

### 2. Mount Data to Minikube

```bash
# Keep this running in a separate terminal
minikube mount ~/kubeflow-data:/mnt/data
```

### 3. Build Docker Image

```bash
cd monai-demo/kubeflow_pipeline

# Use Minikube's Docker daemon
eval $(minikube docker-env)

# Build image
docker build -t spleen-pipeline:v1 .

# Verify
docker images | grep spleen-pipeline
```

### 4. Deploy PersistentVolume

```bash
kubectl apply -f pv.yaml

# Verify
kubectl get pv
kubectl get pvc -n kubeflow
```

### 5. Compile Pipeline

```bash
python pipeline.py
```

This creates `spleen_pipeline.yaml`.

### 6. Upload to Kubeflow

1. Open Kubeflow UI: `http://localhost:8080` (or your Kubeflow address)
2. Go to **Pipelines** → **Upload Pipeline**
3. Upload `spleen_pipeline.yaml`
4. Click **Create Run**
5. Select namespace: `kubeflow`
6. Click **Start**

### 7. Setup Recurring Schedule (Optional)

For weekly automatic runs:

1. In Kubeflow UI, go to **Recurring Runs**
2. Click **Create Recurring Run**
3. Select the pipeline
4. Set schedule: `0 0 * * 6` (Every Saturday at midnight)
5. Click **Create**

## Pipeline Flow

```
For each patient (spleen_12, spleen_19, spleen_29, spleen_9):
  1. Preprocess   → Load NIfTI, normalize, save tensor
  2. Inference    → Run MONAI UNet, save segmentation mask
  3. Visualize    → Create 3-view PNG overlays
```

## Output Files

After pipeline runs successfully:

```bash
~/kubeflow-data/outputs/week_current/spleen_12/
├── segmentation.nii.gz  # Segmentation mask (NIfTI format)
├── axial.png           # Axial view (top-down)
├── coronal.png         # Coronal view (front)
└── sagittal.png        # Sagittal view (side)
```

## Troubleshooting

### Pipeline fails at preprocess
- Check if input files exist: `ls ~/kubeflow-data/inputs/week_current/spleen_*/imaging.nii.gz`
- Check minikube mount is running
- Check PVC is bound: `kubectl get pvc -n kubeflow`

### Pipeline fails at inference
- Check model exists in Docker image: `docker run --rm spleen-pipeline:v1 ls /app/models/spleen_ct_segmentation/models/`
- Check if model.pt is valid

### Pipeline fails at visualize
- Check if segmentation.nii.gz was created
- Check logs: `kubectl logs <pod-name> -n kubeflow`

## Logs

View logs for a specific run:

```bash
# List pods
kubectl get pods -n kubeflow

# View logs
kubectl logs <pod-name> -n kubeflow

# Follow logs
kubectl logs -f <pod-name> -n kubeflow
```

## Clean Up

```bash
# Delete PVC
kubectl delete pvc data-pvc -n kubeflow

# Delete PV
kubectl delete pv spleen-data-pv

# Stop minikube mount (Ctrl+C in terminal)
```

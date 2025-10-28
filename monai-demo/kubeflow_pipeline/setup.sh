#!/bin/bash
# Quick Setup Script for Spleen Segmentation Pipeline

set -e  # Exit on error

echo "======================================================================"
echo "KUBEFLOW PIPELINE SETUP: SPLEEN SEGMENTATION"
echo "======================================================================"

# Step 1: Check prerequisites
echo ""
echo "[Step 1/5] Checking prerequisites..."
command -v minikube >/dev/null 2>&1 || { echo "ERROR: minikube not found"; exit 1; }
command -v kubectl >/dev/null 2>&1 || { echo "ERROR: kubectl not found"; exit 1; }
command -v docker >/dev/null 2>&1 || { echo "ERROR: docker not found"; exit 1; }
command -v python >/dev/null 2>&1 || { echo "ERROR: python not found"; exit 1; }
echo "  [OK] All prerequisites found"

# Step 2: Create data directories
echo ""
echo "[Step 2/5] Creating data directories..."
mkdir -p ~/kubeflow-data/inputs/week_current
mkdir -p ~/kubeflow-data/outputs/week_current
echo "  [OK] Created ~/kubeflow-data/"

# Step 3: Build Docker image
echo ""
echo "[Step 3/5] Building Docker image..."
echo "  Setting up Minikube Docker environment..."
eval $(minikube docker-env)

echo "  Building spleen-pipeline:v1..."
docker build -t spleen-pipeline:v1 .

echo "  [OK] Image built: spleen-pipeline:v1"
docker images | grep spleen-pipeline

# Step 4: Deploy PersistentVolume
echo ""
echo "[Step 4/5] Deploying PersistentVolume..."
kubectl apply -f pv.yaml
echo "  [OK] PV and PVC deployed"

# Wait for PVC to be bound
echo "  Waiting for PVC to be bound..."
kubectl wait --for=condition=Bound pvc/data-pvc -n kubeflow --timeout=60s || true

# Step 5: Compile pipeline
echo ""
echo "[Step 5/5] Compiling pipeline..."
python pipeline.py
echo "  [OK] Pipeline compiled: spleen_pipeline.yaml"

echo ""
echo "======================================================================"
echo "SETUP COMPLETE!"
echo "======================================================================"
echo ""
echo "Next steps:"
echo "1. Start minikube mount (in separate terminal):"
echo "   minikube mount ~/kubeflow-data:/mnt/data"
echo ""
echo "2. Copy patient data to:"
echo "   ~/kubeflow-data/inputs/week_current/spleen_XX/imaging.nii.gz"
echo ""
echo "3. Upload to Kubeflow UI:"
echo "   - Open http://localhost:8080"
echo "   - Pipelines → Upload → spleen_pipeline.yaml"
echo "   - Create Run"
echo ""
echo "4. View results at:"
echo "   ~/kubeflow-data/outputs/week_current/"
echo ""
echo "======================================================================"

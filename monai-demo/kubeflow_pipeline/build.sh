#!/bin/bash
# Build Script for Spleen Pipeline Docker Image

set -e  # Exit on error

echo "======================================================================"
echo "BUILD DOCKER IMAGE: spleen-pipeline:v1"
echo "======================================================================"

# Check Docker is running
echo ""
echo "[Step 1/4] Checking Docker..."
if ! docker info >/dev/null 2>&1; then
    echo "ERROR: Docker is not running!"
    echo "Please start Docker Desktop and try again."
    exit 1
fi
echo "  [OK] Docker is running"

# Check Minikube is running
echo ""
echo "[Step 2/4] Checking Minikube..."
if ! minikube status >/dev/null 2>&1; then
    echo "Minikube is not running. Starting Minikube..."
    minikube start
fi
echo "  [OK] Minikube is running"

# Configure Docker to use Minikube's daemon
echo ""
echo "[Step 3/4] Configuring Minikube Docker environment..."
eval $(minikube docker-env)
echo "  [OK] Docker environment configured"

# Build image
echo ""
echo "[Step 4/4] Building Docker image..."
echo "  Context: $(pwd)/../.."
echo "  Dockerfile: monai-demo/kubeflow_pipeline/Dockerfile"

cd ../..  # Go to root (E:\monai-kubeflow-demo)
docker build -f monai-demo/kubeflow_pipeline/Dockerfile -t spleen-pipeline:v1 .

echo ""
echo "======================================================================"
echo "BUILD COMPLETE!"
echo "======================================================================"
docker images | grep spleen-pipeline
echo ""

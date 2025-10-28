# Build Script for Spleen Pipeline Docker Image (PowerShell)

$ErrorActionPreference = "Stop"

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "BUILD DOCKER IMAGE: spleen-pipeline:v1" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

# Check Docker is running
Write-Host ""
Write-Host "[Step 1/4] Checking Docker..." -ForegroundColor Yellow
try {
    docker info *>$null
    Write-Host "  [OK] Docker is running" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Docker is not running!" -ForegroundColor Red
    Write-Host "Please start Docker Desktop and try again." -ForegroundColor Red
    exit 1
}

# Check Minikube is running
Write-Host ""
Write-Host "[Step 2/4] Checking Minikube..." -ForegroundColor Yellow
$minikubeStatus = minikube status 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Minikube is not running. Starting Minikube..." -ForegroundColor Yellow
    minikube start
}
Write-Host "  [OK] Minikube is running" -ForegroundColor Green

# Configure Docker to use Minikube's daemon
Write-Host ""
Write-Host "[Step 3/4] Configuring Minikube Docker environment..." -ForegroundColor Yellow
& minikube -p minikube docker-env --shell powershell | Invoke-Expression
Write-Host "  [OK] Docker environment configured" -ForegroundColor Green

# Build image
Write-Host ""
Write-Host "[Step 4/4] Building Docker image..." -ForegroundColor Yellow
Write-Host "  Context: E:\monai-kubeflow-demo" -ForegroundColor Gray
Write-Host "  Dockerfile: monai-demo/kubeflow_pipeline/Dockerfile" -ForegroundColor Gray

Set-Location E:\monai-kubeflow-demo
docker build -f monai-demo/kubeflow_pipeline/Dockerfile -t spleen-pipeline:v1 .

Write-Host ""
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "BUILD COMPLETE!" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
docker images | Select-String "spleen-pipeline"
Write-Host ""

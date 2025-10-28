# Spleen Segmentation Pipeline Docker Image
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
# Install torch first from PyTorch CPU index
RUN pip install --no-cache-dir torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu

# Install other packages from PyPI
RUN pip install --no-cache-dir \
    monai==1.3.0 \
    nibabel==5.1.0 \
    matplotlib==3.7.1 \
    numpy==1.24.3

# Create app directory
WORKDIR /app

# Copy components
COPY monai-demo/kubeflow_pipeline/components/ /app/components/

# Copy models
COPY models/ /app/models/

# Set Python path
ENV PYTHONPATH=/app

# Default command (will be overridden by Kubeflow)
CMD ["python", "--version"]

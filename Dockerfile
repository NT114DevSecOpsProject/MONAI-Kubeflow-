FROM python:3.10-slim

WORKDIR /app

# Install system dependencies including build tools
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip first
RUN pip install --no-cache-dir --upgrade pip

# FIX: Install NumPy < 2.0 FIRST to avoid compatibility issues
RUN pip install --no-cache-dir "numpy<2.0"

# Install PyTorch (CPU version)
RUN pip install --no-cache-dir \
    torch==2.0.0 \
    torchvision==0.15.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Install MONAI without optional dependencies
RUN pip install --no-cache-dir \
    monai==1.3.0 \
    nibabel \
    pillow \
    matplotlib \
    scikit-image \
    tqdm

# Verify NumPy version
RUN python -c "import numpy; print(f'NumPy version: {numpy.__version__}')" && \
    python -c "import torch; print(f'PyTorch version: {torch.__version__}')" && \
    python -c "import monai; print(f'MONAI version: {monai.__version__}')"

# Copy training script
COPY train_simple.py /app/train_simple.py

# Download MedNIST dataset
RUN python -c "from monai.apps import download_and_extract; \
    print('Downloading MedNIST dataset...'); \
    download_and_extract( \
        url='https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/MedNIST.tar.gz', \
        output_dir='/app/data' \
    ); \
    print('Download completed!')"

# Verify data exists
RUN ls -lh /app/data/MedNIST && \
    echo "Classes available:" && \
    ls /app/data/MedNIST

# Create output directory
RUN mkdir -p /output

# Clean up to reduce image size
RUN apt-get purge -y gcc g++ wget && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

CMD ["python", "train_simple.py"]
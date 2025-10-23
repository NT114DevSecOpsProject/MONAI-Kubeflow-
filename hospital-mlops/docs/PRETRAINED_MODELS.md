# Pretrained Models Chi Tiết - Đầy đủ nhất

## 🎯 Tổng quan

Danh sách đầy đủ các pretrained models có thể sử dụng cho CT lung segmentation tại bệnh viện.

---

## 1. MONAI Model Zoo (Official Medical AI)

### 1.1 Whole Body CT Segmentation ⭐⭐⭐ BEST

**Download**:
```bash
python -m monai.bundle download "wholeBody_ct_segmentation" --bundle_dir ./models/
```

**Thông tin**:
- **Training dataset**: 1,204 CT volumes
- **Output**: 104 anatomical structures
  - Lungs (left, right, upper/middle/lower lobes)
  - Trachea
  - Bronchi
  - Pulmonary vessels
  - Airways
  - + 99 cơ quan khác (heart, liver, kidney...)
- **Accuracy**: Dice 0.85 average (lung: ~0.90)
- **Model**: SegResNet ensemble
- **Size**: 500 MB
- **Inference time**: ~30s/volume (V100 GPU)
- **Link**: https://github.com/Project-MONAI/model-zoo/tree/dev/models/wholeBody_ct_segmentation

**Sử dụng**:
```python
from monai.bundle import ConfigParser
import torch

config = ConfigParser()
config.read_config("models/wholeBody_ct_segmentation/configs/inference.json")

# Load model
model = config.get_parsed_content("network_def")
model.load_state_dict(torch.load("models/wholeBody_ct_segmentation/models/model.pt"))
model.eval()

# Inference
output = model(ct_volume)  # [B, 104, H, W, D]

# Extract lung only
lung_mask = output[:, [1, 2], ...]  # Channels 1,2 = left/right lung
```

**Download offline**:
```bash
# Trên máy có Internet
wget https://github.com/Project-MONAI/model-zoo/releases/download/hosting_storage_v1/wholeBody_ct_segmentation_v0.1.0.zip
unzip wholeBody_ct_segmentation_v0.1.0.zip
```

---

### 1.2 COVID-19 Lung CT Segmentation ⭐⭐

**Download**:
```bash
python -m monai.bundle download "covid19_lung_ct_segmentation"
```

**Thông tin**:
- **Training dataset**: 1,000+ COVID-19 CT scans
- **Output**:
  - Channel 0: Background
  - Channel 1: Lung tissue
  - Channel 2: COVID-19 infection/lesions
- **Accuracy**:
  - Lung: Dice 0.88
  - Infection: Dice 0.75
- **Model**: UNet
- **Size**: 80 MB
- **Inference time**: ~10s/volume
- **Link**: https://github.com/Project-MONAI/model-zoo/tree/dev/models/covid19_lung_ct_segmentation

**Use case**:
- Segment lung + lesions
- Fine-tune cho general lesions (not just COVID)

---

### 1.3 Lung Nodule CT Detection

**Download**:
```bash
python -m monai.bundle download "lung_nodule_ct_detection"
```

**Thông tin**:
- **Task**: Detection (bounding boxes), not segmentation
- **Training dataset**: LUNA16 (888 CT scans)
- **Output**: Nodule locations + probabilities
- **Model**: RetinaNet
- **Size**: 120 MB
- **Accuracy**: FROC 0.90

**Use case**: Pre-screening để tìm nodules

---

## 2. GitHub Models (Proven in Research)

### 2.1 LungMask ⭐⭐⭐ FASTEST & MOST ACCURATE

**Repository**: https://github.com/JoHof/lungmask

**Install**:
```bash
pip install git+https://github.com/JoHof/lungmask
```

**Usage**:
```python
from lungmask import mask
import SimpleITK as sitk

ct_scan = sitk.ReadImage("patient.nii.gz")
segmentation = mask.apply(ct_scan, model='R231')  # or 'R231CovidWeb'
sitk.WriteImage(segmentation, "lung_mask.nii.gz")
```

**Models**:
- **R231**: General lung segmentation
  - Training: 1,000+ CT scans
  - Accuracy: Dice **0.98**
  - Download: https://github.com/JoHof/lungmask/releases/download/v0.2.5/unet_r231-d5d2fc3d.pth

- **R231CovidWeb**: COVID-19 optimized
  - Training: COVID-19 + general CT scans
  - Accuracy: Dice 0.97
  - Download: https://github.com/JoHof/lungmask/releases/download/v0.2.5/unet_r231CovidWeb-0de78a7e.pth

**Specs**:
- **Size**: 30 MB each
- **Inference**: 5s/volume (GPU), 30s (CPU)
- **Architecture**: UNet

**Pros**:
- ✅ Cực kỳ chính xác (Dice 0.98)
- ✅ Nhanh
- ✅ SỬ DỤNG NGAY (không cần fine-tune)
- ✅ Đơn giản

**Cons**:
- ❌ Chỉ segment lung boundary (không segment lesions bên trong)

---

### 2.2 TotalSegmentator

**Repository**: https://github.com/wasserth/TotalSegmentator

**Install**:
```bash
pip install TotalSegmentator
```

**Usage**:
```bash
TotalSegmentator -i input.nii.gz -o output_folder/
```

**Thông tin**:
- **Output**: 117 anatomical structures
  - Lungs (upper, middle, lower lobes)
  - Trachea
  - Bronchi
  - Vessels
  - + 112 organs khác
- **Training dataset**: 1,228 CT scans
- **Accuracy**: Dice 0.85-0.95 (varies by organ)
- **Model**: nnU-Net
- **Size**: 400 MB
- **Inference**: ~45s/volume

**Use case**: Comprehensive organ segmentation

---

## 3. Hugging Face Models

### 3.1 SAM (Segment Anything Model)

**Repository**: https://huggingface.co/facebook/sam-vit-base

**Download**:
```python
from transformers import SamModel, SamProcessor

model = SamModel.from_pretrained("facebook/sam-vit-base")
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
```

**Thông tin**:
- **Architecture**: Vision Transformer (ViT)
- **Size**: 350 MB
- **Task**: Interactive segmentation (với prompts)
- **Training**: SA-1B dataset (11M images)

**Medical fine-tuning**:
- Repository: https://github.com/rekalantar/MedSegmentAnything_SAM_LungCT
- Fine-tuned cho lung CT với bounding box prompts

**Limitation**: SAM là 2D model, cần loop qua CT slices

---

### 3.2 CT-CLIP Foundation Model

**Dataset**: https://huggingface.co/datasets/ibrahimhamamci/CT-RATE

**Thông tin**:
- **Type**: Vision-language foundation model
- **Training data**: 25,692 CT volumes + radiology reports
- **Use case**: Feature extraction → add segmentation head
- **Size**: ~400 MB

**Workflow**:
```python
# 1. Load CT-CLIP for feature extraction
features = ct_clip.encode_image(ct_volume)

# 2. Add segmentation head
class SegmentationModel(nn.Module):
    def __init__(self):
        self.encoder = ct_clip  # Frozen
        self.decoder = UNetDecoder()

    def forward(self, x):
        features = self.encoder(x)
        return self.decoder(features)

# 3. Fine-tune decoder only
```

---

## 4. Download Script cho Air-Gapped Environment

```python
# pretrained-models/download_all.py

import os
import urllib.request
from monai.bundle import download

def download_monai_models():
    """Download MONAI models"""
    models = [
        "wholeBody_ct_segmentation",
        "covid19_lung_ct_segmentation",
        "lung_nodule_ct_detection"
    ]

    os.makedirs("./monai-models", exist_ok=True)

    for model in models:
        print(f"Downloading {model}...")
        download(name=model, bundle_dir="./monai-models")
        print(f"✓ {model} downloaded")

def download_lungmask():
    """Download LungMask weights"""
    urls = {
        "R231": "https://github.com/JoHof/lungmask/releases/download/v0.2.5/unet_r231-d5d2fc3d.pth",
        "R231CovidWeb": "https://github.com/JoHof/lungmask/releases/download/v0.2.5/unet_r231CovidWeb-0de78a7e.pth"
    }

    os.makedirs("./lungmask", exist_ok=True)

    for name, url in urls.items():
        filename = f"./lungmask/{name}.pth"
        print(f"Downloading LungMask {name}...")
        urllib.request.urlretrieve(url, filename)
        print(f"✓ Saved to {filename}")

def download_huggingface_sam():
    """Download SAM from Hugging Face"""
    from transformers import SamModel, SamProcessor

    print("Downloading SAM...")
    model = SamModel.from_pretrained("facebook/sam-vit-base")
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

    model.save_pretrained("./huggingface/sam-vit-base")
    processor.save_pretrained("./huggingface/sam-vit-base")
    print("✓ SAM saved to ./huggingface/sam-vit-base")

if __name__ == "__main__":
    print("="*60)
    print("Downloading All Pretrained Models")
    print("="*60)

    download_monai_models()
    download_lungmask()
    download_huggingface_sam()

    print("\n" + "="*60)
    print("✓ All downloads complete!")
    print("Total size: ~2 GB")
    print("\nPackage for offline use:")
    print("  tar -czf pretrained_models.tar.gz monai-models/ lungmask/ huggingface/")
    print("="*60)
```

---

## 5. Comparison Table

| Model | Source | Accuracy (Dice) | Speed | Size | Use Case |
|-------|--------|----------------|-------|------|----------|
| **LungMask R231** | GitHub | **0.98** | 5s | 30MB | Lung boundary ⭐ |
| **MONAI Whole Body** | MONAI | 0.85-0.90 | 30s | 500MB | Multi-organ |
| **MONAI COVID-19** | MONAI | 0.88 lung, 0.75 lesion | 10s | 80MB | Lung + lesions ⭐ |
| **TotalSegmentator** | GitHub | 0.85-0.95 | 45s | 400MB | 117 organs |
| **SAM Medical** | HF | 0.80 | 15s | 350MB | Interactive |

---

## 6. Recommendations

### Nếu cần NHANH và CHÍNH XÁC:
→ **LungMask R231** (Dice 0.98, 5s, không cần fine-tune)

### Nếu cần segment LESIONS:
→ **MONAI COVID-19** (fine-tune 20 epochs → Dice 0.88)

### Nếu cần MULTI-ORGAN (future-proof):
→ **MONAI Whole Body** (104 organs)

### Nếu cần RESEARCH/EXPERIMENTS:
→ **SAM** hoặc **CT-CLIP** (foundation models)

---

**Next**: [FINE_TUNING_GUIDE.md](FINE_TUNING_GUIDE.md)

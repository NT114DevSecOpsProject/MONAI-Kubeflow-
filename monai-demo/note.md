# MONAI Pretrained Models - Hướng Dẫn Chi Tiết

## Mục Lục
1. [MONAI có bao nhiêu model train sẵn?](#1-monai-có-bao-nhiêu-model-train-sẵn)
2. [Có thể thêm dữ liệu vào được không?](#2-có-thể-thêm-dữ-liệu-vào-được-không)
3. [Model lá lách được train như thế nào?](#3-model-lá-lách-được-train-như-thế-nào)
4. [So sánh: MONAI vs Hugging Face](#4-so-sánh-monai-vs-hugging-face)
5. [Xem thông tin ở đâu?](#5-xem-thông-tin-ở-đâu)

---

## 1. MONAI có bao nhiêu model train sẵn?

### ✅ Tổng cộng: 35 pretrained models trong MONAI Model Zoo

### 🫀 Organ Segmentation (Phân đoạn cơ quan)
- `spleen_ct_segmentation` ← Đang dùng
- `pancreas_ct_dints_segmentation` - Phân đoạn tụy
- `prostate_mri_anatomy` - Phân đoạn tuyến tiền liệt
- `wholeBody_ct_segmentation` - Phân đoạn toàn thân (104 cơ quan!)
- `renalStructures_UNEST_segmentation` - Phân đoạn thận
- `multi_organ_segmentation` - Đa cơ quan
- `pediatric_abdominal_ct_segmentation` - Bụng trẻ em

### 🧠 Brain Imaging (Hình ảnh não)
- `brats_mri_segmentation` - Khối u não (BraTS dataset)
- `wholeBrainSeg_Large_UNEST_segmentation` - Phân đoạn não

### 🫁 Lung/Chest (Phổi/Ngực)
- `lung_nodule_ct_detection` - Phát hiện nốt phổi
- `cxr_image_synthesis_latent_diffusion_model` - Tạo ảnh X-quang ngực

### 🔬 Pathology (Mô bệnh học)
- `pathology_tumor_detection` - Phát hiện khối u
- `pathology_nuclei_segmentation_classification` - Phân đoạn nhân tế bào
- `pathology_nuclick_annotation` - Công cụ annotation

### 👁️ Retina/Eye (Võng mạc)
- `retinalOCT_RPD_segmentation` - Phân đoạn võng mạc OCT

### 🎥 Endoscopy (Nội soi)
- `endoscopic_tool_segmentation` - Phân đoạn dụng cụ nội soi
- `endoscopic_inbody_classification` - Phân loại trong/ngoài cơ thể

### 🤖 AI Generation (Tạo ảnh AI)
- `maisi_ct_generative` - Tạo CT scan giả
- `brats_mri_generative_diffusion` - Tạo MRI não
- `brain_image_synthesis_latent_diffusion_model` - Latent diffusion cho não

### 🏆 Advanced Models
- `vista3d` - Model foundation 3D (universal segmentation)
- `vista2d` - Model foundation 2D
- `swin_unetr_btcv_segmentation` - BTCV dataset (đa cơ quan)

---

## 2. Có thể thêm dữ liệu vào được không?

### ✅ CÓ - Có 3 cách chính:

### 🔹 Cách 1: Fine-tuning (Đề xuất)

**Kịch bản:** Bệnh viện có 50 CT scan riêng của mình

**Các bước:**
1. Load pretrained model
2. Freeze các layer đầu (feature extraction)
3. Train lại các layer cuối trên 50 CT scan mới
4. Đánh giá trên validation set

**Kết quả:**
- Thời gian: 2-4 giờ training
- Dice tăng từ 0.97 → 0.98-0.99
- Tiết kiệm thời gian (không train từ đầu)
- Cần ít data hơn (50 mẫu thay vì 500+)
- Model học được đặc điểm riêng của bệnh viện

### 🔹 Cách 2: Transfer Learning

**Kịch bản:** Muốn phân đoạn cơ quan khác (ví dụ: gan)

**Các bước:**
1. Load spleen_ct_segmentation model
2. Thay đổi output layer (2 classes → 3 classes nếu cần)
3. Train trên dataset gan
4. Model học nhanh vì đã biết cấu trúc CT scan

**Kết quả:**
- Thời gian: 5-10 giờ training
- Nhanh hơn train từ đầu 5-10 lần

### 🔹 Cách 3: Incremental Learning

**Kịch bản:** Có data mới liên tục đến (bệnh viện mỗi tháng thêm 10 ca)

**Các bước:**
1. Start với pretrained model
2. Mỗi tháng train thêm trên 10 ca mới
3. Model cải thiện dần theo thời gian

**Lợi ích:**
- Model ngày càng tốt hơn với data riêng
- Không cần retrain toàn bộ

---

## 3. Model lá lách được train như thế nào?

### 📊 Dataset Chính Thức

**Tên:** Medical Segmentation Decathlon - **Task09_Spleen**

**Nguồn:**
- 🌐 Website: http://medicaldecathlon.com/
- 📄 Paper: Nature Communications (2022)
- 🏥 Bệnh viện: Memorial Sloan Kettering Cancer Center (New York, USA)

**Kích thước:**
```
Tổng số: 61 CT scans
├─ Training: 41 CT scans (với ground truth)
└─ Testing:  20 CT scans (public test set)
```

**Đặc điểm:**
- Modality: CT scan (Computed Tomography)
- Target: Spleen (lá lách)
- Format: NIfTI (.nii.gz)
- Challenge: Large-ranging foreground size (lá lách có kích thước thay đổi lớn)

### 🏗️ Architecture

**Model:** UNet 3D
- Input: 1 channel (CT scan)
- Output: 2 channels (background, spleen)
- Layers: [16, 32, 64, 128, 256] channels
- Parameters: 4.8 triệu
- Patch size: 96×96×96 voxels

### ⚙️ Training Configuration

**Hardware:**
- GPU: Ít nhất 12GB VRAM
- Actual GPU used: NVIDIA V100/A100

**Hyperparameters:**
- Epochs: 800 (actual best: ~1260 epochs)
- Optimizer: Novograd
- Learning rate: 0.002
- Loss function: DiceCELoss (50% Dice + 50% CrossEntropy)
- AMP (Automatic Mixed Precision): True
- Dataset Manager: CacheDataset

**Data Split:**
- Training: 32 CT scans
- Validation: 9 CT scans

### 🔄 Preprocessing (Training time)

1. `LoadImaged` - Load CT scan và ground truth
2. `EnsureChannelFirstd` - Đảm bảo channel đầu tiên
3. `Orientationd` - Chuẩn hóa hướng (RAS)
4. `Spacingd` - Resample spacing
5. `ScaleIntensityRanged` - Normalize intensity
6. `CropForegroundd` - Crop foreground
7. `RandCropByPosNegLabeld` - Random crop patches 96×96×96
8. `RandFlipd` - Random flip augmentation
9. `RandRotate90d` - Random rotation
10. `EnsureTyped` - Convert to tensor

### 📈 Performance Đạt Được

**Kết quả chính thức:**
- Mean Dice Score: **0.961 (96.1%)**
- Trên validation set (9 CT scans)
- Đạt Runner-up Award tại Medical Segmentation Decathlon 2018

**Training graphs:**
- Training loss: Giảm dần qua 1260 epochs
- Validation Dice: Đạt 0.96-0.97

### 📊 Dataset Statistics

**Training data (41 CT scans):**
- Patient ages: 18-90 years old
- Both genders
- Different spleen conditions:
  * Normal spleen
  * Splenomegaly (enlarged)
  * Post-trauma
  * Cancer patients

**Image characteristics:**
- Voxel spacing: Variable (0.5-1.0 mm)
- Image size: Variable (typically 512×512×[100-300] slices)
- Hounsfield Units: -1024 to 3071 HU
- File format: NIfTI (.nii.gz)

**Ground truth:**
- Manual annotation by expert radiologists
- Each voxel labeled: 0 (background) or 1 (spleen)
- Inter-rater agreement: >95% Dice

---

## 4. So sánh: MONAI vs Hugging Face

### 📊 Bảng So Sánh

| Tiêu chí | MONAI Model Zoo | Hugging Face |
|----------|-----------------|--------------|
| **Số lượng medical models** | 35 models chuyên sâu | ~10-15 models general |
| **Chuyên môn hóa** | 100% medical imaging | Đa lĩnh vực (NLP, CV, Audio) |
| **Data format** | NIfTI, DICOM (medical) | PNG, JPG, MP4 (general) |
| **Preprocessing** | HU normalization, spacing | RGB normalization |
| **Model architecture** | UNet 3D, UNETR (medical) | ViT, SAM (general vision) |

### 🤝 Medical Models trên Hugging Face

1. **facebook/sam-vit-huge** (Segment Anything Model)
   - General purpose segmentation
   - Có thể dùng cho medical (nhưng kém hơn MONAI)
   - Không hiểu HU values, spacing

2. **microsoft/swin-tiny-patch4-window7-224**
   - Vision Transformer
   - Có thể fine-tune cho medical classification
   - Nhưng thiếu medical-specific features

3. **google/vit-base-patch16-224**
   - ViT for image classification
   - Dùng cho phân loại bệnh (có/không có khối u)

### ✅ Có lấy model từ Hugging Face vào được không?

**CÓ - Nhưng phải làm thêm nhiều việc:**

**Ví dụ: Dùng SAM cho medical:**
1. Download SAM từ Hugging Face
2. Convert CT scan sang format SAM hiểu (PNG)
3. Mất medical metadata (spacing, orientation)
4. Inference với SAM
5. Post-process kết quả

**Kết quả:**
- Dice ~0.70-0.80 (kém hơn MONAI 0.97)
- Lý do: SAM không hiểu medical imaging specifics

### 📤 Upload MONAI model lên Hugging Face?

**CÓ THỂ - Để share với cộng đồng:**

1. Save MONAI model weights
2. Tạo Hugging Face repository
3. Upload model + config + preprocessing code
4. Tạo model card (README)
5. Người khác download và dùng

**URL:** huggingface.co/your-name/monai-spleen-ct

---

## 5. Xem thông tin ở đâu?

### 💻 Local Files (Trên máy)

**Thư mục model:**
```
E:\monai-kubeflow-demo\models\spleen_ct_segmentation\
├─ docs/
│  └─ README.md          ← Chi tiết đầy đủ về model
├─ configs/
│  ├─ metadata.json      ← Thông tin metadata
│  ├─ train.json         ← Config training
│  ├─ inference.json     ← Config inference
│  └─ evaluate.json      ← Config evaluation
└─ models/
   └─ model.pt           ← Model weights (220MB)
```

### 🌐 Online Resources

**a) MONAI Model Zoo:**
- URL: https://monai.io/model-zoo.html
- Tìm "Spleen CT Segmentation"
- Có link download, documentation, demo

**b) GitHub Repository:**
- URL: https://github.com/Project-MONAI/model-zoo/tree/dev/models/spleen_ct_segmentation
- Source code đầy đủ
- Config files
- Training scripts
- Version history

**c) Medical Segmentation Decathlon:**
- URL: http://medicaldecathlon.com/
- Download dataset gốc (Task09_Spleen.tar)
- Xem leaderboard
- Đọc paper: https://doi.org/10.1038/s41467-022-30695-9

**d) NVIDIA Clara:**
- URL: https://catalog.ngc.nvidia.com/orgs/nvidia/teams/monaitoolkit/models/monai_spleen_ct_segmentation
- NVIDIA hosting
- Pre-trained weights
- Deployment guides

### 📚 Papers và References

**Paper 1: Training Method**
- Title: "3D Semi-Supervised Learning with Uncertainty-Aware Multi-View Co-Training"
- Authors: Xia, Yingda, et al.
- Year: 2018
- Link: https://arxiv.org/abs/1811.12506
- Nội dung: Phương pháp training được dùng cho model này

**Paper 2: Architecture**
- Title: "Left-Ventricle Quantification Using Residual U-Net"
- Authors: Kerfoot E., et al.
- Year: 2019
- Link: https://doi.org/10.1007/978-3-030-12029-0_40
- Nội dung: UNet architecture với residual connections

**Paper 3: Dataset**
- Title: "A large annotated medical image dataset for the development and evaluation of segmentation algorithms"
- Authors: Medical Segmentation Decathlon Consortium
- Year: 2022
- Journal: Nature Communications
- Link: https://doi.org/10.1038/s41467-022-30695-9
- Nội dung: Giới thiệu toàn bộ Medical Decathlon dataset

---

## 📋 Tóm Tắt Thông Tin Model

| Thông tin | Giá trị |
|-----------|--------|
| **Dataset** | Medical Segmentation Decathlon Task09_Spleen |
| **Nguồn** | Memorial Sloan Kettering Cancer Center (USA) |
| **Kích thước** | 61 CT scans (41 train + 20 test) |
| **Training data** | 32 training + 9 validation |
| **Model** | UNet 3D (4.8M parameters) |
| **Training time** | 1260 epochs (~30 giờ trên V100) |
| **Dice Score** | 0.961 (96.1%) |
| **Award** | Runner-up - Medical Decathlon 2018 |
| **Xem thông tin** | models/spleen_ct_segmentation/docs/README.md |
| **Download dataset** | http://medicaldecathlon.com/ |

---

## 🎯 Khuyến Nghị Cho Ứng Dụng Bệnh Viện

### ✅ NÊN DÙNG:
- MONAI Model Zoo - 35 models chuyên medical imaging
- Pretrained sẵn, chỉ cần download
- Fine-tune được với data riêng
- Hỗ trợ DICOM, NIfTI, medical preprocessing

### 🤝 KẾT HỢP:
- MONAI (backend - AI model)
- Hugging Face Gradio (frontend - web interface)
- Hugging Face Spaces (hosting - deploy miễn phí)

### ❌ KHÔNG NÊN:
- Dùng Hugging Face general models (SAM, ViT) cho medical
- Lý do: Kém chính xác, mất medical metadata

### 🔄 Workflow Đề Xuất

```
[Bước 1] Download MONAI pretrained model
         ↓
[Bước 2] Test trên data thực tế bệnh viện
         ↓
[Bước 3] Nếu Dice > 0.90 → Dùng luôn
         Nếu Dice < 0.90 → Fine-tune với data riêng
         ↓
[Bước 4] Deploy với Gradio web interface
         ↓
[Bước 5] (Optional) Upload lên Hugging Face Spaces
         để bác sĩ dùng qua web browser
```

**Kết quả cuối cùng:**
- Model chính xác cao (Dice > 0.95)
- Web interface thân thiện
- Bác sĩ upload CT scan → Nhận kết quả phân đoạn
- Tiết kiệm 95% thời gian so với vẽ tay

---

**Last updated:** October 26, 2025

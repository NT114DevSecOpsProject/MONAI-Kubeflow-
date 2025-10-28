# Kubeflow Pipeline: Spleen Segmentation Flow

## Overview

Pipeline xử lý 4 bệnh nhân qua 4 bước: **Load Data → Preprocess → Inference → Visualize**

---

## Data Flow

```
PVC (/mnt/data)
├── test_data/
│   └── Task09_Spleen/
│       └── imagesTr/
│           ├── spleen_12.nii.gz  ← Source data
│           ├── spleen_19.nii.gz
│           ├── spleen_29.nii.gz
│           └── spleen_9.nii.gz
│
├── inputs/
│   └── week_current/
│       └── {patient_id}/
│           ├── imaging.nii.gz      ← Step 1: Load Data
│           └── preprocessed.pt     ← Step 2: Preprocess
│
└── outputs/
    └── week_current/
        └── {patient_id}/
            ├── segmentation.nii.gz  ← Step 3: Inference
            ├── axial.png            ← Step 4: Visualize
            ├── coronal.png
            └── sagittal.png
```

---

## Pipeline Steps

### Step 1: Load Data
**Component**: `load_data.py`
- **Input**: `/mnt/data/test_data/Task09_Spleen/imagesTr/{patient_id}.nii.gz`
- **Output**: `/mnt/data/inputs/week_current/{patient_id}/imaging.nii.gz`
- **Action**: Copy CT scan từ test dataset vào pipeline input directory

### Step 2: Preprocess
**Component**: `preprocess.py`
- **Input**: `/mnt/data/inputs/week_current/{patient_id}/imaging.nii.gz`
- **Output**: `/mnt/data/inputs/week_current/{patient_id}/preprocessed.pt`
- **Action**: Load NIfTI và apply MONAI normalization
  - Orientation: RAS
  - Spacing: (1.5, 1.5, 2.0)
  - Intensity scaling: [-175, 250] → [0, 1]
  - Crop foreground

### Step 3: Inference
**Component**: `inference.py`
- **Input**: `/mnt/data/inputs/week_current/{patient_id}/preprocessed.pt`
- **Model**: `/app/models/spleen_ct_segmentation/models/model.pt`
- **Output**: `/mnt/data/outputs/week_current/{patient_id}/segmentation.nii.gz`
- **Action**: Run MONAI UNet segmentation
  - Sliding window inference (96×96×96)
  - Softmax + threshold > 0.5

### Step 4: Visualize
**Component**: `visualize.py`
- **Input**:
  - CT: `/mnt/data/inputs/week_current/{patient_id}/imaging.nii.gz`
  - Mask: `/mnt/data/outputs/week_current/{patient_id}/segmentation.nii.gz`
- **Output**:
  - `/mnt/data/outputs/week_current/{patient_id}/axial.png`
  - `/mnt/data/outputs/week_current/{patient_id}/coronal.png`
  - `/mnt/data/outputs/week_current/{patient_id}/sagittal.png`
- **Action**: Create 3-view overlay images (CT + segmentation)

---

## Test Patients

Pipeline xử lý 4 bệnh nhân test set (chưa từng dùng trong training):
1. `spleen_12`
2. `spleen_19`
3. `spleen_29`
4. `spleen_9`

Nguồn: `data_split_mapping.json` → test split

---

## Execution Order

Mỗi patient chạy tuần tự qua 4 bước:

```
spleen_12: Load → Preprocess → Inference → Visualize
spleen_19: Load → Preprocess → Inference → Visualize
spleen_29: Load → Preprocess → Inference → Visualize
spleen_9:  Load → Preprocess → Inference → Visualize
```

Các bước trong mỗi patient có dependency:
- Preprocess depends on Load
- Inference depends on Preprocess
- Visualize depends on Inference

---

## Resource Requirements

- **PVC**: `data-pvc` mounted at `/mnt/data`
- **Docker Image**: `spleen-pipeline:v1`
- **Caching**: Disabled (set_caching_options(False))

---

## Output Files

Sau khi pipeline chạy xong, mỗi patient sẽ có:

```
/mnt/data/outputs/week_current/{patient_id}/
├── segmentation.nii.gz    (3D segmentation mask)
├── axial.png              (Axial view visualization)
├── coronal.png            (Coronal view visualization)
└── sagittal.png           (Sagittal view visualization)
```

---

## Next Steps

1. Compile pipeline: `python pipeline.py`
2. Build Docker image: `docker build -t spleen-pipeline:v1 .`
3. Upload `spleen_pipeline.yaml` to Kubeflow UI
4. Run pipeline and monitor results

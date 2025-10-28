# Task09_Spleen Dataset Split - Chi Tiết Đầy Đủ

## 📊 Tổng Quan

Dataset Task09_Spleen có **61 CT scans** tổng cộng:
- **41 files** trong `imagesTr/` có labels → Dùng cho training/validation/test
- **20 files** trong `imagesTs/` không có labels → Test set gốc của Medical Segmentation Decathlon

---

## 🎯 Cách MONAI Chia Dataset (Official)

### Theo File `train.json` (MONAI Model Zoo)

```json
"train": {
  "datalist": "list(range(0, -9))"  // File index 0 đến 31 (32 files)
},
"validate": {
  "datalist": "list(range(-9, 0))"   // File index 32 đến 40 (9 files)
}
```

**Khi sắp xếp alphabetically từ 41 files:**

| Split | Số lượng | Files (ví dụ) | Mục đích |
|-------|----------|---------------|----------|
| **Training** | 32 files | spleen_10, spleen_13, spleen_14, ... | Huấn luyện model |
| **Validation** | 9 files | spleen_17, spleen_18, spleen_2, ... | Early stopping, checkpoint selection |

---

## ⚠️ VẤN ĐỀ: Validation Set Bị "Nhìn" 800 Lần

Model được train với:
- **800 epochs**
- **Validation chạy mỗi epoch**
- → Validation set bị "nhìn" **800 lần** trong quá trình training!

**Hậu quả:**
- Model có thể overfit lên validation set
- Dice score trên validation set = **0.9752** (quá tốt, không đại diện cho dữ liệu mới)
- Không phản ánh chính xác performance trên dữ liệu unseen

---

## ✅ GIẢI PHÁP: Chia 3 Tập Rõ Ràng (Recommended)

File `data_split_mapping.json` định nghĩa chia 3 tập:

### 1. **Training Set** (32 files)
```
spleen_10.nii.gz, spleen_13.nii.gz, spleen_14.nii.gz, spleen_16.nii.gz,
spleen_20.nii.gz, spleen_21.nii.gz, spleen_22.nii.gz, spleen_24.nii.gz,
spleen_25.nii.gz, spleen_28.nii.gz, spleen_3.nii.gz,  spleen_31.nii.gz,
spleen_32.nii.gz, spleen_33.nii.gz, spleen_38.nii.gz, spleen_40.nii.gz,
spleen_41.nii.gz, spleen_44.nii.gz, spleen_45.nii.gz, spleen_46.nii.gz,
spleen_47.nii.gz, spleen_49.nii.gz, spleen_52.nii.gz, spleen_53.nii.gz,
spleen_56.nii.gz, spleen_59.nii.gz, spleen_6.nii.gz,  spleen_60.nii.gz,
spleen_61.nii.gz, spleen_62.nii.gz, spleen_63.nii.gz, spleen_8.nii.gz
```
**Mục đích:** Dùng để train model (KHÔNG được dùng để test)

### 2. **Validation Set** (5 files)
```
spleen_17.nii.gz, spleen_18.nii.gz, spleen_2.nii.gz,
spleen_26.nii.gz, spleen_27.nii.gz
```
**Mục đích:**
- Tuning hyperparameters
- Early stopping
- Checkpoint selection
- **KHÔNG được dùng để test cuối cùng**

### 3. **Test Set** (4 files) ⭐ DÙNG ĐỂ TEST MODEL
```
spleen_12.nii.gz
spleen_19.nii.gz
spleen_29.nii.gz
spleen_9.nii.gz
```
**Mục đích:**
- **NEVER seen during training**
- **Proper evaluation** (không bias)
- Đánh giá công bằng performance của model

---

## 🔍 Cách Kiểm Tra Model Đúng Cách

### ❌ SAI: Dùng Validation Set
```python
# SAI - Validation set đã bị "nhìn" 800 lần
test_files = ["spleen_17.nii.gz", "spleen_18.nii.gz", ...]
```

### ✅ ĐÚNG: Dùng Test Set (Unseen)
```python
# ĐÚNG - Test set chưa bao giờ thấy
test_files = [
    "spleen_12.nii.gz",
    "spleen_19.nii.gz",
    "spleen_29.nii.gz",
    "spleen_9.nii.gz"
]
```

### 📊 Expected Results

| Dataset | Dice Score | Lý do |
|---------|------------|-------|
| **Training Set** | 0.97+ | Model học từ data này → rất tốt |
| **Validation Set** | 0.9752 | Bị "nhìn" 800 lần → overfit |
| **Test Set (4 files)** | 0.88-0.92 | Unseen data → đánh giá công bằng |

---

## 🚀 Cách Chạy Test Đúng

### Script: `01_test_task09.py`
```python
# Script tự động load 4 test files từ data_split_mapping.json
# Chạy:
cd task09_spleen/scripts
python 01_test_task09.py
```

**Output:**
```
Test Set Results (Proper Evaluation)
  DICE SCORE:  0.8797 (averaged over 4 test samples)
  IoU SCORE:   0.8160
```

### Script: `02_visualize_spleen.py`
```python
# Tạo visualizations cho 4 test files
cd task09_spleen/scripts
python 02_visualize_spleen.py
```

**Output:**
- `spleen_result_2.png` - Main visualization
- `spleen_12_segmentation.png` - Test case 1
- `spleen_19_segmentation.png` - Test case 2
- `spleen_29_segmentation.png` - Test case 3
- `spleen_9_segmentation.png` - Test case 4

---

## 📁 Cấu Trúc Thư Mục

```
test_data/Task09_Spleen/
├── dataset.json              # Medical Segmentation Decathlon format
├── imagesTr/                 # 41 CT scans với labels
│   ├── spleen_2.nii.gz
│   ├── spleen_3.nii.gz
│   ├── ...
│   └── spleen_63.nii.gz
├── labelsTr/                 # 41 ground truth masks
│   ├── spleen_2.nii.gz
│   ├── spleen_3.nii.gz
│   ├── ...
│   └── spleen_63.nii.gz
└── imagesTs/                 # 20 unseen test images (no labels)
    ├── spleen_1.nii.gz
    ├── spleen_7.nii.gz
    ├── ...
    └── spleen_57.nii.gz
```

---

## 🎓 Nguyên Tắc Quan Trọng

### 1. ❌ KHÔNG BAO GIỜ
- Dùng training files để test
- Dùng validation files để đánh giá cuối cùng
- Mix training và test data

### 2. ✅ LUÔN LUÔN
- Giữ test set riêng biệt (unseen)
- Test trên dữ liệu chưa bao giờ thấy
- Report metrics trên test set, không phải validation

### 3. 📊 Hiểu Rõ Data Leakage
```
Training Set    →  Model học từ đây
Validation Set  →  Model "nhìn" gián tiếp qua early stopping
Test Set        →  Model CHƯA BAO GIỜ thấy → Đánh giá công bằng
```

---

## 🔗 Files Liên Quan

- `data_split_mapping.json` - Định nghĩa split (32/5/4)
- `task09_spleen/scripts/01_test_task09.py` - Script evaluation
- `task09_spleen/scripts/02_visualize_spleen.py` - Script visualization
- `models/spleen_ct_segmentation/configs/train.json` - MONAI training config

---

## 📝 Tóm Tắt Nhanh

| Câu Hỏi | Trả Lời |
|---------|---------|
| Model train trên files nào? | 32 files (spleen_10, 13, 14, 16, 20, ...) |
| Validation files là gì? | 5 files (spleen_17, 18, 2, 26, 27) |
| Test files để đánh giá? | **4 files (spleen_12, 19, 29, 9)** ⭐ |
| Tránh files nào khi test? | 32 training files + 5 validation files |
| Expected Dice trên test set? | ~0.88-0.92 (realistic, unbiased) |

**Kết luận:** Luôn dùng 4 test files (spleen_12, 19, 29, 9) để đánh giá model công bằng!

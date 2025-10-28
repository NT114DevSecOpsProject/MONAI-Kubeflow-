# TotalSegmentator - Analysis & Comparison

Complete analysis of MONAI model on **TotalSegmentator Small dataset** (different domain).

## 🎯 About TotalSegmentator

- **Source**: https://zenodo.org/records/8367169
- **Data**: Whole-body CT scans (102 cases)
- **Organs**: 117 different organs (we use spleen only)
- **Protocol**: Multiple hospitals, scanners, protocols

---

## ⚡ Quick Start

```bash
# Reorganize data (if needed)
cd scripts
python 02_reorganize_data.py

# Test on TotalSegmentator
python 02_test_totalseg.py

# Create comparison chart
python 01_compare_results.py

# Visualize results
python 02_visualize_cases.py

# View results
cat ../outputs/results.csv
```

---

## 📊 Expected Results

**Model Performance on TotalSegmentator**:
- Dice: **0.46** ❌ (Moderate - not great!)
- IoU: **0.37**
- Excellent cases: **30%** (many failures)
- Consistency: **73.0% variability** (Highly unstable!)

This is a **different domain** → Model struggles!

---

## 📁 Structure

```
totalsegmentator/
├── README.md                    ← You are here
├── scripts/
│   ├── 01_compare_results.py    ← Create comparison chart
│   ├── 02_reorganize_data.py    ← Prepare data
│   ├── 02_test_totalseg.py      ← Main evaluation
│   └── 02_visualize_cases.py    ← Create visualizations
└── outputs/                     ← Results
```

---

## 🚀 Scripts in Order

### 1. **02_reorganize_data.py** (Optional - run if data needs reorganizing)
Reorganize TotalSegmentator from nested to flat structure.

```bash
python scripts/02_reorganize_data.py
```

### 2. **02_test_totalseg.py** (Main evaluation)
Evaluate MONAI model on TotalSegmentator dataset.

```bash
python scripts/02_test_totalseg.py
```

**Output**:
- `outputs/results.csv` - Per-case metrics
- `outputs/summary.json` - Summary

### 3. **01_compare_results.py** (Create comparison)
Compare Task09_Spleen vs TotalSegmentator performance.

```bash
python scripts/01_compare_results.py
```

**Output**:
- `outputs/comparison_chart.png` - 6-panel comparison

### 4. **02_visualize_cases.py** (Create visualizations)
Generate 3-panel case visualizations.

```bash
python scripts/02_visualize_cases.py
```

**Output**:
- `outputs/visualizations/` - Individual case images

---

## 📊 Output Files

```
outputs/
├── results.csv              ← Per-case metrics
├── summary.json             ← Summary statistics
├── comparison_chart.png     ← 6-panel comparison
└── visualizations/          ← 3-panel case images
    ├── s0011.png
    ├── s0223.png
    └── ... (more cases)
```

---

## 🔑 Key Findings

Model achieves only **Dice 0.46** on TotalSegmentator because:

1. ❌ **Different data distribution**
   - Whole-body CT vs single-organ
   - Multiple protocols vs consistent

2. ❌ **Domain shift impact**
   - Different scanners & hospitals
   - Different preprocessing methods
   - Greater anatomical variation

3. ❌ **High variability**
   - 73% coefficient of variation
   - 3.8x MORE variable than Task09_Spleen
   - Performance unpredictable across cases

---

## 📈 Comparison Results

| Metric | Task09 | TotalSeg | Difference |
|--------|--------|----------|-----------|
| Dice | 0.88 | 0.46 | **-47.7%** |
| IoU | 0.82 | 0.37 | **-54.9%** |
| Excellent cases | 90% | 30% | **-60%** |
| Variability | 19.2% | 73.0% | **+3.8x** |

**Conclusion**: Domain shift is SIGNIFICANT!

---

## 💡 Recommendations

### ❌ Don't
Use MONAI model directly on TotalSegmentator without modification.

### ✅ Do
1. **Short-term** (1-2 weeks)
   - Fine-tune on TotalSegmentator
   - Expected Dice: 0.75-0.85

2. **Medium-term** (1-2 months)
   - Domain adaptation
   - Transfer learning optimization

3. **Long-term** (2+ months)
   - Retrain from scratch
   - Use full TotalSegmentator (1204 cases)
   - Expected Dice: 0.85-0.92

---

## 📝 Data Structure

### Before Reorganization
```
test_data/TotalSegmentator_small/
└── s0011/
    ├── ct.nii.gz
    └── segmentations/
        ├── spleen.nii.gz
        ├── liver.nii.gz
        └── ... (115 more organs)
```

### After Reorganization (Run 02_reorganize_data.py)
```
test_data/TotalSegmentator_small/
├── images/
│   ├── s0011_ct.nii.gz
│   ├── s0058_ct.nii.gz
│   └── ... (100 more)
└── labels/
    ├── s0011_spleen.nii.gz
    ├── s0058_spleen.nii.gz
    └── ... (100 more)
```

---

## 🎓 What This Teaches

1. **Pre-trained models have limitations**
   - Perfect on training domain
   - Fail on different domain

2. **Domain shift is critical**
   - Different acquisition protocols
   - Different patient populations
   - Different anatomical variations

3. **Validation on target domain is essential**
   - Always test before deployment
   - Results may vary significantly

---

**For detailed analysis, see: [../docs/RESULTS.md](../docs/RESULTS.md)**

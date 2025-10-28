# MONAI Spleen Segmentation - Domain Shift Analysis

Complete testing & analysis of MONAI pretrained spleen segmentation model on two datasets.

**Key Finding**: 47.7% performance drop due to domain shift (Dice 0.88 → 0.46)

---

## 🚀 Quick Start

```bash
# Read overview
cat docs/00_READ_ME.md

# Test on Task09_Spleen (training domain)
cd task09_spleen/scripts
python 01_test_task09.py

# Test on TotalSegmentator (test domain)
cd ../../totalsegmentator/scripts
python 02_test_totalseg.py

# Create comparison chart
python 01_compare_results.py

# View results
cat ../outputs/results.csv
```

---

## 📊 Results Summary

| Dataset | Dice | Status | Cases |
|---------|------|--------|-------|
| **Task09_Spleen** | 0.8797 | ✅ Excellent | 90% excellent |
| **TotalSegmentator** | 0.4597 | ❌ Moderate | 30% excellent |
| **Difference** | **-47.7%** | ⚠️ Domain shift | -60% |

---

## 📁 Folder Structure

```
monai-demo/
├── README.md                           ← You are here
├── docs/
│   ├── 00_READ_ME.md                  ← Start here
│   ├── RESULTS.md                     ← Key findings
│   └── USAGE.md                       ← How to run
├── task09_spleen/                     ← Task09_Spleen section
│   ├── README.md                      ← Task09 guide
│   ├── scripts/
│   │   └── 01_test_task09.py
│   └── outputs/
├── totalsegmentator/                  ← TotalSegmentator section
│   ├── README.md                      ← TotalSeg guide
│   ├── scripts/
│   │   ├── 01_compare_results.py
│   │   ├── 02_reorganize_data.py
│   │   ├── 02_test_totalseg.py
│   │   └── 02_visualize_cases.py
│   └── outputs/
├── test_data/                         ← Datasets
│   ├── Task09_Spleen/
│   └── TotalSegmentator_small/
└── [Images & other files]
```

---

## 📚 Documentation

| File | Purpose |
|------|---------|
| [docs/00_READ_ME.md](docs/00_READ_ME.md) | Overview & quick start |
| [docs/RESULTS.md](docs/RESULTS.md) | Detailed findings & analysis |
| [docs/USAGE.md](docs/USAGE.md) | How to run scripts |

---

## ⚡ Quick Commands

```bash
# Test TotalSegmentator
cd scripts/evaluation && python 02_test_totalseg.py

# Test Task09_Spleen
cd scripts/evaluation && python 01_test_task09.py

# Create comparison chart
cd scripts/visualization && python 01_compare_results.py

# View results
cat outputs/totalsegmentator_test/results.csv
```

---

## 🎯 What This Shows

This project demonstrates **domain shift** in machine learning:

1. **Model trains on Task09_Spleen**: Achieves Dice 0.88 ✅
2. **Model tested on TotalSegmentator**: Achieves Dice 0.46 ❌
3. **Conclusion**: 47.7% performance drop due to different data distribution

**Key insight**: Pre-trained models don't work on different domains without fine-tuning!

---

## 🔑 Key Findings

✅ **Task09_Spleen (Training Data)**
- Dice: 0.8797 ± 0.1690
- 90% excellent cases
- 19.2% variability (consistent)

❌ **TotalSegmentator (Test Data)**
- Dice: 0.4597 ± 0.3356
- 30% excellent cases
- 73.0% variability (inconsistent)

⚠️ **Domain Shift Impact**
- Dice drop: -47.7%
- Variability increase: +3.8x
- Excellent cases decrease: -60%

---

## 📊 Main Visualization

**File**: `outputs/comparison_task09_vs_totalseg.png`

6-panel comparison showing:
1. Dice score (0.88 vs 0.46)
2. IoU score
3. Variability (19% vs 73%)
4. Distribution histogram
5. Performance categories
6. Statistics table

---

## 📖 Script Organization

### `scripts/evaluation/` - Test Models
- `01_test_task09.py` - Test on Task09_Spleen (training data)
- `02_test_totalseg.py` - Test on TotalSegmentator

### `scripts/visualization/` - Create Charts
- `01_compare_results.py` - Create comparison chart
- `02_visualize_cases.py` - 3-panel case visualizations

### `scripts/data_prep/` - Prepare Data
- `01_download_data.py` - Download TotalSegmentator
- `02_reorganize_data.py` - Reorganize data structure

### `scripts/utilities/` - Utilities
- `01_demo.py` - Demo script

---

## 📈 Output Files

```
outputs/
├── comparison_task09_vs_totalseg.png    ← Main chart
├── task09_spleen_test/
│   ├── results.csv
│   └── summary.json
├── totalsegmentator_test/
│   ├── results.csv
│   ├── metrics_report.png
│   └── summary_statistics.png
└── totalsegmentator_visualizations_fixed/
    ├── s0011_segmentation.png (excellent)
    ├── s0310_segmentation.png (poor)
    └── ... (12 more cases)
```

---

## 🎓 Why This Matters

1. **Pre-trained models have limitations**
   - Work well on training domain
   - Fail on different domains

2. **Domain shift is critical challenge**
   - Different hospitals, scanners, protocols
   - Same organ, different image characteristics

3. **Always validate on target domain**
   - Before deployment
   - Before making decisions
   - Before claiming success

---

## 💡 Recommendations

❌ **Don't**: Use model directly on TotalSegmentator

✅ **Do**:
1. Fine-tune on TotalSegmentator (expected Dice 0.75-0.85)
2. Or retrain from scratch (expected Dice 0.85-0.92)
3. Always validate on your target domain

---

## 📞 Need Help?

1. **Getting started?**
   → Read [docs/00_READ_ME.md](docs/00_READ_ME.md)

2. **Want detailed results?**
   → See [docs/RESULTS.md](docs/RESULTS.md)

3. **How to run tests?**
   → Check [docs/USAGE.md](docs/USAGE.md)

4. **Find a script?**
   → Browse `scripts/` folder by purpose

---

## 🚀 Next Steps

1. Read [docs/00_READ_ME.md](docs/00_READ_ME.md)
2. Review [docs/RESULTS.md](docs/RESULTS.md)
3. Run a test: `cd scripts/evaluation && python test_totalsegmentator_spleen.py`
4. View chart: `outputs/comparison_task09_vs_totalseg.png`

---

**For complete details, see documentation in `docs/` folder**

Generated: 2025-10-27 | Updated: 2025-10-27

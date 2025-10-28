# MONAI Spleen Segmentation - Complete Analysis

## 🚀 Quick Start

```bash
# View key findings
cat docs/RESULTS.md

# Run test
cd scripts/evaluation
python test_totalsegmentator_spleen.py

# View output
open outputs/comparison_task09_vs_totalseg.png
```

---

## 📋 What This Project Does

Tests MONAI spleen model on two datasets to show **domain shift impact**:

| Dataset | Dice | Status |
|---------|------|--------|
| **Task09_Spleen** (training) | 0.88 | ✅ Excellent |
| **TotalSegmentator** (test) | 0.46 | ❌ Moderate |
| **Difference** | -47.7% | ⚠️ Domain shift |

---

## 📁 Folder Structure

```
monai-demo/
├── README.md                    ← You are here
├── docs/
│   ├── 00_READ_ME.md           ← Overview
│   ├── RESULTS.md              ← Key findings
│   └── USAGE.md                ← How to run
├── scripts/
│   ├── evaluation/             ← Test models
│   ├── visualization/          ← Create charts
│   ├── data_preparation/       ← Prep data
│   └── utilities/              ← Helper scripts
├── outputs/                    ← Results & charts
└── test_data/                  ← Datasets
```

---

## 📚 Documentation

- **[RESULTS.md](RESULTS.md)** - Key findings & analysis
- **[USAGE.md](USAGE.md)** - How to run scripts & reproduce results

---

## ⚡ Quick Commands

```bash
# Test TotalSegmentator
cd scripts/evaluation && python test_totalsegmentator_spleen.py

# Test Task09_Spleen
cd scripts/evaluation && python test_task09_spleen.py

# Create comparison chart
cd scripts/visualization && python compare_datasets.py

# View results
outputs/comparison_task09_vs_totalseg.png
```

---

## 🔑 Key Finding

**Domain shift = 47.7% performance drop!**

- Model: MONAI pretrained spleen segmentation (UNet 3D)
- Train data (Task09_Spleen): Dice **0.88** ✅
- Test data (TotalSegmentator): Dice **0.46** ❌
- Difference: **-0.42** (-47.7%)

---

## 💡 What This Teaches

1. Pre-trained models don't generalize to all domains
2. Always validate on target domain before deployment
3. Domain shift is a critical challenge in ML
4. Fine-tuning or retraining may be necessary

---

## 📞 Need Help?

- **Getting started?** → Read [RESULTS.md](RESULTS.md)
- **Want to run tests?** → See [USAGE.md](USAGE.md)
- **Find a script?** → Check `scripts/` folder
- **View results?** → Look in `outputs/`

---

Next: Read [RESULTS.md](RESULTS.md) for detailed findings!

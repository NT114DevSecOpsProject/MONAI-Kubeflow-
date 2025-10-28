# Task09_Spleen - Training & Evaluation

Testing MONAI spleen model on **Task09_Spleen dataset** (original training data).

## 🎯 About Task09_Spleen

- **Source**: Medical Segmentation Decathlon
- **Data**: Single-organ focused CT scans (spleen only)
- **Size**: 82 training cases, test set available
- **Protocol**: Well-controlled, consistent acquisition

---

## ⚡ Quick Start

```bash
# Test on Task09_Spleen
cd scripts
python 01_test_task09.py

# View results
cat ../outputs/results.csv
```

---

## 📊 Expected Results

**Model Performance on Task09_Spleen**:
- Dice: **0.88** ✅ (Excellent!)
- IoU: **0.82**
- Excellent cases: **90%**
- Consistency: **19.2% variability** (Very stable!)

This is the **training domain** → Model performs very well!

---

## 📁 Structure

```
task09_spleen/
├── README.md                    ← You are here
├── scripts/
│   ├── 01_test_task09.py        ← Evaluation script
│   └── 02_visualize_spleen.py   ← Visualization script
└── outputs/                     ← Results
    ├── results.csv              ← Metrics
    ├── summary.json             ← Summary
    ├── spleen_result_2.png      ← Main visualization
    └── spleen_*_segmentation.png ← Individual cases
```

---

## 🚀 Scripts in Order

### 1. `01_test_task09.py` - Evaluate Model
Evaluate MONAI model on Task09_Spleen dataset.

```bash
python scripts/01_test_task09.py
```

**Output**:
- `outputs/results.csv` - Per-case metrics (Dice, IoU)
- `outputs/summary.json` - Summary statistics

### 2. `02_visualize_spleen.py` - Create Visualizations
Generate 3-panel visualizations (Input CT | Probability Map | Predicted Mask).

```bash
python scripts/02_visualize_spleen.py
```

**Output**:
- `outputs/spleen_result_2.png` - Main summary visualization
- `outputs/spleen_*_segmentation.png` - Individual case visualizations (4 test samples)

---

## 📊 Output Files

```
outputs/
├── results.csv                  ← Per-case metrics (Dice, IoU)
├── summary.json                 ← Summary statistics
├── spleen_result_2.png          ← Main 3-panel visualization
├── spleen_12_segmentation.png   ← Test case 1
├── spleen_19_segmentation.png   ← Test case 2
├── spleen_29_segmentation.png   ← Test case 3
└── spleen_9_segmentation.png    ← Test case 4
```

---

## 🔑 Key Insight

Model achieves **Dice 0.88** on Task09_Spleen because:
1. ✅ Well-controlled data
2. ✅ Single-organ focused
3. ✅ Consistent CT protocol
4. ✅ Training domain (what model was trained on!)

---

## 📝 Notes

- Task09_Spleen is the **original training domain**
- Model knows this data distribution very well
- This is the baseline for comparison with TotalSegmentator
- See `docs/RESULTS.md` in parent folder for full comparison

---

**For detailed analysis, see: [../docs/RESULTS.md](../docs/RESULTS.md)**

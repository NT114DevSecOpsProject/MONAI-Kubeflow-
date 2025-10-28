# How to Use - Quick Reference

## 🚀 Quick Commands

### 1. Test on TotalSegmentator
```bash
cd scripts/evaluation
python 02_test_totalseg.py
# Output: outputs/totalsegmentator_test/results.csv
```

### 2. Test on Task09_Spleen
```bash
cd scripts/evaluation
python 01_test_task09.py
# Output: outputs/task09_spleen_test/results.csv
```

### 3. Create Comparison Chart
```bash
cd scripts/visualization
python 01_compare_results.py
# Output: outputs/comparison_task09_vs_totalseg.png
```

### 4. Generate Visualizations
```bash
cd scripts/visualization
python 02_visualize_cases.py
# Output: outputs/totalsegmentator_visualizations_fixed/
```

---

## 📁 Script Organization

### `scripts/data_prep/` - Data Preparation
- `01_download_data.py` - Download TotalSegmentator dataset
- `02_reorganize_data.py` - Reorganize data structure

### `scripts/evaluation/` - Model Testing
- `01_test_task09.py` - Test on Task09_Spleen (training data)
- `02_test_totalseg.py` - Test on TotalSegmentator

### `scripts/visualization/` - Charts & Plots
- `01_compare_results.py` - Create comparison chart
- `02_visualize_cases.py` - Create 3-panel case visualizations

### `scripts/utilities/` - Helper Scripts
- `01_demo.py` - Simple demonstration

---

## 📊 Output Files

### Evaluation Results
```
outputs/
├── task09_spleen_test/
│   ├── results.csv        ← Per-case metrics
│   └── summary.json       ← Aggregate metrics
│
└── totalsegmentator_test/
    ├── results.csv
    ├── summary.json
    ├── metrics_report.png
    └── summary_statistics.png
```

### Visualizations
```
outputs/
├── comparison_task09_vs_totalseg.png    ← Main chart!
└── totalsegmentator_visualizations_fixed/
    ├── s0011_segmentation.png
    ├── s0223_segmentation.png
    ├── ... (14 cases)
    ├── grid_5cases_fixed.png
    └── diagnostic_analysis.png
```

---

## 🔄 Common Workflows

### Workflow 1: Quick Test
```bash
# Test TotalSeg and view results
cd scripts/evaluation
python 02_test_totalseg.py
cat ../../outputs/totalsegmentator_test/results.csv
```

### Workflow 2: Compare Both Datasets
```bash
# Test both datasets
cd scripts/evaluation
python 01_test_task09.py
python 02_test_totalseg.py

# Create comparison
cd ../visualization
python 01_compare_results.py

# View result
open ../../outputs/comparison_task09_vs_totalseg.png
```

### Workflow 3: Full Analysis
```bash
# Prepare data (if needed)
cd scripts/data_prep
python 02_reorganize_data.py

# Run evaluations
cd ../evaluation
python 01_test_task09.py
python 02_test_totalseg.py

# Create visualizations
cd ../visualization
python 02_visualize_cases.py
python 01_compare_results.py

# View results
cd ../../outputs && ls -lh
```

---

## 📖 Reading Results

### CSV Files
```bash
# View as table
cat outputs/task09_spleen_test/results.csv

# Column meanings:
# case,dice,iou,status
# - case: Patient ID
# - dice: Dice score (0-1)
# - iou: Intersection over Union (0-1)
# - status: OK or error message
```

### JSON Files
```bash
# View summary metrics
cat outputs/task09_spleen_test/summary.json
# Contains: timestamps, device, model, metrics
```

### PNG Charts
```bash
# View visualizations (depends on your OS)
# Linux: feh outputs/comparison_task09_vs_totalseg.png
# Mac: open outputs/comparison_task09_vs_totalseg.png
# Windows: start outputs/comparison_task09_vs_totalseg.png
```

---

## 🔧 Configuration

### Change Test Size
Edit evaluation scripts, change this line:
```python
test_data = test_data[:10]  # Change 10 to your number
```

### Change Output Directory
Modify in evaluation scripts:
```python
OUTPUT_DIR = Path("./outputs/your_folder_name")
```

### Change Device (CPU/GPU)
```python
device = torch.device("cuda")  # Use GPU
device = torch.device("cpu")   # Use CPU (default)
```

---

## 📊 Understanding Metrics

| Metric | Range | Meaning |
|--------|-------|---------|
| **Dice** | 0-1 | Overlap between prediction and ground truth |
| **IoU** | 0-1 | Intersection over Union (stricter than Dice) |
| **Sensitivity** | 0-1 | True Positive Rate (spleen detection) |
| **Specificity** | 0-1 | True Negative Rate (background detection) |

**Clinical threshold**: Dice > 0.7 is considered good.

---

## 🐛 Troubleshooting

### Error: "cuda out of memory"
```python
# Use CPU instead
device = torch.device("cpu")
```

### Error: "File not found"
```bash
# Check if test_data exists
ls test_data/Task09_Spleen/imagesTr/
ls test_data/TotalSegmentator_small/images/
```

### Slow performance
- Close other programs
- Use GPU if available
- Test on smaller subset (reduce test_data[:10])

### Permission denied
```bash
chmod +x scripts/*/*.py
```

---

## 📝 Tips & Tricks

1. **Run in background**
   ```bash
   nohup python test_totalsegmentator_spleen.py > output.log &
   ```

2. **Monitor progress**
   ```bash
   tail -f output.log
   ```

3. **Save all output**
   ```bash
   python test_totalsegmentator_spleen.py | tee results.log
   ```

4. **Profile performance**
   ```bash
   time python test_totalsegmentator_spleen.py
   ```

---

## 📚 Next Steps

1. **Understand results**
   - Read [RESULTS.md](RESULTS.md)
   - View comparison chart

2. **Reproduce results**
   - Run test scripts
   - Compare with provided outputs

3. **Extend analysis**
   - Fine-tune model on TotalSegmentator
   - Try different thresholds
   - Add more evaluation metrics

4. **Production use**
   - Validate on your target domain
   - Fine-tune if necessary
   - Monitor performance in production

---

## ❓ FAQ

**Q: How long do tests take?**
A: ~10 seconds per case on CPU. 100+ cases on GPU = few minutes.

**Q: Can I modify the model?**
A: Yes, edit scripts. Default is MONAI pretrained UNet 3D.

**Q: How do I add more test data?**
A: Put in `test_data/` and update script paths.

**Q: What's the difference between Dice and IoU?**
A: Dice = 2×intersection/(sum), IoU = intersection/union. IoU is stricter.

---

**For detailed results, see [RESULTS.md](RESULTS.md)**

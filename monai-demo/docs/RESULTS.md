# Results & Analysis - Domain Shift Impact

## 📊 Executive Summary

Compared MONAI spleen model performance on:
- **Task09_Spleen**: Original training dataset
- **TotalSegmentator**: Different domain (whole-body CT)

**Result**: 47.7% performance drop due to domain shift.

---

## 🎯 Quantitative Results

### Task09_Spleen (Training Data)
```
Dice Score:    0.8797 ± 0.1690
IoU Score:     0.8160 ± 0.2057
Excellent:     9/10 cases (90%)
Variability:   19.2% (Very consistent)
Verdict:       ✅ EXCELLENT
```

### TotalSegmentator (Test Data)
```
Dice Score:    0.4597 ± 0.3356
IoU Score:     0.3682 ± 0.3239
Excellent:     3/10 cases (30%)
Variability:   73.0% (Highly variable!)
Verdict:       ❌ MODERATE
```

### Impact of Domain Shift
```
Dice Drop:           -47.7%  (0.88 → 0.46)
IoU Drop:            -54.9%  (0.82 → 0.37)
Variability Increase: +3.8x  (19% → 73%)
Excellent Cases:     -60%    (9 → 3 cases)
```

---

## 📈 Performance Distribution

### Task09_Spleen
- ✅ Excellent (>0.8): 9 cases
- ✓ Good (0.6-0.8): 1 case
- ⚠️ Fair (0.4-0.6): 0 cases
- ❌ Poor (<0.4): 0 cases

**Result**: 90% excellent - Model performs consistently well!

### TotalSegmentator
- ✅ Excellent (>0.8): 3 cases
- ✓ Good (0.6-0.8): 1 case
- ⚠️ Fair (0.4-0.6): 2 cases
- ❌ Poor (<0.4): 4 cases

**Result**: 30% excellent - Performance highly variable!

---

## 🔍 Why Domain Shift Happens

### Task09_Spleen Characteristics
- Single-organ focused (spleen only)
- Specific CT protocol
- Well-controlled acquisition
- Consistent preprocessing

### TotalSegmentator Characteristics
- Whole-body CT scans
- Multiple CT protocols
- Various hospitals & scanners
- Different preprocessing
- Greater anatomical variation

**Result**: Model learned Task09-specific patterns, not generalizable features.

---

## 💡 Key Insights

1. **Overfitting to Domain**
   - Model achieves Dice 0.88 on training domain
   - Performance drops to 0.46 on different domain
   - Shows poor generalization

2. **High Variability**
   - Task09: 19.2% variability (consistent)
   - TotalSeg: 73.0% variability (inconsistent)
   - 3.8x MORE variable on different domain!

3. **Confidence vs Accuracy**
   - Model has high confidence on both domains
   - But accuracy differs significantly
   - Confidence ≠ Reliability under domain shift

4. **Specificity is Strong**
   - Specificity: 0.9999 on both datasets
   - Model rarely makes false positives
   - Good at identifying non-spleen regions

5. **Sensitivity is Weak**
   - Task09: Sensitivity 0.89+ (good)
   - TotalSeg: Sensitivity 0.39 (poor)
   - Model misses many spleen voxels on TotalSeg

---

## 📊 Visualizations

### Main Comparison Chart
**File**: `outputs/comparison_task09_vs_totalseg.png`

6-panel visualization showing:
1. **Dice Score**: Green (0.88) vs Red (0.46)
2. **IoU Score**: Green (0.82) vs Red (0.37)
3. **Variability**: Green 19% vs Red 73% (huge difference!)
4. **Distribution**: Concentrated vs Spread
5. **Categories**: 90% excellent vs 30% excellent
6. **Stats Table**: Full quantitative summary

### Case Visualizations
**Folder**: `outputs/totalsegmentator_visualizations_fixed/`

3-panel format for each case:
- **Left**: Input CT Scan
- **Middle**: Probability Heatmap (normalized)
- **Right**: Predicted Mask (yellow=spleen, purple=background)

Examples:
- `s0011_segmentation.png` - Excellent (Dice 0.91)
- `s0310_segmentation.png` - Poor (Dice 0.03)

---

## 📉 Performance Breakdown

### By Dice Score Range
| Range | Task09 | TotalSeg | Interpretation |
|-------|--------|----------|-----------------|
| > 0.8 | 9 | 3 | Excellent cases |
| 0.6-0.8 | 1 | 1 | Good cases |
| 0.4-0.6 | 0 | 2 | Fair cases |
| < 0.4 | 0 | 4 | Poor cases |

### Variability
- **Task09**: 19.2% CV (consistent)
- **TotalSeg**: 73.0% CV (inconsistent)
- **Ratio**: 3.8x more variable

---

## ✅ What Works Well

1. **On Task09_Spleen**
   - Consistent performance (19% variability)
   - High Dice (0.88)
   - 90% excellent cases
   - Learned patterns specific to this domain

2. **General Properties**
   - High specificity (0.9999)
   - Few false positives
   - Stable across both domains
   - No crashes or errors

---

## ❌ What Doesn't Work

1. **On TotalSegmentator**
   - Low average Dice (0.46)
   - High variability (73%)
   - Only 30% excellent cases
   - Patterns don't transfer

2. **Generalization**
   - Model overfit to Task09
   - Doesn't handle domain shift
   - Sensitive to protocol differences
   - Not reliable on different data

---

## 🎓 Lessons Learned

### For ML Practitioners
1. **Always validate on target domain**
   - Pre-trained models ≠ production-ready
   - Domain shift is common and significant

2. **Domain-specific patterns matter**
   - Model learned Task09 specifics
   - Patterns don't generalize to other domains

3. **Variability is as important as accuracy**
   - High variability = unreliable
   - TotalSeg's 73% variability is problematic

4. **Confidence ≠ Accuracy**
   - Model confident on both domains
   - But accuracy differs by 47.7%

---

## 🔧 Solutions

### Short-term (1-2 weeks)
- Fine-tune on TotalSegmentator subset
- Expected Dice: 0.75-0.85

### Medium-term (1-2 months)
- Domain adaptation techniques
- Transfer learning optimization
- Multi-domain training

### Long-term (2+ months)
- Retrain from scratch on TotalSegmentator
- Use full dataset (1204 cases)
- Expected Dice: 0.85-0.92

---

## 📈 Recommendation

### ❌ DO NOT
Use this model directly on TotalSegmentator

### ✅ DO
1. Fine-tune on TotalSegmentator
2. Or retrain from scratch
3. Always validate on target domain
4. Monitor variability, not just mean accuracy

---

## 📊 Files

- **Results CSV**:
  - `outputs/task09_spleen_test/results.csv`
  - `outputs/totalsegmentator_test/results.csv`

- **Comparison Chart**: `outputs/comparison_task09_vs_totalseg.png`

- **Individual Cases**: `outputs/totalsegmentator_visualizations_fixed/`

---

## 🔗 Related

- **Task09_Spleen**: Medical Segmentation Decathlon
- **TotalSegmentator**: https://zenodo.org/records/8367169
- **MONAI Model Zoo**: https://github.com/Project-MONAI/model-zoo

---

**Generated**: 2025-10-27

**Key Takeaway**: Domain shift is real and significant. Always validate on your target domain!

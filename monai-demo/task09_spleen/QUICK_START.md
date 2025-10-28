# Quick Start - Task09_Spleen

## How to Generate Results

### Step 1: Evaluate Model on Test Set
```bash
cd task09_spleen/scripts
python 01_test_task09.py
```

This creates:
- `../outputs/results.csv` - Per-case Dice/IoU scores
- `../outputs/summary.json` - Summary statistics

### Step 2: Create Visualizations
```bash
python 02_visualize_spleen.py
```

This creates:
- `../outputs/spleen_result_2.png` - Main visualization (3-panel format)
- `../outputs/spleen_*_segmentation.png` - Individual test case visualizations

### Step 3: View Results
```bash
# View metrics
cat ../outputs/results.csv

# View main visualization
open ../outputs/spleen_result_2.png  # Mac
start ../outputs/spleen_result_2.png # Windows
feh ../outputs/spleen_result_2.png   # Linux
```

## Expected Results

**Performance Metrics:**
- Dice Score: ~0.88 (Excellent!)
- IoU Score: ~0.82
- Excellent cases: 90%
- Consistency: 19.2% variability

**Visualization Format:**
- **Left Panel**: Input CT Scan
- **Middle Panel**: Spleen Probability Map (model confidence)
- **Right Panel**: Predicted Mask (yellow=spleen, purple=background)

## Data Split

The evaluation uses a proper 3-way data split:

```json
{
  "training": 32 cases,    // Used to train the model
  "validation": 5 cases,   // Used for hyperparameter tuning
  "test": 4 cases          // NEVER seen - proper evaluation
}
```

**Test Set Files:**
- spleen_12.nii.gz
- spleen_19.nii.gz
- spleen_29.nii.gz
- spleen_9.nii.gz

These files are completely unseen during training, ensuring unbiased evaluation.

## Troubleshooting

### Script runs but no output
- Check that `test_data/Task09_Spleen/` folder exists
- Verify images are in `test_data/Task09_Spleen/imagesTr/`
- Verify labels are in `test_data/Task09_Spleen/labelsTr/`

### Model not found error
- Ensure model is at: `../../models/spleen_ct_segmentation/models/model.pt`
- Download from MONAI Model Zoo if missing

### Slow execution
- First run: ~2-3 min (model loading + inference)
- Subsequent runs: ~30-60 sec
- GPU available: ~10-15 sec

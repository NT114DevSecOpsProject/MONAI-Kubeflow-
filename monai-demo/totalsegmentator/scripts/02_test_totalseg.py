"""
Test MONAI Spleen Segmentation Model on TotalSegmentator Small Dataset
Chỉ test trên spleen (cơ quan duy nhất mà MONAI model được trained)
"""

import os
import sys
import json
import torch
import numpy as np
import nibabel as nib
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm
import pandas as pd

print("=" * 100)
print("TEST MONAI SPLEEN MODEL ON TOTALSEGMENTATOR SMALL DATASET")
print("=" * 100)

# Check environment
print("\n[STEP 1] Check Environment")
print("-" * 100)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch version: {torch.__version__}")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

try:
    from monai import __version__ as monai_version
    from monai.networks.nets import UNet
    from monai.transforms import (
        Compose, LoadImaged, EnsureChannelFirstd,
        Orientationd, Spacingd, ScaleIntensityRanged,
        CropForegroundd, EnsureTyped
    )
    from monai.inferers import sliding_window_inference
    print(f"MONAI version: {monai_version}")
    print("[OK] All libraries loaded")
except Exception as e:
    print(f"[ERROR] Failed to import: {e}")
    sys.exit(1)

# Setup paths
print("\n[STEP 2] Setup Paths")
print("-" * 100)

# Paths (relative to script location)
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
MODEL_PATH = PROJECT_ROOT / "models/spleen_ct_segmentation/models/model.pt"
DATA_DIR = PROJECT_ROOT / "test_data/TotalSegmentator_small"
OUTPUT_DIR = SCRIPT_DIR.parent / "outputs"
VIZ_DIR = OUTPUT_DIR / "visualizations"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VIZ_DIR.mkdir(parents=True, exist_ok=True)

if not MODEL_PATH.exists():
    print(f"[ERROR] Model not found: {MODEL_PATH}")
    sys.exit(1)

if not DATA_DIR.exists():
    print(f"[ERROR] Data dir not found: {DATA_DIR}")
    sys.exit(1)

print(f"[OK] Model: {MODEL_PATH}")
print(f"[OK] Data:  {DATA_DIR}")
print(f"[OK] Output: {OUTPUT_DIR}")

# Load model
print("\n[STEP 3] Load Model")
print("-" * 100)

model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm="batch",
)

checkpoint = torch.load(str(MODEL_PATH), map_location=device, weights_only=True)
model.load_state_dict(checkpoint, strict=False)
model = model.to(device)
model.eval()

num_params = sum(p.numel() for p in model.parameters())
print(f"[OK] Model loaded")
print(f"    Architecture: UNet 3D")
print(f"    Parameters: {num_params:,}")
print(f"    Device: {device}")

# Load data
print("\n[STEP 4] Load Dataset")
print("-" * 100)

images_dir = DATA_DIR / "images"
labels_dir = DATA_DIR / "labels"

image_files = sorted(images_dir.glob("*_ct.nii.gz"))
test_data = []

for image_file in image_files:
    patient_id = image_file.stem.replace("_ct.nii", "")
    spleen_label = labels_dir / f"{patient_id}_spleen.nii.gz"

    if spleen_label.exists():
        test_data.append({
            "image": str(image_file),
            "label": str(spleen_label),
            "case": patient_id
        })

print(f"[OK] Found {len(test_data)} cases with spleen labels")

# Limit to first 10 cases for speed
test_data = test_data[:10]
print(f"[INFO] Testing on first {len(test_data)} cases")
for i, data in enumerate(test_data[:3]):
    print(f"  {i+1}. {data['case']}")

# Setup transforms
print("\n[STEP 5] Setup Transforms")
print("-" * 100)

val_transforms = Compose([
    LoadImaged(keys=["image", "label"], image_only=False),
    EnsureChannelFirstd(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(
        keys=["image", "label"],
        pixdim=(1.5, 1.5, 2.0),
        mode=("bilinear", "nearest"),
    ),
    ScaleIntensityRanged(
        keys=["image"],
        a_min=-175,
        a_max=250,
        b_min=0.0,
        b_max=1.0,
        clip=True,
    ),
    CropForegroundd(keys=["image", "label"], source_key="image"),
    EnsureTyped(keys=["image", "label"]),
])

print("[OK] Transforms ready")

# Metric functions
print("\n[STEP 6] Setup Metric Functions")
print("-" * 100)

def compute_dice_score(pred: np.ndarray, target: np.ndarray) -> float:
    """Dice Score"""
    if np.sum(target) == 0 and np.sum(pred) == 0:
        return 1.0
    if np.sum(target) == 0 or np.sum(pred) == 0:
        return 0.0
    intersection = np.sum(pred * target)
    dice = 2.0 * intersection / (np.sum(pred) + np.sum(target))
    return float(dice)

def compute_iou(pred: np.ndarray, target: np.ndarray) -> float:
    """Intersection over Union (Jaccard)"""
    if np.sum(target) == 0 and np.sum(pred) == 0:
        return 1.0
    if np.sum(target) == 0 or np.sum(pred) == 0:
        return 0.0
    intersection = np.sum(pred * target)
    union = np.sum(pred | target)
    iou = intersection / union if union > 0 else 0.0
    return float(iou)

def compute_sensitivity(pred: np.ndarray, target: np.ndarray) -> float:
    """Sensitivity (Recall) - True positive rate"""
    tp = np.sum(pred * target)
    fn = np.sum(target * ~pred)
    if (tp + fn) == 0:
        return 0.0
    return tp / (tp + fn)

def compute_specificity(pred: np.ndarray, target: np.ndarray) -> float:
    """Specificity - True negative rate"""
    tn = np.sum((~pred) * (~target))
    fp = np.sum(pred * (~target))
    if (tn + fp) == 0:
        return 0.0
    return tn / (tn + fp)

print("[OK] Metrics: Dice, IoU, Sensitivity, Specificity")

# Run inference
print("\n[STEP 7] Run Inference")
print("-" * 100)

results = []
dice_scores = []
iou_scores = []
sensitivity_scores = []
specificity_scores = []

for idx, case_data in enumerate(tqdm(test_data, desc="Testing")):
    case_name = case_data['case']

    try:
        sample = val_transforms(case_data)
        image = sample["image"]
        label = sample["label"]

        image_batch = image.unsqueeze(0).to(device)
        label_np = label[0].cpu().numpy().astype(bool)

        with torch.no_grad():
            output = sliding_window_inference(
                inputs=image_batch,
                roi_size=(96, 96, 96),
                sw_batch_size=4,
                predictor=model,
                overlap=0.5,
                mode="gaussian",
                device=device,
            )

        probs = torch.softmax(output, dim=1)[0, 1].cpu().numpy()
        pred_mask = (probs > 0.5).astype(bool)

        dice = compute_dice_score(pred_mask, label_np)
        iou = compute_iou(pred_mask, label_np)
        sensitivity = compute_sensitivity(pred_mask, label_np)
        specificity = compute_specificity(pred_mask, label_np)

        result = {
            "case": case_name,
            "dice": dice,
            "iou": iou,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "avg_prob": float(np.mean(probs[pred_mask])) if np.sum(pred_mask) > 0 else 0.0,
            "status": "OK"
        }

        results.append(result)
        dice_scores.append(dice)
        iou_scores.append(iou)
        sensitivity_scores.append(sensitivity)
        specificity_scores.append(specificity)

    except Exception as e:
        result = {
            "case": case_name,
            "dice": -1,
            "iou": -1,
            "sensitivity": -1,
            "specificity": -1,
            "avg_prob": -1,
            "status": f"ERROR: {str(e)}"
        }
        results.append(result)

# Print results
print("\n" + "=" * 100)
print("EVALUATION RESULTS")
print("=" * 100)

valid_dice = [d for d in dice_scores if d >= 0]
valid_iou = [i for i in iou_scores if i >= 0]
valid_sens = [s for s in sensitivity_scores if s >= 0]
valid_spec = [s for s in specificity_scores if s >= 0]

print("\n[DICE SCORE]")
if valid_dice:
    print(f"  Mean:   {np.mean(valid_dice):.4f} +/- {np.std(valid_dice):.4f}")
    print(f"  Min:    {np.min(valid_dice):.4f}")
    print(f"  Max:    {np.max(valid_dice):.4f}")
    print(f"  Median: {np.median(valid_dice):.4f}")

print("\n[INTERSECTION OVER UNION]")
if valid_iou:
    print(f"  Mean:   {np.mean(valid_iou):.4f} +/- {np.std(valid_iou):.4f}")
    print(f"  Min:    {np.min(valid_iou):.4f}")
    print(f"  Max:    {np.max(valid_iou):.4f}")
    print(f"  Median: {np.median(valid_iou):.4f}")

print("\n[SENSITIVITY (True Positive Rate)]")
if valid_sens:
    print(f"  Mean:   {np.mean(valid_sens):.4f} +/- {np.std(valid_sens):.4f}")
    print(f"  Min:    {np.min(valid_sens):.4f}")
    print(f"  Max:    {np.max(valid_sens):.4f}")

print("\n[SPECIFICITY (True Negative Rate)]")
if valid_spec:
    print(f"  Mean:   {np.mean(valid_spec):.4f} +/- {np.std(valid_spec):.4f}")
    print(f"  Min:    {np.min(valid_spec):.4f}")
    print(f"  Max:    {np.max(valid_spec):.4f}")

successful = sum(1 for r in results if r["status"] == "OK")
print(f"\n[SUMMARY]")
print(f"  Successful: {successful}/{len(results)}")
print(f"  Failed: {len(results) - successful}/{len(results)}")

# Save results
print("\n[STEP 8] Save Results")
print("-" * 100)

results_df = pd.DataFrame(results)
csv_path = OUTPUT_DIR / "results.csv"
results_df.to_csv(csv_path, index=False)
print(f"[OK] Saved to: {csv_path}")

summary = {
    "timestamp": datetime.now().isoformat(),
    "device": str(device),
    "model": "MONAI spleen_ct_segmentation",
    "dataset": "TotalSegmentator Small (spleen only)",
    "num_cases": len(results),
    "successful": successful,
    "metrics": {
        "dice": {"mean": float(np.mean(valid_dice)) if valid_dice else 0},
        "iou": {"mean": float(np.mean(valid_iou)) if valid_iou else 0},
        "sensitivity": {"mean": float(np.mean(valid_sens)) if valid_sens else 0},
        "specificity": {"mean": float(np.mean(valid_spec)) if valid_spec else 0},
    }
}

json_path = OUTPUT_DIR / "summary.json"
with open(json_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"[OK] Saved to: {json_path}")

print("\n" + "=" * 100)
print("[SUCCESS] Test Complete!")
print("=" * 100)
print(f"\nOutput files:")
print(f"  {csv_path}")
print(f"  {json_path}")

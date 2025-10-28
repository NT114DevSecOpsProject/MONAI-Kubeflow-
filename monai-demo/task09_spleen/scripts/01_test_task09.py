"""
Test MONAI Spleen Model on Task09_Spleen (Original Training Dataset)
Compare with TotalSegmentator results
"""

import os
import torch
import numpy as np
import nibabel as nib
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import json

print("=" * 100)
print("TEST MONAI MODEL ON TASK09_SPLEEN (Original Training Dataset)")
print("=" * 100)

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from monai.networks.nets import UNet
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    Orientationd, Spacingd, ScaleIntensityRanged,
    CropForegroundd, EnsureTyped
)
from monai.inferers import sliding_window_inference

# Paths (relative to script location)
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
REPO_ROOT = PROJECT_ROOT.parent  # Go up to E:\monai-kubeflow-demo
MODEL_PATH = REPO_ROOT / "models/spleen_ct_segmentation/models/model.pt"
TASK09_DIR = PROJECT_ROOT / "test_data/Task09_Spleen"
OUTPUT_DIR = SCRIPT_DIR.parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load model
print("\n[STEP 1] Load Model")
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
print(f"[OK] Model loaded on {device}")

# Load Task09_Spleen data - ONLY TEST SET (unseen files)
print("\n[STEP 2] Load Task09_Spleen TEST SET (Unseen During Training)")
print("-" * 100)

# Load data split mapping to get proper test files
split_file = PROJECT_ROOT / "data_split_mapping.json"
if split_file.exists():
    with open(split_file) as f:
        split_mapping = json.load(f)
    test_files = split_mapping["splits"]["test"]["files"]
    print(f"[OK] Using data_split_mapping.json")
    print(f"[OK] Test set: 4 files (NEVER seen during training)")
else:
    # Fallback: use specific test files
    test_files = ["spleen_12.nii.gz", "spleen_19.nii.gz", "spleen_29.nii.gz", "spleen_9.nii.gz"]
    print(f"[WARNING] data_split_mapping.json not found, using default test files")

# Task09_Spleen structure: imagesTr, imagesTs, labelsTr
test_images_dir = TASK09_DIR / "imagesTr"
test_labels_dir = TASK09_DIR / "labelsTr"

test_data = []

print(f"\nTest Set Files (Unseen):")
for filename in test_files:
    image_file = test_images_dir / filename
    case_name = filename.replace('.nii.gz', '').replace('.nii', '')
    label_file = test_labels_dir / filename

    if image_file.exists() and label_file.exists():
        test_data.append({
            "image": str(image_file),
            "label": str(label_file),
            "case": case_name
        })
        print(f"  - {filename}")
    else:
        print(f"  [WARNING] File not found: {filename}")

print(f"\n[OK] Loaded {len(test_data)} test cases (UNSEEN - proper evaluation)")

if len(test_data) == 0:
    print("[WARNING] No test data found!")
    print(f"Expected structure:")
    print(f"  {TASK09_DIR}/imagesTr/spleen_*.nii.gz")
    print(f"  {TASK09_DIR}/labelsTr/spleen_*.nii.gz")

# Setup transforms
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
def compute_dice_score(pred: np.ndarray, target: np.ndarray) -> float:
    if np.sum(target) == 0 and np.sum(pred) == 0:
        return 1.0
    if np.sum(target) == 0 or np.sum(pred) == 0:
        return 0.0
    intersection = np.sum(pred * target)
    dice = 2.0 * intersection / (np.sum(pred) + np.sum(target))
    return float(dice)

def compute_iou(pred: np.ndarray, target: np.ndarray) -> float:
    if np.sum(target) == 0 and np.sum(pred) == 0:
        return 1.0
    if np.sum(target) == 0 or np.sum(pred) == 0:
        return 0.0
    intersection = np.sum(pred * target)
    union = np.sum(pred | target)
    iou = intersection / union if union > 0 else 0.0
    return float(iou)

# Run inference
print("\n[STEP 3] Run Inference on Task09_Spleen TEST SET")
print("-" * 100)

results = []
dice_scores = []
iou_scores = []

print(f"Testing on {len(test_data)} test cases (UNSEEN during training)...\n")

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

        result = {
            "case": case_name,
            "dice": dice,
            "iou": iou,
            "status": "OK"
        }

        results.append(result)
        dice_scores.append(dice)
        iou_scores.append(iou)

    except Exception as e:
        result = {
            "case": case_name,
            "dice": -1,
            "iou": -1,
            "status": f"ERROR: {str(e)}"
        }
        results.append(result)

# Print results
print("\n" + "=" * 100)
print("TASK09_SPLEEN TEST RESULTS")
print("=" * 100)

valid_dice = [d for d in dice_scores if d >= 0]
valid_iou = [i for i in iou_scores if i >= 0]

print("\n[DICE SCORE]")
if valid_dice:
    print(f"  Mean:   {np.mean(valid_dice):.4f} +/- {np.std(valid_dice):.4f}")
    print(f"  Min:    {np.min(valid_dice):.4f}")
    print(f"  Max:    {np.max(valid_dice):.4f}")
    print(f"  Median: {np.median(valid_dice):.4f}")

print("\n[IoU SCORE]")
if valid_iou:
    print(f"  Mean:   {np.mean(valid_iou):.4f} +/- {np.std(valid_iou):.4f}")
    print(f"  Min:    {np.min(valid_iou):.4f}")
    print(f"  Max:    {np.max(valid_iou):.4f}")
    print(f"  Median: {np.median(valid_iou):.4f}")

successful = sum(1 for r in results if r["status"] == "OK")
print(f"\n[SUMMARY]")
print(f"  Successful: {successful}/{len(results)}")
print(f"  Failed: {len(results) - successful}/{len(results)}")

# Save results
print("\n[STEP 4] Save Results")
print("-" * 100)

results_df = pd.DataFrame(results)
csv_path = OUTPUT_DIR / "results.csv"
results_df.to_csv(csv_path, index=False)
print(f"[OK] Saved to: {csv_path}")

summary = {
    "timestamp": datetime.now().isoformat(),
    "device": str(device),
    "model": "MONAI spleen_ct_segmentation",
    "dataset": "Task09_Spleen (Training dataset)",
    "num_cases": len(results),
    "successful": successful,
    "metrics": {
        "dice": {"mean": float(np.mean(valid_dice)) if valid_dice else 0},
        "iou": {"mean": float(np.mean(valid_iou)) if valid_iou else 0},
    }
}

json_path = OUTPUT_DIR / "summary.json"
with open(json_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"[OK] Saved to: {json_path}")

print("\n" + "=" * 100)
print("[SUCCESS] Task09_Spleen Test Complete!")
print("=" * 100)

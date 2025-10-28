"""
Visualize MONAI Model Predictions on Task09_Spleen Test Set
Creates 3-panel visualization: Input CT | Probability Map | Predicted Mask
"""

import torch
import numpy as np
import os
import sys
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

print("=" * 100)
print("VISUALIZE MONAI MODEL ON TASK09_SPLEEN TEST SET")
print("=" * 100)

# Paths (relative to script location)
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
REPO_ROOT = PROJECT_ROOT.parent  # Go up to E:\monai-kubeflow-demo
MODEL_PATH = REPO_ROOT / "models/spleen_ct_segmentation/models/model.pt"
DATA_DIR = PROJECT_ROOT / "test_data/Task09_Spleen"
OUTPUT_DIR = SCRIPT_DIR.parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load data split mapping
split_file = PROJECT_ROOT / "data_split_mapping.json"
if split_file.exists():
    with open(split_file) as f:
        split_mapping = json.load(f)
    test_files = split_mapping["splits"]["test"]["files"]
else:
    # Fallback: use last 4 files
    all_files = sorted([f for f in (DATA_DIR / "imagesTr").glob("*.nii.gz")])
    test_files = [f.name for f in all_files[-4:]]

print(f"\nTest Set Files (Unseen During Training):")
for f in test_files:
    print(f"  - {f}")

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")

# Load model
print(f"\n[STEP 1] Load Model")
print("-" * 100)

try:
    from monai.networks.nets import UNet

    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm="batch",
    )

    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()

    print(f"[OK] Model loaded from: {MODEL_PATH}")
    print(f"[OK] Parameters: {sum(p.numel() for p in model.parameters()):,}")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    sys.exit(1)

# Setup transforms
print(f"\n[STEP 2] Setup Preprocessing")
print("-" * 100)

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    Spacingd, Orientationd, ScaleIntensityRanged,
    CropForegroundd, EnsureTyped
)

transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Spacingd(
        keys=["image", "label"],
        pixdim=(1.5, 1.5, 2.0),
        mode=("bilinear", "nearest"),
    ),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    ScaleIntensityRanged(
        keys=["image"],
        a_min=-100,
        a_max=240,
        b_min=0.0,
        b_max=1.0,
        clip=True,
    ),
    CropForegroundd(keys=["image", "label"], source_key="image"),
    EnsureTyped(keys=["image", "label"]),
])

print("[OK] Preprocessing pipeline configured")

# Process each test sample
print(f"\n[STEP 3] Run Inference and Visualization")
print("-" * 100)

import matplotlib.pyplot as plt
from monai.inferers import sliding_window_inference

images_dir = DATA_DIR / "imagesTr"
labels_dir = DATA_DIR / "labelsTr"

for sample_idx, test_file in enumerate(test_files, 1):
    image_path = images_dir / test_file
    label_path = labels_dir / test_file

    if not image_path.exists():
        print(f"[WARNING] Image not found: {test_file}")
        continue

    print(f"\n{sample_idx}. Processing {test_file}...", end=" ")

    # Load and preprocess
    data = {"image": str(image_path), "label": str(label_path)}
    processed = transforms(data)

    image = processed["image"].unsqueeze(0).to(device)
    label = processed["label"].unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        outputs = sliding_window_inference(
            inputs=image,
            roi_size=(96, 96, 96),
            sw_batch_size=4,
            predictor=model,
            overlap=0.5,
            mode="gaussian",
            device=device,
        )

        # Post-processing
        probs = torch.softmax(outputs, dim=1)
        prob_spleen = probs[0, 1].cpu().numpy()  # Probability for spleen class
        predictions = torch.argmax(probs, dim=1, keepdim=True)
        pred_mask = predictions[0, 0].cpu().numpy()
        label_mask = label[0, 0].cpu().numpy()
        image_np = image[0, 0].cpu().numpy()

    # Find best slice (where spleen has most pixels)
    spleen_count = np.sum(label_mask, axis=(0, 1))
    if np.max(spleen_count) > 0:
        slice_idx = np.argmax(spleen_count)
    else:
        slice_idx = label_mask.shape[2] // 2

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'MONAI Pretrained Model - Spleen Segmentation\nData Source: Task09_Spleen Test Set',
                 fontsize=14, fontweight='bold')

    # Panel 1: Input CT Scan
    ax = axes[0]
    im = ax.imshow(image_np[:, :, slice_idx], cmap='gray')
    ax.set_title(f'Input CT Scan\n{test_file}\n(Slice {slice_idx}/{image_np.shape[2]})',
                 fontsize=11, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, label='Intensity')

    # Panel 2: Spleen Probability Map
    ax = axes[1]
    prob_slice = prob_spleen[:, :, slice_idx]
    # Normalize per-image for better visibility
    prob_norm = prob_slice / (np.max(prob_slice) + 1e-6)
    im = ax.imshow(prob_norm, cmap='hot')
    ax.set_title(f'Spleen Probability Map\n(avg: {np.mean(prob_slice):.3f})',
                 fontsize=11, fontweight='bold')
    ax.axis('off')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Probability')

    # Panel 3: Predicted Mask
    ax = axes[2]
    mask_display = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3))
    pred_slice = pred_mask[:, :, slice_idx]

    # Yellow = Spleen, Purple = Background
    spleen_pixels = pred_slice > 0.5
    background_pixels = ~spleen_pixels

    # Create RGB image
    mask_display[spleen_pixels, 0] = 1.0    # Red channel
    mask_display[spleen_pixels, 1] = 1.0    # Green channel (Red + Green = Yellow)
    mask_display[background_pixels, 0] = 0.6  # Red channel (Purple)
    mask_display[background_pixels, 2] = 1.0  # Blue channel

    ax.imshow(mask_display)
    spleen_volume = np.sum(pred_mask) / pred_mask.size * 100
    ax.set_title(f'Predicted Mask\n({spleen_volume:.2f}% spleen in volume)',
                 fontsize=11, fontweight='bold')
    ax.axis('off')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='yellow', label='Spleen (Predicted)'),
        Patch(facecolor='purple', label='Background')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

    plt.tight_layout()

    # Save
    output_file = OUTPUT_DIR / f"{test_file.replace('.nii.gz', '')}_segmentation.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"[OK] Saved: {output_file.name}")
    plt.close()

# Create summary grid for first sample
print(f"\n[STEP 4] Create Summary Visualization")
print("-" * 100)

if test_files:
    test_file = test_files[0]
    image_path = images_dir / test_file
    label_path = labels_dir / test_file

    data = {"image": str(image_path), "label": str(label_path)}
    processed = transforms(data)

    image = processed["image"].unsqueeze(0).to(device)
    label = processed["label"].unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = sliding_window_inference(
            inputs=image,
            roi_size=(96, 96, 96),
            sw_batch_size=4,
            predictor=model,
            overlap=0.5,
            mode="gaussian",
            device=device,
        )

        probs = torch.softmax(outputs, dim=1)
        prob_spleen = probs[0, 1].cpu().numpy()
        predictions = torch.argmax(probs, dim=1, keepdim=True)
        pred_mask = predictions[0, 0].cpu().numpy()
        label_mask = label[0, 0].cpu().numpy()
        image_np = image[0, 0].cpu().numpy()

    # Find best slice
    spleen_count = np.sum(label_mask, axis=(0, 1))
    slice_idx = np.argmax(spleen_count) if np.max(spleen_count) > 0 else label_mask.shape[2] // 2

    # Create main visualization (spleen_result_2.png style)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'MONAI Pretrained Model - Spleen Segmentation\nData Source: Task09_Spleen Test Set',
                 fontsize=14, fontweight='bold')

    # Input
    ax = axes[0]
    ax.imshow(image_np[:, :, slice_idx], cmap='gray')
    ax.set_title(f'Input CT Scan\n{test_file}\n(Slice {slice_idx}/{image_np.shape[2]})',
                 fontsize=11, fontweight='bold')
    ax.axis('off')

    # Probability
    ax = axes[1]
    prob_slice = prob_spleen[:, :, slice_idx]
    prob_norm = prob_slice / (np.max(prob_slice) + 1e-6)
    im = ax.imshow(prob_norm, cmap='hot')
    ax.set_title(f'Spleen Probability Map\n(avg: {np.mean(prob_slice):.3f})',
                 fontsize=11, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax)

    # Mask
    ax = axes[2]
    mask_display = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3))
    pred_slice = pred_mask[:, :, slice_idx]
    spleen_pixels = pred_slice > 0.5
    background_pixels = ~spleen_pixels
    mask_display[spleen_pixels, 0] = 1.0
    mask_display[spleen_pixels, 1] = 1.0
    mask_display[background_pixels, 0] = 0.6
    mask_display[background_pixels, 2] = 1.0
    ax.imshow(mask_display)
    spleen_volume = np.sum(pred_mask) / pred_mask.size * 100
    ax.set_title(f'Predicted Mask\n({spleen_volume:.2f}% spleen in volume)',
                 fontsize=11, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()
    output_file = OUTPUT_DIR / "spleen_result_2.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n[OK] Main visualization saved: {output_file.name}")
    plt.close()

print(f"\n[OK] Visualization complete!")
print(f"[OK] Output directory: {OUTPUT_DIR}")
print("\n" + "=" * 100)

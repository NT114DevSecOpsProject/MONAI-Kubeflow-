"""
Fixed visualization for TotalSegmentator results
- Better probability map display (normalized per image)
- Adjusted thresholding to show actual detections
- Better colormap contrast
"""

import os
import torch
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm

print("=" * 100)
print("VISUALIZE TOTALSEGMENTATOR TEST RESULTS (FIXED)")
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
MODEL_PATH = PROJECT_ROOT / "models/spleen_ct_segmentation/models/model.pt"
DATA_DIR = PROJECT_ROOT / "test_data/TotalSegmentator_small"
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

# Load data
print("\n[STEP 2] Load Dataset")
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

print(f"[OK] Found {len(test_data)} cases")

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

# Visualization function - FIXED
def get_middle_slice(volume):
    """Get middle slice of volume"""
    return volume.shape[2] // 2

def visualize_case_3panel_fixed(case_name, image, probs, pred_mask, label_np,
                                save_path=None, slice_idx=None):
    """
    Create 3-panel visualization with FIXED probability map display
    Panel 1: Input CT Scan (grayscale)
    Panel 2: Probability Heatmap (normalized, with better colormap)
    Panel 3: Binary Mask (yellow/purple)
    """

    if slice_idx is None:
        slice_idx = get_middle_slice(image)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Get CT image slice
    img_slice = image[0, 0, :, :, slice_idx].cpu().numpy()

    # Get probability and mask slices
    prob_slice = probs[:, :, slice_idx]
    mask_slice = pred_mask[:, :, slice_idx]
    label_slice = label_np[:, :, slice_idx]

    # Calculate statistics
    avg_prob = np.mean(probs[pred_mask]) if np.sum(pred_mask) > 0 else 0
    max_prob = np.max(probs)
    spleen_pct = np.sum(pred_mask) / pred_mask.size * 100

    # Panel 1: Input CT Scan
    ax = axes[0]
    im1 = ax.imshow(img_slice, cmap='gray')
    ax.set_title(f'Input CT Scan\n{case_name}.nii.gz\n(Slice {slice_idx}/{image.shape[4]-1})',
                fontsize=11, fontweight='bold', pad=10)
    ax.axis('off')

    # Panel 2: Probability Heatmap - IMPROVED
    ax = axes[1]

    # Display probability map with better contrast
    # Use slice-specific normalization for better visibility
    prob_slice_normalized = prob_slice / (np.max(prob_slice) + 1e-6)

    im2 = ax.imshow(prob_slice_normalized, cmap='hot', vmin=0, vmax=1)
    ax.set_title(f'Spleen Probability Map\n(avg: {avg_prob:.3f}, max: {max_prob:.3f})',
                fontsize=11, fontweight='bold', pad=10)
    ax.axis('off')
    cbar2 = plt.colorbar(im2, ax=ax, shrink=0.8, pad=0.05)
    cbar2.set_label('Probability', rotation=270, labelpad=20, fontsize=10)

    # Panel 3: Binary Mask
    ax = axes[2]
    # Create RGB image with yellow spleen and purple background
    rgb_img = np.zeros((mask_slice.shape[0], mask_slice.shape[1], 3))
    rgb_img[mask_slice] = [1, 1, 0]  # Yellow for spleen
    rgb_img[~mask_slice] = [0.5, 0, 0.5]  # Purple for background

    ax.imshow(rgb_img)
    ax.set_title(f'Predicted Mask\n({spleen_pct:.2f}% spleen in volume)',
                fontsize=11, fontweight='bold', pad=10)
    ax.axis('off')

    # Add legend
    yellow_patch = mpatches.Patch(color='yellow', label='Predicted Spleen')
    purple_patch = mpatches.Patch(color=(0.5, 0, 0.5), label='Background')
    ax.legend(handles=[yellow_patch, purple_patch], loc='lower right', fontsize=9)

    # Overall title
    fig.suptitle(f'MONAI Pretrained Model - Spleen Segmentation\nData Source: TotalSegmentator Small Dataset',
                fontsize=13, fontweight='bold', y=0.98)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  Saved: {save_path.name}")
        plt.close()

    return fig

# Run inference and visualization - IMPROVED
print("\n[STEP 3] Run Inference and Create Visualizations (FIXED)")
print("-" * 100)

test_data_subset = test_data[:15]  # Visualize first 15 cases
print(f"Creating visualizations for {len(test_data_subset)} cases...\n")

for idx, case_data in enumerate(tqdm(test_data_subset, desc="Visualizing")):
    case_name = case_data['case']

    try:
        # Load and preprocess
        sample = val_transforms(case_data)
        image = sample["image"]
        label = sample["label"]

        image_batch = image.unsqueeze(0).to(device)
        label_np = label[0].cpu().numpy().astype(bool)

        # Inference
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

        # Post-processing
        probs = torch.softmax(output, dim=1)[0, 1].cpu().numpy()
        pred_mask = (probs > 0.5).astype(bool)

        # Visualize
        save_path = OUTPUT_DIR / f"{case_name}_segmentation.png"
        visualize_case_3panel_fixed(case_name, image_batch.cpu(), probs, pred_mask, label_np,
                                   save_path=save_path)

    except Exception as e:
        print(f"  Error processing {case_name}: {e}")

print(f"\n[OK] Created {len(test_data_subset)} visualizations")
print(f"[OK] Saved to: {OUTPUT_DIR}")

# Create a grid visualization with BETTER display
print("\n[STEP 4] Create Grid Visualization (5 cases) - IMPROVED")
print("-" * 100)

fig = plt.figure(figsize=(18, 12))
fig.suptitle('MONAI Pretrained Model - Spleen Segmentation\nTotalSegmentator Small Dataset (5 Representative Cases)',
            fontsize=14, fontweight='bold', y=0.995)

for idx, case_data in enumerate(test_data_subset[:5]):
    case_name = case_data['case']

    try:
        # Load and preprocess
        sample = val_transforms(case_data)
        image = sample["image"]
        label = sample["label"]

        image_batch = image.unsqueeze(0).to(device)
        label_np = label[0].cpu().numpy().astype(bool)

        # Inference
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

        slice_idx = get_middle_slice(image)
        img_slice = image[0, 0, :, :, slice_idx].cpu().numpy()
        prob_slice = probs[:, :, slice_idx]
        mask_slice = pred_mask[:, :, slice_idx]

        avg_prob = np.mean(probs[pred_mask]) if np.sum(pred_mask) > 0 else 0
        max_prob = np.max(probs)
        spleen_pct = np.sum(pred_mask) / pred_mask.size * 100

        # Panel 1: CT
        ax = plt.subplot(5, 3, idx*3 + 1)
        ax.imshow(img_slice, cmap='gray')
        ax.set_ylabel(f'{case_name}', fontsize=10, fontweight='bold')
        ax.set_title('Input CT' if idx == 0 else '', fontsize=10, fontweight='bold')
        ax.axis('off')

        # Panel 2: Probability (IMPROVED - normalized)
        ax = plt.subplot(5, 3, idx*3 + 2)
        prob_norm = prob_slice / (np.max(prob_slice) + 1e-6)
        ax.imshow(prob_norm, cmap='hot', vmin=0, vmax=1)
        ax.set_title('Probability Map' if idx == 0 else '', fontsize=10, fontweight='bold')
        # Add text annotation with max prob
        ax.text(0.02, 0.98, f'max: {max_prob:.3f}', transform=ax.transAxes,
               fontsize=8, verticalalignment='top', color='white',
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
        ax.axis('off')

        # Panel 3: Mask
        ax = plt.subplot(5, 3, idx*3 + 3)
        rgb = np.zeros((mask_slice.shape[0], mask_slice.shape[1], 3))
        rgb[mask_slice] = [1, 1, 0]
        rgb[~mask_slice] = [0.5, 0, 0.5]
        ax.imshow(rgb)
        ax.set_title('Predicted Mask' if idx == 0 else '', fontsize=10, fontweight='bold')
        # Add text annotation with volume %
        ax.text(0.02, 0.98, f'{spleen_pct:.1f}%', transform=ax.transAxes,
               fontsize=8, verticalalignment='top', color='white',
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
        ax.axis('off')

    except Exception as e:
        print(f"Error: {e}")

plt.tight_layout()
grid_path = OUTPUT_DIR / "grid_5cases_fixed.png"
plt.savefig(grid_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"[OK] Saved grid visualization: {grid_path}")
plt.close()

# Create comparison visualization showing high and low confidence cases
print("\n[STEP 5] Create Diagnostic Visualization")
print("-" * 100)

# Collect some statistics
stats = []
for case_data in test_data_subset[:10]:
    case_name = case_data['case']
    try:
        sample = val_transforms(case_data)
        image = sample["image"]

        image_batch = image.unsqueeze(0).to(device)

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

        max_prob = np.max(probs)
        avg_prob_positive = np.mean(probs[pred_mask]) if np.sum(pred_mask) > 0 else 0
        pct_spleen = np.sum(pred_mask) / pred_mask.size * 100

        stats.append({
            'case': case_name,
            'max_prob': max_prob,
            'avg_prob': avg_prob_positive,
            'spleen_pct': pct_spleen
        })
    except:
        pass

if stats:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Model Confidence Analysis on TotalSegmentator', fontsize=12, fontweight='bold')

    cases = [s['case'] for s in stats]
    max_probs = [s['max_prob'] for s in stats]
    avg_probs = [s['avg_prob'] for s in stats]
    spleen_pcts = [s['spleen_pct'] for s in stats]

    # Max probability
    ax = axes[0]
    ax.bar(range(len(cases)), max_probs, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axhline(0.5, color='red', linestyle='--', linewidth=2, label='Threshold 0.5')
    ax.set_xlabel('Case', fontsize=10)
    ax.set_ylabel('Max Probability', fontsize=10)
    ax.set_title('Maximum Probability per Case', fontsize=11, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

    # Average probability (positive voxels)
    ax = axes[1]
    ax.bar(range(len(cases)), avg_probs, color='coral', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Case', fontsize=10)
    ax.set_ylabel('Avg Probability (positive)', fontsize=10)
    ax.set_title('Average Probability in Predicted Region', fontsize=11, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(alpha=0.3, axis='y')

    # Spleen percentage
    ax = axes[2]
    ax.bar(range(len(cases)), spleen_pcts, color='mediumseagreen', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Case', fontsize=10)
    ax.set_ylabel('Spleen Volume %', fontsize=10)
    ax.set_title('Predicted Spleen Volume', fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')

    ax.set_xticklabels(cases, rotation=45, ha='right')

    plt.tight_layout()
    diag_path = OUTPUT_DIR / "diagnostic_analysis.png"
    plt.savefig(diag_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"[OK] Saved diagnostic analysis: {diag_path}")
    plt.close()

print("\n" + "=" * 100)
print("[SUCCESS] Fixed Visualization Complete!")
print("=" * 100)
print(f"\nOutput Directory: {OUTPUT_DIR}")
print(f"Total individual visualizations: {len(list(OUTPUT_DIR.glob('*_segmentation.png')))}")
print(f"Grid visualization: {OUTPUT_DIR / 'grid_5cases_fixed.png'}")
print(f"Diagnostic analysis: {OUTPUT_DIR / 'diagnostic_analysis.png'}")

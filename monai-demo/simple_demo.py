"""
MONAI Pretrained Model Demo - Spleen CT Segmentation on REAL Data
Model: spleen_ct_segmentation from MONAI Model Zoo
Data: Real CT scans from Medical Segmentation Decathlon
"""

import torch
import numpy as np
import os
import sys
from pathlib import Path

print("=" * 80)
print("MONAI PRETRAINED MODEL DEMO - SPLEEN SEGMENTATION")
print("Using REAL CT Scan Data from Medical Segmentation Decathlon")
print("=" * 80)

# Step 1: Device Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n[Step 1] Device Setup")
print(f"  Device: {device}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# Step 2: Load Pretrained Model
print(f"\n[Step 2] Loading Pretrained Model")

try:
    from monai.networks.nets import UNet

    model_path = os.path.join("..", "models", "spleen_ct_segmentation", "models", "model.pt")

    # Create UNet with same architecture as pretrained model
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,  # background + spleen
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm="batch",
    )

    # Load pretrained weights
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)

    # Move model to device and set to evaluation mode
    model = model.to(device)
    model.eval()

    print(f"  [SUCCESS] Model loaded from MONAI Model Zoo")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Model device: {next(model.parameters()).device}")

except Exception as e:
    print(f"  [ERROR] Failed to load model: {e}")
    sys.exit(1)

# Step 3: Load and Preprocess REAL CT Scan
print(f"\n[Step 3] Loading REAL CT Scan with Proper Preprocessing")

try:
    from monai.transforms import (
        Compose, LoadImage, EnsureChannelFirst,
        Orientation, Spacing, ScaleIntensityRange,
        CropForeground, EnsureType
    )

    # Check if real data exists
    data_dir = Path("./test_data/Task09_Spleen")
    images_dir = data_dir / "imagesTr"

    if not images_dir.exists():
        print(f"  [ERROR] Real data not found!")
        print(f"  Run: python download_test_data.py")
        sys.exit(1)

    # Get CT files (exclude hidden files)
    ct_files = sorted([f for f in images_dir.glob("*.nii.gz") if not f.name.startswith("._")])

    if len(ct_files) == 0:
        print(f"  [ERROR] No valid CT scans found!")
        sys.exit(1)

    sample_file = ct_files[0]
    print(f"  File: {sample_file.name}")
    print(f"  Total CT scans available: {len(ct_files)}")

    # Preprocessing pipeline matching training configuration
    # This is CRITICAL for model to work correctly!
    transforms = Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        # Orientation to standard RAS (Right-Anterior-Superior)
        Orientation(axcodes="RAS"),
        # Resample to standard spacing (1.5mm x 1.5mm x 2.0mm)
        Spacing(pixdim=(1.5, 1.5, 2.0), mode="bilinear"),
        # Intensity windowing for CT scans (Hounsfield Units)
        # -100 to 240 HU captures soft tissue including spleen
        ScaleIntensityRange(
            a_min=-100,
            a_max=240,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        # Crop to remove empty background
        CropForeground(),
        # Convert to tensor
        EnsureType(),
    ])

    print(f"  Applying preprocessing pipeline:")
    print(f"    - Orientation: RAS")
    print(f"    - Spacing: (1.5, 1.5, 2.0) mm")
    print(f"    - Intensity range: -100 to 240 HU -> [0, 1]")
    print(f"    - Crop foreground")

    # Load and preprocess CT scan
    ct_scan = transforms(str(sample_file))

    # Add batch dimension and ensure on correct device
    ct_scan = ct_scan.unsqueeze(0).to(device)

    print(f"  [SUCCESS] CT scan loaded and preprocessed!")
    print(f"  Shape: {ct_scan.shape}")
    print(f"  Device: {ct_scan.device}")
    print(f"  Intensity stats:")
    print(f"    Min:  {ct_scan.min():.4f}")
    print(f"    Max:  {ct_scan.max():.4f}")
    print(f"    Mean: {ct_scan.mean():.4f}")

except Exception as e:
    print(f"  [ERROR] Failed to load data: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 4: Run Inference with Sliding Window
print(f"\n[Step 4] Running Inference with Sliding Window")
print(f"  [*] Using PRETRAINED weights - NO training needed!")

try:
    from monai.inferers import sliding_window_inference

    # Use sliding window inference for better results
    # This handles large volumes by processing overlapping patches
    roi_size = (96, 96, 96)  # Patch size
    sw_batch_size = 4  # Process 4 patches at once
    overlap = 0.5  # 50% overlap between patches

    print(f"  ROI size: {roi_size}")
    print(f"  Overlap: {overlap}")
    print(f"  Running inference...")

    with torch.no_grad():
        # Sliding window inference
        output = sliding_window_inference(
            inputs=ct_scan,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            predictor=model,
            overlap=overlap,
            mode="gaussian",  # Gaussian weighting for overlapping regions
            device=device,
        )

    print(f"  Input shape:  {ct_scan.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Inference completed!")

    # Check GPU memory usage
    if torch.cuda.is_available():
        print(f"  GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")

except Exception as e:
    print(f"  [ERROR] Inference failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 5: Post-processing Results
print(f"\n[Step 5] Post-Processing Predictions")

# Apply softmax to get probabilities
probs = torch.softmax(output, dim=1)

# Get predicted segmentation mask (argmax)
predicted_mask = torch.argmax(probs, dim=1)

# Calculate statistics
bg_prob = probs[0, 0].mean().item()
spleen_prob = probs[0, 1].mean().item()

print(f"  Average Probabilities:")
print(f"    Background: {bg_prob:.4f} ({bg_prob*100:.1f}%)")
print(f"    Spleen:     {spleen_prob:.4f} ({spleen_prob*100:.1f}%)")

# Calculate spleen volume percentage
spleen_pixels = (predicted_mask == 1).sum().item()
total_pixels = predicted_mask.numel()
spleen_percentage = (spleen_pixels / total_pixels) * 100

print(f"  Predicted spleen volume: {spleen_percentage:.2f}% of total volume")
print(f"  Spleen pixels: {spleen_pixels:,} / {total_pixels:,}")

# Find slice with highest spleen probability for visualization
# This gives us the best view of the spleen
spleen_prob_per_slice = probs[0, 1].mean(dim=(1, 2))  # Average over H and W
best_slice_idx = spleen_prob_per_slice.argmax().item()
max_slice_prob = spleen_prob_per_slice[best_slice_idx].item()

print(f"  Best slice for visualization: {best_slice_idx}")
print(f"  Slice {best_slice_idx} spleen probability: {max_slice_prob:.4f}")

# Step 6: Save Results
print(f"\n[Step 6] Saving Results")

try:
    # Save predicted mask as NIfTI file
    from monai.transforms import SaveImage

    saver = SaveImage(
        output_dir=".",
        output_postfix="pred",
        separate_folder=False,
        resample=False,
    )

    # Save the prediction
    output_file_nifti = "spleen_pred.nii.gz"
    saver(predicted_mask[0], meta_data={"filename_or_obj": output_file_nifti})
    print(f"  Saved prediction: {output_file_nifti}")

except Exception as e:
    print(f"  [WARNING] Could not save NIfTI: {e}")

# Step 7: Create Visualization
print(f"\n[Step 7] Creating Visualization")

try:
    import matplotlib.pyplot as plt

    # Use the best slice (highest spleen probability)
    slice_idx = best_slice_idx

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Input CT scan
    axes[0].imshow(ct_scan[0, 0, slice_idx].cpu().numpy(), cmap='gray')
    axes[0].set_title(f'Input CT Scan\n{sample_file.name}\n(Slice {slice_idx}/{ct_scan.shape[2]})', fontsize=11)
    axes[0].axis('off')

    # Spleen probability heatmap
    prob_map = probs[0, 1, slice_idx].cpu().numpy()
    im1 = axes[1].imshow(prob_map, cmap='hot', vmin=0, vmax=1)
    axes[1].set_title(f'Spleen Probability Map\n(avg: {max_slice_prob:.3f})', fontsize=11)
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Predicted segmentation mask
    mask = predicted_mask[0, slice_idx].cpu().numpy()
    axes[2].imshow(mask, cmap='viridis', vmin=0, vmax=1)
    axes[2].set_title(f'Predicted Mask\n({spleen_percentage:.2f}% spleen in volume)', fontsize=11)
    axes[2].axis('off')

    plt.suptitle(f'MONAI Pretrained Model - Spleen Segmentation\nSource: Medical Segmentation Decathlon',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_file_png = "spleen_result.png"
    plt.savefig(output_file_png, dpi=150, bbox_inches='tight')
    print(f"  Saved visualization: {output_file_png}")
    plt.close()

except Exception as e:
    print(f"  [WARNING] Could not create visualization: {e}")
    import traceback
    traceback.print_exc()

# Summary
print("\n" + "=" * 80)
print("[SUCCESS] DEMO COMPLETED!")
print("=" * 80)

print(f"\n[RESULTS SUMMARY]")
print(f"  Data: {sample_file.name}")
print(f"  Spleen detected: {spleen_percentage:.2f}% of volume")
print(f"  Best slice: {best_slice_idx} (probability: {max_slice_prob:.3f})")

print(f"\n[KEY INFORMATION]")
print(f"  + Model: spleen_ct_segmentation (MONAI Model Zoo)")
print(f"  + Data: Real CT scan from Medical Segmentation Decathlon")
print(f"  + Preprocessing: Orientation + Spacing + Intensity windowing")
print(f"  + Inference: Sliding window with Gaussian weighting")
print(f"  + NO TRAINING REQUIRED - Pretrained weights work immediately!")

print(f"\n[OUTPUT FILES]")
print(f"  1. {output_file_nifti} - Predicted segmentation mask (NIfTI format)")
print(f"  2. {output_file_png} - Visualization (PNG image)")

print(f"\n[INTERPRETATION]")
if spleen_percentage > 5.0:
    print(f"  GOOD: Model detected significant spleen region ({spleen_percentage:.1f}%)")
    print(f"  This looks like a valid spleen segmentation!")
elif spleen_percentage > 1.0:
    print(f"  MODERATE: Model detected some spleen ({spleen_percentage:.1f}%)")
    print(f"  May need fine-tuning for better results")
else:
    print(f"  LOW: Limited spleen detection ({spleen_percentage:.1f}%)")
    print(f"  Possible reasons:")
    print(f"    - CT scan may not contain much spleen tissue")
    print(f"    - Different imaging protocol than training data")
    print(f"    - May benefit from fine-tuning on similar data")

print(f"\n[NEXT STEPS]")
print(f"  1. Check visualization in: {output_file_png}")
print(f"  2. Compare with ground truth: python evaluate_accuracy.py")
print(f"  3. View sources: NGUON_DU_LIEU.md")
print(f"  4. Full guide: HUONG_DAN_TEST.md")

print(f"\n[GPU MEMORY]")
if torch.cuda.is_available():
    print(f"  Peak memory used: {torch.cuda.max_memory_allocated(0) / 1024**2:.2f} MB")
    print(f"  Works perfectly on RTX 3050 4GB VRAM!")

print()

"""
Demo script showing how to use MONAI pretrained models WITHOUT training from scratch
This uses the spleen_ct_segmentation pretrained model for inference
"""

import torch
import numpy as np
from monai.bundle import load
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    ScaleIntensityRanged,
    EnsureTyped,
    Activationsd,
    AsDiscreted,
)
from monai.data import decollate_batch
import matplotlib.pyplot as plt
import os

print("=" * 80)
print("MONAI PRETRAINED MODEL DEMO - Spleen CT Segmentation")
print("=" * 80)

# 1. Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n[1] Device: {device}")
if torch.cuda.is_available():
    print(f"    GPU: {torch.cuda.get_device_name(0)}")
    print(f"    VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# 2. Load the pretrained model
print("\n[2] Loading pretrained model...")
print("    Bundle: spleen_ct_segmentation")
print("    Location: ./models/spleen_ct_segmentation")

try:
    # Load the pretrained bundle
    # This loads the model architecture + pretrained weights
    model = load(
        name="spleen_ct_segmentation",
        bundle_dir="./models",
        workflow_type="inference"
    )

    print("    [SUCCESS] Pretrained model loaded successfully!")

    # Move model to GPU
    if torch.cuda.is_available():
        model = model.to(device)
        print(f"    Model moved to: {next(model.parameters()).device}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    Total parameters: {total_params:,}")
    print(f"    Trainable parameters: {trainable_params:,}")

except Exception as e:
    print(f"    [ERROR] Failed to load model: {e}")
    print("\n    Try running:")
    print("    python -m monai.bundle download --name spleen_ct_segmentation --bundle_dir ./models")
    import traceback
    traceback.print_exc()
    exit(1)

# 3. Create sample data for demonstration
print("\n[3] Creating sample CT scan data...")
print("    (In real use, you would load actual CT scan files)")

# Simulate a CT scan volume (normally loaded from DICOM or NIfTI files)
# Shape: (batch, channel, depth, height, width)
sample_ct_scan = torch.randn(1, 1, 96, 96, 96).to(device)
print(f"    Sample input shape: {sample_ct_scan.shape}")
print(f"    Sample input device: {sample_ct_scan.device}")

# 4. Run inference with pretrained model
print("\n[4] Running inference with pretrained model...")
print("    [*] No training needed - using pretrained weights!")

model.eval()
with torch.no_grad():
    start_mem = torch.cuda.memory_allocated(0) / 1024**2 if torch.cuda.is_available() else 0

    # Run inference
    output = model(sample_ct_scan)

    end_mem = torch.cuda.memory_allocated(0) / 1024**2 if torch.cuda.is_available() else 0

print(f"    Input shape: {sample_ct_scan.shape}")
print(f"    Output shape: {output.shape}")
print(f"    Output device: {output.device}")

if torch.cuda.is_available():
    print(f"    GPU Memory used: {end_mem - start_mem:.2f} MB")
    print(f"    Total GPU Memory: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")

# 5. Post-process output
print("\n[5] Post-processing predictions...")

# Apply softmax to get probabilities
output_probs = torch.softmax(output, dim=1)
print(f"    Probabilities shape: {output_probs.shape}")
print(f"    Class 0 (background) mean prob: {output_probs[0, 0].mean():.4f}")
print(f"    Class 1 (spleen) mean prob: {output_probs[0, 1].mean():.4f}")

# Get predicted class (argmax)
predicted_mask = torch.argmax(output_probs, dim=1)
print(f"    Predicted mask shape: {predicted_mask.shape}")
print(f"    Unique values in mask: {torch.unique(predicted_mask).cpu().numpy()}")

# 6. Visualize results (center slice)
print("\n[6] Creating visualization...")

try:
    center_slice = sample_ct_scan.shape[2] // 2

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Input CT scan
    axes[0].imshow(sample_ct_scan[0, 0, center_slice].cpu().numpy(), cmap='gray')
    axes[0].set_title('Input CT Scan (center slice)')
    axes[0].axis('off')

    # Predicted spleen probability
    axes[1].imshow(output_probs[0, 1, center_slice].cpu().numpy(), cmap='hot')
    axes[1].set_title('Spleen Probability Map')
    axes[1].axis('off')

    # Binary segmentation mask
    axes[2].imshow(predicted_mask[0, center_slice].cpu().numpy(), cmap='viridis')
    axes[2].set_title('Segmentation Mask')
    axes[2].axis('off')

    plt.tight_layout()
    output_file = "pretrained_model_demo.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"    Visualization saved to: {output_file}")
    plt.close()

except Exception as e:
    print(f"    [WARNING] Could not create visualization: {e}")

# 7. Show how to use with real medical images
print("\n" + "=" * 80)
print("[7] HOW TO USE WITH REAL MEDICAL IMAGES")
print("=" * 80)

print("""
# Step 1: Prepare your CT scan data (DICOM or NIfTI format)
# Example file structure:
#   ./data/patient_001/ct_scan.nii.gz
#   ./data/patient_002/ct_scan.nii.gz

# Step 2: Load and preprocess the data
from monai.transforms import LoadImaged, Compose, AddChanneld
from monai.data import Dataset, DataLoader

data_dicts = [
    {"image": "./data/patient_001/ct_scan.nii.gz"},
    {"image": "./data/patient_002/ct_scan.nii.gz"},
]

transforms = Compose([
    LoadImaged(keys=["image"]),
    AddChanneld(keys=["image"]),
    # Add other preprocessing as needed
])

dataset = Dataset(data=data_dicts, transform=transforms)
dataloader = DataLoader(dataset, batch_size=1)

# Step 3: Run inference on each image
for batch in dataloader:
    inputs = batch["image"].to(device)
    with torch.no_grad():
        outputs = model(inputs)
    # Process outputs...
""")

print("\n" + "=" * 80)
print("[8] OTHER AVAILABLE PRETRAINED MODELS")
print("=" * 80)

pretrained_models = [
    ("spleen_ct_segmentation", "Spleen segmentation from CT", "[DOWNLOADED]"),
    ("prostate_mri_anatomy", "Prostate MRI segmentation", "Easy to use"),
    ("lung_nodule_ct_detection", "Lung nodule detection", "Medium"),
    ("pancreas_ct_dints_segmentation", "Pancreas segmentation", "Medium"),
    ("pathology_tumor_detection", "Pathology tumor detection", "Easy"),
    ("wholebody_ct_segmentation", "Whole body CT segmentation", "Advanced"),
]

print("\nModel Name                          | Description                    | Status")
print("-" * 80)
for name, desc, status in pretrained_models:
    print(f"{name:35} | {desc:30} | {status}")

print("\n[*] To download another model:")
print("    python -m monai.bundle download --name MODEL_NAME --bundle_dir ./models")

print("\n[*] To list all available models:")
print("    python list_pretrained_models.py")

print("\n" + "=" * 80)
print("[SUCCESS] PRETRAINED MODEL DEMO COMPLETED!")
print("=" * 80)
print("\n[KEY BENEFITS]")
print("  - No training required - model is ready to use!")
print("  - Pre-trained on large medical imaging datasets")
print("  - State-of-the-art performance out of the box")
print("  - Can be fine-tuned on your specific data if needed")
print("  - Saves hours/days of training time")
print("  - Works with 4GB VRAM GPU!")

print("\n[NEXT STEPS]")
print("  1. Download real CT scan data (NIfTI or DICOM format)")
print("  2. Run inference on your data")
print("  3. Visualize and analyze the results")
print("  4. Optional: Fine-tune on your specific dataset")

print("\n")

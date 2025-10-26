"""
Download sample CT scan data for testing the pretrained model
Uses MONAI's built-in data download utilities
"""

import os
from monai.apps import download_and_extract
import gdown

print("=" * 80)
print("DOWNLOADING SAMPLE CT SCAN DATA FOR TESTING")
print("=" * 80)

# Create data directory
data_dir = "./test_data"
os.makedirs(data_dir, exist_ok=True)
print(f"\n[1] Created data directory: {data_dir}")

# Download Medical Segmentation Decathlon - Spleen dataset (Task09)
# This is a small subset for testing
print("\n[2] Downloading sample spleen CT scans...")
print("    Source: Medical Segmentation Decathlon")
print("    Task: Spleen Segmentation")

try:
    # MONAI provides access to sample data
    from monai.apps import DecathlonDataset

    print("\n    Attempting to download via MONAI DecathlonDataset...")

    # This will download a small subset of the data
    resource = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar"

    print(f"    Downloading from: {resource}")
    print("    Note: This may take a few minutes (file size: ~1.4 GB)")
    print("    Please wait...")

    compressed_file = os.path.join(data_dir, "Task09_Spleen.tar")
    data_folder = os.path.join(data_dir, "Task09_Spleen")

    if not os.path.exists(data_folder):
        download_and_extract(
            url=resource,
            output_dir=data_dir,
            filepath=compressed_file,
        )
        print(f"\n    [SUCCESS] Data downloaded to: {data_folder}")
    else:
        print(f"\n    [INFO] Data already exists at: {data_folder}")

    # Check what was downloaded
    print("\n[3] Checking downloaded data...")

    imagesTr_dir = os.path.join(data_folder, "imagesTr")
    labelsTr_dir = os.path.join(data_folder, "labelsTr")

    if os.path.exists(imagesTr_dir):
        images = [f for f in os.listdir(imagesTr_dir) if f.endswith('.nii.gz')]
        print(f"    Training images: {len(images)} CT scans")
        print(f"    Location: {imagesTr_dir}")

        # Show first few files
        print("\n    Sample files:")
        for img in images[:5]:
            print(f"      - {img}")
        if len(images) > 5:
            print(f"      ... and {len(images) - 5} more")

    if os.path.exists(labelsTr_dir):
        labels = [f for f in os.listdir(labelsTr_dir) if f.endswith('.nii.gz')]
        print(f"\n    Ground truth labels: {len(labels)} segmentation masks")
        print(f"    Location: {labelsTr_dir}")

    print("\n[4] Data Structure:")
    print(f"    {data_folder}/")
    print(f"    ├── imagesTr/          # CT scan images")
    print(f"    ├── labelsTr/          # Ground truth segmentation masks")
    print(f"    └── dataset.json       # Dataset metadata")

except Exception as e:
    print(f"\n    [ERROR] Failed to download data: {e}")
    print("\n    Alternative: Download manually from:")
    print("    https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("[INFO] NEXT STEPS")
print("=" * 80)
print("\n1. Data is ready for testing")
print("2. Run evaluation script:")
print("   python evaluate_model.py")
print("\n3. This will test the pretrained model on real CT scans")
print("4. Calculate accuracy metrics (Dice score, IoU, etc.)")
print()

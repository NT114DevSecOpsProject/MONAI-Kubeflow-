"""
Fast reorganization: Simply move CT images to images/ and all segmentation files to labels/
"""

import os
import shutil
from pathlib import Path

# Define paths (relative to script location)
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
base_dir = project_root / "test_data/TotalSegmentator_small"
output_dir = base_dir.parent / "TotalSegmentator_small_organized"
images_dir = os.path.join(output_dir, "images")
labels_dir = os.path.join(output_dir, "labels")

# Convert to strings for os.path
base_dir = str(base_dir)
output_dir = str(output_dir)

# Create output directories
os.makedirs(images_dir, exist_ok=True)
os.makedirs(labels_dir, exist_ok=True)

print(f"Reorganizing data to: {output_dir}")

# Get all patient directories
patient_dirs = sorted([d for d in os.listdir(base_dir)
                       if os.path.isdir(os.path.join(base_dir, d)) and d.startswith('s')])

print(f"Found {len(patient_dirs)} patient directories\n")

# Process each patient
for i, patient_id in enumerate(patient_dirs, 1):
    patient_path = os.path.join(base_dir, patient_id)
    ct_file = os.path.join(patient_path, "ct.nii.gz")
    segmentations_dir = os.path.join(patient_path, "segmentations")

    # Copy CT image
    if os.path.exists(ct_file):
        output_ct_file = os.path.join(images_dir, f"{patient_id}_ct.nii.gz")
        shutil.copy2(ct_file, output_ct_file)

    # Copy all segmentation files
    if os.path.exists(segmentations_dir):
        seg_files = [f for f in os.listdir(segmentations_dir) if f.endswith('.nii.gz')]
        for seg_file in seg_files:
            src = os.path.join(segmentations_dir, seg_file)
            dst = os.path.join(labels_dir, f"{patient_id}_{seg_file}")
            shutil.copy2(src, dst)

        if i % 10 == 0:
            print(f"[{i}/{len(patient_dirs)}] Processed {patient_id} ({len(seg_files)} organ masks)")

print(f"\n✓ Done! Data reorganized in: {output_dir}")
print(f"  - Images: {len(os.listdir(images_dir))} files")
print(f"  - Labels: {len(os.listdir(labels_dir))} files")

# Replace original with reorganized
print("\nReplacing original directory...")
backup_dir = base_dir + "_original_backup"
if os.path.exists(backup_dir):
    shutil.rmtree(backup_dir)
os.rename(base_dir, backup_dir)
os.rename(output_dir, base_dir)
print(f"✓ Original backed up to: {backup_dir}")
print(f"✓ New structure at: {base_dir}")

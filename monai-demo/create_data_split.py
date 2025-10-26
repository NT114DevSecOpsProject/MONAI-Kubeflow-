"""
Create and display data split for train/validation/test sets
This ensures transparency about which files are used where
"""

import random
import json
from pathlib import Path

print("=" * 90)
print("MEDICAL DECATHLON TASK09_SPLEEN - DATA SPLIT CREATION")
print("=" * 90)

data_dir = Path("./test_data/Task09_Spleen")
images_dir = data_dir / "imagesTr"

# Get all image files
image_files = sorted([f for f in images_dir.glob("*.nii.gz") if not f.name.startswith("._")])

print(f"\n[STEP 1] Load All Files")
print(f"  Total CT scans found: {len(image_files)}")

# Apply random shuffle with fixed seed for reproducibility
print(f"\n[STEP 2] Apply Random Split (seed=42)")
random.seed(42)
indices = list(range(len(image_files)))
random.shuffle(indices)

# Create splits
train_indices = indices[:32]    # 32 scans
val_indices = indices[32:37]    # 5 scans
test_indices = indices[37:41]   # 4 scans

train_files = [image_files[i].name for i in train_indices]
val_files = [image_files[i].name for i in val_indices]
test_files = [image_files[i].name for i in test_indices]

# Display split
print(f"\n[SPLIT RESULT]")
print(f"  Random seed: 42 (reproducible)")
print(f"  Total: 41 samples")
print(f"    - Training:   32 samples (used to train model)")
print(f"    - Validation:  5 samples (used for tuning/early stopping)")
print(f"    - Test:        4 samples (UNSEEN - for proper evaluation)")

print(f"\n" + "=" * 90)
print(f"[TRAINING SET] 32 samples")
print("=" * 90)
for idx, f in enumerate(sorted(train_files), 1):
    print(f"{idx:2d}. {f}")

print(f"\n" + "=" * 90)
print(f"[VALIDATION SET] 5 samples")
print("=" * 90)
for idx, f in enumerate(sorted(val_files), 1):
    print(f"{idx}. {f}")

print(f"\n" + "=" * 90)
print(f"[TEST SET] 4 samples - UNSEEN during training")
print("=" * 90)
for idx, f in enumerate(sorted(test_files), 1):
    print(f"{idx}. {f}")

# Save to JSON
split_data = {
    "dataset": "Medical Segmentation Decathlon - Task09_Spleen",
    "source": "http://medicaldecathlon.com/",
    "total_samples": len(image_files),
    "random_seed": 42,
    "splits": {
        "training": {
            "count": len(train_files),
            "files": sorted(train_files),
            "purpose": "Used to train the model"
        },
        "validation": {
            "count": len(val_files),
            "files": sorted(val_files),
            "purpose": "Used for model tuning (early stopping, hyperparameter adjustment)"
        },
        "test": {
            "count": len(test_files),
            "files": sorted(test_files),
            "purpose": "NEVER seen during training - PROPER EVALUATION"
        }
    }
}

with open("data_split_mapping.json", "w") as f:
    json.dump(split_data, f, indent=2)

print(f"\n[SAVED] data_split_mapping.json")
print(f"\nUsage:")
print(f"  - evaluate_accuracy.py: Tests on VALIDATION set (may be optimistic)")
print(f"  - evaluate_on_test_set.py: Tests on TEST set (unbiased)")


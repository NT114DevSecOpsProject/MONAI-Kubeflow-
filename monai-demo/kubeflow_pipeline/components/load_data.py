"""
Load Data Component: Copy CT scan from source to pipeline input directory
Source: /mnt/data/test_data/Task09_Spleen/imagesTr/spleen_XX.nii.gz
Target: /mnt/data/inputs/week_current/spleen_XX/imaging.nii.gz
"""

import sys
import shutil
from pathlib import Path

def load_data(patient_id: str):
    """Copy CT scan from test dataset to pipeline input directory"""
    print(f"\n{'='*60}")
    print(f"LOAD DATA: {patient_id}")
    print(f"{'='*60}")

    try:
        # Source: test dataset
        source_file = Path(f"/mnt/data/test_data/Task09_Spleen/imagesTr/{patient_id}.nii.gz")

        # Target: pipeline input directory
        target_dir = Path(f"/mnt/data/inputs/week_current/{patient_id}")
        target_file = target_dir / "imaging.nii.gz"

        print(f"Source: {source_file}")
        print(f"Target: {target_file}")

        # Check source exists
        if not source_file.exists():
            raise FileNotFoundError(f"Source file not found: {source_file}")

        # Create target directory
        print(f"[Step 1/2] Creating target directory...")
        target_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Created: {target_dir}")

        # Copy file (skip if already exists)
        print(f"[Step 2/2] Copying CT scan...")
        if target_file.exists():
            print(f"  File already exists, skipping copy")
        else:
            shutil.copy2(source_file, target_file)
            print(f"  Copied successfully")

        # Verify
        file_size = target_file.stat().st_size / (1024 * 1024)  # MB
        print(f"  Size: {file_size:.2f} MB")
        print(f"  Location: {target_file}")

        print(f"[OK] Data loading complete!")
        return 0

    except Exception as e:
        print(f"[ERROR] Data loading failed: {e}")
        return 1

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python load_data.py <patient_id>")
        sys.exit(1)

    patient_id = sys.argv[1]
    sys.exit(load_data(patient_id))

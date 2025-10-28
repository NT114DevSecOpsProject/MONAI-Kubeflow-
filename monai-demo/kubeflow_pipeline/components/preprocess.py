"""
Preprocess Component: Load NIfTI CT scan and apply MONAI normalization
Input: /mnt/data/inputs/week_current/spleen_XX/imaging.nii.gz
Output: /mnt/data/inputs/week_current/spleen_XX/preprocessed.pt
"""

import sys
import torch
from pathlib import Path

def preprocess(patient_id: str):
    """Load and preprocess CT scan"""
    print(f"\n{'='*60}")
    print(f"PREPROCESS: {patient_id}")
    print(f"{'='*60}")

    try:
        # Paths
        input_dir = Path(f"/mnt/data/inputs/week_current/{patient_id}")
        input_file = input_dir / "imaging.nii.gz"
        output_file = input_dir / "preprocessed.pt"

        print(f"Input: {input_file}")
        print(f"Output: {output_file}")

        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        # Import MONAI
        from monai.transforms import (
            Compose, LoadImage, EnsureChannelFirst,
            Orientation, Spacing, ScaleIntensityRange,
            CropForeground, EnsureType
        )

        # Setup transforms (same as 01_test_task09.py)
        transforms = Compose([
            LoadImage(image_only=True),
            EnsureChannelFirst(),
            Orientation(axcodes="RAS"),
            Spacing(pixdim=(1.5, 1.5, 2.0), mode="bilinear"),
            ScaleIntensityRange(
                a_min=-175,
                a_max=250,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForeground(),
            EnsureType(),
        ])

        print("[Step 1/2] Loading and transforming...")
        tensor = transforms(str(input_file))

        print(f"[Step 2/2] Saving preprocessed tensor...")
        print(f"  Shape: {tensor.shape}")
        print(f"  Dtype: {tensor.dtype}")

        torch.save(tensor, output_file)

        print(f"[OK] Preprocessing complete!")
        return 0

    except Exception as e:
        print(f"[ERROR] Preprocessing failed: {e}")
        return 1

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python preprocess.py <patient_id>")
        sys.exit(1)

    patient_id = sys.argv[1]
    sys.exit(preprocess(patient_id))

"""
Visualize Component: Create 3-view overlay images of CT scan + segmentation
Input: /mnt/data/inputs/week_current/spleen_XX/imaging.nii.gz
       /mnt/data/outputs/week_current/spleen_XX/segmentation.nii.gz
Output: /mnt/data/outputs/week_current/spleen_XX/{axial,coronal,sagittal}.png
"""

import sys
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path

def visualize(patient_id: str):
    """Create 3-view visualizations"""
    print(f"\n{'='*60}")
    print(f"VISUALIZE: {patient_id}")
    print(f"{'='*60}")

    try:
        # Paths
        input_file = Path(f"/mnt/data/inputs/week_current/{patient_id}/imaging.nii.gz")
        mask_file = Path(f"/mnt/data/outputs/week_current/{patient_id}/segmentation.nii.gz")
        output_dir = Path(f"/mnt/data/outputs/week_current/{patient_id}")

        print(f"CT Input: {input_file}")
        print(f"Mask Input: {mask_file}")
        print(f"Output Dir: {output_dir}")

        if not input_file.exists():
            raise FileNotFoundError(f"CT scan not found: {input_file}")

        if not mask_file.exists():
            raise FileNotFoundError(f"Segmentation mask not found: {mask_file}")

        # Load images
        print("[Step 1/4] Loading CT scan and mask...")
        ct_img = nib.load(input_file)
        ct_data = ct_img.get_fdata()

        mask_img = nib.load(mask_file)
        mask_data = mask_img.get_fdata()

        print(f"  CT shape: {ct_data.shape}")
        print(f"  Mask shape: {mask_data.shape}")

        # Normalize CT for display
        ct_data = np.clip(ct_data, -175, 250)
        ct_data = (ct_data - ct_data.min()) / (ct_data.max() - ct_data.min() + 1e-8)

        # Find center slices with spleen
        print("[Step 2/4] Finding best slices...")
        spleen_mask = mask_data > 0.5

        # Axial (Z-axis)
        z_count = np.sum(spleen_mask, axis=(0, 1))
        z_slice = np.argmax(z_count) if z_count.max() > 0 else ct_data.shape[2] // 2

        # Coronal (Y-axis)
        y_count = np.sum(spleen_mask, axis=(0, 2))
        y_slice = np.argmax(y_count) if y_count.max() > 0 else ct_data.shape[1] // 2

        # Sagittal (X-axis)
        x_count = np.sum(spleen_mask, axis=(1, 2))
        x_slice = np.argmax(x_count) if x_count.max() > 0 else ct_data.shape[0] // 2

        print(f"  Axial slice: {z_slice}/{ct_data.shape[2]}")
        print(f"  Coronal slice: {y_slice}/{ct_data.shape[1]}")
        print(f"  Sagittal slice: {x_slice}/{ct_data.shape[0]}")

        # Create visualizations
        print("[Step 3/4] Creating visualizations...")

        views = [
            ("axial", ct_data[:, :, z_slice], mask_data[:, :, z_slice], f"Axial (Z={z_slice})"),
            ("coronal", ct_data[:, y_slice, :], mask_data[:, y_slice, :], f"Coronal (Y={y_slice})"),
            ("sagittal", ct_data[x_slice, :, :], mask_data[x_slice, :, :], f"Sagittal (X={x_slice})")
        ]

        for view_name, ct_slice, mask_slice, title in views:
            fig, ax = plt.subplots(figsize=(8, 8))

            # Show CT scan
            ax.imshow(ct_slice.T, cmap='gray', origin='lower')

            # Overlay segmentation mask in yellow with transparency
            mask_overlay = np.ma.masked_where(mask_slice.T < 0.5, mask_slice.T)
            ax.imshow(mask_overlay, cmap='autumn', alpha=0.5, origin='lower')

            ax.set_title(f'{patient_id} - Spleen Segmentation\n{title}',
                        fontsize=14, fontweight='bold')
            ax.axis('off')

            # Save
            output_file = output_dir / f"{view_name}.png"
            plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()

            print(f"  Saved: {view_name}.png")

        print("[Step 4/4] Complete!")
        print(f"[OK] Visualization complete!")
        return 0

    except Exception as e:
        print(f"[ERROR] Visualization failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python visualize.py <patient_id>")
        sys.exit(1)

    patient_id = sys.argv[1]
    sys.exit(visualize(patient_id))

"""
Visualize Component: Create 3-panel visualization (CT + Probability + Mask)
FINAL VERSION - Auto-resize mask to match CT
"""

import sys
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path

def visualize(patient_id: str):
    """Create 3-panel visualization with auto-resize"""
    print(f"\n{'='*60}")
    print(f"VISUALIZE: {patient_id}")
    print(f"{'='*60}")

    try:
        # Paths
        ct_file = Path(f"/mnt/data/inputs/week_current/{patient_id}/imaging.nii.gz")
        mask_file = Path(f"/mnt/data/outputs/week_current/{patient_id}/segmentation.nii.gz")
        prob_file = Path(f"/mnt/data/outputs/week_current/{patient_id}/probability.nii.gz")
        output_dir = Path(f"/mnt/data/outputs/week_current/{patient_id}")
        output_file = output_dir / "result.png"

        print(f"CT Input: {ct_file}")
        print(f"Mask Input: {mask_file}")
        print(f"Probability Input: {prob_file}")

        if not ct_file.exists():
            raise FileNotFoundError(f"CT scan not found: {ct_file}")
        if not mask_file.exists():
            raise FileNotFoundError(f"Mask not found: {mask_file}")
        if not prob_file.exists():
            raise FileNotFoundError(f"Probability map not found: {prob_file}")

        # Load data
        print("[Step 1/5] Loading data...")
        ct_data = nib.load(ct_file).get_fdata()
        mask_data = nib.load(mask_file).get_fdata()
        prob_data = nib.load(prob_file).get_fdata()

        print(f"  CT shape: {ct_data.shape}")
        print(f"  Mask shape: {mask_data.shape}")
        print(f"  Probability shape: {prob_data.shape}")

        # AUTO-RESIZE mask and prob if needed
        if mask_data.shape != ct_data.shape or prob_data.shape != ct_data.shape:
            print("[Step 2/5] Auto-resizing mask and probability to match CT...")
            import torch
            from monai.transforms import Resize

            # Resize mask
            if mask_data.shape != ct_data.shape:
                print(f"  Resizing mask from {mask_data.shape} to {ct_data.shape}...")
                resize_transform = Resize(spatial_size=ct_data.shape, mode='nearest')
                mask_tensor = torch.from_numpy(mask_data).unsqueeze(0).unsqueeze(0).float()
                mask_data = resize_transform(mask_tensor)[0, 0].numpy()

            # Resize probability
            if prob_data.shape != ct_data.shape:
                print(f"  Resizing probability from {prob_data.shape} to {ct_data.shape}...")
                resize_transform = Resize(spatial_size=ct_data.shape, mode='trilinear')
                prob_tensor = torch.from_numpy(prob_data).unsqueeze(0).unsqueeze(0).float()
                prob_data = resize_transform(prob_tensor)[0, 0].numpy()

            print(f"  After resize - Mask: {mask_data.shape}, Prob: {prob_data.shape}")

        # Normalize CT
        print("[Step 3/5] Normalizing CT scan...")
        ct_display = np.clip(ct_data, -175, 250)
        ct_display = (ct_display - ct_display.min()) / (ct_display.max() - ct_display.min() + 1e-8)

        # Find best slice
        print("[Step 4/5] Finding best slice...")
        z_counts = np.sum(mask_data > 0.5, axis=(0, 1))
        best_slice = np.argmax(z_counts) if z_counts.max() > 0 else ct_data.shape[2] // 2
        # Ensure slice is in bounds
        best_slice = min(best_slice, ct_data.shape[2] - 1)
        total_slices = ct_data.shape[2]
        print(f"  Best axial slice: {best_slice}/{total_slices}")

        # Extract slices
        ct_slice = ct_display[:, :, best_slice].T
        prob_slice = prob_data[:, :, best_slice].T
        mask_slice = mask_data[:, :, best_slice].T

        # Calculate statistics
        avg_prob = prob_slice[mask_slice > 0.5].mean() if (mask_slice > 0.5).sum() > 0 else 0.0
        spleen_voxels = np.sum(mask_data > 0.5)
        total_voxels = np.prod(mask_data.shape)
        spleen_percentage = (spleen_voxels / total_voxels) * 100

        print(f"  Average probability: {avg_prob:.3f}")
        print(f"  Spleen percentage: {spleen_percentage:.2f}%")

        # Create visualization
        print("[Step 5/5] Creating 3-panel visualization...")
        fig = plt.figure(figsize=(18, 6))

        # Main title
        fig.suptitle('MONAI Pretrained Model - Spleen Segmentation\nData Source: Task09_Spleen Test Set',
                    fontsize=16, fontweight='bold', y=0.98)

        # Panel 1: CT Scan
        ax1 = plt.subplot(131)
        ax1.imshow(ct_slice, cmap='gray', origin='lower')
        ax1.set_title(f'Input CT Scan\n{patient_id}.nii.gz\n(slice {best_slice}/{total_slices})',
                     fontsize=12, fontweight='bold')
        ax1.axis('off')

        # Panel 2: Probability Map
        ax2 = plt.subplot(132)
        im = ax2.imshow(prob_slice, cmap='hot', vmin=0, vmax=1, origin='lower')
        ax2.set_title(f'Spleen Probability Map\n(avg: {avg_prob:.3f})',
                     fontsize=12, fontweight='bold')
        ax2.axis('off')
        cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        cbar.set_label('Probability', rotation=270, labelpad=15)

        # Panel 3: Predicted Mask
        ax3 = plt.subplot(133)
        # Purple background
        background = np.ones((*mask_slice.shape, 3))
        background[:, :, 0] = 0.58
        background[:, :, 1] = 0.0
        background[:, :, 2] = 1.0
        ax3.imshow(background, origin='lower')

        # Yellow spleen mask
        spleen_mask_display = np.ma.masked_where(mask_slice < 0.5, mask_slice)
        ax3.imshow(spleen_mask_display, cmap='spring', vmin=0, vmax=1, origin='lower')

        ax3.set_title(f'Predicted Mask\n({spleen_percentage:.2f}% spleen in volume)',
                     fontsize=12, fontweight='bold')
        ax3.axis('off')

        # Save
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"  Saved: result.png")
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

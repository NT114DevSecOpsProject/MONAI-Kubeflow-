"""
Visualize Component: Create 3-panel overlay images (CT | Probability | Mask)
Input: /mnt/data/inputs/week_current/spleen_XX/preprocessed.pt
       /mnt/data/outputs/week_current/spleen_XX/probability.npy
       /mnt/data/outputs/week_current/spleen_XX/segmentation.nii.gz
Output: /mnt/data/outputs/week_current/spleen_XX/visualization.png
"""

import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

def visualize(patient_id: str):
    """Create 3-panel visualization"""
    print(f"\n{'='*60}")
    print(f"VISUALIZE: {patient_id}")
    print(f"{'='*60}")

    try:
        # Paths
        preprocessed_file = Path(f"/mnt/data/inputs/week_current/{patient_id}/preprocessed.pt")
        prob_file = Path(f"/mnt/data/outputs/week_current/{patient_id}/probability.npy")
        mask_file = Path(f"/mnt/data/outputs/week_current/{patient_id}/segmentation.nii.gz")
        output_file = Path(f"/mnt/data/outputs/week_current/{patient_id}/visualization.png")

        print(f"Preprocessed: {preprocessed_file}")
        print(f"Probability: {prob_file}")
        print(f"Mask: {mask_file}")
        print(f"Output: {output_file}")

        # Verify files
        if not preprocessed_file.exists():
            raise FileNotFoundError(f"Preprocessed file not found: {preprocessed_file}")
        if not prob_file.exists():
            raise FileNotFoundError(f"Probability file not found: {prob_file}")
        if not mask_file.exists():
            raise FileNotFoundError(f"Mask file not found: {mask_file}")

        # Load data
        print("[Step 1/4] Loading data...")
        ct_tensor = torch.load(preprocessed_file, map_location='cpu').numpy()
        if ct_tensor.ndim == 4:  # (1, H, W, D)
            ct_tensor = ct_tensor[0]

        probs = np.load(prob_file)

        import nibabel as nib
        mask_data = nib.load(mask_file).get_fdata()

        print(f"  CT shape: {ct_tensor.shape}")
        print(f"  Prob shape: {probs.shape}")
        print(f"  Mask shape: {mask_data.shape}")

        # Find best slice (where spleen has most pixels)
        print("[Step 2/4] Finding best slice...")
        spleen_count = np.sum(mask_data, axis=(0, 1))
        if np.max(spleen_count) > 0:
            slice_idx = np.argmax(spleen_count)
        else:
            slice_idx = ct_tensor.shape[2] // 2

        print(f"  Best slice: {slice_idx}/{ct_tensor.shape[2]}")

        # Create 3-panel visualization
        print("[Step 3/4] Creating visualization...")

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'MONAI Spleen Segmentation - Kubeflow Pipeline\n{patient_id}',
                     fontsize=14, fontweight='bold')

        # Panel 1: Input CT Scan
        ax = axes[0]
        ax.imshow(ct_tensor[:, :, slice_idx], cmap='gray')
        ax.set_title(f'Input CT Scan\n{patient_id}.nii.gz\n(slice {slice_idx}/{ct_tensor.shape[2]})',
                     fontsize=11, fontweight='bold')
        ax.axis('off')

        # Panel 2: Spleen Probability Map
        ax = axes[1]
        prob_slice = probs[:, :, slice_idx]
        im = ax.imshow(prob_slice, cmap='hot', vmin=0, vmax=1)
        ax.set_title(f'Spleen Probability Map\n(avg: {np.mean(prob_slice):.3f})',
                     fontsize=11, fontweight='bold')
        ax.axis('off')
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Probability')

        # Panel 3: Predicted Mask (Yellow on Purple)
        ax = axes[2]
        mask_display = np.zeros((mask_data.shape[0], mask_data.shape[1], 3))
        pred_slice = mask_data[:, :, slice_idx]

        spleen_pixels = pred_slice > 0.5
        background_pixels = ~spleen_pixels

        # Yellow = Spleen, Purple = Background
        mask_display[spleen_pixels, 0] = 1.0    # Red
        mask_display[spleen_pixels, 1] = 1.0    # Green (Red+Green=Yellow)
        mask_display[background_pixels, 0] = 0.6  # Purple
        mask_display[background_pixels, 2] = 1.0

        ax.imshow(mask_display)
        spleen_volume = np.sum(mask_data) / mask_data.size * 100
        ax.set_title(f'Predicted Mask\n({spleen_volume:.2f}% spleen in volume)',
                     fontsize=11, fontweight='bold')
        ax.axis('off')

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='yellow', label='Spleen (Predicted)'),
            Patch(facecolor='purple', label='Background')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

        plt.tight_layout()

        # Save
        print("[Step 4/4] Saving visualization...")
        plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"[OK] Visualization complete!")
        print(f"[OK] Saved: {output_file}")
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

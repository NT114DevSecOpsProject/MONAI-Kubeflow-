"""
Visualize kết quả segmentation
So sánh Ground Truth vs Prediction

Output: PNG files với 2x3 subplots cho mỗi patient
"""

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from pathlib import Path
import json

def visualize_patient(ct_path, gt_path, pred_path, output_path):
    """
    Tạo visualization cho 1 patient

    Layout: 2x3 grid
    Row 1: Original CT | Ground Truth | Prediction
    Row 2: GT Overlay | Pred Overlay | Difference

    Args:
        ct_path: Path to original CT scan
        gt_path: Path to ground truth segmentation
        pred_path: Path to prediction
        output_path: Output PNG file
    """
    # Load images
    ct = sitk.GetArrayFromImage(sitk.ReadImage(str(ct_path)))
    gt = sitk.GetArrayFromImage(sitk.ReadImage(str(gt_path)))
    pred = sitk.GetArrayFromImage(sitk.ReadImage(str(pred_path)))

    # Get middle slice
    slice_idx = ct.shape[0] // 2

    ct_slice = ct[slice_idx]
    gt_slice = gt[slice_idx]
    pred_slice = pred[slice_idx]

    # Calculate difference
    diff = np.abs(pred_slice.astype(float) - gt_slice.astype(float))

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'{ct_path.stem}\nSlice {slice_idx}/{ct.shape[0]}', fontsize=14, fontweight='bold')

    # Row 1: Original images
    # Original CT
    axes[0, 0].imshow(ct_slice, cmap='gray', vmin=-1000, vmax=500)
    axes[0, 0].set_title('Original CT', fontsize=12)
    axes[0, 0].axis('off')

    # Ground Truth
    axes[0, 1].imshow(ct_slice, cmap='gray', vmin=-1000, vmax=500)
    axes[0, 1].imshow(gt_slice, cmap='Reds', alpha=0.5)
    axes[0, 1].set_title('Ground Truth', fontsize=12)
    axes[0, 1].axis('off')

    # Prediction
    axes[0, 2].imshow(ct_slice, cmap='gray', vmin=-1000, vmax=500)
    axes[0, 2].imshow(pred_slice, cmap='Blues', alpha=0.5)
    axes[0, 2].set_title('Prediction (LungMask)', fontsize=12)
    axes[0, 2].axis('off')

    # Row 2: Overlays and comparison
    # GT Mask only
    axes[1, 0].imshow(gt_slice, cmap='Reds')
    axes[1, 0].set_title('GT Mask', fontsize=12)
    axes[1, 0].axis('off')

    # Pred Mask only
    axes[1, 1].imshow(pred_slice, cmap='Blues')
    axes[1, 1].set_title('Predicted Mask', fontsize=12)
    axes[1, 1].axis('off')

    # Difference (error map)
    im = axes[1, 2].imshow(diff, cmap='hot')
    axes[1, 2].set_title('Difference (Error Map)', fontsize=12)
    axes[1, 2].axis('off')
    plt.colorbar(im, ax=axes[1, 2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()


def create_summary_plot(results_json_path, output_path):
    """
    Tạo summary plot: Bar chart của Dice scores

    Args:
        results_json_path: Path to test_results.json
        output_path: Output PNG file
    """
    with open(results_json_path, 'r') as f:
        data = json.load(f)

    results = data['individual_results']

    # Extract data
    patients = [r['patient'] for r in results]
    dice_scores = [r['dice'] for r in results]
    avg_dice = data['summary']['avg_dice']

    # Create bar chart
    fig, ax = plt.subplots(figsize=(12, 6))

    bars = ax.bar(range(len(patients)), dice_scores, color='steelblue', alpha=0.8, edgecolor='black')

    # Add average line
    ax.axhline(y=avg_dice, color='red', linestyle='--', linewidth=2, label=f'Average: {avg_dice:.4f}')

    # Add threshold lines
    ax.axhline(y=0.95, color='green', linestyle=':', linewidth=1, alpha=0.5, label='Excellent (≥0.95)')
    ax.axhline(y=0.90, color='orange', linestyle=':', linewidth=1, alpha=0.5, label='Good (≥0.90)')

    # Formatting
    ax.set_xlabel('Patient', fontsize=12, fontweight='bold')
    ax.set_ylabel('Dice Score', fontsize=12, fontweight='bold')
    ax.set_title('LungMask Performance on Medical Decathlon Dataset', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(patients)))
    ax.set_xticklabels([p.replace('lung_', '') for p in patients], rotation=45, ha='right')
    ax.set_ylim([0.80, 1.0])
    ax.grid(axis='y', alpha=0.3)
    ax.legend(loc='lower right')

    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars, dice_scores)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{score:.4f}',
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()


def main():
    """
    Visualize tất cả kết quả từ test_lungmask.py
    """
    print("\n" + "="*60)
    print("Visualization Script")
    print("Creating comparison images: GT vs Prediction")
    print("="*60)

    # Check if test results exist
    results_file = Path("./test_results.json")
    if not results_file.exists():
        print("\n✗ Error: test_results.json not found!")
        print("Please run test script first:")
        print("  python demo/test_lungmask.py")
        return

    # Load results
    with open(results_file, 'r') as f:
        results_data = json.load(f)

    results = results_data['individual_results']
    print(f"\nFound {len(results)} patients to visualize")

    # Paths
    data_dir = Path("./sample-data/Task06_Lung")
    images_dir = data_dir / "imagesTr"
    labels_dir = data_dir / "labelsTr"
    pred_dir = Path("./sample-data/predictions")

    # Create output directory
    output_dir = Path("./visualizations")
    output_dir.mkdir(exist_ok=True)

    # Visualize each patient
    print("\nGenerating visualizations...")
    for result in results:
        patient_id = result['patient']

        # Paths
        ct_path = images_dir / f"{patient_id}.nii.gz"
        gt_path = labels_dir / f"{patient_id}.nii.gz"
        pred_path = pred_dir / f"{patient_id}_pred.nii.gz"

        # Check if all files exist
        if not ct_path.exists():
            print(f"  ⚠ Warning: {ct_path.name} not found, skipping...")
            continue
        if not gt_path.exists():
            print(f"  ⚠ Warning: {gt_path.name} not found, skipping...")
            continue
        if not pred_path.exists():
            print(f"  ⚠ Warning: {pred_path.name} not found, skipping...")
            continue

        # Output path
        output_path = output_dir / f"{patient_id}_comparison.png"

        print(f"\n[{patient_id}]")
        print(f"  Dice: {result['dice']:.4f}")

        try:
            visualize_patient(ct_path, gt_path, pred_path, output_path)
        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue

    # Create summary plot
    print("\nGenerating summary plot...")
    summary_path = output_dir / "summary_dice_scores.png"
    try:
        create_summary_plot(results_file, summary_path)
    except Exception as e:
        print(f"  ✗ Error creating summary plot: {e}")

    # Summary
    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60)
    print(f"✓ Visualizations saved to: {output_dir}/")
    print(f"\nGenerated files:")
    print(f"  - Individual comparisons: {len(results)} files")
    print(f"  - Summary plot: summary_dice_scores.png")

    avg_dice = results_data['summary']['avg_dice']
    print(f"\nAverage Dice Score: {avg_dice:.4f}")

    if avg_dice >= 0.95:
        print("✓ Excellent performance! Model ready for production.")
    elif avg_dice >= 0.90:
        print("✓ Good performance! Model works well.")
    else:
        print("⚠ Consider fine-tuning for better accuracy.")

    print("\nNext step:")
    print("  Deploy service: python deployment/serve.py")
    print("="*60)


if __name__ == "__main__":
    main()

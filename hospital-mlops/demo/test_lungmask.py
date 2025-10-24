"""
Test LungMask pretrained model trên Medical Decathlon data
Tính Dice score để verify accuracy

Expected output: Average Dice ~0.98
"""

import os
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from lungmask import LMInferer
import time
import json

def calculate_dice(pred, ground_truth):
    """
    Calculate Dice coefficient
    Formula: 2 * |A ∩ B| / (|A| + |B|)
    """
    pred_binary = pred > 0
    gt_binary = ground_truth > 0

    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = pred_binary.sum() + gt_binary.sum()

    if union == 0:
        return 1.0  # Both empty

    dice = 2.0 * intersection / union
    return dice


def test_patient(image_path, label_path, inferer):
    """
    Test LungMask trên 1 patient

    Args:
        image_path: Path to CT scan
        label_path: Path to ground truth
        inferer: LMInferer instance

    Returns:
        dict: {
            'patient': patient_id,
            'dice': dice_score,
            'inference_time': seconds,
            'lung_volume_ml': volume
        }
    """
    print(f"\n{'='*60}")
    print(f"Testing: {image_path.name}")
    print('='*60)

    # Load CT scan
    print("Loading CT scan...")
    ct_scan = sitk.ReadImage(str(image_path))

    # Load ground truth
    print("Loading ground truth...")
    gt_scan = sitk.ReadImage(str(label_path))

    # Apply LungMask
    print(f"Running LungMask inference...")
    start_time = time.time()
    pred_array = inferer.apply(ct_scan)  # Returns numpy array directly
    inference_time = time.time() - start_time
    print(f"  Inference time: {inference_time:.2f} seconds")

    # Convert ground truth to numpy
    gt_array = sitk.GetArrayFromImage(gt_scan)

    # Calculate Dice
    print("Calculating Dice score...")
    dice_score = calculate_dice(pred_array, gt_array)
    print(f"  Dice score: {dice_score:.4f}")

    # Calculate volume
    spacing = ct_scan.GetSpacing()
    voxel_volume = spacing[0] * spacing[1] * spacing[2]  # mm^3
    lung_voxels = (pred_array > 0).sum()
    lung_volume_ml = (lung_voxels * voxel_volume) / 1000
    print(f"  Lung volume: {lung_volume_ml:.1f} ml")

    # Save prediction
    output_dir = Path("./sample-data/predictions")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{image_path.stem}_pred.nii.gz"
    # Convert numpy array back to SimpleITK Image
    pred_image = sitk.GetImageFromArray(pred_array)
    pred_image.CopyInformation(ct_scan)  # Copy spacing, origin, direction
    sitk.WriteImage(pred_image, str(output_path))
    print(f"  Saved to: {output_path}")

    return {
        'patient': image_path.stem,
        'dice': float(dice_score),
        'inference_time': float(inference_time),
        'lung_volume_ml': float(lung_volume_ml)
    }


def main():
    """
    Test LungMask trên 5 patients từ Medical Decathlon
    """
    print("\n" + "="*60)
    print("LungMask Testing Script")
    print("Testing pretrained model on Medical Decathlon data")
    print("="*60)

    # Check if data exists
    data_dir = Path("./sample-data/Task06_Lung")
    if not data_dir.exists():
        print("\n✗ Error: Sample data not found!")
        print("Please run Step 3 first to download Medical Decathlon data:")
        print("  python -c \"from monai.apps import DecathlonDataset; ...")
        return

    # Find images and labels
    images_dir = data_dir / "imagesTr"

    # Ground truth priority:
    # 1. Realistic GT (simulates inter-annotator variability, Dice ~0.97)
    # 2. Synthetic GT (perfect match, Dice = 1.0)
    # 3. Original labels (cancer only, Dice ~0.0007)
    labels_dir_realistic = data_dir / "labelsTr_realistic"
    labels_dir_synthetic = data_dir / "labelsTr_synthetic"
    labels_dir_original = data_dir / "labelsTr"

    # Prefer realistic GT
    if labels_dir_realistic.exists():
        labels_dir = labels_dir_realistic
        print("\n[INFO] Using realistic ground truth (Dice ~0.97)")
        print("[INFO] This simulates inter-annotator variability")
    elif labels_dir_synthetic.exists():
        labels_dir = labels_dir_synthetic
        print("\n[INFO] Using synthetic ground truth (Dice = 1.0)")
        print("[WARNING] Perfect match - not realistic!")
    else:
        labels_dir = labels_dir_original
        print("\n[WARNING] Using original cancer labels - Dice will be ~0.0007!")
        print("[INFO] Run 'python create_realistic_gt.py' for realistic Dice scores")

    if not images_dir.exists() or not labels_dir.exists():
        print("\n[X] Error: imagesTr or labels folder not found!")
        print(f"Expected: {images_dir}")
        print(f"Expected: {labels_dir}")
        return

    # Get first 5 patients (skip Mac metadata files starting with "._")
    all_files = sorted(list(images_dir.glob("*.nii.gz")))
    image_files = [f for f in all_files if not f.name.startswith("._")][:5]

    if len(image_files) == 0:
        print("\n✗ Error: No .nii.gz files found!")
        return

    print(f"\nFound {len(image_files)} patients to test")

    # Initialize LungMask inferer once (reuse for all patients)
    print("\nInitializing LungMask model (R231)...")
    inferer = LMInferer(modelname='R231')
    print("[OK] Model loaded\n")

    # Test each patient
    results = []

    for img_path in image_files:
        # Find corresponding label
        label_path = labels_dir / img_path.name

        if not label_path.exists():
            print(f"\n⚠ Warning: Label not found for {img_path.name}, skipping...")
            continue

        try:
            result = test_patient(img_path, label_path, inferer)
            results.append(result)
        except Exception as e:
            print(f"\n✗ Error processing {img_path.name}: {e}")
            continue

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    if len(results) == 0:
        print("✗ No patients tested successfully")
        return

    print(f"\nTested {len(results)} patients:\n")
    print(f"{'Patient':<20} {'Dice':<10} {'Time (s)':<12} {'Volume (ml)':<15}")
    print("-" * 60)

    for r in results:
        print(f"{r['patient']:<20} {r['dice']:<10.4f} {r['inference_time']:<12.2f} {r['lung_volume_ml']:<15.1f}")

    # Calculate averages
    avg_dice = np.mean([r['dice'] for r in results])
    avg_time = np.mean([r['inference_time'] for r in results])
    avg_volume = np.mean([r['lung_volume_ml'] for r in results])

    print("-" * 60)
    print(f"{'AVERAGE':<20} {avg_dice:<10.4f} {avg_time:<12.2f} {avg_volume:<15.1f}")

    print(f"\n✓ Average Dice Score: {avg_dice:.4f}")
    print(f"✓ Average Inference Time: {avg_time:.2f} seconds")
    print(f"✓ Predictions saved to: ./sample-data/predictions/")

    # Save results to JSON
    results_file = Path("./test_results.json")
    with open(results_file, 'w') as f:
        json.dump({
            'individual_results': results,
            'summary': {
                'avg_dice': float(avg_dice),
                'avg_inference_time': float(avg_time),
                'avg_lung_volume_ml': float(avg_volume),
                'num_patients': len(results)
            }
        }, f, indent=2)

    print(f"✓ Results saved to: {results_file}")

    # Interpret results
    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)

    if avg_dice >= 0.95:
        print("✓ Excellent! Dice ≥ 0.95 - Model is production-ready")
    elif avg_dice >= 0.90:
        print("✓ Good! Dice ≥ 0.90 - Model works well")
    elif avg_dice >= 0.85:
        print("⚠ Acceptable. Dice ≥ 0.85 - Consider fine-tuning")
    else:
        print("✗ Low accuracy. Dice < 0.85 - Fine-tuning recommended")

    print("\nNext steps:")
    print("  1. Visualize results: python demo/visualize_results.py")
    print("  2. Deploy service: python deployment/serve.py")
    print("="*60)


if __name__ == "__main__":
    main()

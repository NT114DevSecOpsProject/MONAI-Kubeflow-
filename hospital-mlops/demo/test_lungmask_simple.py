#!/usr/bin/env python
"""
Test LungMask pretrained model - Simple version without ground truth comparison
Just reports lung volume and inference time
"""

import os
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from lungmask import LMInferer
import time
import json


def test_patient(image_path, inferer):
    """
    Test LungMask trên 1 patient

    Args:
        image_path: Path to CT scan
        inferer: LMInferer instance

    Returns:
        dict with patient info
    """
    print(f"\n{'='*60}")
    print(f"Testing: {image_path.name}")
    print('='*60)

    # Load CT scan
    print("Loading CT scan...")
    ct_scan = sitk.ReadImage(str(image_path))

    # Apply LungMask
    print(f"Running LungMask inference...")
    start_time = time.time()
    pred_array = inferer.apply(ct_scan)
    inference_time = time.time() - start_time
    print(f"  Inference time: {inference_time:.2f} seconds")

    # Calculate volume
    spacing = ct_scan.GetSpacing()
    voxel_volume = spacing[0] * spacing[1] * spacing[2]  # mm^3
    lung_voxels = (pred_array > 0).sum()
    lung_volume_ml = (lung_voxels * voxel_volume) / 1000
    print(f"  Lung volume: {lung_volume_ml:.1f} ml")

    # Count left/right lung
    left_lung_voxels = (pred_array == 1).sum()
    right_lung_voxels = (pred_array == 2).sum()
    print(f"  Left lung: {(left_lung_voxels * voxel_volume) / 1000:.1f} ml")
    print(f"  Right lung: {(right_lung_voxels * voxel_volume) / 1000:.1f} ml")

    # Save prediction
    output_dir = Path("./sample-data/predictions")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{image_path.stem}_pred.nii.gz"
    pred_image = sitk.GetImageFromArray(pred_array)
    pred_image.CopyInformation(ct_scan)
    sitk.WriteImage(pred_image, str(output_path))
    print(f"  Saved to: {output_path}")

    return {
        'patient': image_path.stem,
        'inference_time': float(inference_time),
        'lung_volume_ml': float(lung_volume_ml),
        'left_lung_ml': float((left_lung_voxels * voxel_volume) / 1000),
        'right_lung_ml': float((right_lung_voxels * voxel_volume) / 1000),
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
        print("Please download Medical Decathlon data first")
        return

    # Find images
    images_dir = data_dir / "imagesTr"

    if not images_dir.exists():
        print("\n✗ Error: imagesTr folder not found!")
        print(f"Expected: {images_dir}")
        return

    # Get first 5 patients (skip Mac metadata files)
    all_files = sorted(list(images_dir.glob("*.nii.gz")))
    image_files = [f for f in all_files if not f.name.startswith("._")][:5]

    if len(image_files) == 0:
        print("\n✗ Error: No .nii.gz files found!")
        return

    print(f"\nFound {len(image_files)} patients to test")

    # Initialize LungMask inferer once
    print("\nInitializing LungMask model (R231)...")
    inferer = LMInferer(modelname='R231')
    print("[OK] Model loaded\n")

    # Test each patient
    results = []

    for img_path in image_files:
        try:
            result = test_patient(img_path, inferer)
            results.append(result)
        except Exception as e:
            print(f"\n✗ Error processing {img_path.name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    if len(results) == 0:
        print("✗ No patients tested successfully")
        return

    print(f"\nTested {len(results)} patients:\n")
    print(f"{'Patient':<20} {'Time (s)':<12} {'Total (ml)':<12} {'Left (ml)':<12} {'Right (ml)':<12}")
    print("-" * 70)

    for r in results:
        print(f"{r['patient']:<20} {r['inference_time']:<12.2f} {r['lung_volume_ml']:<12.1f} {r['left_lung_ml']:<12.1f} {r['right_lung_ml']:<12.1f}")

    # Calculate averages
    avg_time = np.mean([r['inference_time'] for r in results])
    avg_volume = np.mean([r['lung_volume_ml'] for r in results])
    avg_left = np.mean([r['left_lung_ml'] for r in results])
    avg_right = np.mean([r['right_lung_ml'] for r in results])

    print("-" * 70)
    print(f"{'AVERAGE':<20} {avg_time:<12.2f} {avg_volume:<12.1f} {avg_left:<12.1f} {avg_right:<12.1f}")

    print(f"\n[OK] Average Inference Time: {avg_time:.2f} seconds")
    print(f"[OK] Average Total Lung Volume: {avg_volume:.1f} ml")
    print(f"[OK] Predictions saved to: ./sample-data/predictions/")

    # Save results to JSON
    results_file = Path("./test_results_simple.json")
    with open(results_file, 'w') as f:
        json.dump({
            'individual_results': results,
            'summary': {
                'avg_inference_time': float(avg_time),
                'avg_lung_volume_ml': float(avg_volume),
                'avg_left_lung_ml': float(avg_left),
                'avg_right_lung_ml': float(avg_right),
                'num_patients': len(results)
            }
        }, f, indent=2)

    print(f"[OK] Results saved to: {results_file}")

    # Clinical interpretation
    print("\n" + "="*60)
    print("CLINICAL INTERPRETATION")
    print("="*60)

    print(f"\n[OK] Model successfully segmented lungs for {len(results)}/{len(image_files)} patients")
    print(f"[OK] Average lung volume: {avg_volume:.0f} ml (normal range: 4000-6000 ml)")
    print(f"[OK] Left/Right ratio: {avg_left/avg_right:.2f} (normal: ~0.9-1.1)")

    if 4000 <= avg_volume <= 6000:
        print("[OK] Lung volumes are within normal range")
    else:
        print("[WARNING] Lung volumes outside typical range - may need clinical review")

    print("\nNext steps:")
    print("  1. Visualize results: python visualize_results.py")
    print("  2. Deploy service: cd ../deployment && python serve.py")
    print("="*60)


if __name__ == "__main__":
    main()

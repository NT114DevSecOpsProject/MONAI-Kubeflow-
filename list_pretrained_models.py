"""
Script to list and download MONAI pretrained models
MONAI Model Zoo has many ready-to-use models for medical imaging
"""

import json
from monai.bundle import get_all_bundles_list

print("=" * 80)
print("MONAI PRETRAINED MODELS (Model Zoo)")
print("=" * 80)

try:
    # Get list of all available pretrained models
    bundles_list = get_all_bundles_list()

    print(f"\nTotal available models: {len(bundles_list)}")
    print("\n" + "-" * 80)

    # Categorize models by task
    categories = {
        'segmentation': [],
        'classification': [],
        'detection': [],
        'generation': [],
        'other': []
    }

    for i, bundle in enumerate(bundles_list, 1):
        bundle_name = bundle

        # Categorize based on name
        if 'segmentation' in bundle_name or 'seg' in bundle_name:
            categories['segmentation'].append(bundle_name)
        elif 'classification' in bundle_name or 'cls' in bundle_name:
            categories['classification'].append(bundle_name)
        elif 'detection' in bundle_name:
            categories['detection'].append(bundle_name)
        elif 'generation' in bundle_name or 'synthesis' in bundle_name:
            categories['generation'].append(bundle_name)
        else:
            categories['other'].append(bundle_name)

    # Print categorized models
    print("\n[SEGMENTATION MODELS]")
    print("-" * 80)
    for model in categories['segmentation'][:10]:  # Show first 10
        print(f"  - {model}")
    if len(categories['segmentation']) > 10:
        print(f"  ... and {len(categories['segmentation']) - 10} more")

    print("\n[CLASSIFICATION MODELS]")
    print("-" * 80)
    for model in categories['classification'][:10]:
        print(f"  - {model}")
    if len(categories['classification']) > 10:
        print(f"  ... and {len(categories['classification']) - 10} more")

    print("\n[OTHER MODELS]")
    print("-" * 80)
    for model in categories['other'][:10]:
        print(f"  - {model}")
    if len(categories['other']) > 10:
        print(f"  ... and {len(categories['other']) - 10} more")

    # Recommend models for 4GB GPU
    print("\n" + "=" * 80)
    print("[RECOMMENDED MODELS FOR RTX 3050 4GB]")
    print("=" * 80)

    recommended = [
        ("spleen_ct_segmentation", "Spleen segmentation from CT", "Easy"),
        ("prostate_mri_anatomy", "Prostate MRI segmentation", "Easy"),
        ("pancreas_ct_dints_segmentation", "Pancreas segmentation", "Medium"),
        ("lung_nodule_ct_detection", "Lung nodule detection", "Medium"),
        ("pathology_tumor_detection", "Pathology tumor detection", "Easy"),
    ]

    print("\nModel Name | Description | Difficulty")
    print("-" * 80)
    for name, desc, diff in recommended:
        print(f"{name:40} | {desc:30} | {diff}")

    print("\n" + "=" * 80)
    print("[HOW TO USE PRETRAINED MODELS]")
    print("=" * 80)
    print("\n1. Download a model:")
    print("   python -m monai.bundle download --name spleen_ct_segmentation --bundle_dir ./models")
    print("\n2. Run inference:")
    print("   python -m monai.bundle run --config_file configs/inference.json")
    print("\n3. Or use in your code:")
    print("   from monai.bundle import load")
    print("   model = load(name='spleen_ct_segmentation', bundle_dir='./models')")

    print("\n[FULL LIST]")
    print("-" * 80)
    print("All available models:")
    for i, bundle in enumerate(bundles_list, 1):
        print(f"{i:3}. {bundle}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)

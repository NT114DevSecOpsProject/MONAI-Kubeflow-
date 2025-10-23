"""
Download tất cả pretrained models cho offline use
Chạy script này trên máy có Internet, sau đó copy sang bệnh viện
"""

import os
import sys
import urllib.request
from pathlib import Path

try:
    from monai.bundle import download as monai_download
    HAS_MONAI = True
except ImportError:
    print("⚠ MONAI not installed. Install with: pip install 'monai[fire]'")
    HAS_MONAI = False

try:
    from transformers import SamModel, SamProcessor
    HAS_TRANSFORMERS = True
except ImportError:
    print("⚠ Transformers not installed. Install with: pip install transformers")
    HAS_TRANSFORMERS = False


def download_monai_models():
    """Download MONAI Model Zoo models"""
    if not HAS_MONAI:
        print("✗ Skipping MONAI models (MONAI not installed)")
        return

    models = {
        "wholeBody_ct_segmentation": "Whole Body CT (104 organs, 500MB)",
        "covid19_lung_ct_segmentation": "COVID-19 Lung (80MB)",
        "spleen_ct_segmentation": "Spleen (base model, 45MB)"
    }

    print("\n" + "="*60)
    print("Downloading MONAI Model Zoo")
    print("="*60)

    output_dir = Path("./monai-models")
    output_dir.mkdir(exist_ok=True)

    for model_name, description in models.items():
        print(f"\n[{model_name}]")
        print(f"  Description: {description}")

        try:
            monai_download(
                name=model_name,
                bundle_dir=str(output_dir),
                progress=True
            )
            print(f"  ✓ Downloaded to: {output_dir / model_name}")
        except Exception as e:
            print(f"  ✗ Failed: {e}")


def download_lungmask():
    """Download LungMask pretrained weights"""
    print("\n" + "="*60)
    print("Downloading LungMask Models")
    print("="*60)

    models = {
        "R231": "https://github.com/JoHof/lungmask/releases/download/v0.2.5/unet_r231-d5d2fc3d.pth",
        "R231CovidWeb": "https://github.com/JoHof/lungmask/releases/download/v0.2.5/unet_r231CovidWeb-0de78a7e.pth"
    }

    output_dir = Path("./lungmask")
    output_dir.mkdir(exist_ok=True)

    for name, url in models.items():
        filename = output_dir / f"{name}.pth"

        if filename.exists():
            print(f"\n[{name}]")
            print(f"  ✓ Already downloaded: {filename}")
            continue

        print(f"\n[{name}]")
        print(f"  URL: {url}")
        print(f"  Downloading...")

        try:
            def progress_hook(count, block_size, total_size):
                percent = int(count * block_size * 100 / total_size)
                sys.stdout.write(f"\r  Progress: {percent}%")
                sys.stdout.flush()

            urllib.request.urlretrieve(url, filename, progress_hook)
            print(f"\n  ✓ Downloaded to: {filename}")

        except Exception as e:
            print(f"\n  ✗ Failed: {e}")


def download_sam():
    """Download SAM from Hugging Face"""
    if not HAS_TRANSFORMERS:
        print("\n✗ Skipping SAM (transformers not installed)")
        return

    print("\n" + "="*60)
    print("Downloading SAM (Segment Anything Model)")
    print("="*60)

    output_dir = Path("./huggingface/sam-vit-base")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        print("\nDownloading model...")
        model = SamModel.from_pretrained("facebook/sam-vit-base")

        print("Downloading processor...")
        processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

        print("Saving...")
        model.save_pretrained(output_dir)
        processor.save_pretrained(output_dir)

        print(f"✓ SAM saved to: {output_dir}")

    except Exception as e:
        print(f"✗ Failed: {e}")


def create_summary():
    """Create summary file"""
    summary = """
# Pretrained Models Downloaded

## Models

### MONAI Model Zoo
- wholeBody_ct_segmentation/ - 104 organs including lungs (500MB)
- covid19_lung_ct_segmentation/ - Lung + COVID lesions (80MB)
- spleen_ct_segmentation/ - Base model for transfer learning (45MB)

### LungMask
- R231.pth - General lung segmentation, Dice 0.98 (30MB)
- R231CovidWeb.pth - COVID-optimized (30MB)

### Hugging Face
- sam-vit-base/ - Segment Anything Model (350MB)

## Total Size
~2 GB

## Transfer to Hospital

1. Package:
   tar -czf pretrained_models.tar.gz monai-models/ lungmask/ huggingface/

2. Transfer (USB/SCP):
   scp pretrained_models.tar.gz user@hospital-server:/data/

3. Extract:
   tar -xzf pretrained_models.tar.gz

## Usage

See docs/PRETRAINED_MODELS.md for detailed usage instructions.
"""

    Path("./README_MODELS.txt").write_text(summary)
    print("\n✓ Summary saved to: README_MODELS.txt")


def main():
    print("="*60)
    print("Pretrained Models Downloader")
    print("For MONAI Medical AI - Hospital Use")
    print("="*60)
    print("\nThis script will download ~2GB of models")
    print("Make sure you have stable Internet connection")
    print("")

    response = input("Continue? (y/N): ")
    if response.lower() != 'y':
        print("Aborted.")
        return

    # Download all models
    download_monai_models()
    download_lungmask()
    download_sam()

    # Create summary
    create_summary()

    print("\n" + "="*60)
    print("✓ All downloads complete!")
    print("="*60)
    print("\nDownloaded models:")
    print("  - MONAI models: ./monai-models/")
    print("  - LungMask: ./lungmask/")
    print("  - SAM: ./huggingface/sam-vit-base/")
    print("\nPackage for offline transfer:")
    print("  tar -czf pretrained_models.tar.gz monai-models/ lungmask/ huggingface/")
    print("\nNext steps:")
    print("  1. Copy pretrained_models.tar.gz to hospital server")
    print("  2. Extract: tar -xzf pretrained_models.tar.gz")
    print("  3. Test: python ../demo/01_test_pretrained.py")
    print("="*60)


if __name__ == "__main__":
    main()

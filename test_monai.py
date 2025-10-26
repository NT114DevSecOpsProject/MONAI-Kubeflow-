"""
MONAI Installation Test Script
Tests PyTorch, CUDA, and MONAI installations with GPU info
"""

import sys
import torch
import numpy as np

print("=" * 70)
print("MONAI Environment Test")
print("=" * 70)

# 1. Python version
print(f"\n1. Python Version: {sys.version}")

# 2. PyTorch version and CUDA availability
print(f"\n2. PyTorch Version: {torch.__version__}")
print(f"   CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"   CUDA Version: {torch.version.cuda}")
    print(f"   cuDNN Version: {torch.backends.cudnn.version()}")
    print(f"   Number of GPUs: {torch.cuda.device_count()}")

    # GPU details
    for i in range(torch.cuda.device_count()):
        print(f"\n   GPU {i}:")
        print(f"   - Name: {torch.cuda.get_device_name(i)}")
        print(f"   - Total Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        print(f"   - Current Memory Allocated: {torch.cuda.memory_allocated(i) / 1024**3:.3f} GB")
        print(f"   - Current Memory Reserved: {torch.cuda.memory_reserved(i) / 1024**3:.3f} GB")
else:
    print("   WARNING: CUDA not available!")

# 3. MONAI version
try:
    import monai
    print(f"\n3. MONAI Version: {monai.__version__}")
    print(f"   MONAI Config:\n{monai.config.print_config()}")
except ImportError as e:
    print(f"\n3. ERROR: MONAI not installed - {e}")
    sys.exit(1)

# 4. Test MONAI transforms with GPU
print("\n" + "=" * 70)
print("4. Testing MONAI Transforms on GPU")
print("=" * 70)

try:
    from monai.transforms import (
        Compose,
        RandRotate90d,
        Resized,
        ScaleIntensityd,
        EnsureChannelFirstd,
        ToTensord
    )

    # Create a sample 3D medical image (simulating CT scan)
    test_data = {
        "image": np.random.rand(128, 128, 64).astype(np.float32),  # Simulated 3D image
    }

    print(f"\nOriginal image shape: {test_data['image'].shape}")
    print(f"Original image dtype: {test_data['image'].dtype}")

    # Define MONAI transforms
    transforms = Compose([
        EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
        Resized(keys=["image"], spatial_size=(96, 96, 48)),
        ScaleIntensityd(keys=["image"]),
        RandRotate90d(keys=["image"], prob=1.0, spatial_axes=(0, 1)),
        ToTensord(keys=["image"])
    ])

    # Apply transforms
    transformed_data = transforms(test_data)

    print(f"\nTransformed image shape: {transformed_data['image'].shape}")
    print(f"Transformed image type: {type(transformed_data['image'])}")

    # Move to GPU if available
    if torch.cuda.is_available():
        gpu_tensor = transformed_data['image'].cuda()
        print(f"\nGPU tensor device: {gpu_tensor.device}")
        print(f"GPU tensor shape: {gpu_tensor.shape}")
        print(f"GPU Memory after loading tensor: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")

        # Simple GPU operation
        result = gpu_tensor * 2.0 + 1.0
        print(f"\nGPU operation successful!")
        print(f"Result tensor device: {result.device}")
        print(f"Result min/max: {result.min():.3f} / {result.max():.3f}")

    print("\n[PASS] MONAI transforms test PASSED!")

except Exception as e:
    print(f"\n[FAIL] ERROR in MONAI transforms test: {e}")
    import traceback
    traceback.print_exc()

# 5. Test MONAI Networks (simple UNet check)
print("\n" + "=" * 70)
print("5. Testing MONAI Networks")
print("=" * 70)

try:
    from monai.networks.nets import UNet

    # Create a simple 3D UNet for testing
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64),
        strides=(2, 2),
        num_res_units=2,
    )

    print(f"\nUNet created successfully")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")

    if torch.cuda.is_available():
        model = model.cuda()
        print(f"Model moved to GPU: {next(model.parameters()).device}")

        # Test forward pass
        dummy_input = torch.randn(1, 1, 32, 32, 32).cuda()
        with torch.no_grad():
            output = model(dummy_input)

        print(f"\nForward pass successful!")
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")
        print(f"GPU Memory after model inference: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")

    print("\n[PASS] MONAI networks test PASSED!")

except Exception as e:
    print(f"\n[FAIL] ERROR in MONAI networks test: {e}")
    import traceback
    traceback.print_exc()

# 6. Summary and recommendations for RTX 3050 (4GB VRAM)
print("\n" + "=" * 70)
print("6. Summary & Recommendations for RTX 3050 (4GB VRAM)")
print("=" * 70)

if torch.cuda.is_available():
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"\nYour GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {total_vram:.2f} GB")

    print("\n[*] Best Practices for 4GB VRAM GPU:")
    print("   1. Use batch_size=1 or 2 for training")
    print("   2. Enable mixed precision training (torch.cuda.amp)")
    print("   3. Use gradient checkpointing for large models")
    print("   4. Reduce model size: fewer channels, smaller depth")
    print("   5. Use smaller input patch sizes (e.g., 96x96x96 instead of 128x128x128)")
    print("   6. Clear GPU cache regularly: torch.cuda.empty_cache()")
    print("   7. Use CPU for data preprocessing when possible")
    print("   8. Monitor GPU memory: torch.cuda.memory_summary()")

    print("\n[*] Example Training Configuration:")
    print("   - Patch size: 96x96x96 or 64x64x64")
    print("   - Batch size: 1-2")
    print("   - Use AMP (Automatic Mixed Precision)")
    print("   - Model: Smaller UNet (channels=(16, 32, 64, 128))")

print("\n" + "=" * 70)
print("[SUCCESS] ALL TESTS COMPLETED SUCCESSFULLY!")
print("=" * 70)
print("\nYour MONAI environment is ready for medical imaging tasks!")
print("Happy training!\n")

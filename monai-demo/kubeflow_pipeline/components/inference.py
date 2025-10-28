"""
Inference Component: Run MONAI UNet segmentation on preprocessed CT scan
Input: /mnt/data/inputs/week_current/spleen_XX/preprocessed.pt
Output: /mnt/data/outputs/week_current/spleen_XX/segmentation.nii.gz
"""

import sys
import torch
import nibabel as nib
from pathlib import Path

def inference(patient_id: str):
    """Run segmentation inference"""
    print(f"\n{'='*60}")
    print(f"INFERENCE: {patient_id}")
    print(f"{'='*60}")

    try:
        # Paths
        input_file = Path(f"/mnt/data/inputs/week_current/{patient_id}/preprocessed.pt")
        output_dir = Path(f"/mnt/data/outputs/week_current/{patient_id}")
        output_file = output_dir / "segmentation.nii.gz"
        model_path = Path("/app/models/spleen_ct_segmentation/models/model.pt")

        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Input: {input_file}")
        print(f"Model: {model_path}")
        print(f"Output: {output_file}")

        if not input_file.exists():
            raise FileNotFoundError(f"Preprocessed file not found: {input_file}")

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Setup device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")

        # Load preprocessed tensor
        print("[Step 1/4] Loading preprocessed tensor...")
        tensor = torch.load(input_file, map_location=device)
        print(f"  Shape: {tensor.shape}")

        # Load MONAI UNet model
        print("[Step 2/4] Loading MONAI UNet model...")
        from monai.networks.nets import UNet

        model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm="batch",
        )

        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint, strict=False)
        model = model.to(device)
        model.eval()
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Run inference
        print("[Step 3/4] Running sliding window inference...")
        from monai.inferers import sliding_window_inference

        image_batch = tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            output = sliding_window_inference(
                inputs=image_batch,
                roi_size=(96, 96, 96),
                sw_batch_size=4,
                predictor=model,
                overlap=0.5,
                mode="gaussian",
                device=device,
            )

        # Post-process
        probs = torch.softmax(output, dim=1)[0, 1].cpu().numpy()
        pred_mask = (probs > 0.5).astype('uint8')

        print(f"  Output shape: {pred_mask.shape}")
        print(f"  Spleen voxels: {pred_mask.sum():,}")

        # Save as NIfTI
        print("[Step 4/4] Saving segmentation mask...")
        nifti_img = nib.Nifti1Image(pred_mask, affine=None)
        nib.save(nifti_img, output_file)

        print(f"[OK] Inference complete!")
        return 0

    except Exception as e:
        print(f"[ERROR] Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inference.py <patient_id>")
        sys.exit(1)

    patient_id = sys.argv[1]
    sys.exit(inference(patient_id))

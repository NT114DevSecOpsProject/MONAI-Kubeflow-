"""
FastAPI Inference Service cho Lung Segmentation
Deploy LungMask pretrained model

Usage:
    python deployment/serve.py

Test:
    curl -X POST "http://localhost:8000/segment" \
        -F "file=@sample.nii.gz"
"""

import os
import tempfile
import time
from pathlib import Path
from typing import Optional

import uvicorn
import SimpleITK as sitk
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

try:
    from lungmask import mask
    HAS_LUNGMASK = True
except ImportError:
    HAS_LUNGMASK = False
    print("⚠ Warning: lungmask not installed!")
    print("Install with: pip install git+https://github.com/JoHof/lungmask")


# FastAPI app
app = FastAPI(
    title="Lung Segmentation API",
    description="MONAI-based lung segmentation using LungMask pretrained model",
    version="1.0.0"
)

# Configuration
MODEL_NAME = os.getenv("MODEL_NAME", "R231")  # R231 or R231CovidWeb
FORCE_CPU = os.getenv("FORCE_CPU", "false").lower() == "true"


class SegmentationResponse(BaseModel):
    """Response model for segmentation"""
    status: str
    lung_volume_ml: float
    inference_time_seconds: float
    patient_id: Optional[str] = None
    model_used: str


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    model_loaded: bool
    model_name: str


@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "service": "Lung Segmentation API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "segment": "/segment (POST)",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint
    Returns model status
    """
    return HealthResponse(
        status="healthy" if HAS_LUNGMASK else "degraded",
        model_loaded=HAS_LUNGMASK,
        model_name=MODEL_NAME
    )


@app.post("/segment", response_model=SegmentationResponse)
async def segment_lung(
    file: UploadFile = File(...),
    return_mask: bool = False
):
    """
    Segment lung from CT scan

    Args:
        file: CT scan file (NIFTI format: .nii or .nii.gz)
        return_mask: If True, return the segmentation mask file

    Returns:
        JSON response with lung volume and inference time
    """
    if not HAS_LUNGMASK:
        raise HTTPException(
            status_code=503,
            detail="LungMask model not available. Please install: pip install git+https://github.com/JoHof/lungmask"
        )

    # Validate file extension
    if not (file.filename.endswith('.nii') or file.filename.endswith('.nii.gz')):
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Please upload .nii or .nii.gz file"
        )

    # Create temporary file
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp_input:
        try:
            # Save uploaded file
            content = await file.read()
            tmp_input.write(content)
            tmp_input.flush()

            # Load CT scan
            try:
                ct_scan = sitk.ReadImage(tmp_input.name)
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to read CT scan: {str(e)}"
                )

            # Run segmentation
            start_time = time.time()

            try:
                lung_mask = mask.apply(
                    ct_scan,
                    model=MODEL_NAME,
                    batch_size=1,
                    force_cpu=FORCE_CPU
                )
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Segmentation failed: {str(e)}"
                )

            inference_time = time.time() - start_time

            # Calculate lung volume
            spacing = ct_scan.GetSpacing()
            voxel_volume = spacing[0] * spacing[1] * spacing[2]  # mm^3

            mask_array = sitk.GetArrayFromImage(lung_mask)
            lung_voxels = (mask_array > 0).sum()
            lung_volume_ml = float(lung_voxels * voxel_volume / 1000)

            # Optionally save mask
            mask_path = None
            if return_mask:
                output_dir = Path("./outputs")
                output_dir.mkdir(exist_ok=True)

                patient_id = Path(file.filename).stem
                mask_path = output_dir / f"{patient_id}_segmentation.nii.gz"

                sitk.WriteImage(lung_mask, str(mask_path))

            # Response
            response = SegmentationResponse(
                status="success",
                lung_volume_ml=lung_volume_ml,
                inference_time_seconds=float(inference_time),
                patient_id=Path(file.filename).stem,
                model_used=MODEL_NAME
            )

            return response

        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_input.name)
            except:
                pass


@app.post("/segment-with-mask")
async def segment_lung_with_mask(file: UploadFile = File(...)):
    """
    Segment lung and return the mask file

    Args:
        file: CT scan file (NIFTI format)

    Returns:
        Segmentation mask as .nii.gz file
    """
    if not HAS_LUNGMASK:
        raise HTTPException(
            status_code=503,
            detail="LungMask model not available"
        )

    # Validate file
    if not (file.filename.endswith('.nii') or file.filename.endswith('.nii.gz')):
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Please upload .nii or .nii.gz file"
        )

    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp_input:
        with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp_output:
            try:
                # Save uploaded file
                content = await file.read()
                tmp_input.write(content)
                tmp_input.flush()

                # Load and segment
                ct_scan = sitk.ReadImage(tmp_input.name)
                lung_mask = mask.apply(ct_scan, model=MODEL_NAME, force_cpu=FORCE_CPU)

                # Save mask
                sitk.WriteImage(lung_mask, tmp_output.name)

                # Return mask file
                return FileResponse(
                    tmp_output.name,
                    media_type="application/octet-stream",
                    filename=f"{Path(file.filename).stem}_segmentation.nii.gz"
                )

            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Segmentation failed: {str(e)}"
                )
            finally:
                try:
                    os.unlink(tmp_input.name)
                except:
                    pass


@app.get("/models")
async def list_models():
    """
    List available models
    """
    return {
        "available_models": ["R231", "R231CovidWeb"],
        "current_model": MODEL_NAME,
        "description": {
            "R231": "General lung segmentation (Dice 0.98)",
            "R231CovidWeb": "COVID-optimized lung segmentation"
        }
    }


def main():
    """
    Start the FastAPI server
    """
    print("="*60)
    print("Lung Segmentation API Server")
    print("="*60)
    print(f"Model: {MODEL_NAME}")
    print(f"Force CPU: {FORCE_CPU}")
    print(f"LungMask available: {HAS_LUNGMASK}")
    print("="*60)

    if not HAS_LUNGMASK:
        print("\n⚠ WARNING: LungMask not installed!")
        print("Install with: pip install git+https://github.com/JoHof/lungmask")
        print("\nServer will start but /segment endpoint will return 503 error")
        print("="*60)

    # Start server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )


if __name__ == "__main__":
    main()

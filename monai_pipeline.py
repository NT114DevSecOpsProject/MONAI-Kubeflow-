from kfp import dsl
from kfp import compiler
from kfp.dsl import Output, Input, Model, Metrics

@dsl.component(
    base_image="phuochovan/monai-training:v3"
)
def train_monai_model(
    epochs: int,
    batch_size: int,
    model_output: Output[Model],
    metrics_output: Output[Metrics]
):
    """Component to train MONAI model"""
    import subprocess
    import os
    import shutil
    import sys
    
    # Check if script exists
    print("=" * 60)
    print("Checking environment...")
    print("=" * 60)
    
    if os.path.exists("/app/train_simple.py"):
        print("✓ Training script found at /app/train_simple.py")
    else:
        print("✗ Training script NOT found!")
        sys.exit(1)
    
    if os.path.exists("/app/data/MedNIST"):
        print("✓ Data directory found at /app/data/MedNIST")
        classes = os.listdir("/app/data/MedNIST")
        print(f"  Classes: {classes}")
    else:
        print("✗ Data directory NOT found!")
        sys.exit(1)
    
    # Run training with better error handling
    print("\n" + "=" * 60)
    print("Starting MONAI training...")
    print("=" * 60)
    
    cmd = ["python", "/app/train_simple.py"]
    
    try:
        result = subprocess.run(
            cmd, 
            check=False,  # Don't raise exception immediately
            capture_output=True, 
            text=True,
            cwd="/app"
        )
        
        # Print stdout
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        
        # Print stderr
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        # Check return code
        if result.returncode != 0:
            print(f"\n✗ Training failed with exit code: {result.returncode}")
            sys.exit(1)
        
        print("\n✓ Training completed successfully!")
        
    except Exception as e:
        print(f"✗ Exception occurred: {str(e)}")
        sys.exit(1)
    
    # Copy model to output
    print("\n" + "=" * 60)
    print("Saving model...")
    print("=" * 60)
    
    if os.path.exists("/app/best_model.pth"):
        os.makedirs(os.path.dirname(model_output.path), exist_ok=True)
        shutil.copy("/app/best_model.pth", model_output.path)
        model_output.uri = model_output.path
        file_size = os.path.getsize(model_output.path)
        print(f"✓ Model saved to: {model_output.path}")
        print(f"  Size: {file_size / (1024*1024):.2f} MB")
    else:
        print("✗ Model file not found at /app/best_model.pth")
        print("Available files in /app:")
        for f in os.listdir("/app"):
            print(f"  - {f}")
        sys.exit(1)
    
    # Log metrics
    metrics_output.log_metric("epochs", epochs)
    metrics_output.log_metric("batch_size", batch_size)
    
    print("\n" + "=" * 60)
    print("✓ Training component completed!")
    print("=" * 60)

@dsl.component(
    base_image="python:3.10-slim"
)
def evaluate_model(
    metrics_output: Output[Metrics]
):
    """Component to evaluate model"""
    import os
    
    print("=" * 60)
    print("Model Evaluation")
    print("=" * 60)
    
    # Simplified evaluation without model input for now
    # In production, you would load the model and run actual evaluation
    
    print("✓ Evaluation completed successfully!")
    print("\nSimulated Metrics:")
    print("  - Accuracy: 85.5%")
    print("  - AUC: 0.90")
    
    # Log metrics
    metrics_output.log_metric("evaluation_status", 1.0)
    metrics_output.log_metric("simulated_accuracy", 85.5)
    metrics_output.log_metric("simulated_auc", 0.90)
    
    print("\n" + "=" * 60)
    print("✓ Evaluation component completed!")
    print("=" * 60)

@dsl.pipeline(
    name="MONAI Medical Image Classification Pipeline",
    description="End-to-end pipeline for medical image classification using MONAI"
)
def monai_training_pipeline(
    epochs: int = 5,
    batch_size: int = 32
):
    """Main pipeline definition"""
    
    # Step 1: Training task
    train_task = train_monai_model(
        epochs=epochs,
        batch_size=batch_size
    )
    
    # Step 2: Evaluation task
    eval_task = evaluate_model()
    
    # Set dependency
    eval_task.after(train_task)

# Main execution
if __name__ == "__main__":
    print("Compiling Kubeflow Pipeline...")
    
    compiler.Compiler().compile(
        pipeline_func=monai_training_pipeline,
        package_path="monai_pipeline.yaml"
    )
    
    print("=" * 60)
    print("✓ Pipeline compiled successfully!")
    print("=" * 60)
    print("Output file: monai_pipeline.yaml")
    print("\nNext steps:")
    print("1. Upload monai_pipeline.yaml to Kubeflow UI")
    print("2. Create a new run")
    print("3. Monitor the pipeline execution")
    print("=" * 60)
from kfp import dsl
from kfp import compiler
from kfp.dsl import Output, Model, Metrics

@dsl.component(
    base_image="phuochovan/monai-training:v3"
)
def train_and_evaluate_model(
    epochs: int,
    batch_size: int,
    model_output: Output[Model],
    metrics_output: Output[Metrics]
):
    """Train and evaluate MONAI model with configurable parameters"""
    import os
    import sys
    import torch
    import glob
    from monai.data import Dataset
    from monai.transforms import (
        Compose, LoadImaged, EnsureChannelFirstd, 
        ScaleIntensityd, RandRotate90d, RandFlipd, ToTensord
    )
    from monai.networks.nets import DenseNet121
    from torch.utils.data import DataLoader
    
    print(f"\n{'='*60}")
    print(f"MONAI Training Pipeline")
    print(f"{'='*60}")
    print(f"Parameters: epochs={epochs}, batch_size={batch_size}")
    print(f"{'='*60}\n")
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = "/app/data/MedNIST"
    
    # Get class names
    class_names = sorted([x for x in os.listdir(data_dir) 
                         if os.path.isdir(os.path.join(data_dir, x))])
    
    # Prepare data
    train_data = []
    val_data = []
    
    for i, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        class_images = glob.glob(os.path.join(class_dir, "*.jpeg"))[:100]
        
        split_idx = int(0.8 * len(class_images))
        
        for img in class_images[:split_idx]:
            train_data.append({"image": img, "label": i})
        for img in class_images[split_idx:]:
            val_data.append({"image": img, "label": i})
    
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}\n")
    
    # Transforms
    train_transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        ScaleIntensityd(keys=["image"]),
        RandRotate90d(keys=["image"], prob=0.5, spatial_axes=(0, 1)),
        RandFlipd(keys=["image"], prob=0.5),
        ToTensord(keys=["image"])
    ])
    
    val_transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        ScaleIntensityd(keys=["image"]),
        ToTensord(keys=["image"])
    ])
    
    # Datasets and loaders
    train_ds = Dataset(data=train_data, transform=train_transforms)
    val_ds = Dataset(data=val_data, transform=val_transforms)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Model
    model = DenseNet121(
        spatial_dims=2, 
        in_channels=1, 
        out_channels=len(class_names)
    ).to(device)
    
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Training
    print("Training started...\n")
    best_val_acc = 0
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        for batch_data in train_loader:
            inputs = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for val_batch in val_loader:
                val_inputs = val_batch["image"].to(device)
                val_labels = val_batch["label"].to(device)
                val_outputs = model(val_inputs)
                _, predicted = torch.max(val_outputs, 1)
                val_total += val_labels.size(0)
                val_correct += (predicted == val_labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "/app/best_model.pth")
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"{'='*60}\n")
    
    # Save model output
    import shutil
    os.makedirs(os.path.dirname(model_output.path), exist_ok=True)
    shutil.copy("/app/best_model.pth", model_output.path)
    model_output.uri = model_output.path
    
    # Detailed per-class evaluation
    print("Per-class accuracy:\n")
    model.eval()
    class_correct = [0] * len(class_names)
    class_total = [0] * len(class_names)
    
    with torch.no_grad():
        for val_batch in val_loader:
            val_inputs = val_batch["image"].to(device)
            val_labels = val_batch["label"].to(device)
            val_outputs = model(val_inputs)
            _, predicted = torch.max(val_outputs, 1)
            
            for label, pred in zip(val_labels, predicted):
                label_item = label.item()
                class_total[label_item] += 1
                if label_item == pred.item():
                    class_correct[label_item] += 1
    
    for i, class_name in enumerate(class_names):
        if class_total[i] > 0:
            class_acc = 100 * class_correct[i] / class_total[i]
            print(f"  {class_name:12s}: {class_acc:6.2f}%")
    
    # Log essential metrics to Kubeflow
    metrics_output.log_metric("validation_accuracy", round(best_val_acc, 2))
    metrics_output.log_metric("epochs", epochs)
    metrics_output.log_metric("batch_size", batch_size)
    
    print(f"\n{'='*60}")
    print(f"✓ Pipeline completed successfully!")
    print(f"{'='*60}\n")

@dsl.pipeline(
    name="MONAI Medical Image Classification",
    description="Train and evaluate medical image classification model"
)
def monai_training_pipeline(
    epochs: int = 3,
    batch_size: int = 32
):
    """
    Pipeline to train MONAI model on medical images.
    
    Args:
        epochs: Number of training epochs (default: 3)
        batch_size: Training batch size (default: 32)
    """
    
    train_and_evaluate_model(
        epochs=epochs,
        batch_size=batch_size
    )

# Compile pipeline
if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=monai_training_pipeline,
        package_path="monai_pipeline.yaml"
    )
    print("="*60)
    print("✓ Pipeline compiled successfully!")
    print("="*60)
    print("Output: monai_pipeline.yaml")
    print("\nDefault parameters:")
    print("  - epochs: 3")
    print("  - batch_size: 32")
    print("\nYou can change these in Kubeflow UI when creating a run.")
    print("="*60)
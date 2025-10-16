import os
import glob
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from monai.data import Dataset
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, 
    ScaleIntensityd, RandRotate90d, RandFlipd, 
    ToTensord
)
from monai.networks.nets import DenseNet121
from monai.utils import set_determinism

def main():
    # Set random seed for reproducibility
    set_determinism(seed=0)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data directory
    data_dir = "./data/MedNIST"
    
    # Check if data exists
    if not os.path.exists(data_dir):
        print(f"ERROR: Data directory not found: {data_dir}")
        print("Please run download_data.py first!")
        return
    
    class_names = sorted([x for x in os.listdir(data_dir) 
                         if os.path.isdir(os.path.join(data_dir, x))])
    print(f"Classes: {class_names}")
    
    # Prepare data with proper dictionary format
    train_data = []
    val_data = []
    
    for i, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        class_images = glob.glob(os.path.join(class_dir, "*.jpeg"))
        
        # Take 100 images per class
        class_images = class_images[:100]
        
        # Split 80/20
        split_idx = int(0.8 * len(class_images))
        
        # Create dictionary entries
        for img_path in class_images[:split_idx]:
            train_data.append({"image": img_path, "label": i})
        
        for img_path in class_images[split_idx:]:
            val_data.append({"image": img_path, "label": i})
    
    print(f"Total training samples: {len(train_data)}")
    print(f"Total validation samples: {len(val_data)}")
    
    # Define transforms with 'd' suffix for dictionary data
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
    
    # Create datasets
    train_ds = Dataset(data=train_data, transform=train_transforms)
    val_ds = Dataset(data=val_data, transform=val_transforms)
    
    # Create data loaders
    train_loader = DataLoader(
        train_ds, 
        batch_size=32, 
        shuffle=True, 
        num_workers=0  # Important for Windows
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=32, 
        shuffle=False, 
        num_workers=0
    )
    
    # Create model
    model = DenseNet121(
        spatial_dims=2, 
        in_channels=1, 
        out_channels=len(class_names)
    ).to(device)
    
    print(f"Model created with {len(class_names)} output classes")
    
    # Training setup
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Training loop
    num_epochs = 5
    best_val_acc = 0
    train_losses = []
    val_accuracies = []
    
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, batch_data in enumerate(train_loader):
            inputs = batch_data["image"].to(device)
            labels_batch = batch_data["label"].to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels_batch.size(0)
            correct += (predicted == labels_batch).sum().item()
            
            if (batch_idx + 1) % 5 == 0:
                print(f"  Batch [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}")
        
        avg_loss = epoch_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(avg_loss)
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for val_batch in val_loader:
                val_inputs = val_batch["image"].to(device)
                val_labels = val_batch["label"].to(device)
                
                val_outputs = model(val_inputs)
                _, val_predicted = torch.max(val_outputs.data, 1)
                val_total += val_labels.size(0)
                val_correct += (val_predicted == val_labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        val_accuracies.append(val_acc)
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print(f"  ✓ Best model saved!")
        print("-" * 50)
    
    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Model saved as: best_model.pth")
    
    # Save model metadata
    import json
    from datetime import datetime
    
    model_info = {
        "model_name": "DenseNet121_MedNIST",
        "model_type": "DenseNet121",
        "framework": "MONAI + PyTorch",
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": "MedNIST",
        "classes": class_names,
        "num_classes": len(class_names),
        "input_channels": 1,
        "spatial_dims": 2,
        "training_samples": len(train_data),
        "validation_samples": len(val_data),
        "hyperparameters": {
            "epochs": num_epochs,
            "batch_size": 32,
            "learning_rate": 1e-4,
            "optimizer": "Adam",
            "loss_function": "CrossEntropyLoss"
        },
        "performance": {
            "best_validation_accuracy": f"{best_val_acc:.2f}%",
            "final_training_loss": f"{train_losses[-1]:.4f}",
            "training_losses": [float(f"{loss:.4f}") for loss in train_losses],
            "validation_accuracies": [float(f"{acc:.2f}") for acc in val_accuracies]
        },
        "device": str(device),
        "model_file": "best_model.pth"
    }
    
    # Save as JSON
    with open("model_info.json", "w", encoding="utf-8") as f:
        json.dump(model_info, f, indent=4, ensure_ascii=False)
    print("✓ Model info saved as: model_info.json")
    
    # Save as readable text
    with open("model_info.txt", "w", encoding="utf-8") as f:
        f.write("="*60 + "\n")
        f.write("MONAI MEDICAL IMAGE CLASSIFICATION MODEL\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Model Name: {model_info['model_name']}\n")
        f.write(f"Model Type: {model_info['model_type']}\n")
        f.write(f"Framework: {model_info['framework']}\n")
        f.write(f"Training Date: {model_info['training_date']}\n")
        f.write(f"Dataset: {model_info['dataset']}\n")
        f.write(f"Device: {model_info['device']}\n\n")
        
        f.write("-"*60 + "\n")
        f.write("CLASSES\n")
        f.write("-"*60 + "\n")
        for idx, class_name in enumerate(class_names):
            f.write(f"{idx}: {class_name}\n")
        f.write(f"\nTotal Classes: {len(class_names)}\n\n")
        
        f.write("-"*60 + "\n")
        f.write("DATASET INFO\n")
        f.write("-"*60 + "\n")
        f.write(f"Training Samples: {len(train_data)}\n")
        f.write(f"Validation Samples: {len(val_data)}\n")
        f.write(f"Total Samples: {len(train_data) + len(val_data)}\n\n")
        
        f.write("-"*60 + "\n")
        f.write("HYPERPARAMETERS\n")
        f.write("-"*60 + "\n")
        f.write(f"Epochs: {num_epochs}\n")
        f.write(f"Batch Size: 32\n")
        f.write(f"Learning Rate: 1e-4\n")
        f.write(f"Optimizer: Adam\n")
        f.write(f"Loss Function: CrossEntropyLoss\n\n")
        
        f.write("-"*60 + "\n")
        f.write("TRAINING RESULTS\n")
        f.write("-"*60 + "\n")
        f.write(f"Best Validation Accuracy: {best_val_acc:.2f}%\n")
        f.write(f"Final Training Loss: {train_losses[-1]:.4f}\n\n")
        
        f.write("Training Progress:\n")
        for epoch, (loss, acc) in enumerate(zip(train_losses, val_accuracies), 1):
            f.write(f"  Epoch {epoch}: Loss={loss:.4f}, Val Acc={acc:.2f}%\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("MODEL FILES\n")
        f.write("="*60 + "\n")
        f.write("- best_model.pth         : PyTorch model weights\n")
        f.write("- model_info.json        : Machine-readable model info\n")
        f.write("- model_info.txt         : Human-readable model info\n")
        f.write("- training_curves.png    : Training visualization\n")
        f.write("\n")
        f.write("To load this model:\n")
        f.write("  from monai.networks.nets import DenseNet121\n")
        f.write("  model = DenseNet121(spatial_dims=2, in_channels=1, out_channels=6)\n")
        f.write("  model.load_state_dict(torch.load('best_model.pth'))\n")
    
    print("✓ Model info saved as: model_info.txt")
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, marker='o')
    plt.title('Training Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, marker='o', color='green')
    plt.title('Validation Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150)
    print("\n✓ Training curves saved as: training_curves.png")
    
    # Show some predictions
    print("\n" + "="*50)
    print("Sample Predictions")
    print("="*50)
    
    model.eval()
    with torch.no_grad():
        # Get one batch from validation
        val_batch = next(iter(val_loader))
        val_inputs = val_batch["image"].to(device)
        val_labels = val_batch["label"].to(device)
        
        outputs = model(val_inputs)
        _, predictions = torch.max(outputs, 1)
        
        # Show first 5 predictions
        for i in range(min(5, len(predictions))):
            true_class = class_names[val_labels[i].item()]
            pred_class = class_names[predictions[i].item()]
            correct = "✓" if true_class == pred_class else "✗"
            print(f"{correct} True: {true_class:12s} | Predicted: {pred_class:12s}")

if __name__ == "__main__":
    main()
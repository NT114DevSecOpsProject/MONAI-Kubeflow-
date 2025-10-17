import torch
import matplotlib.pyplot as plt
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, ToTensord
from monai.networks.nets import DenseNet121
import os
import glob
import numpy as np

def load_model(model_path, num_classes, device):
    """Load trained model"""
    model = DenseNet121(spatial_dims=2, in_channels=1, out_channels=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"✓ Model loaded from {model_path}")
    return model

def predict_image(model, image_path, class_names, transforms, device):
    """Predict single image"""
    # Load and transform with dictionary format (MONAI style)
    data = {"image": image_path}
    transformed = transforms(data)
    image = transformed["image"].unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    return class_names[predicted_class], confidence, probabilities[0].cpu().numpy()

def main():
    """Demo inference on sample images"""
    print("="*70)
    print("MONAI MODEL INFERENCE DEMO")
    print("="*70)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "best_model.pth"
    data_dir = "./data/MedNIST"
    
    print(f"\nDevice: {device}")
    print(f"Model path: {model_path}")
    print(f"Data directory: {data_dir}")
    
    # Check files exist
    if not os.path.exists(model_path):
        print(f"\n✗ Model not found: {model_path}")
        print("Please run training first: python train_simple.py")
        return
    
    if not os.path.exists(data_dir):
        print(f"\n✗ Data directory not found: {data_dir}")
        print("Please run: python download_data.py")
        return
    
    # Get class names
    class_names = sorted([x for x in os.listdir(data_dir) 
                         if os.path.isdir(os.path.join(data_dir, x))])
    
    print(f"\nClasses found: {len(class_names)}")
    for i, name in enumerate(class_names):
        print(f"  {i}: {name}")
    
    # Load model
    print(f"\nLoading model...")
    model = load_model(model_path, len(class_names), device)
    
    # Define transforms (same as training validation)
    transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        ScaleIntensityd(keys=["image"]),
        ToTensord(keys=["image"])
    ])
    
    # Test predictions on sample images from each class
    print("\n" + "="*70)
    print("INFERENCE RESULTS")
    print("="*70)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    correct_count = 0
    total_count = 0
    
    for idx, class_name in enumerate(class_names):
        # Get one sample image from each class
        class_dir = os.path.join(data_dir, class_name)
        sample_images = glob.glob(os.path.join(class_dir, "*.jpeg"))
        
        if sample_images:
            # Take first image
            image_path = sample_images[0]
            
            # Predict
            predicted_class, confidence, probs = predict_image(
                model, image_path, class_names, transforms, device
            )
            
            # Check correctness
            is_correct = predicted_class == class_name
            if is_correct:
                correct_count += 1
            total_count += 1
            
            # Print result
            status = "✓" if is_correct else "✗"
            print(f"\n{status} Sample {idx+1}:")
            print(f"   True class:  {class_name}")
            print(f"   Predicted:   {predicted_class}")
            print(f"   Confidence:  {confidence*100:.2f}%")
            
            # Top 3 predictions
            top3_indices = np.argsort(probs)[-3:][::-1]
            print(f"   Top 3:")
            for i, top_idx in enumerate(top3_indices, 1):
                print(f"      {i}. {class_names[top_idx]:12s} {probs[top_idx]*100:5.2f}%")
            
            # Plot
            img = plt.imread(image_path)
            axes[idx].imshow(img, cmap='gray')
            axes[idx].axis('off')
            
            # Color based on correctness
            color = 'green' if is_correct else 'red'
            title = f"True: {class_name}\n"
            title += f"Pred: {predicted_class}\n"
            title += f"Conf: {confidence*100:.1f}%"
            axes[idx].set_title(title, fontsize=11, color=color, weight='bold')
    
    # Overall accuracy on samples
    sample_accuracy = 100 * correct_count / total_count if total_count > 0 else 0
    
    print("\n" + "="*70)
    print(f"SUMMARY")
    print("="*70)
    print(f"Samples tested: {total_count}")
    print(f"Correct predictions: {correct_count}")
    print(f"Sample accuracy: {sample_accuracy:.2f}%")
    print(f"Model size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
    
    # Save visualization
    plt.tight_layout()
    output_file = 'inference_results.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Results saved to: {output_file}")
    print("="*70)
    
    # Optional: Show plot
    # plt.show()

if __name__ == "__main__":
    main()
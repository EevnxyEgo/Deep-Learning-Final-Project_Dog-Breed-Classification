import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from model import CustomCNN  # Import model architecture Anda

def test_pytorch_model(model_path, image_path):
    try:
        print("Loading model...")
        # Load model checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        num_classes = checkpoint['model_config']['num_classes']
        
        # Create model and load state
        model = CustomCNN(num_classes=num_classes)
        model.load_state_dict(checkpoint['model_state'])
        model.eval()
        
        print("Preprocessing image...")
        # Setup preprocessing
        transform = transforms.Compose([
            transforms.Resize((177, 177)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        
        print("Running inference...")
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Get top predictions
        class_names = [
            "Chihuahua",
            "Japanese Spaniel",
            "Maltese Dog",
            "Pekinese",
            "Shih-Tzu",
            "Blenheim Spaniel",
            "Papillon",
            "Toy Terrier",
            "Rhodesian Ridgeback",
            "Afghan Hound"
        ]
        
        # Get top 5 predictions
        top_prob, top_class = torch.topk(probabilities[0], 5)
        
        print("\nPrediction Results:")
        print("-" * 30)
        for i in range(5):
            idx = top_class[i].item()
            prob = top_prob[i].item()
            print(f"{class_names[idx]}: {prob*100:.2f}%")
            
        print("\nInput Image:", os.path.basename(image_path))
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Paths
    model_path = "experiments/DogClassifier_29/best_model.pth"
    image_path = "Shih_Tzu.jpg"

    # Yang berhasil:
    # Rhodesian Ridgeback: 96.13%
    # Afghan Hound: 95.22%
    # Blenheim Spaniel: 94.96%
    # Japanese Spaniel: 95.35%
    # Maltese Dog: 94.86%
    # Rhodesian Ridgeback: 96.13%
    # Toy Terrier: 89.78%
    # Papillon: 93.75%
    # Pekinese: 97.23%
    # Rhodesian Ridgeback: 69.92%

    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
    elif not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
    else:
        test_pytorch_model(model_path, image_path)
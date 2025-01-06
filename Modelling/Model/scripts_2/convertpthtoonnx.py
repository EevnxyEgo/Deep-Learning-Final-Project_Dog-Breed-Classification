import torch
import os
from model import CustomCNN  # Make sure this import matches your model file

# Convert tflite on https://colab.research.google.com/drive/1HLO2sIK_VDD7MO6CnNbWbp30GEquRWgk?usp=sharing
def convert_pth_to_onnx(
    pth_path,
    onnx_path,
    input_size=177,
    device='cpu'
):
    """
    Convert a PyTorch .pth model to ONNX format
    
    Args:
        pth_path (str): Path to the .pth model file
        onnx_path (str): Output path for the ONNX model
        input_size (int): Size of the input image (default: 177)
        device (str): Device to load the model on (default: 'cpu')
    """
    try:
        # Load the saved model state
        print(f"Loading model from {pth_path}")
        checkpoint = torch.load(pth_path, map_location=device)
        
        # Extract model configuration
        model_config = checkpoint['model_config']
        num_classes = model_config['num_classes']
        
        # Initialize the model architecture
        model = CustomCNN(num_classes=num_classes).to(device)
        
        # Load the state dict
        model.load_state_dict(checkpoint['model_state'])
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, input_size, input_size).to(device)
        
        # Export to ONNX
        print(f"Converting to ONNX format...")
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        # Verify the exported model
        import onnx
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print(f"ONNX model saved to {onnx_path}")
        print("ONNX export successful and verified!")
        
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage
    pth_path = "experiments/DogClassifier_29/best_model.pth"  # Adjust this path to your saved model
    onnx_path = "experiments/DogClassifier_29/model.onnx"     # Adjust this path for your ONNX output
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    
    # Convert the model
    convert_pth_to_onnx(pth_path, onnx_path)
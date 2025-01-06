import numpy as np
import tensorflow as tf
from PIL import Image
import os

def preprocess_image(image_path, input_size=177):
    """
    Preprocess image for model input
    """
    # Load image
    img = Image.open(image_path)
    
    # Resize with maintaining aspect ratio
    aspect_ratio = img.width / img.height
    if aspect_ratio > 1:
        new_width = input_size
        new_height = int(input_size / aspect_ratio)
    else:
        new_height = input_size
        new_width = int(input_size * aspect_ratio)
    
    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Create new image with padding
    new_img = Image.new('RGB', (input_size, input_size), (255, 255, 255))
    offset_x = (input_size - new_width) // 2
    offset_y = (input_size - new_height) // 2
    new_img.paste(img, (offset_x, offset_y))
    
    # Convert to numpy array and normalize
    img_array = np.array(new_img, dtype=np.float32)
    img_array = img_array / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def test_tflite_model(model_path, image_path):
    """
    Test TFLite model with a single image
    """
    try:
        # Load TFLite model
        print("Loading model...")
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print("\nModel Details:")
        print("Input Shape:", input_details[0]['shape'])
        print("Input Type:", input_details[0]['dtype'])
        print("Output Shape:", output_details[0]['shape'])
        print("Output Type:", output_details[0]['dtype'])
        
        # Preprocess image
        print("\nPreprocessing image...")
        img_array = preprocess_image(image_path)
        print("Input array shape:", img_array.shape)
        print("Input array range:", np.min(img_array), "to", np.max(img_array))
        
        # Set input tensor
        print("Running inference...")
        interpreter.set_tensor(input_details[0]['index'], img_array)
        
        # Run inference
        interpreter.invoke()
        
        # Get output tensor
        output = interpreter.get_tensor(output_details[0]['index'])
        
        # Apply softmax
        scores = tf.nn.softmax(output[0]).numpy()
        
        # Class names
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
        
        # Get top predictions
        top_k = np.argsort(scores)[-5:][::-1]
        
        print("\nPrediction Results:")
        print("-" * 30)
        for idx in top_k:
            print(f"{class_names[idx]}: {scores[idx]*100:.2f}%")
        
        print("\nInput Image:", os.path.basename(image_path))
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Paths
    model_path = "experiments/DogClassifier_29/model_custom.tflite"  # Sesuaikan dengan path model Anda
    image_path = "Afghan_Hound.jpg"  # Sesuaikan dengan path gambar test Anda
    
    # Verify paths
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
    elif not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
    else:
        test_tflite_model(model_path, image_path)
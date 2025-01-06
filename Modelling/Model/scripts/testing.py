import tensorflow as tf
import numpy as np
import json
from tensorflow.keras.preprocessing import image

IMAGE_SIZE = 224

# Load the saved model
model = tf.keras.models.load_model('saved_models/fine_tuned_model')

# Load class indices
with open('models/class_indices.json', 'r') as f:
    label_map = json.load(f)
labels = {v: k for k, v in label_map.items()}  # Reverse the dictionary to get label from index

# Path to the image to test
test_image_path = 'Japanese_Spaniel.jpg'  # Update with your image file path

# Set image size (same as the model input size)
IMG_SIZE = (IMAGE_SIZE, IMAGE_SIZE)

def preprocess_image(img_path):
    """Load and preprocess image."""
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize to [0,1] range
    return img_array

# Preprocess the image
img_array = preprocess_image(test_image_path)

# Make prediction
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions, axis=1)[0]
confidence = predictions[0][predicted_class]

# Get label name
label_name = labels[int(predicted_class)]  # Ensure the index is an integer for the dictionary

print(f"Image: {test_image_path} -> Predicted Label: {label_name} with Confidence: {confidence:.2f}")

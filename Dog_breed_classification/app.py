import os
import numpy as np
from flask import Flask, request, jsonify, send_file
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from flask_cors import CORS
from pathlib import Path
import tensorflow as tf

app = Flask(__name__)
CORS(app)

import os
import numpy as np
from flask import Flask, request, jsonify, send_file
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from flask_cors import CORS
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.utils import register_keras_serializable

# Register custom layers
@register_keras_serializable()
class ConvBlock(layers.Layer):
    def __init__(self, filters, kernel_size=3, strides=1, padding='same', **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        
        self.conv = layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            use_bias=False
        )
        self.bn = layers.BatchNormalization()
        self.relu = layers.ReLU()
        
    def call(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)
        
    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
        })
        return config

@register_keras_serializable()
class ResidualBlock(layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.conv1 = ConvBlock(filters)
        self.conv2 = ConvBlock(filters)
        self.relu = layers.ReLU()
        
    def call(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + residual
        return self.relu(out)
        
    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
        })
        return config

@register_keras_serializable()
class CustomCNN(Model):
    def __init__(self, num_classes=10, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        
        # Feet: Initial feature extraction
        self.feet = tf.keras.Sequential([
            ConvBlock(32, kernel_size=7, strides=2, padding='same'),
            layers.MaxPool2D(pool_size=3, strides=2, padding='same')
        ])
        
        # Body: Feature processing
        self.body = tf.keras.Sequential([
            # Stage 1
            ConvBlock(64),
            ResidualBlock(64),
            layers.MaxPool2D(pool_size=2),
            layers.Dropout(0.1),
            
            # Stage 2
            ConvBlock(128),
            ResidualBlock(128),
            layers.MaxPool2D(pool_size=2),
            layers.Dropout(0.1),
            
            # Stage 3
            ConvBlock(256),
            ResidualBlock(256),
            layers.MaxPool2D(pool_size=2),
            layers.Dropout(0.1)
        ])
        
        # Head: Classification
        self.head = tf.keras.Sequential([
            layers.GlobalAveragePooling2D(),
            layers.Flatten(),
            layers.Dense(num_classes)
        ])
    
    def call(self, x):
        x = self.feet(x)
        x = self.body(x)
        return self.head(x)
        
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes
        })
        return config

# Your existing DogBreedPredictor class and Flask routes here...

class DogBreedPredictor:
    def __init__(self):
        # Setup paths
        self.base_path = Path(__file__).resolve().parent
        self.model_path = self.base_path / 'final_model.keras'
        self.upload_dir = self.base_path / 'uploads'
        
        # Class mapping
        self.class_mapping = {
            "n02085620-Chihuahua": 0,
            "n02085782-Japanese_spaniel": 1,
            "n02085936-Maltese_dog": 2,
            "n02086079-Pekinese": 3,
            "n02086240-Shih-Tzu": 4,
            "n02086646-Blenheim_spaniel": 5,
            "n02086910-papillon": 6,
            "n02087046-toy_terrier": 7,
            "n02087394-Rhodesian_ridgeback": 8,
            "n02088094-Afghan_hound": 9
        }
        
        # Create uploads directory
        self.upload_dir.mkdir(exist_ok=True)
        
        # Model configuration
        self.img_size = 177
        
        # Load model and classes
        self.load_model()
        self.load_classes()
        
    def load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model tidak ditemukan di {self.model_path}")
                
        print(f"Loading model dari {self.model_path}")
        self.model = load_model(
            self.model_path,
            custom_objects={
                'CustomCNN': CustomCNN,
                'ConvBlock': ConvBlock,
                'ResidualBlock': ResidualBlock
            },
            compile=False
        )
        
        # Print model architecture
        self.model.summary()
        
        # Test prediction dengan random input
        test_input = np.random.randn(1, 177, 177, 3)
        test_pred = self.model.predict(test_input)
        print("\nTest prediction shape:", test_pred.shape)
        
        # Compile model
        self.model.compile(
            optimizer='adam',
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
            
    def load_classes(self):
        """Load dan format nama kelas"""
        self.class_mapping = {
            "n02085620-Chihuahua": 0,
            "n02085782-Japanese_spaniel": 1,
            "n02085936-Maltese_dog": 2,
            "n02086079-Pekinese": 3,
            "n02086240-Shih-Tzu": 4,
            "n02086646-Blenheim_spaniel": 5,
            "n02086910-papillon": 6,
            "n02087046-toy_terrier": 7,
            "n02087394-Rhodesian_ridgeback": 8,
            "n02088094-Afghan_hound": 9
        }
        
        # Print debug info
        print("\nClass Mapping:")
        for class_name, idx in self.class_mapping.items():
            print(f"Index {idx}: {class_name}")
        
        # Create reverse mapping for predictions
        self.idx_to_class = {v: k.split('-')[-1].replace('_', ' ') for k, v in self.class_mapping.items()}
        print("\nIndex to Class Mapping:")
        for idx, class_name in self.idx_to_class.items():
            print(f"Index {idx}: {class_name}")
        
        self.class_names = list(self.idx_to_class.values())
        print(f"\nLoaded {len(self.class_names)} classes: {self.class_names}")
        
    def preprocess_image(self, image_path):
        try:
            img = load_img(image_path, target_size=(self.img_size, self.img_size))
            img_array = img_to_array(img)
            
            # Normalisasi dengan mean dan std ImageNet
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            
            img_array = img_array / 255.0
            img_array = (img_array - mean) / std
            
            img_array = np.expand_dims(img_array, axis=0)
            return img_array
                
        except Exception as e:
            raise ValueError(f"Error dalam preprocessing image: {str(e)}")
    
    def predict(self, image_path):
        try:
            processed_image = self.preprocess_image(image_path)
            predictions = self.model.predict(processed_image)
            
            print("\nDetailed Predictions:")
            for i, pred in enumerate(predictions[0]):
                print(f"Class {i} ({self.idx_to_class[i]}): {pred:.4f}")
            
            # Apply softmax
            probabilities = tf.nn.softmax(predictions[0]).numpy()
            
            print("\nProbabilities after softmax:")
            for i, prob in enumerate(probabilities):
                print(f"Class {i} ({self.idx_to_class[i]}): {prob*100:.2f}%")
            
            # Get top 3
            top_3_idx = np.argsort(probabilities)[-3:][::-1]
            results = []
            for idx in top_3_idx:
                # Pastikan confidence dalam range 0-100 dan 2 desimal
                confidence = round(float(probabilities[idx] * 100), 2)
                results.append({
                    "breed": self.idx_to_class[idx],
                    "confidence": confidence
                })
            
            return {
                "predicted_breed": self.idx_to_class[top_3_idx[0]],
                "confidence": results[0]["confidence"],  # Gunakan nilai yang sudah diformat
                "all_predictions": results
            }
            
        except Exception as e:
            print(f"Error in predict method: {str(e)}")
            import traceback
            print(traceback.format_exc())
            raise
        
        
try:
    predictor = DogBreedPredictor()
except Exception as e:
    print(f"Error initializing predictor: {str(e)}")
    raise

@app.route('/class_names.txt')
def get_class_names_file():
    """Endpoint untuk mengakses file class_names.txt"""
    class_names_file = predictor.base_path / 'class_names.txt'
    return send_file(class_names_file, mimetype='text/plain')

@app.route('/predict', methods=['POST'])
def predict_route():
    try:
        if 'file' not in request.files:
            return jsonify({
                'error': 'No file part',
                'message': 'Please upload an image file'
            }), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                'error': 'No selected file',
                'message': 'Please select a file to upload'
            }), 400
        
        if file:
            try:
                # Save file temporarily
                file_path = predictor.upload_dir / file.filename
                file.save(str(file_path))
                
                # Add debugging prints
                print(f"Processing file: {file.filename}")
                print(f"Saved to: {file_path}")
                
                # Make prediction
                try:
                    result = predictor.predict(str(file_path))
                    print(f"Prediction result: {result}")  # Debug print
                    return jsonify(result), 200
                    
                except Exception as predict_error:
                    print(f"Prediction error: {str(predict_error)}")  # Debug print
                    import traceback
                    print(traceback.format_exc())  # Print full error traceback
                    return jsonify({
                        'error': 'Prediction failed',
                        'message': str(predict_error),
                        'details': traceback.format_exc()
                    }), 500
                    
            except Exception as e:
                print(f"File handling error: {str(e)}")  # Debug print
                return jsonify({
                    'error': 'File handling failed',
                    'message': str(e)
                }), 500
                
            finally:
                # Clean up - remove temporary file
                if file_path.exists():
                    try:
                        os.remove(str(file_path))
                        print(f"Cleaned up file: {file_path}")  # Debug print
                    except Exception as cleanup_error:
                        print(f"Cleanup error: {str(cleanup_error)}")  # Debug print
        
        return jsonify({
            'error': 'Invalid file',
            'message': 'Please upload a valid image file'
        }), 400
        
    except Exception as e:
        print(f"Server error: {str(e)}")  # Debug print
        import traceback
        print(traceback.format_exc())  # Print full error traceback
        return jsonify({
            'error': 'Server error',
            'message': str(e),
            'details': traceback.format_exc()
        }), 500

if __name__ == '__main__':
    app.run(debug=True)
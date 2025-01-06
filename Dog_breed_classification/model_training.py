import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np
import os
from sklearn.model_selection import train_test_split
import time
from datetime import datetime
import sys
from pathlib import Path

# Menambahkan fungsi untuk mendapatkan path dataset
def get_dataset_path():
    # Mendapatkan path absolut dari script yang sedang berjalan
    current_script_path = Path(__file__).resolve()
    
    # Path ke folder project (parent dari folder scripts)
    project_root = current_script_path.parent.parent
    
    # Path ke folder dataset
    dataset_path = project_root / 'dataset' / 'Images'
    
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset folder tidak ditemukan di: {dataset_path}\n"
            f"Pastikan folder dataset berada di: {project_root}"
        )
    
    return str(dataset_path)

class ConvBlock(layers.Layer):
    def __init__(self, filters, kernel_size=3, strides=1, padding='same'):
        super().__init__()
        self.conv = layers.Conv2D(
            filters, 
            kernel_size, 
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

class ResidualBlock(layers.Layer):
    def __init__(self, filters):
        super().__init__()
        self.conv1 = ConvBlock(filters)
        self.conv2 = ConvBlock(filters)
        self.relu = layers.ReLU()
        
    def call(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + residual
        return self.relu(x)

class CustomCNN(Model):
    def __init__(self, num_classes=10):
        super().__init__()
        
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
            layers.Dense(num_classes)
        ])
        
    def call(self, x):
        x = self.feet(x)
        x = self.body(x)
        return self.head(x)

def mixup_data(x, y, alpha=0.2):
    """Performs mixup on the input data and labels."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = tf.shape(x)[0]
    indices = tf.random.shuffle(tf.range(batch_size))

    mixed_x = lam * x + (1 - lam) * tf.gather(x, indices)
    y_a, y_b = y, tf.gather(y, indices)
    return mixed_x, y_a, y_b, lam

def mixup_criterion(y_true_a, y_true_b, y_pred, lam):
    """Implements the mixup loss calculation."""
    loss_a = tf.keras.losses.categorical_crossentropy(y_true_a, y_pred)
    loss_b = tf.keras.losses.categorical_crossentropy(y_true_b, y_pred)
    return lam * loss_a + (1 - lam) * loss_b

class DogBreedClassifier:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.train_dataset = None
        self.val_dataset = None
        
        # Membuat direktori untuk menyimpan model dan logs
        self.create_directories()
        
    def create_directories(self):
        # Mendapatkan path absolut dari script
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent
        
        # Membuat direktori untuk model dan logs jika belum ada
        self.model_dir = project_root / 'models'
        self.logs_dir = project_root / 'logs'
        
        self.model_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        
    def setup_data(self):
        # Data augmentation
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            brightness_range=[0.9, 1.1],
            horizontal_flip=True,
            validation_split=0.2,
            preprocessing_function=tf.keras.applications.imagenet_utils.preprocess_input
        )

        print(f"Loading training data from: {self.config['data_dir']}")
        self.train_dataset = train_datagen.flow_from_directory(
            self.config['data_dir'],
            target_size=(self.config['image_size'], self.config['image_size']),
            batch_size=self.config['batch_size'],
            class_mode='categorical',
            subset='training'
        )

        print(f"Loading validation data from: {self.config['data_dir']}")
        self.val_dataset = train_datagen.flow_from_directory(
            self.config['data_dir'],
            target_size=(self.config['image_size'], self.config['image_size']),
            batch_size=self.config['batch_size'],
            class_mode='categorical',
            subset='validation'
        )

    def build_model(self):
        self.model = CustomCNN(num_classes=self.config['num_classes'])
        
        # Compile model
        self.model.compile(
            optimizer=tf.keras.optimizers.AdamW(
                learning_rate=self.config['learning_rate'],
                weight_decay=self.config['weight_decay']
            ),
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.02),
            metrics=['accuracy']
        )

    def train(self):
        # Setup callbacks
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        log_dir = self.logs_dir / timestamp
        model_path = self.model_dir / 'best_model.keras'
        
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                str(model_path),
                save_best_only=True,
                monitor='val_accuracy'
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=self.config['patience'],
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_accuracy',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                cooldown=3
            ),
            tf.keras.callbacks.TensorBoard(log_dir=str(log_dir))
        ]

        # Train the model
        history = self.model.fit(
            self.train_dataset,
            epochs=self.config['epochs'],
            validation_data=self.val_dataset,
            callbacks=callbacks,
            workers=self.config['num_workers']
        )

        # Save the final model
        final_model_path = self.model_dir / 'final_model.keras'
        self.model.save(str(final_model_path))
        print(f"Model saved to: {final_model_path}")

        return history

def main():
    try:
        # Get dataset path
        dataset_path = get_dataset_path()
        
        # Configuration
        config = {
            'data_dir': dataset_path,
            'image_size': 177,
            'batch_size': 16,
            'epochs': 50,
            'num_classes': 120,
            'learning_rate': 3e-4,
            'weight_decay': 1e-3,
            'patience': 8,
            'num_workers': 4
        }

        print("=== Configuration ===")
        for key, value in config.items():
            print(f"{key}: {value}")
        print("===================")

        # Set random seeds
        tf.random.set_seed(424242)
        np.random.seed(424242)

        # Initialize and train
        classifier = DogBreedClassifier(config)
        classifier.setup_data()
        classifier.build_model()
        history = classifier.train()
        
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()
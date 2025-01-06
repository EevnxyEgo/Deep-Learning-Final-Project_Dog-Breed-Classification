import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np
from pathlib import Path
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

class ConvBlock(layers.Layer):
    def __init__(self, filters, kernel_size=3, strides=1, padding='same'):
        super().__init__()
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

class ResidualBlock(layers.Layer):
    def __init__(self, filters):
        super().__init__()
        self.conv1 = ConvBlock(filters)
        self.conv2 = ConvBlock(filters)
        self.relu = layers.ReLU()
        
    def call(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + residual
        return self.relu(out)

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
            layers.Flatten(),
            layers.Dense(num_classes)
        ])
        
        # Build model
        self.build((None, 177, 177, 3))
        self._initialize_weights()
        
    def _initialize_weights(self):
        for layer in self.layers:
            if isinstance(layer, layers.Conv2D):
                # Kaiming/He initialization
                fan_in = layer.kernel_size[0] * layer.kernel_size[1] * layer.input_shape[-1]
                std = tf.sqrt(2.0 / fan_in)
                layer.kernel.assign(tf.random.normal(layer.kernel.shape, mean=0.0, stddev=std))
            elif isinstance(layer, layers.BatchNormalization):
                layer.gamma.assign(tf.ones_like(layer.gamma))
                layer.beta.assign(tf.zeros_like(layer.beta))
            elif isinstance(layer, layers.Dense):
                std = 0.01
                layer.kernel.assign(tf.random.normal(layer.kernel.shape, mean=0.0, stddev=std))
                layer.bias.assign(tf.zeros_like(layer.bias))
    
    def call(self, x):
        x = self.feet(x)
        x = self.body(x)
        return self.head(x)

def get_dataset_path():
    current_script_path = Path(__file__).resolve()
    project_root = current_script_path.parent.parent
    dataset_path = project_root / 'dataset' / 'Images'
    
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset folder tidak ditemukan di: {dataset_path}\n"
            f"Pastikan folder dataset berada di: {project_root}"
        )
    
    return str(dataset_path)

def save_training_plots(history, save_path):
    """
    Save training and validation metrics plots as PNG files
    """
    # Create directory if it doesn't exist
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Accuracy plot
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], color='blue', linewidth=2, label='Training')
    plt.plot(history.history['val_accuracy'], color='red', linewidth=2, label='Validation')
    plt.title('Model Accuracy', size=14, pad=15)
    plt.xlabel('Epoch', size=12)
    plt.ylabel('Accuracy', size=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path / 'accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], color='blue', linewidth=2, label='Training')
    plt.plot(history.history['val_loss'], color='red', linewidth=2, label='Validation')
    plt.title('Model Loss', size=14, pad=15)
    plt.xlabel('Epoch', size=12)
    plt.ylabel('Loss', size=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path / 'loss.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_confusion_matrix(model, val_dataset, save_path):
    """
    Generate and save confusion matrix as PNG
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    # Get predictions
    val_dataset.reset()
    y_true = val_dataset.classes
    y_pred = np.argmax(model.predict(val_dataset), axis=1)
    
    # Get class names
    class_names = list(val_dataset.class_indices.keys())
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    im = plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im)
    
    # Add numbers to the plot
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")

    plt.title('Confusion Matrix', size=14, pad=15)
    plt.ylabel('True Label', size=12)
    plt.xlabel('Predicted Label', size=12)
    
    # Add ticks
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)
    
    plt.tight_layout()
    plt.savefig(save_path / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    # Get dataset path
    dataset_path = get_dataset_path()
    
    # Configuration
    config = {
        'data_dir': dataset_path,
        'image_size': 177,
        'batch_size': 16,
        'epochs': 50,
        'num_classes': 10,
        'learning_rate': 1e-4,
        'weight_decay': 1e-3
    }

    print("\n=== Configuration ===")
    for key, value in config.items():
        print(f"{key}: {value}")
    print("===================\n")

    # Data augmentation and preprocessing
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        brightness_range=[0.8, 1.2],
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2,
        preprocessing_function=tf.keras.applications.imagenet_utils.preprocess_input
    )

    # Load datasets
    print("Loading training data...")
    train_dataset = train_datagen.flow_from_directory(
        config['data_dir'],
        target_size=(config['image_size'], config['image_size']),
        batch_size=config['batch_size'],
        class_mode='categorical',
        subset='training'
    )

    print("\nLoading validation data...")
    val_dataset = train_datagen.flow_from_directory(
        config['data_dir'],
        target_size=(config['image_size'], config['image_size']),
        batch_size=config['batch_size'],
        class_mode='categorical',
        subset='validation'
    )

    # Initialize model
    print("\nInitializing model...")
    model = CustomCNN(num_classes=config['num_classes'])
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=config['learning_rate'],
            weight_decay=config['weight_decay'],
            clipnorm=1.0
        ),
        loss=tf.keras.losses.CategoricalCrossentropy(
            from_logits=True,
            label_smoothing=0.1
        ),
        metrics=['accuracy']
    )

    # Create directories for model saving
    model_dir = Path(__file__).resolve().parent.parent / 'models'
    model_dir.mkdir(exist_ok=True)
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(model_dir / 'best_model.keras'),
            save_best_only=True,
            monitor='val_accuracy'
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=8,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
    ]

    # Print model summary
    model.summary()

    print("\nStarting training...")
    # Train
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config['epochs'],
        callbacks=callbacks
    )
    
    # Create visualizations directory
    viz_dir = model_dir / 'visualizations'
    
    # Save training plots
    print("\nSaving training plots...")
    save_training_plots(history, viz_dir)
    
    # Save confusion matrix
    print("Saving confusion matrix...")
    save_confusion_matrix(model, val_dataset, viz_dir)

    # Save final model
    model.save(str(model_dir / 'final_model.keras'))
    print(f"\nModel saved to: {model_dir}")

if __name__ == "__main__":
    # Test model architecture
    print("Testing model architecture...")
    model = CustomCNN(num_classes=10)
    x = tf.random.normal((1, 177, 177, 3))
    output = model(x)
    print(f"Output shape: {output.shape}")
    
    # Count parameters
    total_params = sum([tf.size(var).numpy() for var in model.trainable_variables])
    print(f"Total parameters: {total_params:,}")
    
    print("\nStarting main training process...")
    # Run training
    main()
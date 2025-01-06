import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import mixed_precision
import os
from pathlib import Path
from model import CustomCNN  # Import the CustomCNN class from model.py
import numpy as np
import random
from collections import Counter
import time
from data_logger import DataLogger  # Make sure DataLogger is imported

# Set seed for reproducibility
SEED = 424242
np.random.seed(SEED)
random.seed(SEED)

# Enable mixed precision if available
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

def get_dataset_path():
    current_script_path = Path(__file__).resolve()
    project_root = current_script_path.parent.parent
    dataset_path = project_root.parent / 'dataset' / 'Images'
    
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset folder tidak ditemukan di: {dataset_path}\n"
            f"Pastikan folder dataset berada di: {project_root}"
        )
    
    return str(dataset_path)

def print_dataset_info(dataset):
    """
    Mencetak informasi detail tentang dataset
    """
    classes = dataset.class_indices
    class_counts = Counter()
    for class_name, idx in dataset.class_indices.items():
        count = len([f for f in dataset.filepaths if f.endswith(f"{class_name}")])
        class_counts[class_name] = count

    print("\n=== Dataset Information ===")
    print(f"Total number of classes: {len(classes)}")
    print("\nClass names and their counts:")
    for class_name, count in class_counts.items():
        print(f"Class: {class_name:20} Count: {count}")
    
    print("\nTotal samples:", len(dataset))
    print("========================\n")

def mixup_data(x, y, alpha=0.2):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.shape[0]

    # Ensure that x and y have matching batch sizes
    assert x.shape[0] == y.shape[0], "Batch sizes of inputs and labels must match."

    index = np.random.permutation(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def train_model(config):
    # Data augmentation and preprocessing
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        brightness_range=[0.8, 1.2],
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2,  # Use this to split the data for training and validation
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

    # Print dataset information
    print_dataset_info(train_dataset)

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
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),
        metrics=['accuracy']
    )

    # Create directories for model saving
    model_dir = Path(__file__).resolve().parent.parent / 'models'
    model_dir.mkdir(exist_ok=True)

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            str(model_dir / 'best_model.keras'),
            save_best_only=True,
            monitor='val_accuracy'
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=8,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
    ]

    # Initialize DataLogger
    logger = DataLogger(base_name='experiment')

    # Print model summary
    model.summary()

    # Training with model.fit
    print("\nStarting training...")
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config['epochs'],
        callbacks=callbacks,
        steps_per_epoch=len(train_dataset),
        validation_steps=len(val_dataset)
    )

    # Save final model
    model.save(str(model_dir / 'final_model.keras'))
    print(f"\nModel saved to: {model_dir}")

    # Save the experiment details and plot training history
    logger.save_experiment_info(config)
    logger.save_plot()

def main():
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

    train_model(config)

if __name__ == "__main__":
    main()

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np
from pathlib import Path
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

def create_resnet_model(num_classes=10, fine_tune_layers=0):
    """
    Create a ResNet50V2 model with custom top layers for classification
    """
    # Load pre-trained ResNet50V2 (using V2 for better performance)
    base_model = tf.keras.applications.ResNet50V2(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3)  # Standard ImageNet size
    )
    
    # Initially freeze all layers
    base_model.trainable = False
    
    # Create custom top layers
    inputs = base_model.input
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    return model, base_model

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

def plot_training_history(history, save_dir):
    """
    Plot training and validation metrics
    """
    fig_dir = Path(save_dir) / 'figures'
    fig_dir.mkdir(exist_ok=True)
    
    # Plot accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(str(fig_dir / 'accuracy_plot.png'))
    plt.close()
    
    # Plot loss
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.savefig(str(fig_dir / 'loss_plot.png'))
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes, save_dir):
    """
    Create and save confusion matrix visualization
    """
    fig_dir = Path(save_dir) / 'figures'
    fig_dir.mkdir(exist_ok=True)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(str(fig_dir / 'confusion_matrix.png'))
    plt.close()

def main():
    # Get dataset path
    dataset_path = get_dataset_path()
    
    # Configuration
    config = {
        'data_dir': dataset_path,
        'image_size': 224,  # Standard ImageNet size
        'batch_size': 32,   # Increased batch size
        'initial_epochs': 10,  # Initial training epochs
        'fine_tune_epochs': 40,  # Fine-tuning epochs
        'num_classes': 10,
        'initial_lr': 1e-3,  # Higher initial learning rate
        'fine_tune_lr': 1e-4,  # Lower learning rate for fine-tuning
        'weight_decay': 1e-5
    }

    print("\n=== Configuration ===")
    for key, value in config.items():
        print(f"{key}: {value}")
    print("===================\n")

    # Data augmentation and preprocessing
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.resnet_v2.preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )

    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.resnet_v2.preprocess_input,
        validation_split=0.2
    )

    # Load datasets
    print("Loading training data...")
    train_dataset = train_datagen.flow_from_directory(
        config['data_dir'],
        target_size=(config['image_size'], config['image_size']),
        batch_size=config['batch_size'],
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    print("\nLoading validation data...")
    val_dataset = val_datagen.flow_from_directory(
        config['data_dir'],
        target_size=(config['image_size'], config['image_size']),
        batch_size=config['batch_size'],
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    # Initialize model
    print("\nInitializing ResNet model...")
    model, base_model = create_resnet_model(num_classes=config['num_classes'])
    
    # First phase: Training only the top layers
    print("\nPhase 1: Training top layers...")
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=config['initial_lr'],
            weight_decay=config['weight_decay']
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
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.2,
            patience=3,
            min_lr=1e-6
        )
    ]

    # Print model summary
    model.summary()

    # Initial training phase
    print("\nStarting initial training phase...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config['initial_epochs'],
        callbacks=callbacks
    )

    # Second phase: Fine-tuning
    print("\nPhase 2: Fine-tuning ResNet layers...")
    # Unfreeze all layers
    base_model.trainable = True
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=config['fine_tune_lr'],
            weight_decay=config['weight_decay']
        ),
        loss=tf.keras.losses.CategoricalCrossentropy(
            from_logits=True,
            label_smoothing=0.1
        ),
        metrics=['accuracy']
    )

    # Fine-tuning phase
    print("\nStarting fine-tuning phase...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config['fine_tune_epochs'],
        callbacks=callbacks
    )
    
    # Plot fine-tuning metrics
    plot_training_history(history, model_dir)

    # Generate predictions for confusion matrix
    print("\nGenerating confusion matrix...")
    val_dataset.reset()
    y_pred = np.argmax(model.predict(val_dataset), axis=1)
    y_true = val_dataset.classes
    class_names = list(train_dataset.class_indices.keys())
    
    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred, class_names, model_dir)

    # Save final model
    model.save(str(model_dir / 'final_model_resnetV2.keras'))
    print(f"\nModel saved to: {model_dir}")
    print(f"Training visualizations saved to: {model_dir}/figures/")

if __name__ == "__main__":
    main()
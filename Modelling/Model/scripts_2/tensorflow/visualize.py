import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from pathlib import Path

def save_training_plots(history, save_path):
    """
    Save training and validation metrics plots as PNG files
    """
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
    
    # Generate predictions
    y_true = val_dataset.classes
    y_pred = model.predict(val_dataset)
    y_pred = np.argmax(y_pred, axis=-1)
    
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalize
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 6))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', cbar=False, 
                xticklabels=val_dataset.class_indices.keys(), 
                yticklabels=val_dataset.class_indices.keys())
    plt.title('Confusion Matrix', size=14, pad=15)
    plt.xlabel('Predicted Label', size=12)
    plt.ylabel('True Label', size=12)
    plt.tight_layout()
    plt.savefig(save_path / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

# Example usage
def main():
    # Replace with actual paths
    history = ...  # History object from training
    val_dataset = ...  # Validation dataset
    model = ...  # Your trained model
    model_dir = 'models/'

    save_training_plots(history, model_dir)
    save_confusion_matrix(model, val_dataset, model_dir)

if __name__ == "__main__":
    main()

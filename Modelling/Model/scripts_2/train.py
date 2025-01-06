import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import os
import random
import numpy as np
from tqdm import tqdm
from collections import Counter
import time
from sklearn.model_selection import StratifiedShuffleSplit

from model import CustomCNN
from logger import DataLogger

# Set seed for reproducibility
SEED = 424242
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.use_deterministic_algorithms(True)

def print_dataset_info(dataset):
    """
    Mencetak informasi detail tentang dataset
    """
    classes = dataset.classes
    class_to_idx = dataset.class_to_idx
    
    print("\n=== Dataset Information ===")
    print(f"Total number of classes: {len(classes)}")
    print("\nClass names and their indices:")
    for class_name, idx in class_to_idx.items():
        print(f"Class: {class_name:20} Index: {idx}")
    
    class_counts = Counter()
    for _, label in dataset.samples:
        class_counts[dataset.classes[label]] += 1
    
    print("\nSamples per class:")
    for class_name in classes:
        count = class_counts[class_name]
        print(f"Class: {class_name:20} Count: {count:5}")
    
    print("\nTotal samples:", len(dataset))
    print("========================\n")

def get_stratified_split(dataset, valid_size=0.2):
    labels = [label for _, label in dataset.samples]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=valid_size, random_state=SEED)
    train_idx, val_idx = next(sss.split(np.zeros(len(labels)), labels))
    
    from torch.utils.data import Subset
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    
    return train_dataset, val_dataset

def mixup_data(x, y, alpha=0.2):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train_model():
    # Constants
    IMAGE_SIZE = 177
    BATCH_SIZE = 16
    EPOCHS = 50
    LEARNING_RATE = 0.0003
    WEIGHT_DECAY = 1e-3
    EARLY_STOPPING_PATIENCE = 8
    EARLY_STOPPING_DELTA = 0.01
    
    # Device configuration
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Data transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = datasets.ImageFolder(
        root='dataset/Images',
        transform=data_transforms['train']
    )
    
    print_dataset_info(train_dataset)
    
    # Save class information
    class_info = {
        'classes': train_dataset.classes,
        'class_to_idx': train_dataset.class_to_idx
    }
    
    # Stratified split
    train_dataset, val_dataset = get_stratified_split(train_dataset)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE,
        num_workers=0,
        pin_memory=False
    )
    
    print(f"\nDataset splits:")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Initialize model, criterion, optimizer
    num_classes = len(train_dataset.dataset.classes)
    print(f"\nInitializing model with {num_classes} output classes...")
    
    model = CustomCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.02)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5,
        min_lr=1e-5,
        cooldown=3
    )
    
    print("\nModel configuration:")
    print(f"Initial learning rate: {LEARNING_RATE}")
    print(f"Weight decay: {WEIGHT_DECAY}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    
    # Initialize logger
    logger = DataLogger('DogClassifier')
    
    # Early stopping setup
    best_val_acc = 0
    patience_counter = 0
    best_epoch = -1
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(EPOCHS):
        start_time = time.time()
        
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS} - Training'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Apply mixup
            inputs_mixed, labels_a, labels_b, lam = mixup_data(inputs, labels)
            
            optimizer.zero_grad()
            outputs = model(inputs_mixed)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Calculate accuracy with original inputs
            with torch.no_grad():
                outputs_orig = model(inputs)
                running_loss += loss.item()
                _, predicted = outputs_orig.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{EPOCHS} - Validation'):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss = running_loss / len(val_loader)
        val_acc = correct / total

        # Update learning rate
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log results
        logger.append(epoch, train_loss, train_acc, val_loss, val_acc)
        
        # Print epoch results
        print(f'\nEpoch {epoch+1}/{EPOCHS}:')
        print(f'Training Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}')
        print(f'Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}')
        print(f'Current Learning Rate: {current_lr:.6f}')
        print(f'Best Accuracy So Far: {best_val_acc:.4f} (Epoch {best_epoch+1})')
        print(f'Epochs without improvement: {patience_counter}')
        print(f'Epoch Time: {time.time() - start_time:.2f}s')

        # Overfitting warning
        if train_acc - val_acc > 0.1:
            print("\nWarning: Possible overfitting detected!")
            print(f"Training-Validation accuracy gap: {(train_acc - val_acc)*100:.2f}%")
        
        # Early stopping check
        if val_acc > (best_val_acc + EARLY_STOPPING_DELTA):
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            
            # Save best model
            state_dict = {
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'epoch': epoch,
                'best_acc': best_val_acc,
                'class_info': class_info,
                'model_config': {
                    'input_size': IMAGE_SIZE,
                    'num_classes': num_classes,
                    'transform_params': {
                        'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]
                    }
                }
            }
            torch.save(state_dict, logger.get_filepath('best_model.pth'))
            
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'best_acc': best_val_acc,
            }, logger.get_filepath('last_checkpoint.pth'))
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                print(f"Best accuracy was {best_val_acc:.4f} at epoch {best_epoch+1}")
                break
    
    # Save final plots
    logger.save_plot()
    
    print("\nTraining Results:")
    print(f"Best Validation Accuracy: {logger.best_accuracy:.4f}")
    print(f"Final Training Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
    print(f"Final Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
    
    try:
        import onnx
        print("\nExporting model to ONNX...")
        dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)
        torch.onnx.export(model, 
                         dummy_input,
                         logger.get_filepath('model.onnx'),
                         export_params=True,
                         opset_version=11,
                         do_constant_folding=True,
                         input_names=['input'],
                         output_names=['output'])
        print("ONNX export successful!")
    except ImportError:
        print("\nSkipping ONNX export (onnx package not installed)")
        print("To enable ONNX export, install onnx package: pip install onnx")
    
    print("\nTraining complete!")
    print(f"Model and training logs saved in: {logger.log_dir}")

if __name__ == "__main__":
    train_model()
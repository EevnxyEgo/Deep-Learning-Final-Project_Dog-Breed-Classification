import os
import matplotlib.pyplot as plt
import json
from datetime import datetime

class DataLogger:
    def __init__(self, base_name):
        # Mencari nomor eksperimen berikutnya
        self.base_dir = 'experiments'
        self.base_name = base_name
        self.experiment_num = self._get_next_experiment_number()
        
        # Membuat nama folder eksperimen
        self.experiment_name = f"{base_name}_{self.experiment_num}"
        self.log_dir = os.path.join(self.base_dir, self.experiment_name)
        
        # Membuat folder eksperimen
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.best_accuracy = 0
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
        
        # Menyimpan waktu mulai eksperimen
        self.start_time = datetime.now()
        
    def _get_next_experiment_number(self):
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
            return 1
        
        existing_dirs = [d for d in os.listdir(self.base_dir) 
                        if os.path.isdir(os.path.join(self.base_dir, d)) 
                        and d.startswith(self.base_name)]
        if not existing_dirs:
            return 1
        
        numbers = [int(d.split('_')[-1]) for d in existing_dirs 
                  if d.split('_')[-1].isdigit()]
        return max(numbers, default=0) + 1
    
    def append(self, epoch, train_loss, train_acc, val_loss, val_acc):
        self.history['train_loss'].append(train_loss)
        self.history['train_acc'].append(train_acc)
        self.history['val_loss'].append(val_loss)
        self.history['val_acc'].append(val_acc)
        
        self.current_epoch_is_best = val_acc > self.best_accuracy
        if self.current_epoch_is_best:
            self.best_accuracy = val_acc
    
    def get_filepath(self, filename):
        return os.path.join(self.log_dir, filename)
    
    def save_plot(self):
        plt.figure(figsize=(15, 5))
        
        # Plot Loss
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(self.history['train_acc'], label='Train Accuracy')
        plt.plot(self.history['val_acc'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.get_filepath('training_history.png'))
        plt.close()
        
    def save_experiment_info(self, config):
        # Menyimpan konfigurasi dan hasil eksperimen
        experiment_info = {
            'experiment_name': self.experiment_name,
            'start_time': self.start_time.strftime('%Y-%m-%d %H:%M:%S'),
            'end_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'best_validation_accuracy': float(self.best_accuracy),
            'configuration': config,
            'final_metrics': {
                'train_loss': float(self.history['train_loss'][-1]),
                'train_acc': float(self.history['train_acc'][-1]),
                'val_loss': float(self.history['val_loss'][-1]),
                'val_acc': float(self.history['val_acc'][-1])
            }
        }
        
        with open(self.get_filepath('experiment_info.json'), 'w') as f:
            json.dump(experiment_info, f, indent=4)
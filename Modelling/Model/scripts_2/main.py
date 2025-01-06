import os
import torch

# Set CUDA deterministic mode
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

if __name__ == "__main__":
    from train import train_model
    train_model()
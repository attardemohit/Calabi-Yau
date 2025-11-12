"""
Training Script for Calabi-Yau to Particle Spectra Neural Networks
Handles training loop, validation, checkpointing, and logging
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import json
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from models import create_model
from data_generator import CalabiYauDataGenerator


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-4, mode: str = 'min'):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
        elif self.mode == 'min' and score > self.best_score - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        elif self.mode == 'max' and score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        
        return self.early_stop


class Trainer:
    """Main trainer class for Calabi-Yau models."""
    
    def __init__(self,
                 model: nn.Module,
                 device: str = 'cpu',
                 task_type: str = 'regression',
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-5,
                 scheduler_patience: int = 5,
                 scheduler_factor: float = 0.5):
        """
        Initialize trainer.
        
        Args:
            model: Neural network model
            device: Device to train on ('cpu' or 'cuda')
            task_type: 'regression' or 'classification'
            learning_rate: Initial learning rate
            weight_decay: L2 regularization weight
            scheduler_patience: Patience for learning rate scheduler
            scheduler_factor: Factor to reduce learning rate
        """
        self.model = model.to(device)
        self.device = device
        self.task_type = task_type
        
        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=scheduler_patience,
            factor=scheduler_factor
        )
        
        # Loss function
        if task_type == 'regression':
            self.criterion = nn.MSELoss()
        else:
            if model.output_dim == 1:
                self.criterion = nn.BCELoss()
            else:
                self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metric': [],
            'val_metric': [],
            'learning_rates': []
        }
    
    def prepare_data(self, 
                    df: pd.DataFrame,
                    target_col: str = 'particle_spectrum',
                    batch_size: int = 32) -> DataLoader:
        """
        Prepare DataLoader from DataFrame.
        
        Args:
            df: DataFrame with features and target
            target_col: Name of target column
            batch_size: Batch size for DataLoader
            
        Returns:
            DataLoader object
        """
        # Separate features and target
        feature_cols = [col for col in df.columns if col != target_col]
        X = df[feature_cols].values.astype(np.float32)
        y = df[target_col].values.astype(np.float32)
        
        # Normalize features (standardization)
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0) + 1e-8  # Add small value to avoid division by zero
        X = (X - X_mean) / X_std
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        
        if self.task_type == 'regression':
            y_tensor = torch.FloatTensor(y.reshape(-1, 1))
        else:
            y_tensor = torch.LongTensor(y)
        
        # Create dataset and dataloader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        return dataloader
    
    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            dataloader: Training data loader
            
        Returns:
            Tuple of (average loss, average metric)
        """
        self.model.train()
        total_loss = 0
        total_metric = 0
        n_batches = 0
        
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(X_batch)
            
            # Calculate loss
            if self.task_type == 'regression':
                loss = self.criterion(outputs, y_batch)
                metric = loss.item()  # Use MSE as metric
            else:
                if self.model.output_dim == 1:
                    loss = self.criterion(outputs, y_batch.float())
                    predictions = (outputs > 0.5).float()
                    metric = (predictions == y_batch).float().mean().item()
                else:
                    loss = self.criterion(outputs, y_batch)
                    predictions = torch.argmax(outputs, dim=1)
                    metric = (predictions == y_batch).float().mean().item()
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            total_metric += metric
            n_batches += 1
        
        return total_loss / n_batches, total_metric / n_batches
    
    def validate(self, dataloader: DataLoader) -> Tuple[float, float]:
        """
        Validate model.
        
        Args:
            dataloader: Validation data loader
            
        Returns:
            Tuple of (average loss, average metric)
        """
        self.model.eval()
        total_loss = 0
        total_metric = 0
        n_batches = 0
        
        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                outputs = self.model(X_batch)
                
                if self.task_type == 'regression':
                    loss = self.criterion(outputs, y_batch)
                    metric = loss.item()
                else:
                    if self.model.output_dim == 1:
                        loss = self.criterion(outputs, y_batch.float())
                        predictions = (outputs > 0.5).float()
                        metric = (predictions == y_batch).float().mean().item()
                    else:
                        loss = self.criterion(outputs, y_batch)
                        predictions = torch.argmax(outputs, dim=1)
                        metric = (predictions == y_batch).float().mean().item()
                
                total_loss += loss.item()
                total_metric += metric
                n_batches += 1
        
        return total_loss / n_batches, total_metric / n_batches
    
    def train(self,
              train_loader: DataLoader,
              val_loader: Optional[DataLoader] = None,
              n_epochs: int = 100,
              early_stopping_patience: int = 15,
              checkpoint_dir: str = 'models',
              verbose: bool = True) -> Dict:
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            n_epochs: Number of epochs to train
            early_stopping_patience: Patience for early stopping
            checkpoint_dir: Directory to save model checkpoints
            verbose: Whether to print progress
            
        Returns:
            Training history dictionary
        """
        # Setup
        Path(checkpoint_dir).mkdir(exist_ok=True)
        early_stopping = EarlyStopping(patience=early_stopping_patience)
        best_val_loss = float('inf')
        
        # Training loop
        for epoch in range(n_epochs):
            start_time = time.time()
            
            # Train
            train_loss, train_metric = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            self.history['train_metric'].append(train_metric)
            
            # Validate
            if val_loader is not None:
                val_loss, val_metric = self.validate(val_loader)
                self.history['val_loss'].append(val_loss)
                self.history['val_metric'].append(val_metric)
                
                # Learning rate scheduling
                self.scheduler.step(val_loss)
                
                # Early stopping
                if early_stopping(val_loss):
                    if verbose:
                        print(f"Early stopping triggered at epoch {epoch + 1}")
                    break
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint(
                        Path(checkpoint_dir) / 'best_model.pth',
                        epoch,
                        val_loss
                    )
            
            # Record learning rate
            self.history['learning_rates'].append(
                self.optimizer.param_groups[0]['lr']
            )
            
            # Print progress
            if verbose and (epoch + 1) % 10 == 0:
                epoch_time = time.time() - start_time
                if val_loader is not None:
                    metric_name = 'Accuracy' if self.task_type == 'classification' else 'MSE'
                    print(f"Epoch [{epoch + 1}/{n_epochs}] "
                          f"Train Loss: {train_loss:.4f}, "
                          f"Val Loss: {val_loss:.4f}, "
                          f"Val {metric_name}: {val_metric:.4f}, "
                          f"Time: {epoch_time:.2f}s")
                else:
                    print(f"Epoch [{epoch + 1}/{n_epochs}] "
                          f"Train Loss: {train_loss:.4f}, "
                          f"Time: {epoch_time:.2f}s")
        
        # Save final model
        self.save_checkpoint(
            Path(checkpoint_dir) / 'final_model.pth',
            epoch,
            self.history['val_loss'][-1] if val_loader else self.history['train_loss'][-1]
        )
        
        return self.history
    
    def save_checkpoint(self, filepath: Path, epoch: int, loss: float):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'history': self.history
        }
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)
        return checkpoint['epoch'], checkpoint['loss']
    
    def plot_history(self, save_path: Optional[str] = None):
        """Plot training history."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Loss plot
        axes[0].plot(self.history['train_loss'], label='Train Loss')
        if self.history['val_loss']:
            axes[0].plot(self.history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Metric plot
        metric_name = 'Accuracy' if self.task_type == 'classification' else 'MSE'
        axes[1].plot(self.history['train_metric'], label=f'Train {metric_name}')
        if self.history['val_metric']:
            axes[1].plot(self.history['val_metric'], label=f'Val {metric_name}')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel(metric_name)
        axes[1].set_title(f'Training and Validation {metric_name}')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Learning rate plot
        axes[2].plot(self.history['learning_rates'])
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Learning Rate')
        axes[2].set_title('Learning Rate Schedule')
        axes[2].set_yscale('log')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.show()


def main():
    """Main training function."""
    # Configuration
    config = {
        'data_dir': 'data',
        'model_type': 'mlp',
        'task_type': 'regression',  # or 'classification'
        'batch_size': 32,
        'n_epochs': 200,
        'learning_rate': 1e-3,
        'early_stopping_patience': 20,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print(f"Training configuration:")
    print(json.dumps(config, indent=2))
    print(f"\nUsing device: {config['device']}")
    
    # Load data
    print("\nLoading data...")
    data_path = Path(config['data_dir'])
    
    # Check if data exists, if not generate it
    if not (data_path / f"calabi_yau_train_{config['task_type']}.csv").exists():
        print("Data not found. Generating synthetic data...")
        from data_generator import generate_and_save_datasets
        generate_and_save_datasets(output_dir=config['data_dir'])
    
    # Load datasets
    train_df = pd.read_csv(data_path / f"calabi_yau_train_{config['task_type']}.csv")
    val_df = pd.read_csv(data_path / f"calabi_yau_val_{config['task_type']}.csv")
    test_df = pd.read_csv(data_path / f"calabi_yau_test_{config['task_type']}.csv")
    
    print(f"Train samples: {len(train_df)}")
    print(f"Val samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # Determine input/output dimensions
    feature_cols = [col for col in train_df.columns if col != 'particle_spectrum']
    input_dim = len(feature_cols)
    
    if config['task_type'] == 'classification':
        output_dim = train_df['particle_spectrum'].nunique()
    else:
        output_dim = 1
    
    print(f"\nModel architecture:")
    print(f"Input dimension: {input_dim}")
    print(f"Output dimension: {output_dim}")
    
    # Create model
    model = create_model(
        model_type=config['model_type'],
        input_dim=input_dim,
        output_dim=output_dim,
        task_type=config['task_type'],
        hidden_dims=[128, 64, 32],
        dropout_rate=0.2
    )
    
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        device=config['device'],
        task_type=config['task_type'],
        learning_rate=config['learning_rate']
    )
    
    # Prepare data loaders
    train_loader = trainer.prepare_data(train_df, batch_size=config['batch_size'])
    val_loader = trainer.prepare_data(val_df, batch_size=config['batch_size'])
    test_loader = trainer.prepare_data(test_df, batch_size=config['batch_size'])
    
    # Train model
    print("\nStarting training...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=config['n_epochs'],
        early_stopping_patience=config['early_stopping_patience'],
        checkpoint_dir='models'
    )
    
    # Plot training history
    trainer.plot_history(save_path='results/training_history.png')
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_metric = trainer.validate(test_loader)
    
    if config['task_type'] == 'regression':
        print(f"Test MSE: {test_loss:.4f}")
        print(f"Test RMSE: {np.sqrt(test_loss):.4f}")
    else:
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_metric:.4f}")
    
    # Save results
    results = {
        'config': config,
        'test_loss': test_loss,
        'test_metric': test_metric,
        'best_val_loss': min(history['val_loss']) if history['val_loss'] else None,
        'n_epochs_trained': len(history['train_loss'])
    }
    
    Path('results').mkdir(exist_ok=True)
    with open('results/training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nTraining complete! Results saved to 'results/' directory.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Multi-task learning for Calabi-Yau manifolds.

This module implements multi-task neural networks that jointly predict
multiple properties (h11, h21, Euler characteristic) while enforcing
physical constraints like mirror symmetry.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional


class MultiTaskCYNet(nn.Module):
    """Multi-task neural network for CY manifold properties."""
    
    def __init__(self, 
                 input_dim: int,
                 shared_dims: List[int] = [256, 128],
                 task_dims: Dict[str, List[int]] = None,
                 dropout: float = 0.2):
        """
        Initialize multi-task network.
        
        Args:
            input_dim: Input feature dimension
            shared_dims: Dimensions of shared layers
            task_dims: Task-specific layer dimensions
            dropout: Dropout rate
        """
        super().__init__()
        
        if task_dims is None:
            task_dims = {
                'h11': [64, 32],
                'h21': [64, 32],
                'euler': [32],
                'mirror': [32]
            }
        
        # Shared layers (common feature extraction)
        shared_layers = []
        prev_dim = input_dim
        
        for dim in shared_dims:
            shared_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = dim
        
        self.shared = nn.Sequential(*shared_layers)
        self.shared_dim = prev_dim
        
        # Task-specific heads
        self.task_heads = nn.ModuleDict()
        
        # H11 prediction head
        h11_layers = []
        prev_dim = self.shared_dim
        for dim in task_dims['h11']:
            h11_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = dim
        h11_layers.append(nn.Linear(prev_dim, 1))
        self.task_heads['h11'] = nn.Sequential(*h11_layers)
        
        # H21 prediction head
        h21_layers = []
        prev_dim = self.shared_dim
        for dim in task_dims['h21']:
            h21_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = dim
        h21_layers.append(nn.Linear(prev_dim, 1))
        self.task_heads['h21'] = nn.Sequential(*h21_layers)
        
        # Euler characteristic head
        euler_layers = []
        prev_dim = self.shared_dim
        for dim in task_dims['euler']:
            euler_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = dim
        euler_layers.append(nn.Linear(prev_dim, 1))
        self.task_heads['euler'] = nn.Sequential(*euler_layers)
        
        # Mirror symmetry classification head
        mirror_layers = []
        prev_dim = self.shared_dim
        for dim in task_dims['mirror']:
            mirror_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = dim
        mirror_layers.append(nn.Linear(prev_dim, 1))
        mirror_layers.append(nn.Sigmoid())
        self.task_heads['mirror'] = nn.Sequential(*mirror_layers)
    
    def forward(self, x):
        """
        Forward pass through multi-task network.
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary of task predictions
        """
        # Shared feature extraction
        shared_features = self.shared(x)
        
        # Task-specific predictions
        outputs = {}
        outputs['h11'] = self.task_heads['h11'](shared_features)
        outputs['h21'] = self.task_heads['h21'](shared_features)
        outputs['euler'] = self.task_heads['euler'](shared_features)
        outputs['mirror'] = self.task_heads['mirror'](shared_features)
        
        return outputs


class MirrorSymmetryConstraint(nn.Module):
    """Enforce mirror symmetry constraints in loss function."""
    
    def __init__(self, alpha: float = 0.1):
        """
        Initialize mirror symmetry constraint.
        
        Args:
            alpha: Weight for symmetry constraint
        """
        super().__init__()
        self.alpha = alpha
    
    def forward(self, h11_pred, h21_pred, h11_true, h21_true):
        """
        Calculate mirror symmetry constraint loss.
        
        Mirror symmetry: (h11, h21) <-> (h21, h11)
        
        Args:
            h11_pred: Predicted h11
            h21_pred: Predicted h21
            h11_true: True h11
            h21_true: True h21
            
        Returns:
            Symmetry constraint loss
        """
        # Check if manifolds are potential mirror pairs
        # Mirror pairs should have swapped Hodge numbers
        is_mirror = torch.abs(h11_true - h21_true) < 0.5  # Binary indicator
        
        # For mirror pairs, predictions should be symmetric
        symmetry_loss = is_mirror * torch.abs(h11_pred - h21_pred)
        
        return self.alpha * symmetry_loss.mean()


class EulerConstraint(nn.Module):
    """Enforce Euler characteristic constraint."""
    
    def __init__(self, beta: float = 0.1):
        """
        Initialize Euler constraint.
        
        Args:
            beta: Weight for Euler constraint
        """
        super().__init__()
        self.beta = beta
    
    def forward(self, h11_pred, h21_pred, euler_pred, euler_true):
        """
        Calculate Euler characteristic constraint loss.
        
        Euler characteristic: χ = 2(h11 - h21)
        
        Args:
            h11_pred: Predicted h11
            h21_pred: Predicted h21
            euler_pred: Predicted Euler characteristic
            euler_true: True Euler characteristic
            
        Returns:
            Euler constraint loss
        """
        # Calculate expected Euler from Hodge numbers
        euler_expected = 2 * (h11_pred - h21_pred)
        
        # Constraint: predicted Euler should match expected
        euler_loss = torch.abs(euler_pred - euler_expected)
        
        return self.beta * euler_loss.mean()


class MultiTaskLoss(nn.Module):
    """Multi-task loss with physical constraints."""
    
    def __init__(self, 
                 task_weights: Dict[str, float] = None,
                 use_mirror_constraint: bool = True,
                 use_euler_constraint: bool = True):
        """
        Initialize multi-task loss.
        
        Args:
            task_weights: Weights for each task
            use_mirror_constraint: Whether to use mirror symmetry constraint
            use_euler_constraint: Whether to use Euler constraint
        """
        super().__init__()
        
        if task_weights is None:
            task_weights = {
                'h11': 1.0,
                'h21': 1.0,
                'euler': 0.5,
                'mirror': 0.3
            }
        
        self.task_weights = task_weights
        self.use_mirror_constraint = use_mirror_constraint
        self.use_euler_constraint = use_euler_constraint
        
        # Task-specific losses
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        
        # Physical constraints
        if use_mirror_constraint:
            self.mirror_constraint = MirrorSymmetryConstraint(alpha=0.1)
        if use_euler_constraint:
            self.euler_constraint = EulerConstraint(beta=0.1)
    
    def forward(self, predictions: Dict, targets: Dict):
        """
        Calculate multi-task loss with constraints.
        
        Args:
            predictions: Dictionary of predictions
            targets: Dictionary of targets
            
        Returns:
            Total loss and individual task losses
        """
        losses = {}
        
        # Task-specific losses
        losses['h11'] = self.mse_loss(predictions['h11'], targets['h11'])
        losses['h21'] = self.mse_loss(predictions['h21'], targets['h21'])
        losses['euler'] = self.mse_loss(predictions['euler'], targets['euler'])
        losses['mirror'] = self.bce_loss(predictions['mirror'], targets['mirror'])
        
        # Weighted sum of task losses
        total_loss = sum(self.task_weights[task] * loss 
                        for task, loss in losses.items())
        
        # Add physical constraints
        if self.use_mirror_constraint:
            mirror_loss = self.mirror_constraint(
                predictions['h11'], predictions['h21'],
                targets['h11'], targets['h21']
            )
            losses['mirror_constraint'] = mirror_loss
            total_loss += mirror_loss
        
        if self.use_euler_constraint:
            euler_loss = self.euler_constraint(
                predictions['h11'], predictions['h21'],
                predictions['euler'], targets['euler']
            )
            losses['euler_constraint'] = euler_loss
            total_loss += euler_loss
        
        losses['total'] = total_loss
        
        return total_loss, losses


class MultiTaskTrainer:
    """Trainer for multi-task CY learning."""
    
    def __init__(self, 
                 model: MultiTaskCYNet,
                 device: str = 'cpu'):
        """
        Initialize multi-task trainer.
        
        Args:
            model: Multi-task model
            device: Device to train on
        """
        self.model = model.to(device)
        self.device = device
        self.loss_fn = MultiTaskLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=10, factor=0.5
        )
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        task_losses = {
            'h11': 0, 'h21': 0, 'euler': 0, 'mirror': 0,
            'mirror_constraint': 0, 'euler_constraint': 0
        }
        
        for batch in train_loader:
            features = batch['features'].to(self.device)
            targets = {
                'h11': batch['h11'].to(self.device),
                'h21': batch['h21'].to(self.device),
                'euler': batch['euler'].to(self.device),
                'mirror': batch['mirror'].to(self.device)
            }
            
            # Forward pass
            predictions = self.model(features)
            
            # Calculate loss
            loss, losses = self.loss_fn(predictions, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Track losses
            total_loss += loss.item()
            for task, task_loss in losses.items():
                if task != 'total':
                    task_losses[task] += task_loss.item()
        
        # Average losses
        n_batches = len(train_loader)
        total_loss /= n_batches
        task_losses = {k: v/n_batches for k, v in task_losses.items()}
        
        return total_loss, task_losses
    
    def evaluate(self, val_loader):
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0
        task_losses = {
            'h11': 0, 'h21': 0, 'euler': 0, 'mirror': 0,
            'mirror_constraint': 0, 'euler_constraint': 0
        }
        
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(self.device)
                targets = {
                    'h11': batch['h11'].to(self.device),
                    'h21': batch['h21'].to(self.device),
                    'euler': batch['euler'].to(self.device),
                    'mirror': batch['mirror'].to(self.device)
                }
                
                # Forward pass
                predictions = self.model(features)
                
                # Calculate loss
                loss, losses = self.loss_fn(predictions, targets)
                
                # Track losses
                total_loss += loss.item()
                for task, task_loss in losses.items():
                    if task != 'total':
                        task_losses[task] += task_loss.item()
        
        # Average losses
        n_batches = len(val_loader)
        total_loss /= n_batches
        task_losses = {k: v/n_batches for k, v in task_losses.items()}
        
        return total_loss, task_losses
    
    def train(self, train_loader, val_loader, n_epochs: int = 100):
        """
        Train multi-task model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            n_epochs: Number of epochs
            
        Returns:
            Training history
        """
        history = {
            'train_loss': [],
            'val_loss': [],
            'task_losses': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(n_epochs):
            # Training
            train_loss, train_task_losses = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_task_losses = self.evaluate(val_loader)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Track history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['task_losses'].append({
                'train': train_task_losses,
                'val': val_task_losses
            })
            
            # Print progress
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{n_epochs}")
                print(f"  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                print(f"  Task Losses (Val):")
                for task, loss in val_task_losses.items():
                    print(f"    {task}: {loss:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'models/best_multitask_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= 20:
                    print(f"Early stopping at epoch {epoch}")
                    break
        
        return history


def create_multitask_dataloader(cicy_data, batch_size: int = 32):
    """
    Create data loader for multi-task learning.
    
    Args:
        cicy_data: CICY manifold data
        batch_size: Batch size
        
    Returns:
        DataLoader for multi-task learning
    """
    from torch.utils.data import Dataset, DataLoader
    
    class MultiTaskDataset(Dataset):
        def __init__(self, data):
            self.data = data
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            manifold = self.data[idx]
            
            # Extract features (excluding targets)
            features = []
            for key in ['num_ps', 'num_pol', 'eta', 'matrix_rank', 
                       'matrix_norm', 'c2_mean', 'c2_std']:
                if key in manifold:
                    features.append(manifold[key])
            
            # Targets
            h11 = manifold.get('h11', 0)
            h21 = manifold.get('h21', 0)
            euler = manifold.get('euler_char', 2*(h11 - h21))
            is_mirror = 1.0 if h11 == h21 else 0.0
            
            return {
                'features': torch.FloatTensor(features),
                'h11': torch.FloatTensor([h11]),
                'h21': torch.FloatTensor([h21]),
                'euler': torch.FloatTensor([euler]),
                'mirror': torch.FloatTensor([is_mirror])
            }
    
    dataset = MultiTaskDataset(cicy_data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


if __name__ == "__main__":
    print("Multi-task Learning for Calabi-Yau Manifolds")
    print("="*60)
    
    # Test multi-task network
    model = MultiTaskCYNet(input_dim=10)
    print(f"Multi-task model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    x = torch.randn(32, 10)
    outputs = model(x)
    
    print("\nOutput shapes:")
    for task, output in outputs.items():
        print(f"  {task}: {output.shape}")
    
    # Test constraints
    print("\nPhysical constraints:")
    print("  - Mirror symmetry: (h11, h21) <-> (h21, h11)")
    print("  - Euler characteristic: χ = 2(h11 - h21)")
    
    print("\nMulti-task learning ready!")

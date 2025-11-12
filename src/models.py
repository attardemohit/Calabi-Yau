"""
Neural Network Models for Calabi-Yau to Particle Spectra Prediction
Implements feed-forward neural networks for both regression and classification tasks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import numpy as np


class CalabiYauMLP(nn.Module):
    """
    Multi-Layer Perceptron for mapping Calabi-Yau geometry to particle spectra.
    
    Flexible architecture supporting both regression and classification tasks.
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int] = [128, 64, 32],
                 output_dim: int = 1,
                 task_type: str = 'regression',
                 dropout_rate: float = 0.2,
                 activation: str = 'relu',
                 use_batch_norm: bool = True):
        """
        Initialize the MLP model.
        
        Args:
            input_dim: Number of input features (geometry properties)
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension (1 for regression, n_classes for classification)
            task_type: 'regression' or 'classification'
            dropout_rate: Dropout probability
            activation: Activation function ('relu', 'leaky_relu', 'elu', 'tanh')
            use_batch_norm: Whether to use batch normalization
        """
        super(CalabiYauMLP, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.task_type = task_type
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        
        # Select activation function
        self.activation_fn = self._get_activation(activation)
        
        # Build the network layers
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None
        self.dropouts = nn.ModuleList()
        
        # Input layer
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            self.dropouts.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer
        self.output_layer = nn.Linear(prev_dim, output_dim)
        
        # Initialize weights
        self._initialize_weights()
    
    def _get_activation(self, activation: str):
        """Get activation function by name."""
        activations = {
            'relu': F.relu,
            'leaky_relu': F.leaky_relu,
            'elu': F.elu,
            'tanh': torch.tanh,
            'gelu': F.gelu
        }
        if activation not in activations:
            raise ValueError(f"Unknown activation: {activation}")
        return activations[activation]
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier/He initialization."""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                # He initialization for ReLU-like activations
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                nn.init.constant_(layer.bias, 0.01)
        
        # Output layer initialization
        nn.init.xavier_normal_(self.output_layer.weight)
        nn.init.constant_(self.output_layer.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        # Pass through hidden layers
        for i, layer in enumerate(self.layers):
            x = layer(x)
            
            if self.use_batch_norm and self.training:
                x = self.batch_norms[i](x)
            
            x = self.activation_fn(x)
            
            if self.training:
                x = self.dropouts[i](x)
        
        # Output layer
        x = self.output_layer(x)
        
        # Apply appropriate output activation
        if self.task_type == 'classification' and self.output_dim > 1:
            # Multi-class classification: softmax will be applied in loss function
            pass
        elif self.task_type == 'classification' and self.output_dim == 1:
            # Binary classification
            x = torch.sigmoid(x)
        # For regression, no activation (linear output)
        
        return x
    
    def predict(self, x: torch.Tensor) -> np.ndarray:
        """
        Make predictions (inference mode).
        
        Args:
            x: Input tensor
            
        Returns:
            Numpy array of predictions
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(x)
            
            if self.task_type == 'classification':
                if self.output_dim > 1:
                    # Multi-class: return class indices
                    predictions = torch.argmax(output, dim=1)
                else:
                    # Binary: threshold at 0.5
                    predictions = (output > 0.5).float()
            else:
                # Regression: return raw values
                predictions = output
            
            return predictions.cpu().numpy()
    
    def get_feature_importance(self, x: torch.Tensor) -> np.ndarray:
        """
        Calculate feature importance using gradient-based method.
        
        Args:
            x: Input tensor
            
        Returns:
            Feature importance scores
        """
        x.requires_grad = True
        output = self.forward(x)
        
        # Calculate gradients
        if self.task_type == 'regression':
            output.sum().backward()
        else:
            # For classification, use the predicted class
            pred_class = torch.argmax(output, dim=1) if self.output_dim > 1 else (output > 0.5).long()
            output.gather(1, pred_class.unsqueeze(1)).sum().backward()
        
        # Get absolute gradients as importance
        importance = torch.abs(x.grad).mean(dim=0).cpu().numpy()
        
        return importance / importance.sum()  # Normalize


class ResidualBlock(nn.Module):
    """Residual block for deeper networks."""
    
    def __init__(self, dim: int, dropout_rate: float = 0.2):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.layer_norm(x + residual)
        return x


class DeepCalabiYauNet(nn.Module):
    """
    Deeper network with residual connections for complex geometry-physics mappings.
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 256,
                 n_residual_blocks: int = 3,
                 output_dim: int = 1,
                 task_type: str = 'regression',
                 dropout_rate: float = 0.2):
        """
        Initialize the deep residual network.
        
        Args:
            input_dim: Number of input features
            hidden_dim: Dimension of hidden layers
            n_residual_blocks: Number of residual blocks
            output_dim: Output dimension
            task_type: 'regression' or 'classification'
            dropout_rate: Dropout probability
        """
        super(DeepCalabiYauNet, self).__init__()
        
        self.task_type = task_type
        self.output_dim = output_dim
        
        # Input projection
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout_rate)
            for _ in range(n_residual_blocks)
        ])
        
        # Output layers
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        # Input projection
        x = self.input_projection(x)
        
        # Pass through residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Output projection
        x = self.output_projection(x)
        
        # Apply appropriate output activation
        if self.task_type == 'classification':
            if self.output_dim == 1:
                x = torch.sigmoid(x)
            # For multi-class, softmax is applied in loss
        
        return x
    
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get intermediate embeddings for visualization.
        
        Args:
            x: Input tensor
            
        Returns:
            Embedding tensor before final output layer
        """
        self.eval()
        with torch.no_grad():
            x = self.input_projection(x)
            for block in self.residual_blocks:
                x = block(x)
            return x


class EnsembleModel(nn.Module):
    """
    Ensemble of multiple models for improved predictions.
    """
    
    def __init__(self, models: List[nn.Module], aggregation: str = 'mean'):
        """
        Initialize ensemble model.
        
        Args:
            models: List of individual models
            aggregation: How to combine predictions ('mean', 'weighted', 'vote')
        """
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)
        self.aggregation = aggregation
        self.weights = nn.Parameter(torch.ones(len(models)) / len(models))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ensemble."""
        predictions = []
        for model in self.models:
            predictions.append(model(x))
        
        predictions = torch.stack(predictions)
        
        if self.aggregation == 'mean':
            return predictions.mean(dim=0)
        elif self.aggregation == 'weighted':
            weights = F.softmax(self.weights, dim=0)
            return (predictions * weights.view(-1, 1, 1)).sum(dim=0)
        elif self.aggregation == 'vote':
            # For classification
            votes = torch.mode(predictions.argmax(dim=-1), dim=0)[0]
            return F.one_hot(votes, predictions.shape[-1]).float()
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation}")


def create_model(model_type: str = 'mlp',
                 input_dim: int = 10,
                 output_dim: int = 1,
                 task_type: str = 'regression',
                 **kwargs) -> nn.Module:
    """
    Factory function to create models.
    
    Args:
        model_type: Type of model ('mlp', 'deep', 'ensemble')
        input_dim: Number of input features
        output_dim: Output dimension
        task_type: 'regression' or 'classification'
        **kwargs: Additional model-specific arguments
        
    Returns:
        Initialized model
    """
    if model_type == 'mlp':
        return CalabiYauMLP(
            input_dim=input_dim,
            output_dim=output_dim,
            task_type=task_type,
            **kwargs
        )
    elif model_type == 'deep':
        return DeepCalabiYauNet(
            input_dim=input_dim,
            output_dim=output_dim,
            task_type=task_type,
            **kwargs
        )
    elif model_type == 'ensemble':
        # Create multiple models for ensemble
        n_models = kwargs.pop('n_models', 3)
        models = []
        for i in range(n_models):
            model = CalabiYauMLP(
                input_dim=input_dim,
                output_dim=output_dim,
                task_type=task_type,
                hidden_dims=[128, 64, 32],
                dropout_rate=0.1 + i * 0.1  # Vary dropout
            )
            models.append(model)
        return EnsembleModel(models, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

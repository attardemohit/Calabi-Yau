#!/usr/bin/env python
"""
Training script for real CICY data with uncertainty quantification.

Features:
- Uses real Calabi-Yau manifold data from CICY dataset
- Monte Carlo dropout for uncertainty estimation
- Ensemble models for robust predictions
- SHAP values for interpretability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import shap

# Import our modules
from models import create_model
from train import Trainer
from evaluate import ModelEvaluator
from cicy_loader import CICYDataLoader


class MCDropoutModel(nn.Module):
    """Model with Monte Carlo Dropout for uncertainty quantification."""
    
    def __init__(self, 
                 input_dim: int,
                 output_dim: int,
                 hidden_dims: List[int] = [128, 64, 32],
                 dropout_rate: float = 0.2,
                 task_type: str = 'regression'):
        """
        Initialize MC Dropout model.
        
        Args:
            input_dim: Input feature dimension
            output_dim: Output dimension
            hidden_dims: Hidden layer dimensions
            dropout_rate: Dropout probability
            task_type: 'regression' or 'classification'
        """
        super().__init__()
        self.task_type = task_type
        self.dropout_rate = dropout_rate
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))  # MC Dropout
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        if task_type == 'classification' and output_dim > 1:
            layers.append(nn.Softmax(dim=1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)
    
    def predict_with_uncertainty(self, 
                                x: torch.Tensor, 
                                n_samples: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with uncertainty estimation using MC Dropout.
        
        Args:
            x: Input tensor
            n_samples: Number of forward passes for MC sampling
            
        Returns:
            Tuple of (mean predictions, uncertainty)
        """
        self.eval()  # Set to eval mode to avoid BatchNorm issues with single samples
        
        predictions = []
        for _ in range(n_samples):
            with torch.no_grad():
                # Temporarily enable dropout for MC sampling
                for module in self.modules():
                    if isinstance(module, nn.Dropout):
                        module.train()
                pred = self.forward(x)
                predictions.append(pred.cpu().numpy())
        
        predictions = np.array(predictions)
        
        # Calculate mean and uncertainty
        mean_pred = predictions.mean(axis=0)
        
        if self.task_type == 'regression':
            # For regression: use standard deviation as uncertainty
            uncertainty = predictions.std(axis=0)
        else:
            # For classification: use entropy of averaged probabilities
            mean_probs = predictions.mean(axis=0)
            entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-8), axis=-1)
            uncertainty = entropy
        
        return torch.tensor(mean_pred), torch.tensor(uncertainty)


class EnsembleModel:
    """Ensemble of models for robust predictions."""
    
    def __init__(self, 
                 n_models: int = 5,
                 input_dim: int = 10,
                 output_dim: int = 1,
                 task_type: str = 'regression'):
        """
        Initialize ensemble model.
        
        Args:
            n_models: Number of models in ensemble
            input_dim: Input feature dimension
            output_dim: Output dimension
            task_type: 'regression' or 'classification'
        """
        self.n_models = n_models
        self.task_type = task_type
        self.models = []
        
        # Create diverse models with different architectures
        architectures = [
            [128, 64, 32],
            [256, 128, 64],
            [64, 64, 64],
            [128, 128],
            [256, 128, 64, 32]
        ]
        
        for i in range(n_models):
            arch = architectures[i % len(architectures)]
            model = MCDropoutModel(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dims=arch,
                dropout_rate=0.2 + (i * 0.05),  # Vary dropout rate
                task_type=task_type
            )
            self.models.append(model)
    
    def train_ensemble(self, 
                      train_loader, 
                      val_loader,
                      n_epochs: int = 100,
                      device: str = 'cpu'):
        """
        Train all models in the ensemble.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            n_epochs: Number of training epochs
            device: Device to train on
        """
        histories = []
        
        for i, model in enumerate(self.models):
            print(f"\nTraining model {i+1}/{self.n_models}...")
            
            trainer = Trainer(
                model=model,
                device=device,
                task_type=self.task_type
            )
            
            history = trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                n_epochs=n_epochs,
                early_stopping_patience=15
            )
            
            histories.append(history)
        
        return histories
    
    def predict_with_uncertainty(self, 
                                x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make ensemble predictions with uncertainty.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (mean predictions, uncertainty)
        """
        predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(x)
                predictions.append(pred.cpu().numpy())
        
        predictions = np.array(predictions)
        
        # Calculate mean and uncertainty across ensemble
        mean_pred = predictions.mean(axis=0)
        
        if self.task_type == 'regression':
            # Total uncertainty = ensemble variance + mean MC dropout uncertainty
            ensemble_var = predictions.var(axis=0)
            
            # Get MC dropout uncertainty from each model
            mc_uncertainties = []
            for model in self.models:
                _, mc_unc = model.predict_with_uncertainty(x, n_samples=20)
                mc_uncertainties.append(mc_unc.numpy())
            
            mc_uncertainty = np.mean(mc_uncertainties, axis=0)
            total_uncertainty = np.sqrt(ensemble_var + mc_uncertainty**2)
        else:
            # For classification: use disagreement between models
            total_uncertainty = predictions.std(axis=0).mean(axis=-1)
        
        return torch.tensor(mean_pred), torch.tensor(total_uncertainty)


def train_on_real_cicy_data(
    target_type: str = 'h11_prediction',
    use_ensemble: bool = True,
    use_mc_dropout: bool = True,
    n_epochs: int = 100,
    device: str = 'cpu'
):
    """
    Train model on real CICY data with uncertainty quantification.
    
    Args:
        target_type: Type of prediction task
        use_ensemble: Whether to use ensemble model
        use_mc_dropout: Whether to use MC dropout
        n_epochs: Number of training epochs
        device: Device to train on
    """
    print("="*60)
    print("TRAINING ON REAL CICY DATA")
    print("="*60)
    
    # Load or generate CICY datasets
    data_path = Path('data')
    train_file = data_path / f'cicy_train_{target_type}.csv'
    
    if not train_file.exists():
        print("Generating CICY datasets...")
        loader = CICYDataLoader('dataset/cicylist.txt')
        loader.save_datasets(target_type=target_type)
    
    # Load datasets
    train_df = pd.read_csv(data_path / f'cicy_train_{target_type}.csv')
    val_df = pd.read_csv(data_path / f'cicy_val_{target_type}.csv')
    test_df = pd.read_csv(data_path / f'cicy_test_{target_type}.csv')
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_df)}")
    print(f"  Val: {len(val_df)}")
    print(f"  Test: {len(test_df)}")
    
    # Prepare features and targets
    feature_cols = [col for col in train_df.columns if col not in ['target', 'id']]
    
    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df['target'].values.astype(np.float32)
    
    X_val = val_df[feature_cols].values.astype(np.float32)
    y_val = val_df['target'].values.astype(np.float32)
    
    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df['target'].values.astype(np.float32)
    
    # Normalize features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Determine task type and output dimension
    if target_type in ['h11_prediction', 'h21_prediction']:
        task_type = 'regression'
        output_dim = 1
        y_train = y_train.reshape(-1, 1)
        y_val = y_val.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)
    else:
        task_type = 'classification'
        n_classes = len(np.unique(y_train))
        output_dim = n_classes if n_classes > 2 else 1
    
    print(f"\nTask type: {task_type}")
    print(f"Input dimension: {X_train.shape[1]}")
    print(f"Output dimension: {output_dim}")
    
    # Create data loaders
    from torch.utils.data import TensorDataset, DataLoader
    
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train) if task_type == 'regression' else torch.LongTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val) if task_type == 'regression' else torch.LongTensor(y_val)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Train model(s)
    if use_ensemble:
        print("\nTraining ensemble model...")
        ensemble = EnsembleModel(
            n_models=5,
            input_dim=X_train.shape[1],
            output_dim=output_dim,
            task_type=task_type
        )
        histories = ensemble.train_ensemble(
            train_loader, val_loader, n_epochs=n_epochs, device=device
        )
        model = ensemble
    else:
        print("\nTraining single model with MC Dropout...")
        model = MCDropoutModel(
            input_dim=X_train.shape[1],
            output_dim=output_dim,
            hidden_dims=[128, 64, 32],
            dropout_rate=0.2,
            task_type=task_type
        )
        
        trainer = Trainer(model=model, device=device, task_type=task_type)
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            n_epochs=n_epochs
        )
    
    # Evaluate with uncertainty
    print("\n" + "="*60)
    print("EVALUATION WITH UNCERTAINTY QUANTIFICATION")
    print("="*60)
    
    X_test_tensor = torch.FloatTensor(X_test)
    
    if isinstance(model, EnsembleModel):
        predictions, uncertainties = model.predict_with_uncertainty(X_test_tensor)
    else:
        predictions, uncertainties = model.predict_with_uncertainty(X_test_tensor, n_samples=100)
    
    predictions = predictions.numpy()
    uncertainties = uncertainties.numpy()
    
    # Calculate metrics
    if task_type == 'regression':
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        print(f"\nRegression Metrics:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R²: {r2:.4f}")
        print(f"  Mean uncertainty: {uncertainties.mean():.4f}")
        print(f"  Uncertainty range: [{uncertainties.min():.4f}, {uncertainties.max():.4f}]")
        
        # Plot predictions with uncertainty
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        # Flatten arrays for plotting
        y_test_flat = y_test.flatten()[:100]
        pred_flat = predictions.flatten()[:100]
        unc_flat = uncertainties.flatten()[:100]
        plt.errorbar(y_test_flat, pred_flat, yerr=unc_flat, 
                    fmt='o', alpha=0.5, capsize=3)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.title('Predictions with Uncertainty')
        
        plt.subplot(1, 2, 2)
        plt.scatter(predictions, uncertainties, alpha=0.5)
        plt.xlabel('Predicted Value')
        plt.ylabel('Uncertainty')
        plt.title('Uncertainty vs Prediction')
        
        plt.tight_layout()
        plt.savefig('results/cicy_predictions_with_uncertainty.png')
        plt.show()
        
    else:
        from sklearn.metrics import accuracy_score, f1_score, classification_report
        
        if output_dim == 1:
            pred_classes = (predictions > 0.5).astype(int).flatten()
        else:
            pred_classes = predictions.argmax(axis=1)
        
        accuracy = accuracy_score(y_test, pred_classes)
        f1 = f1_score(y_test, pred_classes, average='weighted')
        
        print(f"\nClassification Metrics:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  Mean uncertainty: {uncertainties.mean():.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, pred_classes))
    
    # SHAP Analysis for Interpretability
    print("\n" + "="*60)
    print("SHAP ANALYSIS FOR INTERPRETABILITY")
    print("="*60)
    
    # For SHAP, we need a predict function
    if isinstance(model, EnsembleModel):
        def predict_fn(x):
            x_tensor = torch.FloatTensor(x)
            pred, _ = model.predict_with_uncertainty(x_tensor)
            return pred.numpy()
    else:
        def predict_fn(x):
            model.eval()
            x_tensor = torch.FloatTensor(x)
            with torch.no_grad():
                return model(x_tensor).numpy()
    
    # Create SHAP explainer
    explainer = shap.KernelExplainer(predict_fn, X_train[:100])  # Use subset for efficiency
    shap_values = explainer.shap_values(X_test[:100])
    
    # Plot SHAP summary
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test[:100], feature_names=feature_cols, show=False)
    plt.tight_layout()
    plt.savefig('results/cicy_shap_summary.png')
    plt.show()
    
    # Feature importance
    if task_type == 'regression':
        # Ensure shap_values is 2D for regression
        if len(shap_values.shape) == 3:
            shap_values = shap_values.squeeze()
        feature_importance = np.abs(shap_values).mean(axis=0)
    else:
        if len(shap_values.shape) == 3:  # Multi-class
            feature_importance = np.abs(shap_values).mean(axis=(0, 2))
        else:
            feature_importance = np.abs(shap_values).mean(axis=0)
    
    # Ensure feature_importance is 1D
    if len(feature_importance.shape) > 1:
        feature_importance = feature_importance.flatten()
    
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(importance_df.head(10))
    
    # Save results
    results = {
        'target_type': target_type,
        'task_type': task_type,
        'use_ensemble': use_ensemble,
        'use_mc_dropout': use_mc_dropout,
        'metrics': {
            'rmse': float(rmse) if task_type == 'regression' else None,
            'mae': float(mae) if task_type == 'regression' else None,
            'r2': float(r2) if task_type == 'regression' else None,
            'accuracy': float(accuracy) if task_type == 'classification' else None,
            'f1': float(f1) if task_type == 'classification' else None,
            'mean_uncertainty': float(uncertainties.mean()),
            'uncertainty_std': float(uncertainties.std())
        },
        'feature_importance': importance_df.to_dict('records')
    }
    
    with open('results/cicy_real_data_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Results saved to results/")
    
    return model, results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train on real CICY data')
    parser.add_argument('--target', type=str, default='h11_prediction',
                       choices=['h11_prediction', 'h21_prediction', 'mirror_symmetry', 'topology_class'],
                       help='Target type for prediction')
    parser.add_argument('--ensemble', action='store_true', help='Use ensemble model')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--device', type=str, default='cpu', help='Device to train on')
    
    args = parser.parse_args()
    
    model, results = train_on_real_cicy_data(
        target_type=args.target,
        use_ensemble=args.ensemble,
        n_epochs=args.epochs,
        device=args.device
    )

"""
Evaluation and Visualization Tools for Calabi-Yau Models
Includes model interpretation, performance metrics, and visualization utilities
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    classification_report
)
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import shap
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json


class ModelEvaluator:
    """Comprehensive model evaluation and analysis."""
    
    def __init__(self, model: nn.Module, device: str = 'cpu', task_type: str = 'regression'):
        """
        Initialize evaluator.
        
        Args:
            model: Trained neural network model
            device: Device for computation
            task_type: 'regression' or 'classification'
        """
        self.model = model.to(device)
        self.device = device
        self.task_type = task_type
        self.model.eval()
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on input data.
        
        Args:
            X: Input features as numpy array
            
        Returns:
            Predictions as numpy array
        """
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            
            if self.task_type == 'regression':
                predictions = outputs.cpu().numpy()
            else:
                if self.model.output_dim == 1:
                    predictions = (outputs > 0.5).float().cpu().numpy()
                else:
                    predictions = torch.argmax(outputs, dim=1).cpu().numpy()
        
        return predictions
    
    def evaluate_regression(self, X: np.ndarray, y_true: np.ndarray) -> Dict:
        """
        Evaluate regression model performance.
        
        Args:
            X: Input features
            y_true: True target values
            
        Returns:
            Dictionary of metrics
        """
        y_pred = self.predict(X).flatten()
        y_true = y_true.flatten()
        
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100,
            'correlation': np.corrcoef(y_true, y_pred)[0, 1]
        }
        
        # Calculate residuals
        residuals = y_true - y_pred
        metrics['residual_mean'] = np.mean(residuals)
        metrics['residual_std'] = np.std(residuals)
        
        return metrics
    
    def evaluate_classification(self, X: np.ndarray, y_true: np.ndarray) -> Dict:
        """
        Evaluate classification model performance.
        
        Args:
            X: Input features
            y_true: True class labels
            
        Returns:
            Dictionary of metrics
        """
        y_pred = self.predict(X)
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred)
        }
        
        # Calculate per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        metrics['precision_per_class'] = precision.tolist()
        metrics['recall_per_class'] = recall.tolist()
        metrics['f1_per_class'] = f1.tolist()
        metrics['support_per_class'] = support.tolist()
        
        # Calculate weighted averages
        metrics['precision_weighted'] = np.average(precision, weights=support)
        metrics['recall_weighted'] = np.average(recall, weights=support)
        metrics['f1_weighted'] = np.average(f1, weights=support)
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
        
        return metrics
    
    def plot_regression_results(self, X: np.ndarray, y_true: np.ndarray, 
                               save_path: Optional[str] = None):
        """
        Create comprehensive regression visualization.
        
        Args:
            X: Input features
            y_true: True target values
            save_path: Path to save figure
        """
        y_pred = self.predict(X).flatten()
        y_true = y_true.flatten()
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. Predicted vs True
        axes[0, 0].scatter(y_true, y_pred, alpha=0.5, s=10)
        axes[0, 0].plot([y_true.min(), y_true.max()], 
                       [y_true.min(), y_true.max()], 
                       'r--', lw=2)
        axes[0, 0].set_xlabel('True Values')
        axes[0, 0].set_ylabel('Predictions')
        axes[0, 0].set_title(f'Predicted vs True (R² = {r2_score(y_true, y_pred):.3f})')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Residual plot
        axes[0, 1].scatter(y_pred, residuals, alpha=0.5, s=10)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Predicted Values')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residual Plot')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Residual histogram
        axes[0, 2].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 2].set_xlabel('Residuals')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title(f'Residual Distribution (μ={np.mean(residuals):.3f}, σ={np.std(residuals):.3f})')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Error distribution
        relative_error = np.abs(residuals) / (np.abs(y_true) + 1e-8) * 100
        axes[1, 1].hist(relative_error, bins=50, edgecolor='black', alpha=0.7)
        axes[1, 1].set_xlabel('Relative Error (%)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title(f'Relative Error Distribution (Median: {np.median(relative_error):.2f}%)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Prediction density
        axes[1, 2].hexbin(y_true, y_pred, gridsize=30, cmap='YlOrRd')
        axes[1, 2].plot([y_true.min(), y_true.max()], 
                       [y_true.min(), y_true.max()], 
                       'b--', lw=2)
        axes[1, 2].set_xlabel('True Values')
        axes[1, 2].set_ylabel('Predictions')
        axes[1, 2].set_title('Prediction Density')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.suptitle('Regression Model Evaluation', fontsize=16, y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.show()
    
    def plot_classification_results(self, X: np.ndarray, y_true: np.ndarray,
                                  class_names: Optional[List[str]] = None,
                                  save_path: Optional[str] = None):
        """
        Create comprehensive classification visualization.
        
        Args:
            X: Input features
            y_true: True class labels
            class_names: Names of classes
            save_path: Path to save figure
        """
        y_pred = self.predict(X)
        cm = confusion_matrix(y_true, y_pred)
        
        n_classes = len(np.unique(y_true))
        if class_names is None:
            class_names = [f'Class {i}' for i in range(n_classes)]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Confusion Matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names,
                   ax=axes[0, 0])
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('True')
        axes[0, 0].set_title('Confusion Matrix')
        
        # 2. Normalized Confusion Matrix
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names,
                   ax=axes[0, 1])
        axes[0, 1].set_xlabel('Predicted')
        axes[0, 1].set_ylabel('True')
        axes[0, 1].set_title('Normalized Confusion Matrix')
        
        # 3. Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        x_pos = np.arange(len(class_names))
        width = 0.25
        
        axes[1, 0].bar(x_pos - width, precision, width, label='Precision', alpha=0.8)
        axes[1, 0].bar(x_pos, recall, width, label='Recall', alpha=0.8)
        axes[1, 0].bar(x_pos + width, f1, width, label='F1-Score', alpha=0.8)
        axes[1, 0].set_xlabel('Classes')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_title('Per-Class Metrics')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(class_names)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Class distribution
        unique, counts = np.unique(y_true, return_counts=True)
        pred_unique, pred_counts = np.unique(y_pred, return_counts=True)
        
        x_pos = np.arange(len(class_names))
        width = 0.35
        
        axes[1, 1].bar(x_pos - width/2, counts, width, label='True', alpha=0.8)
        axes[1, 1].bar(x_pos + width/2, pred_counts, width, label='Predicted', alpha=0.8)
        axes[1, 1].set_xlabel('Classes')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Class Distribution')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(class_names)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'Classification Model Evaluation (Accuracy: {accuracy_score(y_true, y_pred):.3f})', 
                    fontsize=16, y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.show()
    
    def get_feature_importance_shap(self, X: np.ndarray, feature_names: List[str],
                                   n_samples: int = 100) -> np.ndarray:
        """
        Calculate feature importance using SHAP values.
        
        Args:
            X: Input features
            feature_names: Names of features
            n_samples: Number of samples to use for SHAP
            
        Returns:
            Feature importance scores
        """
        # Sample data if too large
        if len(X) > n_samples:
            indices = np.random.choice(len(X), n_samples, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X
        
        # Create SHAP explainer
        X_tensor = torch.FloatTensor(X_sample).to(self.device)
        
        def model_predict(x):
            x_tensor = torch.FloatTensor(x).to(self.device)
            with torch.no_grad():
                output = self.model(x_tensor)
                if self.task_type == 'classification' and self.model.output_dim > 1:
                    output = torch.softmax(output, dim=1)
            return output.cpu().numpy()
        
        # Use KernelExplainer for neural networks
        explainer = shap.KernelExplainer(model_predict, X_sample)
        shap_values = explainer.shap_values(X_sample[:min(50, len(X_sample))])
        
        # Calculate mean absolute SHAP values
        if isinstance(shap_values, list):
            # Multi-class classification
            importance = np.abs(shap_values[0]).mean(axis=0)
        else:
            importance = np.abs(shap_values).mean(axis=0)
        
        return importance
    
    def plot_feature_importance(self, X: np.ndarray, feature_names: List[str],
                              save_path: Optional[str] = None):
        """
        Plot feature importance.
        
        Args:
            X: Input features
            feature_names: Names of features
            save_path: Path to save figure
        """
        # Get gradient-based importance
        X_tensor = torch.FloatTensor(X).to(self.device)
        X_tensor.requires_grad = True
        
        output = self.model(X_tensor)
        if self.task_type == 'regression':
            output.sum().backward()
        else:
            if self.model.output_dim > 1:
                output.max(dim=1)[0].sum().backward()
            else:
                output.sum().backward()
        
        importance = torch.abs(X_tensor.grad).mean(dim=0).cpu().numpy()
        importance = importance / importance.sum()
        
        # Sort features by importance
        indices = np.argsort(importance)[::-1]
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(importance)), importance[indices], alpha=0.8)
        plt.xlabel('Features')
        plt.ylabel('Importance Score')
        plt.title('Feature Importance (Gradient-based)')
        plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.show()
        
        return dict(zip(feature_names, importance))
    
    def visualize_embeddings(self, X: np.ndarray, y: np.ndarray,
                           method: str = 'tsne', save_path: Optional[str] = None):
        """
        Visualize learned embeddings using dimensionality reduction.
        
        Args:
            X: Input features
            y: Target values/labels
            method: 'tsne' or 'pca'
            save_path: Path to save figure
        """
        # Get embeddings from the model
        if hasattr(self.model, 'get_embeddings'):
            X_tensor = torch.FloatTensor(X).to(self.device)
            with torch.no_grad():
                embeddings = self.model.get_embeddings(X_tensor).cpu().numpy()
        else:
            # Use activations from second-to-last layer
            embeddings = X  # Fallback to original features
        
        # Dimensionality reduction
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
            embeddings_2d = reducer.fit_transform(embeddings)
        elif method == 'pca':
            reducer = PCA(n_components=2)
            embeddings_2d = reducer.fit_transform(embeddings)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Plot
        plt.figure(figsize=(10, 8))
        
        if self.task_type == 'regression':
            scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                c=y, cmap='viridis', alpha=0.6, s=20)
            plt.colorbar(scatter, label='Target Value')
        else:
            unique_classes = np.unique(y)
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_classes)))
            for i, cls in enumerate(unique_classes):
                mask = y == cls
                plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                          c=[colors[i]], label=f'Class {cls}', alpha=0.6, s=20)
            plt.legend()
        
        plt.xlabel(f'{method.upper()} Component 1')
        plt.ylabel(f'{method.upper()} Component 2')
        plt.title(f'Learned Embeddings Visualization ({method.upper()})')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.show()


def evaluate_model(model_path: str, data_path: str, task_type: str = 'regression',
                  output_dir: str = 'results'):
    """
    Complete model evaluation pipeline.
    
    Args:
        model_path: Path to saved model
        data_path: Path to test data
        task_type: 'regression' or 'classification'
        output_dir: Directory to save results
    """
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Path(output_dir).mkdir(exist_ok=True)
    
    # Load model
    from models import create_model
    checkpoint = torch.load(model_path, map_location=device)
    
    # Load test data
    test_df = pd.read_csv(data_path)
    feature_cols = [col for col in test_df.columns if col != 'particle_spectrum']
    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df['particle_spectrum'].values.astype(np.float32)
    
    # Normalize features (using training data statistics if available, else use test data)
    # For now, use test data statistics (in production, save training stats)
    X_mean = X_test.mean(axis=0)
    X_std = X_test.std(axis=0) + 1e-8
    X_test = (X_test - X_mean) / X_std
    
    # Recreate model architecture
    input_dim = len(feature_cols)
    output_dim = 1 if task_type == 'regression' else len(np.unique(y_test))
    
    model = create_model(
        model_type='mlp',
        input_dim=input_dim,
        output_dim=output_dim,
        task_type=task_type,
        hidden_dims=[128, 64, 32]
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create evaluator
    evaluator = ModelEvaluator(model, device, task_type)
    
    # Evaluate performance
    print("Evaluating model performance...")
    if task_type == 'regression':
        metrics = evaluator.evaluate_regression(X_test, y_test)
        print("\nRegression Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
        
        # Create visualizations
        evaluator.plot_regression_results(
            X_test, y_test, 
            save_path=f"{output_dir}/regression_evaluation.png"
        )
    else:
        metrics = evaluator.evaluate_classification(X_test, y_test)
        print("\nClassification Metrics:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1 (weighted): {metrics['f1_weighted']:.4f}")
        
        # Create visualizations
        evaluator.plot_classification_results(
            X_test, y_test,
            save_path=f"{output_dir}/classification_evaluation.png"
        )
    
    # Feature importance
    print("\nCalculating feature importance...")
    importance = evaluator.plot_feature_importance(
        X_test[:100], feature_cols,
        save_path=f"{output_dir}/feature_importance.png"
    )
    
    # Visualize embeddings
    print("Visualizing learned embeddings...")
    evaluator.visualize_embeddings(
        X_test[:500], y_test[:500],
        method='tsne',
        save_path=f"{output_dir}/embeddings_tsne.png"
    )
    
    # Save metrics (convert numpy types to Python types for JSON serialization)
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    metrics_serializable = convert_to_serializable(metrics)
    with open(f"{output_dir}/evaluation_metrics.json", 'w') as f:
        json.dump(metrics_serializable, f, indent=2)
    
    print(f"\nEvaluation complete! Results saved to '{output_dir}/'")
    
    return metrics


if __name__ == "__main__":
    # Example usage
    evaluate_model(
        model_path='models/best_model.pth',
        data_path='data/calabi_yau_test_regression.csv',
        task_type='regression',
        output_dir='results'
    )

#!/usr/bin/env python
"""
Advanced models: Bayesian Neural Networks and Symbolic Regression.

This module implements:
1. Bayesian Neural Networks using Pyro for uncertainty quantification
2. Symbolic regression using PySR for equation discovery
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

# Bayesian Neural Network with Pyro
try:
    import pyro
    import pyro.distributions as dist
    from pyro.nn import PyroModule, PyroSample
    from pyro.infer import SVI, Trace_ELBO, Predictive
    from pyro.optim import Adam
    PYRO_AVAILABLE = True
except ImportError:
    PYRO_AVAILABLE = False
    print("Warning: Pyro not installed. Bayesian models unavailable.")

# Symbolic Regression with PySR
try:
    from pysr import PySRRegressor
    PYSR_AVAILABLE = True
except ImportError:
    PYSR_AVAILABLE = False
    print("Warning: PySR not installed. Symbolic regression unavailable.")


class BayesianNN(PyroModule):
    """Bayesian Neural Network with Pyro."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int):
        """
        Initialize Bayesian NN.
        
        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension
        """
        super().__init__()
        
        if not PYRO_AVAILABLE:
            raise ImportError("Pyro is required for Bayesian models. Install with: pip install pyro-ppl")
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Build layers with Bayesian priors
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            # Bayesian linear layer
            layer = PyroModule[nn.Linear](prev_dim, hidden_dim)
            
            # Set priors for weights and biases
            layer.weight = PyroSample(dist.Normal(0., 1.).expand([hidden_dim, prev_dim]).to_event(2))
            layer.bias = PyroSample(dist.Normal(0., 1.).expand([hidden_dim]).to_event(1))
            
            layers.append(layer)
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        # Output layer
        output_layer = PyroModule[nn.Linear](prev_dim, output_dim)
        output_layer.weight = PyroSample(dist.Normal(0., 1.).expand([output_dim, prev_dim]).to_event(2))
        output_layer.bias = PyroSample(dist.Normal(0., 1.).expand([output_dim]).to_event(1))
        layers.append(output_layer)
        
        self.model = PyroModule[nn.Sequential](*layers)
    
    def forward(self, x, y=None):
        """
        Forward pass with optional observed data.
        
        Args:
            x: Input tensor
            y: Target tensor (optional, for training)
            
        Returns:
            Predictions or samples
        """
        mean = self.model(x)
        
        # Observation noise
        sigma = pyro.sample("sigma", dist.Uniform(0., 10.))
        
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mean, sigma), obs=y)
        
        return mean


class BayesianEnsemble:
    """Ensemble of Bayesian Neural Networks."""
    
    def __init__(self, 
                 n_models: int = 3,
                 input_dim: int = 10,
                 output_dim: int = 1):
        """
        Initialize Bayesian ensemble.
        
        Args:
            n_models: Number of Bayesian models
            input_dim: Input dimension
            output_dim: Output dimension
        """
        if not PYRO_AVAILABLE:
            raise ImportError("Pyro is required for Bayesian models")
        
        self.n_models = n_models
        self.models = []
        self.guides = []
        self.predictives = []
        
        # Different architectures for diversity
        architectures = [
            [64, 32],
            [128, 64, 32],
            [64, 64],
        ]
        
        for i in range(n_models):
            arch = architectures[i % len(architectures)]
            model = BayesianNN(input_dim, arch, output_dim)
            guide = pyro.infer.autoguide.AutoDiagonalNormal(model)
            
            self.models.append(model)
            self.guides.append(guide)
    
    def train_ensemble(self, X_train, y_train, n_epochs: int = 1000):
        """
        Train all Bayesian models in the ensemble.
        
        Args:
            X_train: Training features
            y_train: Training targets
            n_epochs: Number of training epochs
        """
        X_train = torch.FloatTensor(X_train)
        y_train = torch.FloatTensor(y_train)
        
        for i, (model, guide) in enumerate(zip(self.models, self.guides)):
            print(f"\nTraining Bayesian model {i+1}/{self.n_models}...")
            
            # Clear param store
            pyro.clear_param_store()
            
            # Setup SVI
            adam = Adam({"lr": 0.01})
            svi = SVI(model, guide, adam, loss=Trace_ELBO())
            
            # Training loop
            for epoch in range(n_epochs):
                loss = svi.step(X_train, y_train)
                if epoch % 100 == 0:
                    print(f"Epoch {epoch}: Loss = {loss:.4f}")
            
            # Create predictive distribution
            predictive = Predictive(model, guide=guide, num_samples=100)
            self.predictives.append(predictive)
    
    def predict_with_uncertainty(self, X_test, n_samples: int = 100):
        """
        Make predictions with full Bayesian uncertainty.
        
        Args:
            X_test: Test features
            n_samples: Number of posterior samples
            
        Returns:
            Tuple of (mean predictions, epistemic uncertainty, aleatoric uncertainty)
        """
        X_test = torch.FloatTensor(X_test)
        
        all_predictions = []
        all_sigmas = []
        
        for predictive in self.predictives:
            samples = predictive(X_test)
            predictions = samples['obs'].numpy()  # Shape: (n_samples, n_data)
            sigmas = samples['sigma'].numpy()  # Shape: (n_samples,)
            
            all_predictions.append(predictions)
            all_sigmas.append(sigmas)
        
        # Combine predictions from all models
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_sigmas = np.concatenate(all_sigmas, axis=0)
        
        # Calculate uncertainties
        mean_pred = all_predictions.mean(axis=0)
        
        # Epistemic uncertainty (model uncertainty)
        epistemic_var = all_predictions.var(axis=0)
        
        # Aleatoric uncertainty (data noise)
        aleatoric_var = (all_sigmas ** 2).mean()
        
        # Total uncertainty
        total_uncertainty = np.sqrt(epistemic_var + aleatoric_var)
        
        return mean_pred, np.sqrt(epistemic_var), np.sqrt(aleatoric_var), total_uncertainty


class SymbolicRegressor:
    """Symbolic regression for discovering physics equations."""
    
    def __init__(self, 
                 niterations: int = 100,
                 binary_operators: List[str] = None,
                 unary_operators: List[str] = None,
                 complexity_penalty: float = 0.001):
        """
        Initialize symbolic regressor.
        
        Args:
            niterations: Number of iterations for equation search
            binary_operators: Binary operations to use
            unary_operators: Unary operations to use
            complexity_penalty: Penalty for equation complexity
        """
        if not PYSR_AVAILABLE:
            raise ImportError("PySR is required for symbolic regression. Install with: pip install pysr")
        
        if binary_operators is None:
            binary_operators = ["+", "-", "*", "/", "^"]
        
        if unary_operators is None:
            unary_operators = ["sin", "cos", "exp", "log", "sqrt", "abs"]
        
        self.model = PySRRegressor(
            niterations=niterations,
            binary_operators=binary_operators,
            unary_operators=unary_operators,
            complexity_of_operators={"/": 2, "^": 3},
            model_selection="best",
            loss="loss(y, y_pred) = (y - y_pred)^2",
            maxsize=20,
            maxdepth=10,
            parsimony=complexity_penalty,
            turbo=True,
            procs=4,
            populations=20,
            population_size=50,
            ncycles_per_iteration=500,
            fraction_replaced=0.1,
            fraction_replaced_hof=0.05,
            optimize_probability=0.9,
            optimize_iterations=10,
            migration=True,
            hof_migration=True,
            should_optimize_constants=True,
            early_stop_condition=1e-6,
            progress=True,
            random_state=42
        )
    
    def discover_equation(self, X, y, feature_names: List[str] = None):
        """
        Discover symbolic equation from data.
        
        Args:
            X: Input features
            y: Target values
            feature_names: Names of features for equation
            
        Returns:
            Dictionary with discovered equations and metrics
        """
        print("Starting symbolic regression...")
        print(f"Data shape: X={X.shape}, y={y.shape}")
        
        # Fit the model
        self.model.fit(X, y, variable_names=feature_names)
        
        # Get best equations
        equations = self.model.equations_
        
        # Get Pareto front (complexity vs accuracy trade-off)
        pareto_front = []
        for i, eq in equations.iterrows():
            pareto_front.append({
                'equation': eq['equation'],
                'complexity': eq['complexity'],
                'loss': eq['loss'],
                'score': eq['score'] if 'score' in eq else None
            })
        
        # Sort by score
        pareto_front = sorted(pareto_front, key=lambda x: x['loss'])
        
        # Best equation
        best_eq = self.model.sympy()
        
        return {
            'best_equation': str(best_eq),
            'latex': self.model.latex(),
            'pareto_front': pareto_front[:5],  # Top 5 equations
            'feature_importance': self._calculate_feature_importance(best_eq, feature_names)
        }
    
    def _calculate_feature_importance(self, equation, feature_names):
        """Calculate feature importance from symbolic equation."""
        if feature_names is None:
            return {}
        
        importance = {}
        eq_str = str(equation)
        
        for feature in feature_names:
            # Count occurrences in equation
            count = eq_str.count(feature)
            importance[feature] = count
        
        # Normalize
        total = sum(importance.values()) + 1e-8
        importance = {k: v/total for k, v in importance.items()}
        
        return importance


def discover_cy_physics_equations(X, y, feature_names):
    """
    Discover physics equations for Calabi-Yau relationships.
    
    Args:
        X: Feature matrix
        y: Target values
        feature_names: Names of features
        
    Returns:
        Dictionary with discovered equations
    """
    if not PYSR_AVAILABLE:
        print("PySR not available. Cannot perform symbolic regression.")
        return None
    
    print("="*60)
    print("DISCOVERING PHYSICS EQUATIONS")
    print("="*60)
    
    # Initialize symbolic regressor with physics-relevant operators
    sr = SymbolicRegressor(
        niterations=50,  # Reduced for faster demo
        binary_operators=["+", "-", "*", "/", "^"],
        unary_operators=["sqrt", "abs", "log", "exp"],
        complexity_penalty=0.001
    )
    
    # Discover equation
    results = sr.discover_equation(X, y, feature_names)
    
    print(f"\nBest equation discovered:")
    print(f"  {results['best_equation']}")
    print(f"\nLaTeX form:")
    print(f"  {results['latex']}")
    
    print(f"\nPareto-optimal equations (complexity vs accuracy):")
    for i, eq in enumerate(results['pareto_front'], 1):
        print(f"  {i}. Complexity={eq['complexity']}, Loss={eq['loss']:.6f}")
        print(f"     {eq['equation']}")
    
    print(f"\nFeature importance in equation:")
    for feature, importance in sorted(results['feature_importance'].items(), 
                                     key=lambda x: x[1], reverse=True):
        if importance > 0:
            print(f"  {feature}: {importance:.3f}")
    
    return results


if __name__ == "__main__":
    print("Advanced Models: Bayesian NN and Symbolic Regression")
    print("="*60)
    
    # Test data
    np.random.seed(42)
    X = np.random.randn(100, 3)
    y = 2*X[:, 0] + X[:, 1]**2 - 0.5*X[:, 2] + np.random.randn(100)*0.1
    
    if PYRO_AVAILABLE:
        print("\nTesting Bayesian Neural Network...")
        bnn = BayesianNN(input_dim=3, hidden_dims=[16, 8], output_dim=1)
        print("Bayesian NN created successfully!")
    
    if PYSR_AVAILABLE:
        print("\nTesting Symbolic Regression...")
        sr = SymbolicRegressor(niterations=10)
        print("Symbolic Regressor created successfully!")
        
        # Quick test
        results = sr.discover_equation(
            X[:50], y[:50], 
            feature_names=['x1', 'x2', 'x3']
        )
        print(f"Discovered equation: {results['best_equation']}")
    
    print("\nAdvanced models ready!")

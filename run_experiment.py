#!/usr/bin/env python
"""
Quick experiment runner for Calabi-Yau to Particle Spectra Learning
Run this script to execute the complete pipeline with a single command
"""

import sys
import os
sys.path.append('src')

import argparse
from pathlib import Path
import json
import time

from src.data_generator import generate_and_save_datasets
from src.train import main as train_main
from src.evaluate import evaluate_model


def run_complete_experiment(task_type='regression', n_samples=5000, n_epochs=100, model_type='mlp'):
    """
    Run the complete experiment pipeline.
    
    Args:
        task_type: 'regression' or 'classification'
        n_samples: Number of samples to generate
        n_epochs: Number of training epochs
        model_type: 'mlp', 'deep', or 'ensemble'
    """
    print("="*60)
    print("CALABI-YAU GEOMETRY TO PARTICLE SPECTRA LEARNING")
    print("="*60)
    
    start_time = time.time()
    
    # Step 1: Generate Data
    print("\n📊 Step 1: Generating synthetic data...")
    print("-"*40)
    data_dir = 'data'
    Path(data_dir).mkdir(exist_ok=True)
    
    if not Path(f'{data_dir}/calabi_yau_train_{task_type}.csv').exists():
        generate_and_save_datasets(output_dir=data_dir, n_samples=n_samples)
    else:
        print("Data already exists, skipping generation.")
    
    # Step 2: Train Model
    print("\n🧠 Step 2: Training neural network...")
    print("-"*40)
    print(f"Model type: {model_type}")
    print(f"Task type: {task_type}")
    print(f"Epochs: {n_epochs}")
    
    # Modify config for training
    import src.train as train_module
    original_main = train_module.main
    
    def custom_train():
        # Override config
        train_module.config = {
            'data_dir': data_dir,
            'model_type': model_type,
            'task_type': task_type,
            'batch_size': 32,
            'n_epochs': n_epochs,
            'learning_rate': 1e-3,
            'early_stopping_patience': 20,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        original_main()
    
    import torch
    custom_train()
    
    # Step 3: Evaluate Model
    print("\n📈 Step 3: Evaluating model performance...")
    print("-"*40)
    
    metrics = evaluate_model(
        model_path='models/best_model.pth',
        data_path=f'{data_dir}/calabi_yau_test_{task_type}.csv',
        task_type=task_type,
        output_dir='results'
    )
    
    # Step 4: Summary
    elapsed_time = time.time() - start_time
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE!")
    print("="*60)
    
    if task_type == 'regression':
        print(f"\n📊 Final Results:")
        print(f"  - Test RMSE: {metrics.get('rmse', 0):.4f}")
        print(f"  - Test R²: {metrics.get('r2', 0):.4f}")
        print(f"  - Test MAE: {metrics.get('mae', 0):.4f}")
    else:
        print(f"\n📊 Final Results:")
        print(f"  - Test Accuracy: {metrics.get('accuracy', 0):.4f}")
        print(f"  - F1 Score (weighted): {metrics.get('f1_weighted', 0):.4f}")
    
    print(f"\n⏱️ Total time: {elapsed_time:.1f} seconds")
    print(f"\n📁 Results saved to:")
    print(f"  - Models: models/")
    print(f"  - Plots: results/")
    print(f"  - Metrics: results/evaluation_metrics.json")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description='Run Calabi-Yau to Particle Spectra Learning Experiment'
    )
    parser.add_argument(
        '--task', 
        type=str, 
        default='regression',
        choices=['regression', 'classification'],
        help='Task type: regression or classification'
    )
    parser.add_argument(
        '--samples', 
        type=int, 
        default=5000,
        help='Number of samples to generate'
    )
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=100,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--model', 
        type=str, 
        default='mlp',
        choices=['mlp', 'deep', 'ensemble'],
        help='Model architecture type'
    )
    
    args = parser.parse_args()
    
    # Run experiment
    metrics = run_complete_experiment(
        task_type=args.task,
        n_samples=args.samples,
        n_epochs=args.epochs,
        model_type=args.model
    )
    
    # Save command line arguments and results
    def convert_to_serializable(obj):
        """Convert numpy types to Python types for JSON serialization."""
        import numpy as np
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
    
    experiment_info = {
        'command': ' '.join(sys.argv),
        'arguments': vars(args),
        'metrics': convert_to_serializable(metrics)
    }
    
    with open('results/experiment_info.json', 'w') as f:
        json.dump(experiment_info, f, indent=2)


if __name__ == "__main__":
    main()

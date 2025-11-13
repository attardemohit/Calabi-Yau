#!/usr/bin/env python
"""Benchmark model performance."""

import time
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import json
from typing import Dict, List
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models import MLPRegressor, DeepResNet, EnsembleModel
from utils import timer, PerformanceMonitor


class ModelBenchmark:
    """Benchmark different model architectures."""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.results = []
    
    def benchmark_model(
        self,
        model: torch.nn.Module,
        model_name: str,
        input_shape: tuple,
        n_iterations: int = 100,
        batch_sizes: List[int] = [1, 8, 32, 128]
    ) -> Dict:
        """
        Benchmark a single model.
        
        Args:
            model: Model to benchmark
            model_name: Name of the model
            input_shape: Input shape (batch_size excluded)
            n_iterations: Number of iterations
            batch_sizes: Batch sizes to test
            
        Returns:
            Benchmark results
        """
        model = model.to(self.device)
        model.eval()
        
        results = {
            "model": model_name,
            "parameters": sum(p.numel() for p in model.parameters()),
            "device": self.device,
            "benchmarks": []
        }
        
        for batch_size in batch_sizes:
            # Prepare input
            x = torch.randn(batch_size, *input_shape).to(self.device)
            
            # Warmup
            for _ in range(10):
                with torch.no_grad():
                    _ = model(x)
            
            # Benchmark forward pass
            torch.cuda.synchronize() if self.device == "cuda" else None
            start_time = time.time()
            
            for _ in range(n_iterations):
                with torch.no_grad():
                    _ = model(x)
            
            torch.cuda.synchronize() if self.device == "cuda" else None
            forward_time = (time.time() - start_time) / n_iterations * 1000  # ms
            
            # Benchmark backward pass
            x.requires_grad = True
            optimizer = torch.optim.Adam(model.parameters())
            
            torch.cuda.synchronize() if self.device == "cuda" else None
            start_time = time.time()
            
            for _ in range(n_iterations):
                optimizer.zero_grad()
                output = model(x)
                loss = output.mean()
                loss.backward()
                optimizer.step()
            
            torch.cuda.synchronize() if self.device == "cuda" else None
            backward_time = (time.time() - start_time) / n_iterations * 1000  # ms
            
            # Memory usage
            if self.device == "cuda":
                memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
                torch.cuda.reset_peak_memory_stats()
            else:
                memory_mb = 0
            
            results["benchmarks"].append({
                "batch_size": batch_size,
                "forward_time_ms": forward_time,
                "backward_time_ms": backward_time,
                "throughput": batch_size / (forward_time / 1000),  # samples/sec
                "memory_mb": memory_mb
            })
            
            print(f"{model_name} - Batch {batch_size}: "
                  f"Forward {forward_time:.2f}ms, "
                  f"Backward {backward_time:.2f}ms, "
                  f"Memory {memory_mb:.1f}MB")
        
        return results
    
    def run_benchmarks(self):
        """Run all benchmarks."""
        input_dim = 10
        input_shape = (input_dim,)
        
        # Models to benchmark
        models = [
            (MLPRegressor(input_dim, [64, 32], 1), "MLP-Small"),
            (MLPRegressor(input_dim, [128, 64, 32], 1), "MLP-Medium"),
            (MLPRegressor(input_dim, [256, 128, 64, 32], 1), "MLP-Large"),
            (DeepResNet(input_dim, 64, 1, 2), "ResNet-2"),
            (DeepResNet(input_dim, 64, 1, 4), "ResNet-4"),
            (EnsembleModel(input_dim, 1, 3), "Ensemble-3"),
            (EnsembleModel(input_dim, 1, 5), "Ensemble-5"),
        ]
        
        for model, name in models:
            print(f"\nBenchmarking {name}...")
            result = self.benchmark_model(model, name, input_shape)
            self.results.append(result)
    
    def save_results(self, output_dir: str = "benchmarks/results"):
        """Save benchmark results."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON
        with open(output_dir / "model_benchmarks.json", "w") as f:
            json.dump(self.results, f, indent=2)
        
        # Create DataFrame for analysis
        rows = []
        for result in self.results:
            for bench in result["benchmarks"]:
                row = {
                    "model": result["model"],
                    "parameters": result["parameters"],
                    **bench
                }
                rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_dir / "model_benchmarks.csv", index=False)
        
        # Create plots
        self.plot_results(df, output_dir)
    
    def plot_results(self, df: pd.DataFrame, output_dir: Path):
        """Create benchmark plots."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Forward time vs batch size
        ax = axes[0, 0]
        for model in df["model"].unique():
            model_df = df[df["model"] == model]
            ax.plot(model_df["batch_size"], model_df["forward_time_ms"], 
                   marker="o", label=model)
        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Forward Time (ms)")
        ax.set_title("Forward Pass Performance")
        ax.legend()
        ax.grid(True)
        
        # Throughput vs batch size
        ax = axes[0, 1]
        for model in df["model"].unique():
            model_df = df[df["model"] == model]
            ax.plot(model_df["batch_size"], model_df["throughput"], 
                   marker="o", label=model)
        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Throughput (samples/sec)")
        ax.set_title("Inference Throughput")
        ax.legend()
        ax.grid(True)
        
        # Memory usage
        ax = axes[1, 0]
        if self.device == "cuda":
            for model in df["model"].unique():
                model_df = df[df["model"] == model]
                ax.plot(model_df["batch_size"], model_df["memory_mb"], 
                       marker="o", label=model)
            ax.set_xlabel("Batch Size")
            ax.set_ylabel("Memory (MB)")
            ax.set_title("GPU Memory Usage")
            ax.legend()
            ax.grid(True)
        else:
            ax.text(0.5, 0.5, "GPU not available", 
                   ha="center", va="center", transform=ax.transAxes)
        
        # Model comparison
        ax = axes[1, 1]
        model_params = df.groupby("model")["parameters"].first()
        model_time = df[df["batch_size"] == 32].groupby("model")["forward_time_ms"].first()
        
        ax.scatter(model_params, model_time)
        for model in model_params.index:
            ax.annotate(model, (model_params[model], model_time[model]),
                       xytext=(5, 5), textcoords="offset points", fontsize=8)
        ax.set_xlabel("Number of Parameters")
        ax.set_ylabel("Forward Time @ Batch=32 (ms)")
        ax.set_title("Model Efficiency")
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(output_dir / "model_benchmarks.png", dpi=150)
        plt.close()


def benchmark_optimizations():
    """Benchmark various optimization techniques."""
    print("\n" + "="*60)
    print("OPTIMIZATION BENCHMARKS")
    print("="*60)
    
    input_dim = 10
    batch_size = 32
    n_iterations = 100
    
    model = MLPRegressor(input_dim, [128, 64, 32], 1)
    x = torch.randn(batch_size, input_dim)
    
    results = {}
    
    # Baseline
    with timer("Baseline"):
        model.train()
        for _ in range(n_iterations):
            output = model(x)
            loss = output.mean()
            loss.backward()
    
    # With torch.no_grad()
    with timer("With no_grad"):
        model.eval()
        with torch.no_grad():
            for _ in range(n_iterations):
                output = model(x)
    
    # With torch.jit.script (if available)
    try:
        scripted_model = torch.jit.script(model)
        with timer("JIT Scripted"):
            scripted_model.eval()
            with torch.no_grad():
                for _ in range(n_iterations):
                    output = scripted_model(x)
    except:
        print("JIT scripting not available")
    
    # With torch.compile (PyTorch 2.0+)
    if hasattr(torch, "compile"):
        try:
            compiled_model = torch.compile(model)
            with timer("Torch Compiled"):
                compiled_model.eval()
                with torch.no_grad():
                    for _ in range(n_iterations):
                        output = compiled_model(x)
        except:
            print("torch.compile not available")
    
    return results


if __name__ == "__main__":
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running benchmarks on {device}")
    
    # Run model benchmarks
    benchmark = ModelBenchmark(device)
    benchmark.run_benchmarks()
    benchmark.save_results()
    
    # Run optimization benchmarks
    benchmark_optimizations()
    
    print("\n" + "="*60)
    print("BENCHMARKING COMPLETE")
    print("Results saved to benchmarks/results/")
    print("="*60)

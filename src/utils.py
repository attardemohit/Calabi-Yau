#!/usr/bin/env python
"""Utility functions for performance, logging, and monitoring."""

import time
import logging
import psutil
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from contextlib import contextmanager
from functools import wraps
import json
import pickle
import joblib
from datetime import datetime


# Configure logging
def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        format_string: Optional custom format string
        
    Returns:
        Configured logger
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        handlers=[
            logging.StreamHandler(),
            *([] if log_file is None else [logging.FileHandler(log_file)])
        ]
    )
    
    return logging.getLogger(__name__)


logger = setup_logging()


# Performance monitoring
class PerformanceMonitor:
    """Monitor performance metrics during training."""
    
    def __init__(self):
        self.metrics = {
            "time": [],
            "memory": [],
            "gpu_memory": [],
            "cpu_percent": [],
            "gpu_utilization": []
        }
        self.start_time = None
    
    def start(self):
        """Start monitoring."""
        self.start_time = time.time()
    
    def record(self):
        """Record current metrics."""
        if self.start_time is None:
            self.start()
        
        # Time
        self.metrics["time"].append(time.time() - self.start_time)
        
        # CPU metrics
        self.metrics["memory"].append(psutil.virtual_memory().percent)
        self.metrics["cpu_percent"].append(psutil.cpu_percent())
        
        # GPU metrics (if available)
        if torch.cuda.is_available():
            self.metrics["gpu_memory"].append(
                torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100
            )
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                self.metrics["gpu_utilization"].append(util.gpu)
            except:
                self.metrics["gpu_utilization"].append(0)
        else:
            self.metrics["gpu_memory"].append(0)
            self.metrics["gpu_utilization"].append(0)
    
    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics."""
        summary = {}
        for key, values in self.metrics.items():
            if values:
                summary[f"{key}_mean"] = np.mean(values)
                summary[f"{key}_max"] = np.max(values)
                summary[f"{key}_min"] = np.min(values)
        return summary
    
    def save(self, path: str):
        """Save metrics to file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.metrics, f, indent=2)


# Timing utilities
@contextmanager
def timer(name: str = "Operation", logger: Optional[logging.Logger] = None):
    """
    Context manager for timing operations.
    
    Usage:
        with timer("Training"):
            train_model()
    """
    start = time.time()
    if logger:
        logger.info(f"{name} started...")
    else:
        print(f"{name} started...")
    
    try:
        yield
    finally:
        elapsed = time.time() - start
        message = f"{name} completed in {elapsed:.2f} seconds"
        if logger:
            logger.info(message)
        else:
            print(message)


def profile_function(func):
    """Decorator to profile function execution time and memory."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Start monitoring
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Execute function
        result = func(*args, **kwargs)
        
        # End monitoring
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Log results
        logger.debug(f"{func.__name__}: Time={end_time-start_time:.3f}s, "
                    f"Memory={end_memory-start_memory:.1f}MB")
        
        return result
    return wrapper


# Caching utilities
class Cache:
    """Simple cache for expensive computations."""
    
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory_cache = {}
    
    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for key."""
        return self.cache_dir / f"{key}.pkl"
    
    def get(self, key: str, compute_fn=None, use_memory: bool = True):
        """
        Get cached value or compute if not exists.
        
        Args:
            key: Cache key
            compute_fn: Function to compute value if not cached
            use_memory: Use in-memory cache
            
        Returns:
            Cached or computed value
        """
        # Check memory cache
        if use_memory and key in self.memory_cache:
            logger.debug(f"Cache hit (memory): {key}")
            return self.memory_cache[key]
        
        # Check disk cache
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            logger.debug(f"Cache hit (disk): {key}")
            with open(cache_path, "rb") as f:
                value = pickle.load(f)
            if use_memory:
                self.memory_cache[key] = value
            return value
        
        # Compute if not cached
        if compute_fn is not None:
            logger.debug(f"Cache miss: {key}, computing...")
            value = compute_fn()
            self.set(key, value, use_memory)
            return value
        
        return None
    
    def set(self, key: str, value: Any, use_memory: bool = True):
        """Set cache value."""
        # Memory cache
        if use_memory:
            self.memory_cache[key] = value
        
        # Disk cache
        cache_path = self._get_cache_path(key)
        with open(cache_path, "wb") as f:
            pickle.dump(value, f)
        
        logger.debug(f"Cached: {key}")
    
    def clear(self):
        """Clear all cache."""
        self.memory_cache.clear()
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
        logger.info("Cache cleared")


# Data optimization
def optimize_dataloader(
    dataloader: torch.utils.data.DataLoader,
    num_workers: Optional[int] = None,
    pin_memory: bool = None,
    prefetch_factor: int = 2
) -> torch.utils.data.DataLoader:
    """
    Optimize DataLoader for performance.
    
    Args:
        dataloader: Original DataLoader
        num_workers: Number of worker processes
        pin_memory: Pin memory for CUDA
        prefetch_factor: Number of batches to prefetch
        
    Returns:
        Optimized DataLoader
    """
    if num_workers is None:
        num_workers = min(4, psutil.cpu_count())
    
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    
    return torch.utils.data.DataLoader(
        dataloader.dataset,
        batch_size=dataloader.batch_size,
        shuffle=isinstance(dataloader.sampler, torch.utils.data.RandomSampler),
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=num_workers > 0
    )


# Model optimization
def optimize_model(model: torch.nn.Module) -> torch.nn.Module:
    """
    Apply optimizations to model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Optimized model
    """
    # Gradient checkpointing for memory efficiency
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")
    
    # Compile model (PyTorch 2.0+)
    if hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="reduce-overhead")
            logger.info("Model compiled with torch.compile")
        except:
            logger.warning("torch.compile failed, using regular model")
    
    # Mixed precision training setup
    if torch.cuda.is_available():
        model = model.cuda()
        # Enable TF32 on Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("CUDA optimizations enabled")
    
    return model


# Batch processing utilities
def batch_process(
    data: Union[List, np.ndarray, torch.Tensor],
    process_fn,
    batch_size: int = 32,
    show_progress: bool = True
) -> List:
    """
    Process data in batches.
    
    Args:
        data: Input data
        process_fn: Function to process each batch
        batch_size: Batch size
        show_progress: Show progress bar
        
    Returns:
        Processed results
    """
    from tqdm import tqdm
    
    n_samples = len(data)
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    results = []
    iterator = range(n_batches)
    
    if show_progress:
        iterator = tqdm(iterator, desc="Processing batches")
    
    for i in iterator:
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_samples)
        batch = data[start_idx:end_idx]
        
        batch_results = process_fn(batch)
        results.extend(batch_results if isinstance(batch_results, list) else [batch_results])
    
    return results


# Parallel processing
def parallel_process(
    data: List,
    process_fn,
    n_jobs: int = -1,
    backend: str = "threading"
) -> List:
    """
    Process data in parallel.
    
    Args:
        data: Input data
        process_fn: Function to process each item
        n_jobs: Number of parallel jobs (-1 for all CPUs)
        backend: Joblib backend (threading, multiprocessing, loky)
        
    Returns:
        Processed results
    """
    from joblib import Parallel, delayed
    
    results = Parallel(n_jobs=n_jobs, backend=backend)(
        delayed(process_fn)(item) for item in data
    )
    
    return results


# Memory optimization
def reduce_memory_usage(df: "pd.DataFrame") -> "pd.DataFrame":
    """
    Reduce memory usage of pandas DataFrame.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Optimized DataFrame
    """
    import pandas as pd
    
    start_mem = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
    
    end_mem = df.memory_usage().sum() / 1024**2
    logger.info(f"Memory usage reduced from {start_mem:.2f} MB to {end_mem:.2f} MB "
                f"({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)")
    
    return df


# Checkpoint management
class CheckpointManager:
    """Manage model checkpoints."""
    
    def __init__(self, checkpoint_dir: str = "checkpoints", max_checkpoints: int = 5):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.checkpoints = []
    
    def save(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False
    ):
        """Save checkpoint."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"checkpoint_epoch{epoch}_{timestamp}.pth"
        filepath = self.checkpoint_dir / filename
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "timestamp": timestamp,
        }
        
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        
        if metrics is not None:
            checkpoint["metrics"] = metrics
        
        torch.save(checkpoint, filepath)
        self.checkpoints.append(filepath)
        
        # Save best model separately
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"Best model saved to {best_path}")
        
        # Remove old checkpoints
        if len(self.checkpoints) > self.max_checkpoints:
            old_checkpoint = self.checkpoints.pop(0)
            old_checkpoint.unlink()
            logger.debug(f"Removed old checkpoint: {old_checkpoint}")
        
        logger.info(f"Checkpoint saved to {filepath}")
    
    def load_latest(self) -> Optional[Dict]:
        """Load latest checkpoint."""
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*.pth"))
        if checkpoints:
            return torch.load(checkpoints[-1])
        return None
    
    def load_best(self) -> Optional[Dict]:
        """Load best checkpoint."""
        best_path = self.checkpoint_dir / "best_model.pth"
        if best_path.exists():
            return torch.load(best_path)
        return None


if __name__ == "__main__":
    # Test utilities
    logger.info("Testing utilities...")
    
    # Test timer
    with timer("Test operation"):
        time.sleep(0.1)
    
    # Test cache
    cache = Cache()
    value = cache.get("test_key", lambda: "computed_value")
    print(f"Cached value: {value}")
    
    # Test performance monitor
    monitor = PerformanceMonitor()
    monitor.start()
    for _ in range(3):
        time.sleep(0.1)
        monitor.record()
    print(f"Performance summary: {monitor.get_summary()}")
    
    logger.info("Utilities test completed!")

#!/usr/bin/env python
"""Configuration management for Calabi-Yau ML."""

import os
import json
import yaml
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Data configuration."""
    
    n_samples: int = 5000
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    batch_size: int = 32
    num_workers: int = 4
    seed: int = 42
    normalize: bool = True
    cache_dir: str = "data/cache"
    
    # CICY specific
    cicy_file: str = "dataset/cicylist.txt"
    max_manifolds: Optional[int] = None
    
    def validate(self):
        """Validate configuration."""
        assert self.train_ratio + self.val_ratio + self.test_ratio == 1.0
        assert self.batch_size > 0
        assert self.num_workers >= 0


@dataclass
class ModelConfig:
    """Model configuration."""
    
    architecture: str = "mlp"  # mlp, resnet, ensemble, gnn
    input_dim: int = 10
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64, 32])
    output_dim: int = 1
    dropout: float = 0.2
    activation: str = "relu"
    batch_norm: bool = True
    
    # Task specific
    task_type: str = "regression"  # regression, classification
    n_classes: int = 3
    
    # Ensemble specific
    n_models: int = 5
    ensemble_strategy: str = "mean"  # mean, weighted, voting
    
    # MC Dropout
    mc_samples: int = 100
    
    def validate(self):
        """Validate configuration."""
        assert self.architecture in ["mlp", "resnet", "ensemble", "gnn"]
        assert self.task_type in ["regression", "classification"]
        assert 0 <= self.dropout < 1
        assert all(d > 0 for d in self.hidden_dims)


@dataclass
class TrainingConfig:
    """Training configuration."""
    
    epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    optimizer: str = "adam"  # adam, sgd, adamw
    scheduler: str = "reduce_on_plateau"  # reduce_on_plateau, cosine, step
    
    # Scheduler params
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5
    scheduler_min_lr: float = 1e-6
    
    # Early stopping
    early_stopping: bool = True
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 1e-4
    
    # Checkpointing
    checkpoint_dir: str = "models/checkpoints"
    save_best_only: bool = True
    save_frequency: int = 10
    
    # Logging
    log_frequency: int = 10
    tensorboard: bool = False
    wandb: bool = False
    
    def validate(self):
        """Validate configuration."""
        assert self.epochs > 0
        assert self.learning_rate > 0
        assert self.optimizer in ["adam", "sgd", "adamw"]


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    
    name: str = "calabi_yau_experiment"
    description: str = ""
    version: str = "1.0.0"
    
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Paths
    output_dir: str = "results"
    log_dir: str = "logs"
    
    # Reproducibility
    deterministic: bool = True
    seed: int = 42
    
    # Device
    device: str = "auto"  # auto, cpu, cuda, mps
    mixed_precision: bool = False
    
    def validate(self):
        """Validate all configurations."""
        self.data.validate()
        self.model.validate()
        self.training.validate()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def save(self, path: str):
        """Save configuration to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if path.suffix == ".json":
            with open(path, "w") as f:
                json.dump(self.to_dict(), f, indent=2)
        elif path.suffix in [".yaml", ".yml"]:
            with open(path, "w") as f:
                yaml.dump(self.to_dict(), f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        logger.info(f"Configuration saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> "ExperimentConfig":
        """Load configuration from file."""
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        if path.suffix == ".json":
            with open(path, "r") as f:
                data = json.load(f)
        elif path.suffix in [".yaml", ".yml"]:
            with open(path, "r") as f:
                data = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        # Create nested dataclasses
        if "data" in data:
            data["data"] = DataConfig(**data["data"])
        if "model" in data:
            data["model"] = ModelConfig(**data["model"])
        if "training" in data:
            data["training"] = TrainingConfig(**data["training"])
        
        config = cls(**data)
        config.validate()
        
        logger.info(f"Configuration loaded from {path}")
        return config
    
    @classmethod
    def from_args(cls, args) -> "ExperimentConfig":
        """Create configuration from command-line arguments."""
        config = cls()
        
        # Update from arguments
        if hasattr(args, "config") and args.config:
            config = cls.load(args.config)
        
        # Override with command-line arguments
        for key, value in vars(args).items():
            if value is not None:
                # Handle nested attributes
                if "." in key:
                    parts = key.split(".")
                    obj = config
                    for part in parts[:-1]:
                        obj = getattr(obj, part)
                    setattr(obj, parts[-1], value)
                elif hasattr(config, key):
                    setattr(config, key, value)
        
        config.validate()
        return config


def get_device(device: str = "auto") -> str:
    """Get the appropriate device for computation."""
    import torch
    
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    return device


def set_seed(seed: int, deterministic: bool = True):
    """Set random seeds for reproducibility."""
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Set environment variable for hash reproducibility
        os.environ["PYTHONHASHSEED"] = str(seed)
    
    logger.info(f"Random seed set to {seed} (deterministic={deterministic})")


# Default configurations for different scenarios
DEFAULT_CONFIGS = {
    "quick": ExperimentConfig(
        name="quick_test",
        data=DataConfig(n_samples=1000, batch_size=64),
        model=ModelConfig(hidden_dims=[64, 32]),
        training=TrainingConfig(epochs=20)
    ),
    "standard": ExperimentConfig(
        name="standard_run",
        data=DataConfig(n_samples=5000),
        model=ModelConfig(hidden_dims=[128, 64, 32]),
        training=TrainingConfig(epochs=100)
    ),
    "production": ExperimentConfig(
        name="production_run",
        data=DataConfig(n_samples=10000, batch_size=128),
        model=ModelConfig(
            architecture="ensemble",
            hidden_dims=[256, 128, 64],
            n_models=5
        ),
        training=TrainingConfig(
            epochs=200,
            learning_rate=0.0005,
            tensorboard=True
        )
    ),
    "cicy": ExperimentConfig(
        name="cicy_real_data",
        data=DataConfig(
            cicy_file="dataset/cicylist.txt",
            batch_size=64
        ),
        model=ModelConfig(
            input_dim=19,
            hidden_dims=[128, 64, 32],
            mc_samples=100
        ),
        training=TrainingConfig(epochs=50)
    )
}


if __name__ == "__main__":
    # Test configuration
    config = ExperimentConfig()
    print("Default configuration:")
    print(json.dumps(config.to_dict(), indent=2))
    
    # Save and load test
    config.save("test_config.yaml")
    loaded_config = ExperimentConfig.load("test_config.yaml")
    print("\nLoaded configuration matches:", config == loaded_config)
    
    # Clean up
    Path("test_config.yaml").unlink(missing_ok=True)

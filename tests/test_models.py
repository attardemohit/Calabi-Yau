#!/usr/bin/env python
"""Tests for neural network models."""

import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models import MLPRegressor, DeepResNet, EnsembleModel


class TestMLPRegressor:
    """Test MLPRegressor model."""
    
    @pytest.fixture
    def model(self):
        """Create a test model."""
        return MLPRegressor(
            input_dim=10,
            hidden_dims=[64, 32],
            output_dim=1,
            dropout=0.2
        )
    
    def test_initialization(self, model):
        """Test model initialization."""
        assert model is not None
        assert isinstance(model, nn.Module)
        assert model.input_dim == 10
        assert model.output_dim == 1
    
    def test_forward_pass(self, model):
        """Test forward pass."""
        batch_size = 32
        x = torch.randn(batch_size, 10)
        
        output = model(x)
        
        assert output.shape == (batch_size, 1)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_different_batch_sizes(self, model):
        """Test with different batch sizes."""
        for batch_size in [1, 16, 32, 128]:
            x = torch.randn(batch_size, 10)
            output = model(x)
            assert output.shape == (batch_size, 1)
    
    def test_gradient_flow(self, model):
        """Test gradient flow through the model."""
        x = torch.randn(32, 10, requires_grad=True)
        target = torch.randn(32, 1)
        
        output = model(x)
        loss = nn.MSELoss()(output, target)
        loss.backward()
        
        # Check that gradients exist
        for param in model.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()
    
    def test_dropout_behavior(self, model):
        """Test dropout behavior in train vs eval mode."""
        x = torch.randn(100, 10)
        
        # Training mode (dropout active)
        model.train()
        outputs_train = [model(x) for _ in range(10)]
        
        # Check that outputs vary due to dropout
        std_train = torch.stack(outputs_train).std(dim=0).mean()
        assert std_train > 0
        
        # Eval mode (dropout inactive)
        model.eval()
        outputs_eval = [model(x) for _ in range(10)]
        
        # Check that outputs are consistent
        std_eval = torch.stack(outputs_eval).std(dim=0).mean()
        assert std_eval < 1e-6
    
    @pytest.mark.parametrize("hidden_dims", [
        [32],
        [64, 32],
        [128, 64, 32],
        [256, 128, 64, 32]
    ])
    def test_different_architectures(self, hidden_dims):
        """Test different architecture configurations."""
        model = MLPRegressor(
            input_dim=10,
            hidden_dims=hidden_dims,
            output_dim=1
        )
        
        x = torch.randn(16, 10)
        output = model(x)
        assert output.shape == (16, 1)


class TestDeepResNet:
    """Test DeepResNet model."""
    
    @pytest.fixture
    def model(self):
        """Create a test model."""
        return DeepResNet(
            input_dim=10,
            hidden_dim=64,
            output_dim=1,
            num_blocks=2
        )
    
    def test_initialization(self, model):
        """Test model initialization."""
        assert model is not None
        assert isinstance(model, nn.Module)
        assert len(model.res_blocks) == 2
    
    def test_forward_pass(self, model):
        """Test forward pass."""
        x = torch.randn(32, 10)
        output = model(x)
        
        assert output.shape == (32, 1)
        assert not torch.isnan(output).any()
    
    def test_residual_connections(self, model):
        """Test that residual connections work."""
        x = torch.randn(16, 10)
        
        # Get intermediate outputs
        model.eval()
        h = model.input_proj(x)
        
        for block in model.res_blocks:
            h_before = h.clone()
            h = block(h)
            
            # Check that output is different from input (transformation applied)
            assert not torch.allclose(h, h_before)
    
    @pytest.mark.parametrize("num_blocks", [1, 2, 4, 8])
    def test_different_depths(self, num_blocks):
        """Test different network depths."""
        model = DeepResNet(
            input_dim=10,
            hidden_dim=32,
            output_dim=1,
            num_blocks=num_blocks
        )
        
        x = torch.randn(8, 10)
        output = model(x)
        assert output.shape == (8, 1)


class TestEnsembleModel:
    """Test EnsembleModel."""
    
    @pytest.fixture
    def ensemble(self):
        """Create a test ensemble."""
        return EnsembleModel(
            input_dim=10,
            output_dim=1,
            n_models=3,
            task_type="regression"
        )
    
    def test_initialization(self, ensemble):
        """Test ensemble initialization."""
        assert ensemble is not None
        assert len(ensemble.models) == 3
        assert ensemble.task_type == "regression"
    
    def test_forward_pass(self, ensemble):
        """Test forward pass."""
        x = torch.randn(32, 10)
        output = ensemble(x)
        
        assert output.shape == (32, 1)
        assert not torch.isnan(output).any()
    
    def test_ensemble_averaging(self, ensemble):
        """Test that ensemble averages predictions."""
        x = torch.randn(16, 10)
        
        # Get individual model predictions
        individual_preds = []
        for model in ensemble.models:
            model.eval()
            pred = model(x)
            individual_preds.append(pred)
        
        # Get ensemble prediction
        ensemble.eval()
        ensemble_pred = ensemble(x)
        
        # Check that ensemble is average of individuals
        expected = torch.stack(individual_preds).mean(dim=0)
        assert torch.allclose(ensemble_pred, expected, atol=1e-5)
    
    def test_classification_ensemble(self):
        """Test ensemble for classification."""
        ensemble = EnsembleModel(
            input_dim=10,
            output_dim=3,
            n_models=3,
            task_type="classification"
        )
        
        x = torch.randn(32, 10)
        output = ensemble(x)
        
        assert output.shape == (32, 3)
        # Check that outputs are probabilities
        assert torch.allclose(output.sum(dim=1), torch.ones(32), atol=1e-5)
        assert (output >= 0).all() and (output <= 1).all()


class TestModelSaveLoad:
    """Test model saving and loading."""
    
    def test_save_load_mlp(self, tmp_path):
        """Test saving and loading MLP model."""
        # Create and save model
        model = MLPRegressor(10, [64, 32], 1)
        save_path = tmp_path / "model.pth"
        torch.save(model.state_dict(), save_path)
        
        # Load model
        loaded_model = MLPRegressor(10, [64, 32], 1)
        loaded_model.load_state_dict(torch.load(save_path))
        
        # Check that predictions match
        x = torch.randn(16, 10)
        model.eval()
        loaded_model.eval()
        
        pred1 = model(x)
        pred2 = loaded_model(x)
        
        assert torch.allclose(pred1, pred2)
    
    def test_checkpoint_format(self, tmp_path):
        """Test checkpoint format with metadata."""
        model = MLPRegressor(10, [64, 32], 1)
        optimizer = torch.optim.Adam(model.parameters())
        
        # Create checkpoint
        checkpoint = {
            "epoch": 10,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": 0.123,
            "config": {"input_dim": 10, "hidden_dims": [64, 32]}
        }
        
        save_path = tmp_path / "checkpoint.pth"
        torch.save(checkpoint, save_path)
        
        # Load checkpoint
        loaded = torch.load(save_path)
        
        assert loaded["epoch"] == 10
        assert loaded["loss"] == 0.123
        assert "model_state_dict" in loaded
        assert "optimizer_state_dict" in loaded


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
))])
class TestModelDevice:
    """Test models on different devices."""
    
    def test_mlp_device(self, device):
        """Test MLP on device."""
        model = MLPRegressor(10, [64, 32], 1).to(device)
        x = torch.randn(32, 10).to(device)
        
        output = model(x)
        assert output.device.type == device
        assert output.shape == (32, 1)
    
    def test_resnet_device(self, device):
        """Test ResNet on device."""
        model = DeepResNet(10, 64, 1, 2).to(device)
        x = torch.randn(32, 10).to(device)
        
        output = model(x)
        assert output.device.type == device
        assert output.shape == (32, 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

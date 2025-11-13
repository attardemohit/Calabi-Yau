#!/usr/bin/env python
"""Tests for data generation and loading."""

import pytest
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_generator import CalabiYauDataGenerator
from cicy_loader import CICYDataLoader


class TestCalabiYauDataGenerator:
    """Test synthetic data generator."""
    
    @pytest.fixture
    def generator(self):
        """Create a test generator."""
        return CalabiYauDataGenerator(seed=42)
    
    def test_initialization(self, generator):
        """Test generator initialization."""
        assert generator is not None
        assert generator.seed == 42
    
    def test_generate_features(self, generator):
        """Test feature generation."""
        features = generator.generate_features()
        
        assert isinstance(features, dict)
        assert "h11" in features
        assert "h21" in features
        assert "euler_char" in features
        assert "intersection_numbers" in features
        
        # Check Euler characteristic formula
        expected_euler = 2 * (features["h11"] - features["h21"])
        assert features["euler_char"] == expected_euler
    
    def test_generate_spectrum(self, generator):
        """Test spectrum generation."""
        features = generator.generate_features()
        spectrum = generator.generate_spectrum(features)
        
        assert isinstance(spectrum, np.ndarray)
        assert len(spectrum) == 10
        assert spectrum.min() >= 0
        assert spectrum.max() <= 100
        assert not np.isnan(spectrum).any()
    
    def test_generate_dataset(self, generator):
        """Test dataset generation."""
        n_samples = 100
        df = generator.generate_dataset(n_samples)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == n_samples
        assert "h11" in df.columns
        assert "h21" in df.columns
        assert "spectrum" in df.columns
        
        # Check data types
        assert df["h11"].dtype == np.int64
        assert df["h21"].dtype == np.int64
        assert df["spectrum"].dtype == np.float64
    
    def test_reproducibility(self):
        """Test that same seed gives same data."""
        gen1 = CalabiYauDataGenerator(seed=123)
        gen2 = CalabiYauDataGenerator(seed=123)
        
        df1 = gen1.generate_dataset(50)
        df2 = gen2.generate_dataset(50)
        
        pd.testing.assert_frame_equal(df1, df2)
    
    def test_different_seeds(self):
        """Test that different seeds give different data."""
        gen1 = CalabiYauDataGenerator(seed=1)
        gen2 = CalabiYauDataGenerator(seed=2)
        
        df1 = gen1.generate_dataset(50)
        df2 = gen2.generate_dataset(50)
        
        assert not df1.equals(df2)
    
    @pytest.mark.parametrize("task_type", ["regression", "classification"])
    def test_save_splits(self, generator, tmp_path, task_type):
        """Test saving train/val/test splits."""
        df = generator.generate_dataset(100)
        generator.save_splits(df, tmp_path, task_type=task_type)
        
        # Check that files exist
        assert (tmp_path / f"calabi_yau_train_{task_type}.json").exists()
        assert (tmp_path / f"calabi_yau_val_{task_type}.json").exists()
        assert (tmp_path / f"calabi_yau_test_{task_type}.json").exists()
        
        # Load and check splits
        train_df = pd.read_json(tmp_path / f"calabi_yau_train_{task_type}.json")
        val_df = pd.read_json(tmp_path / f"calabi_yau_val_{task_type}.json")
        test_df = pd.read_json(tmp_path / f"calabi_yau_test_{task_type}.json")
        
        # Check sizes (70/15/15 split)
        assert len(train_df) == 70
        assert len(val_df) == 15
        assert len(test_df) == 15
        
        # Check no overlap
        train_ids = set(train_df.index)
        val_ids = set(val_df.index)
        test_ids = set(test_df.index)
        
        assert len(train_ids & val_ids) == 0
        assert len(train_ids & test_ids) == 0
        assert len(val_ids & test_ids) == 0


class TestCICYDataLoader:
    """Test CICY data loader."""
    
    @pytest.fixture
    def sample_data(self, tmp_path):
        """Create sample CICY data file."""
        data_file = tmp_path / "test_cicy.txt"
        content = """Num: 1
NumPs: 3
NumPol: 4
Eta: 0
H11: 15
H21: 15
C2: {1, 2, 3}
Redun: {0}
{1 1 0 0}
{0 1 1 0}
{1 0 1 1}
{0 0 0 2}

Num: 2
NumPs: 2
NumPol: 3
Eta: 1
H11: 10
H21: 20
C2: {2, 3}
Redun: {1}
{2 1 0}
{1 2 1}
{0 1 2}

"""
        data_file.write_text(content)
        return str(data_file)
    
    def test_initialization(self, sample_data):
        """Test loader initialization."""
        loader = CICYDataLoader(sample_data)
        assert loader is not None
        assert len(loader.manifolds) == 2
    
    def test_parse_manifold(self, sample_data):
        """Test manifold parsing."""
        loader = CICYDataLoader(sample_data)
        
        # Check first manifold
        m1 = loader.manifolds[0]
        assert m1["id"] == 1
        assert m1["h11"] == 15
        assert m1["h21"] == 15
        assert m1["euler_char"] == 0  # 2*(15-15)
        assert m1["config_matrix"].shape == (4, 4)
        
        # Check second manifold
        m2 = loader.manifolds[1]
        assert m2["id"] == 2
        assert m2["h11"] == 10
        assert m2["h21"] == 20
        assert m2["euler_char"] == -20  # 2*(10-20)
        assert m2["config_matrix"].shape == (3, 3)
    
    def test_to_dataframe(self, sample_data):
        """Test conversion to DataFrame."""
        loader = CICYDataLoader(sample_data)
        df = loader.to_dataframe()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "h11" in df.columns
        assert "h21" in df.columns
        assert "euler_char" in df.columns
        
        # Check derived features
        assert "matrix_rank" in df.columns
        assert "matrix_norm" in df.columns
        assert "c2_mean" in df.columns
    
    def test_filter_manifolds(self, sample_data):
        """Test filtering manifolds."""
        loader = CICYDataLoader(sample_data)
        
        # Filter by h11
        filtered = loader.filter_manifolds(min_h11=12)
        assert len(filtered) == 1
        assert filtered[0]["h11"] == 15
        
        # Filter by h21
        filtered = loader.filter_manifolds(max_h21=18)
        assert len(filtered) == 1
        assert filtered[0]["h21"] == 15
    
    def test_create_ml_dataset(self, sample_data):
        """Test ML dataset creation."""
        loader = CICYDataLoader(sample_data)
        
        # Test h11 prediction
        X, y, features = loader.create_ml_dataset(target="h11_prediction")
        assert X.shape[0] == 2
        assert y.shape == (2,)
        assert len(features) > 0
        
        # Test mirror symmetry classification
        X, y, features = loader.create_ml_dataset(target="mirror_symmetry")
        assert X.shape[0] == 2
        assert y.shape == (2,)
        assert set(y) <= {0, 1}


class TestDataAugmentation:
    """Test data augmentation techniques."""
    
    def test_noise_augmentation(self):
        """Test adding noise to features."""
        data = np.array([[1, 2, 3], [4, 5, 6]])
        noise_level = 0.1
        
        # Add Gaussian noise
        augmented = data + np.random.normal(0, noise_level, data.shape)
        
        # Check that data changed but not too much
        assert not np.allclose(data, augmented)
        assert np.abs(data - augmented).max() < 3 * noise_level  # 3-sigma
    
    def test_feature_scaling(self):
        """Test feature scaling."""
        from sklearn.preprocessing import StandardScaler
        
        data = np.random.randn(100, 10) * 10 + 5
        
        scaler = StandardScaler()
        scaled = scaler.fit_transform(data)
        
        # Check that scaling worked
        assert np.abs(scaled.mean()) < 1e-10
        assert np.abs(scaled.std() - 1) < 1e-10


class TestDataValidation:
    """Test data validation utilities."""
    
    def test_check_nans(self):
        """Test NaN detection."""
        # Good data
        data = np.random.randn(100, 10)
        assert not np.isnan(data).any()
        
        # Data with NaN
        data[5, 3] = np.nan
        assert np.isnan(data).any()
        
        # Find NaN location
        nan_mask = np.isnan(data)
        assert nan_mask[5, 3]
    
    def test_check_infinities(self):
        """Test infinity detection."""
        # Good data
        data = np.random.randn(100, 10)
        assert not np.isinf(data).any()
        
        # Data with infinity
        data[7, 2] = np.inf
        data[9, 5] = -np.inf
        assert np.isinf(data).any()
        
        # Find infinity locations
        inf_mask = np.isinf(data)
        assert inf_mask[7, 2]
        assert inf_mask[9, 5]
    
    def test_check_range(self):
        """Test range validation."""
        data = np.random.uniform(0, 100, (50, 5))
        
        # Check valid range
        assert (data >= 0).all()
        assert (data <= 100).all()
        
        # Check invalid values
        data[10, 2] = -5
        data[20, 4] = 150
        
        assert (data < 0).any()
        assert (data > 100).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

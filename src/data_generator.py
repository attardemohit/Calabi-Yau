"""
Synthetic Data Generator for Calabi-Yau Geometry to Particle Spectra
Generates synthetic Calabi-Yau manifold features and corresponding particle observables
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any
import json
from pathlib import Path


class CalabiYauDataGenerator:
    """Generate synthetic Calabi-Yau geometry data with corresponding particle spectra."""
    
    def __init__(self, seed: Optional[int] = 42):
        """
        Initialize the data generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
    
    def generate_hodge_numbers(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate realistic Hodge numbers h^{1,1} and h^{2,1}.
        
        Based on known constraints from the Kreuzer-Skarke database:
        - h^{1,1} typically ranges from 1 to ~500
        - h^{2,1} typically ranges from 1 to ~500
        - Euler characteristic χ = 2(h^{1,1} - h^{2,1})
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Tuple of (h11, h21) arrays
        """
        # Generate h^{1,1} with a log-normal distribution (more small values)
        h11 = np.random.lognormal(mean=2.5, sigma=1.0, size=n_samples)
        h11 = np.clip(h11, 1, 500).astype(int)
        
        # Generate h^{2,1} with similar distribution
        h21 = np.random.lognormal(mean=2.5, sigma=1.0, size=n_samples)
        h21 = np.clip(h21, 1, 500).astype(int)
        
        return h11, h21
    
    def compute_euler_characteristic(self, h11: np.ndarray, h21: np.ndarray) -> np.ndarray:
        """
        Compute Euler characteristic from Hodge numbers.
        
        χ = 2(h^{1,1} - h^{2,1})
        """
        return 2 * (h11 - h21)
    
    def generate_intersection_numbers(self, h11: np.ndarray, n_samples: int) -> np.ndarray:
        """
        Generate simplified intersection numbers.
        
        In reality, these form a tensor, but we'll use summary statistics.
        """
        # Generate average intersection number based on h^{1,1}
        avg_intersection = np.random.gamma(shape=2, scale=h11/10, size=n_samples)
        return avg_intersection
    
    def generate_particle_spectrum(self, 
                                  h11: np.ndarray, 
                                  h21: np.ndarray,
                                  chi: np.ndarray,
                                  intersection: np.ndarray,
                                  spectrum_type: str = 'regression') -> np.ndarray:
        """
        Generate synthetic particle spectrum based on geometry features.
        
        This is a simplified model that creates plausible relationships between
        geometry and physics observables.
        
        Args:
            h11: Hodge number h^{1,1}
            h21: Hodge number h^{2,1}
            chi: Euler characteristic
            intersection: Average intersection numbers
            spectrum_type: 'regression' for continuous, 'classification' for discrete
            
        Returns:
            Array of particle spectrum values
        """
        if spectrum_type == 'regression':
            # Synthetic formula for number of particle generations or energy scale
            # Inspired by phenomenological observations
            base_spectrum = (
                0.5 * np.abs(h11 - h21) +  # Difference in Hodge numbers
                0.1 * np.sqrt(h11 * h21) +  # Geometric mean contribution
                0.05 * np.abs(chi) +  # Euler characteristic contribution
                0.2 * intersection  # Intersection number contribution
            )
            
            # Normalize to reasonable range [0, 100]
            base_spectrum = (base_spectrum - base_spectrum.min()) / (base_spectrum.max() - base_spectrum.min() + 1e-8)
            base_spectrum = base_spectrum * 100
            
            # Add realistic noise
            noise = np.random.normal(0, 5, size=len(base_spectrum))  # Smaller noise
            spectrum = base_spectrum + noise
            
            # Ensure positive values (e.g., for energy scales)
            spectrum = np.maximum(spectrum, 0.1)
            
        elif spectrum_type == 'classification':
            # Generate discrete classes (e.g., gauge group types)
            # Use a nonlinear combination to determine class
            score = (
                np.tanh((h11 - h21) / 50) +
                0.5 * np.sin(h11 / 20) +
                0.3 * np.cos(h21 / 30)
            )
            
            # Map to discrete classes (e.g., 3 types of gauge groups)
            percentiles = np.percentile(score, [33, 67])
            spectrum = np.zeros(len(score), dtype=int)
            spectrum[score > percentiles[0]] = 1
            spectrum[score > percentiles[1]] = 2
            
        else:
            raise ValueError(f"Unknown spectrum_type: {spectrum_type}")
        
        return spectrum
    
    def generate_dataset(self, 
                        n_samples: int = 1000,
                        spectrum_type: str = 'regression',
                        include_advanced_features: bool = False) -> pd.DataFrame:
        """
        Generate complete synthetic dataset.
        
        Args:
            n_samples: Number of samples to generate
            spectrum_type: 'regression' or 'classification'
            include_advanced_features: Whether to include additional geometric features
            
        Returns:
            DataFrame with geometry features and particle spectrum
        """
        # Generate basic Hodge numbers
        h11, h21 = self.generate_hodge_numbers(n_samples)
        
        # Compute derived quantities
        chi = self.compute_euler_characteristic(h11, h21)
        intersection = self.generate_intersection_numbers(h11, n_samples)
        
        # Generate particle spectrum
        spectrum = self.generate_particle_spectrum(
            h11, h21, chi, intersection, spectrum_type
        )
        
        # Create DataFrame
        data = {
            'h11': h11,
            'h21': h21,
            'euler_char': chi,
            'avg_intersection': intersection,
            'particle_spectrum': spectrum
        }
        
        if include_advanced_features:
            # Add more sophisticated features
            data['h11_h21_ratio'] = h11 / (h21 + 1e-6)
            data['hodge_product'] = h11 * h21
            data['hodge_sum'] = h11 + h21
            data['normalized_chi'] = chi / (h11 + h21 + 1e-6)
            
            # Add synthetic "topological invariants"
            data['invariant_1'] = np.sin(h11/10) * np.cos(h21/10)
            data['invariant_2'] = np.exp(-np.abs(chi)/100)
        
        df = pd.DataFrame(data)
        
        # Add metadata
        df.attrs['spectrum_type'] = spectrum_type
        df.attrs['n_features'] = len(df.columns) - 1  # Exclude target
        
        return df
    
    def split_dataset(self, 
                      df: pd.DataFrame,
                      test_size: float = 0.2,
                      val_size: float = 0.1) -> Dict[str, pd.DataFrame]:
        """
        Split dataset into train, validation, and test sets.
        
        Args:
            df: Complete dataset
            test_size: Fraction for test set
            val_size: Fraction for validation set
            
        Returns:
            Dictionary with 'train', 'val', 'test' DataFrames
        """
        n = len(df)
        n_test = int(n * test_size)
        n_val = int(n * val_size)
        n_train = n - n_test - n_val
        
        # Shuffle the data
        df_shuffled = df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        
        # Split
        splits = {
            'train': df_shuffled[:n_train],
            'val': df_shuffled[n_train:n_train + n_val],
            'test': df_shuffled[n_train + n_val:]
        }
        
        return splits
    
    def save_dataset(self, df: pd.DataFrame, filepath: str):
        """Save dataset to CSV file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath, index=False)
        
        # Save metadata
        metadata = {
            'spectrum_type': df.attrs.get('spectrum_type', 'unknown'),
            'n_features': df.attrs.get('n_features', len(df.columns) - 1),
            'n_samples': len(df),
            'columns': list(df.columns)
        }
        
        metadata_path = Path(filepath).with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    @staticmethod
    def load_dataset(filepath: str) -> pd.DataFrame:
        """Load dataset from CSV file."""
        df = pd.read_csv(filepath)
        
        # Load metadata if available
        metadata_path = Path(filepath).with_suffix('.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                for key, value in metadata.items():
                    df.attrs[key] = value
        
        return df


def generate_and_save_datasets(output_dir: str = 'data',
                               n_samples: int = 5000,
                               seed: int = 42):
    """
    Generate and save both regression and classification datasets.
    
    Args:
        output_dir: Directory to save datasets
        n_samples: Number of samples to generate
        seed: Random seed
    """
    generator = CalabiYauDataGenerator(seed=seed)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Generate regression dataset
    print("Generating regression dataset...")
    df_reg = generator.generate_dataset(
        n_samples=n_samples,
        spectrum_type='regression',
        include_advanced_features=True
    )
    
    splits_reg = generator.split_dataset(df_reg)
    for split_name, split_df in splits_reg.items():
        filepath = output_path / f'calabi_yau_{split_name}_regression.csv'
        generator.save_dataset(split_df, str(filepath))
        print(f"  Saved {split_name} set: {len(split_df)} samples to {filepath}")
    
    # Generate classification dataset
    print("\nGenerating classification dataset...")
    df_class = generator.generate_dataset(
        n_samples=n_samples,
        spectrum_type='classification',
        include_advanced_features=True
    )
    
    splits_class = generator.split_dataset(df_class)
    for split_name, split_df in splits_class.items():
        filepath = output_path / f'calabi_yau_{split_name}_classification.csv'
        generator.save_dataset(split_df, str(filepath))
        print(f"  Saved {split_name} set: {len(split_df)} samples to {filepath}")
    
    print("\nDataset generation complete!")
    
    # Print summary statistics
    print("\nDataset Summary:")
    print(f"Total samples: {n_samples}")
    print(f"Features: {list(df_reg.columns[:-1])}")
    print(f"Regression target range: [{df_reg['particle_spectrum'].min():.2f}, {df_reg['particle_spectrum'].max():.2f}]")
    print(f"Classification classes: {df_class['particle_spectrum'].nunique()}")


if __name__ == "__main__":
    # Generate datasets when run as script
    generate_and_save_datasets(n_samples=5000)

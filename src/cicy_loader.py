#!/usr/bin/env python
"""
CICY Data Loader for Real Calabi-Yau Manifold Data

This module parses the CICY (Complete Intersection Calabi-Yau) dataset
to extract real geometric features for machine learning.

Data format:
- Num: Manifold ID
- NumPs: Number of projective spaces
- NumPol: Number of polynomials
- Eta: Eta invariant
- H11, H21: Hodge numbers
- C2: Second Chern class values
- Configuration matrix: Defines the complete intersection
"""

import numpy as np
import pandas as pd
import re
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json


class CICYDataLoader:
    """Load and parse CICY manifold data from text files."""
    
    def __init__(self, data_path: str = "dataset/cicylist.txt"):
        """
        Initialize CICY data loader.
        
        Args:
            data_path: Path to CICY data file
        """
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"CICY data file not found: {data_path}")
        
        self.manifolds = []
        self._parse_file()
    
    def _parse_file(self):
        """Parse the CICY data file."""
        with open(self.data_path, 'r') as f:
            content = f.read()
        
        # Split into individual manifold entries
        entries = content.strip().split('\n\n')
        
        for entry in entries:
            if not entry.strip():
                continue
            
            manifold = self._parse_manifold(entry)
            if manifold:
                self.manifolds.append(manifold)
    
    def _parse_manifold(self, entry: str) -> Optional[Dict]:
        """
        Parse a single manifold entry.
        
        Args:
            entry: Text block for one manifold
            
        Returns:
            Dictionary with manifold properties
        """
        lines = entry.strip().split('\n')
        manifold = {}
        config_matrix = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Parse key-value pairs
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                
                if key == 'Num':
                    manifold['id'] = int(value)
                elif key == 'NumPs':
                    manifold['num_ps'] = int(value)
                elif key == 'NumPol':
                    manifold['num_pol'] = int(value)
                elif key == 'Eta':
                    manifold['eta'] = int(value)
                elif key == 'H11':
                    manifold['h11'] = int(value)
                elif key == 'H21':
                    manifold['h21'] = int(value)
                elif key == 'C2':
                    # Parse Chern class values
                    c2_values = re.findall(r'\d+', value)
                    manifold['c2'] = [int(v) for v in c2_values]
                elif key == 'Redun':
                    # Parse redundancy vector
                    redun_values = re.findall(r'\d+', value)
                    manifold['redundancy'] = [int(v) for v in redun_values]
            
            # Parse configuration matrix rows
            elif line.startswith('{') and line.endswith('}'):
                row = re.findall(r'-?\d+', line)
                config_matrix.append([int(v) for v in row])
        
        if config_matrix:
            manifold['config_matrix'] = np.array(config_matrix)
        
        # Calculate derived features
        if 'h11' in manifold and 'h21' in manifold:
            manifold['euler_char'] = 2 * (manifold['h11'] - manifold['h21'])
            manifold['h11_h21_ratio'] = manifold['h11'] / (manifold['h21'] + 1e-8)
            manifold['hodge_sum'] = manifold['h11'] + manifold['h21']
            manifold['hodge_product'] = manifold['h11'] * manifold['h21']
            manifold['hodge_diff'] = abs(manifold['h11'] - manifold['h21'])
        
        # Calculate configuration matrix statistics
        if 'config_matrix' in manifold:
            matrix = manifold['config_matrix']
            manifold['matrix_rank'] = np.linalg.matrix_rank(matrix)
            manifold['matrix_norm'] = np.linalg.norm(matrix)
            manifold['matrix_trace'] = np.trace(matrix) if matrix.shape[0] == matrix.shape[1] else 0
            manifold['matrix_det'] = np.linalg.det(matrix) if matrix.shape[0] == matrix.shape[1] else 0
            manifold['matrix_mean'] = np.mean(matrix)
            manifold['matrix_std'] = np.std(matrix)
            manifold['matrix_max'] = np.max(matrix)
            manifold['matrix_min'] = np.min(matrix)
            manifold['matrix_nonzero'] = np.count_nonzero(matrix)
        
        # Calculate Chern class statistics
        if 'c2' in manifold:
            manifold['c2_mean'] = np.mean(manifold['c2'])
            manifold['c2_std'] = np.std(manifold['c2'])
            manifold['c2_sum'] = np.sum(manifold['c2'])
            manifold['c2_max'] = np.max(manifold['c2'])
            manifold['c2_min'] = np.min(manifold['c2'])
        
        return manifold if 'id' in manifold else None
    
    def to_dataframe(self, 
                     include_matrix_features: bool = True,
                     include_chern_features: bool = True) -> pd.DataFrame:
        """
        Convert manifold data to pandas DataFrame.
        
        Args:
            include_matrix_features: Include configuration matrix statistics
            include_chern_features: Include Chern class statistics
            
        Returns:
            DataFrame with manifold features
        """
        rows = []
        
        for manifold in self.manifolds:
            row = {
                'id': manifold.get('id'),
                'num_ps': manifold.get('num_ps'),
                'num_pol': manifold.get('num_pol'),
                'eta': manifold.get('eta'),
                'h11': manifold.get('h11'),
                'h21': manifold.get('h21'),
                'euler_char': manifold.get('euler_char'),
                'h11_h21_ratio': manifold.get('h11_h21_ratio'),
                'hodge_sum': manifold.get('hodge_sum'),
                'hodge_product': manifold.get('hodge_product'),
                'hodge_diff': manifold.get('hodge_diff')
            }
            
            if include_matrix_features:
                row.update({
                    'matrix_rank': manifold.get('matrix_rank'),
                    'matrix_norm': manifold.get('matrix_norm'),
                    'matrix_trace': manifold.get('matrix_trace'),
                    'matrix_det': manifold.get('matrix_det'),
                    'matrix_mean': manifold.get('matrix_mean'),
                    'matrix_std': manifold.get('matrix_std'),
                    'matrix_max': manifold.get('matrix_max'),
                    'matrix_min': manifold.get('matrix_min'),
                    'matrix_nonzero': manifold.get('matrix_nonzero')
                })
            
            if include_chern_features:
                row.update({
                    'c2_mean': manifold.get('c2_mean'),
                    'c2_std': manifold.get('c2_std'),
                    'c2_sum': manifold.get('c2_sum'),
                    'c2_max': manifold.get('c2_max'),
                    'c2_min': manifold.get('c2_min')
                })
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Remove any columns with all NaN values
        df = df.dropna(axis=1, how='all')
        
        # Fill remaining NaN values with 0
        df = df.fillna(0)
        
        return df
    
    def generate_ml_dataset(self,
                           target_type: str = 'clustering',
                           test_size: float = 0.2,
                           val_size: float = 0.1,
                           random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Generate ML-ready dataset with train/val/test splits.
        
        Args:
            target_type: Type of target to generate
                - 'clustering': No target, for unsupervised learning
                - 'h11_prediction': Predict h11 from other features
                - 'h21_prediction': Predict h21 from other features
                - 'mirror_symmetry': Binary classification of mirror pairs
                - 'topology_class': Multi-class based on Hodge numbers
            test_size: Fraction for test set
            val_size: Fraction for validation set
            random_state: Random seed
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        from sklearn.model_selection import train_test_split
        
        # Get base dataframe
        df = self.to_dataframe()
        
        # Generate target based on type
        if target_type == 'clustering':
            # No target for unsupervised learning
            pass
        
        elif target_type == 'h11_prediction':
            # Predict h11 from other features
            df['target'] = df['h11']
            # Remove h11 and related features from inputs
            df = df.drop(['h11', 'h11_h21_ratio', 'hodge_sum', 'hodge_product', 'hodge_diff'], axis=1)
        
        elif target_type == 'h21_prediction':
            # Predict h21 from other features
            df['target'] = df['h21']
            # Remove h21 and related features from inputs
            df = df.drop(['h21', 'h11_h21_ratio', 'hodge_sum', 'hodge_product', 'hodge_diff'], axis=1)
        
        elif target_type == 'mirror_symmetry':
            # Binary classification: is this a potential mirror manifold?
            # Mirror symmetry swaps h11 and h21
            df['target'] = (df['h11'] == df['h21']).astype(int)
        
        elif target_type == 'topology_class':
            # Multi-class classification based on Hodge number patterns
            def classify_topology(row):
                if row['h11'] < 10 and row['h21'] < 10:
                    return 0  # Small Hodge numbers
                elif row['h11'] > 20 or row['h21'] > 20:
                    return 2  # Large Hodge numbers
                else:
                    return 1  # Medium Hodge numbers
            
            df['target'] = df.apply(classify_topology, axis=1)
        
        else:
            raise ValueError(f"Unknown target type: {target_type}")
        
        # Split data
        train_val_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_state
        )
        
        train_df, val_df = train_test_split(
            train_val_df, test_size=val_size/(1-test_size), random_state=random_state
        )
        
        print(f"Dataset created with target '{target_type}':")
        print(f"  Train: {len(train_df)} samples")
        print(f"  Val: {len(val_df)} samples")
        print(f"  Test: {len(test_df)} samples")
        
        if 'target' in df.columns:
            print(f"  Target distribution:")
            print(df['target'].value_counts().sort_index())
        
        return train_df, val_df, test_df
    
    def save_datasets(self, 
                     output_dir: str = "data",
                     target_type: str = 'h11_prediction'):
        """
        Save train/val/test datasets to CSV files.
        
        Args:
            output_dir: Directory to save files
            target_type: Type of target to generate
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        train_df, val_df, test_df = self.generate_ml_dataset(target_type=target_type)
        
        # Save datasets
        train_df.to_csv(output_path / f'cicy_train_{target_type}.csv', index=False)
        val_df.to_csv(output_path / f'cicy_val_{target_type}.csv', index=False)
        test_df.to_csv(output_path / f'cicy_test_{target_type}.csv', index=False)
        
        # Save metadata
        metadata = {
            'source': str(self.data_path),
            'target_type': target_type,
            'num_manifolds': len(self.manifolds),
            'num_features': len(train_df.columns) - 1 if 'target' in train_df.columns else len(train_df.columns),
            'train_size': len(train_df),
            'val_size': len(val_df),
            'test_size': len(test_df),
            'feature_names': list(train_df.columns)
        }
        
        with open(output_path / f'cicy_metadata_{target_type}.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Datasets saved to {output_path}/")
        print(f"  - cicy_train_{target_type}.csv")
        print(f"  - cicy_val_{target_type}.csv")
        print(f"  - cicy_test_{target_type}.csv")
        print(f"  - cicy_metadata_{target_type}.json")


def main():
    """Example usage of CICY data loader."""
    
    # Load CICY data
    loader = CICYDataLoader('dataset/cicylist.txt')
    
    print(f"Loaded {len(loader.manifolds)} CICY manifolds")
    
    # Convert to DataFrame
    df = loader.to_dataframe()
    print(f"\nDataFrame shape: {df.shape}")
    print(f"Features: {list(df.columns)}")
    
    # Show statistics
    print("\nHodge number statistics:")
    print(df[['h11', 'h21', 'euler_char']].describe())
    
    # Generate different ML datasets
    print("\n" + "="*50)
    print("Generating ML datasets...")
    
    # 1. H11 prediction task
    loader.save_datasets(target_type='h11_prediction')
    
    # 2. Mirror symmetry classification
    loader.save_datasets(target_type='mirror_symmetry')
    
    # 3. Topology classification
    loader.save_datasets(target_type='topology_class')
    
    print("\nDone! Real CICY data is ready for training.")


if __name__ == "__main__":
    main()

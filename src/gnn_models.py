#!/usr/bin/env python
"""
Graph Neural Network models for Calabi-Yau configuration matrices.

This module implements GNN architectures to process CY manifolds as graphs,
where the configuration matrix defines the graph structure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data, DataLoader
import numpy as np
from typing import List, Tuple, Optional


class CYGraphEncoder:
    """Convert CY configuration matrices to graph representations."""
    
    @staticmethod
    def matrix_to_graph(config_matrix: np.ndarray, 
                       hodge_numbers: dict = None) -> Data:
        """
        Convert configuration matrix to PyTorch Geometric graph.
        
        The graph structure:
        - Nodes: Represent projective spaces and polynomials
        - Edges: Non-zero entries in the configuration matrix
        - Node features: Derived from matrix rows/columns
        - Edge features: Matrix values
        
        Args:
            config_matrix: CY configuration matrix
            hodge_numbers: Dictionary with h11, h21, etc.
            
        Returns:
            PyTorch Geometric Data object
        """
        n_rows, n_cols = config_matrix.shape
        
        # Create node features
        # First n_cols nodes: projective spaces
        # Next n_rows nodes: polynomials
        num_nodes = n_rows + n_cols
        
        node_features = []
        
        # Features for projective space nodes
        for j in range(n_cols):
            col_sum = config_matrix[:, j].sum()
            col_mean = config_matrix[:, j].mean()
            col_std = config_matrix[:, j].std()
            col_max = config_matrix[:, j].max()
            col_nonzero = np.count_nonzero(config_matrix[:, j])
            node_features.append([col_sum, col_mean, col_std, col_max, col_nonzero, 0])  # 0 = PS node
        
        # Features for polynomial nodes
        for i in range(n_rows):
            row_sum = config_matrix[i, :].sum()
            row_mean = config_matrix[i, :].mean()
            row_std = config_matrix[i, :].std()
            row_max = config_matrix[i, :].max()
            row_nonzero = np.count_nonzero(config_matrix[i, :])
            node_features.append([row_sum, row_mean, row_std, row_max, row_nonzero, 1])  # 1 = Poly node
        
        # Add global features if available
        if hodge_numbers:
            h11 = hodge_numbers.get('h11', 0)
            h21 = hodge_numbers.get('h21', 0)
            euler = hodge_numbers.get('euler_char', 0)
            
            # Add to all nodes
            for i in range(num_nodes):
                node_features[i].extend([h11, h21, euler])
        
        x = torch.tensor(node_features, dtype=torch.float)
        
        # Create edges based on non-zero matrix entries
        edge_index = []
        edge_attr = []
        
        for i in range(n_rows):
            for j in range(n_cols):
                if config_matrix[i, j] != 0:
                    # Edge from PS node j to Poly node (n_cols + i)
                    edge_index.append([j, n_cols + i])
                    edge_index.append([n_cols + i, j])  # Bidirectional
                    
                    # Edge features: matrix value and normalized value
                    val = config_matrix[i, j]
                    edge_attr.append([val, val / (np.abs(config_matrix).max() + 1e-8)])
                    edge_attr.append([val, val / (np.abs(config_matrix).max() + 1e-8)])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


class CYGraphNet(nn.Module):
    """Graph Neural Network for Calabi-Yau manifolds."""
    
    def __init__(self, 
                 input_dim: int = 9,  # Node feature dimension
                 hidden_dim: int = 64,
                 output_dim: int = 1,
                 num_layers: int = 3,
                 dropout: float = 0.2,
                 task_type: str = 'regression'):
        """
        Initialize CY Graph Neural Network.
        
        Args:
            input_dim: Dimension of node features
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
            num_layers: Number of GCN layers
            dropout: Dropout rate
            task_type: 'regression' or 'classification'
        """
        super().__init__()
        self.task_type = task_type
        self.dropout = dropout
        
        # Graph convolution layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        self.convs.append(GCNConv(input_dim, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Readout layers
        self.fc1 = nn.Linear(hidden_dim * 3, hidden_dim)  # *3 for mean, max, add pooling
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        """
        Forward pass through GNN.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features (optional)
            batch: Batch assignment for nodes
            
        Returns:
            Output predictions
        """
        # Graph convolutions
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout_layer(x)
        
        # Global pooling (combine node features into graph-level features)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Multiple pooling strategies
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_add = global_add_pool(x, batch)
        
        # Concatenate pooled features
        x = torch.cat([x_mean, x_max, x_add], dim=1)
        
        # Final MLP
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout_layer(x)
        
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout_layer(x)
        
        x = self.fc3(x)
        
        if self.task_type == 'classification' and self.fc3.out_features > 1:
            x = F.softmax(x, dim=1)
        
        return x


class AttentionGNN(MessagePassing):
    """Graph Attention Network for CY manifolds."""
    
    def __init__(self, input_dim: int, hidden_dim: int, heads: int = 4):
        """
        Initialize attention-based GNN.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            heads: Number of attention heads
        """
        super().__init__(aggr='add')
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.heads = heads
        
        # Multi-head attention
        self.lin_key = nn.Linear(input_dim, hidden_dim * heads)
        self.lin_query = nn.Linear(input_dim, hidden_dim * heads)
        self.lin_value = nn.Linear(input_dim, hidden_dim * heads)
        
        # Output projection
        self.lin_out = nn.Linear(hidden_dim * heads, hidden_dim)
        
        # Layer norm
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x, edge_index):
        """Forward pass with attention."""
        # Compute key, query, value
        key = self.lin_key(x)
        query = self.lin_query(x)
        value = self.lin_value(x)
        
        # Propagate with attention
        out = self.propagate(edge_index, query=query, key=key, value=value)
        
        # Output projection and normalization
        out = self.lin_out(out)
        out = self.norm(out)
        
        return out
    
    def message(self, query_i, key_j, value_j):
        """Compute attention-weighted messages."""
        # Reshape for multi-head attention
        B = query_i.size(0)
        H = self.heads
        D = self.hidden_dim
        
        query_i = query_i.view(B, H, D)
        key_j = key_j.view(B, H, D)
        value_j = value_j.view(B, H, D)
        
        # Compute attention scores
        scores = (query_i * key_j).sum(dim=-1) / np.sqrt(D)
        alpha = F.softmax(scores, dim=0)
        
        # Apply attention to values
        alpha = alpha.unsqueeze(-1)
        out = alpha * value_j
        
        return out.view(B, -1)


class CYGraphEnsemble:
    """Ensemble of GNN models for robust predictions."""
    
    def __init__(self, 
                 n_models: int = 5,
                 input_dim: int = 9,
                 output_dim: int = 1,
                 task_type: str = 'regression'):
        """
        Initialize GNN ensemble.
        
        Args:
            n_models: Number of models in ensemble
            input_dim: Input feature dimension
            output_dim: Output dimension
            task_type: Task type
        """
        self.models = []
        self.n_models = n_models
        self.task_type = task_type
        
        # Create diverse GNN architectures
        architectures = [
            {'hidden_dim': 32, 'num_layers': 2, 'dropout': 0.1},
            {'hidden_dim': 64, 'num_layers': 3, 'dropout': 0.2},
            {'hidden_dim': 128, 'num_layers': 4, 'dropout': 0.3},
            {'hidden_dim': 64, 'num_layers': 2, 'dropout': 0.15},
            {'hidden_dim': 96, 'num_layers': 3, 'dropout': 0.25},
        ]
        
        for i in range(n_models):
            arch = architectures[i % len(architectures)]
            model = CYGraphNet(
                input_dim=input_dim,
                output_dim=output_dim,
                task_type=task_type,
                **arch
            )
            self.models.append(model)
    
    def train_ensemble(self, train_loader, val_loader, epochs: int = 100, device: str = 'cpu'):
        """Train all models in the ensemble."""
        histories = []
        
        for i, model in enumerate(self.models):
            print(f"\nTraining GNN {i+1}/{self.n_models}...")
            model = model.to(device)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            if self.task_type == 'regression':
                criterion = nn.MSELoss()
            else:
                criterion = nn.CrossEntropyLoss()
            
            history = {'train_loss': [], 'val_loss': []}
            
            for epoch in range(epochs):
                # Training
                model.train()
                train_loss = 0
                for batch in train_loader:
                    batch = batch.to(device)
                    optimizer.zero_grad()
                    
                    out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                    loss = criterion(out, batch.y)
                    
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                
                # Validation
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch in val_loader:
                        batch = batch.to(device)
                        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                        loss = criterion(out, batch.y)
                        val_loss += loss.item()
                
                train_loss /= len(train_loader)
                val_loss /= len(val_loader)
                
                history['train_loss'].append(train_loss)
                history['val_loss'].append(val_loss)
                
                if epoch % 10 == 0:
                    print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
            
            histories.append(history)
        
        return histories
    
    def predict_with_uncertainty(self, data_loader, device: str = 'cpu'):
        """Make ensemble predictions with uncertainty."""
        all_predictions = []
        
        for model in self.models:
            model = model.to(device)
            model.eval()
            
            predictions = []
            with torch.no_grad():
                for batch in data_loader:
                    batch = batch.to(device)
                    out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                    predictions.append(out.cpu().numpy())
            
            predictions = np.concatenate(predictions, axis=0)
            all_predictions.append(predictions)
        
        all_predictions = np.array(all_predictions)
        
        # Calculate mean and uncertainty
        mean_pred = all_predictions.mean(axis=0)
        
        if self.task_type == 'regression':
            uncertainty = all_predictions.std(axis=0)
        else:
            # For classification: entropy of averaged probabilities
            mean_probs = all_predictions.mean(axis=0)
            entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-8), axis=-1)
            uncertainty = entropy
        
        return mean_pred, uncertainty


def create_cy_graph_dataset(cicy_loader, target_type: str = 'h11_prediction'):
    """
    Create graph dataset from CICY data.
    
    Args:
        cicy_loader: CICYDataLoader instance
        target_type: Type of prediction target
        
    Returns:
        List of PyTorch Geometric Data objects
    """
    encoder = CYGraphEncoder()
    graph_data = []
    
    for manifold in cicy_loader.manifolds:
        if 'config_matrix' not in manifold:
            continue
        
        # Create graph from configuration matrix
        hodge_dict = {
            'h11': manifold.get('h11', 0),
            'h21': manifold.get('h21', 0),
            'euler_char': manifold.get('euler_char', 0)
        }
        
        graph = encoder.matrix_to_graph(
            manifold['config_matrix'],
            hodge_numbers=hodge_dict
        )
        
        # Add target based on type
        if target_type == 'h11_prediction':
            graph.y = torch.tensor([manifold['h11']], dtype=torch.float)
        elif target_type == 'h21_prediction':
            graph.y = torch.tensor([manifold['h21']], dtype=torch.float)
        elif target_type == 'mirror_symmetry':
            is_mirror = 1 if manifold['h11'] == manifold['h21'] else 0
            graph.y = torch.tensor([is_mirror], dtype=torch.long)
        elif target_type == 'topology_class':
            if manifold['h11'] < 10 and manifold['h21'] < 10:
                cls = 0
            elif manifold['h11'] > 20 or manifold['h21'] > 20:
                cls = 2
            else:
                cls = 1
            graph.y = torch.tensor([cls], dtype=torch.long)
        
        graph_data.append(graph)
    
    return graph_data


if __name__ == "__main__":
    print("GNN Models for Calabi-Yau Configuration Matrices")
    print("="*50)
    
    # Example: Create a simple graph from a configuration matrix
    config_matrix = np.array([
        [1, 1, 0, 0],
        [0, 1, 1, 0],
        [1, 0, 1, 1],
        [0, 0, 0, 2]
    ])
    
    encoder = CYGraphEncoder()
    graph = encoder.matrix_to_graph(config_matrix, {'h11': 15, 'h21': 15})
    
    print(f"Graph created:")
    print(f"  Nodes: {graph.x.shape[0]}")
    print(f"  Node features: {graph.x.shape[1]}")
    print(f"  Edges: {graph.edge_index.shape[1]}")
    print(f"  Edge features: {graph.edge_attr.shape}")
    
    # Test GNN forward pass
    model = CYGraphNet(input_dim=graph.x.shape[1])
    output = model(graph.x, graph.edge_index, graph.edge_attr)
    print(f"\nGNN output shape: {output.shape}")
    print("GNN models ready for CY manifold processing!")

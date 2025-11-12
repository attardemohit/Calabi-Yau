#!/usr/bin/env python
"""Simple test to debug the training issue"""

import sys
sys.path.append('src')

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

# Load data
train_df = pd.read_csv('data/calabi_yau_train_regression.csv')
print("Data loaded successfully")
print(f"Shape: {train_df.shape}")
print(f"Target stats:\n{train_df['particle_spectrum'].describe()}")

# Prepare data
feature_cols = [col for col in train_df.columns if col != 'particle_spectrum']
X = train_df[feature_cols].values.astype(np.float32)
y = train_df['particle_spectrum'].values.astype(np.float32).reshape(-1, 1)

print(f"\nX shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"X range: [{X.min():.2f}, {X.max():.2f}]")
print(f"y range: [{y.min():.2f}, {y.max():.2f}]")

# Create simple model
model = nn.Sequential(
    nn.Linear(X.shape[1], 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
)

# Test forward pass
X_tensor = torch.FloatTensor(X[:10])
y_tensor = torch.FloatTensor(y[:10])

output = model(X_tensor)
print(f"\nModel output shape: {output.shape}")
print(f"Model output range: [{output.min().item():.2f}, {output.max().item():.2f}]")

# Test loss
criterion = nn.MSELoss()
loss = criterion(output, y_tensor)
print(f"Loss: {loss.item():.4f}")

# Train for a few steps
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("\nTraining for 10 steps...")
for i in range(10):
    # Get batch
    idx = np.random.choice(len(X), 32)
    X_batch = torch.FloatTensor(X[idx])
    y_batch = torch.FloatTensor(y[idx])
    
    # Forward pass
    optimizer.zero_grad()
    output = model(X_batch)
    loss = criterion(output, y_batch)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    print(f"Step {i+1}: Loss = {loss.item():.4f}")

print("\nTest complete!")

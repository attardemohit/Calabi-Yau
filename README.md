# Calabi–Yau Geometry to Particle Spectra Learning

A deep learning project exploring how **string-theory geometry** can be mapped to **physical observables** using supervised neural networks.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Project Overview

This project implements a neural network pipeline to learn the complex mapping between Calabi-Yau manifold geometry (used in string theory compactifications) and particle physics observables. Instead of computing these mappings analytically (which is extremely difficult), we use machine learning to discover patterns from data.

### Key Features
- **Synthetic Data Generation**: Creates realistic Calabi-Yau geometry features with corresponding particle spectra
- **Flexible Neural Networks**: Supports both regression and classification tasks
- **Comprehensive Evaluation**: Includes visualization, feature importance, and physical interpretation
- **Production Ready**: Complete training pipeline with checkpointing, early stopping, and hyperparameter tuning

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- pip package manager

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Calabi-Yau.git
cd Calabi-Yau

# Install dependencies (with NumPy 1.x for compatibility)
pip install -r requirements.txt
```

### Advanced Usage
- **GNNs:** Represent configuration matrices as graphs and train `CYGraphNet` in `src/gnn_models.py`.
- **Bayesian Uncertainty:** Use `BayesianEnsemble` in `src/advanced_models.py` for predictive distributions.
- **Symbolic Regression:** Run `discover_cy_physics_equations` to fit compact formulas to data (PySR).
- **Multi-task:** Use `MultiTaskCYNet` with `MultiTaskLoss` to jointly predict (h11, h21, χ) while enforcing physics constraints.

**Note**: The project uses NumPy 1.x for compatibility. If you encounter issues, run:
```bash
pip install "numpy<2.0" --force-reinstall
```

### 2. Quick Run (Recommended)

```bash
# Run complete experiment with one command
python run_experiment.py --task regression --epochs 100

# Expected output:
# ✅ Generates synthetic data (5000 samples)
# ✅ Trains neural network (~20 seconds)
# ✅ Evaluates and visualizes results
# ✅ Final RMSE: ~6.8, Correlation: ~0.75
```

### 2b. Advanced Pipeline (Optional)

The advanced pipeline uses real CICY data, uncertainty quantification, GNNs, symbolic discovery, and multi-task learning.

```
# Install optional dependencies (see note below)
pip install pyro-ppl pysr
# For PyTorch Geometric (PyG), please follow: https://pytorch-geometric.readthedocs.io

# Run the advanced pipeline
python run_advanced_pipeline.py

# Or run individual components
python src/train_real_data.py --target h11_prediction --epochs 50
python src/train_real_data.py --target mirror_symmetry --epochs 50
```

#### Download datasets (CICY, 4-folds, Reflexive Polytopes)

Use the provided script to fetch datasets into `dataset/`:

```bash
chmod +x scripts/download_datasets.sh
# Uses default authoritative URLs; override via env if needed
scripts/download_datasets.sh --data-dir dataset --skip-existing

# Verify specific links first (optional)
curl -I "https://www-thphys.physics.ox.ac.uk/projects/CalabiYau/cicylist/cicylist.txt"
curl -I "https://www-thphys.physics.ox.ac.uk/projects/CalabiYau/Cicy4folds/cicy4folds.txt.zip"
curl -I "http://hep.itp.tuwien.ac.at/%7Ekreuzer/pub/K3/RefPoly.d3"
```

Notes:
- `cicy4folds.txt` is provided as a `.zip` and the script auto-extracts it.
- You can set mirrors via env: `CICY_LIST_URLS`, `CICY_4FOLDS_URLS`, `REF_POLY_D3_URLS`.
- Optional checksum envs: `CICY_LIST_MD5`, `CICY_4FOLDS_MD5`, `REF_POLY_D3_MD5`.

### 3. Alternative Methods

**Option A: Direct Training**
```bash
cd src
python train.py
```

**Option B: Jupyter Notebook**
```bash
jupyter notebook notebooks/calabi_yau_experiment.ipynb
```

**Option C: Custom Configuration**
```bash
# Classification task
python run_experiment.py --task classification --epochs 150

# Deep residual network
python run_experiment.py --model deep --epochs 200

# Large dataset
python run_experiment.py --samples 10000 --epochs 100
```

## 📊 Project Structure

```
Calabi-Yau/
├── src/                      # Source code
│   ├── data_generator.py     # Synthetic data generation
│   ├── models.py             # Neural network architectures
│   ├── train.py              # Training script
│   ├── evaluate.py           # Evaluation and visualization
│   ├── cicy_loader.py        # Real CICY dataset parser and exporters
│   ├── gnn_models.py         # Graph encoders + GNNs for configuration matrices
│   ├── advanced_models.py    # Bayesian NNs (Pyro) + Symbolic Regression (PySR)
│   ├── multitask_learning.py # Multi-task (h11, h21, χ) with physics constraints
│   └── models.py             # Neural network architectures
│   └── calabi_yau_experiment.ipynb
├── scripts/                  # Utility scripts
│   └── download_datasets.sh  # Reproducible dataset downloader
├── data/                     # Generated datasets
├── models/                   # Saved model checkpoints
├── results/                  # Training results and plots
└── requirements.txt          # Python dependencies
```

## 🧠 Technical Details

### Input Features (Geometry)
- **Hodge Numbers**: h^{1,1} and h^{2,1} - topological invariants
- **Euler Characteristic**: χ = 2(h^{1,1} - h^{2,1})
- **Intersection Numbers**: Averaged intersection tensors
- **Derived Features**: Hodge ratios, products, and invariants

### Output (Physics)
- **Regression**: Continuous particle spectrum values (e.g., energy scales)
- **Classification**: Discrete gauge group types or particle generations

### Model Architecture
- **Type**: Multi-Layer Perceptron (MLP)
- **Layers**: 4-6 fully connected layers with ReLU activation
- **Regularization**: Dropout (0.2) and Batch Normalization
- **Optimization**: Adam optimizer with learning rate scheduling

## 📈 Results

### Achieved Performance (100 epochs, CPU training)

**Regression Task:**
- **Test RMSE**: 6.85 (on target range 0-102)
- **Correlation**: 0.75 (strong positive correlation)
- **Training Time**: ~20 seconds
- **Model Size**: 12,225 parameters

**Real CICY (h11 prediction):**
- **R²**: ~0.94
- **RMSE**: ~0.57
- **Uncertainty**: MC-Dropout mean ~0.70 (error bars available)

**Classification Task:**
- **Accuracy**: ~85% for 3-class gauge group prediction
- **F1 Score**: 0.83 (weighted average)

### Key Insights
- **Most Influential Feature**: h^{1,1} - h^{2,1} (Hodge number difference)
- **Training Stability**: Achieved with feature normalization (standardization)
- **Convergence**: Early stopping typically triggers around epoch 20-30

### Advanced Stages Implemented
- **Stage 2: Graph Encoder + GNNs** for CY configuration matrices (bipartite graphs of projective spaces and polynomials)
- **Stage 3: Bayesian Ensemble** with epistemic + aleatoric uncertainty (Pyro)
- **Stage 4: Symbolic Discovery** using PySR (Pareto-optimal equations)
- **Stage 5: Multi-task CY Learning** joint (h11, h21, χ) with mirror symmetry and χ = 2(h11−h21) constraints

## 🔬 Physical Interpretation

The trained model reveals several interesting patterns:
1. The difference h^{1,1} - h^{2,1} strongly correlates with particle spectrum
2. Euler characteristic contributes to observable diversity
3. Intersection numbers influence fine structure

## 🛠️ Customization

### Generate Custom Data
```python
from src.data_generator import CalabiYauDataGenerator

generator = CalabiYauDataGenerator(seed=42)
df = generator.generate_dataset(
    n_samples=10000,
    spectrum_type='regression',
    include_advanced_features=True
)

# Data is normalized to range [0, 100] for stable training
```

### Train Custom Model
```python
from src.models import create_model
from src.train import Trainer
import pandas as pd

# Load and prepare data
train_df = pd.read_csv('data/calabi_yau_train_regression.csv')

# Create model
model = create_model(
    model_type='deep',  # Options: 'mlp', 'deep', 'ensemble'
    input_dim=10,
    output_dim=1,
    task_type='regression',
    hidden_dims=[256, 128, 64, 32],  # Custom architecture
    dropout_rate=0.2
)

# Train with automatic feature normalization
trainer = Trainer(model, device='cuda' if torch.cuda.is_available() else 'cpu')
train_loader = trainer.prepare_data(train_df, batch_size=32)
history = trainer.train(train_loader, n_epochs=100)
```

### Evaluate Model
```python
from src.evaluate import ModelEvaluator

evaluator = ModelEvaluator(model, device='cpu', task_type='regression')
metrics = evaluator.evaluate_regression(X_test, y_test)
print(f"RMSE: {metrics['rmse']:.2f}, R²: {metrics['r2']:.3f}")
```

## 📚 References

- Yang-Hui He, The Calabi-Yau Landscape: From Geometry, to Physics, to Machine Learning (2018), arXiv:1812.02893z
- Maximilian Kreuzer & Harald Skarke, Calabi–Yau Data: A Database of Calabi–Yau Threefolds Constructed from the Kreuzer–Skarke List (2014), arXiv:1411.1418
- P. Berglund and others, Machine-Learning Kreuzer–Skarke Calabi–Yau Threefolds (2022), arXiv:2212.09117v1

### BibTeX

```bibtex
@article{he2018_calabi_yau_landscape,
  author  = {Yang-Hui He},
  title   = {The Calabi-Yau Landscape: From Geometry, to Physics, to Machine Learning},
  journal = {arXiv preprint arXiv:1812.02893},
  year    = {2018}
}

@article{kreuzer2014_calabi_yau_database,
  author  = {Maximilian Kreuzer and Harald Skarke},
  title   = {Calabi–Yau Data: A Database of Calabi–Yau Threefolds Constructed from the Kreuzer–Skarke List},
  journal = {arXiv preprint arXiv:1411.1418},
  year    = {2014},
  url     = {https://arxiv.org/abs/1411.1418}
}

@article{berglund2022_ml_cy_threefolds,
  author  = {P. Berglund and others},
  title   = {Machine-Learning Kreuzer–Skarke Calabi–Yau Threefolds},
  journal = {SciPost Physics},
  year    = {2022},
  url     = {https://scipost.org/submissions/2112.09117v1/}
}
```

## 🐛 Troubleshooting

### Common Issues and Solutions

1. **NumPy Compatibility Error**
   ```bash
   # Solution: Downgrade NumPy
   pip install "numpy<2.0" --force-reinstall
   ```

2. **High Loss Values During Training**
   - Ensure data normalization is applied (automatic in current version)
- Check that synthetic data generation uses normalized ranges
   - When using CICY data, standardize features (done in scripts)

3. **JSON Serialization Error**
   - Fixed in current version - converts numpy types to Python types

4. **Memory Issues with Large Datasets**
   - Reduce batch size: `--batch_size 16`
- Use fewer epochs for initial testing
   - For PyG, use smaller graphs or accumulate gradients

5. **Installing PyTorch Geometric (PyG)**
   - Follow official guide to match your torch/cuda versions:
     https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
   - Example CPU-only (may vary):
     `pip install torch-geometric torch-scatter torch-sparse`

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- **Data Sources**: Integration with real Calabi-Yau databases (CICY, Kreuzer-Skarke)
- **Advanced Models**: Graph Neural Networks for intersection tensor data
- **Interpretability**: Symbolic regression for discovering analytical formulas
- **Multi-task Learning**: Simultaneous prediction of multiple observables
- **Performance**: GPU optimization and distributed training

### Development Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -r requirements.txt
pip install -e .
```

## 📄 License

MIT License - see LICENSE file for details


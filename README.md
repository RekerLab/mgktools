# mgktools

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**mgktools** is a Python package for molecular property prediction using Marginalized Graph Kernels (MGK). It provides a comprehensive framework for training Gaussian Process and Support Vector Machine models on molecular datasets, with built-in support for hyperparameter optimization and model interpretability.

## Features

- **Graph Kernel Methods**: Marginalized Graph Kernel (MGK) for molecular similarity computation
- **Multiple Model Types**: Gaussian Process Regression (GPR), Gaussian Process Classification (GPC), and SVM models
- **Hyperparameter Optimization**: Bayesian optimization via Optuna and gradient-based optimization
- **Model Interpretability**: Atomic and molecular attribution for understanding predictions
- **Flexible Molecular Representations**: Support for 13+ fingerprint types and molecular descriptors
- **Scalable Methods**: Nystrom approximation and local expert models for large datasets
- **Cross-Validation**: Built-in k-fold, leave-one-out, and Monte Carlo cross-validation

## Installation

### Requirements
- Python == 3.12
- GCC == 9 or 11
- CUDA >= 11 (No CPU implementation for graph kernel computation)

### Install from PyPI

```bash
# Install graphdot dependency (required)
pip install git+https://gitlab.com/Xiangyan93/graphdot.git@v0.8.2

# Install mgktools from PyPI
pip install mgktools
```

### Install from Source

```bash
# Install graphdot dependency (required)
pip install git+https://gitlab.com/Xiangyan93/graphdot.git@v0.8.2

git clone https://github.com/Xiangyan93/mgktools.git
cd mgktools
pip install -e .
```

## Quick Start with Google Colab Tutorial

GPU is required to compute graph kernels. Try the interactive tutorial:

- [Interpretability Marginalized Graph Kernel](https://colab.research.google.com/drive/1Z6tx4_3FBhZB9SwOvxex9jIP-XZ40hi9?usp=sharing)

## Command-Line Tools

mgktools provides several CLI commands:

### mgk_cache_data
Pre-cache graph objects and molecular features for faster processing.
```bash
mgk_cache_data --data_paths data.csv --smiles_columns smiles --cache_graph --cache_path cache.pkl
```

### mgk_kernel_calc
Compute and save kernel matrices.
```bash
mgk_kernel_calc --save_dir output/ --data_path data.csv --smiles_columns smiles \
    --graph_kernel_type graph --graph_hyperparameters additive-pnorm.json
```

### mgk_cross_validation
Run cross-validation experiments.
```bash
mgk_cross_validation --save_dir output/ --data_path data.csv --smiles_columns smiles \
    --targets_columns target --graph_kernel_type graph --graph_hyperparameters additive-pnorm.json \
    --model_type gpr --task_type regression --cross_validation kFold --n_splits 5 \
    --alpha 0.01 --metric rmse
```

### mgk_optuna
Bayesian hyperparameter optimization with Optuna.
```bash
mgk_optuna --save_dir output/ --data_path data.csv --smiles_columns smiles \
    --targets_columns target --graph_kernel_type graph --graph_hyperparameters additive-pnorm.json \
    --model_type gpr --task_type regression --cross_validation leave-one-out \
    --alpha 0.01 --alpha_bounds 0.001 0.1 --metric rmse --num_iters 100
```

### mgk_embedding
Compute molecular embeddings (t-SNE or kPCA).
```bash
mgk_embedding --save_dir output/ --data_path data.csv --smiles_columns smiles \
    --targets_columns target --graph_kernel_type graph --graph_hyperparameters additive-pnorm.json \
    --embedding_algorithm tSNE --n_components 2 --perplexity 30
```

### Basic Usage

```python
import pandas as pd
from mgktools.data.data import Dataset
from mgktools.kernels.utils import get_kernel_config
from mgktools.evaluators.cross_validation import Evaluator

# Load data from DataFrame
df = pd.DataFrame({
    'smiles': ['CCO', 'CCC', 'CCCC', 'CCCCC', 'CCCCCC'],
    'target': [1.0, 2.0, 3.0, 4.0, 5.0]
})

# Create dataset
dataset = Dataset.from_df(
    df,
    smiles_columns=['smiles'],
    targets_columns=['target']
)

# Set up for graph kernel computation
dataset.set_status(graph_kernel_type='graph')
dataset.create_graphs(n_jobs=4)
dataset.unify_datatype()

# Configure kernel (using additive kernel with p-normalization)
kernel_config = get_kernel_config(
    dataset=dataset,
    graph_kernel_type='graph',
    mgk_hyperparameters_files=['mgktools/hyperparameters/configs/additive-pnorm.json']
)

# Run cross-validation
evaluator = Evaluator(
    dataset=dataset,
    model_type='gpr',
    task_type='regression',
    kernel_config=kernel_config,
    split_type='random',
    n_splits=5,
    metrics=['rmse', 'r2']
)
results = evaluator.run_cross_validation()
```

### Using Molecular Fingerprints

```python
from mgktools.features_mol.features_generators import FeaturesGenerator

# Create feature generator
fg = FeaturesGenerator('morgan', radius=2, num_bits=2048)

# Set up dataset with molecular features
dataset.set_status(
    graph_kernel_type='graph',
    features_generators=[fg],
    features_combination='concat'
)
dataset.create_graphs(n_jobs=4)
dataset.create_features_mol(n_jobs=4)
```

### Hyperparameter Optimization with Optuna

```python
from mgktools.hyperparameters.optuna import bayesian_optimization

# Run Bayesian optimization
best_params, best_score = bayesian_optimization(
    dataset_train=train_dataset,
    dataset_val=val_dataset,
    kernel_config=kernel_config,
    model_type='gpr',
    task_type='regression',
    metric='rmse',
    n_trials=100
)
```

## Package Structure

```
mgktools/
├── data/               # Dataset handling and caching
│   ├── data.py        # CachedDict, Datapoint, Dataset classes
│   └── split.py       # Train/val/test splitting utilities
├── kernels/           # Kernel implementations
│   ├── base.py        # MicroKernel, BaseKernelConfig
│   ├── GraphKernel.py # MGK and GraphKernelConfig
│   ├── FeatureKernel.py
│   ├── HybridKernel.py
│   └── utils.py       # get_kernel_config factory
├── models/            # Model implementations
│   ├── regression/    # GPR, NLE, ensemble models
│   └── classification/# GPC, SVM classifiers
├── evaluators/        # Evaluation utilities
│   ├── cross_validation.py  # Evaluator class
│   └── metric.py      # Metric computation
├── features_mol/      # Molecular feature generators
│   └── features_generators.py
├── graph/             # Graph conversion utilities
│   └── hashgraph.py   # HashGraph class
├── interpret/         # Model interpretability
│   └── interpret.py   # Atomic/molecular attribution
├── hyperparameters/   # Optimization and configs
│   ├── optuna.py      # Bayesian optimization
│   └── configs/       # Pre-defined kernel configs
└── exe/               # CLI entry points
    └── run.py         # Command-line tools
```

## Supported Feature Generators

| Name | Description | Default Size |
|------|-------------|--------------|
| `morgan` | Binary Morgan fingerprint | 2048 |
| `morgan_count` | Count-based Morgan fingerprint | 2048 |
| `rdkit_2d` | RDKit 2D descriptors | ~200 |
| `rdkit_2d_normalized` | Normalized RDKit 2D descriptors | ~200 |
| `rdkit_208` | RDKit 208 descriptors | ~210 |
| `rdkit_topol` | RDKit topological fingerprint | 2048 |
| `layered` | Layered fingerprint | 2048 |
| `torsion` | Topological torsion fingerprint | 2048 |
| `atom_pair` | Atom pair fingerprint | 2048 |
| `avalon` | Avalon fingerprint | 2048 |
| `avalon_count` | Count-based Avalon fingerprint | 2048 |
| `maccskey` | MACCS keys fingerprint | 167 |
| `pattern` | Pattern fingerprint | 2048 |

## Hyperparameter Configurations

Pre-defined kernel configurations are available in `mgktools/hyperparameters/configs/`:

- **Additive kernels**: `additive-norm.json`, `additive-pnorm.json`, `additive-msnorm.json`
- **Product kernels**: `product-norm.json`, `product-pnorm.json`, `product-msnorm.json`
- **Feature kernels**: `rbf.json`, `dot_product.json`


## Testing

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/kernel/test_kernel.py

# Run with verbose output
pytest tests/ -v
```

## Related Publications

* [Predicting Single-Substance Phase Diagrams: A Kernel Approach on Graph Representations of Molecules](https://pubs.acs.org/doi/full/10.1021/acs.jpca.1c02391)
* [A Comparative Study of Marginalized Graph Kernel and Message-Passing Neural Network](https://pubs.acs.org/doi/full/10.1021/acs.jcim.1c01118)
* [Interpretable Molecular Property Predictions Using Marginalized Graph Kernels](https://pubs.acs.org/doi/full/10.1021/acs.jcim.3c00396)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

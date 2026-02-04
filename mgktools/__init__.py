#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
mgktools: Marginalized Graph Kernel Tools for Molecular Property Prediction.

This package provides tools for molecular property prediction using Marginalized
Graph Kernels (MGK) with Gaussian Process and Support Vector Machine models.

Key Features
------------
- Graph kernel-based molecular representations
- Gaussian Process Regression (GPR) and Classification (GPC)
- Support Vector Machine (SVM) models
- Hyperparameter optimization via Optuna and gradient-based methods
- Model interpretability through atomic and molecular attribution
- Multiple molecular fingerprint generators
- Cross-validation and evaluation utilities

Modules
-------
data
    Dataset handling, caching, and splitting utilities.
kernels
    Graph kernels, feature kernels, and hybrid kernel configurations.
models
    GPR, GPC, SVM, and ensemble model implementations.
evaluators
    Cross-validation and metric computation.
features_mol
    Molecular fingerprint and descriptor generators.
graph
    SMILES to graph conversion utilities.
interpret
    Atomic and molecular attribution for model interpretability.
hyperparameters
    Hyperparameter optimization with Optuna.
exe
    Command-line interface utilities.

Examples
--------
>>> from mgktools.data.data import Dataset
>>> from mgktools.kernels.utils import get_kernel_config
>>> from mgktools.evaluators.cross_validation import Evaluator
"""

__version__ = '3.4.0'

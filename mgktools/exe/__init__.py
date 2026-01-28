#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Command-line interface module.

This module provides command-line tools for common mgktools operations
including data caching, kernel computation, cross-validation, and
hyperparameter optimization.

CLI Commands
------------
mgk_cache_data
    Pre-cache graph objects and molecular features.
mgk_kernel_calc
    Compute and save kernel matrices.
mgk_cross_validation
    Run cross-validation experiments.
mgk_gradientopt
    Gradient-based hyperparameter optimization.
mgk_optuna
    Bayesian hyperparameter optimization with Optuna.
mgk_optuna_multi_datasets
    Optuna optimization across multiple datasets.
mgk_embedding
    Compute molecular embeddings (t-SNE or kPCA).
"""

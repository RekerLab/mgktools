#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Kernel module for molecular property prediction.

This module provides kernel implementations for Gaussian Process and SVM models,
including Marginalized Graph Kernels (MGK), feature kernels, and hybrid kernels
that combine multiple kernel types.

Key Classes
-----------
MicroKernel
    Smallest unit of a kernel with a single hyperparameter.
GraphKernelConfig
    Configuration for graph kernels with atom/bond/probability microkernels.
FeatureKernelConfig
    Configuration for feature-based kernels (RBF, DotProduct).
HybridKernelConfig
    Configuration combining multiple kernel types.
PreComputedKernelConfig
    Configuration for pre-computed kernel matrices.
MGK
    Marginalized Graph Kernel wrapper with sklearn compatibility.

Key Functions
-------------
get_kernel_config
    Factory function to create kernel configurations from hyperparameter files.
"""

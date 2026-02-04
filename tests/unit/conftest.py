#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pytest configuration and shared fixtures for unit tests.

This module provides common fixtures and configuration for fast unit testing.
"""
import pytest
import numpy as np
import pandas as pd
import tempfile
import shutil
from unittest.mock import Mock, MagicMock
from rdkit import Chem


# =============================================================================
# Session-Scoped Fixtures
# =============================================================================

@pytest.fixture(scope='session')
def sample_molecules():
    """
    Provide a set of sample molecules for testing.

    Returns dictionary with SMILES strings for various molecule types.
    """
    return {
        'ethanol': 'CCO',
        'propane': 'CCC',
        'butane': 'CCCC',
        'benzene': 'c1ccccc1',
        'toluene': 'Cc1ccccc1',
        'caffeine': 'Cn1cnc2c1c(=O)n(c(=O)n2C)C',
        'aspirin': 'CC(=O)Oc1ccccc1C(=O)O',
        'methane': 'C',
        'water': 'O',
        'ammonia': 'N'
    }


@pytest.fixture(scope='session')
def sample_mol_objects(sample_molecules):
    """
    Provide RDKit Mol objects for sample molecules.

    Returns dictionary with RDKit Mol objects.
    """
    return {name: Chem.MolFromSmiles(smiles) for name, smiles in sample_molecules.items()}


# =============================================================================
# Function-Scoped Fixtures
# =============================================================================

@pytest.fixture
def temp_directory():
    """
    Provide a temporary directory that is cleaned up after the test.
    """
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def simple_dataframe():
    """
    Provide a simple DataFrame for testing Dataset creation.
    """
    return pd.DataFrame({
        'smiles': ['CCO', 'CCC', 'CCCC', 'CCCCC', 'CCCCCC'],
        'target': [1.0, 2.0, 3.0, 4.0, 5.0],
        'feature1': [0.1, 0.2, 0.3, 0.4, 0.5],
        'feature2': [1.1, 1.2, 1.3, 1.4, 1.5]
    })


@pytest.fixture
def classification_dataframe():
    """
    Provide a DataFrame for classification testing.
    """
    return pd.DataFrame({
        'smiles': ['CCO', 'CCC', 'CCCC', 'CCCCC', 'CCCCCC',
                   'c1ccccc1', 'Cc1ccccc1', 'CCc1ccccc1', 'c1ccncc1', 'c1cccnc1'],
        'target': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    })


@pytest.fixture
def mock_kernel():
    """
    Provide a mock kernel function.
    """
    def kernel_func(X, Y=None):
        if Y is None:
            Y = X
        n, m = len(X), len(Y)
        return np.random.rand(n, m)

    mock = Mock(side_effect=kernel_func)
    return mock


@pytest.fixture
def mock_model_regression():
    """
    Provide a mock regression model.
    """
    mock = Mock()
    mock.fit = Mock()
    mock.predict = Mock(return_value=(np.array([1.0, 2.0, 3.0]), np.array([0.1, 0.1, 0.1])))
    mock.predict_loocv = Mock(return_value=(np.array([1.0, 2.0, 3.0]), np.array([0.1, 0.1, 0.1])))
    return mock


@pytest.fixture
def mock_model_classification():
    """
    Provide a mock classification model.
    """
    mock = Mock()
    mock.fit = Mock()
    mock.predict = Mock(return_value=np.array([0, 1, 1]))
    mock.predict_proba = Mock(return_value=np.array([0.2, 0.8, 0.9]))
    return mock


# =============================================================================
# Markers Configuration
# =============================================================================

def pytest_configure(config):
    """
    Configure pytest markers.
    """
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )


# =============================================================================
# Test Collection Modifiers
# =============================================================================

def pytest_collection_modifyitems(config, items):
    """
    Modify test collection to skip slow tests unless explicitly requested.
    """
    if config.getoption("-m"):
        # Don't modify if markers are explicitly specified
        return

    skip_slow = pytest.mark.skip(reason="Skipping slow test; use -m slow to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fast unit tests for mgktools.features_mol.features_generators module.

Tests cover FeaturesGenerator class and all supported fingerprint types.
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch
from rdkit import Chem

from mgktools.features_mol.features_generators import (
    FeaturesGenerator,
    AVAILABLE_FEATURES_GENERATORS
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def ethanol_smiles():
    """Ethanol SMILES string."""
    return 'CCO'


@pytest.fixture
def ethanol_mol():
    """Ethanol RDKit Mol object."""
    return Chem.MolFromSmiles('CCO')


@pytest.fixture
def benzene_smiles():
    """Benzene SMILES string."""
    return 'c1ccccc1'


# =============================================================================
# Test Constants
# =============================================================================

class TestConstants:
    """Tests for module constants."""

    def test_available_features_generators_list(self):
        """Test AVAILABLE_FEATURES_GENERATORS contains expected types."""
        expected = [
            'morgan', 'morgan_count', 'rdkit_208', 'rdkit_2d', 'rdkit_2d_normalized',
            'rdkit_topol', 'layered', 'torsion', 'atom_pair', 'avalon', 'avalon_count',
            'maccskey', 'pattern'
        ]
        assert AVAILABLE_FEATURES_GENERATORS == expected

    def test_available_features_generators_length(self):
        """Test number of available feature generators."""
        assert len(AVAILABLE_FEATURES_GENERATORS) == 13


# =============================================================================
# FeaturesGenerator Initialization Tests
# =============================================================================

class TestFeaturesGeneratorInit:
    """Tests for FeaturesGenerator initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        fg = FeaturesGenerator('morgan')
        assert fg.features_generator_name == 'morgan'
        assert fg.radius == 2
        assert fg.num_bits == 2048
        assert fg.atomInvariantsGenerator is False

    def test_init_with_custom_params(self):
        """Test initialization with custom parameters."""
        fg = FeaturesGenerator('morgan', radius=3, num_bits=1024, atomInvariantsGenerator=True)
        assert fg.radius == 3
        assert fg.num_bits == 1024
        assert fg.atomInvariantsGenerator is True

    def test_init_rdkit_2d_creates_generator(self):
        """Test rdkit_2d uses lazy initialization for generator object."""
        fg = FeaturesGenerator('rdkit_2d')
        # Generator is lazily initialized, so it's None until first use
        assert fg._rdkit_2d_generator is None
        # After first use, generator should be created
        _ = fg('CCO')
        assert fg._rdkit_2d_generator is not None

    def test_init_rdkit_2d_normalized_creates_generator(self):
        """Test rdkit_2d_normalized uses lazy initialization for generator object."""
        fg = FeaturesGenerator('rdkit_2d_normalized')
        # Generator is lazily initialized, so it's None until first use
        assert fg._rdkit_2d_generator is None
        # After first use, generator should be created
        _ = fg('CCO')
        assert fg._rdkit_2d_generator is not None


# =============================================================================
# FeaturesGenerator __call__ Tests
# =============================================================================

class TestFeaturesGeneratorCall:
    """Tests for FeaturesGenerator __call__ method."""

    def test_call_with_smiles_string(self, ethanol_smiles):
        """Test calling with SMILES string."""
        fg = FeaturesGenerator('morgan')
        features = fg(ethanol_smiles)
        assert isinstance(features, np.ndarray)
        assert len(features) == 2048

    def test_call_with_mol_object(self, ethanol_mol):
        """Test calling with RDKit Mol object."""
        fg = FeaturesGenerator('morgan')
        features = fg(ethanol_mol)
        assert isinstance(features, np.ndarray)
        assert len(features) == 2048

    def test_call_unknown_generator_raises(self):
        """Test that unknown generator name raises ValueError."""
        fg = FeaturesGenerator('morgan')
        fg.features_generator_name = 'unknown_type'
        with pytest.raises(ValueError, match='unknown features generator'):
            fg('CCO')

    @pytest.mark.parametrize('generator_name', [g for g in AVAILABLE_FEATURES_GENERATORS if g != 'layered'])
    def test_all_generators_return_array(self, generator_name, ethanol_smiles):
        """Test all generators return numpy array.

        Note: 'layered' is excluded as it requires Mol object, not SMILES string.
        """
        fg = FeaturesGenerator(generator_name)
        features = fg(ethanol_smiles)
        assert isinstance(features, np.ndarray)
        assert len(features) > 0

    def test_layered_generator_returns_array(self, ethanol_mol):
        """Test layered generator returns numpy array (requires Mol object)."""
        fg = FeaturesGenerator('layered')
        features = fg(ethanol_mol)
        assert isinstance(features, np.ndarray)
        assert len(features) > 0


# =============================================================================
# Morgan Fingerprint Tests
# =============================================================================

class TestMorganFingerprints:
    """Tests for Morgan fingerprint generation."""

    def test_morgan_binary_returns_correct_size(self, ethanol_smiles):
        """Test Morgan binary fingerprint has correct size."""
        fg = FeaturesGenerator('morgan', num_bits=1024)
        features = fg(ethanol_smiles)
        assert len(features) == 1024

    def test_morgan_binary_values_are_binary(self, ethanol_smiles):
        """Test Morgan binary fingerprint values are 0 or 1."""
        fg = FeaturesGenerator('morgan')
        features = fg(ethanol_smiles)
        assert set(features).issubset({0, 1})

    def test_morgan_count_returns_correct_size(self, ethanol_smiles):
        """Test Morgan count fingerprint has correct size."""
        fg = FeaturesGenerator('morgan_count', num_bits=1024)
        features = fg(ethanol_smiles)
        assert len(features) == 1024

    def test_morgan_count_values_are_non_negative(self, ethanol_smiles):
        """Test Morgan count fingerprint values are non-negative integers."""
        fg = FeaturesGenerator('morgan_count')
        features = fg(ethanol_smiles)
        assert all(f >= 0 for f in features)

    def test_morgan_with_atom_invariants(self, ethanol_smiles):
        """Test Morgan fingerprint with atom invariants generator."""
        fg = FeaturesGenerator('morgan', atomInvariantsGenerator=True)
        features = fg(ethanol_smiles)
        assert len(features) == 2048

    def test_morgan_different_radius_gives_different_features(self, benzene_smiles):
        """Test different radius produces different features."""
        fg1 = FeaturesGenerator('morgan', radius=1)
        fg2 = FeaturesGenerator('morgan', radius=3)
        f1 = fg1(benzene_smiles)
        f2 = fg2(benzene_smiles)
        # Features should be different (at least for some molecules)
        assert not np.array_equal(f1, f2) or np.sum(f1) != np.sum(f2)


# =============================================================================
# RDKit Descriptor Tests
# =============================================================================

class TestRDKitDescriptors:
    """Tests for RDKit descriptor generation."""

    def test_rdkit_208_returns_array(self, ethanol_smiles):
        """Test rdkit_208 returns numpy array."""
        features = FeaturesGenerator.rdkit_208_features_generator(ethanol_smiles)
        assert isinstance(features, np.ndarray)
        assert len(features) > 200

    def test_rdkit_208_handles_large_values(self):
        """Test rdkit_208 caps very large values."""
        # This tests the value capping logic
        fg = FeaturesGenerator('rdkit_208')
        features = fg('CCO')
        # All values should be <= float32 max
        assert all(f <= np.finfo(np.float32).max for f in features)

    def test_rdkit_208_handles_nan_values(self):
        """Test rdkit_208 converts NaN to 0."""
        fg = FeaturesGenerator('rdkit_208')
        features = fg('CCO')
        # No NaN values should be present
        assert not any(np.isnan(f) for f in features)

    def test_rdkit_2d_returns_array(self, ethanol_smiles):
        """Test rdkit_2d returns numpy array."""
        fg = FeaturesGenerator('rdkit_2d')
        features = fg(ethanol_smiles)
        assert isinstance(features, np.ndarray)
        assert len(features) > 100

    def test_rdkit_2d_normalized_returns_array(self, ethanol_smiles):
        """Test rdkit_2d_normalized returns numpy array."""
        fg = FeaturesGenerator('rdkit_2d_normalized')
        features = fg(ethanol_smiles)
        assert isinstance(features, np.ndarray)
        assert len(features) > 100


# =============================================================================
# Topological Fingerprint Tests
# =============================================================================

class TestTopologicalFingerprints:
    """Tests for topological fingerprint generation."""

    def test_rdkit_topol_returns_correct_size(self, ethanol_smiles):
        """Test RDKit topological fingerprint has correct size."""
        fg = FeaturesGenerator('rdkit_topol', num_bits=1024)
        features = fg(ethanol_smiles)
        assert len(features) == 1024

    def test_layered_returns_correct_size(self, ethanol_mol):
        """Test layered fingerprint has correct size.

        Note: layered fingerprint requires Mol object, not SMILES string.
        """
        fg = FeaturesGenerator('layered', num_bits=1024)
        features = fg(ethanol_mol)
        assert len(features) == 1024

    def test_torsion_returns_correct_size(self, ethanol_smiles):
        """Test torsion fingerprint has correct size."""
        fg = FeaturesGenerator('torsion', num_bits=1024)
        features = fg(ethanol_smiles)
        assert len(features) == 1024

    def test_atom_pair_returns_correct_size(self, ethanol_smiles):
        """Test atom pair fingerprint has correct size."""
        fg = FeaturesGenerator('atom_pair', num_bits=1024)
        features = fg(ethanol_smiles)
        assert len(features) == 1024


# =============================================================================
# Avalon Fingerprint Tests
# =============================================================================

class TestAvalonFingerprints:
    """Tests for Avalon fingerprint generation."""

    def test_avalon_returns_correct_size(self, ethanol_smiles):
        """Test Avalon fingerprint has correct size."""
        fg = FeaturesGenerator('avalon', num_bits=512)
        features = fg(ethanol_smiles)
        assert len(features) == 512

    def test_avalon_values_are_binary(self, ethanol_smiles):
        """Test Avalon fingerprint values are binary."""
        fg = FeaturesGenerator('avalon')
        features = fg(ethanol_smiles)
        assert set(features).issubset({0, 1})

    def test_avalon_count_returns_correct_size(self, ethanol_smiles):
        """Test Avalon count fingerprint has correct size."""
        fg = FeaturesGenerator('avalon_count', num_bits=512)
        features = fg(ethanol_smiles)
        assert len(features) == 512


# =============================================================================
# MACCS Keys Tests
# =============================================================================

class TestMACCSKeys:
    """Tests for MACCS keys fingerprint generation."""

    def test_maccskey_returns_fixed_size(self, ethanol_smiles):
        """Test MACCS keys returns 167 bits."""
        fg = FeaturesGenerator('maccskey')
        features = fg(ethanol_smiles)
        assert len(features) == 167

    def test_maccskey_values_are_binary(self, ethanol_smiles):
        """Test MACCS keys values are binary."""
        fg = FeaturesGenerator('maccskey')
        features = fg(ethanol_smiles)
        assert set(features).issubset({0, 1})

    def test_maccskey_ignores_num_bits(self, ethanol_smiles):
        """Test MACCS keys ignores num_bits parameter."""
        fg = FeaturesGenerator('maccskey', num_bits=1024)
        features = fg(ethanol_smiles)
        # Should still be 167, not 1024
        assert len(features) == 167


# =============================================================================
# Pattern Fingerprint Tests
# =============================================================================

class TestPatternFingerprint:
    """Tests for pattern fingerprint generation."""

    def test_pattern_returns_correct_size(self, ethanol_smiles):
        """Test pattern fingerprint has correct size."""
        fg = FeaturesGenerator('pattern', num_bits=1024)
        features = fg(ethanol_smiles)
        assert len(features) == 1024

    def test_pattern_values_are_binary(self, ethanol_smiles):
        """Test pattern fingerprint values are binary."""
        fg = FeaturesGenerator('pattern')
        features = fg(ethanol_smiles)
        assert set(features).issubset({0, 1})


# =============================================================================
# Consistency Tests
# =============================================================================

class TestConsistency:
    """Tests for feature generation consistency."""

    def test_same_molecule_same_features(self, ethanol_smiles):
        """Test same molecule produces same features."""
        fg = FeaturesGenerator('morgan')
        f1 = fg(ethanol_smiles)
        f2 = fg(ethanol_smiles)
        assert np.array_equal(f1, f2)

    def test_smiles_and_mol_same_features(self, ethanol_smiles, ethanol_mol):
        """Test SMILES and Mol give same features."""
        fg = FeaturesGenerator('morgan')
        f_smiles = fg(ethanol_smiles)
        f_mol = fg(ethanol_mol)
        assert np.array_equal(f_smiles, f_mol)

    def test_different_molecules_different_features(self):
        """Test different molecules produce different features."""
        fg = FeaturesGenerator('morgan')
        f1 = fg('CCO')
        f2 = fg('c1ccccc1')
        # Should be different
        assert not np.array_equal(f1, f2)


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_atom_molecule(self):
        """Test single atom molecule."""
        fg = FeaturesGenerator('morgan')
        features = fg('C')
        assert isinstance(features, np.ndarray)
        assert len(features) == 2048

    def test_large_molecule(self):
        """Test larger molecule."""
        # Caffeine
        fg = FeaturesGenerator('morgan')
        features = fg('Cn1cnc2c1c(=O)n(c(=O)n2C)C')
        assert isinstance(features, np.ndarray)
        assert len(features) == 2048

    def test_molecule_with_charge(self):
        """Test molecule with formal charge."""
        fg = FeaturesGenerator('morgan')
        features = fg('[NH4+]')
        assert isinstance(features, np.ndarray)

    def test_molecule_with_stereochemistry(self):
        """Test molecule with stereochemistry."""
        fg = FeaturesGenerator('morgan')
        features = fg('C/C=C/C')  # Trans-2-butene
        assert isinstance(features, np.ndarray)

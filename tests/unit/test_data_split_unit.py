#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fast unit tests for mgktools.data.split module.

Tests cover data splitting utilities with various strategies.
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch
from rdkit import Chem

from mgktools.data.split import (
    get_data_from_index,
    generate_scaffold,
    scaffold_to_smiles,
    get_split_sizes,
    data_split_index,
    dataset_split
)
from mgktools.data.data import Dataset, Datapoint, CachedDict


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def simple_smiles_list():
    """Simple list of SMILES for testing."""
    return ['CCO', 'CCC', 'CCCC', 'c1ccccc1', 'CC(C)C']


@pytest.fixture
def mock_dataset():
    """Create a mock dataset for testing."""
    mock = Mock(spec=Dataset)
    mock.data = [
        Mock(smiles_list=['CCO']),
        Mock(smiles_list=['CCC']),
        Mock(smiles_list=['CCCC']),
        Mock(smiles_list=['c1ccccc1']),
        Mock(smiles_list=['CC(C)C'])
    ]
    mock.mols = np.array([[Chem.MolFromSmiles('CCO')],
                          [Chem.MolFromSmiles('CCC')],
                          [Chem.MolFromSmiles('CCCC')],
                          [Chem.MolFromSmiles('c1ccccc1')],
                          [Chem.MolFromSmiles('CC(C)C')]])
    mock.y = np.array([[1], [2], [3], [4], [5]])
    mock.copy = Mock(return_value=Mock(spec=Dataset, data=mock.data.copy()))
    # Add __len__ for len(dataset) calls
    mock.__len__ = Mock(return_value=5)
    return mock


# =============================================================================
# get_data_from_index Tests
# =============================================================================

class TestGetDataFromIndex:
    """Tests for get_data_from_index function."""

    def test_returns_new_dataset(self, mock_dataset):
        """Test that function returns a new dataset."""
        result = get_data_from_index(mock_dataset, [0, 2, 4])
        mock_dataset.copy.assert_called_once()

    def test_selects_correct_indices(self):
        """Test that correct indices are selected."""
        mock_ds = Mock(spec=Dataset)
        mock_ds.data = ['a', 'b', 'c', 'd', 'e']
        mock_ds.copy = Mock(return_value=Mock(spec=Dataset, data=list(mock_ds.data)))

        result = get_data_from_index(mock_ds, [1, 3])
        assert result.data == ['b', 'd']

    def test_empty_indices(self):
        """Test with empty indices."""
        mock_ds = Mock(spec=Dataset)
        mock_ds.data = ['a', 'b', 'c']
        mock_ds.copy = Mock(return_value=Mock(spec=Dataset, data=list(mock_ds.data)))

        result = get_data_from_index(mock_ds, [])
        assert result.data == []


# =============================================================================
# generate_scaffold Tests
# =============================================================================

class TestGenerateScaffold:
    """Tests for generate_scaffold function."""

    def test_returns_string(self):
        """Test that function returns a string."""
        scaffold = generate_scaffold('CCO')
        assert isinstance(scaffold, str)

    def test_benzene_scaffold(self):
        """Test benzene returns itself as scaffold."""
        scaffold = generate_scaffold('c1ccccc1')
        # Benzene scaffold is benzene
        assert scaffold == 'c1ccccc1'

    def test_toluene_scaffold(self):
        """Test toluene returns benzene scaffold."""
        scaffold = generate_scaffold('Cc1ccccc1')
        # Toluene scaffold is benzene
        assert scaffold == 'c1ccccc1'

    def test_accepts_mol_object(self):
        """Test function accepts RDKit Mol object."""
        mol = Chem.MolFromSmiles('Cc1ccccc1')
        scaffold = generate_scaffold(mol)
        assert scaffold == 'c1ccccc1'

    def test_with_chirality(self):
        """Test with include_chirality=True."""
        # Chiral molecule
        scaffold = generate_scaffold('C[C@H](O)c1ccccc1', include_chirality=True)
        assert isinstance(scaffold, str)

    def test_aliphatic_molecule(self):
        """Test aliphatic molecule scaffold."""
        scaffold = generate_scaffold('CCCCCCCC')
        # Long chain has empty scaffold
        assert scaffold == ''


# =============================================================================
# scaffold_to_smiles Tests
# =============================================================================

class TestScaffoldToSmiles:
    """Tests for scaffold_to_smiles function."""

    def test_returns_dict(self, simple_smiles_list):
        """Test that function returns a dictionary."""
        result = scaffold_to_smiles(simple_smiles_list[:3])
        assert isinstance(result, dict)

    def test_use_indices_false(self):
        """Test with use_indices=False."""
        mols = ['Cc1ccccc1', 'CCc1ccccc1']  # Both have benzene scaffold
        result = scaffold_to_smiles(mols, use_indices=False)

        # Should have benzene scaffold with both molecules
        assert 'c1ccccc1' in result
        assert 'Cc1ccccc1' in result['c1ccccc1'] or 'CCc1ccccc1' in result['c1ccccc1']

    def test_use_indices_true(self):
        """Test with use_indices=True."""
        mols = ['Cc1ccccc1', 'CCc1ccccc1']
        result = scaffold_to_smiles(mols, use_indices=True)

        # Should have indices 0 and 1 for benzene scaffold
        assert 'c1ccccc1' in result
        assert 0 in result['c1ccccc1']
        assert 1 in result['c1ccccc1']

    def test_different_scaffolds(self):
        """Test molecules with different scaffolds."""
        mols = ['c1ccccc1', 'c1ccncc1']  # Benzene and pyridine
        result = scaffold_to_smiles(mols, use_indices=True)

        # Should have two different scaffolds
        assert len(result) == 2


# =============================================================================
# get_split_sizes Tests
# =============================================================================

class TestGetSplitSizes:
    """Tests for get_split_sizes function."""

    def test_basic_split(self):
        """Test basic 80/20 split."""
        sizes = get_split_sizes(100, [0.8, 0.2])
        assert sum(sizes) == 100
        assert sizes[0] == 80
        assert sizes[1] == 20

    def test_three_way_split(self):
        """Test three-way split."""
        sizes = get_split_sizes(100, [0.6, 0.2, 0.2])
        assert sum(sizes) == 100
        assert len(sizes) == 3

    def test_non_divisible_split(self):
        """Test split with non-divisible numbers."""
        sizes = get_split_sizes(10, [0.8, 0.2])
        assert sum(sizes) == 10

    def test_ratios_not_sum_to_one_raises(self):
        """Test that ratios not summing to 1 raises error."""
        with pytest.raises(ValueError, match='do not sum to 1'):
            get_split_sizes(100, [0.5, 0.3])

    def test_negative_ratio_raises(self):
        """Test that negative ratio raises error."""
        with pytest.raises(ValueError, match='non-negative'):
            get_split_sizes(100, [1.2, -0.2])

    def test_small_dataset(self):
        """Test with very small dataset."""
        sizes = get_split_sizes(3, [0.67, 0.33])
        assert sum(sizes) == 3

    def test_single_sample(self):
        """Test with single sample."""
        sizes = get_split_sizes(1, [1.0])
        assert sizes == [1]


# =============================================================================
# data_split_index Tests
# =============================================================================

class TestDataSplitIndex:
    """Tests for data_split_index function."""

    def test_random_split(self):
        """Test random split."""
        indices = data_split_index(
            n_samples=100,
            split_type='random',
            sizes=[0.8, 0.2],
            seed=42
        )
        assert len(indices) == 2
        assert len(indices[0]) + len(indices[1]) == 100
        assert len(set(indices[0]) | set(indices[1])) == 100

    def test_random_split_reproducible(self):
        """Test random split is reproducible with same seed."""
        indices1 = data_split_index(n_samples=100, split_type='random', sizes=[0.8, 0.2], seed=42)
        indices2 = data_split_index(n_samples=100, split_type='random', sizes=[0.8, 0.2], seed=42)
        assert indices1[0] == indices2[0]
        assert indices1[1] == indices2[1]

    def test_random_split_different_seeds(self):
        """Test random split differs with different seeds."""
        indices1 = data_split_index(n_samples=100, split_type='random', sizes=[0.8, 0.2], seed=42)
        indices2 = data_split_index(n_samples=100, split_type='random', sizes=[0.8, 0.2], seed=123)
        assert indices1[0] != indices2[0]

    def test_stratified_split(self):
        """Test stratified split."""
        targets = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  # 5 of each class
        indices = data_split_index(
            n_samples=10,
            targets=targets,
            split_type='stratified',
            sizes=[0.8, 0.2],
            seed=42
        )
        assert len(indices) == 2
        assert len(indices[0]) + len(indices[1]) == 10

    def test_three_way_split(self):
        """Test three-way split."""
        indices = data_split_index(
            n_samples=100,
            split_type='random',
            sizes=[0.6, 0.2, 0.2],
            seed=42
        )
        assert len(indices) == 3
        assert len(indices[0]) + len(indices[1]) + len(indices[2]) == 100

    def test_unsupported_split_type_raises(self):
        """Test unsupported split type raises error."""
        with pytest.raises(ValueError, match='not supported'):
            data_split_index(
                n_samples=100,
                split_type='unsupported_type',
                sizes=[0.8, 0.2],
                seed=42
            )

    def test_scaffold_split_requires_mols(self):
        """Test scaffold split needs mols parameter."""
        # This should work with mols provided
        mols = [Chem.MolFromSmiles('CCO')] * 10
        indices = data_split_index(
            n_samples=10,
            mols=mols,
            split_type='scaffold_random',
            sizes=[0.8, 0.2],
            seed=42
        )
        assert len(indices) == 2


# =============================================================================
# dataset_split Tests
# =============================================================================

class TestDatasetSplit:
    """Tests for dataset_split function."""

    def test_returns_list_of_datasets(self, mock_dataset):
        """Test that function returns a list."""
        with patch('mgktools.data.split.data_split_index', return_value=[[0, 1, 2], [3, 4]]):
            with patch('mgktools.data.split.get_data_from_index') as mock_get:
                # Mock get_data_from_index to return mock datasets
                mock_get.return_value = Mock()
                result = dataset_split(
                    mock_dataset,
                    split_type='random',
                    sizes=[0.6, 0.4],
                    seed=42
                )
                assert len(result) == 2

    def test_random_split_no_mols_needed(self, mock_dataset):
        """Test random split doesn't require mols extraction."""
        with patch('mgktools.data.split.data_split_index') as mock_split:
            mock_split.return_value = [[0, 1, 2], [3, 4]]
            with patch('mgktools.data.split.get_data_from_index') as mock_get:
                mock_get.return_value = Mock()
                dataset_split(
                    mock_dataset,
                    split_type='random',
                    sizes=[0.6, 0.4],
                    seed=42
                )
                # Check mols parameter is None for random split
                call_kwargs = mock_split.call_args[1]
                assert call_kwargs['mols'] is None

    def test_scaffold_split_extracts_mols(self, mock_dataset):
        """Test scaffold split extracts mols from dataset."""
        with patch('mgktools.data.split.data_split_index') as mock_split:
            mock_split.return_value = [[0, 1, 2], [3, 4]]
            with patch('mgktools.data.split.get_data_from_index') as mock_get:
                mock_get.return_value = Mock()
                dataset_split(
                    mock_dataset,
                    split_type='scaffold_random',
                    sizes=[0.6, 0.4],
                    seed=42
                )
                # Check mols parameter is provided for scaffold split
                call_kwargs = mock_split.call_args[1]
                assert call_kwargs['mols'] is not None


# =============================================================================
# Edge Cases and Error Handling Tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_split_single_sample(self):
        """Test splitting single sample dataset."""
        indices = data_split_index(
            n_samples=1,
            split_type='random',
            sizes=[1.0],
            seed=42
        )
        assert len(indices) == 1
        assert indices[0] == [0]

    def test_split_two_samples_equal(self):
        """Test splitting two samples equally."""
        indices = data_split_index(
            n_samples=2,
            split_type='random',
            sizes=[0.5, 0.5],
            seed=42
        )
        assert len(indices[0]) == 1
        assert len(indices[1]) == 1

    def test_all_same_scaffold(self):
        """Test all molecules with same scaffold."""
        mols = [Chem.MolFromSmiles(s) for s in ['c1ccccc1', 'Cc1ccccc1', 'CCc1ccccc1']]
        result = scaffold_to_smiles(mols, use_indices=True)

        # All should have benzene scaffold
        assert 'c1ccccc1' in result
        assert len(result['c1ccccc1']) == 3

    def test_init_al_split(self):
        """Test init_al split for active learning."""
        targets = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  # 5 of each class
        indices = data_split_index(
            n_samples=10,
            targets=targets,
            split_type='init_al',
            sizes=[0.4, 0.6],
            n_samples_per_class=2,
            seed=42
        )
        assert len(indices) == 2
        # Training set should have 2 per class = 4 total
        assert len(indices[0]) == 4
        # Remaining 6 in test
        assert len(indices[1]) == 6


# =============================================================================
# Determinism and Reproducibility Tests
# =============================================================================

class TestReproducibility:
    """Tests for deterministic behavior."""

    def test_scaffold_split_deterministic(self):
        """Test scaffold split is deterministic."""
        mols = [Chem.MolFromSmiles(s) for s in ['CCO', 'CCCO', 'c1ccccc1', 'Cc1ccccc1']]

        result1 = scaffold_to_smiles(mols, use_indices=True)
        result2 = scaffold_to_smiles(mols, use_indices=True)

        assert result1 == result2

    def test_generate_scaffold_deterministic(self):
        """Test generate_scaffold is deterministic."""
        smiles = 'CC(C)(C)c1ccc(cc1)C(O)c2ccccc2'

        scaffold1 = generate_scaffold(smiles)
        scaffold2 = generate_scaffold(smiles)

        assert scaffold1 == scaffold2

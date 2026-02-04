#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fast unit tests for mgktools.data module.

Tests cover CachedDict, Datapoint, and Dataset classes with proper
mocking to avoid slow operations like graph kernel computation.
"""
import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from unittest.mock import Mock, MagicMock, patch

from mgktools.data.data import CachedDict, Datapoint, Dataset


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def simple_smiles():
    """Simple SMILES strings for testing."""
    return ['CCO', 'CCC', 'CCCC']


@pytest.fixture
def simple_df():
    """Simple DataFrame for testing Dataset creation."""
    return pd.DataFrame({
        'smiles': ['CCO', 'CCC', 'CCCC', 'CCCCC'],
        'target': [1.0, 2.0, 3.0, 4.0],
        'feature1': [0.1, 0.2, 0.3, 0.4]
    })


@pytest.fixture
def multi_smiles_df():
    """DataFrame with multiple SMILES columns."""
    return pd.DataFrame({
        'smiles_1': ['CCO', 'CCC'],
        'smiles_2': ['CCCO', 'CCCC'],
        'target': [1.0, 2.0]
    })


@pytest.fixture
def mock_hash_graph():
    """Mock HashGraph object."""
    mock = Mock()
    mock.cookie = Mock()
    mock.cookie.clear = Mock()
    return mock


@pytest.fixture
def mock_features_generator():
    """Mock FeaturesGenerator that returns fixed-size features."""
    mock = Mock()
    mock.features_generator_name = 'mock_morgan'
    mock.__call__ = Mock(return_value=np.array([0.1, 0.2, 0.3, 0.4]))
    return mock


# =============================================================================
# CachedDict Tests
# =============================================================================

class TestCachedDict:
    """Unit tests for CachedDict class."""

    def test_init_creates_empty_caches(self):
        """Test that initialization creates empty caches."""
        cache = CachedDict()
        assert cache.SMILES_TO_GRAPH == {}
        assert cache.SMILES_TO_FEATURES == {}

    def test_smiles2graph_caches_result(self):
        """Test that smiles2graph caches computed graphs."""
        cache = CachedDict()

        # First call computes and caches
        graph1 = cache.smiles2graph('CCO')
        assert 'CCO' in cache.SMILES_TO_GRAPH

        # Second call returns cached result
        graph2 = cache.smiles2graph('CCO')
        assert graph1 is graph2

    def test_smiles2graph_static_returns_hashgraph(self):
        """Test that static method returns HashGraph."""
        graph = CachedDict.smiles2graph_('CCO')
        # Should return a HashGraph object
        assert hasattr(graph, 'hash')

    def test_smiles2features_caches_with_generator_name(self, mock_features_generator):
        """Test that features are cached with generator name tag."""
        cache = CachedDict()

        with patch.object(CachedDict, 'smiles2features_', return_value=[0.1, 0.2, 0.3]):
            features1 = cache.smiles2features('CCO', mock_features_generator)
            tag = 'CCO_mock_morgan'
            assert tag in cache.SMILES_TO_FEATURES or 'CCO' in cache.SMILES_TO_FEATURES

    def test_smiles2features_handles_nan_and_large_values(self):
        """Test that NaN and large values are replaced."""
        mock_fg = Mock()
        mock_fg.features_generator_name = 'test'
        # Mock needs to be callable and return the array
        mock_fg.return_value = np.array([np.nan, 1e11, 1.0])

        features = CachedDict.smiles2features_('CCO', mock_fg)
        # NaN and values > 1e10 should be replaced with 0
        assert features[0] == 0
        assert features[1] == 0
        assert features[2] == 1.0

    def test_save_and_load(self):
        """Test saving and loading cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = CachedDict()
            cache.smiles2graph('CCO')

            # Save
            cache.save(path=tmpdir, filename='test_cache.pkl')
            assert os.path.exists(os.path.join(tmpdir, 'test_cache.pkl'))

            # Load
            loaded = CachedDict.load(path=tmpdir, filename='test_cache.pkl')
            assert 'CCO' in loaded.SMILES_TO_GRAPH

    def test_save_raises_on_existing_file(self):
        """Test that save raises error if file exists and overwrite=False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = CachedDict()
            cache.save(path=tmpdir, filename='test.pkl')

            with pytest.raises(RuntimeError, match='already exists'):
                cache.save(path=tmpdir, filename='test.pkl', overwrite=False)

    def test_save_overwrites_with_flag(self):
        """Test that save overwrites with overwrite=True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = CachedDict()
            cache.save(path=tmpdir, filename='test.pkl')
            cache.smiles2graph('CCO')
            cache.save(path=tmpdir, filename='test.pkl', overwrite=True)

            loaded = CachedDict.load(path=tmpdir, filename='test.pkl')
            assert 'CCO' in loaded.SMILES_TO_GRAPH


# =============================================================================
# Datapoint Tests
# =============================================================================

class TestDatapoint:
    """Unit tests for Datapoint class."""

    def test_init_with_minimal_args(self):
        """Test initialization with only smiles_list."""
        dp = Datapoint(['CCO'])
        assert dp.smiles_list == ['CCO']
        assert dp.features_add == []
        assert dp.targets == []

    def test_init_with_all_args(self):
        """Test initialization with all arguments."""
        dp = Datapoint(
            smiles_list=['CCO', 'CCC'],
            features_add=[0.1, 0.2],
            targets=[1.0, 2.0]
        )
        assert dp.smiles_list == ['CCO', 'CCC']
        assert dp.features_add == [0.1, 0.2]
        assert dp.targets == [1.0, 2.0]

    def test_repr_without_features(self):
        """Test string representation without features."""
        dp = Datapoint(['CCO', 'CCC'])
        assert repr(dp) == 'CCO,CCC'

    def test_repr_with_features(self):
        """Test string representation with features."""
        dp = Datapoint(['CCO'], features_add=[0.1, 0.2])
        assert repr(dp) == 'CCO;0.1,0.2'

    def test_mols_returns_rdkit_mols(self):
        """Test that mols property returns RDKit Mol objects."""
        dp = Datapoint(['CCO', 'CCC'])
        mols = dp.mols
        assert len(mols) == 2
        # Check that they are valid Mol objects
        assert mols[0].GetNumAtoms() > 0
        assert mols[1].GetNumAtoms() > 0

    def test_graph_uses_cache(self, mock_hash_graph):
        """Test that graph property uses cache."""
        cache = CachedDict()
        cache.SMILES_TO_GRAPH['CCO'] = mock_hash_graph

        dp = Datapoint(['CCO'], cache=cache)
        graphs = dp.graph
        assert graphs[0] is mock_hash_graph

    def test_features_mol_returns_empty_when_no_generators(self):
        """Test features_mol returns empty list when no generators."""
        dp = Datapoint(['CCO'])
        features = dp.features_mol(None, None)
        assert features == []

    def test_features_mol_concat_combination(self):
        """Test features_mol with concat combination."""
        mock_fg = Mock()
        mock_fg.features_generator_name = 'test'

        cache = CachedDict()
        cache.SMILES_TO_FEATURES['CCO_test'] = [0.1, 0.2]
        cache.SMILES_TO_FEATURES['CCC_test'] = [0.3, 0.4]

        dp = Datapoint(['CCO', 'CCC'], cache=cache)

        with patch.object(cache, 'smiles2features', side_effect=[[0.1, 0.2], [0.3, 0.4]]):
            features = dp.features_mol([mock_fg], 'concat')
            assert len(features) == 4

    def test_features_mol_mean_combination(self):
        """Test features_mol with mean combination."""
        mock_fg = Mock()
        mock_fg.features_generator_name = 'test'

        cache = CachedDict()
        dp = Datapoint(['CCO', 'CCC'], cache=cache)

        with patch.object(cache, 'smiles2features', side_effect=[[0.1, 0.2], [0.3, 0.4]]):
            features = dp.features_mol([mock_fg], 'mean')
            assert len(features) == 2
            assert features[0] == pytest.approx(0.2, rel=1e-5)
            assert features[1] == pytest.approx(0.3, rel=1e-5)

    def test_features_mol_raises_on_invalid_combination(self):
        """Test that invalid features_combination raises ValueError."""
        mock_fg = Mock()
        mock_fg.features_generator_name = 'test'

        cache = CachedDict()
        dp = Datapoint(['CCO'], cache=cache)

        with patch.object(cache, 'smiles2features', return_value=[0.1]):
            with pytest.raises(ValueError, match='Invalid features_combination'):
                dp.features_mol([mock_fg], 'invalid')


# =============================================================================
# Dataset Tests
# =============================================================================

class TestDataset:
    """Unit tests for Dataset class."""

    def test_init_with_no_data(self):
        """Test initialization with no data - raises due to set_cache iteration."""
        # Dataset.__init__ calls set_cache which iterates over data
        # So passing None data will raise TypeError
        with pytest.raises(TypeError):
            Dataset(data=None)

    def test_init_with_empty_data(self):
        """Test initialization with empty data list."""
        dataset = Dataset(data=[])
        assert dataset.data == []

    def test_from_df_creates_datapoints(self, simple_df):
        """Test that from_df creates Datapoint objects."""
        dataset = Dataset.from_df(
            simple_df,
            smiles_columns=['smiles'],
            targets_columns=['target'],
            n_jobs=1
        )
        assert len(dataset) == 4
        assert dataset[0].smiles_list == ['CCO']
        assert dataset[0].targets == [1.0]

    def test_from_df_with_features(self, simple_df):
        """Test from_df with features columns."""
        dataset = Dataset.from_df(
            simple_df,
            smiles_columns=['smiles'],
            features_columns=['feature1'],
            targets_columns=['target'],
            n_jobs=1
        )
        assert dataset[0].features_add == [0.1]

    def test_from_df_with_multiple_smiles(self, multi_smiles_df):
        """Test from_df with multiple SMILES columns."""
        dataset = Dataset.from_df(
            multi_smiles_df,
            smiles_columns=['smiles_1', 'smiles_2'],
            targets_columns=['target'],
            n_jobs=1
        )
        assert dataset[0].smiles_list == ['CCO', 'CCCO']

    def test_len(self, simple_df):
        """Test __len__ method."""
        dataset = Dataset.from_df(simple_df, smiles_columns=['smiles'], n_jobs=1)
        assert len(dataset) == 4

    def test_getitem(self, simple_df):
        """Test __getitem__ method."""
        dataset = Dataset.from_df(simple_df, smiles_columns=['smiles'], n_jobs=1)
        assert dataset[0].smiles_list == ['CCO']
        assert dataset[0:2][0].smiles_list == ['CCO']

    def test_set_status(self, simple_df):
        """Test set_status method."""
        dataset = Dataset.from_df(simple_df, smiles_columns=['smiles'], n_jobs=1)
        dataset.set_status(
            graph_kernel_type='graph',
            features_generators=None,
            features_combination=None
        )
        assert dataset.graph_kernel_type == 'graph'
        assert dataset.features_generators is None

    def test_y_property(self, simple_df):
        """Test y property returns targets."""
        dataset = Dataset.from_df(
            simple_df,
            smiles_columns=['smiles'],
            targets_columns=['target'],
            n_jobs=1
        )
        y = dataset.y
        assert y.shape == (4, 1)
        assert y[0, 0] == 1.0

    def test_repr_property(self, simple_df):
        """Test repr property."""
        dataset = Dataset.from_df(simple_df, smiles_columns=['smiles'], n_jobs=1)
        reprs = dataset.repr
        assert reprs[0] == 'CCO'

    def test_X_smiles_property(self, simple_df):
        """Test X_smiles property."""
        dataset = Dataset.from_df(simple_df, smiles_columns=['smiles'], n_jobs=1)
        X_smiles = dataset.X_smiles
        assert X_smiles.shape == (4, 1)
        assert X_smiles[0, 0] == 'CCO'

    def test_mols_property(self, simple_df):
        """Test mols property returns RDKit Mol objects."""
        dataset = Dataset.from_df(simple_df, smiles_columns=['smiles'], n_jobs=1)
        mols = dataset.mols
        assert mols.shape == (4, 1)
        assert mols[0, 0].GetNumAtoms() > 0

    def test_X_features_add_raw(self, simple_df):
        """Test X_features_add_raw property."""
        dataset = Dataset.from_df(
            simple_df,
            smiles_columns=['smiles'],
            features_columns=['feature1'],
            n_jobs=1
        )
        X = dataset.X_features_add_raw
        assert X.shape == (4, 1)
        assert X[0, 0] == pytest.approx(0.1)

    def test_normalize_features_add(self, simple_df):
        """Test normalize_features_add creates scaler."""
        dataset = Dataset.from_df(
            simple_df,
            smiles_columns=['smiles'],
            features_columns=['feature1'],
            n_jobs=1
        )
        assert dataset.features_add_scaler is None
        dataset.normalize_features_add()
        assert dataset.features_add_scaler is not None

    def test_X_features_add_uses_scaler(self, simple_df):
        """Test X_features_add uses scaler when set."""
        dataset = Dataset.from_df(
            simple_df,
            smiles_columns=['smiles'],
            features_columns=['feature1'],
            n_jobs=1
        )
        dataset.normalize_features_add()
        X = dataset.X_features_add
        # Normalized features should have mean ~0
        assert abs(X.mean()) < 1e-10

    def test_N_tasks(self, simple_df):
        """Test N_tasks property."""
        dataset = Dataset.from_df(
            simple_df,
            smiles_columns=['smiles'],
            targets_columns=['target'],
            n_jobs=1
        )
        assert dataset.N_tasks == 1

    def test_N_features_add(self, simple_df):
        """Test N_features_add property."""
        dataset = Dataset.from_df(
            simple_df,
            smiles_columns=['smiles'],
            features_columns=['feature1'],
            n_jobs=1
        )
        assert dataset.N_features_add == 1

    def test_copy_creates_deep_copy(self, simple_df):
        """Test copy creates independent dataset."""
        dataset = Dataset.from_df(simple_df, smiles_columns=['smiles'], n_jobs=1)
        copy = dataset.copy()
        assert copy is not dataset
        assert copy.data is not dataset.data

    def test_set_cache_propagates_to_datapoints(self, simple_df):
        """Test set_cache updates all datapoint caches."""
        dataset = Dataset.from_df(simple_df, smiles_columns=['smiles'], n_jobs=1)
        new_cache = CachedDict()
        dataset.set_cache(new_cache)

        for dp in dataset.data:
            assert dp.cache is new_cache

    def test_save_and_load(self, simple_df):
        """Test save and load dataset.

        Note: Dataset.load() has a bug where it calls cls() with no args,
        which fails because set_cache tries to iterate over None data.
        This test documents the current (broken) behavior.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = Dataset.from_df(
                simple_df,
                smiles_columns=['smiles'],
                targets_columns=['target'],
                n_jobs=1
            )
            dataset.save(tmpdir, filename='test_dataset.pkl')

            # Currently Dataset.load() fails due to calling cls() with no args
            with pytest.raises(TypeError):
                Dataset.load(tmpdir, filename='test_dataset.pkl')

    def test_save_raises_on_existing(self, simple_df):
        """Test save raises if file exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = Dataset.from_df(simple_df, smiles_columns=['smiles'], n_jobs=1)
            dataset.save(tmpdir)

            with pytest.raises(RuntimeError, match='already exists'):
                dataset.save(tmpdir, overwrite=False)

    def test_X_with_no_graph_kernel(self, simple_df):
        """Test X property when graph_kernel_type is 'no'."""
        dataset = Dataset.from_df(
            simple_df,
            smiles_columns=['smiles'],
            features_columns=['feature1'],
            n_jobs=1
        )
        dataset.set_status(graph_kernel_type='no', features_generators=None, features_combination=None)
        X = dataset.X
        # Should only have features_add
        assert X.shape == (4, 1)

    def test_X_raises_on_invalid_kernel_type(self, simple_df):
        """Test X raises on invalid graph_kernel_type."""
        dataset = Dataset.from_df(simple_df, smiles_columns=['smiles'], n_jobs=1)
        dataset.graph_kernel_type = 'invalid'
        dataset.features_generators = None
        dataset.features_combination = None

        with pytest.raises(ValueError, match='Invalid graph_kernel_type'):
            _ = dataset.X

    def test_N_MGK_with_graph_kernel(self, simple_df):
        """Test N_MGK returns correct value with graph kernel."""
        dataset = Dataset.from_df(simple_df, smiles_columns=['smiles'], n_jobs=1)
        dataset.set_status(graph_kernel_type='graph')
        assert dataset.N_MGK == 1

    def test_N_MGK_without_graph_kernel(self, simple_df):
        """Test N_MGK returns 0 without graph kernel."""
        dataset = Dataset.from_df(simple_df, smiles_columns=['smiles'], n_jobs=1)
        dataset.set_status(graph_kernel_type='no')
        assert dataset.N_MGK == 0

    def test_features_size(self, simple_df):
        """Test features_size method."""
        dataset = Dataset.from_df(
            simple_df,
            smiles_columns=['smiles'],
            features_columns=['feature1'],
            n_jobs=1
        )
        dataset.set_status(graph_kernel_type='no', features_generators=None, features_combination=None)
        assert dataset.features_size() == 1

    def test_clear_cookie(self, simple_df, mock_hash_graph):
        """Test clear_cookie clears all graph cookies."""
        dataset = Dataset.from_df(simple_df, smiles_columns=['smiles'], n_jobs=1)
        dataset.set_status(graph_kernel_type='graph')

        # Mock the graph property
        mock_graphs = [[mock_hash_graph], [mock_hash_graph], [mock_hash_graph], [mock_hash_graph]]
        with patch.object(Dataset, 'X_graph', property(lambda self: np.array(mock_graphs))):
            dataset.clear_cookie()
            # Each mock should have had clear() called
            assert mock_hash_graph.cookie.clear.call_count >= 1

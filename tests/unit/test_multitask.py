#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unit tests for multi-task learning support.

Tests cover NaN handling in models, Evaluator weighted averaging,
NaN-aware metric computation, LOOCV with NaN targets, and gradient
optimization guard.
"""
import pytest
import numpy as np
import pandas as pd
import tempfile
import shutil
from unittest.mock import Mock, patch


# =============================================================================
# SVRegressor NaN handling
# =============================================================================

class TestSVRegressorNaN:
    """Tests for SVRegressor NaN filtering in fit()."""

    def test_fit_filters_nan_from_y(self):
        from mgktools.models.regression.svr import SVRegressor
        model = SVRegressor(kernel='rbf', C=1.0)
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([1.0, np.nan, 3.0, 4.0])
        model.fit(X, y)
        # Model should have been trained on 3 valid samples
        assert model.support_vectors_.shape[0] <= 3

    def test_fit_all_valid(self):
        from mgktools.models.regression.svr import SVRegressor
        model = SVRegressor(kernel='rbf', C=1.0)
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([1.0, 2.0, 3.0])
        model.fit(X, y)
        assert model.support_vectors_.shape[0] <= 3

    def test_remove_nan_X_y_static(self):
        from mgktools.models.regression.svr import SVRegressor
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([1.0, np.nan, 3.0])
        X_, y_ = SVRegressor._remove_nan_X_y(X, y)
        assert len(X_) == 2
        assert len(y_) == 2
        np.testing.assert_array_equal(y_, [1.0, 3.0])

    def test_remove_nan_X_y_none_values(self):
        from mgktools.models.regression.svr import SVRegressor
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([1.0, None, 3.0], dtype=object)
        X_, y_ = SVRegressor._remove_nan_X_y(X, y)
        assert len(X_) == 2


# =============================================================================
# EnsembleRegressor NaN handling
# =============================================================================

class TestEnsembleRegressorNaN:
    """Tests for EnsembleRegressor NaN filtering in fit()."""

    def test_fit_filters_nan_rows(self):
        from mgktools.models.regression.consensus import EnsembleRegressor
        # Use a mock base model to avoid graphdot dependency
        mock_model = Mock()
        mock_model.fit = Mock()

        ensemble = EnsembleRegressor(
            model=mock_model,
            n_estimators=2,
            n_samples_per_model=2,
            n_jobs=1
        )

        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([1.0, np.nan, 3.0, np.nan, 5.0])

        ensemble.fit(X, y)

        # Each model should have been fitted
        assert len(ensemble.models) == 2
        # The n_samples_per_model should be restored
        assert ensemble.n_samples_per_model == 2

    def test_fit_clamps_n_samples(self):
        from mgktools.models.regression.consensus import EnsembleRegressor
        mock_model = Mock()
        mock_model.fit = Mock()

        ensemble = EnsembleRegressor(
            model=mock_model,
            n_estimators=1,
            n_samples_per_model=100,  # More than available valid samples
            n_jobs=1
        )

        X = np.array([[1], [2], [3]])
        y = np.array([1.0, np.nan, 3.0])

        ensemble.fit(X, y)
        assert len(ensemble.models) == 1


# =============================================================================
# NaiveLocalExpertGP NaN handling
# =============================================================================

class TestNaiveLocalExpertGPNaN:
    """Tests for NaiveLocalExpertGP NaN filtering in fit()."""

    def test_fit_filters_nan(self):
        from mgktools.models.regression.scalable.NLE import NaiveLocalExpertGP
        mock_kernel = Mock()
        model = NaiveLocalExpertGP(kernel=mock_kernel, alpha=1e-8, n_local=5)

        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([1.0, np.nan, 3.0, np.nan])

        model.fit(X, y)
        assert len(model.X) == 2
        assert len(model.y) == 2
        np.testing.assert_array_equal(model.y, [1.0, 3.0])

    def test_fit_all_valid(self):
        from mgktools.models.regression.scalable.NLE import NaiveLocalExpertGP
        mock_kernel = Mock()
        model = NaiveLocalExpertGP(kernel=mock_kernel, alpha=1e-8, n_local=5)

        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([1.0, 2.0, 3.0])

        model.fit(X, y)
        assert len(model.X) == 3
        assert len(model.y) == 3

    def test_fit_integer_y_no_filter(self):
        from mgktools.models.regression.scalable.NLE import NaiveLocalExpertGP
        mock_kernel = Mock()
        model = NaiveLocalExpertGP(kernel=mock_kernel, alpha=1e-8, n_local=5)

        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([1, 2, 3])

        model.fit(X, y)
        assert len(model.X) == 3
        assert len(model.y) == 3


# =============================================================================
# Evaluator._weighted_mean()
# =============================================================================

class TestWeightedMean:
    """Tests for Evaluator._weighted_mean static method."""

    def test_basic_weighted_average(self):
        from mgktools.evaluators.cross_validation import Evaluator
        df = pd.DataFrame({
            'value': [0.8, 0.9],
            'n_samples': [10, 20]
        })
        result = Evaluator._weighted_mean(df)
        expected = (0.8 * 10 + 0.9 * 20) / 30
        assert abs(result - expected) < 1e-10

    def test_all_nan_values(self):
        from mgktools.evaluators.cross_validation import Evaluator
        df = pd.DataFrame({
            'value': [np.nan, np.nan],
            'n_samples': [10, 20]
        })
        result = Evaluator._weighted_mean(df)
        assert np.isnan(result)

    def test_zero_n_samples(self):
        from mgktools.evaluators.cross_validation import Evaluator
        df = pd.DataFrame({
            'value': [0.8, 0.9],
            'n_samples': [0, 0]
        })
        result = Evaluator._weighted_mean(df)
        assert np.isnan(result)

    def test_unequal_weights(self):
        from mgktools.evaluators.cross_validation import Evaluator
        df = pd.DataFrame({
            'value': [1.0, 0.0],
            'n_samples': [100, 1]
        })
        result = Evaluator._weighted_mean(df)
        expected = (1.0 * 100 + 0.0 * 1) / 101
        assert abs(result - expected) < 1e-10

    def test_fallback_without_n_samples_column(self):
        from mgktools.evaluators.cross_validation import Evaluator
        df = pd.DataFrame({
            'value': [0.8, 0.9, 1.0]
        })
        result = Evaluator._weighted_mean(df)
        expected = (0.8 + 0.9 + 1.0) / 3
        assert abs(result - expected) < 1e-10

    def test_mixed_nan_and_valid(self):
        from mgktools.evaluators.cross_validation import Evaluator
        df = pd.DataFrame({
            'value': [0.8, np.nan, 0.9],
            'n_samples': [10, 5, 20]
        })
        result = Evaluator._weighted_mean(df)
        expected = (0.8 * 10 + 0.9 * 20) / 30
        assert abs(result - expected) < 1e-10


# =============================================================================
# Evaluator.evaluate_train_test() with NaN targets
# =============================================================================

class TestEvaluateTrainTestNaN:
    """Tests for NaN-aware metric computation in evaluate_train_test."""

    @pytest.fixture
    def temp_dir(self):
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
        shutil.rmtree(tmpdir, ignore_errors=True)

    def _make_evaluator(self, mock_model, mock_dataset, temp_dir, task_type='regression'):
        from mgktools.evaluators.cross_validation import Evaluator
        return Evaluator(
            save_dir=temp_dir,
            dataset=mock_dataset,
            model=mock_model,
            task_type=task_type,
            metrics=['rmse'],
            cross_validation='Monte-Carlo',
            split_type='random',
            split_sizes=[0.8, 0.2],
            verbose=False
        )

    def test_n_samples_column_present(self, temp_dir):
        from mgktools.evaluators.cross_validation import Evaluator
        from mgktools.data.data import Dataset

        mock_train = Mock(spec=Dataset)
        mock_train.X = np.array([[1, 2], [3, 4], [5, 6]])
        mock_train.y = np.array([[1.0], [2.0], [3.0]])
        mock_train.N_tasks = 1

        mock_test = Mock(spec=Dataset)
        mock_test.X = np.array([[7, 8], [9, 10]])
        mock_test.y = np.array([[4.0], [5.0]])
        mock_test.N_tasks = 1
        mock_test.repr = ['test1', 'test2']

        def mock_predict(X, return_std=False):
            if return_std:
                return np.array([3.9, 4.8]), np.array([0.1, 0.1])
            return np.array([3.9, 4.8])

        mock_model = Mock()
        mock_model.predict = mock_predict
        mock_model.fit = Mock()

        evaluator = self._make_evaluator(mock_model, mock_train, temp_dir)
        df_predict, df_metrics = evaluator.evaluate_train_test(mock_train, mock_test, loocv=False)

        assert 'n_samples' in df_metrics.columns
        assert df_metrics['n_samples'].iloc[0] == 2

    def test_nan_targets_filtered_before_metrics(self, temp_dir):
        from mgktools.evaluators.cross_validation import Evaluator
        from mgktools.data.data import Dataset

        mock_train = Mock(spec=Dataset)
        mock_train.X = np.array([[1, 2], [3, 4], [5, 6]])
        mock_train.y = np.array([[1.0], [2.0], [3.0]])
        mock_train.N_tasks = 1

        mock_test = Mock(spec=Dataset)
        mock_test.X = np.array([[7, 8], [9, 10], [11, 12]])
        mock_test.y = np.array([[4.0], [np.nan], [6.0]])
        mock_test.N_tasks = 1
        mock_test.repr = ['test1', 'test2', 'test3']

        def mock_predict(X, return_std=False):
            if return_std:
                return np.array([3.9, 4.8, 5.9]), np.array([0.1, 0.1, 0.1])
            return np.array([3.9, 4.8, 5.9])

        mock_model = Mock()
        mock_model.predict = mock_predict
        mock_model.fit = Mock()

        evaluator = self._make_evaluator(mock_model, mock_train, temp_dir)
        df_predict, df_metrics = evaluator.evaluate_train_test(mock_train, mock_test, loocv=False)

        # Only 2 valid samples (NaN at index 1 filtered out)
        assert df_metrics['n_samples'].iloc[0] == 2

    def test_all_nan_targets_returns_nan_metric(self, temp_dir):
        from mgktools.evaluators.cross_validation import Evaluator
        from mgktools.data.data import Dataset

        mock_train = Mock(spec=Dataset)
        mock_train.X = np.array([[1, 2], [3, 4]])
        mock_train.y = np.array([[1.0], [2.0]])
        mock_train.N_tasks = 1

        mock_test = Mock(spec=Dataset)
        mock_test.X = np.array([[5, 6], [7, 8]])
        mock_test.y = np.array([[np.nan], [np.nan]])
        mock_test.N_tasks = 1
        mock_test.repr = ['test1', 'test2']

        def mock_predict(X, return_std=False):
            if return_std:
                return np.array([2.5, 3.5]), np.array([0.1, 0.1])
            return np.array([2.5, 3.5])

        mock_model = Mock()
        mock_model.predict = mock_predict
        mock_model.fit = Mock()

        evaluator = self._make_evaluator(mock_model, mock_train, temp_dir)
        df_predict, df_metrics = evaluator.evaluate_train_test(mock_train, mock_test, loocv=False)

        assert df_metrics['n_samples'].iloc[0] == 0
        assert np.isnan(df_metrics['value'].iloc[0])


# =============================================================================
# Evaluator LOOCV with NaN
# =============================================================================

class TestLOOCVNaN:
    """Tests for LOOCV with NaN targets."""

    @pytest.fixture
    def temp_dir(self):
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_loocv_nan_reconstructs_full_arrays(self, temp_dir):
        from mgktools.evaluators.cross_validation import Evaluator
        from mgktools.data.data import Dataset

        mock_dataset = Mock(spec=Dataset)
        mock_dataset.X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        mock_dataset.y = np.array([[1.0], [np.nan], [3.0], [4.0]])
        mock_dataset.N_tasks = 1
        mock_dataset.repr = ['mol1', 'mol2', 'mol3', 'mol4']

        # predict_loocv returns results only for valid (non-NaN) rows
        mock_model = Mock()
        mock_model.predict_loocv = Mock(
            return_value=(np.array([1.1, 3.1, 4.1]), np.array([0.1, 0.1, 0.1]))
        )

        evaluator = Evaluator(
            save_dir=temp_dir,
            dataset=mock_dataset,
            model=mock_model,
            task_type='regression',
            metrics=['rmse'],
            cross_validation='leave-one-out',
            verbose=False
        )

        df_predict, df_metrics = evaluator.evaluate_train_test(
            mock_dataset, mock_dataset, loocv=True
        )

        # Should have full-length arrays (4 samples, with NaN at index 1)
        assert len(df_predict) == 4
        assert np.isnan(df_predict['predict_0'].iloc[1])
        assert np.isnan(df_predict['uncertainty_0'].iloc[1])
        # Valid predictions should be placed correctly
        assert abs(df_predict['predict_0'].iloc[0] - 1.1) < 1e-10

    def test_loocv_no_nan_passes_through(self, temp_dir):
        from mgktools.evaluators.cross_validation import Evaluator
        from mgktools.data.data import Dataset

        mock_dataset = Mock(spec=Dataset)
        mock_dataset.X = np.array([[1, 2], [3, 4], [5, 6]])
        mock_dataset.y = np.array([[1.0], [2.0], [3.0]])
        mock_dataset.N_tasks = 1
        mock_dataset.repr = ['mol1', 'mol2', 'mol3']

        mock_model = Mock()
        mock_model.predict_loocv = Mock(
            return_value=(np.array([1.1, 2.1, 3.1]), np.array([0.1, 0.1, 0.1]))
        )

        evaluator = Evaluator(
            save_dir=temp_dir,
            dataset=mock_dataset,
            model=mock_model,
            task_type='regression',
            metrics=['rmse'],
            cross_validation='leave-one-out',
            verbose=False
        )

        df_predict, df_metrics = evaluator.evaluate_train_test(
            mock_dataset, mock_dataset, loocv=True
        )

        assert len(df_predict) == 3
        assert not np.any(np.isnan(df_predict['predict_0'].values))


# =============================================================================
# Primary metric filtering
# =============================================================================

class TestPrimaryMetricFiltering:
    """Tests for filtering to primary metric when multiple metrics are used."""

    @pytest.fixture
    def temp_dir(self):
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_returns_only_primary_metric_value(self, temp_dir):
        """Test that return value uses only primary metric, not all metrics."""
        from mgktools.evaluators.cross_validation import Evaluator
        from mgktools.data.data import Dataset

        mock_dataset = Mock(spec=Dataset)
        mock_dataset.X = np.array([[1, 2], [3, 4], [5, 6]])
        mock_dataset.y = np.array([[1.0], [2.0], [3.0]])
        mock_dataset.N_tasks = 1
        mock_dataset.repr = ['mol1', 'mol2', 'mol3']

        mock_model = Mock()
        mock_model.predict_loocv = Mock(
            return_value=(np.array([1.1, 2.1, 3.1]), np.array([0.1, 0.1, 0.1]))
        )

        # Use two metrics: rmse (primary) and mae (extra)
        evaluator = Evaluator(
            save_dir=temp_dir,
            dataset=mock_dataset,
            model=mock_model,
            task_type='regression',
            metrics=['rmse', 'mae'],  # rmse is primary
            cross_validation='leave-one-out',
            verbose=False
        )

        result = evaluator.eval_loocv()

        # Read the metrics CSV to verify both metrics are computed
        df_metrics = pd.read_csv(f"{temp_dir}/loocv_metrics.csv")
        assert 'rmse' in df_metrics['metric'].values
        assert 'mae' in df_metrics['metric'].values

        # The returned value should be only for rmse (primary), not averaged with mae
        rmse_value = df_metrics[df_metrics['metric'] == 'rmse']['value'].values[0]
        assert abs(result - rmse_value) < 1e-10


# =============================================================================
# Gradient optimization guard
# =============================================================================

class TestGradientOptGuard:
    """Tests for gradient optimization single-task guard."""

    def test_raises_for_multi_task(self):
        from mgktools.data.data import Dataset

        mock_dataset = Mock(spec=Dataset)
        mock_dataset.N_tasks = 2
        mock_dataset.y = np.array([[1.0, 2.0], [3.0, 4.0]])

        mock_args = Mock()
        mock_args.data_path = 'dummy.csv'
        mock_args.smiles_columns = ['smiles']
        mock_args.features_columns = None
        mock_args.targets_columns = ['t1', 't2']
        mock_args.n_jobs = 1
        mock_args.graph_kernel_type = 'graph'
        mock_args.features_generators = None
        mock_args.features_combination = None
        mock_args.cache_path = None
        mock_args.graph_hyperparameters = None
        mock_args.features_hyperparameters = None
        mock_args.save_dir = tempfile.mkdtemp()
        mock_args.optimizer = None
        mock_args.alpha_ = 0.1
        mock_args.loss = 'loocv'

        with patch('mgktools.exe.run.GradientOptArgs') as MockArgs, \
             patch('mgktools.exe.run.Dataset') as MockDataset, \
             patch('mgktools.exe.run.get_kernel_config') as MockKernel, \
             patch('mgktools.exe.run.set_model') as MockSetModel, \
             patch('mgktools.exe.run.CachedDict'), \
             patch('mgktools.exe.run.pd.read_csv', return_value=pd.DataFrame()):

            MockArgs.return_value.parse_args.return_value = mock_args
            MockDataset.from_df.return_value = mock_dataset
            mock_dataset.set_status = Mock()
            mock_dataset.unify_datatype = Mock()
            MockKernel.return_value = Mock()
            MockKernel.return_value.kernel = Mock()
            MockSetModel.return_value = Mock()

            from mgktools.exe.run import mgk_gradientopt
            with pytest.raises(ValueError, match="single-task"):
                mgk_gradientopt()


# =============================================================================
# Model factory wiring
# =============================================================================

class TestModelFactoryWiring:
    """Tests that SVRegressor is wired into the model factory."""

    def test_svr_returns_svregressor(self):
        from mgktools.models import set_model
        from mgktools.models.regression.svr import SVRegressor
        model = set_model(model_type='svr', kernel='rbf', C=1.0)
        assert isinstance(model, SVRegressor)

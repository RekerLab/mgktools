#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fast unit tests for mgktools.evaluators.cross_validation module.

Tests cover Evaluator class with proper mocking to avoid slow operations.
"""
import pytest
import numpy as np
import pandas as pd
import tempfile
import os
import shutil
from unittest.mock import Mock, MagicMock, patch, PropertyMock

from mgktools.evaluators.cross_validation import Evaluator
from mgktools.data.data import Dataset, Datapoint, CachedDict


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_dataset():
    """Create a mock dataset for testing."""
    mock = Mock(spec=Dataset)
    mock.X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
    mock.y = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
    mock.N_tasks = 1
    mock.repr = ['mol1', 'mol2', 'mol3', 'mol4', 'mol5', 'mol6']
    mock.mols = np.array([['mol1'], ['mol2'], ['mol3'], ['mol4'], ['mol5'], ['mol6']])
    return mock


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    mock = Mock()
    mock.fit = Mock()
    mock.predict = Mock(return_value=(np.array([1.5, 2.5, 3.5]), np.array([0.1, 0.1, 0.1])))
    mock.predict_loocv = Mock(return_value=(np.array([1.1, 2.1, 3.1, 4.1, 5.1, 6.1]), np.array([0.1]*6)))
    mock.predict_proba = Mock(return_value=np.array([0.8, 0.2, 0.9]))
    return mock


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir)


# =============================================================================
# Evaluator Initialization Tests
# =============================================================================

class TestEvaluatorInit:
    """Tests for Evaluator initialization."""

    def test_init_creates_save_dir(self, mock_dataset, mock_model, temp_dir):
        """Test that initialization creates save directory."""
        save_path = os.path.join(temp_dir, 'new_dir')
        evaluator = Evaluator(
            save_dir=save_path,
            dataset=mock_dataset,
            model=mock_model,
            task_type='regression',
            metrics=['rmse'],
            verbose=False
        )
        assert os.path.exists(save_path)

    def test_init_creates_logfile(self, mock_dataset, mock_model, temp_dir):
        """Test that initialization creates log file."""
        evaluator = Evaluator(
            save_dir=temp_dir,
            dataset=mock_dataset,
            model=mock_model,
            task_type='regression',
            metrics=['rmse'],
            verbose=False
        )
        assert os.path.exists(os.path.join(temp_dir, 'results.log'))

    def test_init_stores_parameters(self, mock_dataset, mock_model, temp_dir):
        """Test that initialization stores all parameters."""
        evaluator = Evaluator(
            save_dir=temp_dir,
            dataset=mock_dataset,
            model=mock_model,
            task_type='regression',
            metrics=['rmse', 'r2'],
            cross_validation='Monte-Carlo',
            split_type='random',
            split_sizes=[0.8, 0.2],
            num_folds=3,
            seed=42,
            verbose=False
        )
        assert evaluator.task_type == 'regression'
        assert evaluator.metrics == ['rmse', 'r2']
        assert evaluator.cross_validation == 'Monte-Carlo'
        assert evaluator.split_type == 'random'
        assert evaluator.split_sizes == [0.8, 0.2]
        assert evaluator.num_folds == 3
        assert evaluator.seed == 42

    def test_init_none_save_dir(self, mock_dataset, mock_model):
        """Test initialization with None save_dir."""
        evaluator = Evaluator(
            save_dir=None,
            dataset=mock_dataset,
            model=mock_model,
            task_type='regression',
            verbose=False
        )
        assert evaluator.write_file is False


# =============================================================================
# Evaluator Properties Tests
# =============================================================================

class TestEvaluatorProperties:
    """Tests for Evaluator properties."""

    def test_write_file_true_when_save_dir_set(self, mock_dataset, mock_model, temp_dir):
        """Test write_file is True when save_dir is set."""
        evaluator = Evaluator(
            save_dir=temp_dir,
            dataset=mock_dataset,
            model=mock_model,
            task_type='regression',
            verbose=False
        )
        assert evaluator.write_file is True

    def test_write_file_false_when_save_dir_none(self, mock_dataset, mock_model):
        """Test write_file is False when save_dir is None."""
        evaluator = Evaluator(
            save_dir=None,
            dataset=mock_dataset,
            model=mock_model,
            task_type='regression',
            verbose=False
        )
        assert evaluator.write_file is False


# =============================================================================
# Cross-Validation Tests
# =============================================================================

class TestCrossValidation:
    """Tests for cross-validation methods."""

    def test_loocv_assertions(self, mock_dataset, mock_model, temp_dir):
        """Test LOOCV parameter assertions."""
        evaluator = Evaluator(
            save_dir=temp_dir,
            dataset=mock_dataset,
            model=mock_model,
            task_type='regression',
            metrics=['rmse'],
            cross_validation='leave-one-out',
            n_splits=5,  # Should be None for LOOCV
            verbose=False
        )
        with pytest.raises(AssertionError, match='nfold must be None'):
            evaluator.run_cross_validation()

    def test_loocv_split_type_assertion(self, mock_dataset, mock_model, temp_dir):
        """Test LOOCV split_type must be None."""
        evaluator = Evaluator(
            save_dir=temp_dir,
            dataset=mock_dataset,
            model=mock_model,
            task_type='regression',
            metrics=['rmse'],
            cross_validation='leave-one-out',
            split_type='random',  # Should be None
            verbose=False
        )
        with pytest.raises(AssertionError, match='split_type must be None'):
            evaluator.run_cross_validation()

    def test_kfold_requires_n_splits(self, mock_dataset, mock_model, temp_dir):
        """Test k-fold requires n_splits parameter."""
        evaluator = Evaluator(
            save_dir=temp_dir,
            dataset=mock_dataset,
            model=mock_model,
            task_type='regression',
            metrics=['rmse'],
            cross_validation='kFold',
            n_splits=None,  # Required
            verbose=False
        )
        with pytest.raises(AssertionError, match='n_splits must be specified'):
            evaluator.run_cross_validation()

    def test_monte_carlo_requires_split_type(self, mock_dataset, mock_model, temp_dir):
        """Test Monte-Carlo requires split_type."""
        evaluator = Evaluator(
            save_dir=temp_dir,
            dataset=mock_dataset,
            model=mock_model,
            task_type='regression',
            metrics=['rmse'],
            cross_validation='Monte-Carlo',
            split_type=None,  # Required
            split_sizes=[0.8, 0.2],
            verbose=False
        )
        with pytest.raises(AssertionError, match='split_type must be specified'):
            evaluator.run_cross_validation()

    def test_monte_carlo_requires_split_sizes(self, mock_dataset, mock_model, temp_dir):
        """Test Monte-Carlo requires split_sizes."""
        evaluator = Evaluator(
            save_dir=temp_dir,
            dataset=mock_dataset,
            model=mock_model,
            task_type='regression',
            metrics=['rmse'],
            cross_validation='Monte-Carlo',
            split_type='random',
            split_sizes=None,  # Required
            verbose=False
        )
        with pytest.raises(AssertionError, match='split_sizes must be specified'):
            evaluator.run_cross_validation()

    def test_atomic_attribution_raises_for_cv(self, mock_dataset, mock_model, temp_dir):
        """Test atomic_attribution raises for cross-validation."""
        evaluator = Evaluator(
            save_dir=temp_dir,
            dataset=mock_dataset,
            model=mock_model,
            task_type='regression',
            metrics=['rmse'],
            cross_validation='Monte-Carlo',
            split_type='random',
            split_sizes=[0.8, 0.2],
            atomic_attribution=True,
            verbose=False
        )
        with pytest.raises(AssertionError, match='Atomic attribution'):
            evaluator.run_cross_validation()

    def test_unsupported_cv_method_raises(self, mock_dataset, mock_model, temp_dir):
        """Test unsupported cross-validation method raises."""
        evaluator = Evaluator(
            save_dir=temp_dir,
            dataset=mock_dataset,
            model=mock_model,
            task_type='regression',
            metrics=['rmse'],
            cross_validation='unsupported',
            verbose=False
        )
        with pytest.raises(ValueError, match='Unsupported cross-validation'):
            evaluator.run_cross_validation()

    def test_cv_no_raises_with_cv(self, mock_dataset, mock_model, temp_dir):
        """Test cross_validation='no' raises when run_cross_validation called."""
        evaluator = Evaluator(
            save_dir=temp_dir,
            dataset=mock_dataset,
            model=mock_model,
            task_type='regression',
            metrics=['rmse'],
            cross_validation='no',
            verbose=False
        )
        with pytest.raises(ValueError, match="use run_external"):
            evaluator.run_cross_validation()


# =============================================================================
# Fit Method Tests
# =============================================================================

class TestFitMethod:
    """Tests for fit method."""

    def test_fit_calls_model_fit(self, mock_dataset, mock_model, temp_dir):
        """Test fit calls model.fit with correct arguments."""
        evaluator = Evaluator(
            save_dir=temp_dir,
            dataset=mock_dataset,
            model=mock_model,
            task_type='regression',
            verbose=False
        )
        X = np.array([[1, 2], [3, 4]])
        y = np.array([1.0, 2.0])
        evaluator.fit(X, y)
        mock_model.fit.assert_called_once_with(X, y)

    def test_fit_with_n_core_uses_subset(self, mock_dataset, mock_model, temp_dir):
        """Test fit with n_core uses core set."""
        evaluator = Evaluator(
            save_dir=temp_dir,
            dataset=mock_dataset,
            model=mock_model,
            task_type='regression',
            n_core=2,
            verbose=False
        )
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([1.0, 2.0, 3.0, 4.0])

        # Fix random seed for reproducibility
        np.random.seed(42)
        evaluator.fit(X, y)

        # Model should be called with 3 arguments (C_train, X, y)
        assert mock_model.fit.call_count == 1
        call_args = mock_model.fit.call_args[0]
        assert len(call_args) == 3
        assert len(call_args[0]) == 2  # C_train has n_core samples


# =============================================================================
# Metric Evaluation Tests
# =============================================================================

class TestEvalMetric:
    """Tests for metric evaluation."""

    def test_eval_metric_regression(self, mock_dataset, mock_model, temp_dir):
        """Test eval_metric for regression task."""
        evaluator = Evaluator(
            save_dir=temp_dir,
            dataset=mock_dataset,
            model=mock_model,
            task_type='regression',
            verbose=False
        )
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 3.1])

        with patch('mgktools.evaluators.cross_validation.metric_regression', return_value=0.1) as mock_metric:
            result = evaluator.eval_metric(y_true, y_pred, 'rmse')
            mock_metric.assert_called_once_with(y_true, y_pred, 'rmse')
            assert result == 0.1

    def test_eval_metric_binary(self, mock_dataset, mock_model, temp_dir):
        """Test eval_metric for binary classification."""
        evaluator = Evaluator(
            save_dir=temp_dir,
            dataset=mock_dataset,
            model=mock_model,
            task_type='binary',
            verbose=False
        )
        y_true = np.array([0, 1, 1])
        y_pred = np.array([0.2, 0.8, 0.9])

        with patch('mgktools.evaluators.cross_validation.metric_binary', return_value=0.9) as mock_metric:
            result = evaluator.eval_metric(y_true, y_pred, 'roc_auc')
            mock_metric.assert_called_once_with(y_true, y_pred, 'roc_auc')
            assert result == 0.9

    def test_eval_metric_multiclass_raises(self, mock_dataset, mock_model, temp_dir):
        """Test eval_metric raises for multi-class."""
        evaluator = Evaluator(
            save_dir=temp_dir,
            dataset=mock_dataset,
            model=mock_model,
            task_type='multi-class',
            verbose=False
        )
        with pytest.raises(NotImplementedError):
            evaluator.eval_metric(np.array([0, 1, 2]), np.array([0, 1, 1]), 'accuracy')


# =============================================================================
# Get Similar Info Tests
# =============================================================================

class TestGetSimilarInfo:
    """Tests for get_similar_info method."""

    def test_get_similar_info_returns_correct_format(self, mock_dataset, mock_model, temp_dir):
        """Test get_similar_info returns properly formatted strings."""
        mock_kernel = Mock(return_value=np.array([[0.9, 0.5, 0.3], [0.4, 0.8, 0.6]]))

        evaluator = Evaluator(
            save_dir=temp_dir,
            dataset=mock_dataset,
            model=mock_model,
            task_type='regression',
            kernel=mock_kernel,
            verbose=False
        )

        X_test = np.array([[1, 2], [3, 4]])
        X_train = np.array([[5, 6], [7, 8], [9, 10]])
        X_repr = ['mol_a', 'mol_b', 'mol_c']

        result = evaluator.get_similar_info(X_test, X_train, X_repr, n_most_similar=2)

        assert len(result) == 2
        # Each result should contain SMILES and similarity values
        assert 'mol_a' in result[0] or 'mol_b' in result[0] or 'mol_c' in result[0]


# =============================================================================
# Logging Tests
# =============================================================================

class TestLogging:
    """Tests for logging functionality."""

    def test_log_writes_to_file(self, mock_dataset, mock_model, temp_dir):
        """Test log writes to file when verbose and write_file."""
        evaluator = Evaluator(
            save_dir=temp_dir,
            dataset=mock_dataset,
            model=mock_model,
            task_type='regression',
            verbose=True
        )
        evaluator.log("Test message")
        evaluator.logfile.flush()

        with open(os.path.join(temp_dir, 'results.log'), 'r') as f:
            content = f.read()
            assert "Test message" in content

    def test_log_silent_when_not_verbose(self, mock_dataset, mock_model, temp_dir, capsys):
        """Test log is silent when verbose=False."""
        evaluator = Evaluator(
            save_dir=None,
            dataset=mock_dataset,
            model=mock_model,
            task_type='regression',
            verbose=False
        )
        evaluator.log("Test message")
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_log_prints_when_no_save_dir(self, mock_dataset, mock_model, capsys):
        """Test log prints to stdout when no save_dir."""
        evaluator = Evaluator(
            save_dir=None,
            dataset=mock_dataset,
            model=mock_model,
            task_type='regression',
            verbose=True
        )
        evaluator.log("Test message")
        captured = capsys.readouterr()
        assert "Test message" in captured.out

    def test_log_metrics_format(self, mock_dataset, mock_model, temp_dir):
        """Test log_metrics correctly formats output."""
        evaluator = Evaluator(
            save_dir=temp_dir,
            dataset=mock_dataset,
            model=mock_model,
            task_type='regression',
            metrics=['rmse', 'r2'],
            verbose=True
        )
        df_metrics = pd.DataFrame({
            'metric': ['rmse', 'rmse', 'r2', 'r2'],
            'no_targets_columns': [0, 0, 0, 0],
            'value': [0.1, 0.15, 0.95, 0.92]
        })
        evaluator.log_metrics(df_metrics)
        evaluator.logfile.flush()

        with open(os.path.join(temp_dir, 'results.log'), 'r') as f:
            content = f.read()
            assert 'rmse' in content
            assert 'r2' in content


# =============================================================================
# Integration-style Unit Tests (with mocks)
# =============================================================================

class TestEvaluateTrainTest:
    """Tests for evaluate_train_test method."""

    def test_evaluate_train_test_regression(self, mock_dataset, mock_model, temp_dir):
        """Test evaluate_train_test for regression."""
        import inspect

        # Setup mock datasets
        mock_train = Mock(spec=Dataset)
        mock_train.X = np.array([[1, 2], [3, 4], [5, 6]])
        mock_train.y = np.array([[1.0], [2.0], [3.0]])
        mock_train.N_tasks = 1

        mock_test = Mock(spec=Dataset)
        mock_test.X = np.array([[7, 8], [9, 10]])
        mock_test.y = np.array([[4.0], [5.0]])
        mock_test.N_tasks = 1
        mock_test.repr = ['test1', 'test2']

        # Create a mock predict that properly handles return_std
        def mock_predict(X, return_std=False):
            if return_std:
                return np.array([3.9, 4.8]), np.array([0.1, 0.1])
            return np.array([3.9, 4.8])

        mock_model.predict = mock_predict

        evaluator = Evaluator(
            save_dir=temp_dir,
            dataset=mock_train,
            model=mock_model,
            task_type='regression',
            metrics=['rmse'],
            cross_validation='Monte-Carlo',
            split_type='random',
            split_sizes=[0.8, 0.2],
            verbose=False
        )

        df_predict, df_metrics = evaluator.evaluate_train_test(mock_train, mock_test, loocv=False)

        assert 'repr' in df_predict.columns
        assert 'predict_0' in df_predict.columns
        assert len(df_predict) == 2

    def test_evaluate_train_test_binary(self, mock_dataset, mock_model, temp_dir):
        """Test evaluate_train_test for binary classification."""
        mock_train = Mock(spec=Dataset)
        mock_train.X = np.array([[1, 2], [3, 4], [5, 6]])
        mock_train.y = np.array([[0], [1], [1]])
        mock_train.N_tasks = 1

        mock_test = Mock(spec=Dataset)
        mock_test.X = np.array([[7, 8], [9, 10]])
        mock_test.y = np.array([[0], [1]])
        mock_test.N_tasks = 1
        mock_test.repr = ['test1', 'test2']

        mock_model.predict_proba.return_value = np.array([0.2, 0.9])

        evaluator = Evaluator(
            save_dir=temp_dir,
            dataset=mock_train,
            model=mock_model,
            task_type='binary',
            metrics=['roc_auc'],
            cross_validation='Monte-Carlo',
            split_type='random',
            split_sizes=[0.8, 0.2],
            verbose=False
        )

        df_predict, df_metrics = evaluator.evaluate_train_test(mock_train, mock_test, loocv=False)

        assert 'predict_0' in df_predict.columns
        mock_model.predict_proba.assert_called()

    def test_evaluate_train_test_loocv(self, mock_dataset, mock_model, temp_dir):
        """Test evaluate_train_test with leave-one-out cross-validation."""
        mock_dataset.N_tasks = 1
        mock_dataset.repr = ['mol1', 'mol2', 'mol3', 'mol4', 'mol5', 'mol6']

        evaluator = Evaluator(
            save_dir=temp_dir,
            dataset=mock_dataset,
            model=mock_model,
            task_type='regression',
            metrics=['rmse'],
            cross_validation='leave-one-out',
            verbose=False
        )

        df_predict, df_metrics = evaluator.evaluate_train_test(mock_dataset, mock_dataset, loocv=True)

        mock_model.predict_loocv.assert_called()
        assert len(df_predict) == 6

    def test_evaluate_train_test_unknown_targets(self, mock_dataset, mock_model, temp_dir):
        """Test evaluate_train_test when test targets unknown."""
        mock_train = Mock(spec=Dataset)
        mock_train.X = np.array([[1, 2], [3, 4]])
        mock_train.y = np.array([[1.0], [2.0]])
        mock_train.N_tasks = 1

        mock_test = Mock(spec=Dataset)
        mock_test.X = np.array([[5, 6]])
        mock_test.y = np.array([]).reshape(1, 0)  # No targets
        mock_test.N_tasks = 0  # Different from train
        mock_test.repr = ['test1']

        # Create a mock predict that properly handles return_std
        def mock_predict(X, return_std=False):
            if return_std:
                return np.array([2.5]), np.array([0.1])
            return np.array([2.5])

        mock_model.predict = mock_predict

        evaluator = Evaluator(
            save_dir=temp_dir,
            dataset=mock_train,
            model=mock_model,
            task_type='regression',
            metrics=['rmse'],
            verbose=False
        )

        df_predict, df_metrics = evaluator.evaluate_train_test(mock_train, mock_test, loocv=False)

        assert df_metrics is None
        assert 'target_0' not in df_predict.columns

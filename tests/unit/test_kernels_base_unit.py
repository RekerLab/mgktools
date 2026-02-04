#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fast unit tests for mgktools.kernels.base module.

Tests cover MicroKernel, ABCKernelConfig, and BaseKernelConfig classes.
"""
import pytest
import numpy as np
import tempfile
import os
from unittest.mock import Mock, MagicMock, patch

from mgktools.kernels.base import MicroKernel, ABCKernelConfig, BaseKernelConfig


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def basic_microkernel():
    """Basic MicroKernel with continuous bounds."""
    return MicroKernel(
        idx=0,
        name='test_kernel',
        kernel_type='sExp',
        value=1.0,
        bounds=(0.1, 10.0)
    )


@pytest.fixture
def fixed_microkernel():
    """MicroKernel with fixed bounds."""
    return MicroKernel(
        idx=0,
        name='fixed_kernel',
        kernel_type='sExp',
        value=1.0,
        bounds='fixed'
    )


@pytest.fixture
def categorical_microkernel():
    """MicroKernel with categorical values."""
    return MicroKernel(
        idx=0,
        name='type_kernel',
        kernel_type='a_type',
        value='Tensorproduct',
        available_values=['Tensorproduct', 'Additive']
    )


# =============================================================================
# MicroKernel Initialization Tests
# =============================================================================

class TestMicroKernelInit:
    """Tests for MicroKernel initialization."""

    def test_init_with_bounds(self):
        """Test initialization with continuous bounds."""
        mk = MicroKernel(
            idx=0,
            name='test',
            kernel_type='sExp',
            value=1.0,
            bounds=(0.1, 10.0)
        )
        assert mk.name == 'test'
        assert mk.kernel_type == 'sExp'
        assert mk.value == 1.0
        assert mk.bounds == (0.1, 10.0)
        assert mk.available_values is None

    def test_init_with_fixed_bounds(self):
        """Test initialization with fixed bounds."""
        mk = MicroKernel(
            idx=0,
            name='test',
            kernel_type='sExp',
            value=1.0,
            bounds='fixed'
        )
        assert mk.bounds == 'fixed'

    def test_init_with_categorical_values(self):
        """Test initialization with categorical values."""
        mk = MicroKernel(
            idx=0,
            name='type',
            kernel_type='a_type',
            value='Tensorproduct',
            available_values=['Tensorproduct', 'Additive']
        )
        assert mk.available_values == ['Tensorproduct', 'Additive']
        assert mk.bounds is None

    def test_unique_name_format(self):
        """Test unique_name is properly formatted."""
        mk = MicroKernel(
            idx=5,
            name='length_scale',
            kernel_type='sExp',
            value=1.0,
            bounds=(0.1, 10.0)
        )
        assert mk.unique_name == '5:length_scale:sExp'

    def test_init_with_delta(self):
        """Test initialization with delta step size."""
        mk = MicroKernel(
            idx=0,
            name='test',
            kernel_type='sExp',
            value=1.0,
            bounds=(0.1, 10.0),
            delta=0.1
        )
        assert mk.delta == 0.1

    def test_bounds_list_converted_to_tuple(self):
        """Test that bounds list is converted to tuple."""
        mk = MicroKernel(
            idx=0,
            name='test',
            kernel_type='sExp',
            value=1.0,
            bounds=[0.1, 10.0]
        )
        assert mk.bounds == (0.1, 10.0)
        assert isinstance(mk.bounds, tuple)

    def test_invalid_bounds_raises(self):
        """Test that invalid bounds raise AssertionError."""
        with pytest.raises(AssertionError):
            MicroKernel(
                idx=0,
                name='test',
                kernel_type='sExp',
                value=1.0,
                bounds=(10.0, 0.1)  # Lower > upper
            )

    def test_both_bounds_and_values_raises(self):
        """Test that having both bounds and available_values raises."""
        with pytest.raises(AssertionError):
            MicroKernel(
                idx=0,
                name='test',
                kernel_type='sExp',
                value=1.0,
                bounds=(0.1, 10.0),
                available_values=['a', 'b']
            )

    def test_categorical_value_not_in_list_raises(self):
        """Test that categorical value not in list raises."""
        with pytest.raises(AssertionError):
            MicroKernel(
                idx=0,
                name='type',
                kernel_type='a_type',
                value='Invalid',
                available_values=['Tensorproduct', 'Additive']
            )


# =============================================================================
# MicroKernel get_kernel Tests
# =============================================================================

class TestMicroKernelGetKernel:
    """Tests for MicroKernel.get_kernel method."""

    def test_get_kernel_rbf(self):
        """Test get_kernel for RBF type."""
        mk = MicroKernel(
            idx=0,
            name='length',
            kernel_type='rbf',
            value=1.0,
            bounds=(0.1, 10.0)
        )
        kernel = mk.get_kernel()
        assert hasattr(kernel, 'length_scale')
        assert kernel.length_scale == 1.0

    def test_get_kernel_dot_product(self):
        """Test get_kernel for dot_product type."""
        mk = MicroKernel(
            idx=0,
            name='sigma',
            kernel_type='dot_product',
            value=0.5,
            bounds=(0.1, 5.0)
        )
        kernel = mk.get_kernel()
        assert hasattr(kernel, 'sigma_0')
        assert kernel.sigma_0 == 0.5

    def test_get_kernel_sExp(self):
        """Test get_kernel for sExp type."""
        mk = MicroKernel(
            idx=0,
            name='test',
            kernel_type='sExp',
            value=1.0,
            bounds=(0.1, 10.0)
        )
        kernel = mk.get_kernel()
        # Should return graphdot SquareExponential
        assert kernel is not None

    def test_get_kernel_kDelta(self):
        """Test get_kernel for kDelta type."""
        mk = MicroKernel(
            idx=0,
            name='test',
            kernel_type='kDelta',
            value=0.5,
            bounds='fixed'
        )
        kernel = mk.get_kernel()
        assert kernel is not None

    def test_get_kernel_Const(self):
        """Test get_kernel for Const type."""
        mk = MicroKernel(
            idx=0,
            name='test',
            kernel_type='Const',
            value=1.0,
            bounds='fixed'
        )
        kernel = mk.get_kernel()
        assert kernel is not None

    def test_get_kernel_tensorproduct_type(self):
        """Test get_kernel for a_type Tensorproduct."""
        mk = MicroKernel(
            idx=0,
            name='a_type',
            kernel_type='a_type',
            value='Tensorproduct',
            available_values=['Tensorproduct', 'Additive']
        )
        kernel_func = mk.get_kernel()
        assert callable(kernel_func) or kernel_func is not None

    def test_get_kernel_additive_type(self):
        """Test get_kernel for a_type Additive."""
        mk = MicroKernel(
            idx=0,
            name='a_type',
            kernel_type='a_type',
            value='Additive',
            available_values=['Tensorproduct', 'Additive']
        )
        kernel_func = mk.get_kernel()
        assert callable(kernel_func)

    def test_get_kernel_invalid_type_raises(self):
        """Test get_kernel raises for invalid kernel_type."""
        mk = MicroKernel(
            idx=0,
            name='test',
            kernel_type='invalid_type',
            value=1.0,
            bounds='fixed'
        )
        # Bypass validation
        mk.kernel_type = 'invalid_type'
        with pytest.raises(ValueError, match='Invalid kernel type'):
            mk.get_kernel()

    def test_get_kernel_invalid_a_type_value_raises(self):
        """Test get_kernel raises for invalid a_type value."""
        mk = MicroKernel(
            idx=0,
            name='a_type',
            kernel_type='a_type',
            value='Tensorproduct',
            available_values=['Tensorproduct', 'Additive']
        )
        mk.value = 'InvalidType'
        with pytest.raises(ValueError, match='not supported'):
            mk.get_kernel()


# =============================================================================
# MicroKernel Update Methods Tests
# =============================================================================

class TestMicroKernelUpdate:
    """Tests for MicroKernel update methods."""

    def test_update_from_trial_updates_value(self, basic_microkernel):
        """Test update_from_trial updates value."""
        trial = {basic_microkernel.unique_name: 5.0}
        basic_microkernel.update_from_trial(trial)
        assert basic_microkernel.value == 5.0

    def test_update_from_trial_ignores_missing(self, fixed_microkernel):
        """Test update_from_trial does nothing if key missing."""
        original_value = fixed_microkernel.value
        fixed_microkernel.update_from_trial({})
        assert fixed_microkernel.value == original_value

    def test_update_from_trial_raises_if_fixed_but_present(self, fixed_microkernel):
        """Test update_from_trial raises if fixed but key present."""
        with pytest.raises(AssertionError):
            fixed_microkernel.update_from_trial({fixed_microkernel.unique_name: 2.0})

    def test_update_from_theta_pops_value(self, basic_microkernel):
        """Test update_from_theta pops and assigns value."""
        values = [2.5, 3.5]  # First value should be popped
        basic_microkernel.update_from_theta(values)
        assert basic_microkernel.value == 2.5
        assert values == [3.5]

    def test_update_from_theta_fixed_does_nothing(self, fixed_microkernel):
        """Test update_from_theta does nothing for fixed bounds."""
        values = [2.5]
        fixed_microkernel.update_from_theta(values)
        assert values == [2.5]  # Not popped

    def test_update_from_theta_clips_to_bounds(self):
        """Test update_from_theta clips values within tolerance to bounds."""
        mk = MicroKernel(
            idx=0,
            name='test',
            kernel_type='sExp',
            value=1.0,
            bounds=(0.1, 10.0)
        )
        # Value within 1% tolerance of lower bound
        values = [0.0995]  # 0.1 * 0.99 = 0.099, so 0.0995 is within tolerance
        mk.update_from_theta(values)
        assert mk.value == 0.1


# =============================================================================
# MicroKernel Optuna Trial Tests
# =============================================================================

class TestMicroKernelTrial:
    """Tests for MicroKernel Optuna trial methods."""

    def test_get_trial_fixed_returns_empty(self, fixed_microkernel):
        """Test get_trial returns empty dict for fixed bounds."""
        mock_trial = Mock()
        result = fixed_microkernel.get_trial(mock_trial)
        assert result == {}

    def test_get_trial_continuous_suggests_float(self, basic_microkernel):
        """Test get_trial calls suggest_float for continuous."""
        mock_trial = Mock()
        mock_trial.suggest_float.return_value = 2.0
        result = basic_microkernel.get_trial(mock_trial)

        mock_trial.suggest_float.assert_called_once()
        assert basic_microkernel.unique_name in result

    def test_get_trial_with_delta_uses_step(self):
        """Test get_trial uses step parameter when delta set."""
        mk = MicroKernel(
            idx=0,
            name='test',
            kernel_type='sExp',
            value=1.0,
            bounds=(0.0, 10.0),
            delta=0.5
        )
        mock_trial = Mock()
        mock_trial.suggest_float.return_value = 5.0
        mk.get_trial(mock_trial)

        # Check step parameter was passed
        call_kwargs = mock_trial.suggest_float.call_args[1]
        assert call_kwargs.get('step') == 0.5

    def test_get_trial_categorical_suggests_categorical(self, categorical_microkernel):
        """Test get_trial calls suggest_categorical for categorical."""
        mock_trial = Mock()
        mock_trial.suggest_categorical.return_value = 'Additive'
        result = categorical_microkernel.get_trial(mock_trial)

        mock_trial.suggest_categorical.assert_called_once()
        assert categorical_microkernel.unique_name in result


# =============================================================================
# MicroKernel Dictionary Methods Tests
# =============================================================================

class TestMicroKernelDict:
    """Tests for MicroKernel dictionary methods."""

    def test_get_microdict(self, basic_microkernel):
        """Test get_microdict returns correct structure."""
        microdict = basic_microkernel.get_microdict()
        assert 'sExp' in microdict
        assert microdict['sExp'] == [1.0, (0.1, 10.0), None, None]

    def test_update_hyperdict_same_name_and_type(self):
        """Test update_hyperdict when name equals kernel_type."""
        mk = MicroKernel(
            idx=0,
            name='sExp',
            kernel_type='sExp',
            value=1.0,
            bounds=(0.1, 10.0)
        )
        hyperdict = {}
        mk.update_hyperdict(hyperdict)
        assert 'sExp' in hyperdict

    def test_update_hyperdict_different_name_and_type(self, basic_microkernel):
        """Test update_hyperdict when name differs from kernel_type."""
        hyperdict = {}
        basic_microkernel.update_hyperdict(hyperdict)
        assert 'test_kernel' in hyperdict
        assert 'sExp' in hyperdict['test_kernel']

    def test_from_microdict(self):
        """Test from_microdict class method."""
        microdict = {'sExp': [1.5, (0.2, 8.0), 0.1, None]}
        mk = MicroKernel.from_microdict(idx=2, name='length', microdict=microdict)

        # idx is used in unique_name but not stored separately
        assert mk.unique_name == '2:length:sExp'
        assert mk.name == 'length'
        assert mk.kernel_type == 'sExp'
        assert mk.value == 1.5
        assert mk.bounds == (0.2, 8.0)
        assert mk.delta == 0.1


# =============================================================================
# BaseKernelConfig Tests
# =============================================================================

class ConcreteKernelConfig(BaseKernelConfig):
    """Concrete implementation of BaseKernelConfig for testing."""

    def _get_kernel(self):
        """Create kernel based on kernel_type."""
        from sklearn.gaussian_process.kernels import RBF, DotProduct
        if self.kernel_type == 'rbf':
            return RBF(length_scale=self.kernel_hyperparameters[0],
                       length_scale_bounds=self.kernel_hyperparameters_bounds[0])
        elif self.kernel_type == 'dot_product':
            return DotProduct(sigma_0=self.kernel_hyperparameters[0],
                              sigma_0_bounds=self.kernel_hyperparameters_bounds[0])
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")

    def update_kernel(self):
        pass

    def get_trial(self, trial):
        return {}

    def update_from_trial(self, trial):
        pass

    def update_from_theta(self):
        pass


class TestBaseKernelConfig:
    """Tests for BaseKernelConfig class."""

    def test_init_rbf_kernel(self):
        """Test initialization with RBF kernel."""
        config = ConcreteKernelConfig(
            kernel_type='rbf',
            kernel_hyperparameters=[1.0],
            kernel_hyperparameters_bounds=[(0.1, 10.0)]
        )
        assert config.kernel_type == 'rbf'
        assert config.kernel is not None

    def test_init_dot_product_kernel(self):
        """Test initialization with dot_product kernel."""
        config = ConcreteKernelConfig(
            kernel_type='dot_product',
            kernel_hyperparameters=[0.5],
            kernel_hyperparameters_bounds=[(0.1, 5.0)]
        )
        assert config.kernel_type == 'dot_product'
        assert config.kernel is not None

    def test_get_kernel_dict(self):
        """Test get_kernel_dict returns correct structure."""
        config = ConcreteKernelConfig(
            kernel_type='rbf',
            kernel_hyperparameters=[1.0],
            kernel_hyperparameters_bounds=[(0.1, 10.0)]
        )
        X = np.random.randn(5, 3)
        labels = ['a', 'b', 'c', 'd', 'e']

        kernel_dict = config.get_kernel_dict(X, labels)

        assert 'X' in kernel_dict
        assert 'K' in kernel_dict
        assert 'theta' in kernel_dict
        assert kernel_dict['X'] == labels
        assert kernel_dict['K'].shape == (5, 5)

    def test_save_kernel_matrix(self):
        """Test save_kernel_matrix creates file."""
        config = ConcreteKernelConfig(
            kernel_type='rbf',
            kernel_hyperparameters=[1.0],
            kernel_hyperparameters_bounds=[(0.1, 10.0)]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            X = np.random.randn(3, 2)
            labels = ['mol1', 'mol2', 'mol3']
            config.save_kernel_matrix(tmpdir, X, labels)

            assert os.path.exists(os.path.join(tmpdir, 'kernel.pkl'))

    def test_combine_dicts(self):
        """Test combine_dicts static method."""
        dicts = [{'a': 1, 'b': 2}, {'c': 3}, {'d': 4, 'e': 5}]
        combined = BaseKernelConfig.combine_dicts(dicts)

        assert combined == {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}

    def test_combine_dicts_raises_on_duplicate_keys(self):
        """Test combine_dicts raises on duplicate keys."""
        dicts = [{'a': 1}, {'a': 2}]
        with pytest.raises(AssertionError):
            BaseKernelConfig.combine_dicts(dicts)


# =============================================================================
# ABCKernelConfig Tests
# =============================================================================

class TestABCKernelConfig:
    """Tests for ABCKernelConfig abstract class."""

    def test_cannot_instantiate_directly(self):
        """Test that ABCKernelConfig cannot be instantiated."""
        with pytest.raises(TypeError):
            ABCKernelConfig()

    def test_subclass_must_implement_methods(self):
        """Test that subclass must implement abstract methods."""
        class IncompleteConfig(ABCKernelConfig):
            pass

        with pytest.raises(TypeError):
            IncompleteConfig()

    def test_subclass_with_all_methods(self):
        """Test that subclass with all methods can be instantiated."""
        class CompleteConfig(ABCKernelConfig):
            def update_kernel(self):
                pass

            def get_trial(self, trial):
                return {}

            def update_from_trial(self, trial):
                pass

            def update_from_theta(self):
                pass

        config = CompleteConfig()
        assert config is not None

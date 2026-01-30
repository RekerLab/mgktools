#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Base kernel classes and microkernel components.

This module provides the foundational classes for building complex kernels
from smaller microkernel components. It supports hyperparameter optimization
through Optuna and gradient-based methods.

Classes
-------
MicroKernel
    Smallest unit of a kernel with a single hyperparameter.
ABCKernelConfig
    Abstract base class for kernel configurations.
BaseKernelConfig
    Base configuration class for sklearn-compatible kernels.
"""
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Callable, Any
import os
import pickle
import numpy as np
from sklearn.gaussian_process.kernels import DotProduct, RBF
from graphdot.microkernel import (
    Additive,
    Constant as Const,
    TensorProduct,
    SquareExponential as sExp,
    KroneckerDelta as kDelta,
    Convolution as kConv,
    Normalize,
)
from graphdot.microprobability import (
    Additive as Additive_p,
    Constant,
    AssignProbability,
)


class MicroKernel:
    """
    Smallest unit of a kernel containing a single hyperparameter.

    MicroKernels are the building blocks for constructing complex graph kernels.
    Each microkernel wraps a specific kernel function type (e.g., RBF, Kronecker Delta,
    Square Exponential) with its associated hyperparameter and optimization bounds.

    Parameters
    ----------
    idx : int
        Index of the kernel configuration this microkernel belongs to.
        Used to create unique names when the same microkernel type appears
        in different kernel configs.
    name : str
        Name of the microkernel (e.g., 'atom_Gasteiger_charge').
    kernel_type : str
        Type of kernel function. Supported types:
        - 'rbf': Radial Basis Function kernel
        - 'dot_product': Dot product kernel
        - 'Const': Constant microkernel
        - 'kDelta': Kronecker Delta microkernel
        - 'kConv': Convolution microkernel
        - 'sExp': Square Exponential microkernel
        - 'Const_p': Constant microprobability
        - 'Assign_p': Assignment probability
        - 'Normalization': Kernel normalization
        - 'a_type', 'b_type', 'p_type': Atom/bond/probability composition types
    value : float or str
        Hyperparameter value or composition type name.
    bounds : Tuple[float, float], optional
        Lower and upper bounds for hyperparameter optimization.
        Use 'fixed' for non-optimizable parameters, by default None.
    delta : float, optional
        Step size for discrete hyperparameter optimization, by default None.
    available_values : List, optional
        List of categorical values for hyperparameter selection, by default None.

    Attributes
    ----------
    unique_name : str
        Unique identifier combining idx, name, and kernel_type.

    Raises
    ------
    AssertionError
        If bounds and available_values are both None or both set,
        or if bounds are invalid.

    Examples
    --------
    >>> mk = MicroKernel(idx=0, name='length_scale', kernel_type='sExp',
    ...                  value=1.0, bounds=(0.1, 10.0))
    >>> kernel = mk.get_kernel()
    >>> trial_params = mk.get_trial(trial)  # For hyperparameter optimization
    """

    def __init__(
        self,
        idx: int,
        name: str,
        kernel_type: str,
        value,
        bounds: Tuple[float, float] = None,
        delta: float = None,
        available_values: List = None,
    ):
        self.name = name
        self.kernel_type = kernel_type
        self.unique_name = f"{idx}:{name}:{kernel_type}"
        self.value = value
        self.bounds = bounds
        self.delta = delta
        self.available_values = available_values
        if self.bounds is None:
            assert self.available_values is not None
            assert self.value in self.available_values
        else:
            assert self.available_values is None
            if self.bounds != "fixed":
                if isinstance(self.bounds, list):
                    self.bounds = tuple(self.bounds)
                assert (
                    len(self.bounds) == 2
                ), "Bounds must be a tuple with two elements."
                assert (
                    self.bounds[0] < self.bounds[1]
                ), "Lower bound must be less than upper bound."
                assert isinstance(self.value, float)

    def get_kernel(self) -> Callable:
        """
        Create and return the kernel function with current hyperparameters.

        Returns
        -------
        Callable
            Kernel function instance with configured hyperparameters.

        Raises
        ------
        ValueError
            If kernel_type is not supported.
        """
        bounds = self.bounds or "fixed"
        if self.kernel_type == "rbf":
            return RBF(length_scale=self.value, length_scale_bounds=bounds)
        elif self.kernel_type == "dot_product":
            return DotProduct(sigma_0=self.value, sigma_0_bounds=bounds)
        elif self.kernel_type == "Const":
            return Const(self.value, bounds)
        elif self.kernel_type == "kDelta":
            return kDelta(self.value, bounds)
        elif self.kernel_type == "kConv":
            return kConv(kDelta(self.value, bounds))
        elif self.kernel_type == "sExp":
            return sExp(self.value, length_scale_bounds=bounds)
        elif self.kernel_type == "Const_p":
            return Constant(self.value, bounds)
        elif self.kernel_type == "Assign_p":
            return AssignProbability(self.value, bounds)
        elif self.kernel_type == "Normalization":
            from mgktools.kernels.normalization import Norm, NormalizationMolSize
            assert self.name == self.kernel_type
            if self.value == True:
                return Norm
            elif self.value == False:
                return lambda x: x
            else:
                assert isinstance(self.value, float)
                return lambda x: NormalizationMolSize(
                    kernel=x, s=self.value, s_bounds=bounds
                )
        elif self.kernel_type in ["a_type", "b_type", "p_type"]:
            if self.value == "Tensorproduct":
                return TensorProduct
            elif self.value == "Additive":
                return lambda **x: Normalize(Additive(**x))
            elif self.value == "Additive_p":
                return Additive_p
            else:
                raise ValueError(
                    "For kernel type (%s), the value (%s) is not supported."
                    % (self.kernel_type, self.value)
                )
        else:
            raise ValueError("Invalid kernel type %s." % self.kernel_type)

    def get_trial(self, trial) -> Dict:
        """
        Get Optuna trial suggestions for this microkernel.

        Parameters
        ----------
        trial : optuna.Trial
            Optuna trial object for suggesting hyperparameters.

        Returns
        -------
        Dict
            Dictionary mapping unique_name to suggested value.
            Empty dict if parameter is fixed.
        """
        if self.available_values is not None:
            return {
                self.unique_name: trial.suggest_categorical(
                    name=self.unique_name, choices=self.available_values
                )
            }
        elif self.bounds == "fixed":
            return {}
        else:
            if self.delta is None:
                return {
                    self.unique_name: trial.suggest_float(
                        name=self.unique_name, low=self.bounds[0], high=self.bounds[1]
                    )
                }
            else:
                return {
                    self.unique_name: trial.suggest_float(
                        name=self.unique_name,
                        low=self.bounds[0],
                        high=self.bounds[1],
                        step=self.delta,
                    )
                }

    def update_from_trial(self, trial: Dict[str, Any]):
        """
        Update hyperparameter value from Optuna trial result.

        Parameters
        ----------
        trial : Dict[str, Any]
            Dictionary of trial hyperparameter values.
        """
        if self.unique_name in trial:
            assert self.bounds != "fixed"
            self.value = trial[self.unique_name]
        else:
            assert self.bounds == "fixed", f"{self.unique_name};{self.bounds};{trial}"

    def update_from_theta(self, values: List):
        """
        Update hyperparameter value from gradient optimization theta.

        Parameters
        ----------
        values : List
            List of hyperparameter values. The first value is popped
            and assigned to this microkernel.
        """
        assert self.available_values is None
        if self.bounds != "fixed":
            assert self.bounds[0] * 0.99 < values[0] < self.bounds[1] * 1.01
            if values[0] < self.bounds[0]:
                values[0] = self.bounds[0]
            elif values[0] > self.bounds[1]:
                values[0] = self.bounds[1]
            self.value = values.pop(0)

    def get_microdict(self) -> Dict:
        """
        Get dictionary representation of microkernel configuration.

        Returns
        -------
        Dict
            Dictionary with kernel_type as key and [value, bounds, delta,
            available_values] as value.
        """
        values = [self.value, self.bounds, self.delta, self.available_values]
        return {f"{self.kernel_type}": values}

    def update_hyperdict(self, hyperdict: Dict):
        """
        Update a hyperparameter dictionary with this microkernel's configuration.

        Parameters
        ----------
        hyperdict : Dict
            Dictionary to update with microkernel configuration.
        """
        if self.name == self.kernel_type:
            assert self.kernel_type not in hyperdict
            hyperdict.update(self.get_microdict())
        else:
            if self.name not in hyperdict:
                hyperdict.update({self.name: {}})
            assert self.kernel_type not in hyperdict[self.name]
            hyperdict[self.name].update(self.get_microdict())

    @classmethod
    def from_microdict(cls, idx: int, name: str, microdict: Dict[str, List]) -> 'MicroKernel':
        """
        Create a MicroKernel from a dictionary representation.

        Parameters
        ----------
        idx : int
            Index for unique naming.
        name : str
            Name of the microkernel.
        microdict : Dict[str, List]
            Dictionary with kernel_type as key and configuration list as value.

        Returns
        -------
        MicroKernel
            New MicroKernel instance.
        """
        assert len(microdict) == 1
        for kernel_type, values in microdict.items():
            return cls(
                idx=idx,
                name=name,
                kernel_type=kernel_type,
                value=values[0],
                bounds=values[1],
                delta=values[2],
                available_values=values[3],
            )


class ABCKernelConfig(ABC):
    """
    Abstract base class for kernel configurations.

    This class defines the interface that all kernel configurations must
    implement to support hyperparameter optimization via Optuna and
    gradient-based methods.
    """

    @abstractmethod
    def update_kernel(self):
        """Update the kernel with current hyperparameter values."""
        pass

    @abstractmethod
    def get_trial(self, trial) -> Dict:
        """Get Optuna trial suggestions for all hyperparameters."""
        pass

    @abstractmethod
    def update_from_trial(self, trial: Dict[str, Any]):
        """Update hyperparameters from Optuna trial result."""
        pass

    @abstractmethod
    def update_from_theta(self):
        """Update hyperparameters from gradient optimization theta."""
        pass


class BaseKernelConfig(ABCKernelConfig):
    """
    Base configuration class for sklearn-compatible kernels.

    This class provides common functionality for configuring feature-based
    kernels (RBF, DotProduct) that work with sklearn's GaussianProcessRegressor.

    Parameters
    ----------
    kernel_type : str
        Type of kernel ('rbf' or 'dot_product').
    kernel_hyperparameters : list
        List of hyperparameter values.
    kernel_hyperparameters_bounds : list
        List of (lower, upper) bound tuples for each hyperparameter.

    Attributes
    ----------
    kernel : sklearn kernel
        The configured sklearn kernel instance.
    """

    def __init__(
        self,
        kernel_type: str,
        kernel_hyperparameters: list,
        kernel_hyperparameters_bounds: list,
    ):
        """Initialize base kernel configuration."""
        self.kernel_type = kernel_type
        self.kernel_hyperparameters = kernel_hyperparameters
        self.kernel_hyperparameters_bounds = kernel_hyperparameters_bounds
        self.kernel = self._get_kernel()

    def get_kernel_dict(self, X: np.ndarray, X_labels: List[str]) -> Dict:
        """Calculate a kernel matrix and save in a dictionary.

        Parameters
        ----------
        X : np.ndarray
            A numpy array with the data to compute the kernel matrix.
        X_labels : List[str]
            A list with the unique text labels for the data.

        Returns
        -------
        dict:
            A dictionary with the kernel matrix and the data labels.
        """
        K = self.kernel(X)
        return {"X": X_labels, "K": K, "theta": self.kernel.theta}

    def save_kernel_matrix(self, path: str, X: np.ndarray, X_labels: List[str]):
        """Save kernel.pkl file that used for preCalc kernels."""
        kernel_dict = self.get_kernel_dict(X, X_labels)
        kernel_pkl = os.path.join(path, "kernel.pkl")
        pickle.dump(kernel_dict, open(kernel_pkl, "wb"), protocol=4)

    @staticmethod
    def combine_dicts(dicts: List[Dict]) -> Dict:
        """Combine a list of dictionaries into one.

        Parameters
        ----------
        dicts : List[Dict]
            A list with dictionaries to be combined.

        Returns
        -------
        Dict:
            A dictionary with the combined dictionaries.
        """
        combined_dict = {}
        n = 0
        for d in dicts:
            combined_dict.update(d)
            n += len(d)
        assert n == len(combined_dict)
        return combined_dict

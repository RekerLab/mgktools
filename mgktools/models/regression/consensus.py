#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Literal
import numpy as np
import copy
import threading
from joblib import Parallel, delayed
from mgktools.models.regression.gpr.gpr import GaussianProcessRegressor as GPRgraphdot


def _parallel_build_models(model, models, X, y, model_idx, n_models,
                           verbose=0):
    """
    Private function used to fit a ensemble model in parallel."""
    if verbose > 1:
        print("building model %d of %d" % (model_idx + 1, n_models))
    np.random.seed(model_idx)
    idx = np.random.choice(np.arange(len(X)), models.n_samples_per_model,
                           replace=False)
    if X.ndim == 1:
        model.fit(X[idx], y[idx])
    elif X.ndim == 2:
        model.fit(X[idx, :], y[idx])
    else:
        raise RuntimeError(
            'X must be 1 or 2 dimensional'
        )
    return model


def _accumulate_prediction(predict, X, out, out_u, lock, return_std=False):
    """
    This is a utility function for joblib's Parallel.

    It can't go locally in ForestClassifier or ForestRegressor, because joblib
    complains that it cannot pickle it when placed there.
    """
    if return_std:
        prediction, uncertainty = predict(X, return_std=True)
        with lock:
            out.append(prediction)
            out_u.append(uncertainty)
    else:
        prediction = predict(X, return_std=False)
        with lock:
            out.append(prediction)


class EnsembleRegressor:
    def __init__(self, model,
                 n_estimators: int = 100,
                 n_samples_per_model: int = 2000,
                 n_jobs: int = 1, verbose=0,
                 ensemble_rule: Literal['smallest_uncertainty', 'weight_uncertainty', 'mean'] = 'smallest_uncertainty'
                 ):
        self.model = model
        self.models = []
        self.n_estimators = n_estimators
        self.n_samples_per_model = n_samples_per_model
        self.n_jobs = 1 # n_jobs > 1 is not supported yet
        self.verbose = verbose
        self.ensemble_rule = ensemble_rule
        assert (n_estimators > 0)

    def fit(self, X, y):
        y = np.asarray(y)
        if y.dtype == float:
            valid_mask = ~np.isnan(y)
        else:
            valid_mask = np.ones(len(y), dtype=bool)
        X_valid = np.asarray(X)[valid_mask]
        y_valid = y[valid_mask]
        n_samples = min(self.n_samples_per_model, len(X_valid))
        original_n_samples = self.n_samples_per_model
        self.n_samples_per_model = n_samples
        models = [copy.deepcopy(self.model) for i in range(self.n_estimators)]
        models = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, prefer='processes')(
            delayed(_parallel_build_models)(
                m, self, X_valid, y_valid, i, len(models), verbose=self.verbose)
            for i, m in enumerate(models))
        self.models.extend(models)
        self.n_samples_per_model = original_n_samples

    def predict(self, X, return_std=False, return_cov=False):
        assert (not return_cov)
        if self.model.__class__ in [GPRgraphdot]:
            return self.predict_gpr(X, return_std=return_std)
        else:
            raise RuntimeError(
                f'The regressor {self.model} are not supported for '
                f'EnsembleRegressor yet'
            )

    def predict_gpr(self, X, return_std=False):
        y_hat = []
        u_hat = []
        # Parallel loop
        lock = threading.Lock()
        Parallel(n_jobs=self.n_jobs, verbose=self.verbose, require='sharedmem')(
            delayed(_accumulate_prediction)(m.predict, X, y_hat, u_hat, lock,
                                            return_std=return_std)
            for m in self.models)
        y, u = self.majority_vote(np.asarray(y_hat), np.asarray(u_hat),
                                  self.ensemble_rule)
        y_hat = y
        u_hat = u
        if return_std:
            return y_hat, u_hat
        else:
            return y_hat

    def majority_vote(self, y, u, rule):
        if rule == 'smallest_uncertainty':
            idx = u.argmin(axis=0)
            return np.array([y[idx[I]][I] for I in np.ndindex(idx.shape)]), \
                   np.array([u[idx[I]][I] for I in np.ndindex(idx.shape)])
        elif rule == 'weight_uncertainty':
            sigma = 10
            weight = np.exp(-sigma*u) / np.exp(-sigma*u).sum(axis=0)
            return (y * weight).sum(axis=0), (u * weight).sum(axis=0)
        elif rule == 'mean':
            return y.mean(axis=0), u.mean(axis=0)
        else:
            raise RuntimeError(
                f'Unknown predict_rule for EnsembleRegressor{rule}'
            )

    def save(self, path, overwrite=False):
        for i, m in enumerate(self.models):
            m.save(path, filename='model_%d.pkl' % i, overwrite=overwrite)

    def load(self, path):
        models = [copy.copy(self.model) for i in range(self.n_estimators)]
        for i, m in enumerate(models):
            m.load(path, filename='model_%d.pkl' % i)
        self.models = models

    @property
    def X_train_(self):
        return np.concatenate([m.X_train_ for m in self.models])

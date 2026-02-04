#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.svm import SVR


class SVRegressor(SVR):
    @staticmethod
    def _remove_nan_X_y(X, y):
        if None in y:
            idx = np.where(y != None)[0]
        elif y.dtype == float:
            idx = ~np.isnan(y)
        else:
            idx = np.arange(len(y))
        return np.asarray(X)[idx], y[idx]

    def fit(self, X, y, sample_weight=None):
        X_, y_ = self._remove_nan_X_y(X, y)
        return super().fit(X_, y_, sample_weight=sample_weight)

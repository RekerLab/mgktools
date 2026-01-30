#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Dict, Any
import os
import json
import numpy as np
from mgktools.kernels.base import BaseKernelConfig, MicroKernel


class FeatureKernelConfig(BaseKernelConfig):
    def __init__(
        self,
        hyperdict: Dict,
        idx: int = 0,
    ):
        """"""
        self.microkernels_feature = []
        assert len(hyperdict) == 1 and "features_kernel" in hyperdict
        for name, intermediate in hyperdict.items():
            for kernel_type, values in intermediate.items():
                microkernel = MicroKernel.from_microdict(
                    idx=idx, name=name, microdict={kernel_type: values}
                )
                self.microkernels_feature.append(microkernel)
        assert len(self.microkernels_feature) == 1
        self.update_kernel()

    def update_kernel(self):
        assert len(self.microkernels_feature) == 1
        self.kernel = self.microkernels_feature[0].get_kernel()

    def get_trial(self, trial) -> Dict:
        """Get Optuna trial suggestions for all hyperparameters.

        Returns:
        --------
        dict:
            A dictionary for hyperparameters space used in optuna.
        """
        return self.combine_dicts(
            [microkernel.get_trial(trial) for microkernel in self.microkernels_feature]
        )

    def update_from_trial(self, trial: Dict[str, Any]):
        """Update the hyperparameters from the trial dictionary.

        Parameters
        ----------
        trial : dict
            A dictionary with the hyperparameters values.
        """
        for microkernel in self.microkernels_feature:
            microkernel.update_from_trial(trial)

    def update_from_theta(self):
        """gradient optimization function. Update the hyperparameters from kernel.theta."""
        values = np.exp(self.kernel.theta).tolist()
        for microkernel in self.microkernels_feature:
            microkernel.update_from_theta(values)
        assert len(values) == 0

    def save(self, path: str, name: str = "features_hyperparameters.json"):
        hyperdict = {}
        for microkernel in self.microkernels_feature:
            microkernel.update_hyperdict(hyperdict)
        open(os.path.join(path, name), "w").write(
            json.dumps(hyperdict, indent=1, sort_keys=False)
        )

    @classmethod
    def load(cls, path: str, name: str = "features_hyperparameters.json", idx: int = 0):
        hyperdict = json.loads(open(os.path.join(path, name), "r").read())
        return cls(
            hyperdict=hyperdict,
            idx=idx,
        )

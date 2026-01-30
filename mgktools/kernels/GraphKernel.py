#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Dict, Any
import os
import json
import numpy as np
from graphdot.kernel.marginalized import MarginalizedGraphKernel
from mgktools.kernels.base import BaseKernelConfig, MicroKernel


class MGK(MarginalizedGraphKernel):
    """
    X and Y could be 2-d numpy array.
    make it compatible with sklearn.
    remove repeated kernel calculations, if set unique=True.
    """

    def __init__(self, unique=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.unique = unique

    @staticmethod
    def _unique(X):
        X_unique = np.sort(np.unique(X))
        X_idx = np.searchsorted(X_unique, X)
        return X_unique, X_idx

    @staticmethod
    def _format_X(X):
        if X.__class__ == np.ndarray:
            return X.ravel()  # .tolist()
        else:
            return X

    def __call__(self, X, Y=None, eval_gradient=False, *args, **kwargs):
        X = self._format_X(X)
        Y = self._format_X(Y)
        if self.unique:
            X_unique, X_idx = self._unique(X)
            if Y is None:
                Y_unique, Y_idx = X_unique, X_idx
            else:
                Y_unique, Y_idx = self._unique(Y)
            if eval_gradient:
                K, K_gradient = super().__call__(
                    X_unique, Y_unique, eval_gradient=True, *args, **kwargs
                )
                return K[X_idx][:, Y_idx], K_gradient[X_idx][:, Y_idx][:]
            else:
                K = super().__call__(
                    X_unique, Y_unique, eval_gradient=False, *args, **kwargs
                )
                return K[X_idx][:, Y_idx]
        else:
            return super().__call__(X, Y, eval_gradient=eval_gradient, *args, **kwargs)

    def diag(self, X, *args, **kwargs):
        X = self._format_X(X)
        if self.unique:
            X_unique, X_idx = self._unique(X)
            diag = super().diag(X_unique, *args, **kwargs)
            return diag[X_idx]
        else:
            return super().diag(X, *args, **kwargs)

    def get_params(self, deep=False):
        return dict(
            node_kernel=self.node_kernel,
            edge_kernel=self.edge_kernel,
            p=self.p,
            q=self.q,
            q_bounds=self.q_bounds,
            backend=self.backend,
        )


class GraphKernelConfig(BaseKernelConfig):
    def __init__(
        self,
        hyperdict: Dict,
        unique: bool = False,
        idx: int = 0,
    ):
        self.unique = unique
        self.microkernels_atom = []
        self.microkernels_bond = []
        self.microkernels_probability = []
        for name, itermediate in hyperdict.items():
            if name.startswith("atom_"):
                mks = []
                for kernel_type, values in itermediate.items():
                    microkernel = MicroKernel.from_microdict(
                        idx=idx, name=name, microdict={kernel_type: values}
                    )
                    mks.append(microkernel)
                self.microkernels_atom.append(mks)
            elif name.startswith("bond_"):
                mks = []
                for kernel_type, values in itermediate.items():
                    microkernel = MicroKernel.from_microdict(
                        idx=idx, name=name, microdict={kernel_type: values}
                    )
                    mks.append(microkernel)
                self.microkernels_bond.append(mks)
            elif name.startswith("probability_"):
                mks = []
                for kernel_type, values in itermediate.items():
                    microkernel = MicroKernel.from_microdict(
                        idx=idx, name=name, microdict={kernel_type: values}
                    )
                    mks.append(microkernel)
                self.microkernels_probability.append(mks)
            elif name == "Normalization":
                self.microkernel_normalization = MicroKernel.from_microdict(
                    idx=idx, name=name, microdict={name: itermediate}
                )
            elif name == "a_type":
                self.microkernel_atype = MicroKernel.from_microdict(
                    idx=idx, name=name, microdict={name: itermediate}
                )
            elif name == "b_type":
                self.microkernel_btype = MicroKernel.from_microdict(
                    idx=idx, name=name, microdict={name: itermediate}
                )
            elif name == "p_type":
                self.microkernel_ptype = MicroKernel.from_microdict(
                    idx=idx, name=name, microdict={name: itermediate}
                )
            elif name == "q":
                self.microkernel_q = MicroKernel.from_microdict(
                    idx=idx, name=name, microdict={name: itermediate}
                )
            else:
                raise ValueError(f"Unknown hyperparameter {name}")
        self.update_kernel()

    def update_kernel(self):
        knode_dict = {}
        for mks in self.microkernels_atom:
            knode_dict.update(
                {mks[0].name[5:]: np.prod([k.get_kernel() for k in mks])}
            )
        knode = self.microkernel_atype.get_kernel()(**knode_dict)
        kedge_dict = {}
        for mks in self.microkernels_bond:
            kedge_dict.update(
                {mks[0].name[5:]: np.prod([k.get_kernel() for k in mks])}
            )
        kedge = self.microkernel_btype.get_kernel()(**kedge_dict)
        p_dict = {}
        for mks in self.microkernels_probability:
            p_dict.update({mks[0].name[12:]: np.prod([k.get_kernel() for k in mks])})
        p = self.microkernel_ptype.get_kernel()(**p_dict)

        kernel = MGK(
            node_kernel=knode,
            edge_kernel=kedge,
            q=self.microkernel_q.value,
            q_bounds=self.microkernel_q.bounds,
            p=p,
            unique=self.unique,
        )
        self.kernel = self.microkernel_normalization.get_kernel()(kernel)

    def get_trial(self, trial) -> Dict:
        trials = []
        for mks in (
            self.microkernels_atom
            + self.microkernels_bond
            + self.microkernels_probability
        ):
            for mk in mks:
                trials.append(mk.get_trial(trial))
        trials.append(self.microkernel_normalization.get_trial(trial))
        trials.append(self.microkernel_atype.get_trial(trial))
        trials.append(self.microkernel_btype.get_trial(trial))
        trials.append(self.microkernel_ptype.get_trial(trial))
        trials.append(self.microkernel_q.get_trial(trial))
        return self.combine_dicts(trials)

    def update_from_trial(self, trial: Dict[str, Any]):
        for mks in (
            self.microkernels_atom
            + self.microkernels_bond
            + self.microkernels_probability
        ):
            for mk in mks:
                mk.update_from_trial(trial)
        self.microkernel_normalization.update_from_trial(trial)
        self.microkernel_atype.update_from_trial(trial)
        self.microkernel_btype.update_from_trial(trial)
        self.microkernel_ptype.update_from_trial(trial)
        self.microkernel_q.update_from_trial(trial)

    def update_from_theta(self):
        """gradient optimization function. Update the hyperparameters from kernel.theta."""
        values = np.exp(self.kernel.theta).tolist()
        for mks in self.microkernels_probability:
            for microkernel in mks:
                microkernel.update_from_theta(values)
        self.microkernel_q.update_from_theta(values)
        for mks in self.microkernels_atom:
            for microkernel in mks:
                microkernel.update_from_theta(values)
        for mks in self.microkernels_bond:
            for microkernel in mks:
                microkernel.update_from_theta(values)
        if isinstance(self.microkernel_normalization.value, float):
            self.microkernel_normalization.update_from_theta(values)
        assert len(values) == 0
    
    def save(self, path: str, name: str = "graph_hyperparameters.json"):
        hyperdict = {}
        for mks in (
            self.microkernels_atom
            + self.microkernels_bond
            + self.microkernels_probability
        ):
            for mk in mks:
                mk.update_hyperdict(hyperdict)
        self.microkernel_normalization.update_hyperdict(hyperdict)
        self.microkernel_atype.update_hyperdict(hyperdict)
        self.microkernel_btype.update_hyperdict(hyperdict)
        self.microkernel_ptype.update_hyperdict(hyperdict)
        self.microkernel_q.update_hyperdict(hyperdict)
        open(os.path.join(path, name), "w").write(
            json.dumps(hyperdict, indent=1, sort_keys=False)
        )

    @classmethod
    def load(
        cls,
        path: str,
        name: str = "graph_hyperparameters.json",
        idx: int = 0,
        unique: bool = False,
    ):
        hyperdict = json.loads(open(os.path.join(path, name), "r").read())
        return cls(hyperdict=hyperdict, idx=idx, unique=unique)

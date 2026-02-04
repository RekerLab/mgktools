from typing import Literal
from mgktools.models.regression.gpr.gpr import GaussianProcessRegressor
from graphdot.model.gaussian_process.nystrom import LowRankApproximateGPR
from mgktools.models.regression.scalable.NLE import NaiveLocalExpertGP
from mgktools.models.regression.consensus import EnsembleRegressor
from mgktools.models.regression.svr import SVRegressor
from mgktools.models.classification.gpc.gpc import GaussianProcessClassifier
from mgktools.models.classification.svm.svm import SVMClassifier


def set_model(model_type: Literal['gpr', 'gpr-nystrom', 'gpr-nle', 'svr', 'gpc', 'svc'],
              kernel,
              graph_kernel_type: Literal['graph', 'precomputed', 'no'] = None,
              # gpr
              optimizer = None,
              alpha: float = None,
              # svm
              C: float = None,
              # sod
              n_estimators: int = None,
              n_samples_per_model: int = None,
              ensemble_rule: Literal['smallest_uncertainty', 'weight_uncertainty', 'mean'] = 'weight_uncertainty',
              n_jobs: int = 1):
    if model_type == 'gpr':
        assert alpha is not None
        model = GaussianProcessRegressor(
            kernel=kernel,
            optimizer=optimizer,
            alpha=alpha,
            normalize_y=False,
        )
        if n_estimators is not None and n_estimators > 1:
            return EnsembleRegressor(
                model,
                n_estimators=n_estimators,
                n_samples_per_model=n_samples_per_model,
                n_jobs=n_jobs,
                ensemble_rule=ensemble_rule
            )
    elif model_type == 'gpr-nystrom':
        assert alpha is not None
        model = LowRankApproximateGPR(
            kernel=kernel,
            optimizer=optimizer,
            alpha=alpha,
            normalize_y=True,
        )
    elif model_type == 'gpr-nle':
        assert alpha is not None
        n_jobs = 1 if graph_kernel_type == 'graph' else n_jobs
        model = NaiveLocalExpertGP(
            kernel=kernel,
            alpha=alpha,
            n_local=n_samples_per_model,
            n_jobs=n_jobs
        )
    elif model_type == 'svr':
        assert C is not None
        model = SVRegressor(kernel=kernel, C=C)
    elif model_type == 'gpc':
        model = GaussianProcessClassifier(
            kernel=kernel,
            optimizer=optimizer,
            n_jobs=n_jobs
        )
    elif model_type == 'svc':
        assert C is not None
        model = SVMClassifier(
            kernel=kernel,
            C=C,
            probability=True
        )
    else:
        raise RuntimeError(f'unsupported model_type: {model_type}')
    return model

#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Dict, List, Union, Literal, Tuple
import numpy as np
import optuna
from optuna.samplers import TPESampler
from mgktools.data.data import Dataset
from mgktools.models import set_model
from mgktools.evaluators.cross_validation import Evaluator, Metric
from mgktools.kernels.PreComputed import calc_precomputed_kernel_config
from mgktools.kernels.base import BaseKernelConfig


def evaluate_model(dataset, kernel_config, model_type, task_type, metric, **kwargs):
    """Evaluate model performance for a single dataset."""
    alpha = kwargs.pop('alpha', 0.01)
    C = kwargs.pop('C', 10)
    
    if dataset.graph_kernel_type == "graph":
        kernel = calc_precomputed_kernel_config(kernel_config, dataset).kernel
        dataset.graph_kernel_type = "pre-computed"
        tag = True
    else:
        kernel = kernel_config.kernel
        tag = False
        
    model = set_model(model_type=model_type, kernel=kernel, alpha=alpha, C=C)
    evaluator = Evaluator(
        dataset=dataset,
        model=model,
        task_type=task_type,
        metrics=[metric],
        verbose=False,
        **kwargs
    )
    score = evaluator.run_cross_validation()
    
    if tag:
        dataset.graph_kernel_type = "graph"
    dataset.clear_cookie()
    
    return score


def save_optimization_results(save_dir: str, best_params: Dict, kernel_config) -> None:
    """Save optimization hyperparameters to files."""
    # Save regularization parameters
    for param in ['alpha', 'C']:
        if param in best_params:
            with open(f"{save_dir}/{param}", "w") as f:
                f.write(str(best_params.pop(param)))
    
    kernel_config.update_from_space(best_params)
    kernel_config.save(path=save_dir)


def bayesian_optimization(
    save_dir: str,
    datasets: List[Dataset],
    kernel_config: BaseKernelConfig,
    task_type: Literal["regression", "binary", "multi-class"],
    model_type: Literal["gpr", "gpr-sod", "gpr-nystrom", "gpr-nle", "svr", "gpc", "svc"],
    metric: Literal[Metric, "log_likelihood"],
    cross_validation: Literal["n-fold", "leave-one-out", "Monte-Carlo"],
    n_splits: int = None,
    split_type: Literal['random', 'scaffold_order', 'scaffold_random'] = None,
    split_sizes: List[float] = None,
    num_folds: int = 10,
    num_iters: int = 100,
    alpha: float = None,
    alpha_bounds: Tuple[float, float] = None,
    d_alpha: float = None,
    C: float = None,
    C_bounds: Tuple[float, float] = None,
    d_C: float = None,
    load_if_exists: bool = True,
    seed: int = 0,
):
    """ Perform Bayesian optimization for hyperparameter tuning.

    Parameters:
    -----------
    save_dir: str
        Directory to save the optimization results.
    datasets: List of Dataset objects. 
        This is designed for splitting a large dataset into small subsets to reduce the computational costs.
    kernel_config: BaseKernelConfig object
        Kernel configuration object.
    task_type: str
        Task type, either "regression", "binary", or "multi-class".
    model_type: str
        Model type, either "gpr", "gpr-sod", "gpr-nystrom", "gpr-nle", "svr", "gpc", or "svc".
    metric: str
        Evaluation metric.
    cross_validation: str
        Cross-validation method, either "n-fold", "leave-one-out", or "Monte-Carlo".
    n_splits: int
        Number of folds for n-fold cross-validation. Valid only if cross_validation is "n-fold".
    split_type: str
        Split type for Monte-Carlo cross-validation. Valid only if cross_validation is "Monte-Carlo".
    split_sizes: List of float
        Split sizes for Monte-Carlo cross-validation. Valid only if cross_validation is "Monte-Carlo".
    num_folds: int
        Number of repeats for cross-validation. Must be 1 for leave-one-out cross-validation.
    num_iters: int
        Number of iterations for Bayesian optimization.
    alpha: float
        Important parameter for Gaussian Process Regression (GPR). Value added to the diagonal of the 
        kernel matrix during fitting. This can prevent a potential numerical issue during fitting, 
        by ensuring that the calculated values form a positive definite matrix. It can also be 
        interpreted as the variance of additional Gaussian measurement noise on the training observations.
    alpha_bounds: Tuple of float
        Optimization bounds for the alpha.
    d_alpha: float
        Optimization step size for alpha.
    C: float
        Regularization parameter for Support Vector Machine (SVM). The strength of the regularization 
        is inversely proportional to C. Must be strictly positive. The penalty is a squared l2 penalty.
    C_bounds: Tuple of float
        Optimization bounds for the C parameter.
    d_C: float
        Optimization step size for C.
    load_if_exists: bool
        Whether to load the existing optimization results.
    seed: int
        Random seed for Optuna.
    """
    # Input check
    if task_type == "regression":
        assert model_type in ["gpr", "gpr-sod", "gpr-nystrom", "gpr-nle", "svr"]
    elif task_type == "binary":
        assert model_type in ["gpr", "gpc", "svc"]
    else:
        # assert model_type in ["gpc", "svc"]
        raise NotImplementedError("Multi-class classification is not supported yet.")
    if model_type.startswith("gpr"):
        assert alpha is not None
        assert C is None and C_bounds is None and d_C is None
    elif model_type in ["svc", "svr"]:
        assert C is not None
        assert alpha is None and alpha_bounds is None and d_alpha is None

    if metric in ["rmse", "mae", "mse", "max"]:
        maximize = False
    else:
        maximize = True
    if cross_validation == "loocv":
        assert num_folds == 1
    elif cross_validation == "n-fold":
        assert n_splits is not None
    elif cross_validation == "Monte-Carlo":
        assert split_type is not None
        assert split_sizes is not None

    def objective(trial) -> Union[float, np.ndarray]:
        hyperdict = kernel_config.get_trial(trial)

        # Handle alpha parameter
        if alpha_bounds:
            assert model_type.startswith("gpr"), "Alpha parameter only supported for GPR models"
            alpha_param = trial.suggest_float(
                name="alpha", 
                low=alpha_bounds[0], 
                high=alpha_bounds[1],
                step=d_alpha,
                log=(d_alpha is None)
            )
            hyperdict["alpha"] = alpha_param

        # Handle C parameter
        if C_bounds:
            C_param = trial.suggest_float(
                name="C",
                low=C_bounds[0],
                high=C_bounds[1],
                step=d_C,
                log=(d_C is None)
            )
            if C_param:
                hyperdict["C"] = C_param

        curr_alpha = hyperdict.pop("alpha", alpha)
        curr_C = hyperdict.pop("C", C)
        kernel_config.update_from_trial(hyperdict)
        kernel_config.update_kernel()
        if metric == "log_likelihood":
            assert model_type in ["gpr", "gpr-sod", "gpr-nystrom", "gpr-nle"], "Log likelihood only supported for GPR models"
            kernel = kernel_config.kernel
            model = set_model(model_type=model_type, kernel=kernel, alpha=curr_alpha)
            scores = []
            for dataset in datasets:
                scores.append(model.log_marginal_likelihood(X=dataset.X, y=dataset.y))
                dataset.clear_cookie()
            return np.mean(scores)
        else:
            scores = []
            for dataset in datasets:
                score = evaluate_model(
                    dataset=dataset,
                    kernel_config=kernel_config,
                    model_type=model_type,
                    task_type=task_type,
                    metric=metric,
                    save_dir=save_dir,
                    cross_validation=cross_validation,
                    n_splits=n_splits,
                    split_type=split_type,
                    split_sizes=split_sizes,
                    num_folds=num_folds,
                    alpha=curr_alpha,
                    C=curr_C,
                )
                scores.append(score)
            score = np.mean(scores)
            if maximize:
                return - score
            else:
                return score

    study = optuna.create_study(
        study_name="optuna-study",
        sampler=TPESampler(seed=seed),
        storage="sqlite:///%s/optuna.db" % save_dir,
        load_if_exists=load_if_exists,
        direction="minimize"
    )
    n_to_run = num_iters - len(study.trials)
    if n_to_run > 0:
        study.optimize(objective, n_trials=n_to_run)
    save_optimization_results(save_dir=save_dir, best_params=study.best_params, kernel_config=kernel_config)
    # optuna.delete_study(study_name="optuna-study", storage="sqlite:///%s/optuna.db" % save_dir)


def bayesian_optimization_gpr_multi_datasets(
    save_dir: str,
    kernel_config: BaseKernelConfig,
    datasets: List[Dataset],
    tasks_type: List[Literal["regression", "binary"]],
    metrics: List[Literal[Metric]],
    cross_validation: Literal["n-fold", "leave-one-out", "Monte-Carlo"],
    n_splits: int = None,
    split_type: Literal['random', 'scaffold_order', 'scaffold_random'] = None,
    split_sizes: List[float] = None,
    num_folds: int = 10,
    num_iters: int = 100,
    alpha: float = None,
    alpha_bounds: Tuple[float, float] = None,
    d_alpha: float = None,
    seed: int = 0,
):
    """ Perform Bayesian optimization for hyperparameter tuning by maximizing the mean performance of 
    several datasets. Gaussian Process Regression is used as the model.

    Parameters:
    -----------
    save_dir: str
        Directory to save the optimization results.
    kernel_config: BaseKernelConfig object
        Kernel configuration object.
    datasets: List of Dataset objects.
        List of datasets for series of cross-validation, respectively. The mean performance of 
    task_type: str
        List of task types for each dataset, either "regression" or "binary".
    metrics: str
        List of evaluation metrics for each dataset.
    cross_validation: str
        Cross-validation method, either "n-fold", "leave-one-out", or "Monte-Carlo".
    n_splits: int
        Number of folds for n-fold cross-validation. Valid only if cross_validation is "n-fold".
    split_type: str
        Split type for Monte-Carlo cross-validation. Valid only if cross_validation is "Monte-Carlo".
    split_sizes: List of float
        Split sizes for Monte-Carlo cross-validation. Valid only if cross_validation is "Monte-Carlo".
    num_folds: int
        Number of repeats for cross-validation. Must be 1 for leave-one-out cross-validation.
    num_iters: int
        Number of iterations for Bayesian optimization.
    alpha: float
        Important parameter for Gaussian Process Regression (GPR). Value added to the diagonal of the 
        kernel matrix during fitting. This can prevent a potential numerical issue during fitting, 
        by ensuring that the calculated values form a positive definite matrix. It can also be 
        interpreted as the variance of additional Gaussian measurement noise on the training observations.
    alpha_bounds: Tuple of float
        Optimization bounds for the alpha.
    d_alpha: float
        Optimization step size for alpha.
    seed: int
        Random seed for Optuna.
    """
    def objective(trial) -> Union[float, np.ndarray]:
        hyperdict = kernel_config.get_trial(trial)

        # Handle alpha parameter
        if alpha_bounds:
            alpha_param = trial.suggest_float(
                name="alpha", 
                low=alpha_bounds[0], 
                high=alpha_bounds[1],
                step=d_alpha,
                log=(d_alpha is None)
            )
            hyperdict["alpha"] = alpha_param

        curr_alpha = hyperdict.pop("alpha", alpha)
        kernel_config.update_from_trial(hyperdict)
        kernel_config.update_kernel()
        scores = []
        for i, dataset in enumerate(datasets):
            score = evaluate_model(
                dataset=dataset,
                kernel_config=kernel_config,
                model_type='gpr',
                task_type=tasks_type[i],
                metric=metrics[i],
                save_dir=save_dir,
                cross_validation=cross_validation,
                n_splits=n_splits,
                split_type=split_type,
                split_sizes=split_sizes,
                num_folds=num_folds,
                alpha=curr_alpha,
            )
            if metrics[i] in ["rmse", "mae", "mse", "max"]:
                scores.append(score)
            else:
                scores.append(-score)
        return np.mean(scores)

    study = optuna.create_study(
        study_name="optuna-study",
        sampler=TPESampler(seed=seed),
        storage="sqlite:///%s/optuna.db" % save_dir,
        load_if_exists=True,
        direction="minimize"
    )
    n_to_run = num_iters - len(study.trials)
    if n_to_run > 0:
        study.optimize(objective, n_trials=n_to_run)
    save_optimization_results(save_dir=save_dir, best_params=study.best_params, kernel_config=kernel_config)

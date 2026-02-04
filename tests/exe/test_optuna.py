# !/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import os
import tempfile
import pandas as pd
from mgktools.hyperparameters import (
    additive, additive_pnorm, additive_msnorm, additive_norm,
    product, product_pnorm, product_msnorm, product_norm,
    rbf, dot_product
)
from mgktools.evaluators.metric import AVAILABLE_METRICS_REGRESSION, AVAILABLE_METRICS_BINARY
from mgktools.exe.run import mgk_optuna, mgk_optuna_multi_datasets


CWD = os.path.dirname(os.path.abspath(__file__))


@pytest.mark.parametrize("input1", [
    ("freesolv", ["smiles"], ["freesolv"], [], "regression"),
    ("bace", ["smiles"], ["bace"], [], "binary"),
    ("clintox", ["smiles"], ["FDA_APPROVED", "CT_TOX"], [], "binary"),
    ("np", ["smiles1", "smiles2"], ["np"], [], "binary"),
    ("st", ["smiles"], ["st"], ["T"], "regression"),
])
@pytest.mark.parametrize("input2", [
    (None, None, None),
    # (["morgan"], "concat", 2048),
    (["rdkit_2d_normalized"], "concat", 200),
    # (["rdkit_2d", "morgan_count"], "concat", 2248),
    # (["rdkit_2d", "morgan_count"], "mean", 2248)
])
@pytest.mark.parametrize("features_scaling", [True, False])
@pytest.mark.parametrize("input3", [
    ("gpr", None),
    # ("svr", None),
    # ("gpr-sod", ("1", "4", "mean")),
    # ("gpr-sod", ("2", "4", "mean")),
    # ("gpr-sod", ("3", "3", "smallest_uncertainty")),
    # ("gpr-sod", ("4", "2", "weight_uncertainty")), # prediction is nan in some cases.
    # ("gpr-nystrom", "4"),
    # ("gpr-nle", "4"),
    # ("gpc", None),
    ("svc", None),
])
@pytest.mark.parametrize("input4", [
    ("leave-one-out", None, None, "1"),
    # ("Monte-Carlo", None, "random", "5"),
    # ("Monte-Carlo", None, "scaffold_order", "5"),
    # ("Monte-Carlo", None, "scaffold_random", "5"),
    # ("kFold", "5", None, "1"),
    # ("no", None, None, "1")
])
@pytest.mark.parametrize("graph_hyperparameters", [
    None, additive_pnorm, product_msnorm
    # additive, additive_pnorm, additive_msnorm, additive_norm,
    # product, product_pnorm, product_msnorm, product_norm,
])
@pytest.mark.parametrize("use_cache", [True, False])
@pytest.mark.parametrize("features_hyperparameters", [rbf, dot_product])
@pytest.mark.parametrize("metric", ["rmse", "roc_auc"])
@pytest.mark.parametrize("num_iters", [10])
@pytest.mark.parametrize("opt_alpha_or_C", [True, False])
@pytest.mark.parametrize("num_splits", [1, 2])
def test_Optuna(input1, input2, features_scaling, input3, input4, 
                         graph_hyperparameters, use_cache,
                         features_hyperparameters, metric,
                         num_iters, opt_alpha_or_C, num_splits):
    dataset, smiles_columns, targets_columns, features_columns, task_type = input1
    features_generators, features_combination, n_features = input2
    model_type, model_params = input3
    cross_validation, n_splits, split, num_folds = input4
    graph_kernel_type = 'graph' if graph_hyperparameters is not None else 'no'
    save_dir = tempfile.mkdtemp()
    ### skip the invalid input combinations
    if task_type == "regression":
        # Skip regression datasets for gpc and svc
        if model_type in ["gpc", "svc"]:
            return
        # Skip regression datasets for classification metrics
        if metric in AVAILABLE_METRICS_BINARY:
            return
    elif task_type == "binary":
        # Skip binary classification datasets for gpr-sod, gpr-nystrom, gpr-nle, svr
        if model_type in ["gpr-sod", "gpr-nystrom", "gpr-nle", "svr"]:
            return
        # Skip binary classification datasets for regression metrics
        if metric in AVAILABLE_METRICS_REGRESSION:
            return
    if graph_hyperparameters is None and features_generators is None:
        # You must use graph kernel or feature kernel.
        return
    if features_generators is None and len(features_columns) == 0 and features_scaling:
        # Skip features scaling when no features are used
        return
    if features_generators is None:
        if features_hyperparameters != rbf:
            return
    if cross_validation == "no":
        if graph_kernel_type == "pre-computed":
            return
        ext_test_path = f"{CWD}/data/{dataset}_test.csv"
        if not os.path.exists(ext_test_path):
            return
    elif cross_validation == "leave-one-out":
        # Leave-one-out is only valid for GPR.
        if model_type != "gpr":
            return
    elif cross_validation == "Monte-Carlo":
        if len(smiles_columns) > 1 and split.startswith("scaffold"):
            return
    # hyperparameters optimization using Optuna
    arguments = [
        "--save_dir", save_dir,
        "--data_path", f"{CWD}/data/{dataset}.csv",
        "--smiles_columns"] + smiles_columns + [
        "--targets_columns"] + targets_columns + [
        "--graph_kernel_type", graph_kernel_type,
        "--task_type", task_type,
        "--model_type", model_type.replace("gpr-sod", "gpr"),
        "--metric", metric,
        "--num_folds", num_folds,
        "--cross_validation", cross_validation
    ]
    if n_splits is not None:
        arguments += ["--n_splits", n_splits]
    if split is not None:
        arguments += ["--split_type", split, "--split_sizes", "0.8", "0.2"]
    if graph_kernel_type == "graph":
        arguments += ["--graph_hyperparameters"] + [graph_hyperparameters] * len(smiles_columns)
    if features_columns:
        arguments += ["--features_columns"] + features_columns
    if features_generators is not None:
        arguments += ["--features_generators_name"] + features_generators
        arguments += ["--features_combination", features_combination]
    if features_generators is not None or features_columns:
        arguments += ["--features_hyperparameters", features_hyperparameters]
    if features_scaling:
        if features_columns:
            arguments += ['--features_add_normalize']
        if features_generators:
            arguments += ['--features_mol_normalize']
    if use_cache:
        arguments += ["--cache_path", f"{CWD}/tmp/cache.pkl"]
    if model_type.startswith("gpr"):
        arguments += ["--alpha", "0.01"]
        if opt_alpha_or_C:
            arguments += ["--alpha_bounds", "0.001", "0.3"]
        if model_type == "gpr-sod":
            arguments += ["--ensemble", "--n_estimators", model_params[0], "--n_samples_per_model", model_params[1], "--ensemble_rule", model_params[2]]
        elif model_type == "gpr-nystrom":
            arguments += ["--n_core", model_params]
        elif model_type == "gpr-nle":
            arguments += ["--n_local", model_params]
    elif model_type.startswith("sv"):
        arguments += ["--C", "1.0"]
        if opt_alpha_or_C:
            arguments += ["--C_bounds", "0.01", "100.0"]
    arguments += ["--num_iters", str(num_iters),
                  "--num_splits", str(num_splits)]
    mgk_optuna(arguments)
    assert os.path.exists(f"{save_dir}/optuna.db")
    os.remove(f"{save_dir}/optuna.db")


@pytest.mark.parametrize("input1", [
    (["freesolv", "bace", "clintox"], "smiles;smiles;smiles", "freesolv;bace;FDA_APPROVED,CT_TOX", ["regression", "binary", "binary"], ["rmse", "roc_auc", "roc_auc"]),
])
@pytest.mark.parametrize("input2", [
    (None, None, None),
    # (["morgan"], "concat", 2048),
    (["rdkit_2d_normalized"], "concat", 200),
    # (["rdkit_2d", "morgan_count"], "concat", 2248),
    # (["rdkit_2d", "morgan_count"], "mean", 2248)
])
@pytest.mark.parametrize("features_scaling", [True, False])
@pytest.mark.parametrize("graph_hyperparameters", [
    None, additive_pnorm, product_msnorm
    # additive, additive_pnorm, additive_msnorm, additive_norm,
    # product, product_pnorm, product_msnorm, product_norm,
])
@pytest.mark.parametrize("use_cache", [True, False])
@pytest.mark.parametrize("features_hyperparameters", [rbf, dot_product])
@pytest.mark.parametrize("opt_alpha_or_C", [True, False])
def test_OptunaMultiDatasets(input1, input2, features_scaling, graph_hyperparameters, use_cache, 
                             features_hyperparameters, opt_alpha_or_C):
    datasets, smiles_columns, targets_columns, tasks_type, metrics = input1
    data_paths = [f"{CWD}/data/{d}.csv" for d in datasets]
    features_generators, features_combination, n_features = input2
    graph_kernel_type = "graph" if graph_hyperparameters is not None else "no"
    if graph_kernel_type == "no" and features_generators is None:
        # You must use graph kernel or feature kernel.
        return
    # hyperparameters optimization using Optuna
    save_dir = tempfile.mkdtemp()
    arguments = [
        "--save_dir", save_dir,
        "--data_paths"] + data_paths + [
        "--smiles_columns", smiles_columns,
        "--targets_columns", targets_columns,
        "--tasks_type"] + tasks_type + [
        "--metrics"] + metrics + [
        "--graph_kernel_type", graph_kernel_type,
    ]
    if graph_kernel_type == "graph":
        arguments += ["--graph_hyperparameters", graph_hyperparameters]

    if features_generators is not None:
        arguments += ["--features_generators_name"] + features_generators
        arguments += ["--features_combination", features_combination,
                      "--features_hyperparameters", features_hyperparameters]
        if features_scaling:
            arguments += ["--features_mol_normalize"]
    if use_cache:
        arguments += ["--cache_path", f"{CWD}/tmp/cache.pkl"]
    arguments += ["--alpha", "0.01"]
    if opt_alpha_or_C:
        arguments += ["--alpha_bounds", "0.001", "0.3"]
    mgk_optuna_multi_datasets(arguments)
    for file in ["optuna.db"]:
        assert os.path.exists(f'{save_dir}/{file}'), f'{save_dir}/{file} does not exist.'
        os.remove(f'{save_dir}/{file}')

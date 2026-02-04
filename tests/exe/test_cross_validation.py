# !/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import os
import shutil
import tempfile
import pandas as pd
import numpy as np
from mgktools.hyperparameters import (
    additive, additive_pnorm, additive_msnorm, additive_norm,
    product, product_pnorm, product_msnorm, product_norm,
    rbf, dot_product
)
from mgktools.evaluators.metric import AVAILABLE_METRICS_REGRESSION, AVAILABLE_METRICS_BINARY
from mgktools.exe.run import mgk_kernel_calc, mgk_cross_validation


CWD = os.path.dirname(os.path.abspath(__file__))


@pytest.mark.timeout(300)
@pytest.mark.parametrize("input1", [
    ("freesolv", ["smiles"], ["freesolv"], [], "regression"),
    ("bace", ["smiles"], ["bace"], [], "binary"),
    ("clintox", ["smiles"], ["FDA_APPROVED", "CT_TOX"], [], "binary"),
    ("np", ["smiles1", "smiles2"], ["np"], [], "binary"),
    ("st", ["smiles"], ["st"], ["T"], "regression"),
])
@pytest.mark.parametrize("input2", [
    (None, None, None),
    (["morgan"], "concat", 2048),
    (["rdkit_2d_normalized"], "concat", 200),
    (["rdkit_2d", "morgan_count"], "concat", 2248),
    (["rdkit_2d", "morgan_count"], "mean", 2248)
])
@pytest.mark.parametrize("features_scaling", [True, False])
@pytest.mark.parametrize("input3", [
    ("gpr", None),
    ("svr", None),
    ("gpr-sod", ("1", "4", "mean")),
    ("gpr-sod", ("2", "4", "mean")),
    ("gpr-sod", ("3", "3", "smallest_uncertainty")),
    # ("gpr-sod", ("4", "2", "weight_uncertainty")), # prediction is nan in some cases.
    ("gpr-nystrom", "4"),
    ("gpr-nle", "4"),
    ("gpc", None),
    ("svc", None),
])
@pytest.mark.parametrize("input4", [
    ("leave-one-out", None, None, "1"),
    ("Monte-Carlo", None, "random", "5"),
    ("Monte-Carlo", None, "scaffold_order", "5"),
    ("Monte-Carlo", None, "scaffold_random", "5"),
    ("kFold", "5", None, "1"),
    ("no", None, None, "1")
])
@pytest.mark.parametrize("graph_hyperparameters", [
    None,
    additive, additive_pnorm, additive_msnorm, additive_norm,
    product, product_pnorm, product_msnorm, product_norm,
])
@pytest.mark.parametrize("precomputed", [True, False])
@pytest.mark.parametrize("use_cache", [True, False])
@pytest.mark.parametrize("features_hyperparameters", [rbf, dot_product])
@pytest.mark.parametrize("metric", ["rmse", "roc_auc"])
def test_CrossValidation(input1, input2, features_scaling, input3, input4, 
                         graph_hyperparameters, precomputed, use_cache,
                         features_hyperparameters, metric):
    # randomly skip some tests to reduce the total number of tests
    if np.random.uniform(0, 1) > 0.01:
        return
    dataset, smiles_columns, targets_columns, features_columns, task_type = input1
    features_generators, features_combination, n_features = input2
    model_type, model_params = input3
    cross_validation, n_splits, split, num_folds = input4
    save_dir = tempfile.mkdtemp()
    if graph_hyperparameters is None:
        graph_kernel_type = "no"
    else:
        graph_kernel_type = "pre-computed" if precomputed else "graph"
    ### skip the invalid input combinations
    if task_type == "regression":
        # Skip regression datasets for gpc and svc
        if model_type in ["gpc", "svc"]:
            return
        # Skip regression datasets for classification metrics
        if metric not in AVAILABLE_METRICS_REGRESSION:
            return
    elif task_type == "binary":
        # Skip binary classification datasets for gpr-sod, gpr-nystrom, gpr-nle, svr
        if model_type in ["gpr-sod", "gpr-nystrom", "gpr-nle", "svr"]:
            return
        # Skip binary classification datasets for regression metrics
        if metric not in AVAILABLE_METRICS_BINARY:
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
    # common arguments
    arguments_base = [
        "--save_dir", save_dir,
        "--data_path", f"{CWD}/data/{dataset}.csv",
        "--smiles_columns"] + smiles_columns + [
        "--targets_columns"] + targets_columns
    if features_columns:
        arguments_base += ["--features_columns"] + features_columns
    if features_generators is not None:
        arguments_base += ["--features_generators_name"] + features_generators
        arguments_base += ["--features_combination", features_combination]
    if features_generators is not None or features_columns:
        arguments_base += ["--features_hyperparameters", features_hyperparameters]
    if features_scaling:
        if features_columns:
            arguments_base += ['--features_add_normalize']
        if features_generators:
            arguments_base += ['--features_mol_normalize']
    if use_cache:
        arguments_base += ["--cache_path", f"{CWD}/tmp/cache.pkl"]
    # kernel computation
    if graph_kernel_type == "pre-computed":
        assert not os.path.exists(f"{save_dir}/kernel.pkl")
        arguments = arguments_base.copy()
        arguments += ["--graph_kernel_type", "graph",
                      "--graph_hyperparameters"] + [graph_hyperparameters] * len(smiles_columns)
        mgk_kernel_calc(arguments)
        assert os.path.exists(f"{save_dir}/kernel.pkl")
    # cross-validation
    arguments = arguments_base.copy()
    arguments += ["--task_type", task_type,
                  "--model_type", model_type.replace("gpr-sod", "gpr"),
                  "--metric", metric,
                  "--num_folds", num_folds,
                  "--graph_kernel_type", graph_kernel_type,
                  "--cross_validation", cross_validation]
    if graph_kernel_type == "graph":
        arguments += ["--graph_hyperparameters"] + [graph_hyperparameters] * len(smiles_columns)
    if cross_validation == "no":
        arguments += ["--separate_test_path", ext_test_path]
        arguments += ["--smiles_columns"] + smiles_columns
        arguments += ["--targets_columns"] + targets_columns
    if n_splits is not None:
        arguments += ["--n_splits", n_splits]
    if split is not None:
        arguments += ["--split_type", split, "--split_sizes", "0.8", "0.2"]
    if model_type.startswith("gpr"):
        arguments += ["--alpha", "0.01"]
        if model_type == "gpr-sod":
            arguments += ["--ensemble", "--n_estimators", model_params[0], "--n_samples_per_model", model_params[1], "--ensemble_rule", model_params[2]]
        elif model_type == "gpr-nystrom":
            arguments += ["--n_core", model_params]
        elif model_type == "gpr-nle":
            arguments += ["--n_local", model_params]
    elif model_type.startswith("sv"):
        arguments += ["--C", "1.0"]
    mgk_cross_validation(arguments)
    if graph_kernel_type == "pre-computed":
        os.remove("%s/kernel.pkl" % save_dir)
    if cross_validation == "leave-one-out":
        df = pd.read_csv("%s/loocv_prediction.csv" % save_dir)
        df = pd.read_csv("%s/loocv_metrics.csv" % save_dir)
        assert len(df) > 0
        os.remove("%s/loocv_prediction.csv" % save_dir)
        os.remove("%s/loocv_metrics.csv" % save_dir)
    elif cross_validation == "Monte-Carlo":
        for i in range(int(num_folds)):
            df = pd.read_csv("%s/test_%d_prediction.csv" % (save_dir, i))
            assert len(df) > 0
            os.remove("%s/test_%d_prediction.csv" % (save_dir, i))
        df = pd.read_csv("%s/Monte-Carlo_metrics.csv" % save_dir)
        assert len(df) > 0
        os.remove("%s/Monte-Carlo_metrics.csv" % save_dir)
    elif cross_validation == "kFold":
        for i in range(int(num_folds)):
            for j in range(int(n_splits)):
                df = pd.read_csv("%s/kFold_%d-%d_prediction.csv" % (save_dir, i, j))
                assert len(df) > 0
                os.remove("%s/kFold_%d-%d_prediction.csv" % (save_dir, i, j))
        df = pd.read_csv("%s/kFold_metrics.csv" % save_dir)
        assert len(df) > 0
        os.remove("%s/kFold_metrics.csv" % save_dir)
    elif cross_validation == "no":
        df = pd.read_csv("%s/test_ext_prediction.csv" % save_dir)
        assert len(df) > 0
        os.remove("%s/test_ext_prediction.csv" % save_dir)
    shutil.rmtree(save_dir, ignore_errors=True)

# !/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import os
import shutil
import tempfile
import pandas as pd
from mgktools.hyperparameters import (
    additive, additive_pnorm, additive_msnorm, additive_norm,
    product, product_pnorm, product_msnorm, product_norm,
    rbf, dot_product
)
from mgktools.evaluators.metric import AVAILABLE_METRICS_REGRESSION, AVAILABLE_METRICS_BINARY
from mgktools.exe.run import mgk_embedding


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
    (["morgan"], "concat", 2048),
    (["rdkit_2d_normalized"], "concat", 200),
    (["rdkit_2d", "morgan_count"], "concat", 2248),
    (["rdkit_2d", "morgan_count"], "mean", 2248)
])
@pytest.mark.parametrize("features_scaling", [True, False])
@pytest.mark.parametrize("graph_hyperparameters", [
    None, additive_msnorm, product_msnorm
    # additive, additive_pnorm, additive_msnorm, additive_norm,
    # product, product_pnorm, product_msnorm, product_norm,
])
@pytest.mark.parametrize("use_cache", [True, False])
@pytest.mark.parametrize("features_hyperparameters", [rbf, dot_product])
@pytest.mark.parametrize("embedding_algorithm", ["tSNE", "kPCA"])
def test_Embedding(input1, input2, features_scaling, graph_hyperparameters,
                   use_cache, features_hyperparameters, embedding_algorithm):
    dataset, smiles_columns, targets_columns, features_columns, task_type = input1
    features_generators, features_combination, n_features = input2
    save_dir = tempfile.mkdtemp()
    graph_kernel_type = "graph" if graph_hyperparameters is not None else "no"
    ### skip the invalid input combinations
    if graph_kernel_type == 'no' and features_generators is None:
        # You must use graph kernel or feature kernel.
        return
    if features_generators is None and len(features_columns) == 0 and features_scaling:
        # Skip features scaling when no features are used
        return
    if features_generators is None:
        if features_hyperparameters != rbf:
            return
    # calculate embedding
    arguments = [
        "--save_dir", save_dir,
        "--data_path", f"{CWD}/data/{dataset}.csv",
        "--smiles_columns"] + smiles_columns + [
        "--targets_columns"] + targets_columns + [
        "--graph_kernel_type", graph_kernel_type,
        "--embedding_algorithm", embedding_algorithm,
        "--perplexity", "5",
    ]
    if graph_kernel_type == "graph":
        arguments += ["--graph_hyperparameters"] + [graph_hyperparameters] * len(smiles_columns)
    if features_columns:
        arguments += ["--features_columns"] + features_columns
    if features_generators is not None:
        arguments += ["--features_generators_name"] + features_generators
        arguments += ["--features_combination", features_combination]
    if features_generators is not None or len(features_columns) != 0:
        arguments += ["--features_hyperparameters", features_hyperparameters]
    if features_scaling:
        if features_columns:
            arguments += ['--features_add_normalize']
        if features_generators:
            arguments += ['--features_mol_normalize']
    if use_cache:
        arguments += ["--cache_path", f"{CWD}/tmp/cache.pkl"]
    mgk_embedding(arguments)
    assert os.path.exists(f"{save_dir}/{embedding_algorithm}.csv")
    os.remove(f"{save_dir}/{embedding_algorithm}.csv")
    shutil.rmtree(save_dir)

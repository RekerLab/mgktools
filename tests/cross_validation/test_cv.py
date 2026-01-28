#!/usr/bin/env python
# -*- coding: utf-8 -*-
import shutil
import tempfile
import pandas as pd
import pytest
from mgktools.data.data import Dataset
from mgktools.kernels.utils import get_kernel_config
from mgktools.hyperparameters import *
from mgktools.models import set_model
from mgktools.evaluators.cross_validation import Evaluator


smiles = ['CCCC', 'CCCCCO', 'c1ccccc1', 'CCNCCO', 'CCCCN', 'NCCCCCO', 'c1ccccc1N', 'NCCNCCO',
        'CNC(CC)CC', 'c1ccccc1', 'c1ccccc1CCCCc1ccccc1', 'CC(=O)OCCO']
targets_regression = [3.1, 14.5, 25.6, 56.7, 9.1, 17.5, 22.6, 36.7, 23.1, 32.1, 1.4, 7.6]
df_regression = pd.DataFrame({'smiles': smiles, 'targets': targets_regression})
targets_classification = [1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1]
df_classification = pd.DataFrame({'smiles': smiles, 'targets': targets_classification})


@pytest.mark.parametrize('mgk_file', [additive_norm, additive_pnorm, additive_msnorm,
                                      product_norm, product_pnorm, product_msnorm])
@pytest.mark.parametrize('model', ['gpc', 'svc'])
@pytest.mark.parametrize('split_type', ['random', 'scaffold_order', 'scaffold_random'])
def test_only_graph_classification(mgk_file, model, split_type):
    dataset = Dataset.from_df(df=df_classification,
                              smiles_columns=['smiles'],
                              targets_columns=['targets'])
    dataset.set_status(graph_kernel_type='graph', features_generators=None, features_combination=None)
    dataset.create_graphs(n_jobs=4)
    dataset.unify_datatype()
    kernel_config = get_kernel_config(dataset=dataset,
                                      graph_kernel_type='graph',
                                      mgk_hyperparameters_files=[mgk_file])
    C = 1.0 if model == 'svc' else None
    model = set_model(model, kernel=kernel_config.kernel, C=C)
    save_dir = tempfile.mkdtemp()
    Evaluator(save_dir=save_dir,
              dataset=dataset,
              model=model,
              task_type='binary',
              metrics=['roc_auc', 'accuracy', 'precision', 'recall', 'f1_score', 'mcc'],
              cross_validation='Monte-Carlo',
              split_type=split_type,
              split_sizes=[0.75, 0.25],
              num_folds=2,
              verbose=True).run_cross_validation()
    shutil.rmtree(save_dir)


@pytest.mark.parametrize('n_splits', [4, 5])
def test_kFold_cv_classification(n_splits):
    model = 'gpc'
    mgk_file = additive_norm
    dataset = Dataset.from_df(df=df_classification,
                              smiles_columns=['smiles'],
                              targets_columns=['targets'])
    dataset.set_status(graph_kernel_type='graph', features_generators=None, features_combination=None)
    dataset.create_graphs(n_jobs=4)
    dataset.unify_datatype()
    kernel_config = get_kernel_config(dataset=dataset,
                                      graph_kernel_type='graph',
                                      mgk_hyperparameters_files=[mgk_file])
    C = 1.0 if model == 'svc' else None
    model = set_model(model, kernel=kernel_config.kernel, C=C)
    save_dir = tempfile.mkdtemp()
    Evaluator(save_dir=save_dir,
              dataset=dataset,
              model=model,
              task_type='binary',
              metrics=['roc_auc', 'accuracy', 'precision', 'recall', 'f1_score', 'mcc'],
              cross_validation='kFold',
              n_splits=n_splits,
              num_folds=2,
              verbose=True).run_cross_validation()
    shutil.rmtree(save_dir)


@pytest.mark.parametrize('mgk_file', [product_msnorm])
@pytest.mark.parametrize('modelsets', [('gpr', None, None, None),
                                       ('gpr', 2, 3, 'smallest_uncertainty'),
                                       # ('gpr', 2, 3, 'weight_uncertainty'),
                                       ('gpr', 2, 3, 'mean'),
                                       ('gpr-nystrom', None, 3, None),
                                       ('gpr-nle', None, 3, None)])
@pytest.mark.parametrize('split_type', ['random'])
def test_only_graph_scalable_gps(mgk_file, modelsets, split_type):
    model_type, n_estimators, n_samples_per_model, ensemble_rule = modelsets
    dataset = Dataset.from_df(df=df_regression,
                              smiles_columns=['smiles'],
                              targets_columns=['targets'])
    dataset.set_status(graph_kernel_type='graph', features_generators=None, features_combination=None)
    dataset.create_graphs(n_jobs=4)
    dataset.unify_datatype()
    kernel_config = get_kernel_config(dataset=dataset,
                                      graph_kernel_type='graph',
                                      mgk_hyperparameters_files=[mgk_file])
    model = set_model(model_type,
                      kernel=kernel_config.kernel,
                      alpha=0.01,
                      n_estimators=n_estimators,
                      n_samples_per_model=n_samples_per_model,
                      ensemble_rule=ensemble_rule)
    save_dir = tempfile.mkdtemp()
    Evaluator(save_dir=save_dir,
              dataset=dataset,
              model=model,
              task_type='regression',
              metrics=['rmse', 'mae', 'mse', 'r2', 'max', 'spearman', 'kendall', 'pearson'],
              split_type=split_type,
              split_sizes=[0.75, 0.25],
              num_folds=2,
              verbose=True,
              n_core=n_samples_per_model if model_type == 'gpr-nystrom' else None).run_cross_validation()
    shutil.rmtree(save_dir)


def test_loocv():
    model_type = 'gpr'
    mgk_file = additive_msnorm
    dataset = Dataset.from_df(df=df_regression,
                              smiles_columns=['smiles'],
                              targets_columns=['targets'])
    dataset.set_status(graph_kernel_type='graph', features_generators=None, features_combination=None)
    dataset.create_graphs(n_jobs=4)
    dataset.unify_datatype()
    kernel_config = get_kernel_config(dataset=dataset,
                                      graph_kernel_type='graph',
                                      mgk_hyperparameters_files=[mgk_file])
    model = set_model(model_type,
                      kernel=kernel_config.kernel,
                      alpha=0.01)
    save_dir = tempfile.mkdtemp()
    Evaluator(save_dir=save_dir,
              dataset=dataset,
              model=model,
              task_type='regression',
              metrics=['rmse', 'mae', 'mse', 'r2', 'max', 'spearman', 'kendall', 'pearson'],
              cross_validation='leave-one-out',
              num_folds=1).run_cross_validation()
    shutil.rmtree(save_dir)

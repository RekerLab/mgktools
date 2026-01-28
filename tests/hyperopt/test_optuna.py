#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import pandas as pd
from mgktools.data.data import Dataset
from mgktools.kernels.utils import get_kernel_config
from mgktools.features_mol.features_generators import FeaturesGenerator
from mgktools.hyperparameters import *
from mgktools.hyperparameters.optuna import bayesian_optimization
import os
import shutil


CWD = os.path.dirname(os.path.abspath(__file__))
smiles = ['CCCC', 'CCCCCO', 'c1ccccc1', 'CCNCCO', 'OCCCO']
targets = [3.1, 14.5, 25.6, 56.7, 12.3]
df = pd.DataFrame({'smiles': smiles, 'targets': targets})
if os.path.exists(f'{CWD}/tmp'):
    shutil.rmtree(f'{CWD}/tmp')
os.mkdir(f'{CWD}/tmp')


@pytest.mark.parametrize('mgk_file', [additive, additive_norm, additive_pnorm, additive_msnorm,
                                      product, product_norm, product_pnorm, product_msnorm])
@pytest.mark.parametrize('split_set', [
    ('leave-one-out', None, None, 1),
    ('Monte-Carlo', None, 'random', 10),
    ('kFold', 5, None, 1)
])
def test_bayesian_Graph(mgk_file, split_set):
    cross_validation, n_splits, split_type, num_folds = split_set
    if split_type is not None:
        split_sizes = [0.8, 0.2]
    else:
        split_sizes = None
    dataset = Dataset.from_df(df=df,
                              smiles_columns=['smiles'],
                              targets_columns=['targets'])
    dataset.set_status(graph_kernel_type='graph', features_generators=None, features_combination=None)
    dataset.create_graphs(n_jobs=4)
    dataset.unify_datatype()
    kernel_config = get_kernel_config(dataset=dataset,
                                      graph_kernel_type='graph',
                                      mgk_hyperparameters_files=[mgk_file])

    save_dir = f'{CWD}/tmp'
    bayesian_optimization(
        save_dir=save_dir,
        datasets=[dataset],
        dataset_val=None,
        dataset_test=None,
        kernel_config=kernel_config,
        model_type='gpr',
        task_type='regression',
        metric='rmse',
        cross_validation=cross_validation,
        split_type=split_type,
        split_sizes=split_sizes,
        num_folds=num_folds,
        n_splits=n_splits,
        num_iters=2,
        alpha=0.01,
        alpha_bounds=(0.001, 0.02),
        d_alpha=0.001)
    for file in ['optuna.db', 'graph_hyperparameters.json']:
        assert os.path.exists(f'{save_dir}/{file}')
        os.remove(f'{save_dir}/{file}')

    bayesian_optimization(
        save_dir=save_dir,
        datasets=[dataset, dataset],
        dataset_val=None,
        dataset_test=None,
        kernel_config=kernel_config,
        model_type='gpr',
        task_type='regression',
        metric='rmse',
        cross_validation=cross_validation,
        split_type=split_type,
        split_sizes=split_sizes,
        num_folds=num_folds,
        n_splits=n_splits,
        num_iters=2,
        alpha=0.01,
        alpha_bounds=(0.001, 0.02),
        d_alpha=0.001)
    for file in ['optuna.db', 'graph_hyperparameters.json']:
        assert os.path.exists(f'{save_dir}/{file}')
        os.remove(f'{save_dir}/{file}')

    bayesian_optimization(
        save_dir=save_dir,
        datasets=[dataset],
        dataset_val=None,
        dataset_test=None,
        kernel_config=kernel_config,
        model_type='gpr',
        task_type='regression',
        metric='log_likelihood',
        cross_validation=cross_validation,
        split_type=split_type,
        split_sizes=split_sizes,
        num_folds=num_folds,
        n_splits=n_splits,
        num_iters=2,
        alpha=0.01,
        alpha_bounds=(0.001, 0.02),
        d_alpha=0.001)
    for file in ['optuna.db', 'graph_hyperparameters.json', 'alpha']:
        assert os.path.exists(f'{save_dir}/{file}')
        os.remove(f'{save_dir}/{file}')


@pytest.mark.parametrize('features_kernel_type', ['dot_product', 'rbf'])
@pytest.mark.parametrize('features_generators', [['morgan'], ['rdkit_2d_normalized']])
@pytest.mark.parametrize('split_type', ['random'])
def test_bayesian_Fingperprint(features_kernel_type, features_generators, split_type):
    dataset = Dataset.from_df(df=df,
                              smiles_columns=['smiles'],
                              targets_columns=['targets'])
    dataset.set_status(graph_kernel_type='no', 
                       features_generators=[FeaturesGenerator(fg) for fg in features_generators], 
                       features_combination='concat')
    dataset.create_graphs(n_jobs=4)
    dataset.unify_datatype()
    kernel_config = get_kernel_config(dataset=dataset,
                                      graph_kernel_type='no',
                                      features_kernel_type=features_kernel_type,
                                      features_hyperparameters=[1.0],
                                      features_hyperparameters_bounds=(0.1, 10.0))

    save_dir = f'{CWD}/tmp'
    bayesian_optimization(
        save_dir=save_dir,
        datasets=[dataset],
        dataset_val=None,
        dataset_test=None,
        kernel_config=kernel_config,
        model_type='gpr',
        task_type='regression',
        metric='rmse',
        cross_validation='Monte-Carlo',
        split_type=split_type,
        split_sizes=[0.8, 0.2],
        num_iters=2,
        alpha=0.01,
        alpha_bounds=(0.001, 0.02),
        d_alpha=0.001)
    for file in ['optuna.db', 'features_hyperparameters.json']:
        assert os.path.exists(f'{save_dir}/{file}')
        os.remove(f'{save_dir}/{file}')

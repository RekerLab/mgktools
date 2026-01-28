#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import numpy as np
import pandas as pd
from mgktools.data.data import Dataset
from mgktools.features_mol.features_generators import FeaturesGenerator


smiles_1 = ['CCCC', 'CCCCCO', 'c1ccccc1', 'CCNCCO']
smiles_2 = ['CCCCC', 'CCCCCCO', 'Cc1ccccc1', 'CCCNCCO']
targets_1 = [3.4, 4.5, 5.6, 6.7]
targets_2 = [3.1, 14.5, 25.6, 56.7]
features_add = [0.6, 0.1, 0.3, 0.9]
df = pd.DataFrame({'smiles_1': smiles_1, 'smiles_2': smiles_2, 'targets_1': targets_1, 'targets_2': targets_2, 'features_add': features_add})


@pytest.mark.parametrize('smiles_columns', [['smiles_1'], ['smiles_1', 'smiles_2']])
@pytest.mark.parametrize('targets_columns', [None, ['targets_1'], ['targets_1', 'targets_2']])
@pytest.mark.parametrize('features_columns', [None, ['features_add']])
@pytest.mark.parametrize('fingerprints', [
    (['morgan'], 'concat', 2048),
    (['morgan_count'], 'concat', 2048),
    (['rdkit_2d'], 'concat', 200),
    (['rdkit_2d_normalized'], 'concat', 200),
    (['rdkit_208'], 'concat', 217),
    (['rdkit_topol'], 'concat', 2048),
    (['layered'], 'concat', 2048),
    (['torsion'], 'concat', 2048),
    (['atom_pair'], 'concat', 2048),
    (['avalon'], 'concat', 2048),
    (['avalon_count'], 'concat', 2048),
    (['maccskey'], 'concat', 167),
    (['pattern'], 'concat', 2048),
    (['morgan', 'rdkit_2d_normalized'], 'concat', 2248),
    (['morgan', 'rdkit_2d_normalized'], 'mean', 2248)
])
@pytest.mark.parametrize('normalize_features_mol', [True, False])
@pytest.mark.parametrize('normalize_features_add', [True, False])
def test_data(smiles_columns, targets_columns, features_columns, fingerprints, normalize_features_mol, normalize_features_add):
    features_generators, features_combination, n_features = fingerprints
    NS = len(smiles_columns)
    NF = len(features_columns) if features_columns is not None else 0
    NT = len(targets_columns) if targets_columns is not None else 0
    fgs = [FeaturesGenerator(features_generator_name=fg) for fg in features_generators]
    dataset = Dataset.from_df(df=df,
                              smiles_columns=smiles_columns,
                              features_columns=features_columns,
                              targets_columns=targets_columns)
    dataset.set_status(graph_kernel_type='graph',
                       features_generators=fgs, 
                       features_combination=features_combination)
    dataset.create_graphs(n_jobs=4)
    dataset.create_features_mol(n_jobs=4)
    # graph only
    dataset.set_status(graph_kernel_type='graph',
                       features_generators=None, 
                       features_combination=None)
    assert dataset.X.shape[0] == 4
    assert dataset.y.shape[0] == 4
    assert dataset.X.shape[1] == NS + NF
    assert dataset.y.shape[1] == NT
    # features only
    dataset.set_status(graph_kernel_type='no',
                       features_generators=fgs, 
                       features_combination=features_combination)
    assert dataset.X.shape[0] == 4
    assert dataset.y.shape[0] == 4
    if features_combination == 'concat':
        assert dataset.X.shape[1] == n_features * NS + NF
    else:
        assert dataset.X.shape[1] == n_features + NF
    assert dataset.y.shape[1] == NT
    # graph+features
    dataset.set_status(graph_kernel_type='graph',
                       features_generators=fgs,
                       features_combination=features_combination)
    assert dataset.X.shape[0] == 4
    assert dataset.y.shape[0] == 4
    if features_combination == 'concat':
        assert dataset.X.shape[1] == (1 + n_features) * NS + NF
    else:
        assert dataset.X.shape[1] == n_features + NS + NF
    assert dataset.y.shape[1] == NT
    if normalize_features_mol:
        assert dataset.features_mol_scaler is None
        dataset.normalize_features_mol()
        assert dataset.features_mol_scaler is not None
        X = dataset.features_mol_scaler.transform(dataset.X_features_mol_raw)
        assert dataset.X_features_mol == pytest.approx(X, 1e-5)
        assert X.mean(axis=0) == pytest.approx(np.zeros(X.shape[1]), 1e-5)
        for i, std in enumerate(X.std(axis=0)):
            assert std == pytest.approx(1.0, 1e-5) or std == pytest.approx(0.0, 1e-5)
            if std == pytest.approx(0.0, 1e-5):
                assert X[:, i] == pytest.approx(np.zeros(len(X)), 1e-5) or X[:, i] == pytest.approx(np.ones(len(X)), 1e-5)
                
    if features_columns is not None and normalize_features_add:
        assert dataset.features_add_scaler is None
        dataset.normalize_features_add()
        X = dataset.features_add_scaler.transform(dataset.X_features_add_raw)
        assert dataset.X_features_add == pytest.approx(X, 1e-5)
        assert X.mean(axis=0) == pytest.approx(np.zeros(X.shape[1]), 1e-5)
        for i, std in enumerate(X.std(axis=0)):
            assert std == pytest.approx(1.0, 1e-5) or std == pytest.approx(0.0, 1e-5)
            if std == pytest.approx(0.0, 1e-5):
                assert X[:, i] == pytest.approx(np.zeros(len(X)), 1e-5) or X[:, i] == pytest.approx(np.ones(len(X)), 1e-5)

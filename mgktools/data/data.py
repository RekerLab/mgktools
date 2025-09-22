#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Dict, List, Union, Literal
import copy
import os
import pickle
import numpy as np
import pandas as pd
from rdkit import Chem
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from mgktools.features_mol.features_generators import FeaturesGenerator
from mgktools.graph.hashgraph import HashGraph


class CachedDict:
    def __init__(self):
        self.SMILES_TO_GRAPH: Dict[str, HashGraph] = {}
        self.SMILES_TO_FEATURES: Dict[str, np.ndarray] = {}

    @staticmethod
    def smiles2graph_(smiles) -> HashGraph:
        mol = Chem.MolFromSmiles(smiles)
        return HashGraph.from_rdkit(mol, hash=smiles)

    def smiles2graph(self, smiles: str) -> HashGraph:
        if smiles in self.SMILES_TO_GRAPH:
            return self.SMILES_TO_GRAPH[smiles]
        else:
            graph = self.smiles2graph_(smiles)
            self.SMILES_TO_GRAPH[smiles] = graph
            return graph

    def cache_graphs(self, smiles_list: List[str], n_jobs: int = 8):
        non_cached_smiles = [smiles for smiles in smiles_list if smiles not in self.SMILES_TO_GRAPH]
        graphs = Parallel(n_jobs=n_jobs, verbose=True, prefer='processes')(
            delayed(self.smiles2graph_)(non_cached_smiles[i]) for i in range(len(non_cached_smiles))
        )
        self.SMILES_TO_GRAPH.update(dict(zip(non_cached_smiles, graphs)))

    @staticmethod
    def smiles2features_(smiles: str, features_generator: FeaturesGenerator) -> List[float]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None and mol.GetNumHeavyAtoms() == 0:
            features = np.zeros(len(features_generator(Chem.MolFromSmiles('C')))).tolist()
        else:
            features = features_generator(mol)
            replace_token = 0
            features = np.where((np.isnan(features)) | (features > 1e10), replace_token, features).tolist()
        return features

    def smiles2features(self, smiles: str, features_generator: FeaturesGenerator) -> List[float]: # 1D array.
        tag = f'{smiles}_{features_generator.features_generator_name}'
        if tag in self.SMILES_TO_FEATURES:
            return self.SMILES_TO_FEATURES[tag]
        else:
            features = self.smiles2features_(smiles, features_generator)
            self.SMILES_TO_FEATURES[smiles] = features
            return features

    def cache_features(self, smiles_list: List[str], features_generators: List[FeaturesGenerator], n_jobs: int = 8):
        for fg in features_generators:
            non_cached_smiles = [smiles for smiles in smiles_list if f'{smiles}_{fg.features_generator_name}' not in self.SMILES_TO_FEATURES]
            features = Parallel(n_jobs=n_jobs, verbose=True, prefer='processes')(
                delayed(self.smiles2features_)(non_cached_smiles[i], fg) for i in range(len(non_cached_smiles))
            )
            self.SMILES_TO_FEATURES.update({f'{smiles}_{fg.features_generator_name}': features for smiles, features in zip(non_cached_smiles, features)})

    def save(self, path=".", filename='cache.pkl', overwrite=False):
        f_cache = os.path.join(path, filename)
        if os.path.isfile(f_cache) and not overwrite:
            raise RuntimeError(
                f'Path {f_cache} already exists. To overwrite, set '
                '`overwrite=True`.'
            )
        store = self.__dict__.copy()
        pickle.dump(store, open(f_cache, 'wb'), protocol=4)

    @classmethod
    def load(cls, path=".", filename='cache.pkl'):
        f_cache = os.path.join(path, filename)
        store = pickle.load(open(f_cache, 'rb'))
        dataset = cls()
        dataset.__dict__.update(**store)
        return dataset


class Datapoint:
    def __init__(self, smiles_list: List[str],
                 features_add: List[float] = None,
                 targets: List[float] = None,
                 cache: CachedDict = None):
        self.smiles_list = smiles_list
        self.features_add = features_add or [] # set None to []
        self.targets = targets or [] # set None to []
        self.cache = cache or CachedDict() # initialize cache if it is None

    def __repr__(self) -> str:
        if self.features_add:
            return ','.join(self.smiles_list) + ';' + ','.join([str(f) for f in self.features_add])
        else:
            return ','.join(self.smiles_list)

    @property
    def mols(self) -> List[Chem.Mol]:
        return [Chem.MolFromSmiles(smiles) for smiles in self.smiles_list]

    @property
    def graph(self) -> List[HashGraph]:
        return [self.cache.smiles2graph(smiles) for smiles in self.smiles_list]

    def features_mol(self, features_generators: List[FeaturesGenerator] = None, 
                     features_combination: Literal['concat', 'mean'] = None) -> List[float]:
        if features_generators is None:
            return []
        features = []
        for smiles in self.smiles_list:
            f = np.concatenate([self.cache.smiles2features(smiles, fg) for fg in features_generators]).tolist()
            features.append(f)
        if features_combination == 'concat':
            return np.ravel(features).tolist()
        elif features_combination == 'mean':
            return np.mean(features, axis=0).tolist()
        else:
            raise ValueError(f'Invalid features_combination: {features_combination}')


class Dataset:
    def __init__(self, data: List[Datapoint] = None,
                 features_mol_scaler: StandardScaler = None,
                 features_add_scaler: StandardScaler = None,
                 cache: CachedDict = None):
        self.data = data
        self.features_mol_scaler = features_mol_scaler
        self.features_add_scaler = features_add_scaler
        self.set_cache(cache or CachedDict()) # initialize cache if it is None

    def __len__(self) -> int:
        """Return the number of data points in the dataset."""
        return len(self.data)

    def __getitem__(self, item) -> Union[Datapoint, List[Datapoint]]:
        return self.data[item]

    def set_status(self, graph_kernel_type: Literal['graph', 'pre-computed', 'no'],
                   features_generators: List[FeaturesGenerator] = None,
                   features_combination: Literal['concat', 'mean'] = None):
        self.graph_kernel_type = graph_kernel_type
        self.features_generators = features_generators
        self.features_combination = features_combination
    
    @property
    def X(self) -> np.ndarray: # 2D array.
        if self.graph_kernel_type == 'no':
            return np.concatenate([self.X_features_mol, self.X_features_add], axis=1)
        elif self.graph_kernel_type == 'graph':
            return np.concatenate([self.X_graph, self.X_features_mol, self.X_features_add], axis=1)
        elif self.graph_kernel_type == 'pre-computed':
            return np.concatenate([self.X_smiles, self.X_features_add], axis=1, dtype=object)
        else:
            raise ValueError(f'Invalid graph_kernel_type: {self.graph_kernel_type}')

    @property
    def y(self) -> np.ndarray[float]: # 2D float array.
        return np.array([d.targets for d in self.data])

    @property
    def repr(self) -> List[str]:
        return [d.__repr__() for d in self.data]

    @property
    def mols(self) -> np.ndarray[Chem.Mol]: # 2D array.
        return np.array([d.mols for d in self.data])

    @property
    def X_graph(self) -> np.ndarray[HashGraph]: # 2D graph array.
        return np.array([d.graph for d in self.data])
        
    @property
    def X_smiles(self) -> np.ndarray[str]:  # 2D str array.
        return np.array([d.smiles_list for d in self.data])
    
    @property
    def X_features_mol_raw(self) -> np.ndarray[float]: # 2D float array.
        return np.array([d.features_mol(self.features_generators, self.features_combination) for d in self.data])

    @property
    def X_features_mol(self) -> np.ndarray[float]: # 2D float array.
        if self.features_mol_scaler is not None:
            return self.features_mol_scaler.transform(self.X_features_mol_raw)
        else:
            return self.X_features_mol_raw

    @property
    def X_features_add_raw(self) -> np.ndarray[float]: # 2D float array.
        return np.array([d.features_add for d in self.data])

    @property
    def X_features_add(self) -> np.ndarray[float]: # 2D float array.
        if self.features_add_scaler is not None:
            return self.features_add_scaler.transform(self.X_features_add_raw)
        else:
            return self.X_features_add_raw

    @property
    def X_mol(self) -> np.ndarray: # 2D graph+float array.
        return np.concatenate([self.X_graph, self.X_features_mol], axis=1)

    @property
    def X_features(self) -> np.ndarray[float]: # 2D float array.
        return np.concatenate([self.X_features_mol, self.X_features_add], axis=1)

    @property
    def N_MGK(self) -> int:
        if self.graph_kernel_type == 'graph':
            return self.X_smiles.shape[1]
        else:
            return 0

    @property
    def N_tasks(self) -> int:
        return self.y.shape[1]

    @property
    def N_features_mol(self):
        return self.X_features_mol.shape[1]

    @property
    def N_features_add(self):
        return self.X_features_add.shape[1]

    def features_size(self):
        return self.N_features_mol + self.N_features_add

    def copy(self):
        return copy.deepcopy(self)

    def normalize_features_mol(self):
        if self.X_features_mol_raw is not None:
            self.features_mol_scaler = StandardScaler().fit(self.X_features_mol_raw)
        else:
            self.features_mol_scaler = None

    def normalize_features_add(self):
        if self.X_features_add_raw is not None:
            self.features_add_scaler = StandardScaler().fit(self.X_features_add_raw)
        else:
            self.features_add_scaler = None

    def unify_datatype(self, X: np.ndarray[HashGraph] = None):
        if X is None:
            X = self.X_graph
        else:
            X = np.concatenate([X, self.X_graph], axis=0)
        for i in range(X.shape[1]):
            HashGraph.unify_datatype(X[:, i], inplace=True)

    def clear_cookie(self):
        """Clear the cookie of all graphs in the dataset. This is crucial to avoid memory leaks during hyperparameter optimization."""
        for x in self.X_graph:
            for g in x:
                g.cookie.clear()

    def create_graphs(self, n_jobs: int = 8):
        unique_smiles_list = np.unique(self.X_smiles)
        self.cache.cache_graphs(unique_smiles_list, n_jobs=n_jobs)
    
    def create_features_mol(self, n_jobs: int = 8):
        unique_smiles_list = np.unique(self.X_smiles)
        self.cache.cache_features(unique_smiles_list, self.features_generators, n_jobs=n_jobs)

    def set_cache(self, cache: CachedDict):
        self.cache = cache
        for d in self.data:
            d.cache = cache

    def save(self, path, filename='dataset.pkl', overwrite=False):
        f_dataset = os.path.join(path, filename)
        if os.path.isfile(f_dataset) and not overwrite:
            raise RuntimeError(
                f'Path {f_dataset} already exists. To overwrite, set '
                '`overwrite=True`.'
            )
        store = self.__dict__.copy()
        pickle.dump(store, open(f_dataset, 'wb'), protocol=4)

    @classmethod
    def load(cls, path, filename='dataset.pkl'):
        f_dataset = os.path.join(path, filename)
        store = pickle.load(open(f_dataset, 'rb'))
        dataset = cls()
        dataset.__dict__.update(**store)
        return dataset

    @classmethod
    def from_df(cls, df: pd.DataFrame,
                smiles_columns: List[str],
                features_columns: List[str] = None,
                targets_columns: List[str] = None,
                n_jobs: int = 8,
                cache: CachedDict = None):
        if cache is None:
            cache = CachedDict()
        I1 = df.get(smiles_columns).to_numpy().tolist()
        I2 = df.get(features_columns).to_numpy().tolist() if features_columns else [None] * len(df)
        I3 = df.get(targets_columns)
        I3 = [None] * len(df) if I3 is None else I3.to_numpy().tolist()
        data = Parallel(
            n_jobs=n_jobs, verbose=True, prefer='processes')(
            delayed(Datapoint)(I1[i], I2[i], I3[i]) for i in range(len(df)))
        return cls(data=data, cache=cache)

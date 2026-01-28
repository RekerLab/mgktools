#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data handling module for molecular datasets.

This module provides classes for managing molecular datasets, including caching
of molecular graphs and features, data point representation, and dataset operations.

Classes
-------
CachedDict
    Dictionary-based cache for SMILES-to-graph and SMILES-to-features conversions.
Datapoint
    Container for a single molecular data point with SMILES, features, and targets.
Dataset
    Main dataset container for molecular property prediction tasks.
"""
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
    """
    Dictionary-based cache for SMILES-to-graph and SMILES-to-features conversions.

    This class provides efficient caching of molecular graph representations and
    computed features to avoid redundant calculations during model training and
    evaluation.

    Attributes
    ----------
    SMILES_TO_GRAPH : Dict[str, HashGraph]
        Cache mapping SMILES strings to HashGraph objects.
    SMILES_TO_FEATURES : Dict[str, np.ndarray]
        Cache mapping SMILES strings (with feature generator name) to feature arrays.

    Examples
    --------
    >>> cache = CachedDict()
    >>> cache.cache_graphs(['CCO', 'CCC'], n_jobs=2)
    >>> graph = cache.smiles2graph('CCO')
    >>> cache.save(path='.', filename='cache.pkl')
    """

    def __init__(self):
        """Initialize an empty cache for graphs and features."""
        self.SMILES_TO_GRAPH: Dict[str, HashGraph] = {}
        self.SMILES_TO_FEATURES: Dict[str, np.ndarray] = {}

    @staticmethod
    def smiles2graph_(smiles: str) -> HashGraph:
        """
        Convert a SMILES string to a HashGraph object.

        Parameters
        ----------
        smiles : str
            SMILES string representation of a molecule.

        Returns
        -------
        HashGraph
            Graph representation of the molecule.
        """
        mol = Chem.MolFromSmiles(smiles)
        return HashGraph.from_rdkit(mol, hash=smiles)

    def smiles2graph(self, smiles: str) -> HashGraph:
        """
        Get or create a HashGraph for a SMILES string with caching.

        Parameters
        ----------
        smiles : str
            SMILES string representation of a molecule.

        Returns
        -------
        HashGraph
            Cached or newly created graph representation of the molecule.
        """
        if smiles in self.SMILES_TO_GRAPH:
            return self.SMILES_TO_GRAPH[smiles]
        else:
            graph = self.smiles2graph_(smiles)
            self.SMILES_TO_GRAPH[smiles] = graph
            return graph

    def cache_graphs(self, smiles_list: List[str], n_jobs: int = 8):
        """
        Pre-cache graphs for a list of SMILES strings in parallel.

        Parameters
        ----------
        smiles_list : List[str]
            List of SMILES strings to convert to graphs.
        n_jobs : int, optional
            Number of parallel jobs for graph creation, by default 8.
        """
        non_cached_smiles = [smiles for smiles in smiles_list if smiles not in self.SMILES_TO_GRAPH]
        graphs = Parallel(n_jobs=n_jobs, verbose=True, prefer='processes')(
            delayed(self.smiles2graph_)(non_cached_smiles[i]) for i in range(len(non_cached_smiles))
        )
        self.SMILES_TO_GRAPH.update(dict(zip(non_cached_smiles, graphs)))

    @staticmethod
    def smiles2features_(smiles: str, features_generator: FeaturesGenerator) -> List[float]:
        """
        Compute molecular features for a SMILES string.

        Parameters
        ----------
        smiles : str
            SMILES string representation of a molecule.
        features_generator : FeaturesGenerator
            Feature generator instance to compute molecular descriptors.

        Returns
        -------
        List[float]
            List of computed molecular features. NaN and extremely large values
            (> 1e10) are replaced with 0.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None and mol.GetNumHeavyAtoms() == 0:
            features = np.zeros(len(features_generator(Chem.MolFromSmiles('C')))).tolist()
        else:
            features = features_generator(mol)
            replace_token = 0
            features = np.where((np.isnan(features)) | (features > 1e10), replace_token, features).tolist()
        return features

    def smiles2features(self, smiles: str, features_generator: FeaturesGenerator) -> List[float]:
        """
        Get or compute molecular features for a SMILES string with caching.

        Parameters
        ----------
        smiles : str
            SMILES string representation of a molecule.
        features_generator : FeaturesGenerator
            Feature generator instance to compute molecular descriptors.

        Returns
        -------
        List[float]
            Cached or newly computed molecular features.
        """
        tag = f'{smiles}_{features_generator.features_generator_name}'
        if tag in self.SMILES_TO_FEATURES:
            return self.SMILES_TO_FEATURES[tag]
        else:
            features = self.smiles2features_(smiles, features_generator)
            self.SMILES_TO_FEATURES[smiles] = features
            return features

    def cache_features(self, smiles_list: List[str], features_generators: List[FeaturesGenerator], n_jobs: int = 8):
        """
        Pre-cache molecular features for a list of SMILES strings in parallel.

        Parameters
        ----------
        smiles_list : List[str]
            List of SMILES strings to compute features for.
        features_generators : List[FeaturesGenerator]
            List of feature generators to apply to each molecule.
        n_jobs : int, optional
            Number of parallel jobs for feature computation, by default 8.

        Notes
        -----
        Some feature generators (e.g., rdkit_2d) may not be picklable and
        require n_jobs=1 to avoid serialization errors.
        """
        for fg in features_generators:
            non_cached_smiles = [smiles for smiles in smiles_list if f'{smiles}_{fg.features_generator_name}' not in self.SMILES_TO_FEATURES]
            features = Parallel(n_jobs=n_jobs, verbose=True, prefer='processes')(
                delayed(self.smiles2features_)(non_cached_smiles[i], fg) for i in range(len(non_cached_smiles))
            )
            self.SMILES_TO_FEATURES.update({f'{smiles}_{fg.features_generator_name}': features for smiles, features in zip(non_cached_smiles, features)})

    def save(self, path: str = ".", filename: str = 'cache.pkl', overwrite: bool = False):
        """
        Save the cache to a pickle file.

        Parameters
        ----------
        path : str, optional
            Directory path to save the cache file, by default ".".
        filename : str, optional
            Name of the cache file, by default 'cache.pkl'.
        overwrite : bool, optional
            Whether to overwrite existing file, by default False.

        Raises
        ------
        RuntimeError
            If the file exists and overwrite is False.
        """
        f_cache = os.path.join(path, filename)
        if os.path.isfile(f_cache) and not overwrite:
            raise RuntimeError(
                f'Path {f_cache} already exists. To overwrite, set '
                '`overwrite=True`.'
            )
        store = self.__dict__.copy()
        pickle.dump(store, open(f_cache, 'wb'), protocol=4)

    @classmethod
    def load(cls, path: str = ".", filename: str = 'cache.pkl') -> 'CachedDict':
        """
        Load a cache from a pickle file.

        Parameters
        ----------
        path : str, optional
            Directory path containing the cache file, by default ".".
        filename : str, optional
            Name of the cache file, by default 'cache.pkl'.

        Returns
        -------
        CachedDict
            Loaded cache instance.
        """
        f_cache = os.path.join(path, filename)
        store = pickle.load(open(f_cache, 'rb'))
        dataset = cls()
        dataset.__dict__.update(**store)
        return dataset


class Datapoint:
    """
    Container for a single molecular data point.

    A data point consists of one or more SMILES strings (for multi-molecule
    inputs like reactions), optional additional features, and optional target values.

    Parameters
    ----------
    smiles_list : List[str]
        List of SMILES strings representing molecules.
    features_add : List[float], optional
        Additional numerical features, by default None.
    targets : List[float], optional
        Target values for prediction, by default None.
    cache : CachedDict, optional
        Cache instance for graph and feature storage, by default None.

    Attributes
    ----------
    smiles_list : List[str]
        SMILES strings for this data point.
    features_add : List[float]
        Additional features (empty list if None provided).
    targets : List[float]
        Target values (empty list if None provided).
    cache : CachedDict
        Cache instance for efficient graph/feature retrieval.

    Examples
    --------
    >>> dp = Datapoint(['CCO'], targets=[1.5])
    >>> dp.mols  # Returns list of RDKit Mol objects
    >>> dp.graph  # Returns list of HashGraph objects
    """

    def __init__(self, smiles_list: List[str],
                 features_add: List[float] = None,
                 targets: List[float] = None,
                 cache: CachedDict = None):
        """Initialize a data point with molecular and target information."""
        self.smiles_list = smiles_list
        self.features_add = features_add or []
        self.targets = targets or []
        self.cache = cache or CachedDict()

    def __repr__(self) -> str:
        """
        Return string representation of the data point.

        Returns
        -------
        str
            SMILES strings joined by comma, optionally with features.
        """
        if self.features_add:
            return ','.join(self.smiles_list) + ';' + ','.join([str(f) for f in self.features_add])
        else:
            return ','.join(self.smiles_list)

    @property
    def mols(self) -> List[Chem.Mol]:
        """
        Get RDKit Mol objects for all SMILES in this data point.

        Returns
        -------
        List[Chem.Mol]
            List of RDKit Mol objects.
        """
        return [Chem.MolFromSmiles(smiles) for smiles in self.smiles_list]

    @property
    def graph(self) -> List[HashGraph]:
        """
        Get HashGraph objects for all SMILES in this data point.

        Returns
        -------
        List[HashGraph]
            List of HashGraph objects from cache.
        """
        return [self.cache.smiles2graph(smiles) for smiles in self.smiles_list]

    def features_mol(self, features_generators: List[FeaturesGenerator] = None,
                     features_combination: Literal['concat', 'mean'] = None) -> List[float]:
        """
        Compute molecular features for this data point.

        Parameters
        ----------
        features_generators : List[FeaturesGenerator], optional
            List of feature generators to apply, by default None.
        features_combination : {'concat', 'mean'}, optional
            How to combine features from multiple molecules, by default None.

        Returns
        -------
        List[float]
            Combined molecular features.

        Raises
        ------
        ValueError
            If features_combination is not 'concat' or 'mean'.
        """
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
    """
    Main dataset container for molecular property prediction.

    This class manages collections of molecular data points and provides
    methods for data loading, preprocessing, feature normalization, and
    graph creation.

    Parameters
    ----------
    data : List[Datapoint], optional
        List of data points, by default None.
    features_mol_scaler : StandardScaler, optional
        Scaler for molecular features normalization, by default None.
    features_add_scaler : StandardScaler, optional
        Scaler for additional features normalization, by default None.
    cache : CachedDict, optional
        Cache for graphs and features, by default None.

    Attributes
    ----------
    data : List[Datapoint]
        Collection of data points.
    features_mol_scaler : StandardScaler
        Fitted scaler for molecular features.
    features_add_scaler : StandardScaler
        Fitted scaler for additional features.
    graph_kernel_type : str
        Type of graph kernel ('graph', 'pre-computed', or 'no').
    features_generators : List[FeaturesGenerator]
        List of feature generators to use.
    features_combination : str
        Method for combining features ('concat' or 'mean').

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'smiles': ['CCO', 'CCC'], 'target': [1.0, 2.0]})
    >>> dataset = Dataset.from_df(df, smiles_columns=['smiles'], targets_columns=['target'])
    >>> dataset.set_status(graph_kernel_type='graph')
    >>> dataset.create_graphs(n_jobs=4)
    >>> X, y = dataset.X, dataset.y
    """

    def __init__(self, data: List[Datapoint] = None,
                 features_mol_scaler: StandardScaler = None,
                 features_add_scaler: StandardScaler = None,
                 cache: CachedDict = None):
        """Initialize a Dataset with optional data and scalers."""
        self.data = data
        self.features_mol_scaler = features_mol_scaler
        self.features_add_scaler = features_add_scaler
        self.set_cache(cache or CachedDict())

    def __len__(self) -> int:
        """
        Return the number of data points in the dataset.

        Returns
        -------
        int
            Number of data points.
        """
        return len(self.data)

    def __getitem__(self, item) -> Union[Datapoint, List[Datapoint]]:
        """
        Get data point(s) by index.

        Parameters
        ----------
        item : int or slice
            Index or slice of data points to retrieve.

        Returns
        -------
        Datapoint or List[Datapoint]
            Requested data point(s).
        """
        return self.data[item]

    def set_status(self, graph_kernel_type: Literal['graph', 'pre-computed', 'no'],
                   features_generators: List[FeaturesGenerator] = None,
                   features_combination: Literal['concat', 'mean'] = None):
        """
        Set the dataset status for kernel computation.

        Parameters
        ----------
        graph_kernel_type : {'graph', 'pre-computed', 'no'}
            Type of graph kernel to use:
            - 'graph': Compute graph kernel on-the-fly
            - 'pre-computed': Use pre-computed kernel matrix
            - 'no': No graph kernel, features only
        features_generators : List[FeaturesGenerator], optional
            List of molecular feature generators, by default None.
        features_combination : {'concat', 'mean'}, optional
            How to combine features from multiple molecules, by default None.
        """
        self.graph_kernel_type = graph_kernel_type
        self.features_generators = features_generators
        self.features_combination = features_combination

    @property
    def X(self) -> np.ndarray:
        """
        Get the feature matrix for the dataset.

        Returns
        -------
        np.ndarray
            2D array of features. Structure depends on graph_kernel_type:
            - 'no': [features_mol, features_add]
            - 'graph': [graphs, features_mol, features_add]
            - 'pre-computed': [smiles, features_add]

        Raises
        ------
        ValueError
            If graph_kernel_type is invalid.
        """
        if self.graph_kernel_type == 'no':
            return np.concatenate([self.X_features_mol, self.X_features_add], axis=1)
        elif self.graph_kernel_type == 'graph':
            return np.concatenate([self.X_graph, self.X_features_mol, self.X_features_add], axis=1)
        elif self.graph_kernel_type == 'pre-computed':
            return np.concatenate([self.X_smiles, self.X_features_add], axis=1, dtype=object)
        else:
            raise ValueError(f'Invalid graph_kernel_type: {self.graph_kernel_type}')

    @property
    def y(self) -> np.ndarray:
        """
        Get the target values for the dataset.

        Returns
        -------
        np.ndarray
            2D array of target values with shape (n_samples, n_tasks).
        """
        return np.array([d.targets for d in self.data])

    @property
    def repr(self) -> List[str]:
        """
        Get string representations of all data points.

        Returns
        -------
        List[str]
            List of string representations.
        """
        return [d.__repr__() for d in self.data]

    @property
    def mols(self) -> np.ndarray:
        """
        Get RDKit Mol objects for all data points.

        Returns
        -------
        np.ndarray
            2D array of RDKit Mol objects with shape (n_samples, n_molecules).
        """
        return np.array([d.mols for d in self.data])

    @property
    def X_graph(self) -> np.ndarray:
        """
        Get graph representations for all data points.

        Returns
        -------
        np.ndarray
            2D array of HashGraph objects with shape (n_samples, n_molecules).
        """
        return np.array([d.graph for d in self.data])

    @property
    def X_smiles(self) -> np.ndarray:
        """
        Get SMILES strings for all data points.

        Returns
        -------
        np.ndarray
            2D array of SMILES strings with shape (n_samples, n_molecules).
        """
        return np.array([d.smiles_list for d in self.data])

    @property
    def X_features_mol_raw(self) -> np.ndarray:
        """
        Get raw (unnormalized) molecular features.

        Returns
        -------
        np.ndarray
            2D array of raw molecular features.
        """
        return np.array([d.features_mol(self.features_generators, self.features_combination) for d in self.data])

    @property
    def X_features_mol(self) -> np.ndarray:
        """
        Get normalized molecular features.

        Returns
        -------
        np.ndarray
            2D array of normalized molecular features, or raw features
            if no scaler is set.
        """
        if self.features_mol_scaler is not None:
            return self.features_mol_scaler.transform(self.X_features_mol_raw)
        else:
            return self.X_features_mol_raw

    @property
    def X_features_add_raw(self) -> np.ndarray:
        """
        Get raw (unnormalized) additional features.

        Returns
        -------
        np.ndarray
            2D array of raw additional features.
        """
        return np.array([d.features_add for d in self.data])

    @property
    def X_features_add(self) -> np.ndarray:
        """
        Get normalized additional features.

        Returns
        -------
        np.ndarray
            2D array of normalized additional features, or raw features
            if no scaler is set.
        """
        if self.features_add_scaler is not None:
            return self.features_add_scaler.transform(self.X_features_add_raw)
        else:
            return self.X_features_add_raw

    @property
    def X_mol(self) -> np.ndarray:
        """
        Get combined graph and molecular features.

        Returns
        -------
        np.ndarray
            2D array combining graphs and molecular features.
        """
        return np.concatenate([self.X_graph, self.X_features_mol], axis=1)

    @property
    def X_features(self) -> np.ndarray:
        """
        Get combined molecular and additional features.

        Returns
        -------
        np.ndarray
            2D array combining molecular and additional features.
        """
        return np.concatenate([self.X_features_mol, self.X_features_add], axis=1)

    @property
    def N_MGK(self) -> int:
        """
        Get the number of graph kernel components.

        Returns
        -------
        int
            Number of molecules per data point if using graph kernel, else 0.
        """
        if self.graph_kernel_type == 'graph':
            return self.X_smiles.shape[1]
        else:
            return 0

    @property
    def N_tasks(self) -> int:
        """
        Get the number of prediction tasks.

        Returns
        -------
        int
            Number of target columns.
        """
        return self.y.shape[1]

    @property
    def N_features_mol(self) -> int:
        """
        Get the number of molecular features.

        Returns
        -------
        int
            Dimension of molecular features.
        """
        return self.X_features_mol.shape[1]

    @property
    def N_features_add(self) -> int:
        """
        Get the number of additional features.

        Returns
        -------
        int
            Dimension of additional features.
        """
        return self.X_features_add.shape[1]

    def features_size(self) -> int:
        """
        Get the total number of features.

        Returns
        -------
        int
            Sum of molecular and additional feature dimensions.
        """
        return self.N_features_mol + self.N_features_add

    def copy(self) -> 'Dataset':
        """
        Create a deep copy of the dataset.

        Returns
        -------
        Dataset
            Deep copy of the dataset.
        """
        return copy.deepcopy(self)

    def normalize_features_mol(self):
        """
        Fit a StandardScaler on molecular features.

        This method fits a scaler on the raw molecular features that will
        be used to normalize features when accessing X_features_mol.
        """
        if self.X_features_mol_raw is not None:
            self.features_mol_scaler = StandardScaler().fit(self.X_features_mol_raw)
        else:
            self.features_mol_scaler = None

    def normalize_features_add(self):
        """
        Fit a StandardScaler on additional features.

        This method fits a scaler on the raw additional features that will
        be used to normalize features when accessing X_features_add.
        """
        if self.X_features_add_raw is not None:
            self.features_add_scaler = StandardScaler().fit(self.X_features_add_raw)
        else:
            self.features_add_scaler = None

    def unify_datatype(self, X: np.ndarray = None):
        """
        Unify data types across all graphs in the dataset.

        This ensures consistent node and edge attribute types across all
        graphs, which is required for kernel computation.

        Parameters
        ----------
        X : np.ndarray, optional
            Additional graph array to include in unification, by default None.
        """
        if X is None:
            X = self.X_graph
        else:
            X = np.concatenate([X, self.X_graph], axis=0)
        for i in range(X.shape[1]):
            HashGraph.unify_datatype(X[:, i], inplace=True)

    def clear_cookie(self):
        """
        Clear the cookie cache of all graphs in the dataset.

        This is crucial to avoid memory leaks during hyperparameter
        optimization, as graph cookies store intermediate kernel computations.
        """
        for x in self.X_graph:
            for g in x:
                g.cookie.clear()

    def create_graphs(self, n_jobs: int = 8):
        """
        Pre-create and cache graphs for all unique SMILES in the dataset.

        Parameters
        ----------
        n_jobs : int, optional
            Number of parallel jobs for graph creation, by default 8.
        """
        unique_smiles_list = np.unique(self.X_smiles)
        self.cache.cache_graphs(unique_smiles_list, n_jobs=n_jobs)

    def create_features_mol(self, n_jobs: int = 8):
        """
        Pre-compute and cache molecular features for all unique SMILES.

        Parameters
        ----------
        n_jobs : int, optional
            Number of parallel jobs for feature computation, by default 8.

        Notes
        -----
        Some feature generators may not support parallel execution.
        Use n_jobs=1 for rdkit_2d descriptors.
        """
        unique_smiles_list = np.unique(self.X_smiles)
        self.cache.cache_features(unique_smiles_list, self.features_generators, n_jobs=n_jobs)

    def set_cache(self, cache: CachedDict):
        """
        Set the cache for the dataset and all its data points.

        Parameters
        ----------
        cache : CachedDict
            Cache instance to use for graph and feature storage.
        """
        self.cache = cache
        for d in self.data:
            d.cache = cache

    def save(self, path: str, filename: str = 'dataset.pkl', overwrite: bool = False):
        """
        Save the dataset to a pickle file.

        Parameters
        ----------
        path : str
            Directory path to save the dataset file.
        filename : str, optional
            Name of the dataset file, by default 'dataset.pkl'.
        overwrite : bool, optional
            Whether to overwrite existing file, by default False.

        Raises
        ------
        RuntimeError
            If the file exists and overwrite is False.
        """
        f_dataset = os.path.join(path, filename)
        if os.path.isfile(f_dataset) and not overwrite:
            raise RuntimeError(
                f'Path {f_dataset} already exists. To overwrite, set '
                '`overwrite=True`.'
            )
        store = self.__dict__.copy()
        pickle.dump(store, open(f_dataset, 'wb'), protocol=4)

    @classmethod
    def load(cls, path: str, filename: str = 'dataset.pkl') -> 'Dataset':
        """
        Load a dataset from a pickle file.

        Parameters
        ----------
        path : str
            Directory path containing the dataset file.
        filename : str, optional
            Name of the dataset file, by default 'dataset.pkl'.

        Returns
        -------
        Dataset
            Loaded dataset instance.
        """
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
                cache: CachedDict = None) -> 'Dataset':
        """
        Create a Dataset from a pandas DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing molecular data.
        smiles_columns : List[str]
            Column names containing SMILES strings.
        features_columns : List[str], optional
            Column names containing additional features, by default None.
        targets_columns : List[str], optional
            Column names containing target values, by default None.
        n_jobs : int, optional
            Number of parallel jobs for data point creation, by default 8.
        cache : CachedDict, optional
            Cache instance to use, by default None (creates new cache).

        Returns
        -------
        Dataset
            Dataset instance populated from the DataFrame.

        Examples
        --------
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     'smiles': ['CCO', 'CCC', 'CCCC'],
        ...     'feature1': [1.0, 2.0, 3.0],
        ...     'target': [0.5, 1.0, 1.5]
        ... })
        >>> dataset = Dataset.from_df(
        ...     df,
        ...     smiles_columns=['smiles'],
        ...     features_columns=['feature1'],
        ...     targets_columns=['target']
        ... )
        """
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

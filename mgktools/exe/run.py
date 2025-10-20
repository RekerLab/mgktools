#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import KernelPCA
from mgktools.data.data import CachedDict, Dataset
from mgktools.kernels.utils import get_kernel_config
from mgktools.kernels.PreComputed import calc_precomputed_kernel_config
from mgktools.evaluators.cross_validation import Evaluator
from mgktools.data.split import dataset_split
from mgktools.hyperparameters.optuna import (
    bayesian_optimization as optuna_bayesian_optimization,
    bayesian_optimization_gpr_multi_datasets as optuna_bayesian_optimization_gpr_multi_datasets,
)
from mgktools.exe.args import (
    CacheArgs,
    CommonArgs,
    KernelArgs,
    TrainArgs,
    GradientOptArgs,
    OptunaArgs,
    OptunaMultiDatasetArgs,
    EmbeddingArgs,
)
from mgktools.exe.model import set_model_from_args, set_model


def mgk_cache_data(arguments=None):
    args = CacheArgs().parse_args(arguments)
    smiles_list = []
    for data_path in args.data_paths:
        df = pd.read_csv(data_path)
        for smiles_column in args.smiles_columns:
            if smiles_column in df:
                smiles_list += df[smiles_column].tolist()
    smiles_list = np.unique(smiles_list).tolist()
    cache = CachedDict()
    if args.cache_graph:
        print("**\tCreating GraphDot graph objects.\t**")
        cache = CachedDict()
        cache.cache_graphs(smiles_list, n_jobs=args.n_jobs)
    if args.features_generators is not None:
        print("**\tCreating molecular descriptors.\t**")
        cache.cache_features(smiles_list, features_generators=args.features_generators, n_jobs=args.n_jobs)
    print(f"**\tSaving cache file: {args.cache_path}.\t**")
    cache.save(filename=args.cache_path, overwrite=True)


def mgk_kernel_calc(arguments=None):
    args = KernelArgs().parse_args(arguments)
    assert args.graph_kernel_type == "graph", "Graph kernel must be used for mgk_kernel_calc."
    # read data set.
    dataset = Dataset.from_df(
        df=pd.read_csv(args.data_path),
        smiles_columns=args.smiles_columns,
        features_columns=args.features_columns,
        targets_columns=args.targets_columns,
        n_jobs=args.n_jobs,
    )
    dataset.set_status(graph_kernel_type=args.graph_kernel_type,
                       features_generators=args.features_generators,
                       features_combination=args.features_combination)
    if args.cache_path is not None:
        print(f"**\tLoading cache file: {args.cache_path}.\t**")
        dataset.set_cache(CachedDict.load(filename=args.cache_path))
    dataset.unify_datatype()
    # set kernel_config
    kernel_config = get_kernel_config(
        dataset=dataset,
        graph_kernel_type=args.graph_kernel_type,
        mgk_hyperparameters_files=args.graph_hyperparameters,
        features_hyperparameters_file=args.features_hyperparameters,
    )
    print("**\tCalculating kernel matrix.\t**")
    kernel_config = calc_precomputed_kernel_config(kernel_config=kernel_config, dataset=dataset)
    print("**\tEnd Calculating kernel matrix.\t**")
    kernel_pkl = os.path.join(args.save_dir, "kernel.pkl")
    pickle.dump(kernel_config, open(kernel_pkl, "wb"), protocol=4)


def mgk_cross_validation(arguments=None):
    args = TrainArgs().parse_args(arguments)
    # read data set.
    dataset = Dataset.from_df(
        df=pd.read_csv(args.data_path),
        smiles_columns=args.smiles_columns,
        features_columns=args.features_columns,
        targets_columns=args.targets_columns,
        n_jobs=args.n_jobs,
    )
    dataset.set_status(graph_kernel_type=args.graph_kernel_type,
                       features_generators=args.features_generators,
                       features_combination=args.features_combination)
    if args.cache_path is not None:
        print(f"**\tLoading cache file: {args.cache_path}.\t**")
        cache = CachedDict.load(filename=args.cache_path)
        dataset.set_cache(cache)
    # set kernel_config
    kernel_config = get_kernel_config(
        dataset=dataset,
        graph_kernel_type=args.graph_kernel_type,
        mgk_hyperparameters_files=args.graph_hyperparameters,
        features_hyperparameters_file=args.features_hyperparameters,
        kernel_pkl=os.path.join(args.save_dir, "kernel.pkl"),
    )
    model = set_model_from_args(args, kernel=kernel_config.kernel)
    evaluator = Evaluator(
        save_dir=args.save_dir,
        dataset=dataset,
        model=model,
        task_type=args.task_type,
        metrics=args.metrics,
        cross_validation=args.cross_validation,
        n_splits=args.n_splits,
        split_type=args.split_type,
        split_sizes=args.split_sizes,
        num_folds=args.num_folds,
        evaluate_train=args.evaluate_train,
        n_similar=args.n_similar,
        kernel=kernel_config.kernel,
        n_core=args.n_core,
        atomic_attribution=args.atomic_attribution,
        molecular_attribution=args.molecular_attribution,
        seed=args.seed,
        verbose=True,
    )
    if args.separate_test_path is not None:
        df = pd.read_csv(args.separate_test_path)
        dataset_test = Dataset.from_df(
            df=df,
            smiles_columns=args.smiles_columns,
            features_columns=args.features_columns,
            targets_columns=args.targets_columns,
            n_jobs=args.n_jobs,
        )
        dataset_test.set_status(graph_kernel_type=args.graph_kernel_type,
                                features_generators=args.features_generators, 
                                features_combination=args.features_combination)
        if args.cache_path is not None:
            dataset_test.set_cache(cache)
        dataset.unify_datatype(dataset_test.X_graph)
        evaluator.run_external(dataset_test)
    else:
        dataset.unify_datatype()
        evaluator.run_cross_validation()


def mgk_gradientopt(arguments=None):
    args = GradientOptArgs().parse_args(arguments)
    # read data set.
    dataset = Dataset.from_df(
        df=pd.read_csv(args.data_path),
        smiles_columns=args.smiles_columns,
        features_columns=args.features_columns,
        targets_columns=args.targets_columns,
        n_jobs=args.n_jobs,
    )
    dataset.set_status(graph_kernel_type=args.graph_kernel_type,
                       features_generators=args.features_generators,
                       features_combination=args.features_combination)
    if args.cache_path is not None:
        print(f"**\tLoading cache file: {args.cache_path}.\t**")
        dataset.set_cache(CachedDict.load(filename=args.cache_path))
    dataset.unify_datatype()
    # set kernel_config
    kernel_config = get_kernel_config(
        dataset=dataset,
        graph_kernel_type=args.graph_kernel_type,
        mgk_hyperparameters_files=args.graph_hyperparameters,
        features_hyperparameters_file=args.features_hyperparameters,
    )
    model = set_model(
        model_type="gpr",
        kernel=kernel_config.kernel,
        optimizer=args.optimizer,
        alpha=args.alpha_,
    )
    model.fit(dataset.X, dataset.y, loss=args.loss, verbose=True)
    kernel_config.update_from_theta()
    kernel_config.update_kernel()
    kernel_config.save(args.save_dir)


def mgk_optuna(arguments=None):
    args = OptunaArgs().parse_args(arguments)
    # read data set.
    dataset = Dataset.from_df(
        df=pd.read_csv(args.data_path),
        smiles_columns=args.smiles_columns,
        features_columns=args.features_columns,
        targets_columns=args.targets_columns,
        n_jobs=args.n_jobs,
    )
    dataset.set_status(graph_kernel_type=args.graph_kernel_type,
                       features_generators=args.features_generators,
                       features_combination=args.features_combination)
    if args.cache_path is not None:
        print(f"**\tLoading cache file: {args.cache_path}.\t**")
        dataset.set_cache(CachedDict.load(filename=args.cache_path))
    dataset.unify_datatype()

    if args.separate_val_path is not None:
        df_val = pd.read_csv(args.separate_val_path)
        dataset_val = Dataset.from_df(
            df=df_val,
            smiles_columns=args.smiles_columns,
            features_columns=args.features_columns,
            targets_columns=args.targets_columns,
            n_jobs=args.n_jobs,
        )
        dataset_val.set_status(graph_kernel_type=args.graph_kernel_type,
                               features_generators=args.features_generators, 
                               features_combination=args.features_combination)
        if args.cache_path is not None:
            dataset_val.set_cache(CachedDict.load(filename=args.cache_path))
        dataset.unify_datatype(dataset_val.X_graph)
    else:
        dataset_val = None

    if args.separate_test_path is not None:
        df_test = pd.read_csv(args.separate_test_path)
        dataset_test = Dataset.from_df(
            df=df_test,
            smiles_columns=args.smiles_columns,
            features_columns=args.features_columns,
            targets_columns=args.targets_columns,
            n_jobs=args.n_jobs,
        )
        dataset_test.set_status(graph_kernel_type=args.graph_kernel_type,
                                features_generators=args.features_generators, 
                                features_combination=args.features_combination)
        if args.cache_path is not None:
            dataset_test.set_cache(CachedDict.load(filename=args.cache_path))
        dataset.unify_datatype(dataset_test.X_graph)
    else:
        dataset_test = None

    if args.num_splits == 1:
        datasets = [dataset]
    else:
        datasets = dataset_split(
            dataset=dataset,
            split_type="random",
            sizes=[1 / args.num_splits] * args.num_splits,
        )
    # set kernel_config
    kernel_config = get_kernel_config(
        dataset=dataset,
        graph_kernel_type=args.graph_kernel_type,
        mgk_hyperparameters_files=args.graph_hyperparameters,
        features_hyperparameters_file=args.features_hyperparameters,
    )
    optuna_bayesian_optimization(
        save_dir=args.save_dir,
        datasets=datasets,
        dataset_val=dataset_val,
        dataset_test=dataset_test,
        kernel_config=kernel_config,
        task_type=args.task_type,
        model_type=args.model_type,
        metric=args.metric,
        cross_validation=args.cross_validation,
        n_splits=args.n_splits,
        split_type=args.split_type,
        split_sizes=args.split_sizes,
        num_folds=args.num_folds,
        num_iters=args.num_iters,
        alpha=args.alpha_,
        alpha_bounds=args.alpha_bounds,
        d_alpha=args.d_alpha,
        C=args.C_,
        C_bounds=args.C_bounds,
        d_C=args.d_C,
        seed=args.seed,
    )


def mgk_optuna_multi_datasets(arguments=None):
    args = OptunaMultiDatasetArgs().parse_args(arguments)
    if args.cache_path is not None:
        print(f"**\tLoading cache file: {args.cache_path}.\t**")
        cache = CachedDict.load(filename=args.cache_path)
    else:
        cache = CachedDict()
    print("Preprocessing Dataset.")
    datasets = []
    for i, data_path in enumerate(args.data_paths):
        dataset = Dataset.from_df(
            df=pd.read_csv(data_path),
            smiles_columns=args.smiles_columns_[i],
            features_columns=args.features_columns_[i],
            targets_columns=args.targets_columns_[i],
            n_jobs=args.n_jobs,
        )
        dataset.set_status(graph_kernel_type=args.graph_kernel_type,
                           features_generators=args.features_generators, 
                           features_combination=args.features_combination)
        dataset.set_cache(cache)
        dataset.unify_datatype()
        datasets.append(dataset)
    print("Preprocessing Dataset Finished.")
    # set kernel_config
    kernel_config = get_kernel_config(
        dataset=dataset,
        graph_kernel_type=args.graph_kernel_type,
        mgk_hyperparameters_files=args.graph_hyperparameters,
        features_hyperparameters_file=args.features_hyperparameters,
    )
    optuna_bayesian_optimization_gpr_multi_datasets(
        save_dir=args.save_dir,
        kernel_config=kernel_config,
        datasets=datasets,
        tasks_type=args.tasks_type,
        metrics=args.metrics,
        cross_validation="leave-one-out",
        num_folds=1,
        num_iters=args.num_iters,
        alpha=args.alpha_,
        alpha_bounds=args.alpha_bounds,
        d_alpha=args.d_alpha,
        seed=args.seed,
    )


def mgk_embedding(arguments=None):
    args = EmbeddingArgs().parse_args(arguments)
    # read data set.
    dataset = Dataset.from_df(
        df=pd.read_csv(args.data_path),
        smiles_columns=args.smiles_columns,
        features_columns=args.features_columns,
        targets_columns=args.targets_columns,
        n_jobs=args.n_jobs,
    )
    dataset.set_status(graph_kernel_type=args.graph_kernel_type,
                       features_generators=args.features_generators,
                       features_combination=args.features_combination)
    if args.cache_path is not None:
        print(f"**\tLoading cache file: {args.cache_path}.\t**")
        dataset.set_cache(CachedDict.load(filename=args.cache_path))
    dataset.unify_datatype()
    # set kernel_config
    kernel_config = get_kernel_config(
        dataset=dataset,
        graph_kernel_type=args.graph_kernel_type,
        mgk_hyperparameters_files=args.graph_hyperparameters,
        features_hyperparameters_file=args.features_hyperparameters,
    )
    if args.embedding_algorithm == "tSNE":
        # compute data embedding.
        R = kernel_config.kernel(dataset.X)
        d = R.diagonal() ** -0.5
        K = d[:, None] * R * d[None, :]
        # get the distance matrix
        D = np.sqrt(np.maximum(0, 2 - 2 * K))
        embedding = TSNE(
            n_components=args.n_components,
            perplexity=args.perplexity,
            n_iter=args.n_iter,
            n_jobs=args.n_jobs,
        ).fit_transform(D)
    elif args.embedding_algorithm == "kPCA":
        R = kernel_config.kernel(dataset.X) + 0.01 * np.eye(len(dataset))
        embedding = KernelPCA(
            n_components=args.n_components, kernel="precomputed", n_jobs=args.n_jobs
        ).fit_transform(R)
    else:
        raise ValueError(f"Invalid embedding algorithm {args.embedding_algorithm}.")
    # embedding dataframe.
    df = pd.DataFrame({"repr": dataset.repr})
    for i in range(args.n_components):
        df["embedding_%d" % i] = embedding[:, i]
    for i in range(dataset.N_tasks):
        df["target_%d" % i] = dataset.y[:, i]
    df.to_csv("%s/%s.csv" % (args.save_dir, args.embedding_algorithm), index=False)

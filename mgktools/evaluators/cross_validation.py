#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cross-validation and model evaluation module.

This module provides the Evaluator class for performing cross-validation,
external validation, and model interpretation on molecular property prediction tasks.

Classes
-------
Evaluator
    Main class for evaluating machine learning models with various cross-validation
    strategies and optional interpretability analysis.
"""
from typing import List, Optional, Literal, Tuple
import os
import math
from tqdm import tqdm
import pandas as pd
import numpy as np
import inspect
from rdkit import Chem
from sklearn.model_selection import KFold
from mgktools.interpret.utils import save_mols_pkl
from mgktools.data.data import Dataset
from mgktools.data.split import get_data_from_index, dataset_split
from mgktools.evaluators.metric import Metric, metric_regression, metric_binary


class Evaluator:
    """
    Evaluator for machine learning models on molecular property prediction tasks.

    This class provides comprehensive evaluation capabilities including multiple
    cross-validation strategies (k-fold, leave-one-out, Monte Carlo), external
    validation, and optional interpretability analysis through atomic and
    molecular attribution.

    Parameters
    ----------
    save_dir : str
        Directory path to save all output files (predictions, metrics, logs).
    dataset : Dataset
        Dataset instance used for cross-validation or as training data.
    model : object
        Machine learning model instance (GPR, GPC, SVM, etc.).
    task_type : {'regression', 'binary', 'multi-class'}
        Type of prediction task.
    metrics : List[Metric], optional
        List of evaluation metrics to compute, by default None.
    cross_validation : {'kFold', 'leave-one-out', 'Monte-Carlo', 'no'}, optional
        Cross-validation strategy, by default 'Monte-Carlo'.
    n_splits : int, optional
        Number of splits for k-fold CV, by default None.
    split_type : {'random', 'scaffold_order', 'scaffold_random'}, optional
        Method for splitting data, by default None.
    split_sizes : List[float], optional
        Proportions for train/val/test splits, by default None.
    num_folds : int, optional
        Number of times to repeat cross-validation, by default 1.
    evaluate_train : bool, optional
        Whether to evaluate on training set, by default False.
    n_similar : int, optional
        Number of similar molecules to output for each prediction, by default None.
    kernel : object, optional
        Kernel function for finding similar molecules, by default None.
    n_core : int, optional
        Number of samples for Nystrom approximation core set, by default None.
    atomic_attribution : bool, optional
        Whether to perform atomic attribution analysis, by default False.
    molecular_attribution : bool, optional
        Whether to perform molecular attribution analysis, by default False.
    seed : int, optional
        Random seed for reproducibility, by default 0.
    verbose : bool, optional
        Whether to print progress information, by default True.

    Attributes
    ----------
    save_dir : str
        Output directory path.
    dataset : Dataset
        Input dataset.
    model : object
        Fitted model.

    Examples
    --------
    >>> evaluator = Evaluator(
    ...     save_dir='./results',
    ...     dataset=dataset,
    ...     model=model,
    ...     task_type='regression',
    ...     cross_validation='kFold',
    ...     n_splits=5,
    ...     metrics=['rmse', 'r2']
    ... )
    >>> mean_score = evaluator.run_cross_validation()
    """

    def __init__(self,
                 save_dir: str,
                 dataset: Dataset,
                 model,
                 task_type: Literal["regression", "binary", "multi-class"],
                 metrics: List[Metric] = None,
                 cross_validation: Literal["kFold", "leave-one-out", "Monte-Carlo", "no"] = "Monte-Carlo",
                 n_splits: int = None,
                 split_type: Literal["random", "scaffold_order", "scaffold_random"] = None,
                 split_sizes: List[float] = None,
                 num_folds: int = 1,
                 evaluate_train: bool = False,
                 n_similar: Optional[int] = None,
                 kernel=None,
                 n_core: int = None,
                 atomic_attribution: bool = False,
                 molecular_attribution: bool = False,
                 seed: int = 0,
                 verbose: bool = True
                 ):
        self.save_dir = save_dir
        if self.write_file:
            if not os.path.exists(self.save_dir):
                os.mkdir(self.save_dir)
            self.logfile = open("%s/results.log" % self.save_dir, "w")
        self.dataset = dataset
        self.model = model
        self.task_type = task_type
        self.cross_validation = cross_validation
        self.n_splits = n_splits
        self.split_type = split_type
        self.split_sizes = split_sizes
        self.metrics = metrics
        self.num_folds = num_folds
        self.evaluate_train = evaluate_train
        self.n_similar = n_similar
        self.kernel = kernel
        self.n_core = n_core
        self.atomic_attribution = atomic_attribution
        self.molecular_attribution = molecular_attribution
        self.seed = seed
        self.verbose = verbose

    @property
    def write_file(self) -> bool:
        if self.save_dir is None:
            return False
        else:
            return True

    def run_cross_validation(self) -> float:
        assert not self.atomic_attribution, "Atomic attribution interpretability is only supported for run_external()."
        assert not self.molecular_attribution, "Molecular attribution interpretability is only supported for run_external()."
        # Leave-One-Out cross validation
        if self.cross_validation == "leave-one-out":
            assert self.n_splits is None, "nfold must be None for LOOCV."
            assert self.split_type is None, "split_type must be None for LOOCV."
            assert self.split_sizes is None, "split_sizes must be None for LOOCV."
            return self.eval_loocv()
        elif self.cross_validation == "kFold":
            assert self.n_splits is not None, "n_splits must be specified for nfold cross-validation."
            # repeat cross-validation for num_folds times
            metrics_list = []
            for i in range(self.num_folds):
                kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed + i)
                kf.get_n_splits(self.dataset.X)
                for i_fold, (train_index, test_index) in enumerate(kf.split(self.dataset.X)):
                    dataset_train = get_data_from_index(self.dataset, train_index)
                    dataset_test = get_data_from_index(self.dataset, test_index)
                    df_predict, df_ = self.evaluate_train_test(dataset_train, dataset_test)
                    df_predict.to_csv("%s/kFold_%d-%d_prediction.csv" % (self.save_dir, i, i_fold), index=False)
                    df_["seed"] = self.seed + i
                    df_["split"] = i_fold
                    metrics_list.append(df_)
            df_metrics = pd.concat(metrics_list, ignore_index=True)
            df_metrics.to_csv("%s/kFold_metrics.csv" % self.save_dir, index=False)
            self.log("kFold cross-validation performance:")
            self.log_metrics(df_metrics)
            return df_metrics['value'].mean()
        elif self.cross_validation == "Monte-Carlo":
            assert self.split_type is not None, "split_type must be specified for Monte-Carlo cross-validation."
            assert self.split_sizes is not None, "split_sizes must be specified for Monte-Carlo cross-validation."
            metrics_list = []
            for i in range(self.num_folds):
                if len(self.split_sizes) == 2:
                    dataset_train, dataset_test = dataset_split(
                        self.dataset,
                        split_type=self.split_type,
                        sizes=self.split_sizes,
                        seed=self.seed + i)
                # the second part, validation set, is abandoned.
                elif len(self.split_sizes) == 3:
                    dataset_train, _, dataset_test = dataset_split(
                        self.dataset,
                        split_type=self.split_type,
                        sizes=self.split_sizes,
                        seed=self.seed + i)
                else:
                    raise ValueError("split_sizes must be 2 or 3.")
                df_predict, df_ = self.evaluate_train_test(dataset_train, dataset_test)
                df_predict.to_csv("%s/test_%d_prediction.csv" % (self.save_dir, i), index=False)
                df_["seed"] = self.seed + i
                metrics_list.append(df_)
            df_metrics = pd.concat(metrics_list, ignore_index=True)
            df_metrics.to_csv("%s/Monte-Carlo_metrics.csv" % self.save_dir, index=False)
            self.log("Monte-Carlo cross-validation performance:")
            self.log_metrics(df_metrics)
            return df_metrics['value'].mean()
        elif self.cross_validation == "no":
            raise ValueError("When set cross_validation to 'no', please use run_external() not eval_cross_validation.")
        else:
            raise ValueError("Unsupported cross-validation method %s." % self.cross_validation)

    def run_external(self, dataset_test: Dataset, name='test_ext'):
        # assert self.cross_validation == "no", "cross_validation must be 'no' for run_external()."
        if self.evaluate_train:
            df_predict, df_metrics = self.evaluate_train_test(self.dataset, self.dataset, loocv=False)
            df_predict.to_csv("%s/train_prediction.csv" % self.save_dir, index=False)
            if df_metrics is not None:
                # Calculate metrics values.
                df_metrics.to_csv("%s/train_metrics.csv" % self.save_dir, index=False)
                self.log("Training set performance:")
                self.log_metrics(df_metrics)
        df_predict, df_metrics = self.evaluate_train_test(self.dataset, dataset_test, loocv=False)
        df_predict.to_csv("%s/%s_prediction.csv" % (self.save_dir, name), index=False)
        if df_metrics is not None:
            # Calculate metrics values.
            df_metrics.to_csv("%s/%s_metrics.csv" % (self.save_dir, name), index=False)
            self.log("External test set performance:")
            self.log_metrics(df_metrics)
        if self.atomic_attribution:
            mols = self.get_interpreted_mols(dataset_test)
            save_mols_pkl(mols=mols, path=self.save_dir, filename="atomic_attribution.pkl")
        if self.molecular_attribution:
            n_mol = 10000
            c_percentage, c_y = self.model.predict_interpretable(dataset_test.X)
            idx_list = np.argsort(-np.fabs(c_percentage))[:, :min(n_mol, c_percentage.shape[1])]
            for i, idx in enumerate(idx_list):
                df = pd.DataFrame({"smiles_train": np.array(self.dataset.repr)[idx],
                                   "contribution_percentage": c_percentage[i][idx],
                                   "contribution_value": c_y[i][idx]})
                df.to_csv("%s/molecular_attribution_mol%d.csv" % (self.save_dir, i), index=False)
        return df_metrics['value'].mean()

    def eval_loocv(self) -> float:
        df_predict, df_metrics = self.evaluate_train_test(self.dataset, self.dataset)
        df_predict.to_csv("%s/loocv_prediction.csv" % self.save_dir, index=False)
        df_metrics.to_csv("%s/loocv_metrics.csv" % self.save_dir, index=False)
        # Calculate metrics values.
        self.log("Leave-one-out cross-validation performance:")
        self.log_metrics(df_metrics)
        return df_metrics['value'].mean()

    def evaluate_train_test(self, dataset_train: Dataset,
                            dataset_test: Dataset, loocv=True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        test_targets_known = (dataset_train.N_tasks == dataset_test.N_tasks)
        pred_dict = {"repr": dataset_test.repr}
        y_preds = []
        for i in range(dataset_train.N_tasks):
            if self.cross_validation == "leave-one-out" and loocv:
                y_pred, y_std = self.model.predict_loocv(dataset_test.X, dataset_test.y[:, i], return_std=True)
            else:
                self.fit(dataset_train.X, dataset_train.y[:, i])
                if self.task_type == "regression":
                    sig = inspect.signature(self.model.predict)
                    if "return_std" in sig.parameters:
                        y_pred, y_std = self.model.predict(dataset_test.X, return_std=True)
                    else:
                        y_pred, y_std = self.model.predict(dataset_test.X), None
                elif self.task_type == "binary":
                    y_std = None
                    if hasattr(self.model, "predict_proba"):
                        y_pred = self.model.predict_proba(dataset_test.X)
                    else:
                        y_pred = self.model.predict(dataset_test.X)
            if test_targets_known:
                pred_dict["target_%d" % i] = dataset_test.y[:, i]
            pred_dict["predict_%d" % i] = y_pred
            if y_std is not None:
                pred_dict["uncertainty_%d" % i] = y_std
            y_preds.append(y_pred.tolist())
        if self.n_similar is not None:
            y_similar = self.get_similar_info(dataset_test.X, dataset_train.X, dataset_train.repr, self.n_similar)
            pred_dict["y_similar"] = y_similar
        df_predict = pd.DataFrame(pred_dict)
        if test_targets_known:
            metrics_data = []
            for metric in self.metrics:
                for i in range(dataset_train.N_tasks):
                    v = self.eval_metric(dataset_test.y[:, i], y_preds[i], metric)
                    metrics_data.append([metric, i, v])
            df_metrics = pd.DataFrame(metrics_data, columns=["metric", "no_targets_columns", "value"])
            return df_predict, df_metrics
        else:
            return df_predict, None

    def fit(self, X, y):
        if self.n_core is not None:
            idx = np.random.choice(np.arange(len(X)), self.n_core, replace=False)
            C_train = X[idx]
            self.model.fit(C_train, X, y)
        else:
            self.model.fit(X, y)

    def get_interpreted_mols(self, dataset_test) -> List[Chem.Mol]:
        X_test = dataset_test.X
        assert dataset_test.mols.shape[1] == 1, "interpretability is only valid for single-graph data"
        mols_to_be_interpret = dataset_test.mols.ravel()
        batch_size = 1
        N_batch = math.ceil(len(mols_to_be_interpret) / batch_size)
        for i in tqdm(range(N_batch)):
            start = batch_size * i
            end = batch_size * (i + 1)
            if end > len(mols_to_be_interpret):
                end = len(mols_to_be_interpret)
            g = X_test.ravel()[start:end]
            y_nodes = self.model.predict_nodal(g)
            print(y_nodes)
            k = 0
            for j in range(start, end):
                mol = mols_to_be_interpret[j]
                for atom in mol.GetAtoms():
                    atom.SetProp("atomNote", "%.6f" % y_nodes[k])
                    k += 1
            assert k == len(y_nodes)
        return mols_to_be_interpret.tolist()

    def get_similar_info(self, X_test: np.ndarray, X_train: np.ndarray, 
                         X_repr: List[str], n_most_similar: int) -> List[str]:
        """Get the most similar molecules in the training set for each molecule in the test set.
        
        Parameters
        ----------
        X_test: np.ndarray
            The test set.
        X_train: np.ndarray
            The training set.
        X_repr: List[str]
            The representation of the training set molecules.
        n_most_similar: int
            The number of most similar molecules to be output.
        
        Returns
        -------
        List[str]
            The information of the most similar molecules in the training set and the cross kernel 
            value (similarity) for each molecule in the test set.
        """
        K = self.kernel(X_test, X_train)
        assert (K.shape == (len(X_test), len(X_train)))
        similar_info = []
        kindex = np.argsort(-K)[:, :min(n_most_similar, K.shape[1])]
        for i, index in enumerate(kindex):
            def round5(x):
                return ",%.5f" % x
            k = list(map(round5, K[i][index]))
            repr = np.asarray(X_repr)[index]
            info = ";".join(list(map(str.__add__, repr, k)))
            similar_info.append(info)
        return similar_info

    def eval_metric(self, y, y_pred, metric):
        if self.task_type == "regression":
            return metric_regression(y, y_pred, metric)
        elif self.task_type == "binary":
            return metric_binary(y, y_pred, metric)
        elif self.task_type == "multi-class":
            raise NotImplementedError("multi-class classification is not supported yet.")
            # return metric_multiclass(y, y_pred, metric)

    def log_metrics(self, df_metrics: pd.DataFrame):
        N_targets_columns = df_metrics["no_targets_columns"].max() + 1
        for metric in self.metrics:
            df_ = df_metrics[df_metrics["metric"] == metric]
            assert len(df_) > 0
            if len(df_) == 1:
                self.log(f"Metric({metric}): %.5f" % df_["value"].iloc[0])
            else:
                self.log(f"Metric({metric}): %.5f +/- %.5f" % (df_["value"].mean(), df_["value"].std()))
        for i in range(N_targets_columns):
            df_ = df_metrics[df_metrics["no_targets_columns"] == i]
            assert len(df_) > 0
            if len(df_) == 1:
                self.log(f"Target({i}): %.5f" % df_["value"].iloc[0])
            else:
                self.log(f"Target({i}): %.5f +/- %.5f" % (df_["value"].mean(), df_["value"].std()))
        for i in range(N_targets_columns):
            for metric in self.metrics:
                df_ = df_metrics[(df_metrics["metric"] == metric) & (df_metrics["no_targets_columns"] == i)]
                assert len(df_) > 0
                if len(df_) == 1:
                    self.log(f"Target({i}),Metric({metric}): %.5f" % df_["value"].iloc[0])
                else:
                    self.log(f"Target({i}),Metric({metric}): %.5f +/- %.5f" % (df_["value"].mean(), df_["value"].std()))

    def log(self, info: str):
        if self.verbose:
            if self.write_file:
                self.logfile.write(info + "\n")
            else:
                print(info)
        else:
            pass
    

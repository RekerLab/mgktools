#!/usr/bin/env python
# -*- coding: utf-8 -*-
from tap import Tap
from typing import List, Literal, Tuple, Optional
import os
from mgktools.evaluators.metric import Metric, AVAILABLE_METRICS_REGRESSION, AVAILABLE_METRICS_BINARY
from mgktools.features_mol.features_generators import FeaturesGenerator


class CacheArgs(Tap):
    data_paths: List[str]
    """The Path of a list of CSV files."""
    smiles_columns: List[str] = None
    """Name of the columns containing single SMILES string."""
    cache_graph: bool = False
    """Convert SMILES into GraphDot graphs and cached."""
    features_generators_name: List[str] = None
    """Method(s) of generating additional features_mol."""
    n_jobs: int = 8
    """The cpu numbers used for parallel computing."""
    cache_path: str = "cache.pkl"
    """The Path of the output cache file."""
    
    @property
    def features_generators(self) -> Optional[List[FeaturesGenerator]]:
        if self.features_generators_name is None:
            return None
        else:
            return [FeaturesGenerator(features_generator_name=fg) for fg in self.features_generators_name]


class CommonArgs(Tap):
    save_dir: str
    """The output directory."""
    cache_path: str = None
    """The Path of the output cache file."""
    data_path: str = None
    """The Path of input data CSV file."""
    smiles_columns: List[str] = None
    """
    Name of the columns containing single SMILES string.
    """
    features_columns: List[str] = None
    """
    Name of the columns containing additional features_mol such as temperature, 
    pressuer.
    """
    targets_columns: List[str] = None
    """
    Name of the columns containing target values. Multi-targets are not implemented yet.
    """
    features_generators_name: List[str] = None
    """Method(s) of generating additional features_mol."""
    features_combination: Literal["concat", "mean"] = None
    """How to combine features vector for mixtures."""
    features_mol_normalize: bool = False
    """Nomralize the molecular features_mol."""
    features_add_normalize: bool = False
    """Nomralize the additonal features_mol."""
    n_jobs: int = 8
    """The cpu numbers used for parallel computing."""

    def __init__(self, *args, **kwargs):
        super(CommonArgs, self).__init__(*args, **kwargs)

    @property
    def features_generators(self) -> Optional[List[FeaturesGenerator]]:
        if self.features_generators_name is None:
            return None
        else:
            return [FeaturesGenerator(features_generator_name=fg) for fg in self.features_generators_name]

    def process_args(self) -> None:
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        if self.cache_path is not None:
            assert os.path.exists(self.cache_path)
        if self.features_generators_name is not None and self.features_combination is None:
            self.features_combination = "concat"


class KernelArgs(CommonArgs):
    graph_kernel_type: Literal["graph", "pre-computed", "no"]
    """The type of kernel to use."""
    graph_hyperparameters: List[str] = None
    """hyperparameters files for graph kernel."""
    features_hyperparameters: str = None
    """hyperparameters file for molecular descriptors."""
    def process_args(self) -> None:
        super().process_args()
        if self.graph_kernel_type == "graph":
            assert self.graph_hyperparameters is not None
        if self.features_generators_name is not None:
            assert self.features_hyperparameters is not None
        if self.graph_kernel_type == "no" and self.features_generators_name is None:
            raise ValueError("At least one of graph kernel or features kernel should be used.")


class ModelArgs(Tap):
    model_type: Literal["gpr", "gpr-nystrom", "gpr-nle", "svr", "gpc", "svc"]
    """The machine learning model to use."""
    alpha: str = None
    """data noise used in gpr."""
    C: str = None
    """C parameter used in Support Vector Machine."""
    ensemble: bool = False
    """use ensemble model."""
    n_estimators: int = 1
    """Ensemble model with n estimators."""
    n_samples_per_model: int = None
    """The number of samples use in each estimator."""
    ensemble_rule: Literal["smallest_uncertainty", "weight_uncertainty",
                           "mean"] = "weight_uncertainty"
    """The rule to combining prediction from estimators."""
    n_local: int = 500
    """The number of samples used in Naive Local Experts."""
    n_core: int = None
    """The number of samples used in Nystrom core set."""


class TrainArgs(KernelArgs, ModelArgs):
    task_type: Literal["regression", "binary", "multi-class"]
    """Type of task."""
    cross_validation: Literal["kFold", "leave-one-out", "Monte-Carlo", "no"] = "no"
    """The way to split data for cross-validation."""
    n_splits: int = None
    """The number of fold for kFold CV."""
    split_type: Literal["random", "scaffold_order", "scaffold_random", "stratified"] = None
    """Method of splitting the data into train/test sets."""
    split_sizes: List[float] = None
    """Split proportions for train/test sets."""
    num_folds: int = 1
    """Number of folds when performing cross validation."""
    seed: int = 0
    """Random seed."""
    metric: Metric = None
    """metric"""
    extra_metrics: List[Metric] = []
    """Metrics"""
    evaluate_train: bool = False
    """If set True, evaluate the model on training set."""
    n_similar: int = None
    """The number of most similar molecules in the training set will be saved."""
    separate_test_path: str = None
    """Path to separate test set, optional."""
    atomic_attribution: bool = False
    """Output interpretable results on atomic attribution."""
    molecular_attribution: bool = False
    """Output interpretable results on molecular attribution."""

    @property
    def metrics(self) -> List[Metric]:
        return [self.metric] + self.extra_metrics

    @property
    def alpha_(self) -> float:
        if self.alpha is None:
            return None
        elif isinstance(self.alpha, float):
            return self.alpha
        elif os.path.exists(self.alpha):
            return float(open(self.alpha, "r").read())
        else:
            return float(self.alpha)

    @property
    def C_(self) -> float:
        if self.C is None:
            return None
        elif isinstance(self.C, float):
            return self.C
        elif os.path.exists(self.C):
            return float(open(self.C, "r").read())
        else:
            return float(self.C)

    def process_args(self) -> None:
        super().process_args()
        if self.task_type == "regression":
            assert self.model_type in ["gpr", "gpr-nystrom", "gpr-nle", "svr"]
            for metric in self.metrics:
                assert metric in AVAILABLE_METRICS_REGRESSION
        elif self.task_type == "binary":
            assert self.model_type in ["gpc", "svc", "gpr"]
            for metric in self.metrics:
                assert metric in AVAILABLE_METRICS_BINARY
        elif self.task_type == "multi-class":
            raise NotImplementedError("Multi-class classification is not implemented yet.")

        if self.cross_validation == "leave-one-out":
            assert self.num_folds == 1
            assert self.model_type == "gpr"
        elif self.cross_validation == "no":
            assert self.separate_test_path is not None, "separate_test_path should be provided for no cross-validation."
        elif self.cross_validation == "Monte-Carlo":
            if self.split_type.startswith("scaffold"):
                assert len(self.smiles_columns) == 1, "Single SMILES column is required for scaffold splitting."
        elif self.cross_validation == "kFold":
            assert self.n_splits is not None, "n_splits should be provided for kFold cross-validation."

        if self.model_type.startswith("gpr"):
            assert self.alpha is not None

        if self.model_type in ["svc", "svr"]:
            assert self.C is not None

        if self.ensemble:
            assert self.n_samples_per_model is not None

        if self.separate_test_path is not None:
            assert self.cross_validation == "no", "cross-validation should be set to no when separate_test_path is provided."

        if self.atomic_attribution or self.molecular_attribution:
            assert self.cross_validation == "no", "cross-validation should be set to no for interpretability."
            assert self.separate_test_path is not None, "separate_test_path should be provided for interpretability."
            assert self.model_type == "gpr", "Set model_type to gpr for interpretability."
            if self.atomic_attribution:
                assert self.graph_kernel_type == "graph", "Set graph_kernel_type to graph for interpretability"


class GradientOptArgs(KernelArgs):
    loss: Literal["loocv", "likelihood"] = "loocv"
    """The target loss function to minimize or maximize."""
    optimizer: str = None
    """Optimizer implemented in scipy.optimize.minimize are valid: L-BFGS-B, SLSQP, Nelder-Mead, Newton-CG, etc."""
    alpha: str = None
    """data noise used in gpr."""

    @property
    def alpha_(self) -> float:
        if self.alpha is None:
            return None
        elif isinstance(self.alpha, float):
            return self.alpha
        elif os.path.exists(self.alpha):
            return float(open(self.alpha, "r").read())
        else:
            return float(self.alpha)

    def process_args(self) -> None:
        super().process_args()


class OptunaArgs(TrainArgs):
    num_iters: int = 100
    """Number of hyperparameter choices to try."""
    alpha_bounds: Tuple[float, float] = None
    """Bounds of alpha used in GPR."""
    d_alpha: float = None
    """The step size of alpha to be optimized."""
    C_bounds: Tuple[float, float] = None #  (1e-3, 1e3)
    """Bounds of C used in SVC."""
    d_C: float = None
    """The step size of C to be optimized."""
    num_splits: int = 1
    """split the dataset randomly into no. subsets to reduce computational costs."""

    def process_args(self) -> None:
        super().process_args()


class OptunaMultiDatasetArgs(Tap):
    save_dir: str
    """The output directory."""
    cache_path: str = None
    """The Path of the output cache file."""
    n_jobs: int = 1
    """The cpu numbers used for parallel computing."""
    data_paths: List[str]
    """The Path of input data CSV files."""
    smiles_columns: str = None
    """
    Name of the columns containing single SMILES string.
    E.g.: "smiles;smiles;smiles1,smiles2"
    """
    features_columns: str = None
    """
    Name of the columns containing additional features_mol such as temperature, 
    pressuer.
    """
    targets_columns: str = None
    """
    Name of the columns containing target values.
    """
    features_generators_name: List[str] = None
    """Method(s) of generating additional features_mol."""
    features_combination: Literal["concat", "mean"] = None
    """How to combine features vector for mixtures."""
    features_mol_normalize: bool = False
    """Nomralize the molecular features_mol."""
    features_add_normalize: bool = False
    """Nomralize the additonal features_mol."""
    tasks_type: List[Literal["regression", "binary", "multi-class"]]
    """
    Type of task.
    """
    metrics: List[Metric]
    """taget metrics to be optimized."""
    num_iters: int = 100
    """Number of hyperparameter choices to try."""
    alpha: str = None
    """data noise used in gpr."""
    alpha_bounds: Tuple[float, float] = None
    """Bounds of alpha used in GPR."""
    d_alpha: float = None
    """The step size of alpha to be optimized."""
    seed: int = 0
    """Random seed."""
    graph_kernel_type: Literal["graph", "pre-computed", "no"]
    """The type of kernel to use."""
    graph_hyperparameters: List[str] = None
    """hyperparameters files for graph kernel."""
    features_hyperparameters: str = None
    """hyperparameters file for molecular descriptors."""

    @property
    def features_generators(self) -> Optional[List[FeaturesGenerator]]:
        if self.features_generators_name is None:
            return None
        else:
            return [FeaturesGenerator(features_generator_name=fg) for fg in self.features_generators_name]

    @property
    def alpha_(self) -> float:
        if self.alpha is None:
            return None
        elif isinstance(self.alpha, float):
            return self.alpha
        elif os.path.exists(self.alpha):
            return float(open(self.alpha, "r").read())
        else:
            return float(self.alpha)

    def process_args(self) -> None:
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        none_list = [None] * len(self.data_paths)
        self.smiles_columns_ = [i.split(",") for i in self.smiles_columns.split(";")]
        self.features_columns_ = [None if i == '' else i.split(",") for i in self.features_columns.split(";")] if self.features_columns is not None else none_list
        self.targets_columns_ = [i.split(",") for i in self.targets_columns.split(";")]

        if self.graph_kernel_type == "graph":
            assert self.graph_hyperparameters is not None
        if self.features_generators_name is not None:
            assert self.features_hyperparameters is not None
        if self.graph_kernel_type == "no" and self.features_generators_name is None:
            raise ValueError("At least one of graph kernel or features kernel should be used.")


class EmbeddingArgs(KernelArgs):
    embedding_algorithm: Literal["tSNE", "kPCA"] = "tSNE"
    """Algorithm for data embedding."""
    n_components: int = 2
    """Dimension of the embedded space."""
    perplexity: float = 30.0
    """
    The perplexity is related to the number of nearest neighbors that
    is used in other manifold learning algorithms. Larger datasets
    usually require a larger perplexity. Consider selecting a value
    different results.
    """
    n_iter: int = 1000
    """Maximum number of iterations for the optimization. Should be at least 250."""

    def process_args(self) -> None:
        super().process_args()

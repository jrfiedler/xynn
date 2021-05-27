"""
Base classes for estimators (Scikit-learn-style models)

"""
import os
import json
import random
import inspect
from abc import ABCMeta, abstractmethod
from typing import Union, List, Tuple, Callable, Dict, Optional, Any, Type

import numpy as np
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch.nn.functional import softmax

from ..embedding import EmbeddingBase, BasicEmbedding, LinearEmbedding, DefaultEmbedding
from ..dataset import TabularDataLoader
from ..train import train, now
from .modules import BaseNN


def _set_seed(seed):
    # https://discuss.pytorch.org/t/reproducibility-with-all-the-bells-and-whistles
    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _log(info, filepath):
    dirpath, _ = os.path.split(filepath)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    with open(filepath, "w") as outfile:
        json.dump(info, outfile, indent=4)
        outfile.flush()
        os.fsync(outfile.fileno())


def _param_json(value):
    if inspect.isclass(value):
        return value.__name__
    if isinstance(value, EmbeddingBase):
        return str(value)
    return value


def _param_repr(value):
    if inspect.isclass(value):
        return value.__name__
    return repr(value)


ESTIMATOR_INIT_DOC = """
Parameters
----------
embedding_num : "auto", embedding.EmbeddingBase, or None, optional
    embedding for numeric fields; default is auto
embedding_cat : "auto", embedding.EmbeddingBase, or None, optional
    embedding for categorical fields; default is auto
embedding_l1_reg : float, optional
    value for l1 regularization of embedding vectors; default is 0.0
embedding_l2_reg : float, optional
    value for l2 regularization of embedding vectors; default is 0.0
{}
mlp_hidden_sizes : int or iterable of int, optional
    sizes for the linear transformations between the MLP input and
    the output size needed based on the target; default is (512, 256, 128, 64)
mlp_activation : subclass of torch.nn.Module, optional
    default is nn.LeakyReLU
mlp_use_bn : boolean, optional
    whether to use batch normalization between MLP linear layers;
    default is True
mlp_bn_momentum : float, optional
    only used if `mlp_use_bn` is True; default is 0.01
mlp_dropout : float, optional
    whether and how much dropout to use between MLP linear layers;
    `0.0 <= mlp_dropout < 1.0`; default is 0.0
mlp_l1_reg : float, optional
    value for l1 regularization of MLP weights; default is 0.0
mlp_l2_reg : float, optional
    value for l2 regularization of MLP weights; default is 0.0
mlp_leaky_gate : boolean, optional
    whether to include a "leaky gate" layer before the MLP layers;
    default is True
mlp_use_skip : boolean, optional
    use a side path in the MLP containing just the optional leaky gate
    plus single linear layer; default is True
loss_fn : "auto" or PyTorch loss function, optional
    if "auto", nn.CrossEntropyLoss is used; default is "auto"
seed : int or None, optional
    if int, seed for `torch.manual_seed` and `numpy.random.seed`;
    if None, no seeding is done; default is None
device : string or torch.device, optional
    default is "cpu"

"""


class BaseEstimator(metaclass=ABCMeta):
    """
    Base class for Scikit-learn style classes in this package

    """

    def __init__(
        self,
        embedding_num: Optional[Union[str, EmbeddingBase]] = "auto",
        embedding_cat: Optional[Union[str, EmbeddingBase]] = "auto",
        loss_fn: Union[str, Callable] = "auto",
        seed: Union[int, None] = None,
        device: Union[str, torch.device] = "cpu",
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.task = ""
        self.embedding_num = embedding_num
        self.embedding_cat = embedding_cat
        self.model_kwargs = model_kwargs if model_kwargs else {}
        self.seed = seed
        self.train_info = []
        self._device = torch.device(device)
        self._model = None
        self._model_class: Optional[Type[BaseNN]] = None
        self._num_numeric_fields = 0
        self._num_categorical_fields = 0
        if seed is not None:
            _set_seed(seed)

        # record init parameters, mostly for logging
        init_params_bef = {
            "embedding_num": embedding_num,
            "embedding_cat": embedding_cat,
        }
        init_params_aft = {
            "loss_fn": loss_fn,
            "seed": seed,
            "device": device
        }
        self.init_parameters = {
            key: val
            for params in (init_params_bef, model_kwargs, init_params_aft)
            for key, val in params.items()
        }

    def __repr__(self):
        init_params = ",\n    ".join(
            f"{key}={_param_repr(val)}" for key, val in self.init_parameters.items()
        )
        repr_str = f"{self.__class__.__name__}(\n    {init_params},\n)"
        return repr_str

    def mlp_weight_sum(self) -> Tuple[float, float]:
        """
        Sum of absolute value and square of weights in MLP layers

        Return
        ------
        w1 : sum of absolute value of MLP weights
        w2 : sum of squared MLP weights

        """
        if self._model:
            return self._model.mlp_weight_sum()
        return 0.0, 0.0

    def embedding_sum(self) -> Tuple[float, float]:
        """
        Sum of absolute value and square of embedding values

        Return
        ------
        e1_sum : sum of absolute value of embedding values
        e2_sum : sum of squared embedding values

        """
        if self._model:
            return self._model.embedding_sum()
        return 0.0, 0.0

    def num_parameters(self) -> int:
        """
        Number of trainable parameters in the model

        Return
        ------
        int number of trainable parameters

        """
        if self._model:
            return self._model.num_parameters()
        return 0

    def _optimizer_init(self, optimizer, opt_kwargs, scheduler, sch_kwargs):
        self._model.set_optimizer(
            optimizer=optimizer,
            opt_kwargs=opt_kwargs,
            scheduler=scheduler,
            sch_kwargs=sch_kwargs,
        )
        self._model.configure_optimizers()

    def _create_embeddings(self, X_num, X_cat):
        # numeric embedding
        if X_num.shape[1]:
            if self.embedding_num is None:
                if self._require_numeric_embedding:
                    raise ValueError(
                        "embedding_num was set to None; "
                        f"expected zero numeric columns, got {X_num.shape[1]}"
                    )
            elif isinstance(self.embedding_num, EmbeddingBase):
                if not self.embedding_num._isfit:
                    self.embedding_num.fit(X_num)
            else: # initialized with embedding_num = "auto"
                self.embedding_num = LinearEmbedding(device=self._device)
                self.embedding_num.fit(X_num)
        else:
            self.embedding_num = None

        # categorical embedding
        if X_cat.shape[1]:
            if self.embedding_cat is None:
                raise ValueError(
                    "embedding_cat was set to None; "
                    f"expected zero categorical columns, got {X_cat.shape[1]}"
                )
            elif isinstance(self.embedding_cat, EmbeddingBase):
                if not self.embedding_cat._isfit:
                    self.embedding_cat.fit(X_cat)
            else:  # initialized with embedding_cat = "auto"
                self.embedding_cat = DefaultEmbedding(device=self._device)
                self.embedding_cat.fit(X_cat)
        else:
            self.embedding_cat = None

    @abstractmethod
    def _create_model(self, embedding_num, embedding_cat):
        return

    @abstractmethod
    def _fit_init(self, X_num, X_cat, y, warm_start=False):
        return X_num, X_cat, y

    def _convert_x(self, X_num, X_cat, y=None) -> Tuple[Tensor, Union[Tensor, np.ndarray]]:
        if X_num is None and X_cat is None:
            raise TypeError("X_num and X_cat cannot both be None")

        if X_num is None:
            X_num = torch.empty((X_cat.shape[0], 0))
            self._num_numeric_fields = 0
        else:
            self._num_numeric_fields = X_num.shape[1]
            if isinstance(X_num, np.ndarray):
                X_num = torch.from_numpy(X_num).to(dtype=torch.float32)

        if X_cat is None:
            X_cat = torch.empty((X_num.shape[0], 0))
            self._num_categorical_fields = 0
        else:
            self._num_categorical_fields = X_cat.shape[1]

        if X_num.shape[0] != X_cat.shape[0]:
            raise ValueError(
                f"mismatch in shapes for X_num {X_num.shape}, X_cat {X_cat.shape}"
            )
        if y is not None and X_num.shape[0] != y.shape[0]:
            raise ValueError(
                f"mismatch in shapes for X_num {X_num.shape}, "
                f"X_cat {X_cat.shape}, y {y.shape}"
            )

        return X_num, X_cat

    @abstractmethod
    def _convert_y(self, y):
        return y

    def _convert_xy(self, X_num, X_cat, y):
        X_num, X_cat = self._convert_x(X_num, X_cat, y)
        y = self._convert_y(y)
        return X_num, X_cat, y

    def fit(
        self,
        X_num: Optional[Union[Tensor, np.ndarray]],
        X_cat: Optional[Union[Tensor, np.ndarray]],
        y: Union[Tensor, np.ndarray],
        optimizer: Callable,
        opt_kwargs: Optional[Dict[str, Any]] = None,
        scheduler: Optional[Callable] = None,
        sch_kwargs: Optional[Dict[str, Any]] = None,
        val_sets: Optional[List[Tuple[Tensor, Tensor, Tensor]]] = None,
        num_epochs: int = 5,
        batch_size: int = 128,
        warm_start: bool = False,
        extra_metrics: Optional[List[Tuple[str, Callable]]] = None,
        early_stopping_metric: str = "val_loss",
        early_stopping_patience: Union[int, float] = float("inf"),
        early_stopping_mode: str = "min",
        shuffle: bool = True,
        log_path: Optional[str] = None,
        param_path: Optional[str] = None,
        verbose: bool = False,
    ):
        """
        Fit the model to the training data

        Parameters
        ----------
        X_num : torch.Tensor, numpy.ndarray, or None
        X_cat : torch.Tensor, numpy.ndarray, or None
        y : torch.Tensor or numpy.ndarray

        optimizer : PyTorch Optimizer class, optional
            uninitialized subclass of Optimizer; default is `torch.optim.Adam`
        opt_kwargs : dict or None, optional
            dict of keyword arguments to initialize optimizer with;
            default is None
        scheduler : PyTorch scheduler class, optional
            example: `torch.optim.lr_scheduler.ReduceLROnPlateau`
            default is None
        sch_kwargs : dict or None, optional
            dict of keyword arguments to initialize scheduler with;
            default is None
        val_sets : list of tuples, or None; optional
            each tuple should be (X_num, X_cat, y) validation data;
            default is None
        num_epochs : int, optional
            default is 5
        batch_size : int, optional
            default is 128
        warm_start : boolean, optional
            whether to re-create the model before fitting (warm_start == False),
            or refine the training (warm_start == True); default is False
        extra_metrics : list of (str, callable) tuples or None, optional
            default is None
        early_stopping_metric : str, optional
            should be "val_loss" or one of the passed `extra_metrics`;
            default is "val_loss"
        early_stopping_patience : int, float; optional
            default is float("inf") (no early stopping)
        early_stopping_mode : {"min", "max"}, optional
            use "min" if smaller values are better; default is "min"
        shuffle : boolean, optional
            default is True
        log_path : str or None, optional
            filename to save output to; default is None
        param_path : str or None, optional
            specify this to have the best parameters reloaded at end of training;
            default is None
        verbose : boolean, optional
            default is False

        """
        time_start = now()

        X_num, X_cat, y = self._fit_init(X_num, X_cat, y, warm_start)
        self._optimizer_init(optimizer, opt_kwargs, scheduler, sch_kwargs)

        train_dl = TabularDataLoader(
            task=self.task,
            X_num=X_num,
            X_cat=X_cat,
            y=y,
            batch_size=batch_size,
            shuffle=shuffle,
            device=self._device,
        )

        if val_sets is not None:
            valid_dl = [
                TabularDataLoader(
                    self.task,
                    *self._convert_x(*val_set),
                    y=self._convert_y(val_set[-1]),
                    batch_size=batch_size,
                    shuffle=False,
                    device=self._device,
                )
                for val_set in val_sets
            ]
        else:
            valid_dl = None

        train_info = train(
            self._model,
            train_data=train_dl,
            val_data=valid_dl,
            num_epochs=num_epochs,
            max_grad_norm=float("inf"),
            extra_metrics=extra_metrics,
            early_stopping_metric=early_stopping_metric,
            early_stopping_patience=early_stopping_patience,
            early_stopping_mode=early_stopping_mode,
            param_path=param_path,
            verbose=verbose,
        )

        if warm_start:
            self.train_info.extend(train_info)
        else:
            self.train_info = train_info

        if log_path:
            info = {
                "init_parameters": {
                    key: _param_json(val) for key, val in self.init_parameters.items()
                },
                "fit_parameters": {
                    "optimizer": str(optimizer.__name__),
                    "opt_kwargs": opt_kwargs,
                    "scheduler": str(scheduler.__name__) if scheduler is not None else None,
                    "sch_kwargs": sch_kwargs,
                    "num_epochs": num_epochs,
                    "batch_size": batch_size,
                    "extra_metrics": [x[0] for x in extra_metrics] if extra_metrics else None,
                    "early_stopping_metric": early_stopping_metric,
                    "early_stopping_patience": early_stopping_patience,
                    "early_stopping_mode": early_stopping_mode,
                    "shuffle": shuffle,
                },
                "num_parameters": self.num_parameters(),
                "time_start": time_start,
                "train_info": self.train_info,
                "time_end": now(),
            }
            _log(info, log_path)


class BaseClassifier(BaseEstimator):
    """
    Base class for Scikit-learn style classification classes in this package

    """

    def __init__(
        self,
        embedding_num: Optional[Union[str, EmbeddingBase]] = "auto",
        embedding_cat: Optional[Union[str, EmbeddingBase]] = "auto",
        loss_fn: Union[str, Callable] = "auto",
        seed: Union[int, None] = None,
        device: Union[str, torch.device] = "cpu",
        **model_kwargs,
    ):
        super().__init__(
            embedding_num=embedding_num,
            embedding_cat=embedding_cat,
            seed=seed,
            device=device,
            model_kwargs=model_kwargs,
        )
        self.task = "classification"
        self.loss_fn = nn.CrossEntropyLoss() if loss_fn == "auto" else loss_fn
        self.classes = {}

    def _create_model(self):
        self._model = self._model_class(
            task="classification",
            output_size=len(self.classes),
            embedding_num=self.embedding_num,
            embedding_cat=self.embedding_cat,
            loss_fn=self.loss_fn,
            device=self._device,
            **self.model_kwargs
        )

    def _convert_y(self, y) -> Tensor:
        if len(y.shape) == 1:
            y = y.reshape((-1, 1))
        y = torch.tensor([self.classes[yval[0].item()] for yval in y])
        return y

    def _fit_init(self, X_num, X_cat, y, warm_start=False):
        if self._model is None or not warm_start:
            self.classes = {old : new for new, old in enumerate(np.unique(y))}
        X_num, X_cat, y = self._convert_xy(X_num, X_cat, y)
        if self._model is None or not warm_start:
            self._create_embeddings(X_num, X_cat)
            self._create_model()
        return X_num, X_cat, y

    def predict(self, X_num, X_cat):
        """
        Calculate class predictions

        Parameters
        ----------
        X_num : torch.Tensor, numpy.ndarray, or None
        X_cat : torch.Tensor, numpy.ndarray, or None

        Return
        ------
        torch.Tensor

        """
        if not self._model:
            raise RuntimeError("you need to fit the model first")
        X_num, X_cat = self._convert_x(X_num, X_cat)
        self._model.eval()
        with torch.no_grad():
            raw = self._model(X_num, X_cat)
            preds = torch.argmax(raw, dim=1)
        return preds

    def predict_proba(self, X_num, X_cat):
        """
        Calculate class "probabilities"

        Parameters
        ----------
        X_num : torch.Tensor, numpy.ndarray, or None
        X_cat : torch.Tensor, numpy.ndarray, or None

        Return
        ------
        torch.Tensor

        """
        if not self._model:
            raise RuntimeError("you need to fit the model first")
        X_num, X_cat = self._convert_x(X_num, X_cat)
        self._model.eval()
        with torch.no_grad():
            raw = self._model(X_num, X_cat)
            proba = softmax(raw, dim=1)
        return proba


class BaseRegressor(BaseEstimator):
    """
    Base class for Scikit-learn style regression classes in this package

    """

    def __init__(
        self,
        embedding_num: Optional[Union[str, EmbeddingBase]] = "auto",
        embedding_cat: Optional[Union[str, EmbeddingBase]] = "auto",
        loss_fn: Union[str, Callable] = "auto",
        seed: Union[int, None] = None,
        device: Union[str, torch.device] = "cpu",
        **model_kwargs,
    ):
        super().__init__(
            embedding_num=embedding_num,
            embedding_cat=embedding_cat,
            seed=seed,
            device=device,
            model_kwargs=model_kwargs,
        )
        self.task = "regression"
        self.loss_fn = nn.MSELoss() if loss_fn == "auto" else loss_fn
        self.num_targets = 0

    def _create_model(self):
        self._model = self._model_class(
            task="regression",
            output_size=self.num_targets,
            embedding_num=self.embedding_num,
            embedding_cat=self.embedding_cat,
            loss_fn=self.loss_fn,
            device=self._device,
            **self.model_kwargs,
        )

    def _convert_y(self, y) -> Tensor:
        if isinstance(y, np.ndarray):
            y = torch.tensor(y.astype("float32"))

        if len(y.shape) == 1:
            y = y.reshape((-1, 1))

        return y

    def _fit_init(self, X_num, X_cat, y, warm_start=False):
        if self._model is None or not warm_start:
            self.num_targets = y.shape[1] if len(y.shape) > 1 else 1
        X_num, X_cat, y = self._convert_xy(X_num, X_cat, y)
        if self._model is None or not warm_start:
            self._create_embeddings(X_num, X_cat)
            self._create_model()
        return X_num, X_cat, y

    def predict(self, X_num, X_cat):
        """
        Calculate class predictions

        Parameters
        ----------
        X_num : torch.Tensor, numpy.ndarray, or None
        X_cat : torch.Tensor, numpy.ndarray, or None

        Return
        ------
        torch.Tensor

        """
        if not self._model:
            raise RuntimeError("you need to fit the model first")
        X_num, X_cat = self._convert_x(X_num, X_cat)
        self._model.eval()
        with torch.no_grad():
            preds = self._model(X_num, X_cat)
        return preds

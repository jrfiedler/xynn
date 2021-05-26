"""
Base classes for modules (PyTorch models)

"""

from abc import ABCMeta, abstractmethod
from typing import Union, List, Tuple, Callable, Dict, Optional, Type, Iterable, Any

import numpy as np
import torch
from torch import Tensor
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau
try:
    import pytorch_lightning as pl
except ImportError:
    HAS_PL = False
else:
    HAS_PL = True

from ..embedding import EmbeddingBase
from ..embedding.ragged import RaggedBase


MODULE_INIT_DOC = """
Parameters
----------
task : {{"regression", "classification"}}
output_size : int
    number of final output values; i.e., number of targets for
    regression or number of classes for classification
embedding_num : EmbeddingBase or None
    initialized and fit embedding for numeric fields
embedding_cat : EmbeddingBase or None
    initialized and fit embedding for categorical fields
{}
mlp_hidden_sizes : int or iterable of int, optional
    sizes for the linear transformations between the MLP input and
    the output size needed based on the target; default is (512, 256, 128, 64)
mlp_activation : subclass of torch.nn.Module (uninitialized), optional
    default is nn.LeakyReLU
mlp_use_bn : boolean, optional
    whether to use batch normalization between MLP linear layers;
    default is True
mlp_bn_momentum : float, optional
    only used if `mlp_use_bn` is True; default is 0.01
mlp_dropout : float, optional
    whether and how much dropout to use between MLP linear layers;
    `0.0 <= mlp_dropout < 1.0`; default is 0.0
mlp_leaky_gate : boolean, optional
    whether to include a "leaky gate" layer before the MLP layers;
    default is True
mlp_use_skip : boolean, optional
    use a side path in the MLP containing just the optional leaky gate
    plus single linear layer; default is True
loss_fn : "auto" or PyTorch loss function, optional
    default is "auto"
device : string or torch.device, optional
    default is "cpu"

"""


BaseClass = pl.LightningModule if HAS_PL else nn.Module


class BaseNN(BaseClass, metaclass=ABCMeta):
    """
    Base class for neural network models

    """

    @abstractmethod
    def __init__(
        self,
        task: str,
        embedding_num: Optional[EmbeddingBase],
        embedding_cat: Optional[EmbeddingBase],
        loss_fn: Union[str, Callable],
        device: Union[str, torch.device] = "cpu",
    ):
        """
        Parameters
        ----------
        task : {"regression", "classification"}
        embedding_num : EmbeddingBase or None
            initialized and fit embedding for numeric fields
        embedding_cat : EmbeddingBase or None
            initialized and fit embedding for categorical fields
        loss_fn : "auto" or PyTorch loss function, optional
            default is "auto"
        device : string or torch.device, optional
            default is "cpu"

        """
        super().__init__()
        if task not in {"regression", "classification"}:
            raise ValueError(
                f"task {task} not recognized; should be 'regression' or 'classification'"
            )

        self.task = task
        self.num_epochs = 0

        if loss_fn != "auto":
            self.loss_fn = loss_fn
        elif task == "regression":
            self.loss_fn = nn.MSELoss()
        else:
            self.loss_fn = nn.CrossEntropyLoss()

        self.embedding_num = embedding_num
        self.embedding_cat = embedding_cat
        self.optimizer: Optional[Callable] = None
        self.optimizer_info: Dict[str, Any] = {}
        self.scheduler: Dict[str, Any] = {}
        self._device = device

    @abstractmethod
    def mlp_weight_sum(self) -> Tuple[float, float]:
        return 0.0, 0.0

    def embedding_sum(self) -> Tuple[float, float]:
        """
        Sum of absolute value and square of embedding values

        Return
        ------
        e1_sum : sum of absolute value of embedding values
        e2_sum : sum of squared embedding values

        """
        e1_sum = 0.0
        e2_sum = 0.0

        if hasattr(self, "embedding_num") and self.embedding_num is not None:
            e1_sum_num, e2_sum_num = self.embedding_num.weight_sum()
            e1_sum += e1_sum_num
            e2_sum += e2_sum_num

        if hasattr(self, "embedding_cat") and self.embedding_cat is not None:
            e1_sum_cat, e2_sum_cat = self.embedding_cat.weight_sum()
            e1_sum += e1_sum_cat
            e2_sum += e2_sum_cat

        return e1_sum, e2_sum

    def num_parameters(self) -> int:
        """
        Number of trainable parameters in the model

        Return
        ------
        int number of trainable parameters

        """
        return sum(param.numel() for param in self.parameters() if param.requires_grad)

    def embed(
        self,
        X_num: Tensor,
        X_cat: Tensor,
        num_dim: int = 3,
        concat: bool = True,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Embed the numeric and categorical input fields.

        Parameters
        ----------
        X_num : torch.Tensor or numpy.ndarray or None
        X_cat : torch.Tensor or numpy.ndarray or None
        num_dim : 2 or 3, optional
            default is 3
        concat : bool, optional
            whether to concatenate outputs into a single Tensor;
            if True, concatenation is on dim 1; default is True

        Return
        ------
        torch.Tensor if concat else (torch.Tensor, torch.Tensor)

        """
        if X_num is None and X_cat is None:
            raise ValueError("X_num and X_cat cannot both be None")

        if num_dim not in (2, 3):
            raise ValueError(f"num_dim should be 2 or 3, got {num_dim}")

        if num_dim == 3 and (
            isinstance(self.embedding_num, RaggedBase)
            or isinstance(self.embedding_cat, RaggedBase)
        ):
            raise ValueError("cannot use num_dim=3 with ragged embeddings")

        # handle X_num
        if X_num is not None and X_num.shape[1] and self.embedding_num:
            X_num_emb = self.embedding_num(X_num)
        elif (X_num is not None and X_num.shape[1]) or not self.embedding_cat:
            if num_dim == 3:
                X_num_emb = X_num.reshape((X_num.shape[0], X_num.shape[1], 1))
            else:
                X_num_emb = X_num
        else:  # (X_num is None or not X_num.shape[1]) and self.embedding_cat
            X_num_emb = torch.empty(
                (X_cat.shape[0], 0, self.embedding_cat.embedding_size),
                device=self._device,
            )

        # handle X_cat
        if X_cat is not None and X_cat.shape[1] and self.embedding_cat:
            X_cat_emb = self.embedding_cat(X_cat)
        else:
            embed_dim = self.embedding_num.embedding_size if self.embedding_num else 1
            X_cat_emb = torch.empty((X_num.shape[0], 0, embed_dim), device=self._device)

        # reshape, if necessary
        if num_dim == 2:
            X_num_emb = X_num_emb.reshape((X_num_emb.shape[0], -1))
            X_cat_emb = X_cat_emb.reshape((X_cat_emb.shape[0], -1))

        if concat:
            return torch.cat([X_num_emb, X_cat_emb], dim=1)

        return X_num_emb, X_cat_emb

    def training_step(self, train_batch: List[Tensor], batch_idx: int) -> Dict:
        """
        Create predictions on batch and compute loss

        Used by PyTorch Lightning and the Scikit-learn-style classes

        Parameters
        ----------
        train_batch : torch.Tensor
        batch_idx : int

        Returns
        -------
        dict mapping "train_step_loss" to torch.Tensor loss value

        """
        X_num, X_cat, y = train_batch
        y_hat = self.forward(X_num, X_cat)
        loss = self.loss_fn(y_hat, y)
        return {"loss": loss}

    def training_epoch_end(self, outputs: List[Dict]):
        """
        Computes and logs average train loss

        Used by PyTorch Lightning and the Scikit-learn-style classes

        Parameters
        ----------
        outputs : list of dicts
            outputs after all of the training steps

        Side effect
        -----------
        logs average loss as "train_loss"

        """
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("train_loss", avg_loss)

    def validation_step(self, val_batch: List[Tensor], batch_idx: int) -> Tuple[Tensor, Tensor]:
        """
        Calculate validation loss

        Used by PyTorch Lightning and the Scikit-learn-style classes

        Parameters
        ----------
        val_batch : torch.Tensor
        batch_idx : int

        Returns
        -------
        (y_pred, y_true) pair of tensors

        """
        X_num, X_cat, y = val_batch
        y_hat = self.forward(X_num, X_cat)
        return (y_hat, y)

    def validation_epoch_end(self, validation_step_outputs: List[Tuple[Tensor, Tensor]]):
        """
        Computes average validation loss

        Used by PyTorch Lightning and the Scikit-learn-style classes

        Parameters
        ----------
        validation_step_outputs : list of (y_pred, y_true) tensors
            outputs after all of the validation steps

        Side effect
        -----------
        logs average loss as "val_loss"

        """
        preds = torch.stack([y_hat for y_hat, _ in validation_step_outputs])
        ytrue = torch.stack([y for _, y in validation_step_outputs])
        val_loss = self.loss_fn(preds, ytrue)
        self.log("val_loss", val_loss)

    def custom_val_epoch_end(
        self,
        validation_step_outputs: List[Tuple[Tensor, Tensor]],
        extra_metrics: Iterable[Tuple[str, Callable]],
    ) -> Dict:
        """
        Calculate validation loss and other metrics if provided

        Parameters
        ----------
        validation_step_outputs : list of (y_pred, y_true) tensors
            outputs after all of the validation steps
        extra_metrics: list of (str, callable)
            tuples of str name and callable metric

        Returns
        -------
        dict
        - maps "val_step_loss" to torch.Tensor loss value
        - maps each name in `extra_metrics` to the metric value

        """
        preds = torch.cat([y_hat for y_hat, _ in validation_step_outputs], dim=0)
        ytrue = torch.cat([y for _, y in validation_step_outputs], dim=0)
        loss = self.loss_fn(preds, ytrue)
        info = {"val_loss": loss.item()}
        for name, fn in extra_metrics:
            loss = fn(preds, ytrue)
            info[name] = loss.item() if isinstance(loss, (np.ndarray, Tensor)) else loss
        return info

    def test_step(self, test_batch: List[Tensor], batch_idx: int) -> Dict:
        """
        Calculate test loss

        Used by PyTorch Lightning

        Parameters
        ----------
        test_batch : list of torch.Tensor
        batch_idx : int

        Returns
        -------
        dict
        - maps "test_step_loss" to torch.Tensor loss value

        """
        X_num, X_cat, y = test_batch
        y_hat = self.forward(X_num, X_cat)
        loss = self.loss_fn(y_hat, y)
        info = {"test_step_loss": loss}
        return info

    def test_epoch_end(self, outputs: List[Dict]):
        """
        Computes average test loss

        Used by PyTorch Lightning

        Parameters
        ----------
        outputs : list of dicts
            outputs after all of the test steps

        Side effect
        -----------
        logs average loss as "test_loss"

        """
        avg_loss = torch.stack([x["test_step_loss"] for x in outputs]).mean()
        self.log("test_loss", avg_loss)

    def set_optimizer(
        self,
        optimizer: Type[Optimizer] = torch.optim.Adam,
        opt_kwargs: Optional[Dict] = None,
        scheduler: Optional[Type[_LRScheduler]] = None,
        sch_kwargs: Optional[Dict] = None,
        sch_options: Optional[Dict] = None,
    ):
        """
        Set the models optimizer and, optionally, the learning rate schedule

        Parameters
        ----------
        optimizer : PyTorch Optimizer class, optional
            uninitialized subclass of Optimizer; default is torch.optim.Adam
        opt_kwargs : dict or None, optional
            dict of keyword arguments to initialize optimizer with;
            default is None
        scheduler : PyTorch scheduler class, optional
            default is None
        sch_kwargs : dict or None, optional
            dict of keyword arguments to initialize scheduler with;
            default is None
        sch_options : dict or None, optional
            options for PyTorch Lightning's call to `configure_optimizers`;
            ignore if not using PyTorch Lightning or no options are needed;
            with PyTorch Lightning, `ReduceLROnPlateau` requires "monitor";
            default is None

        """
        if sch_options is None:
            sch_options = {}
        if scheduler is ReduceLROnPlateau and "monitor" not in sch_options:
            sch_options["monitor"] = "val_loss"

        self.optimizer_info = {
            "optimizer": optimizer,
            "opt_kwargs": opt_kwargs if opt_kwargs is not None else {},
            "scheduler": scheduler,
            "sch_kwargs": sch_kwargs if sch_kwargs is not None else {},
            "sch_options": sch_options,
        }

    def configure_optimizers(
        self
    ) -> Union[Optimizer, Tuple[List[Optimizer], List[_LRScheduler]]]:
        """
        Initializes the optimizer and learning rate scheduler

        The optimizer and learning rate info needs to first be set with
        the `set_optimizer` method

        Used by PyTorch Lightning and the Scikit-learn-style classes

        Returns
        -------
        if no scheduler is being used
            initialized optimizer
        else
            tuple with
                list containing just the initialized optimizer
                dict containing scheduler information

        """
        if not self.optimizer_info:
            raise RuntimeError(
                "The optimizer and learning rate info needs to first be set "
                "with the `set_optimizer` method"
            )

        optimizer = self.optimizer_info["optimizer"]
        opt_kwargs = self.optimizer_info["opt_kwargs"]
        self.optimizer = optimizer(self.parameters(), **opt_kwargs)

        if self.optimizer_info["scheduler"] is None:
            return self.optimizer

        scheduler = self.optimizer_info["scheduler"]
        sch_kwargs = self.optimizer_info["sch_kwargs"]
        sch_options = self.optimizer_info["sch_options"]
        self.scheduler = {"scheduler": scheduler(self.optimizer, **sch_kwargs)}
        self.scheduler.update(sch_options)

        return [self.optimizer], [self.scheduler]

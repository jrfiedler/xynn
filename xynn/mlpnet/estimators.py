"""
Scikit-learn style classes for the MLP model

"""

from typing import Union, Callable, Optional, Type, List, Tuple

import torch
from torch import nn

from ..base_classes.estimators import BaseClassifier, BaseRegressor, ESTIMATOR_INIT_DOC
from ..embedding import EmbeddingBase
from .modules import MLPNet


INIT_DOC = ESTIMATOR_INIT_DOC.format("")


class MLPClassifier(BaseClassifier):
    """
    Scikit-learn style classification model for the MLP model

    """

    diagram = MLPNet.diagram

    def __init__(
        self,
        embedding_num: Optional[Union[str, EmbeddingBase]] = "auto",
        embedding_cat: Optional[Union[str, EmbeddingBase]] = "auto",
        embedding_l1_reg: float = 0.0,
        embedding_l2_reg: float = 0.0,
        mlp_hidden_sizes: Union[int, Tuple[int, ...], List[int]] = (512, 256, 128, 64),
        mlp_activation: Type[nn.Module] = nn.LeakyReLU,
        mlp_use_bn: bool = True,
        mlp_bn_momentum: float = 0.1,
        mlp_dropout: float = 0.0,
        mlp_l1_reg: float = 0.0,
        mlp_l2_reg: float = 0.0,
        mlp_leaky_gate: bool = True,
        mlp_use_skip: bool = True,
        weighted_sum: bool = True,
        loss_fn: Union[str, Callable] = "auto",
        seed: Union[int, None] = None,
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__(
            embedding_num=embedding_num,
            embedding_cat=embedding_cat,
            embedding_l1_reg=embedding_l1_reg,
            embedding_l2_reg=embedding_l2_reg,
            mlp_hidden_sizes=mlp_hidden_sizes,
            mlp_activation=mlp_activation,
            mlp_use_bn=mlp_use_bn,
            mlp_bn_momentum=mlp_bn_momentum,
            mlp_dropout=mlp_dropout,
            mlp_l1_reg=mlp_l1_reg,
            mlp_l2_reg=mlp_l2_reg,
            mlp_leaky_gate=mlp_leaky_gate,
            mlp_use_skip=mlp_use_skip,
            weighted_sum=weighted_sum,
            loss_fn=loss_fn,
            seed=seed,
            device=device,
        )
        self._model_class = MLPNet
        self._require_numeric_embedding = False

    __init__.__doc__ = INIT_DOC


class MLPRegressor(BaseRegressor):
    """
    Scikit-learn style regression model for the MLP model

    """

    diagram = MLPNet.diagram

    def __init__(
        self,
        embedding_num: Optional[Union[str, EmbeddingBase]] = "auto",
        embedding_cat: Optional[Union[str, EmbeddingBase]] = "auto",
        embedding_l1_reg: float = 0.0,
        embedding_l2_reg: float = 0.0,
        mlp_hidden_sizes: Union[int, Tuple[int, ...], List[int]] = (512, 256, 128, 64),
        mlp_activation: Type[nn.Module] = nn.LeakyReLU,
        mlp_use_bn: bool = True,
        mlp_bn_momentum: float = 0.1,
        mlp_dropout: float = 0.0,
        mlp_l1_reg: float = 0.0,
        mlp_l2_reg: float = 0.0,
        mlp_leaky_gate: bool = True,
        mlp_use_skip: bool = True,
        weighted_sum: bool = True,
        loss_fn: Union[str, Callable] = "auto",
        seed: Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__(
            embedding_num=embedding_num,
            embedding_cat=embedding_cat,
            embedding_l1_reg=embedding_l1_reg,
            embedding_l2_reg=embedding_l2_reg,
            mlp_hidden_sizes=mlp_hidden_sizes,
            mlp_activation=mlp_activation,
            mlp_use_bn=mlp_use_bn,
            mlp_bn_momentum=mlp_bn_momentum,
            mlp_dropout=mlp_dropout,
            mlp_l1_reg=mlp_l1_reg,
            mlp_l2_reg=mlp_l2_reg,
            mlp_leaky_gate=mlp_leaky_gate,
            mlp_use_skip=mlp_use_skip,
            weighted_sum=weighted_sum,
            loss_fn=loss_fn,
            seed=seed,
            device=device,
        )
        self._model_class = MLPNet
        self._require_numeric_embedding = False

    __init__.__doc__ = INIT_DOC

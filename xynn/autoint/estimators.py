"""
Scikit-learn style classes for the AutoInt model

"""

import textwrap
from typing import Type, Union, Callable, Tuple, List, Optional

import torch
from torch import nn

from .modules import AutoInt
from ..base_classes.estimators import BaseClassifier, BaseRegressor, ESTIMATOR_INIT_DOC


INIT_DOC = ESTIMATOR_INIT_DOC.format(
    textwrap.dedent(
        """\
        attn_embedding_size : int, optional
            default is 8
        attn_num_layer : int, optional
            default is 3
        attn_num_head : int, optional
            default is 2
        attn_activation : subclass of torch.nn.Module or None, optional
            default is None
        attn_use_residual : bool, optional
            default is True
        attn_dropout : float, optional
            default is 0.1
        attn_normalize : bool, optional
            default is True"""
    )
)


class AutoIntClassifier(BaseClassifier):
    """
    Scikit-learn style classification model for the AutoInt model

    """

    diagram = AutoInt.diagram

    def __init__(
        self,
        embedding_size: int=10,
        embedding_alpha: int=20,
        embedding_l1_reg: float=0.0,
        embedding_l2_reg: float=0.0,
        attn_embedding_size: int = 8,
        attn_num_layer: int = 3,
        attn_num_head: int = 2,
        attn_activation: Optional[Type[nn.Module]] = None,
        attn_use_residual: bool = True,
        attn_dropout: float = 0.1,
        attn_normalize: bool = True,
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
            embedding_size=embedding_size,
            embedding_alpha=embedding_alpha,
            embedding_l1_reg=embedding_l1_reg,
            embedding_l2_reg=embedding_l2_reg,
            attn_embedding_size=attn_embedding_size,
            attn_num_layer=attn_num_layer,
            attn_num_head=attn_num_head,
            attn_activation=attn_activation,
            attn_use_residual=attn_use_residual,
            attn_dropout=attn_dropout,
            attn_normalize=attn_normalize,
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
        self._model_class = AutoInt

    __init__.__doc__ = INIT_DOC


class AutoIntRegressor(BaseRegressor):
    """
    Scikit-learn style regression model for the AutoInt model

    """

    diagram = AutoInt.diagram

    def __init__(
        self,
        embedding_size: int=10,
        embedding_alpha: int=20,
        embedding_l1_reg: float=0.0,
        embedding_l2_reg: float=0.0,
        attn_embedding_size: int = 8,
        attn_num_layer: int = 3,
        attn_num_head: int = 2,
        attn_activation: Optional[Type[nn.Module]] = None,
        attn_use_residual: bool = True,
        attn_dropout: float = 0.1,
        attn_normalize: bool = True,
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
            embedding_size=embedding_size,
            embedding_alpha=embedding_alpha,
            embedding_l1_reg=embedding_l1_reg,
            embedding_l2_reg=embedding_l2_reg,
            attn_embedding_size=attn_embedding_size,
            attn_num_layer=attn_num_layer,
            attn_num_head=attn_num_head,
            attn_activation=attn_activation,
            attn_use_residual=attn_use_residual,
            attn_dropout=attn_dropout,
            attn_normalize=attn_normalize,
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
        self._model_class = AutoInt

    __init__.__doc__ = INIT_DOC

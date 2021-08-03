"""
Scikit-learn style classes for the xDeepFM model

"""

import textwrap
from typing import Type, Union, Callable, Tuple, List, Optional

import torch
from torch import nn

from .modules import XDeepFM
from ..base_classes.estimators import BaseClassifier, BaseRegressor, ESTIMATOR_INIT_DOC
from ..embedding import EmbeddingBase


INIT_DOC = ESTIMATOR_INIT_DOC.format(
    textwrap.dedent(
        """\
        cin_layer_sizes : int, tuple of int or list of int; optional
            if `cin_full_agg` is False, all sizes except the last must be even;
            default is (128, 128)
        cin_activation : subclass of torch.nn.Module, optional
            default is nn.Identity
        cin_full_agg : bool, optional
            if True, each intermediate output is aggregated in the final CIN output;
            if False, half of each intermediate output is aggregated;
            default is False
        cin_use_bn : bool, optional
            default is True
        cin_bn_momentum: float, optional
            default is 0.1
        cin_use_residual: bool, optional
            whether to use a skip connection from CIN to output; default is True
        cin_use_mlp : bool, optional
            default is True"""
    )
)


class XDeepFMClassifier(BaseClassifier):
    """
    Scikit-learn style classification model for the xDeepFM model

    """

    diagram = XDeepFM.diagram

    def __init__(
        self,
        embedding_num: Optional[Union[str, EmbeddingBase]] = "auto",
        embedding_cat: Optional[Union[str, EmbeddingBase]] = "auto",
        embedding_l1_reg: float=0.0,
        embedding_l2_reg: float=0.0,
        cin_layer_sizes: Union[int, Tuple[int, ...], List[int]] = (128, 128),
        cin_activation: Type[nn.Module] = nn.Identity,
        cin_full_agg: bool = False,
        cin_use_bn: bool = True,
        cin_bn_momentum: float = 0.1,
        cin_use_residual: bool = True,
        cin_use_mlp: bool = True,
        mlp_hidden_sizes: Union[int, Tuple[int, ...], List[int]] = (512, 256, 128, 64),
        mlp_activation: Type[nn.Module] = nn.LeakyReLU,
        mlp_use_bn: bool = True,
        mlp_bn_momentum: float = 0.1,
        mlp_ghost_batch: Optional[int] = None,
        mlp_dropout: float = 0.0,
        mlp_l1_reg: float = 0.0,
        mlp_l2_reg: float = 0.0,
        mlp_use_skip: bool = True,
        use_leaky_gate: bool = True,
        loss_fn: Union[str, Callable] = "auto",
        seed: Union[int, None] = None,
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__(
            embedding_num=embedding_num,
            embedding_cat=embedding_cat,
            embedding_l1_reg=embedding_l1_reg,
            embedding_l2_reg=embedding_l2_reg,
            cin_layer_sizes=cin_layer_sizes,
            cin_activation=cin_activation,
            cin_full_agg=cin_full_agg,
            cin_use_bn=cin_use_bn,
            cin_bn_momentum=cin_bn_momentum,
            cin_use_residual=cin_use_residual,
            cin_use_mlp=cin_use_mlp,
            mlp_hidden_sizes=mlp_hidden_sizes,
            mlp_activation=mlp_activation,
            mlp_use_bn=mlp_use_bn,
            mlp_bn_momentum=mlp_bn_momentum,
            mlp_ghost_batch=mlp_ghost_batch,
            mlp_dropout=mlp_dropout,
            mlp_l1_reg=mlp_l1_reg,
            mlp_l2_reg=mlp_l2_reg,
            mlp_use_skip=mlp_use_skip,
            use_leaky_gate=use_leaky_gate,
            loss_fn=loss_fn,
            seed=seed,
            device=device,
        )
        self._model_class = XDeepFM

    __init__.__doc__ = INIT_DOC


class XDeepFMRegressor(BaseRegressor):
    """
    Scikit-learn style regression model for the xDeepFM model

    """

    diagram = XDeepFM.diagram

    def __init__(
        self,
        embedding_num: Optional[Union[str, EmbeddingBase]] = "auto",
        embedding_cat: Optional[Union[str, EmbeddingBase]] = "auto",
        embedding_l1_reg: float=0.0,
        embedding_l2_reg: float=0.0,
        cin_layer_sizes: Union[int, Tuple[int, ...], List[int]] = (128, 128),
        cin_activation: Type[nn.Module] = nn.Identity,
        cin_full_agg: bool = False,
        cin_use_bn: bool = True,
        cin_bn_momentum: float = 0.1,
        cin_use_residual: bool = True,
        cin_use_mlp: bool = True,
        mlp_hidden_sizes: Union[int, Tuple[int, ...], List[int]] = (512, 256, 128, 64),
        mlp_activation: Type[nn.Module] = nn.LeakyReLU,
        mlp_use_bn: bool = True,
        mlp_bn_momentum: float = 0.1,
        mlp_ghost_batch: Optional[int] = None,
        mlp_dropout: float = 0.0,
        mlp_l1_reg: float = 0.0,
        mlp_l2_reg: float = 0.0,
        mlp_use_skip: bool = True,
        use_leaky_gate: bool = True,
        loss_fn: Union[str, Callable] = "auto",
        seed: Union[int, None] = None,
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__(
            embedding_num=embedding_num,
            embedding_cat=embedding_cat,
            embedding_l1_reg=embedding_l1_reg,
            embedding_l2_reg=embedding_l2_reg,
            cin_layer_sizes=cin_layer_sizes,
            cin_activation=cin_activation,
            cin_full_agg=cin_full_agg,
            cin_use_bn=cin_use_bn,
            cin_bn_momentum=cin_bn_momentum,
            cin_use_residual=cin_use_residual,
            cin_use_mlp=cin_use_mlp,
            mlp_hidden_sizes=mlp_hidden_sizes,
            mlp_activation=mlp_activation,
            mlp_use_bn=mlp_use_bn,
            mlp_bn_momentum=mlp_bn_momentum,
            mlp_ghost_batch=mlp_ghost_batch,
            mlp_dropout=mlp_dropout,
            mlp_l1_reg=mlp_l1_reg,
            mlp_l2_reg=mlp_l2_reg,
            mlp_use_skip=mlp_use_skip,
            use_leaky_gate=use_leaky_gate,
            loss_fn=loss_fn,
            seed=seed,
            device=device,
        )
        self._model_class = XDeepFM

    __init__.__doc__ = INIT_DOC

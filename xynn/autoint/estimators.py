"""
Scikit-learn style classes for the AutoInt model

"""

import textwrap
from typing import Type, Union, Callable, Tuple, List, Optional

import torch
from torch import nn

from .modules import AutoInt
from ..base_classes.estimators import BaseClassifier, BaseRegressor, ESTIMATOR_INIT_DOC
from ..embedding import EmbeddingBase


INIT_DOC = ESTIMATOR_INIT_DOC.format(
    textwrap.dedent(
        """\
        attn_embedding_size : int, optional
            default is 8
        attn_num_layers : int, optional
            default is 3
        attn_num_head : int, optional
            default is 2
        attn_activation : subclass of torch.nn.Module or None, optional
            applied to the transformation tensors; default is None
        attn_use_residual : bool, optional
            default is True
        attn_dropout : float, optional
            amount of dropout to use on the product of queries and keys;
            default is 0.1
        attn_normalize : bool, optional
            whether to normalize each attn layer output; default is True"""
    )
)


class AutoIntClassifier(BaseClassifier):
    """
    Scikit-learn style classification model for the AutoInt model

    """

    diagram = AutoInt.diagram

    def __init__(
        self,
        embedding_num: Optional[Union[str, EmbeddingBase]] = "auto",
        embedding_cat: Optional[Union[str, EmbeddingBase]] = "auto",
        embedding_l1_reg: float=0.0,
        embedding_l2_reg: float=0.0,
        attn_embedding_size: int = 8,
        attn_num_layers: int = 3,
        attn_num_heads: int = 2,
        attn_activation: Optional[Type[nn.Module]] = None,
        attn_use_residual: bool = True,
        attn_dropout: float = 0.1,
        attn_normalize: bool = True,
        attn_use_mlp: bool = True,
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
            attn_embedding_size=attn_embedding_size,
            attn_num_layers=attn_num_layers,
            attn_num_heads=attn_num_heads,
            attn_activation=attn_activation,
            attn_use_residual=attn_use_residual,
            attn_dropout=attn_dropout,
            attn_normalize=attn_normalize,
            attn_use_mlp=attn_use_mlp,
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
            weighted_sum=weighted_sum,
            loss_fn=loss_fn,
            seed=seed,
            device=device,
        )
        self._model_class = AutoInt
        self._require_numeric_embedding = True

    __init__.__doc__ = INIT_DOC


class AutoIntRegressor(BaseRegressor):
    """
    Scikit-learn style regression model for the AutoInt model

    """

    diagram = AutoInt.diagram

    def __init__(
        self,
        embedding_num: Optional[Union[str, EmbeddingBase]] = "auto",
        embedding_cat: Optional[Union[str, EmbeddingBase]] = "auto",
        embedding_l1_reg: float=0.0,
        embedding_l2_reg: float=0.0,
        attn_embedding_size: int = 8,
        attn_num_layers: int = 3,
        attn_num_heads: int = 2,
        attn_activation: Optional[Type[nn.Module]] = None,
        attn_use_residual: bool = True,
        attn_dropout: float = 0.1,
        attn_normalize: bool = True,
        attn_use_mlp: bool = True,
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
            attn_embedding_size=attn_embedding_size,
            attn_num_layers=attn_num_layers,
            attn_num_heads=attn_num_heads,
            attn_activation=attn_activation,
            attn_use_residual=attn_use_residual,
            attn_dropout=attn_dropout,
            attn_normalize=attn_normalize,
            attn_use_mlp=attn_use_mlp,
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
            weighted_sum=weighted_sum,
            loss_fn=loss_fn,
            seed=seed,
            device=device,
        )
        self._model_class = AutoInt
        self._require_numeric_embedding = True

    __init__.__doc__ = INIT_DOC

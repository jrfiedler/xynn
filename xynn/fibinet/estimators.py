"""
Scikit-learn style classes for the FiBiNet model

"""

import textwrap
from typing import Type, Union, Callable, List, Tuple, Optional

import torch
from torch import nn

from .modules import FiBiNet
from ..base_classes.estimators import BaseClassifier, BaseRegressor, ESTIMATOR_INIT_DOC
from ..embedding import EmbeddingBase


INIT_DOC = ESTIMATOR_INIT_DOC.format(
    textwrap.dedent(
        """\
        fibi_reduction_ratio : int, optional
            used in the SENET layer; default is 3
        fibi_activation : subclass of torch.nn.Module, optional
            activation used in the SENET layer; default is nn.LeakyReLU
        fibi_senet_product : str, optional
            options:
                - "field-all"
                - "field-each"
                - "field-interaction"
                - "sym-all"
                - "sym-each"
                - "sym-interaction"
                - "hadamard"
            "field" :
                the original asymmetric bilinear products, with products like
                `linear(field_1) * field_2`
                where `*` is elementwise multiplication
            "sym" :
                symmetric versions of the "field" products
                `(linear(field_1) * field_2 + field_1 * linear(field_2)) / 2`
            "all" : a single product matrix is shared across all pairs of fields
            "each" : each field has an associated product matrix
            "interaction" : each pair of fields has an associated product matrix
            "hadamard" : elementwise multiplication of each pair of fields
            default is \"sym-interaction\"
        fibi_embed_product : str, optional
            options:
                - "shared"
                - "field-all"
                - "field-each"
                - "field-interaction"
                - "sym-all"
                - "sym-each"
                - "sym-interaction"
                - "hadamard"
            "shared" :
                use the same product layer (not just the same option) as the SENET
                product (previous parameter)
            for descriptions of other options, see notes under `fibi_senet_product`;
            default is \"sym-interaction\"
        fibi_senet_skip: bool, optional
            whether SENET output should also be used in both the MLP and Bilinear
            layer (True), or just the Bilinear layer (False); see FiBiNet.diagram();
            default is True"""
    )
)


class FiBiNetClassifier(BaseClassifier):
    """
    Scikit-learn style classification model for the FiBiNet model

    """

    diagram = FiBiNet.diagram

    def __init__(
        self,
        embedding_num: Optional[Union[str, EmbeddingBase]] = "auto",
        embedding_cat: Optional[Union[str, EmbeddingBase]] = "auto",
        embedding_l1_reg: float = 0.0,
        embedding_l2_reg: float = 0.0,
        fibi_reduction_ratio: int = 3,
        fibi_activation: Type[nn.Module] = nn.LeakyReLU,
        fibi_senet_product: str = "sym-interaction",
        fibi_embed_product: str = "sym-interaction",
        fibi_senet_skip: bool = True,
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
            fibi_reduction_ratio=fibi_reduction_ratio,
            fibi_activation=fibi_activation,
            fibi_senet_product=fibi_senet_product,
            fibi_embed_product=fibi_embed_product,
            fibi_senet_skip=fibi_senet_skip,
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
        self._model_class = FiBiNet

    __init__.__doc__ = INIT_DOC

    def _create_model(self):
        model_kwargs = {
            k: v for k, v in self.model_kwargs.items() if k != "embed_numeric_fields"
        }
        self._model = self._model_class(
            task="classification",
            output_size=len(self.classes),
            embedding_num=self.embedding_num,
            embedding_cat=self.embedding_cat,
            num_numeric_fields=self._num_numeric_fields,
            loss_fn=self.loss_fn,
            device=self._device,
            **model_kwargs,
        )


class FiBiNetRegressor(BaseRegressor):
    """
    Scikit-learn style regression model for the FiBiNet model

    """

    diagram = FiBiNet.diagram

    def __init__(
        self,
        embedding_num: Optional[Union[str, EmbeddingBase]] = "auto",
        embedding_cat: Optional[Union[str, EmbeddingBase]] = "auto",
        embedding_l1_reg: float = 0.0,
        embedding_l2_reg: float = 0.0,
        fibi_reduction_ratio: int = 3,
        fibi_activation: Type[nn.Module] = nn.LeakyReLU,
        fibi_senet_product: str = "sym-interaction",
        fibi_embed_product: str = "sym-interaction",
        fibi_senet_skip: bool = True,
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
            fibi_reduction_ratio=fibi_reduction_ratio,
            fibi_activation=fibi_activation,
            fibi_senet_product=fibi_senet_product,
            fibi_embed_product=fibi_embed_product,
            fibi_senet_skip=fibi_senet_skip,
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
        self._model_class = FiBiNet

    __init__.__doc__ = INIT_DOC

    def _create_model(self):
        model_kwargs = {
            k: v for k, v in self.model_kwargs.items() if k != "embed_numeric_fields"
        }
        self._model = self._model_class(
            task="regression",
            output_size=self.num_targets,
            embedding_num=self.embedding_num,
            embedding_cat=self.embedding_cat,
            num_numeric_fields=self._num_numeric_fields,
            loss_fn=self.loss_fn,
            device=self._device,
            **model_kwargs,
        )

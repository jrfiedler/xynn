"""
PyTorch module for the MLP model

"""

import textwrap
from typing import Union, Tuple, Callable, Optional, Type, List

import torch
from torch import Tensor
from torch import nn

from ..base_classes.modules import BaseNN, MODULE_INIT_DOC
from ..mlp import MLP
from ..embedding import check_embeddings
from ..embedding.common import EmbeddingBase


INIT_DOC = MODULE_INIT_DOC.format(
    textwrap.dedent(
        """\
        num_numeric_fields : int or "auto", optional
            an integer must be specified when embedding_num is None;
            default is \"auto\""""
    )
)


class MLPNet(BaseNN):
    """ A model consisting of just an MLP """

    def __init__(
        self,
        task: str,
        output_size: int,
        embedding_num: Optional[EmbeddingBase],
        embedding_cat: Optional[EmbeddingBase],
        embedding_l1_reg: float = 0.0,
        embedding_l2_reg: float = 0.0,
        num_numeric_fields: Union[int, str] = "auto",
        mlp_hidden_sizes: Union[int, Tuple[int, ...], List[int]] = (512, 256, 128, 64),
        mlp_activation: Type[nn.Module] = nn.LeakyReLU,
        mlp_use_bn: bool = True,
        mlp_bn_momentum: float = 0.1,
        mlp_ghost_batch: Optional[int] = None,
        mlp_dropout: float = 0.0,
        mlp_use_skip: bool = True,
        mlp_l1_reg: float = 0.0,
        mlp_l2_reg: float = 0.0,
        use_leaky_gate: bool = True,
        weighted_sum: bool = True,
        loss_fn: Union[str, Callable] = "auto",
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__(
            task,
            embedding_num,
            embedding_cat,
            embedding_l1_reg,
            embedding_l2_reg,
            mlp_l1_reg,
            mlp_l2_reg,
            loss_fn,
            device,
        )

        embed_info = check_embeddings(embedding_num, embedding_cat)

        if embedding_num is not None:
            input_size = embed_info.output_size
        elif not isinstance(num_numeric_fields, int):
            raise TypeError(
                "when embedding_num is None, num_numeric_fields must be an integer"
            )
        else:
            input_size = embed_info.output_size + num_numeric_fields

        self.mlp = MLP(
            task,
            input_size=input_size,
            hidden_sizes=mlp_hidden_sizes,
            output_size=output_size,
            activation=mlp_activation,
            dropout=mlp_dropout,
            dropout_first=True,
            use_bn=mlp_use_bn,
            bn_momentum=mlp_bn_momentum,
            ghost_batch=mlp_ghost_batch,
            leaky_gate=use_leaky_gate,
            use_skip=mlp_use_skip,
            weighted_sum=weighted_sum,
            device=device,
        )

        self.mix = self.mlp.mix
        #self.to(device)

    __init__.__doc__ = INIT_DOC

    @staticmethod
    def diagram():
        """ Print a text diagram of this model """
        gram = """\

        if mlp_use_skip=True (default)
        ------------------------------
        X_num ─ Num. embedding? ┐ ┌─── MLP ──┐
                                ├─┤          w+ ── output
        X_cat ─ Cat. embedding ─┘ └─ Linear ─┘

        if mlp_use_skip=False
        ---------------------
        X_num ─ Num. embedding? ┐
                                ├─── MLP ── output
        X_cat ─ Cat. embedding ─┘

        splits are copies and joins are concatenations;
        'w+' is weighted element-wise addition;
        the numeric embedding is optional
        """
        print("\n" + textwrap.dedent(gram))

    def mlp_weight_sum(self) -> Tuple[Tensor, Tensor]:
        """
        Sum of absolute value and square of weights in MLP layers

        Return
        ------
        w1 : sum of absolute value of MLP weights
        w2 : sum of squared MLP weights

        """
        return  self.mlp.weight_sum()

    def forward(self, X_num: Tensor, X_cat: Tensor) -> Tensor:
        """
        Transform the input tensor

        Parameters
        ----------
        X_num : torch.Tensor
            numeric fields
        X_cat : torch.Tensor
            categorical fields

        Return
        ------
        torch.Tensor

        """
        embedded = self.embed(X_num, X_cat, num_dim=2)
        return self.mlp(embedded)

"""
PyTorch modules for the PNN and PNNPlus models

"""
# product implementations adapted from
# https://github.com/JianzhouZhan/Awesome-RecSystem-Models/blob/master/Model/PNN_PyTorch.py
# Paper: https://arxiv.org/pdf/1611.00144.pdf

import textwrap
from typing import Union, Tuple, Iterable, Callable, Optional, Type, List

import torch
from torch import Tensor
from torch import nn

from ..base_classes.modules import BaseNN, MODULE_INIT_DOC
from ..mlp import MLP
from ..embedding import check_uniform_embeddings
from ..embedding.common import EmbeddingBase


INIT_DOC = MODULE_INIT_DOC.format("")


class MLPNet(BaseNN):
    """ A model consisting of just an MLP """

    def __init__(
        self,
        task: str,
        output_size: int,
        embedding_num: Optional[EmbeddingBase],
        embedding_cat: Optional[EmbeddingBase],
        mlp_hidden_sizes: Union[int, Iterable[int]] = (512, 256, 128, 64),
        mlp_activation: Type[nn.Module] = nn.LeakyReLU,
        mlp_use_bn: bool = True,
        mlp_bn_momentum: float = 0.1,
        mlp_dropout: float = 0.0,
        mlp_leaky_gate: bool = True,
        mlp_use_skip: bool = True,
        weighted_sum: bool = True,
        loss_fn: Union[str, Callable] = "auto",
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__(task, embedding_num, embedding_cat, loss_fn, device)

        embed_info = check_uniform_embeddings(embedding_num, embedding_cat)

        self.mlp = MLP(
            task,
            input_size=embed_info.output_size,
            hidden_sizes=mlp_hidden_sizes,
            output_size=output_size,
            activation=mlp_activation,
            dropout=mlp_dropout,
            dropout_first=True,
            use_bn=mlp_use_bn,
            bn_momentum=mlp_bn_momentum,
            leaky_gate=mlp_leaky_gate,
            use_skip=mlp_use_skip,
            weighted_sum=weighted_sum,
            device=device,
        )

        self.mix = self.mlp.mix
        self.embedding_size = embed_info.embedding_size
        #self.to(device)

    __init__.__doc__ = INIT_DOC

    @staticmethod
    def diagram():
        """ Print a text diagram of this model """
        gram = """\
        if use_skip=False (default)
        ---------------------------
        X_num ─ Num. embedding ┐
                               ├─── MLP ── output
        X_cat ─ Cat. embedding ┘

        if use_skip=True
        ----------------------
        X_num ─ Num. embedding ┐ ┌─── MLP ──┐
                               ├─┤          w+ ── output
        X_cat ─ Cat. embedding ┘ └─ Linear ─┘

        splits are copies and joins are concatenations;
        'w+' is weighted element-wise addition
        """
        print("\n" + textwrap.dedent(gram))

    def mlp_weight_sum(self) -> Tuple[float, float]:
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
        embedded = torch.cat(self.embed(X_num, X_cat), dim=1)
        embedded = embedded.reshape((X_num.shape[0], -1))
        return self.mlp(embedded)

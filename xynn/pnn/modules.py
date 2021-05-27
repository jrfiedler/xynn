"""
PyTorch modules for the PNN and PNNPlus models

"""
# product implementations adapted from
# https://github.com/JianzhouZhan/Awesome-RecSystem-Models/blob/master/Model/PNN_PyTorch.py
# Paper: https://arxiv.org/pdf/1611.00144.pdf

import textwrap
from typing import Union, Tuple, Callable, Optional, Type, List

import torch
from torch import Tensor
from torch import nn

from ..base_classes.modules import BaseNN, MODULE_INIT_DOC
from ..mlp import MLP
from ..embedding import check_uniform_embeddings
from ..embedding.common import EmbeddingBase


INIT_DOC = MODULE_INIT_DOC.format(
    textwrap.dedent(
        """\
        pnn_product_type : {"inner", "outer", "both"}, optional
            default is "outer"
        pnn_product_size : int, optional
            size of overall product output after transformation; after
            transformation, the batch size is num_rows x pnn_product_size;
            default is 10"""
    )
)


def xavier_linear(size: Union[int, Tuple[int, ...]]) -> nn.Parameter:
    """
    Create a tensor with given size, initial with Xavier uniform weights,
    and convert to nn.Parameter

    Parameters
    ----------
    size : int or tuple of ints

    Return
    ------
    nn.Parameter

    """
    weights = torch.empty(size)
    nn.init.xavier_uniform_(weights)
    return nn.Parameter(weights)


class InnerProduct(nn.Module):
    """
    Inner product of embedded vectors, originally used in the PNN model

    Input needs to be shaped like (num_rows, num_fields, embedding_size)

    """

    def __init__(
        self,
        num_fields: int,
        output_size: int = 10,
        device: Union[str, torch.device] = "cpu",
    ):
        """
        Parameters
        ----------
        task : {"regression", "classification"}
        num_fields : int
            number of input fields
        output_size : int, optional
            size of output after product and transformation; after
            transformation, the batch size is num_rows x output_size;
            default is 10
        device : string or torch.device, optional
            default is "cpu"

        """
        super().__init__()
        self.weights = xavier_linear((output_size, num_fields))
        self.to(device=device)

    def forward(self, x: Tensor) -> Tensor:
        """
        Transform the input tensor

        Parameters
        ----------
        x : torch.Tensor
            shape should be (num_rows, num_fields, embedding_size)

        Return
        ------
        torch.Tensor

        """
        # r = # rows (batch size)
        # f = # fields
        # e = embedding size
        # p = product output size
        delta = torch.einsum('rfe,pf->rpfe', x, self.weights)
        lp = torch.einsum('rpfe,rpfe->rp', delta, delta)
        return lp


class OuterProduct(nn.Module):
    """
    Outer product of embedded vectors, originally used in the PNN model

    Input needs to be shaped like (num_rows, num_fields, embedding_size)

    """

    def __init__(
        self,
        embedding_size: int,
        output_size: int = 10,
        device: Union[str, torch.device] = "cpu"
    ):
        """
        Parameters
        ----------
        task : {"regression", "classification"}
        embedding_size : int
            length of embedding vectors in input; all inputs are assumed
            to be embedded values
        output_size : int, optional
            size of output after product and transformation; after
            transformation, the batch size is num_rows x output_size;
            default is 10
        device : string or torch.device, optional
            default is "cpu"

        """
        super().__init__()
        self.weights = xavier_linear((output_size, embedding_size, embedding_size))
        self.to(device=device)

    def forward(self, x: Tensor) -> Tensor:
        """
        Transform the input tensor

        Parameters
        ----------
        x : torch.Tensor
            shape should be (num_rows, num_fields, embedding_size)

        Return
        ------
        torch.Tensor

        """
        # r = # rows (batch size)
        # f = # fields
        # e, m = embedding size (two letters are needed)
        # p = product output size
        f_sigma = x.sum(dim=1)  # rfe -> re
        p = torch.einsum('re,rm->rem', f_sigma, f_sigma)
        lp = torch.einsum('rem,pem->rp', p, self.weights)
        return lp


class PNNCore(nn.Module):
    """
    The core components and calculations of PNN models, to be used in
    PNN and PNNPlus model classes

    """

    def __init__(
        self,
        task: str,
        output_size: int,
        embedding_num: Optional[EmbeddingBase],
        embedding_cat: Optional[EmbeddingBase],
        pnn_product_type: str = "outer",
        pnn_product_size: int = 10,
        mlp_hidden_sizes: Union[int, Tuple[int, ...], List[int]] = (512, 256, 128, 64),
        mlp_activation: Type[nn.Module] = nn.LeakyReLU,
        mlp_use_bn: bool = True,
        mlp_bn_momentum: float = 0.1,
        mlp_dropout: float = 0.0,
        mlp_leaky_gate: bool = True,
        mlp_use_skip: bool = True,
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__()

        if pnn_product_type not in {"inner", "outer", "both"}:
            raise ValueError("pnn_product_type should be 'inner', 'outer', or 'both'")

        embed_info = check_uniform_embeddings(embedding_num, embedding_cat)

        self.product_type = pnn_product_type

        # linear
        self.weights_linear = xavier_linear(
            (pnn_product_size, embed_info.num_fields, embed_info.embedding_size)
        )

        # inner product
        if self.product_type in ('inner', 'both'):
            self.inner = InnerProduct(
                embed_info.num_fields, pnn_product_size, device
            )
        else:
            self.inner = None

        # outer product
        if self.product_type in ('outer', 'both'):
            self.outer = OuterProduct(
                embed_info.embedding_size, pnn_product_size, device
            )
        else:
            self.outer = None

        # MLP
        mlp_input_size = pnn_product_size * (2 if pnn_product_type != "both" else 3)

        self.mlp = MLP(
            task,
            input_size=mlp_input_size,
            hidden_sizes=mlp_hidden_sizes,
            output_size=output_size,
            activation=mlp_activation,
            dropout=mlp_dropout,
            dropout_first=True,
            use_bn=mlp_use_bn,
            bn_momentum=mlp_bn_momentum,
            leaky_gate=mlp_leaky_gate,
            use_skip=mlp_use_skip,
            device=device,
        )

        self.to(device=device)

    __init__.__doc__ = INIT_DOC

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transform the input tensor

        Parameters
        ----------
        x : torch.Tensor

        Return
        ------
        torch.Tensor

        """
        outputs = []

        # r = # rows (batch size)
        # f = # fields
        # e, m = embedding size (two letters are needed below)
        # p = product output size

        # linear
        lz = torch.einsum('rfe,pfe->rp', x, self.weights_linear)
        outputs.append(lz)

        # inner product
        if self.product_type in ('inner', 'both'):
            outputs.append(self.inner(x))

        # outer product
        if self.product_type in ('outer', 'both'):
            outputs.append(self.outer(x))

        out = torch.cat(outputs, dim=1)
        out = self.mlp(out)
        return out


class PNN(BaseNN):
    """
    The PNN model. See PNN.diagram() for the general structure of the model.

    Paper for the original PNN model: https://arxiv.org/pdf/1611.00144.pdf

    """

    def __init__(
        self,
        task: str,
        output_size: int,
        embedding_num: Optional[EmbeddingBase],
        embedding_cat: Optional[EmbeddingBase],
        embedding_l1_reg: float = 0.0,
        embedding_l2_reg: float = 0.0,
        pnn_product_type: str = "outer",
        pnn_product_size: int = 10,
        mlp_hidden_sizes: Union[int, Tuple[int, ...], List[int]] = (512, 256, 128, 64),
        mlp_activation: Type[nn.Module] = nn.LeakyReLU,
        mlp_use_bn: bool = True,
        mlp_bn_momentum: float = 0.1,
        mlp_dropout: float = 0.0,
        mlp_leaky_gate: bool = True,
        mlp_use_skip: bool = True,
        mlp_l1_reg: float = 0.0,
        mlp_l2_reg: float = 0.0,
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

        self.pnn = PNNCore(
            task="classification",
            output_size=output_size,
            embedding_num=embedding_num,
            embedding_cat=embedding_cat,
            pnn_product_type=pnn_product_type,
            pnn_product_size=pnn_product_size,
            mlp_hidden_sizes=mlp_hidden_sizes,
            mlp_activation=mlp_activation,
            mlp_use_bn=mlp_use_bn,
            mlp_bn_momentum=mlp_bn_momentum,
            mlp_dropout=mlp_dropout,
            mlp_leaky_gate=mlp_leaky_gate,
            mlp_use_skip=mlp_use_skip,
            device=device,
        )
        #self.to(device)

    __init__.__doc__ = INIT_DOC

    @staticmethod
    def diagram():
        """ Print a text diagram of this model """
        gram = """\
        if pnn_product_type="inner" or pnn_product_type="outer"
        -------------------------------------------------------
        X_num ─ Num. embedding ┬─┬─ Linear ──────────────┬─ MLP
        X_cat ─ Cat. embedding ┘ └─ inner/outer product ─┘

        if pnn_product_type="both"
        --------------------------
        X_num ─ Num. embedding ┬─┬─ Linear ────────┬─ MLP
        X_cat ─ Cat. embedding ┘ ├─ inner product ─┤
                                 └─ outer product ─┘

        splits are copies and joins are concatenations
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
        return self.pnn.mlp.weight_sum()

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
        embedded = self.embed(X_num, X_cat)
        return self.pnn(embedded)


class PNNPlus(BaseNN):
    """
    The PNN model, with a side MLP component. See PNNPlus.diagram()
    for the general structure of the model.

    Paper for the original PNN model: https://arxiv.org/pdf/1611.00144.pdf

    """

    def __init__(
        self,
        task: str,
        output_size: int,
        embedding_num: Optional[EmbeddingBase],
        embedding_cat: Optional[EmbeddingBase],
        embedding_l1_reg: float = 0.0,
        embedding_l2_reg: float = 0.0,
        pnn_product_type: str = "outer",
        pnn_product_size: int = 10,
        mlp_hidden_sizes: Union[int, Tuple[int, ...], List[int]] = (512, 256, 128, 64),
        mlp_activation: Type[nn.Module] = nn.LeakyReLU,
        mlp_use_bn: bool = True,
        mlp_bn_momentum: float = 0.1,
        mlp_dropout: float = 0.0,
        mlp_leaky_gate: bool = True,
        mlp_use_skip: bool = True,
        mlp_l1_reg: float = 0.0,
        mlp_l2_reg: float = 0.0,
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

        if pnn_product_type not in {"inner", "outer", "both"}:
            raise ValueError("pnn_product_type should be 'inner', 'outer', or 'both'")

        embed_info = check_uniform_embeddings(embedding_num, embedding_cat)

        self.pnn = PNNCore(
            task="classification",
            output_size=output_size,
            embedding_num=embedding_num,
            embedding_cat=embedding_cat,
            pnn_product_type=pnn_product_type,
            pnn_product_size=pnn_product_size,
            mlp_hidden_sizes=mlp_hidden_sizes,
            mlp_activation=mlp_activation,
            mlp_use_bn=mlp_use_bn,
            mlp_bn_momentum=mlp_bn_momentum,
            mlp_dropout=mlp_dropout,
            mlp_leaky_gate=mlp_leaky_gate,
            mlp_use_skip=mlp_use_skip,
            device=device,
        )

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
            device=device,
        )

        self.embedding_size = embed_info.embedding_size
        if weighted_sum:
            self.mix = nn.Parameter(torch.tensor([0.0], device=device))
        else:
            self.mix = torch.tensor([0.0], device=device)
        #self.to(device)

    __init__.__doc__ = INIT_DOC

    @staticmethod
    def diagram():
        """ Print a text diagram of this model """
        gram = """\
        if pnn_product_type="outer" (default) or pnn_product_type="inner"
        ---------------------------------------------------------
        X_num ─ Num. embedding ┐ ┌─────── Linear ────────┬─ MLP ─┐
                               ├─┼─ inner/outer product ─┘       w+ ── output
        X_cat ─ Cat. embedding ┘ └───────────────────────── MLP ─┘

        if pnn_product_type="both"
        ----------------------   ┌──── Linear ─────┐
        X_num ─ Num. embedding ┐ ├─ inner product ─┼─ MLP ─┐
                               ├─┼─ outer product ─┘       w+ ── output
        X_cat ─ Cat. embedding ┘ └─────────────────── MLP ─┘

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
        pnn_w1, pnn_w2 = self.pnn.mlp.weight_sum()
        mlp_w1, mlp_w2 = self.mlp.weight_sum()
        return pnn_w1 + mlp_w1, pnn_w2 + mlp_w2

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
        embedded = self.embed(X_num, X_cat)
        mix = torch.sigmoid(self.mix)
        out_1 = self.pnn(embedded)
        out_2 = self.mlp(embedded.reshape((X_num.shape[0], -1)))
        out = mix * out_1 + (1 - mix) * out_2
        return out

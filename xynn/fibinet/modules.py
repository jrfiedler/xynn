"""
PyTorch modules for the FiBiNet model

"""
# Paper: https://arxiv.org/pdf/1905.09433v1.pdf
# Official implementation: 

import textwrap
from itertools import combinations
from typing import Union, Type, Optional, Callable, Tuple, List

import torch
from torch import nn, Tensor

from ..embedding.common import EmbeddingBase
from ..embedding import check_uniform_embeddings
from ..base_classes.modules import BaseNN, MODULE_INIT_DOC
from ..mlp import MLP


INIT_DOC = MODULE_INIT_DOC.format(
    textwrap.dedent(
        """\
        num_numeric_fields : int or "auto", optional
            an integer must be specified when embedding_num is None;
            default is \"auto\"
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
        fibi_senet_skip : bool, optional
            whether SENET output should also be used in both the MLP and Bilinear
            layer (True), or just the Bilinear layer (False); see FiBiNet.diagram();
            default is True"""
    )
)


class SENET(nn.Module):
    """
    A "squeeze and excitation" layer to create an alternate or modified embedding,
    originally for the FiBiNet model.

    Paper for the original FiBiNet model: https://arxiv.org/pdf/1905.09433v1.pdf

    """

    def __init__(
        self,
        num_fields: int,
        reduction_ratio: int = 3,
        activation: Type[nn.Module] = nn.LeakyReLU,
        device: Union[str, torch.device] = "cpu",
    ):
        if not isinstance(reduction_ratio, int) or reduction_ratio <= 0:
            raise ValueError(
                f"reduction_ratio should be a positive integer, got {reduction_ratio}"
            )

        super().__init__()

        width = max(1, num_fields // reduction_ratio)
        self.layers = nn.Sequential(
            nn.Linear(num_fields, width, bias=False),
            activation(),
            nn.Linear(width, num_fields, bias=False),
            activation(),
        )

        self.to(device)

    def forward(self, x: Tensor) -> Tensor:
        """
        Transform the input tensor to create an alternate embedding

        Parameters
        ----------
        x : torch.Tensor
            3-d tensor, usually embedded numeric and/or categorical values

        Return
        ------
        torch.Tensor

        """
        out = x.mean(dim=-1)
        out = self.layers(out)
        out = x * out.unsqueeze(dim=-1)
        return out


class Bilinear(nn.Module):
    """
    Bilinear interaction layer, originally for the FiBiNet model.

    Paper for the original FiBiNet model: https://arxiv.org/pdf/1905.09433v1.pdf

    """

    def __init__(
        self,
        num_fields: int,
        embedding_size: int,
        product_type: str = "sym-interaction",
        device: Union[str, torch.device] = "cpu",
    ):
        """
        Parameters
        ----------
        num_fields : int
        embedding_size : int
        product_type : str, optional
            options:
                - "field-all"
                - "field-each"
                - "field-interaction"
                - "sym-all"
                - "sym-each"
                - "sym-interaction"
            "field" :
                the original asymmetric bilinear products, with products like
                `linear(field_1) * field_2` where `*` is elementwise multiplication
            "sym" :
                symmetric versions of the "field" products
                `(linear(field_1) * field_2 + field_1 * linear(field_2)) / 2`
            "all" : a single product matrix is shared across all pairs of fields
            "each" : each field has an associated product matrix
            "interaction" : each pair of fields has an associated product matrix
            default is \"sym-interaction\"
        device : string or torch.device, optional
            default is "cpu"

        """
        super().__init__()

        product_type = product_type.lower()

        if not product_type in {
            "field-all",
            "field-each",
            "field-interaction",
            "sym-all",
            "sym-each",
            "sym-interaction",
        }:
            raise ValueError(f"unrecognized product type '{product_type}'")
        elif product_type.startswith("sym-"):
            self.symmetric = True
            product_type = product_type[len("sym-"):]
        else:
            self.symmetric = False
            product_type = product_type[len("field-"):]

        if product_type == "all":
            num_layers = 1
        elif product_type == "each":
            num_layers = num_fields
        else:  # "interaction"
            num_layers = (num_fields * (num_fields - 1)) // 2

        self.layers = nn.ModuleList(
            [
                nn.Linear(embedding_size, embedding_size, bias=False)
                for _ in range(num_layers)
            ]
        )
        self.product_type = product_type

        self.to(device)

    def _product(self, x: Tensor) -> Tensor:
        """
        Transform the input tensor

        Parameters
        ----------
        x : torch.Tensor
            3-d tensor with embedded numeric and/or categorical values

        Return
        ------
        torch.Tensor

        """
        columns = x.split(1, dim=1)
        layers = self.layers
        if self.product_type == "all":
            linear = layers[0]
            col_pairs = combinations(columns, 2)
            products = [linear(v_i) * v_j for v_i, v_j in col_pairs]
        elif self.product_type == "each":
            lin_col_pairs = combinations(zip(layers, columns), 2)
            products = [linear(v_i) * v_j for (linear, v_i), (_, v_j) in lin_col_pairs]
        else:  # "interaction"
            col_pairs = combinations(columns, 2)
            products = [linear(v_i) * v_j for (v_i, v_j), linear in zip(col_pairs, layers)]

        return torch.cat(products, dim=1)

    def _symmetric_product(self, x: Tensor) -> Tensor:
        """
        Transform the input tensor

        Parameters
        ----------
        x : torch.Tensor
            3-d tensor with embedded numeric and/or categorical values

        Return
        ------
        torch.Tensor

        """
        columns = x.split(1, dim=1)
        layers = self.layers
        if self.product_type == "all":
            linear = layers[0]
            col_pairs = combinations(columns, 2)
            products = [
                (linear(v_i) * v_j + v_i * linear(v_j)) / 2
                for v_i, v_j in col_pairs
            ]
        elif self.product_type == "each":
            lin_col_pairs = combinations(zip(layers, columns), 2)
            products = [
                (linear_i(v_i) * v_j + v_i * linear_j(v_j)) / 2
                for (linear_i, v_i), (linear_j, v_j) in lin_col_pairs
            ]
        else:  # "interaction"
            col_pairs = combinations(columns, 2)
            products = [
                (linear(v_i) * v_j + v_i * linear(v_j)) / 2
                for (v_i, v_j), linear in zip(col_pairs, layers)
            ]

        return torch.cat(products, dim=1)

    def forward(self, x: Tensor) -> Tensor:
        """
        Transform the input tensor

        Parameters
        ----------
        x : torch.Tensor
            3-d tensor with embedded numeric and/or categorical values

        Return
        ------
        torch.Tensor

        """
        if self.symmetric:
            return self._symmetric_product(x)
        return self._product(x)


class Hadamard(nn.Module):
    """
    Hadamard interaction layer, originally for the FiBiNet model.

    Paper for the original FiBiNet model: https://arxiv.org/pdf/1905.09433v1.pdf

    """

    def forward(self, x: Tensor) -> Tensor:
        """
        Transform the input tensor

        Parameters
        ----------
        x : torch.Tensor
            2-d or 3-d tensor

        Return
        ------
        torch.Tensor

        """
        x_cols = x.split(1, dim=1)
        col_pairs = combinations(x_cols, 2)
        products = [x_i * x_j for x_i, x_j in col_pairs]
        return torch.cat(products, dim=1)


class FiBiNet(BaseNN):
    """
    The FiBiNet model, with modifications. See FiBiNet.diagram() for the general
    structure of the model.

    Paper for the original FiBiNet model: https://arxiv.org/pdf/1905.09433v1.pdf

    """

    def __init__(
        self,
        task: str,
        output_size: int,
        embedding_num: Optional[EmbeddingBase],
        embedding_cat: Optional[EmbeddingBase],
        embedding_l1_reg: float = 0.0,
        embedding_l2_reg: float = 0.0,
        num_numeric_fields: Union[int, str] = "auto",
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

        embed_info = check_uniform_embeddings(embedding_num, embedding_cat)
        num_fields, embed_size, tot_embed_size = embed_info

        if not isinstance(num_numeric_fields, int):
            if embedding_num is None:
                msg = "when embedding_num is None, num_numeric_fields must be an integer"
                raise TypeError(msg)
            num_numeric_fields = embedding_num.num_fields

        self.senet = SENET(
            num_fields=num_fields,
            reduction_ratio=fibi_reduction_ratio,
            activation=fibi_activation,
            device=device,
        )

        senet_product_type = fibi_senet_product.lower()
        if senet_product_type == "hadamard":
            self.senet_product = Hadamard()
        else:
            self.senet_product = Bilinear(
                num_fields=num_fields,
                embedding_size=embed_size,
                product_type=fibi_senet_product,
                device=device,
            )

        embed_product_type = fibi_embed_product.lower()
        if embed_product_type == "shared":
            self.embed_product = self.senet_product
        elif embed_product_type == "hadamard":
            self.embed_product = Hadamard()
        else:
            self.embed_product = Bilinear(
                num_fields=embed_info.num_fields,
                embedding_size=embed_info.embedding_size,
                product_type=fibi_embed_product,
                device=device,
            )

        num_products = (num_fields * (num_fields - 1)) // 2
        mlp_input_size = num_numeric_fields + embed_size * num_products * 2
        if fibi_senet_skip:
            mlp_input_size += tot_embed_size

        self.mlp = MLP(
            task=task,
            input_size=mlp_input_size,
            hidden_sizes=mlp_hidden_sizes,
            output_size=output_size,
            activation=mlp_activation,
            dropout=mlp_dropout,
            use_bn=mlp_use_bn,
            bn_momentum=mlp_bn_momentum,
            ghost_batch=mlp_ghost_batch,
            leaky_gate=mlp_leaky_gate,
            use_skip=mlp_use_skip,
            device=device,
        )

        self.senet_skip = fibi_senet_skip
        #self.to(device)

    __init__.__doc__ = INIT_DOC

    @staticmethod
    def diagram():
        """ Print a text diagram of this model """
        gram = """\
        if fibi_senet_skip (default)
        ----------------------------

            if embedding_num is not None (default)
            --------------------------------------
            X_num ┬──────────────────────────────────────────────┐
                  └ Num. embedding ┐ ┌ SENET ┬───────────────────┤
            X_cat ─ Cat. embedding ┴─┤       └ Bilinear/Hadamard ┼─ MLP
                                     └──────── Bilinear/Hadamard ┘

            if embedding_num is None
            ------------------------
            X_num ──────────────────────────────────────────────┐
                                    ┌ SENET ┬───────────────────┤
            X_cat ─ Cat. embedding ─┤       └ Bilinear/Hadamard ┼─ MLP
                                    └──────── Bilinear/Hadamard ┘

        if not fibi_senet_skip
        ----------------------

            if embedding_num is not None (default)
            --------------------------------------
            X_num ┬──────────────────────────────────────────────┐
                  └ Num. embedding ┐ ┌ SENET ─ Bilinear/Hadamard ┼─ MLP
            X_cat ─ Cat. embedding ┴─┴──────── Bilinear/Hadamard ┘

            if embedding_num is None
            ------------------------
            X_num ───────────────────────────────────────────────┐
                                     ┌ SENET ─ Bilinear/Hadamard ┼─ MLP
            X_cat ─ Cat. embedding ──┴──────── Bilinear/Hadamard ┘


        splits are copies and joins are concatenations
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
        return self.mlp.weight_sum()

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
        out_s = self.senet(embedded)
        out_p = self.senet_product(out_s).reshape((embedded.shape[0], -1))
        out_q = self.embed_product(embedded).reshape((embedded.shape[0], -1))

        if self.senet_skip:
            out_s = out_s.reshape((embedded.shape[0], -1))
            out = self.mlp(torch.cat([X_num, out_s, out_p, out_q], dim=1))
        else:
            out = self.mlp(torch.cat([X_num, out_p, out_q], dim=1))

        return out

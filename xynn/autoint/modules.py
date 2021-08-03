"""
PyTorch modules for the AutoInt model

"""
# Paper: https://arxiv.org/pdf/1810.11921v2.pdf
# Official implementation: https://github.com/DeepGraphLearning/RecommenderSystems

import textwrap
from typing import Optional, Type, Union, Callable, Tuple, List

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from ..embedding import check_uniform_embeddings
from ..embedding import EmbeddingBase
from ..base_classes.modules import BaseNN, MODULE_INIT_DOC
from ..mlp import MLP
from ..ghost_norm import GhostLayerNorm


INIT_DOC = MODULE_INIT_DOC.format(
    textwrap.dedent(
        """\
        attn_embedding_size : int, optional
            default is 8
        attn_num_layers : int, optional
            default is 3
        attn_num_heads : int, optional
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


def _initialized_tensor(*sizes):
    weight = nn.Parameter(torch.Tensor(*sizes))
    nn.init.kaiming_uniform_(weight)
    return weight


class AttnInteractionLayer(nn.Module):
    """
    The attention interaction layer for the AutoInt model.

    Paper for the original AutoInt model: https://arxiv.org/pdf/1810.11921v2.pdf

    """

    def __init__(
        self,
        field_input_size: int,
        field_output_size: int = 8,
        num_heads: int = 2,
        activation: Optional[Type[nn.Module]] = None,
        use_residual: bool = True,
        dropout: float = 0.1,
        normalize: bool = True,
        ghost_batch_size: Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
    ):
        """
        Parameters
        ----------
        field_input_size : int
            original embedding size for each field
        field_output_size : int, optional
            embedding size after transformation; default is 8
        num_heads : int, optional
            number of attention heads; default is 2
        activation : subclass of torch.nn.Module or None, optional
            applied to the W tensors; default is None
        use_residual : bool, optional
            default is True
        dropout : float, optional
            default is 0.1
        normalize : bool, optional
            default is True
        ghost_batch_size : int or None, optional
            only used if `use_bn` is True; size of batch in "ghost batch norm";
            if None, normal batch norm is used; defualt is None
        device : string or torch.device, optional
            default is "cpu"

        """
        super().__init__()

        self.use_residual = use_residual

        self.W_q = _initialized_tensor(field_input_size, field_output_size, num_heads)
        self.W_k = _initialized_tensor(field_input_size, field_output_size, num_heads)
        self.W_v = _initialized_tensor(field_input_size, field_output_size, num_heads)

        if use_residual:
            self.W_r = _initialized_tensor(field_input_size, field_output_size * num_heads)
        else:
            self.W_r = None

        if activation:
            self.w_act = activation()
        else:
            self.w_act = nn.Identity()

        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()

        if normalize:
            if ghost_batch_size is not None:
                self.layer_norm = GhostLayerNorm(
                    field_output_size * num_heads, ghost_batch_size
                )
            else:
                self.layer_norm = nn.LayerNorm(field_output_size * num_heads)
        else:
            self.layer_norm = nn.Identity()

        self.to(device)

    def forward(self, x: Tensor) -> Tensor:
        """
        Transform the input tensor with attention interaction

        Parameters
        ----------
        x : torch.Tensor
            3-d tensor; for example, embedded numeric and/or categorical values,
            or the output of a previous attention interaction layer

        Return
        ------
        torch.Tensor

        """
        # R : # rows
        # F, D : # fields
        # I : field embedding size in
        # O : field embedding size out
        # H : # heads
        num_rows, num_fields, _ = x.shape  # R, F, I

        # (R, F, I) * (I, O, H) -> (R, F, O, H)
        qrys = torch.tensordot(x, self.w_act(self.W_q), dims=([-1], [0]))
        keys = torch.tensordot(x, self.w_act(self.W_k), dims=([-1], [0]))
        vals = torch.tensordot(x, self.w_act(self.W_v), dims=([-1], [0]))
        if self.use_residual:
            rsdl = torch.tensordot(x, self.w_act(self.W_r), dims=([-1], [0]))

        product = torch.einsum("rdoh,rfoh->rdfh", qrys, keys)  # (R, F, F, H)

        alpha = F.softmax(product, dim=2)  # (R, F, F, H)
        alpha = self.dropout(alpha)

        # (R, F, F, H) * (R, F, O, H) -> (R, F, O, H)
        out = torch.einsum("rfdh,rfoh->rfoh", alpha, vals)
        out = out.reshape((num_rows, num_fields, -1))  # (R, F, O * H)
        if self.use_residual:
            out = out + rsdl  # (R, F, O * H)
        out = F.leaky_relu(out)
        out = self.layer_norm(out)

        return out


class AttnInteractionBlock(nn.Module):
    """
    A collection of AttnInteractionLayers, followed by an optional "leaky gate"
    and then a linear layer. This block is originally for the AutoInt model.

    Paper for the original AutoInt model: https://arxiv.org/pdf/1810.11921v2.pdf

    """

    def __init__(
        self,
        field_input_size: int,
        field_output_size: int = 8,
        num_layers: int = 3,
        num_heads: int = 2,
        activation: Optional[Type[nn.Module]] = None,
        use_residual: bool = True,
        dropout: float = 0.1,
        normalize: bool = True,
        ghost_batch_size: Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
    ):
        """
        Parameters
        ----------
        field_input_size : int
            original embedding size for each field
        field_output_size : int, optional
            embedding size after transformation; default is 8
        num_layers : int, optional
            number of attention layers; default is 3
        num_heads : int, optional
            number of attention heads per layer; default is 2
        activation : subclass of torch.nn.Module or None, optional
            applied to the W tensors; default is None
        use_residual : bool, optional
            default is True
        dropout : float, optional
            default is 0.0
        normalize : bool, optional
            default is True
        ghost_batch_size : int or None, optional
            only used if `use_bn` is True; size of batch in "ghost batch norm";
            if None, normal batch norm is used; defualt is None
        device : string or torch.device, optional
            default is "cpu"

        """
        super().__init__()

        layers = []
        for _ in range(num_layers):
            layers.append(
                AttnInteractionLayer(
                    field_input_size,
                    field_output_size,
                    num_heads,
                    activation,
                    use_residual,
                    dropout,
                    normalize,
                    ghost_batch_size,
                    device,
                )
            )
            field_input_size = field_output_size * num_heads

        self.layers = nn.Sequential(*layers)
        self.to(device)

    def forward(self, x: Tensor) -> Tensor:
        """
        Transform the input tensor

        Parameters
        ----------
        x : torch.Tensor
            3-d tensor, usually embedded numeric and/or categorical values

        Return
        ------
        torch.Tensor

        """
        out = self.layers(x)
        return out


class AutoInt(BaseNN):
    """
    The AutoInt model, with a side MLP component, aka "AutoInt+", with modifications.
    See AutoInt.diagram() for the general structure of the model.

    Paper for the original AutoInt model: https://arxiv.org/pdf/1810.11921v2.pdf

    """

    def __init__(
        self,
        task: str,
        output_size: int,
        embedding_num: Optional[EmbeddingBase],
        embedding_cat: Optional[EmbeddingBase],
        embedding_l1_reg: float = 0.0,
        embedding_l2_reg: float = 0.0,
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

        device = torch.device(device)
        embed_info = check_uniform_embeddings(embedding_num, embedding_cat)

        self.attn_interact = AttnInteractionBlock(
            field_input_size=embed_info.embedding_size,
            field_output_size=attn_embedding_size,
            num_layers=attn_num_layers,
            num_heads=attn_num_heads,
            activation=attn_activation,
            use_residual=attn_use_residual,
            dropout=attn_dropout,
            normalize=attn_normalize,
            ghost_batch_size=mlp_ghost_batch,
            device=device,
        )

        self.attn_final = MLP(
            task=task,
            input_size=embed_info.num_fields * attn_embedding_size * attn_num_heads,
            hidden_sizes=(mlp_hidden_sizes if mlp_hidden_sizes and attn_use_mlp else []),
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

        if mlp_hidden_sizes:
            self.mlp = MLP(
                task=task,
                input_size=embed_info.output_size,
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
            if weighted_sum:
                self.mix = nn.Parameter(torch.tensor([0.0], device=device))
            else:
                self.mix = torch.tensor([0.0], device=device)
        else:
            self.mlp = None
            self.mix = None

        #self.to(device)

    __init__.__doc__ = INIT_DOC

    @staticmethod
    def diagram():
        """ Print a text diagram of this model """
        gram = """\
        if mlp_hidden_sizes (default)
        -----------------------------
        X_num ─ Num. embedding ┐ ┌─ Attn Int ─ ... ─ Attn Int ─ MLP ─┐
                               ├─┤                                   w+ ── output
        X_cat ─ Cat. embedding ┘ └─────────────── MLP ───────────────┘

        if no mlp_hidden_sizes
        ----------------------
        X_num ─ Num. embedding ┬─ Attn Int ─ ... ─ Attn Int ─ Linear ─ output
        X_cat ─ Cat. embedding ┘ 

        splits are copies and joins are concatenations;
        'w+' is weighted element-wise addition;
        "Attn Int" is AutoInt's AttentionInteractionLayer
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
        w1, w2 = self.attn_final.weight_sum()
        if self.mlp is not None:
            side_w1, side_w2 = self.mlp.weight_sum()
            w1 += side_w1
            w2 += side_w2
        return w1, w2

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
        out = self.attn_interact(embedded)
        out = self.attn_final(out.reshape((out.shape[0], -1)))
        if self.mlp is not None:
            embedded_2d = embedded.reshape((embedded.shape[0], -1))
            mix = torch.sigmoid(self.mix)
            out = mix * out + (1 - mix) * self.mlp(embedded_2d)
        return out

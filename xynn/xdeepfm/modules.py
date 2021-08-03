"""
PyTorch modules for the xDeepFM model

"""
# Paper:

import textwrap
from typing import Optional, Type, Union, Callable, Tuple, List

import torch
from torch import nn, Tensor

from ..base_classes.modules import BaseNN, MODULE_INIT_DOC
from ..embedding import check_uniform_embeddings
from ..embedding.common import EmbeddingBase
from ..mlp import MLP, LeakyGate
from ..ghost_norm import GhostBatchNorm


INIT_DOC = MODULE_INIT_DOC.format(
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


class CIN(nn.Module):
    """
    Compressed Interaction Network layer originally used in the xDeepFM model.

    """

    def __init__(
        self,
        num_fields: int,
        layer_sizes: Union[int, Tuple[int, ...], List[int]] = (128, 128),
        activation: Type[nn.Module] = nn.Identity,
        full_agg: bool = False,
        use_bn: bool = True,
        bn_momentum: float = 0.1,
        ghost_batch_size: Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__()

        if not full_agg and any(size % 2 != 0 for size in layer_sizes[:-1]):
            raise ValueError(
                "when using full_agg=False, all but the last layer size must be even"
            )

        self.convs = nn.ModuleList()
        self.actns = nn.ModuleList()
        self.norms = nn.ModuleList()
        final_size = 0
        conv_size = num_fields
        for layer_num, size in enumerate(layer_sizes):
            input_size = num_fields * conv_size

            self.convs.append(
                nn.Conv1d(in_channels=input_size, out_channels=size, kernel_size=1)
            )

            if use_bn:
                if ghost_batch_size is not None:
                    bn = GhostBatchNorm(size, ghost_batch_size, momentum=bn_momentum)
                else:
                    bn = nn.BatchNorm1d(size, momentum=bn_momentum)
            else:
                bn = nn.Identity()

            self.norms.append(bn)
            self.actns.append(activation())

            # for output size / next input size
            if not full_agg and layer_num < len(layer_sizes) - 1:
                conv_size = size // 2
            else:
                conv_size = size
            final_size += conv_size

        self.layer_sizes = layer_sizes
        self.full_agg = full_agg
        self.output_size = final_size
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
        torch.Tensor, 2-d

        """
        # R, r : rows
        # F, f : fields
        # E, e : embedding
        # N, n : convolution size
        num_rows, _, emb_size = x.shape  # R, F, E

        agg = []
        out = x
        for layer_num, (conv, actn, norm, size) in enumerate(
            zip(self.convs, self.actns, self.norms, self.layer_sizes)
        ):
            out = torch.einsum("rne,rfe->rnfe", out, x)
            out = out.reshape(num_rows, -1, emb_size)
            out = conv(out)
            out = norm(out)
            out = actn(out)
            if not self.full_agg and layer_num < len(self.layer_sizes) - 1:
                out, direct = out.split(size // 2, dim=1)
            else:
                direct = out
            agg.append(direct)

        out = torch.cat(agg, dim=1)
        out = out.sum(dim=-1)

        return out


class XDeepFM(BaseNN):
    """
    The xDeepFM model, with modifications.
    See XDeepFM.diagram() for the general structure of the model.

    Paper for the original xDeepFM model: https://arxiv.org/pdf/1803.05170v3.pdf

    """

    def __init__(
        self,
        task: str,
        output_size: int,
        embedding_num: Optional[EmbeddingBase],
        embedding_cat: Optional[EmbeddingBase],
        embedding_l1_reg: float = 0.0,
        embedding_l2_reg: float = 0.0,
        cin_layer_sizes: Union[int, Tuple[int, ...], List[int]] = (128, 128),
        cin_activation: Type[nn.Module] = nn.LeakyReLU,
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
        mlp_use_skip: bool = True,
        mlp_l1_reg: float = 0.0,
        mlp_l2_reg: float = 0.0,
        use_leaky_gate: bool = True,
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

        if use_leaky_gate:
            self.cin_gate = LeakyGate(embed_info.output_size, device=device)
        else:
            self.cin_gate = nn.Identity()

        self.cin = CIN(
            num_fields=embed_info.num_fields,
            layer_sizes=cin_layer_sizes,
            activation=cin_activation,
            full_agg=cin_full_agg,
            use_bn=cin_use_bn,
            bn_momentum=cin_bn_momentum,
            ghost_batch_size=mlp_ghost_batch,
            device=device,
        )

        self.cin_final = MLP(
            task=task,
            input_size=self.cin.output_size,
            hidden_sizes=(mlp_hidden_sizes if mlp_hidden_sizes and cin_use_mlp else []),
            output_size=output_size,
            activation=mlp_activation,
            dropout=mlp_dropout,
            use_bn=mlp_use_bn,
            bn_momentum=mlp_bn_momentum,
            ghost_batch=mlp_ghost_batch,
            leaky_gate=use_leaky_gate,
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
                leaky_gate=use_leaky_gate,
                use_skip=mlp_use_skip,
                device=device,
            )
            self.mix = nn.Parameter(torch.tensor([0.0], device=device))
        else:
            self.mlp = None
            self.mix = None

        self.use_residual = cin_use_residual
        #self.to(device)

    __init__.__doc__ = INIT_DOC

    @staticmethod
    def diagram():
        """ Print a text diagram of this model """
        gram = """\
        if mlp_hidden_sizes (default)
        -----------------------------

            if cin_use_residual (default)
            -----------------------------
            X_num ─ Num. embedding ┐ ┌─ CIN ─┬─ MLP/Linear ─ + ─┐
                                   ├─┤       └─ sum embed. ──┘  w+ ── output
            X_cat ─ Cat. embedding ┘ └─ MLP ────────────────────┘


            if not cin_use_residual
            -----------------------
            X_num ─ Num. embedding ┐ ┌─ CIN ── MLP/Linear ─┐
                                   ├─┤                     w+ ── output
            X_cat ─ Cat. embedding ┘ └────── MLP ──────────┘


        if not mlp_hidden_sizes
        -----------------------

            if cin_use_residual (default)
            -----------------------------
            X_num ─ Num. embedding ┐        ┌─── Linear ───┐
                                   ├── CIN ─┤              + ── output
            X_cat ─ Cat. embedding ┘        └─ sum embed. ─┘


            if not cin_use_residual
            -----------------------
            X_num ─ Num. embedding ┐
                                   ├── CIN ── Linear ── output
            X_cat ─ Cat. embedding ┘

        splits are copies and joins are concatenations;
        'w+' is weighted element-wise addition
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
        w1, w2 = self.mlp.weight_sum()
        if isinstance(self.cin_final, MLP):
            cin_final_w1, cin_final_w2 = self.cin_final.weight_sum()
            w1 += cin_final_w1
            w2 += cin_final_w2
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
        cin_out = self.cin_gate(embedded)
        cin_out = self.cin(cin_out)
        out = self.cin_final(cin_out)
        if self.use_residual:
            out = cin_out.sum(dim=1, keepdim=True) + out
        if self.mlp is not None:
            out_mlp = embedded.reshape((embedded.shape[0], -1))
            out_mlp = self.mlp(out_mlp)
            mix = torch.sigmoid(self.mix)
            out = mix * out + (1 - mix) * out_mlp
        return out

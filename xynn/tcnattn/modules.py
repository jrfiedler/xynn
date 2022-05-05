"""
PyTorch module for the TCN-Attn model

"""

# Some modules borrow heavily from https://github.com/locuslab/TCN

import textwrap
from math import ceil
from typing import Union, Tuple, Callable, Optional, Type, List, Literal

import torch
from torch import Tensor, nn
from torch.nn.utils import weight_norm
from entmax import entmax15

from ..base_classes.modules import BaseNN, MODULE_INIT_DOC
from ..mlp import MLP, LeakyGate
from ..embedding import check_uniform_embeddings
from ..embedding.common import EmbeddingBase
from ..autoint.modules import AttnInteractionBlock


INIT_DOC = MODULE_INIT_DOC.format(
    textwrap.dedent(
        """\
        tcn_temporal : bool, optional
            only used when `tcn_outputs` is "single"; if True, the final
            field is used, otherwise the middle field is used; default is False
        tcn_dilations : int, tuple of ints, list of ints, or "auto"; optional
            default is "auto"
        tcn_hidden_sizes : int, tuple of ints, or list of ints; optional
            default is (30, 30, 30, 30, 30, 30)
        tcn_kernel_size : int, optional
            default is 5
        tcn_use_linear : bool, optional
            default is False
        tcn_dropout : float, optional
            default is 0.0
        tcn_outputs : {"all", "single"}, optional
            default is "all"
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


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size] #.contiguous()


class Linear1d(nn.Module):
    def __init__(self, size_in, size_out, bias=True):
        super().__init__()
        self.linear = nn.Linear(size_in, size_out, bias=bias)
        self.weight = self.linear.weight

    def forward(self, x):
        out = x.permute(0, 2, 1)  #.contiguous()
        out = self.linear(out)
        out = out.permute(0, 2, 1)  #.contiguous()
        return out


class Linear1d(nn.Module):
    def __init__(
        self,
        size_in,
        size_out,
        bias=True,
        transform=False,
        reorder_dims=True,
        device="cpu"
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(size_in, size_out, dtype=torch.float32))
        self.bias = torch.zeros(size_out, dtype=torch.float32)
        if bias:
            self.bias = nn.Parameter(self.bias)
        self.transform = transform
        self.reorder_dims = reorder_dims
        self.to(device)

    def forward(self, x):
        weight = self.weight
        if self.transform:
            weight = entmax15(weight, dim=0)

        if self.reorder_dims:
            out = x.permute(0, 2, 1)  #.contiguous()
            out = out.matmul(weight) + self.bias
            out = out.permute(0, 2, 1)  #.contiguous()
        else:
            out = x.matmul(weight) + self.bias

        return out


class TemporalBlock(nn.Module):
    def __init__(
        self,
        n_inputs,
        n_outputs,
        num_fields,
        kernel_size,
        stride,
        dilation,
        padding,
        use_linear=False,
        dropout=0.2,
        device="cpu",
    ):
        super(TemporalBlock, self).__init__()

        self.use_linear = use_linear
        if use_linear:
            self.linear1 = Linear1d(n_inputs, n_inputs, bias=True)
        else:
            self.linear1 = nn.Identity()

        self.conv1 = weight_norm(
            nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                device=device,
            )
        )
        self.chomp1 = nn.Identity() if padding == "same" else Chomp1d(padding)
        self.norm1 = nn.LayerNorm((num_fields,))
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        if use_linear:
            self.linear2 = Linear1d(n_outputs, n_outputs, bias=True)
        else:
            self.linear2 = nn.Identity()

        self.conv2 = weight_norm(
            nn.Conv1d(
                n_outputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                device=device,
            )
        )
        self.chomp2 = nn.Identity() if padding == "same" else Chomp1d(padding)
        self.norm2 = nn.LayerNorm((num_fields,))
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.linear1,
            self.conv1,
            self.chomp1,
            self.norm1,
            self.relu1,
            self.dropout1,
            self.linear2,
            self.conv2,
            self.chomp2,
            self.norm2,
            self.relu2,
            self.dropout2,
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1, device=device) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()
        self.to(device)

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(
        self,
        temporal,
        num_inputs,
        num_channels,
        num_fields,
        dilations="auto",
        kernel_size=5,
        use_linear=False,
        ghost_batch_size=None,
        dropout=0.2,
        device="cpu",
    ):
        super().__init__()
        
        num_levels = len(num_channels)
        if dilations == "auto":
            dilations = [2 ** i for i in range(num_levels)]

        if not temporal:
            padding = "same"

        layers = []
        for i in range(num_levels):
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            dilation = dilations[i]
            if temporal:
                padding = (kernel_size - 1) * dilation
            new_block = TemporalBlock(
                in_channels,
                out_channels,
                num_fields,
                kernel_size,
                stride=1,
                dilation=dilation,
                padding=padding,
                use_linear=use_linear,
                dropout=dropout,
                device=device,
            )
            layers.append(new_block)

        self.network = nn.Sequential(*layers)
        self.virtual_batch_size = ghost_batch_size
        self.to(device)

    def forward(self, x):
        return self.network(x)


class TCNAttn(BaseNN):

    def __init__(
        self,
        task: str,
        output_size: int,
        embedding_num: Optional[EmbeddingBase],
        embedding_cat: Optional[EmbeddingBase],
        embedding_l1_reg: float = 0.0,
        embedding_l2_reg: float = 0.0,
        tcn_output: Literal["temporal", "non-temporal", "all"] = "all",
        tcn_dilations: Union[Tuple[int, ...], List[int], str] = "auto",
        tcn_hidden_sizes: Union[int, Tuple[int, ...], List[int]] = (30, 30, 30, 30, 30, 30),
        tcn_kernel_size: int = 5,
        tcn_use_linear: bool = False,
        tcn_dropout: float = 0.0,
        attn_embedding_size: int = 8,
        attn_num_layers: int = 3,
        attn_num_heads: int = 2,
        attn_activation: Optional[Type[nn.Module]] = None,
        attn_use_residual: bool = True,
        attn_dropout: float = 0.1,
        attn_normalize: bool = True,
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

        embed_info = check_uniform_embeddings(embedding_num, embedding_cat)

        self.tcn_output = tcn_output

        if use_leaky_gate:
            self.main_gate = LeakyGate(embed_info.output_size, device=device)
        else:
            self.main_gate = nn.Identity()

        self.tcn = TemporalConvNet(
            temporal=(self.tcn_output == "temporal"),
            num_inputs=embed_info.embedding_size,
            num_channels=tcn_hidden_sizes,
            num_fields=embed_info.num_fields,
            dilations=tcn_dilations,
            kernel_size=tcn_kernel_size,
            ghost_batch_size=mlp_ghost_batch,
            use_linear=tcn_use_linear,
            dropout=tcn_dropout,
            device=device,
        )

        self.attn_interact = AttnInteractionBlock(
            field_input_size=tcn_hidden_sizes[-1],
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

        if self.tcn_output == "all":
            mlp_input_size = attn_embedding_size * attn_num_heads * embed_info.num_fields
        else:
            mlp_input_size = attn_embedding_size * attn_num_heads

        self.main_mlp = MLP(
            task,
            input_size=mlp_input_size,
            hidden_sizes=mlp_hidden_sizes,
            output_size=output_size,
            activation=mlp_activation,
            dropout=mlp_dropout,
            use_bn=mlp_use_bn,
            bn_momentum=mlp_bn_momentum,
            ghost_batch=mlp_ghost_batch,
            leaky_gate=use_leaky_gate,
            use_skip=mlp_use_skip,
            weighted_sum=weighted_sum,
            device=device,
        )

        self.side_mlp = MLP(
            task,
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
            weighted_sum=weighted_sum,
            device=device,
        )

        self.mix = nn.Parameter(torch.tensor([0.0], device=device))
        self.to(device)

    __init__.__doc__ = INIT_DOC

    @staticmethod
    def diagram():
        """ Print a text diagram of this model """
        gram = """\
        X_num ─ Num. embedding ┐ ┌─ TCN ─ Attn ─ MLP ─┐
                               ├─┤                    w+ ── output
        X_cat ─ Cat. embedding ┘ └──────── MLP ───────┘

        splits are copies and joins are concatenations;
        'w+' is weighted element-wise addition;
        "Attn" indicates 1 or more of AutoInt's AttentionInteractionLayer
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
        w1, w2 = self.main_mlp.weight_sum()
        side_w1, side_w2 = self.side_mlp.weight_sum()
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
        embedded_2d = embedded.flatten(start_dim=1)

        out_main = self.main_gate(embedded)
        out_main = self.tcn(out_main.movedim(2, 1))
        if self.tcn_output == "temporal":
            out_main = out_main[:, :, [-1]]
        elif self.tcn_output == "non-temporal":
            out_main = out_main[:, :, [out_main.shape[-1] // 2]]
        out_main = self.attn_interact(out_main.movedim(2, 1))
        out_main = self.main_mlp(out_main.flatten(start_dim=1))

        out_side = self.side_mlp(embedded_2d)

        mix = torch.sigmoid(self.mix)
        out = mix * out_main + (1 - mix) * out_side

        return out


"""
PyTorch modules for the AutoInt model

"""
# Paper: https://arxiv.org/pdf/1810.11921v2.pdf
# Official implementation: https://github.com/DeepGraphLearning/RecommenderSystems

import textwrap
from typing import Optional, Type, Union, Callable, Tuple, List

import torch
from torch import nn
from torch.nn import functional as F

from ..embedding import check_uniform_embeddings
from ..embedding import EmbeddingBase
from ..base_classes.modules import BaseNN, MODULE_INIT_DOC
from ..mlp import MLP, LeakyGate


INIT_DOC = MODULE_INIT_DOC.format(
    textwrap.dedent(
        """\
        attn_embedding_size : int, optional
            default is 8
        attn_num_layer : int, optional
            default is 3
        attn_num_head : int, optional
            default is 2
        attn_activation : subclass of torch.nn.Module or None, optional
            default is None
        attn_use_residual : bool, optional
            default is True
        attn_dropout : float, optional
            default is 0.1
        attn_normalize : bool, optional
            default is True"""
    )
)


def _initialized_tensor(input_dim, output_dim):
    weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
    nn.init.kaiming_uniform_(weight)
    return weight


class AttnInteractionLayer(nn.Module):

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_head: int,
        activation: Optional[Type[nn.Module]],
        use_residual: bool,
        dropout: float,
        normalize: bool,
        device: Union[str, torch.device],
    ):
        super().__init__()

        self.num_head = num_head
        self.output_dim = output_dim
        self.use_residual = use_residual

        self.W_q = _initialized_tensor(input_dim, output_dim * num_head)
        self.W_k = _initialized_tensor(input_dim, output_dim * num_head)
        self.W_v = _initialized_tensor(input_dim, output_dim * num_head)

        if use_residual:
            self.W_r = _initialized_tensor(input_dim, output_dim * num_head)
            #self.mix = nn.Parameter(torch.tensor([0.0]))
            self.mix = torch.tensor([0.0])
        else:
            self.W_r = None
            self.mix = 1.0

        if activation:
            self.w_act = activation()
        else:
            self.w_act = nn.Identity()

        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()

        if normalize:
            self.layer_norm = nn.LayerNorm(output_dim * num_head)
        else:
            self.layer_norm = nn.Identity()

        self.to(device)

    def forward(self, x):
        # R : # rows
        # F, D : # fields
        # I : input embedding size
        # O : attention output size
        # H : # heads
        num_rows, num_fields, _ = x.shape  # R, F, I
        num_head = self.num_head           # H
        output_dim = self.output_dim       # O

        # (R, F, I) * (I, O * H) -> (R, F, O * H)
        qrys = torch.tensordot(x, self.w_act(self.W_q), dims=([-1], [0]))
        keys = torch.tensordot(x, self.w_act(self.W_k), dims=([-1], [0]))
        vals = torch.tensordot(x, self.w_act(self.W_v), dims=([-1], [0]))
        if self.use_residual:
            rsdl = torch.tensordot(x, self.w_act(self.W_r), dims=([-1], [0]))

        qrys = qrys.reshape((num_rows, num_fields, output_dim, num_head))  # (R, F, O, H)
        keys = keys.reshape((num_rows, num_fields, output_dim, num_head))
        vals = vals.reshape((num_rows, num_fields, output_dim, num_head))

        product = torch.einsum("rdoh,rfoh->rdfh", qrys, keys)  # (R, F, F, H)

        alpha = F.softmax(product, dim=2)  # (R, F, F, H)
        alpha = self.dropout(alpha)

        # (R, F, F, H) * (R, F, O, H) -> (R, F, O, H)
        out = torch.einsum("rfdh,rfoh->rfoh", alpha, vals)
        out = out.reshape((num_rows, num_fields, -1))  # (R, F, O * H)
        if self.use_residual:
            mix = torch.sigmoid(self.mix)
            out = mix * out + (1 - mix) * rsdl  # (R, F, O * H)
        out = F.relu(out)
        out = self.layer_norm(out)

        return out


class AttnInteractionBlock(nn.Module):

    def __init__(
        self,
        task: str,
        input_dim: int,
        attn_output_dim: int,
        output_dim: int,
        feature_dim: int,
        num_layers: int,
        num_head: int,
        activation: Optional[Type[nn.Module]],
        use_residual: bool = True,
        dropout: float = 0.0,
        normalize: bool = True,
        leaky_gate: bool = True,
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__()

        self.layers = nn.ModuleList()

        for _ in range(num_layers):
            self.layers.append(
                AttnInteractionLayer(
                    input_dim,
                    attn_output_dim,
                    num_head,
                    activation,
                    use_residual,
                    dropout,
                    normalize,
                    device,
                )
            )
            input_dim = attn_output_dim * num_head

        if leaky_gate:
            self.layers.append(LeakyGate(feature_dim * attn_output_dim * num_head))

        self.final = nn.Linear(
            feature_dim * attn_output_dim * num_head,
            output_dim,
            bias=(task != "classification"),
        )

        self.to(device)

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        out = out.reshape(out.shape[0], -1)
        out = self.final(out)
        return out


class AutoInt(BaseNN):

    def __init__(
        self,
        task: str,
        output_size: int,
        embedding_num: Optional[EmbeddingBase],
        embedding_cat: Optional[EmbeddingBase],
        attn_embedding_size: int = 8,
        attn_num_layer: int = 3,
        attn_num_head: int = 2,
        attn_activation: Optional[Type[nn.Module]] = None,
        attn_use_residual: bool = True,
        attn_dropout: float = 0.1,
        attn_normalize: bool = True,
        mlp_hidden_sizes: Union[int, Tuple[int, ...], List[int]] = (512, 256, 128, 64),
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
        device = torch.device(device)
        embed_info = check_uniform_embeddings(embedding_num, embedding_cat)

        self.attn_interact = AttnInteractionBlock(
            task=task,
            input_dim=embed_info.embedding_size,
            attn_output_dim=attn_embedding_size,
            output_dim=output_size,
            feature_dim=embed_info.num_fields,
            num_layers=attn_num_layer,
            num_head=attn_num_head,
            activation=attn_activation,
            use_residual=attn_use_residual,
            dropout=attn_dropout,
            normalize=attn_normalize,
            leaky_gate=mlp_leaky_gate,
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
                dropout_first=True,
                use_bn=mlp_use_bn,
                bn_momentum=mlp_bn_momentum,
                leaky_gate=mlp_leaky_gate,
                use_skip=mlp_use_skip,
                device=device,
            )
            if weighted_sum:
                self.mix = nn.Parameter(torch.tensor([0.0]))
            else:
                self.mix = torch.tensor([0.0])
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
        X_num ─ Lin. embedding ┐ ┌─ Attn Int ─ ... ─ Attn Int ─ Linear ─┐
                               ├─┤                                      w+ ── output
        X_cat ─ Cat. embedding ┘ └──────────────── MLP ─────────────────┘

        if no mlp_hidden_sizes
        ----------------------
        X_num ─ Lin. embedding ┬─ Attn Int ─ ... ─ Attn Int ─ Linear ─ output
        X_cat ─ Cat. embedding ┘ 

        splits are copies and joins are concatenations;
        'w+' is weighted element-wise addition;
        "Attn Int" is AutoInt's AttentionInteractionLayer
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
        return self.mlp.weight_sum()

    def forward(self, X_num, X_cat):
        embedded = torch.cat(self.embed(X_num, X_cat), dim=1)
        out = self.attn_interact(embedded)
        if self.mlp is not None:
            embedded_2d = embedded.reshape((embedded.shape[0], -1))
            mix = torch.sigmoid(self.mix)
            out = mix * out + (1 - mix) * self.mlp(embedded_2d)
        return out

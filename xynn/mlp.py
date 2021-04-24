"""
Module for MLP (multi-layer perceptron) and related modules

"""


from typing import Union, Tuple, List, Type

import torch
from torch import Tensor
from torch import nn
from entmax import sparsemax, entmax15


class LeakyGate(nn.Module):
    """
    This performs an element-wise linear transformation followed by a chosen
    activation; the default activation is nn.LeakyReLU. Fields may be
    represented by individual values or vectors of values (i.e., embedded).

    Input needs to be shaped like (num_rows, num_fields) or
    (num_rows, num_fields, embedding_size)

    """

    def __init__(
        self,
        input_size: int,
        bias: bool = True,
        activation: Type[nn.Module] = nn.LeakyReLU,
    ):
        """
        Parameters
        ----------
        input_size : int
        bias : boolean, optional
            whether to include an additive bias; default is True
        activation : torch.nn.Module, optional
            default is nn.LeakyReLU

        """
        super().__init__()
        self.weight = nn.Parameter(torch.normal(mean=0, std=1.0, size=(1, input_size)))
        self.bias = nn.Parameter(torch.zeros(size=(1, input_size)), requires_grad=bias)
        self.activation = activation()

    def forward(self, X: Tensor) -> Tensor:
        """
        Transform the input tensor

        Parameters
        ----------
        X : torch.Tensor

        Return
        ------
        torch.Tensor

        """
        out = X
        if len(X.shape) > 2:
            out = out.reshape((X.shape[0], -1))
        out = out * self.weight + self.bias
        if len(X.shape) > 2:
            out = out.reshape(X.shape)
        out = self.activation(out)
        return out


class MLP(nn.Module):
    """
    A "multi-layer perceptron". This forms layes of fully-connected linear
    transformations, with opional batch norm, dropout, and an initial
    "leaky gate".

    Input should be shaped like (num_rows, num_fields)

    """

    def __init__(
        self,
        task: str,
        input_size: int,
        hidden_sizes: Union[int, Tuple[int, ...], List[int]],
        output_size: int,
        activation: Type[nn.Module] = nn.LeakyReLU,
        dropout: Union[float, Tuple[float], List[float]] = 0.0,
        dropout_first: bool = False,
        use_bn: bool = True,
        bn_momentum: float = 0.1,
        leaky_gate: bool = True,
        use_skip: bool = True,
        weighted_sum: bool = True,
        device: Union[str, torch.device] = "cpu",
    ):
        """
        Parameters
        ----------
        task : {"regression", "classification"}
        input_size : int
            the number of inputs into the first layer
        hidden_sizes : iterable of int
            intermediate sizes between `input_size` and `output_size`
        output_size : int
            the number of outputs from the last layer
        activation : subclass of torch.nn.Module (uninitialized), optional
            default is nn.LeakyReLU
        dropout : float or iterable of float
            should be between 0.0 and 1.0; if iterable of float, there
            should be one value for each hidden size, plus an additional
            value if `use_bn` is True
        dropout_first : boolean, optional
            whether to include dropout before the first fully-connected
            linear layer (and after "leaky_gate", if using);
            default is False
        use_bn : boolean, optional
            whether to use batch normalization; default is True
        bn_momentum : float, optional
            default is 0.1
        leaky_gate : boolean, optional
            whether to include a LeakyGate layer before the linear layers;
            default is True
        use_skip : boolean, optional
            use a side path containing just the optional leaky gate plust
            a single linear layer; default is True
        weighted_sum : boolean, optional
            only used with use_skip; when adding main MLP output with side
            "skip" output, use a weighted sum with learnable weight; default is True
        device : string or torch.device, optional
            default is "cpu"

        """
        super().__init__()

        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]

        dropout_len = len(hidden_sizes) + (1 if dropout_first else 0)

        if isinstance(dropout, float):
            dropout = [dropout] * dropout_len
        elif not len(dropout) == dropout_len:
            raise ValueError(
                f"expected a single dropout value or {dropout_len} values "
                f"({'one more than' if dropout_first else 'same as'} hidden_sizes)"
            )

        main_layers: List[nn.Module] = []

        if leaky_gate:
            main_layers.append(LeakyGate(input_size))

        if dropout_first and dropout[0] > 0:
            main_layers.append(nn.Dropout(dropout[0]))
            dropout = dropout[1:]

        input_size_i = input_size
        for hidden_size_i, dropout_i in zip(hidden_sizes, dropout):
            main_layers.append(nn.Linear(input_size_i, hidden_size_i, bias=(not use_bn)))
            if use_bn:
                main_layers.append(nn.BatchNorm1d(hidden_size_i, momentum=bn_momentum))
            main_layers.append(activation())
            if dropout_i > 0:
                main_layers.append(nn.Dropout(dropout_i))
            input_size_i = hidden_size_i

        main_layers.append(
            nn.Linear(input_size_i, output_size, bias=(task != "classification"))
        )

        self.main_layers = nn.Sequential(*main_layers)

        self.use_skip = use_skip
        if use_skip:
            skip_linear = nn.Linear(input_size, output_size, bias=(task != "classification"))
            if leaky_gate:
                self.skip_layers = nn.Sequential(LeakyGate(input_size), skip_linear)
            else:
                self.skip_layers = skip_linear
            if weighted_sum:
                self.mix = nn.Parameter(torch.tensor([0.0]))
            else:
                self.mix = torch.tensor([0.0])
        else:
            self.skip_layers = None
            self.mix = None

        self.to(device)

    def weight_sum(self) -> Tuple[float, float]:
        """
        Sum of absolute value and squared weights, for regularization

        Return
        ------
        w1 : float
            sum of absolute value of weights
        w2 : float
            sum of squared weights

        """
        w1_sum = 0.0
        w2_sum = 0.0
        for layer_group in (self.main_layers, self.skip_layers):
            if layer_group is None:
                continue
            for layer in layer_group:
                if not isinstance(layer, nn.Linear):
                    continue
                w1_sum += layer.weight.abs().sum().item()
                w2_sum += (layer.weight ** 2).sum().item()
        return w1_sum, w2_sum

    def forward(self, X: Tensor) -> Tuple[float, float]:
        """
        Transform the input tensor

        Parameters
        ----------
        X : torch.Tensor

        Return
        ------
        torch.Tensor

        """
        out = self.main_layers(X)
        if self.use_skip:
            mix = torch.sigmoid(self.mix)
            skip_out = self.skip_layers(X)
            out = mix * skip_out + (1 - mix) * out
        return out

"""
Classes for embedding numeric fields

LinearEmbedding
  - embed each numeric *field* with a vector; for each numeric value, multiply the
    field vector by the value
DenseEmbedding
  - a dense linear layer followed by an activation

"""

from typing import Union, List, Optional, Tuple, Type
from functools import reduce
import operator

import torch
from torch import Tensor
from torch import nn

from ..common import _isnan
from .base import UniformBase


class LinearEmbedding(UniformBase):
    """
    An embedding for numeric fields. There is one embedded vector for each field.
    The embedded vector for a value is that value times its field's vector.

    """

    def __init__(self, embedding_size: int = 10, device: Union[str, torch.device] = "cpu"):
        """
        Parameters
        ----------
        embedding_size : int, optional
            size of each value's embedding vector; default is 10
        device : string or torch.device

        """
        super().__init__()
        self.num_fields = 0
        self.output_size = 0
        self.embedding: Optional[nn.Embedding] = None
        self.embedding_size = embedding_size
        self._device = device
        self.to(device)
        self._isfit = False

    def __repr__(self):
        return f"LinearEmbedding({self.embedding_size}, {repr(self._device)})"

    def from_values(self, num_fields: int):
        """
        Create the embedding for the given number of fields

        Parameters
        ----------
        num_fields : int

        Return
        ------
        self

        """
        self.num_fields = num_fields
        self.output_size = num_fields * self.embedding_size
        self.embedding = nn.Embedding(num_fields, self.embedding_size).to(device=self._device)
        nn.init.xavier_uniform_(self.embedding.weight)

        self._isfit = True

        return self

    def _fit_array(self, X):
        self.from_values(X.shape[1])

    def _fit_iterable(self, X):
        for batch in X:
            self._fit_array(batch)
            break

    def forward(self, X: Tensor) -> Tensor:
        """
        Produce embedding for each value in input

        Parameters
        ----------
        X : torch.Tensor

        Return
        ------
        torch.Tensor

        """
        if not self._isfit:
            raise RuntimeError("need to call `fit` or `from_values` first")
        return self.embedding.weight * X.unsqueeze(dim=-1)


class DenseEmbedding(UniformBase):
    """
    An embedding for numeric fields, consisting of just a linear transformation with
    an activation. Maps an input with shape n_rows * n_fields to an output with shape
    n_rows * 1 * embedding_size if one value passed for embedding_size or
    n_rows * embeddin_size[0] * embedding_size[1] if two values are passed

    """

    def __init__(
        self,
        embedding_size: Union[int, Tuple[int, ...], List[int]] = 10,
        activation: Type[nn.Module] = nn.LeakyReLU,
        device: Union[str, torch.device] = "cpu",
    ):
        """
        Parameters
        ----------
        embedding_size : int, tuple of ints, or list of ints; optional
            size of each value's embedding vector; default is 10
        activation : subclass of torch.nn.Module, optional
            default is nn.LeakyReLU
        device : string or torch.device

        """
        super().__init__()

        if isinstance(embedding_size, int):
            embedding_size = (1, embedding_size)
        elif len(embedding_size) == 1:
            embedding_size = (1, embedding_size[0])

        self.num_fields = 0
        self.output_size = 0
        self.embedding_w: Optional[nn.Parameter] = None
        self.embedding_b: Optional[nn.Parameter] = None
        self.embedding_size = embedding_size
        self.activation = activation().to(device=device)
        self._device = device
        self.to(device)
        self._isfit = False

    def __repr__(self):
        embed_size = self.embedding_size
        activation = self.activation.__class__.__name__
        device = repr(self._device)
        return f"DenseEmbedding({embed_size}, {activation}, {device})"

    def from_values(self, num_fields: int):
        """
        Create the embedding for the given number of fields

        Parameters
        ----------
        num_fields : int

        Return
        ------
        self

        """
        self.num_fields = num_fields
        self.output_size = reduce(operator.mul, self.embedding_size, 1)
        self.embedding_w = nn.Parameter(
            torch.zeros((num_fields, *self.embedding_size))
        ).to(device=self._device)
        self.embedding_b = nn.Parameter(
            torch.zeros(self.embedding_size)
        ).to(device=self._device)
        nn.init.xavier_uniform_(self.embedding_w)

        self._isfit = True

        return self

    def _fit_array(self, X):
        self.from_values(X.shape[1])

    def _fit_iterable(self, X):
        for batch in X:
            self._fit_array(batch)
            break

    def forward(self, X: Tensor) -> Tensor:
        """
        Produce embedding for each value in input

        Parameters
        ----------
        X : torch.Tensor

        Return
        ------
        torch.Tensor

        """
        if not self._isfit:
            raise RuntimeError("need to call `fit` or `from_values` first")
        embedded = self.embedding_w.T.matmul(X.T.to(dtype=torch.float)).T + self.embedding_b
        embedded = self.activation(embedded.reshape((X.shape[0], -1)))
        return embedded.reshape((X.shape[0], *self.embedding_size))

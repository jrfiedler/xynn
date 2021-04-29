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

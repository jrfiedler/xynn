"""
Classes for embedding categorical fields

FastBasicEmbedding
  - requires that each field's values are integers 0, 1, ...
  - embed each value with single vector
FastDefaultEmbedding
  - requires that each field's values are integers 0, 1, ...
  - like FastBasicEmbedding, but include a "default" vector for each field
  - returned vector is a weighted combination between the value's own vector
    and the field's "default" vector
  - the weighting is based on the count of value in the training set; a higher
    count puts more weight the value's own vector
  - values not seen in the training data are embedded with the default vector

"""

from typing import Any, Union, List, Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn, Tensor

from ..common import _isnan, FastBasicBase, FastDefaultBase
from .base import UniformBase


class FastBasicEmbedding(UniformBase, FastBasicBase):
    """
    A basic embedding that creates an embedded vector for each field value.

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
        embed_size = self.embedding_size
        device = repr(self._device)
        return f"FastBasicEmbedding({embed_size}, {device})"

    def from_summary(self, num_classes: List[int]) -> "FastBasicEmbedding":
        """
        Create the embedding from category values for each field

        Parameters
        ----------
        num_classes : list of int
            number of category values for each field

        Return
        ------
        self

        """
        self.num_fields = len(num_classes)
        self.output_size = self.num_fields * self.embedding_size
        self.offsets = torch.tensor([[0] + list(np.cumsum(num_classes[:-1]))], device=self._device)
        self.embedding = nn.Embedding(
            sum(num_classes), self.embedding_size
        ).to(device=self._device)
        nn.init.xavier_uniform_(self.embedding.weight)

        self._isfit = True

        return self

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
            raise RuntimeError("need to call `fit` or `from_summary` first")

        return self.embedding(X + self.offsets)


class FastDefaultEmbedding(UniformBase, FastDefaultBase):
    """
    An embedding with a default value for each field. The default is returned for
    any field value not seen when the embedding was initialized (using `fit` or
    `from_summary`). For any value seen at initialization, a weighted average of
    that value's embedding and the default embedding is returned. The weights for
    the average are determined by the parameter `alpha`:

    weight = count / (count + alpha)
    final = embedding * weight + default * (1 - weight)

    """

    def __init__(
        self,
        embedding_size: int = 10,
        alpha: int = 20,
        device: Union[str, torch.device] = "cpu",
    ):
        """
        Parameters
        ----------
        embedding_size : int, optional
            size of each value's embedding vector; default is 10
        alpha : int, optional
            controls the weighting of each embedding vector with the default;
            when `alpha`-many values are seen at initialization; the final
            vector is evenly weighted; the influence of the default is decreased
            with either higher counts or lower `alpha`; default is 20
        device : string or torch.device

        """
        super().__init__()
        #self.offsets
        self.embedding_size = embedding_size
        self.alpha = alpha
        self.num_fields = 0
        self.output_size = 0
        self.max_values: Optional[Tensor] = None
        self.offsets: Optional[Tensor] = None
        self.counts: Optional[Tensor] = None
        self.num_cat_vals = 0
        self.embedding: Optional[nn.Embedding] = None
        self._device = device
        self.to(device)
        self._isfit = False

    def __repr__(self):
        embed_size = self.embedding_size
        alpha = self.alpha
        device = repr(self._device)
        return f"FastDefaultEmbedding({embed_size}, {alpha}, {device})"

    def from_summary(self, class_counts: List[List[int]]) -> "FastDefaultEmbedding":
        """
        Create the embedding from known value counts for each field

        Parameters
        ----------
        class_counts : list of list of int
            each sub-list has count of category occurrences,
            one sub-list for each field

        Return
        ------
        self

        """
        num_fields = len(class_counts)
        num_uniques = [len(counts) for counts in class_counts]
        max_values = [x - 1 for x in num_uniques]
        offsets = [0] + list(np.cumsum(num_uniques[:-1]))
        num_embed = sum(num_uniques) + num_fields
        counts_flat = [count for field in class_counts for count in field]

        self.num_fields = num_fields
        self.output_size = self.num_fields * self.embedding_size
        self.max_values = torch.tensor(max_values, device=self._device).reshape((1, -1))
        self.offsets = torch.tensor(offsets, device=self._device)
        self.counts = torch.tensor(counts_flat).to(self._device)
        self.num_cat_vals = sum(num_uniques)
        self.embedding = nn.Embedding(num_embed, self.embedding_size).to(self._device)
        nn.init.xavier_uniform_(self.embedding.weight)

        self._isfit = True

        return self

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
            raise RuntimeError("need to call `fit` or `from_summary` first")

        offsets = self.offsets.expand(X.shape[0], self.offsets.shape[0])
        X_offset = X + offsets

        unxpcted = (X > self.max_values)
        X_offset[unxpcted] = offsets[unxpcted]  # block any unexpected categories

        counts = self.counts.expand(X_offset.shape[0], self.counts.shape[0])
        counts = torch.gather(counts, dim=1, index=X_offset)
        weights = (counts / (counts + self.alpha)).unsqueeze(-1)
        weights[unxpcted] = 0  # block any unexpected categories
        primary = self.embedding(X_offset)
        default = self.embedding.weight[self.num_cat_vals:, :].unsqueeze(0)
        return weights * primary + (1 - weights) * default

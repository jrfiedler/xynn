"""
Embeddings that allow embedding of multiple fields together and
allow a different vector size for each field

FastRaggedEmbedding
  - requires that each field's values are integers 0, 1, ...
  - embed each value with single vector
  - allows a different vector size for each field
FastRaggedDefaultEmbedding
  - requires that each field's values are integers 0, 1, ...
  - like RaggedEmbedding, but include a "default" vector for each field
  - returned vector is a weighted combination between the value's own vector
    and the field's "default" vector
  - the weighting is based on the count of value in the training set; a higher
    count puts more weight the value's own vector
  - values not seen in the training data are embedded with the default vector
  - allows a different vector size for each field

"""

from collections.abc import Iterable as IterableClass
from typing import Any, Union, List, Dict, Iterable, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch import nn

from ..common import _isnan
from ..common import FastBasicBase, FastDefaultBase
from .common import RaggedBase, _check_embedding_size, _parse_embedding_size


class FastRaggedEmbedding(RaggedBase, FastBasicBase):
    """
    Creates an embedded vector for each field value, with each field allowed
    a different size of embedding

    """

    def __init__(
        self,
        embedding_size: Union[str, int, Iterable[int]] = "sqrt",
        max_size: int = 100,
        device: Union[str, torch.device] = "cpu",
    ):
        """
        Parameters
        ----------
        embedding_size : {"sqrt", "log", "fastai"} or iterable of int; optional
            - "sqrt": square root of number of classes in each field, rounded up
            - "log": log of number of classes in each field, rounded up
            - "fastai": `round(1.6 * num_classes**0.56))`
            if iterable of int, the number of values must match the number of
            fields when calling `fit`; the embedding size can also be
            passed in later with `fit` or `from_summary`; default is "sqrt"
        max_size : int, optional
            maximum embedding size if using "sqrt", "log", or "fastai";
            default is 100
        device : string or torch.device, optional

        """
        super().__init__()
        embedding_size = _check_embedding_size(embedding_size)
        self.num_fields = 0
        self.output_size = 0
        self.num_classes: List[int] = []
        self.embedding: Optional[nn.ModuleList] = None
        self.embedding_size_orig = embedding_size
        self.embedding_size = embedding_size
        self.max_size = max_size
        self._device = device
        self.to(device)
        self._isfit = False

    def __repr__(self):
        embed_size = repr(self.embedding_size_orig)
        max_size = self.max_size
        device = repr(self._device)
        return f"FastRaggedEmbedding({embed_size}, {max_size}, {device})"

    def from_summary(self, num_classes: List[int]) -> "FastRaggedEmbedding":
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
        embedding_size = _parse_embedding_size(
            self.embedding_size, self.max_size, num_classes
        )

        self.embedding = nn.ModuleList([])
        for num_cats, size in zip(num_classes, embedding_size):
            embedding = nn.Embedding(num_cats, size).to(device=self._device)
            nn.init.xavier_uniform_(embedding.weight)
            self.embedding.append(embedding)

        self.num_fields = len(num_classes)
        self.output_size = sum(embedding_size)
        self.num_classes = num_classes
        self.embedding_size = embedding_size

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

        embedded = [
            embedding(column).reshape((X.shape[0], -1))
            for embedding, column in zip(self.embedding, X.split(1, dim=1))
        ]

        return torch.cat(embedded, dim=1)


class FastRaggedDefaultEmbedding(RaggedBase, FastDefaultBase):
    """
    An embedding with a default value for each field and which allows a different
    embedding size for each field. The default is returned for any field value
    not seen when the embedding was initialized (using `fit` or `from_summary`).
    For any value seen at initialization, a weighted average of that value's
    embedding and the default embedding is returned. The weights for the average
    are determined by the parameter `alpha`:

    weight = count / (count + alpha)
    final = embedding * weight + default * (1 - weight)

    """

    def __init__(
        self,
        embedding_size: Union[str, Iterable[int]] = "sqrt",
        max_size: int = 100,
        alpha: int = 20,
        device: Union[str, torch.device] = "cpu",
    ):
        """
        Parameters
        ----------
        embedding_size : {"sqrt", "log", "fastai"} or iterable of int; optional
            - "sqrt": square root of number of classes in each field, rounded up
            - "log": log of number of classes in each field, rounded up
            - "fastai": `round(1.6 * num_classes**0.56))`
            if iterable of int, the number of values must match the number of
            fields when calling `fit`; the embedding size can also be
            passed in later with `fit` or `from_summary`; default is "sqrt"
        max_size : int, optional
            maximum embedding size if using "sqrt", "log", or "fastai";
            default is 100
        alpha : int, optional
            controls the weighting of each embedding vector with the default;
            when `alpha`-many values are seen at initialization; the final
            vector is evenly weighted; the influence of the default is decreased
            with either higher counts or lower `alpha`; default is 20
        device : string or torch.device

        """
        super().__init__()
        embedding_size = _check_embedding_size(embedding_size)
        self.num_fields = 0
        self.output_size = 0
        self.alpha = alpha
        self.num_classes: List[int] = []
        self.embedding: Optional[nn.ModuleList] = None
        self.embedding_size_orig = embedding_size
        self.embedding_size = embedding_size
        self.max_size = max_size
        self._device = device
        self.to(device)
        self._isfit = False

    def __repr__(self):
        embed_size = repr(self.embedding_size_orig)
        max_size = self.max_size
        alpha = self.alpha
        device = repr(self._device)
        return f"FastRaggedDefaultEmbedding({embed_size}, {max_size}, {alpha}, {device})"

    def from_summary(self, class_counts: List[List[int]]) -> "FastRaggedDefaultEmbedding":
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
        num_classes = [len(counts) for counts in class_counts]

        embedding_size = _parse_embedding_size(
            self.embedding_size, self.max_size, num_classes
        )

        self.embedding = nn.ModuleList([])
        for num_cls, size in zip(num_classes, embedding_size):
            embedding = nn.Embedding(num_cls + 1, size).to(device=self._device)
            nn.init.xavier_uniform_(embedding.weight)
            self.embedding.append(embedding)

        self.num_fields = len(class_counts)
        self.output_size = sum(embedding_size)
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.max_values = torch.tensor([[x - 1 for x in num_classes]], device=self._device)
        self.counts = [
            torch.tensor(counts, device=self._device) for counts in class_counts
        ]

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

        embedded = []
        for embedding, counts, num_cls, X_col in zip(
            self.embedding, self.counts, self.num_classes, X.split(1, dim=1)
        ):
            unxpctd = (X_col >= num_cls)
            idx = torch.clone(X_col)
            idx[unxpctd] = 0    # block any unexpected categories

            counts = counts.expand(idx.shape[0], num_cls)
            counts = torch.gather(counts, dim=1, index=idx)
            weights = (counts / (counts + self.alpha)).unsqueeze(-1)
            weights[unxpctd] = 0  # block any unexpected categories

            primary = embedding(X_col)
            default = embedding.weight[num_cls:, :].unsqueeze(0)
            output = (weights * primary + (1 - weights) * default).reshape(X_col.shape[0], -1)
            embedded.append(output)

        return torch.cat(embedded, dim=1)

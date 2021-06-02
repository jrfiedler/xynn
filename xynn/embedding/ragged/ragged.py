"""
Embeddings that allow embedding of multiple fields together and
allow a different vector size for each field

RaggedEmbedding
  - embed each value with single vector
  - allows a different vector size for each field
RaggedDefaultEmbedding
  - like RaggedEmbedding, but include a "default" vector for each field
  - returned vector is a weighted combination between the value's own vector
    and the field's "default" vector
  - the weighting is based on the count of value in the training set; a higher
    count puts more weight the value's own vector
  - values not seen in the training data are embedded with the default vector
  - allows a different vector size for each field

"""

from math import ceil, sqrt, log
from collections.abc import Iterable as IterableClass
from typing import Any, Union, List, Dict, Iterable, Optional, Tuple

import numpy as np
import torch
from torch import nn, Tensor

from ..common import _isnan
from ..common import BasicBase, DefaultBase
from .common import RaggedBase, _check_embedding_size, _parse_embedding_size


class RaggedEmbedding(RaggedBase, BasicBase):
    """
    Creates an embedded vector for each field value, with each field allowed
    a different size of embedding

    """

    def __init__(
        self,
        embedding_size: Union[str, Iterable[int]] = "sqrt",
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
        self.lookup: Dict[Tuple[int, Any], int] = {}
        self.lookup_nan: Dict[int, int] = {}
        self.num_values: List[int] = []
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
        return f"RaggedEmbedding({embed_size}, {max_size}, {device})"

    def from_summary(
        self,
        uniques: List[Union[List, np.ndarray]],
        has_nan: List[bool],
    ) -> "RaggedEmbedding":
        """
        Create the embedding from category values for each field

        Parameters
        ----------
        uniques : list of array-like
            all possible category values for each field
        has_nan : list of boolean
            whether each field can have NaN

        Return
        ------
        self

        """
        if not len(uniques) == len(has_nan):
            raise ValueError(
                "length of uniques and has_nan should be equal, "
                f"got {len(uniques)}, {len(has_nan)}"
            )

        lookup = {}
        lookup_nan = {}
        num_values = [0] * len(uniques)
        for fieldnum, (field, use_nan) in enumerate(zip(uniques, has_nan)):
            for value in field:
                if (fieldnum, value) in lookup:
                    # extra defense against repeated values
                    continue
                lookup[(fieldnum, value)] = num_values[fieldnum]
                num_values[fieldnum] += 1
            if use_nan:
                lookup_nan[fieldnum] = num_values[fieldnum]
                num_values[fieldnum] += 1

        embedding_size = _parse_embedding_size(
            self.embedding_size, self.max_size, num_values
        )

        self.embedding = nn.ModuleList([])
        for num_cats, size in zip(num_values, embedding_size):
            embedding = nn.Embedding(num_cats, size).to(device=self._device)
            nn.init.xavier_uniform_(embedding.weight)
            self.embedding.append(embedding)

        self.num_fields = len(uniques)
        self.output_size = sum(embedding_size)
        self.lookup = lookup
        self.lookup_nan = lookup_nan
        self.num_values = num_values
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

        idxs: List[List[int]] = [[] for _ in range(self.num_fields)]
        for row in X:
            for col, val in enumerate(row):
                val = val.item()
                if _isnan(val):
                    idx = self.lookup_nan[col]
                else:
                    idx = self.lookup[(col, val)]
                idxs[col].append(idx)

        embedded = [
            embedding(torch.tensor(col_idxs, dtype=torch.int64, device=self._device))
            for embedding, col_idxs in zip(self.embedding, idxs)
        ]

        return torch.cat(embedded, dim=1)


class RaggedDefaultEmbedding(RaggedBase, DefaultBase):
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
        self.lookup: Dict[Tuple[int, Any], Tuple[int, int]] = {}
        self.lookup_nan: Dict[int, Tuple[int, int]] = {}
        self.lookup_default: Dict[int, Tuple[int, int]] = {}
        self.num_values: List[int] = []
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
        return f"RaggedDefaultEmbedding({embed_size}, {max_size}, {alpha}, {device})"

    def from_summary(
        self,
        unique_counts: List[Dict[Any, int]],
        nan_counts: List[int],
    ) -> "RaggedDefaultEmbedding":
        """
        Create the embedding from known value counts for each field

        Parameters
        ----------
        unique_counts : list of dicts
            each dict is a mapping from Python object to count of occurrences,
            one dict for each field
        nan_counts : list of int
            count of NaN occurrences for each field

        Return
        ------
        self

        """
        if not len(unique_counts) == len(nan_counts):
            raise ValueError(
                "length of unique_counts and nan_count should be equal, "
                f"got {len(unique_counts)}, {len(nan_counts)}"
            )

        lookup = {}
        lookup_nan = {}
        num_values = [0] * len(unique_counts)
        for fieldnum, (counts, nan_count) in enumerate(zip(unique_counts, nan_counts)):
            num_values[fieldnum] += 1  # default embedding is counted as the 0th value
            for value, count in counts.items():
                lookup[(fieldnum, value)] = (num_values[fieldnum], count)
                num_values[fieldnum] += 1
            if nan_count:
                lookup_nan[fieldnum] = (num_values[fieldnum], nan_count)
                num_values[fieldnum] += 1

        embedding_size = _parse_embedding_size(
            self.embedding_size, self.max_size, num_values
        )

        self.embedding = nn.ModuleList([])
        for num_cats, size in zip(num_values, embedding_size):
            embedding = nn.Embedding(num_cats, size).to(device=self._device)
            nn.init.xavier_uniform_(embedding.weight)
            self.embedding.append(embedding)

        self.num_fields = len(unique_counts)
        self.output_size = sum(embedding_size)
        self.lookup = lookup
        self.lookup_nan = lookup_nan
        self.num_values = num_values
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

        list_weights: List[List[List[float]]] = [[] for _ in range(self.num_fields)]
        idxs_primary: List[List[int]] = [[] for _ in range(self.num_fields)]
        for row in X:
            for col, val in enumerate(row):
                val = val.item()
                if _isnan(val):
                    idx, count = self.lookup_nan.get(col, (0, 0))
                else:
                    idx, count = self.lookup.get((col, val), (0, 0))
                list_weights[col].append([count / (count + self.alpha)])
                idxs_primary[col].append(idx)

        tsr_weights = torch.tensor(list_weights, dtype=torch.float32, device=self._device)
        embedded = []
        for embedding, col_w, idxs in zip(self.embedding, tsr_weights, idxs_primary):
            emb_primary = embedding(torch.tensor(idxs, dtype=torch.int64, device=self._device))
            emb_default = embedding(torch.tensor([0], device=self._device)).reshape((1, -1))
            embedded.append(col_w * emb_primary + (1 - col_w) * emb_default)

        return torch.cat(embedded, dim=1)

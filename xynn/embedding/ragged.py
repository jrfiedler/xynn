"""
Embeddings that allow embedding of multiple fields together and
allow a different vector size for each field

RaggedEmbedding
  - like uniform.BasicEmbedding, but allows a different vector size for each field
RaggedDefaultEmbedding
  - like uniform.DefaultEmbedding, but allows a different vector size for each field

"""

from math import ceil, sqrt, log
from collections.abc import Iterable as IterableClass
from typing import Any, Union, List, Dict, Iterable, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch import nn

from .common import _isnan
from .common import EmbeddingBase, BasicBase, DefaultBase


def _check_embedding_size(embedding_size, num_categories=None):
    """Check that given `embedding_size` makes sense for ragged embeddings"""
    if isinstance(embedding_size, str):
        if embedding_size not in ("sqrt", "log"):
            raise ValueError(
                "str embedding_size value must be one of {'sqrt', 'log'}; "
                f"got '{embedding_size}'"
            )
    elif not isinstance(embedding_size, IterableClass) or not all(
        isinstance(size, int) for size in embedding_size
    ):
        raise TypeError(f"embedding_size {repr(embedding_size)} not understood")
    elif num_categories is not None and len(embedding_size) != len(num_categories):
        raise ValueError(
            "number of embeddings must match number of fields, got "
            f"{len(embedding_size)} sizes and {len(num_categories)} fields"
        )


def _parse_embedding_size(embedding_size, num_categories) -> List[int]:
    """
    Parse given `embedding_size into` a list of individual sizes,
    for ragged embeddings
    """
    _check_embedding_size(embedding_size, num_categories)
    # calculate the individual values if "sqrt" or "log"
    if embedding_size == "sqrt":
        embedding_size = [int(ceil(sqrt(num_cat))) for num_cat in num_categories]
    elif embedding_size == "log":
        embedding_size = [
            int(ceil(log(num_cat))) if num_cat > 1 else 1 for num_cat in num_categories
        ]
    return embedding_size


class RaggedBase(EmbeddingBase):
    """Base class for embeddings that allow a different vector size for each field"""

    def __init__(self):
        super().__init__()
        self.embedding: Optional[nn.ModuleList] = None

    def weight_sum(self) -> Tuple[float, float]:
        """
        Sum of absolute value and square of embedding weights

        Return
        ------
        e1_sum : sum of absolute value of embedding values
        e2_sum : sum of squared embedding values
        """
        if not self._isfit:
            return 0.0, 0.0
        e1_sum = 0.0
        e2_sum = 0.0
        for embedding in self.embedding:
            e1_sum += embedding.weight.abs().sum().item()
            e2_sum += (embedding.weight ** 2).sum().item()
        return e1_sum, e2_sum


class RaggedEmbedding(RaggedBase, BasicBase):
    """
    Creates an embedded vector for each field value, with each field allowed
    a different size of embedding

    """

    def __init__(
        self,
        embedding_size: Union[str, Iterable[int]] = "sqrt",
        device: Union[str, torch.device] = "cpu",
    ):
        """
        Parameters
        ----------
        embedding_size : {"sqrt", "log"} or iterable of int; optional
            if "sqrt" each size will be the square root of the number of
            categories in each field, rounded up; "log" is likewise, except
            the log of number of categories; if iterable of int, the number
            of values must match the number of fields when calling size of
            each field's embedding vector; the embedding size can also be
            passed in later with `fit` or `from_values`; default is "sqrt"
        device : string or torch.device, optional

        """
        super().__init__()
        _check_embedding_size(embedding_size)
        self.num_fields = 0
        self.output_size = 0
        self.lookup: Dict[Tuple[int, Any], int] = {}
        self.lookup_nan: Dict[int, int] = {}
        self.num_values: List[int] = []
        self.embedding: Optional[nn.ModuleList] = None
        self.embedding_size = embedding_size
        self._device = device
        self.to(device)
        self._isfit = False

    def __repr__(self):
        return f"RaggedEmbedding({repr(self.embedding_size)}, {repr(self._device)})"

    def from_values(
        self,
        uniques: List[Union[List, np.ndarray]],
        has_nan: List[bool],
        embedding_size: Optional[Union[str, Iterable[int]]] = None,
    ):
        """
        Create the embedding from category values for each field

        Parameters
        ----------
        uniques : list of array-like
            all possible category values for each field
        has_nan : list of boolean
            whether each field can have NaN
        embedding_size : {"sqrt", "log"}, iterable of int, or None; optional
            if None, the value from initialization will be used; if "sqrt"
            each size will be the square root of the number of categories in
            each field, rounded up; "log" is likewise, except the log of
            number of categories; if iterable of int, the number of values
            must match the number of fields when calling size of each field's
            embedding vector; default is None

        Return
        ------
        self

        """
        if not len(uniques) == len(has_nan):
            raise ValueError(
                "length of uniques and has_nan should be equal, "
                f"got {len(uniques)}, {len(has_nan)}"
            )
        if embedding_size is None:
            embedding_size = self.embedding_size

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

        embedding_size = _parse_embedding_size(embedding_size, num_values)

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
            raise RuntimeError("need to call `fit` or `from_values` first")

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
    not seen when the embedding was initialized (using `fit` or `from_values`).
    For any value seen at initialization, a weighted average of that value's
    embedding and the default embedding is returned. The weights for the average
    are determined by the parameter `alpha`:

    weight = count / (count + alpha)
    final = embedding * weight + default * (1 - weight)

    """

    def __init__(
        self,
        embedding_size: Union[str, Iterable[int]] = "sqrt",
        alpha: int = 20,
        device: Union[str, torch.device] = "cpu",
    ):
        """
        Parameters
        ----------
        embedding_size : int
            size of each value's embedding vector
        alpha : int, optional
            controls the weighting of each embedding vector with the default;
            when `alpha`-many values are seen at initialization; the final
            vector is evenly weighted; the influence of the default is decreased
            with either higher counts or lower `alpha`; default is 20
        device : string or torch.device

        """
        super().__init__()
        _check_embedding_size(embedding_size)
        self.num_fields = 0
        self.output_size = 0
        self.alpha = alpha
        self.lookup: Dict[Tuple[int, Any], Tuple[int, int]] = {}
        self.lookup_nan: Dict[int, Tuple[int, int]] = {}
        self.lookup_default: Dict[int, Tuple[int, int]] = {}
        self.num_values: List[int] = []
        self.embedding: Optional[nn.ModuleList] = None
        self.embedding_size = embedding_size
        self._device = device
        self.to(device)
        self._isfit = False

    def __repr__(self):
        embed_size = repr(self.embedding_size)
        alpha = self.alpha
        device = repr(self._device)
        return f"RaggedDefaultEmbedding({embed_size}, {alpha}, {device})"

    def from_values(
        self,
        unique_counts: List[Dict[Any, int]],
        nan_counts: List[int],
        embedding_size: Optional[Union[str, Iterable[int]]] = None,
    ):
        """
        Create the embedding from known value counts for each field

        Parameters
        ----------
        unique_counts : list of dicts
            each dict is a mapping from Python object to count of occurrences,
            one dict for each field
        nan_counts : list of int
            count of NaN occurrences for each field
        embedding_size : {"sqrt", "log"}, iterable of int, or None; optional
            if None, the value from initialization will be used; if "sqrt"
            each size will be the square root of the number of categories in
            each field, rounded up; "log" is likewise, except the log of
            number of categories; if iterable of int, the number of values
            must match the number of fields when calling size of each field's
            embedding vector; default is None

        Return
        ------
        self

        """
        if not len(unique_counts) == len(nan_counts):
            raise ValueError(
                "length of unique_counts and nan_count should be equal, "
                f"got {len(unique_counts)}, {len(nan_counts)}"
            )
        if embedding_size is None:
            embedding_size = self.embedding_size

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

        embedding_size = _parse_embedding_size(embedding_size, num_values)

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
            raise RuntimeError("need to call `fit` or `from_values` first")

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

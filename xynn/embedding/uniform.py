"""
Embedding classes that allow embedding of multiple fields together,
with fixed vector size across fields

BasicEmbedding
  - embed each value with single vector
LinearEmbedding
  - embed each *field* with a vector; for each numeric value, multiply the
    field vector by the value
DefaultEmbedding
  - like BasicEmbedding, but include a "default" vector for each field
  - returned embedding vector for a value is a weighted combination between
    the value's own vector and the field's "default" vector
  - the weighting is based on the count of value in the training set; a higher
    count puts more weight the value's own vector
  - values not seen in the training data are embedded with the default vector

"""

from typing import Any, Union, List, Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch import nn

from .common import _isnan
from .common import EmbeddingBase, BasicBase, DefaultBase


class UniformBase(EmbeddingBase):
    """Base class for embeddings that have a single vector size for all fields"""

    def __init__(self):
        super().__init__()
        self.embedding: Optional[nn.Embedding] = None

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
        e1_sum = self.embedding.weight.abs().sum().item()
        e2_sum = (self.embedding.weight ** 2).sum().item()
        return e1_sum, e2_sum


class BasicEmbedding(UniformBase, BasicBase):
    """
    A basic embedding that creates an embedded vector for each field value.

    """

    def __init__(self, embedding_size: int, device: Union[str, torch.device] = "cpu"):
        """
        Parameters
        ----------
        embedding_size : int
            size of each value's embedding vector
        device : string or torch.device

        """
        super().__init__()
        self.num_fields = 0
        self.output_size = 0
        self.lookup: Dict[Tuple[int, Any], int] = {}
        self.lookup_nan: Dict[int, int] = {}
        self.num_values = 0
        self.embedding: Optional[nn.Embedding] = None
        self.embedding_size = embedding_size
        self._device = device
        self.to(device)
        self._isfit = False

    def from_values(
        self, uniques: List[Union[List, Tensor, np.ndarray]], has_nan: List[bool]
    ):
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
        num_values = 0
        for fieldnum, (field, use_nan) in enumerate(zip(uniques, has_nan)):
            for value in field:
                if (fieldnum, value) in lookup:
                    # extra defense against repeated values
                    continue
                lookup[(fieldnum, value)] = num_values
                num_values += 1
            if use_nan:
                lookup_nan[fieldnum] = num_values
                num_values += 1

        self.num_fields = len(uniques)
        self.output_size = self.num_fields * self.embedding_size
        self.lookup = lookup
        self.lookup_nan = lookup_nan
        self.num_values = num_values
        self.embedding = nn.Embedding(num_values, self.embedding_size).to(device=self._device)
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
            raise RuntimeError("need to call `fit` or `from_values` first")

        idxs: List[List[int]] = []
        for row in X:
            idxs.append([])
            for col, val in enumerate(row):
                val = val.item()
                if _isnan(val):
                    idx = self.lookup_nan[col]
                else:
                    idx = self.lookup[(col, val)]
                idxs[-1].append(idx)

        return self.embedding(torch.LongTensor(idxs, device=self._device))


class LinearEmbedding(UniformBase):
    """
    An embedding for numeric fields. There is one embedded vector for each field.
    The embedded vector for a value is that value times its field's vector.

    """

    def __init__(self, embedding_size: int, device: Union[str, torch.device] = "cpu"):
        """
        Parameters
        ----------
        embedding_size : int
            size of each value's embedding vector
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


class DefaultEmbedding(UniformBase, DefaultBase):
    """
    An embedding with a default value for each field. The default is returned for
    any field value not seen when the embedding was initialized (using `fit` or
    `from_values`). For any value seen at initialization, a weighted average of
    that value's embedding and the default embedding is returned. The weights for
    the average are determined by the parameter `alpha`:

    weight = count / (count + alpha)
    final = embedding * weight + default * (1 - weight)

    """

    def __init__(
        self, embedding_size: int, alpha: int, device: Union[str, torch.device] = "cpu"
    ):
        """
        Parameters
        ----------
        embedding_size : int
            size of each value's embedding vector
        alpha : int
            controls the weighting of each embedding vector with the default;
            when `alpha`-many values are seen at initialization; the final
            vector is evenly weighted; the influence of the default is decreased
            with either higher counts or lower `alpha`
        device : string or torch.device

        """
        super().__init__()
        self.num_fields = 0
        self.output_size = 0
        self.alpha = alpha
        self.lookup: Dict[Tuple[int, Any], Tuple[int, int]] = {}
        self.lookup_nan: Dict[int, Tuple[int, int]] = {}
        self.lookup_default: Dict[int, Tuple[int, int]] = {}
        self.num_values = 0
        self.embedding: Optional[nn.Embedding] = None
        self.embedding_size = embedding_size
        self._device = device
        self.to(device)
        self._isfit = False

    def from_values(self, unique_counts: List[Dict[Any, int]], nan_counts: List[int]):
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
                "length of unique_counts and nan_counts should be equal, "
                f"got {len(unique_counts)}, {len(nan_counts)}"
            )

        lookup = {}
        lookup_nan = {}
        lookup_default = {}
        num_values = 0
        for fieldnum, (counts, nan_count) in enumerate(zip(unique_counts, nan_counts)):
            lookup_default[fieldnum] = (num_values, 0)
            num_values += 1
            for value, count in counts.items():
                lookup[(fieldnum, value)] = (num_values, count)
                num_values += 1
            if nan_count:
                lookup_nan[fieldnum] = (num_values, nan_count)
                num_values += 1

        self.num_fields = len(unique_counts)
        self.output_size = self.num_fields * self.embedding_size
        self.lookup = lookup
        self.lookup_nan = lookup_nan
        self.lookup_default = lookup_default
        self.num_values = num_values
        self.embedding = nn.Embedding(num_values, self.embedding_size).to(device=self._device)
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
            raise RuntimeError("need to call `fit` or `from_values` first")

        list_weights: List[List[List[float]]] = []
        idxs_primary: List[List[int]] = []
        idxs_default: List[List[int]] = []
        for row in X:
            list_weights.append([])
            idxs_primary.append([])
            idxs_default.append([])
            for col, val in enumerate(row):
                val = val.item()
                default = self.lookup_default[col]
                if _isnan(val):
                    idx, count = self.lookup_nan.get(col, default)
                else:
                    idx, count = self.lookup.get((col, val), default)
                list_weights[-1].append([count / (count + self.alpha)])
                idxs_primary[-1].append(idx)
                idxs_default[-1].append(default[0])
        tsr_weights = torch.FloatTensor(list_weights)
        emb_primary = self.embedding(torch.LongTensor(idxs_primary, device=self._device))
        emb_default = self.embedding(torch.LongTensor(idxs_default, device=self._device))
        return tsr_weights * emb_primary + (1 - tsr_weights) * emb_default

"""
The base classes and common functions for embeddings

"""

from abc import ABCMeta, abstractmethod
from collections import defaultdict
from typing import Any, Union, List, Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch import nn
from torch.utils.data import DataLoader

from ..preprocessing import IntegerEncoder


def _isnan(value: Any) -> bool:
    return isinstance(value, float) and np.isnan(value)


def _isnan_index(series: pd.Series) -> np.ndarray:
    return np.array([_isnan(value) for value in series.index])


def _linear_agg(num_fields, empty_param, batch):
    if num_fields != 0:
        return num_fields, empty_param
    return batch.shape[1], empty_param


def _unique(
    X: Union[Tensor, np.ndarray, pd.DataFrame]
) -> Tuple[List[Iterable], List[bool]]:
    if isinstance(X, pd.DataFrame):
        uniques = [X[col].unique() for col in X.columns]
    elif isinstance(X, np.ndarray):
        uniques = [pd.unique(X[:, i]) for i in range(X.shape[1])]
    elif isinstance(X, Tensor):
        uniques = [torch.unique(X[:, i]).numpy() for i in range(X.shape[1])]
    else:
        raise TypeError(
            "input should be Pandas DataFrame, NumPy array, or PyTorch Tensor"
        )
    nan_chk = [np.array([_isnan(value) for value in group]) for group in uniques]
    has_nan = [np.any(check) for check in nan_chk]
    uniques = [group[~check] for group, check in zip(uniques, nan_chk)]
    return uniques, has_nan


def _unique_agg(uniques, has_nan, batch):
    for row in batch:
        for colnum, value in enumerate(row):
            value = value.item()
            if colnum >= len(uniques):
                uniques.append(set())
                has_nan.append(False)
            if _isnan(value):
                has_nan[colnum] = True
            else:
                uniques[colnum].add(value)
    return uniques, has_nan


def _value_counts(
    X: Union[Tensor, np.ndarray, pd.DataFrame]
) -> Tuple[List[Dict[Any, int]], List[int]]:
    if isinstance(X, (np.ndarray, pd.DataFrame)):
        if isinstance(X, pd.DataFrame):
            counts = [
                X[col].value_counts(dropna=False, ascending=True) for col in X.columns
            ]
        else:
            counts = [
                pd.value_counts(X[:, i], dropna=False, ascending=True)
                for i in range(X.shape[1])
            ]
        nan_check = [_isnan_index(count) for count in counts]
        nan_counts = [sum(count.loc[isnan]) for count, isnan in zip(counts, nan_check)]
        unique_counts = [
            count.loc[~isnan].to_dict() for count, isnan in zip(counts, nan_check)
        ]
    elif isinstance(X, Tensor):
        counts = [
            [values.numpy() for values in torch.unique(X[:, i], return_counts=True)]
            for i in range(X.shape[1])
        ]
        nan_check = [np.array([_isnan(val) for val in values]) for values, _ in counts]
        nan_counts = [np.sum(check) for check in nan_check]
        unique_counts = [
            dict(zip(vals[~check], counts[~check]))
            for (vals, counts), check in zip(counts, nan_check)
        ]
    else:
        raise TypeError(
            "input should be Pandas DataFrame, NumPy array, or PyTorch Tensor"
        )
    return unique_counts, nan_counts


def _value_counts_agg(unique_counts, nan_counts, batch):
    for row in batch:
        for colnum, value in enumerate(row):
            value = value.item()
            if colnum >= len(unique_counts):
                unique_counts.append(defaultdict(int))
                nan_counts.append(0)
            if _isnan(value):
                nan_counts[colnum] += 1
            else:
                unique_counts[colnum][value] += 1
    return unique_counts, nan_counts


def _flatten_counts(unique_counts: List[Dict[int, int]]) -> List[int]:
    counts = [
        [count.get(i, 0) for i in range(max(count) + 1)]
        for count in unique_counts
    ]
    return counts


class EmbeddingBase(nn.Module, metaclass=ABCMeta):
    """
    Base class for embeddings

    """

    def __init__(self):
        super().__init__()
        self._isfit = False

    @abstractmethod
    def _fit_array(self, X):
        return

    @abstractmethod
    def _fit_iterable(self, X):
        return

    def fit(self, X) -> "EmbeddingBase":
        """
        Create the embedding from training data

        Parameters
        ----------
        X : array-like or iterable of array-like
            should be a PyTorch Tensor, NumPy array, Pandas DataFrame
            or iterable of arrays/tensors (i.e., batches)

        Return
        ------
        self

        """
        if isinstance(X, (np.ndarray, Tensor, pd.DataFrame)):
            self._fit_array(X)
        elif isinstance(X, DataLoader):
            self._fit_iterable(X)
        else:
            raise TypeError(
                "input X must be a PyTorch Tensor, PyTorch DataLoader, "
                "NumPy array, or Pandas DataFrame"
            )

        self._isfit = True

        return self


class BasicBase(EmbeddingBase):
    """Base class for embeddings that do not have defaults"""

    @abstractmethod
    def from_summary(self, uniques, has_nan) -> "BasicBase":
        return self

    def _fit_array(self, X):
        uniques, has_nan = _unique(X)
        self.from_summary(uniques, has_nan)

    def _fit_iterable(self, X):
        uniques = []
        has_nan = []
        for batch in X:
            _unique_agg(uniques, has_nan, batch)
        self.from_summary(uniques, has_nan)


class DefaultBase(EmbeddingBase):
    """Base class for embeddings that have a default embedding for each field"""

    @abstractmethod
    def from_summary(self, unique_counts, nan_counts) -> "DefaultBase":
        return self

    def _fit_array(self, X):
        unique_counts, nan_counts = _value_counts(X)
        self.from_summary(unique_counts, nan_counts)

    def _fit_iterable(self, X):
        unique_counts = []
        nan_counts = []
        for batch in X:
            _value_counts_agg(unique_counts, nan_counts, batch)
        self.from_summary(unique_counts, nan_counts)


class FastBasicBase(EmbeddingBase):
    """Base class for embeddings that do not have defaults"""

    @abstractmethod
    def from_summary(self, num_classes: List[int]) -> "FastBasicBase":
        return self

    def from_encoder(self, encoder: IntegerEncoder) -> "FastBasicBase":
        if not isinstance(encoder, IntegerEncoder):
            raise TypeError("encoder needs to be a fit IntegerEncoder")
        if not encoder._isfit:
            raise ValueError("encoder needs to be fit")
        return self.from_summary(encoder.num_classes)

    def _fit_array(self, X):
        uniques, has_nan = _unique(X)
        if any(has_nan):
            raise ValueError("NaN found in categorical data")
        self.from_summary([max(col_uniques) + 1 for col_uniques in uniques])

    def _fit_iterable(self, X):
        uniques = []
        has_nan = []
        for batch in X:
            _unique_agg(uniques, has_nan, batch)
        if any(has_nan):
            raise ValueError("NaN found in categorical data")
        self.from_summary([max(col_uniques) + 1 for col_uniques in uniques])


class FastDefaultBase(EmbeddingBase):
    """Base class for embeddings that have a default embedding for each field"""

    @abstractmethod
    def from_summary(self, class_counts: List[List[int]]) -> "FastDefaultBase":
        return self

    def from_encoder(self, encoder: IntegerEncoder) -> "FastDefaultBase":
        if not isinstance(encoder, IntegerEncoder):
            raise TypeError("encoder needs to be a fit IntegerEncoder")
        if not encoder._isfit:
            raise ValueError("encoder needs to be fit")
        return self.from_summary(encoder.class_counts)

    def _fit_array(self, X):
        unique_counts, nan_counts = _value_counts(X)
        if any(nan_counts):
            raise ValueError("NaN found in categorical data")
        counts = _flatten_counts(unique_counts)
        self.from_summary(counts)

    def _fit_iterable(self, X):
        unique_counts = []
        nan_counts = []
        for batch in X:
            _value_counts_agg(unique_counts, nan_counts, batch)
            if any(nan_counts):
                raise ValueError("NaN found in categorical data")
        counts = _flatten_counts(unique_counts)
        self.from_summary(counts)

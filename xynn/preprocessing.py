"""
Data preprocessing

"""

from typing import Union, Any, Iterator, List

import torch
from torch import Tensor
import numpy as np
from sklearn.preprocessing import LabelEncoder


UTA = Union[Tensor, np.ndarray]


def _ismissing(column: UTA) -> UTA:
    if isinstance(column, Tensor):
        return torch.isnan(column)
    return np.isnan(column)


def _columns(X: UTA) -> Iterator[UTA]:
    """
    Split 2d input into 1d columns

    Parameters
    ----------
    X : NumPy array or PyTorch Tensor
        should be 2d

    Yields
    ------
    NumPy arrays or PyTorch Tensors

    """
    if isinstance(X, Tensor):
        columns = X.split(1, dim=1)
    else:
        columns = np.split(X, X.shape[1], axis=1)

    for column in columns:
        yield column.reshape((-1,))


def _isin(column: UTA, test_values: np.ndarray) -> UTA:
    if isinstance(column, Tensor):
        test_values = torch.from_numpy(test_values)
        return (column[..., None] == test_values).any(-1)
    return np.isin(column, test_values)


class IntegerEncoder:
    """
    Convert categorical inputs to integers.
    Input: 2d Tensor or NumPy array
    Output: 2d integer-valued Tensor

    """

    def __init__(self, unexpected="increment"):
        """
        Parameters
        ----------
        unexpected : {"increment", "raise"}, optional
            when encountering unexpected values in `transform`,
            whether to use a new label for them ("increment") or
            whether to raise an error ("raise"); default is "increment"

        """
        self.encoders: List[LabelEncoder] = []
        self.classes_: List[np.ndarray] = []
        self.nan_labels: List[int] = []
        self.num_classes: List[int] = []
        self.class_counts: List[List[int]] = []
        self._isfit = False
        self.unexpected = unexpected

    def fit(self, X: UTA, y: Any = None) -> "IntegerEncoder":
        """
        Fit encoder values from the input data

        Parameters
        ----------
        X : NumPy array or PyTorch Tensor
            should be 2d
        y : any, optional
            not used; parameter provided to imitate Scikit-learn transformers;
            default is None

        """
        for column in _columns(X):
            missing = _ismissing(column)
            encoder = LabelEncoder()
            encoder.fit(column[~missing])
            self.encoders.append(encoder)
            self.classes_.append(encoder.classes_)

            self.num_classes.append(len(encoder.classes_))
            self.class_counts.append(
                [(column == val).sum().item() for val in encoder.classes_]
            )
            num_missing = missing.sum().item()
            if num_missing:
                self.nan_labels.append(len(encoder.classes_))
                self.num_classes[-1] += 1
                self.class_counts[-1].append(int(missing.sum().item()))
            else:
                self.nan_labels.append(-1)

        self._isfit = True
        return self

    def _unexpected(self, column: UTA, col_idx: int) -> UTA:
        if not self._isfit:
            raise RuntimeError("encoder needs to be fit first")
        unexp = ~_isin(column, self.classes_[col_idx])
        if self.nan_labels[col_idx] != -1:
            unexp[_ismissing(column)] = False
        return unexp

    def transform(self, X: UTA, y: Any = None) -> Tensor:
        """
        Encode the input with integers

        Parameters
        ----------
        X : NumPy array or PyTorch Tensor
            should be 2d
        y : any, optional
            not used; parameter provided to imitate Scikit-learn transformers;
            default is None

        Returns
        -------
        PyTorch Tensor, with each column transformed to integers, from zero
        up to (not including) the number of classes in that column

        """
        if not self._isfit:
            raise RuntimeError("encoder needs to be fit first")
        if not X.shape[1] == len(self.encoders):
            raise ValueError(
                "input has the wrong shape, expected "
                f"{len(self.encoders)} columns, got {X.shape[1]}"
            )

        encoded_cols = []
        for col_idx, column in enumerate(_columns(X)):
            unxpctd = self._unexpected(column, col_idx)
            if unxpctd.sum() and self.unexpected == "raise":
                values = ", ".join(str(x) for x in column[unxpctd][:3])
                if unxpctd.sum() > 3:
                    values += ", ..."
                raise ValueError(f"unexpected values found in input: {values}")
            encoder = self.encoders[col_idx]
            missing = _ismissing(column)
            allgood = ~missing & ~unxpctd
            encoded = torch.empty(column.shape, dtype=torch.int64)
            encoded[allgood] = torch.from_numpy(encoder.transform(column[allgood]))
            encoded[missing] = self.nan_labels[col_idx]
            encoded[unxpctd] = self.num_classes[col_idx]
            encoded_cols.append(encoded.reshape((-1, 1)))
        return torch.cat(encoded_cols, dim=1)

    def fit_transform(self, X: UTA, y: Any = None) -> Tensor:
        """
        Fit encoder values and encode the input

        Parameters
        ----------
        X : NumPy array or PyTorch Tensor
            should be 2d
        y : any, optional
            not used; parameter provided to imitate Scikit-learn transformers;
            default is None

        Returns
        -------
        PyTorch Tensor, with each column transformed to integers, from zero
        up to (not including) the number of classes in that column

        """
        self.fit(X)
        return self.transform(X)

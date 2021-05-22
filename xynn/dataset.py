"""
Simple DataLoader-like class for tabular X_num, X_cat, y

"""

from typing import Union, Tuple, Optional

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


def _validate_x(X, y, X_name, device):
    if isinstance(X, (Tensor, np.ndarray)):
        if not X.shape[0] == y.shape[0]:
            raise ValueError(
                f"shape mismatch; got y.shape[0] == {y.shape[0]}, "
                f"{X_name}.shape[0] == {X.shape[0]}"
            )
        if len(X.shape) != 2:
            raise ValueError(
                f"{X_name} should be 2-d; got shape {X.shape}"
            )
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).to(device=device, dtype=torch.float32)
    elif X is None:
        X = torch.empty((y.shape[0], 0), device=device)
    else:
        raise TypeError(f"input {X_name} should be Tensor, NumPy array, or None")
    return X


def _validate_y(y, task, device):
    if isinstance(y, (Tensor, np.ndarray)):
        if any(size == 0 for size in y.shape):
            raise ValueError(f"y has a zero-sized dimension; got shape {y.shape}")

        if task == "regression" and len(y.shape) == 1:
            y = y.reshape((-1, 1))
        elif task == "classification" and len(y.shape) == 2:
            if y.shape[1] != 1:
                raise ValueError("for classification y must be 1-d or 2-d with one column")
            y = y.reshape((-1,))
        elif len(y.shape) > 2:
            raise ValueError(f"y has too many dimensions; got shape {y.shape}")

        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).to(device=device, dtype=torch.float32)
    else:
        raise TypeError("y should be Tensor or NumPy array")
    return y


class TabularDataLoader:
    """
    A DataLoader-like class that aims to be faster for tabular data.

    Based on `FastTensorDataLoader` by Jesse Mu
    https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6

    """
    def __init__(
        self,
        task: str,
        X_num: Optional[Union[np.ndarray, Tensor]],
        X_cat: Optional[Union[np.ndarray, Tensor]],
        y: Union[np.ndarray, Tensor],
        batch_size: int = 32,
        shuffle: bool = False,
        device: Union[str, torch.device] = "cpu",
    ):
        """
        Parameters
        ----------
        task : {"regression", "classification"}
        X_num : PyTorch Tensor, NumPy array, or None
            numeric input fields
        X_cat : PyTorch Tensor, NumPy array, or None
            categorical input fields (represented as numeric values)
        y : PyTorch Tensor, NumPy array, or None
            target field
        batch_size : int, optional
            default is 32
        shuffle : bool, optional
            default is False
        device : string or torch.device, optional
            default is "cpu"

        """
        if X_num is None and X_cat is None:
            raise TypeError("X_num and X_cat cannot both be None")

        self.y = _validate_y(y, task, device)
        self.X_num = _validate_x(X_num, self.y, "X_num", device)
        self.X_cat = _validate_x(X_cat, self.y, "X_cat", device)
        self.dataset_len = y.shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            self.indices = torch.randperm(self.dataset_len)
        else:
            self.indices = None
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        if self.indices is not None:
            indices = self.indices[self.i:self.i+self.batch_size]
            batch = (
                torch.index_select(self.X_num, 0, indices).to(device=self.device),
                torch.index_select(self.X_cat, 0, indices).to(device=self.device),
                torch.index_select(self.y, 0, indices).to(device=self.device),
            )
        else:
            batch = (
                self.X_num[self.i:self.i+self.batch_size].to(device=self.device),
                self.X_cat[self.i:self.i+self.batch_size].to(device=self.device),
                self.y[self.i:self.i+self.batch_size].to(device=self.device),
            )
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches

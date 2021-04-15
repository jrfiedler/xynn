"""
Simple Dataset class for tabular X_num, X_cat, y

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


class TabularDataset(Dataset):
    """
    Simple Dataset class for tabular X_num, X_cat, y

    """

    def __init__(
        self,
        X_num: Optional[Union[np.ndarray, Tensor]],
        X_cat: Optional[Union[np.ndarray, Tensor]],
        y: Union[np.ndarray, Tensor],
        task: str,
        device: Union[str, torch.device] = "cpu",
    ):
        """
        Parameters
        ----------
        X_num : PyTorch Tensor, NumPy array, or None
            numeric input fields
        X_cat : PyTorch Tensor, NumPy array, or None
            categorical input fields (represented as numeric values)
        y : PyTorch Tensor, NumPy array, or None
            target field
        task : {"regression", "classification"}
        device : string or torch.device, optional
            default is "cpu"

        """
        if X_num is None and X_cat is None:
            raise TypeError("X_num and X_cat cannot both be None")

        self.y = _validate_y(y, task, device)
        self.X_num = _validate_x(X_num, self.y, "X_num", device)
        self.X_cat = _validate_x(X_cat, self.y, "X_cat", device)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        return self.X_num[idx], self.X_cat[idx], self.y[idx]

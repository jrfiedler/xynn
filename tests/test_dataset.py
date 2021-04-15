
import numpy as np
import torch
import pytest

from xynn.dataset import _validate_x, _validate_y, TabularDataset


def test_that__validate_x_raises_error_with_wrong_type():
    X = list(range(100))
    y = torch.tensor(list(range(100)))
    with pytest.raises(TypeError, match="input X should be Tensor, NumPy array, or None"):
        _validate_x(X, y, "X", "cpu")


def test_that__validate_x_raises_error_with_wrong_size():
    X = np.array(list(range(99)))
    y = torch.tensor(list(range(100)))
    with pytest.raises(
        ValueError,
        match=r"shape mismatch; got y.shape\[0\] == 100, X.shape\[0\] == 99",
    ):
        _validate_x(X, y, "X", "cpu")


def test_that__validate_x_raises_error_with_wrong_shape():
    X = np.array(list(range(100)))
    y = torch.tensor(list(range(100)))
    with pytest.raises(ValueError, match=r"X should be 2-d; got shape \(100,\)"):
        _validate_x(X, y, "X", "cpu")

    X = np.array(list(range(100))).reshape((100, 1, 1))
    y = torch.tensor(list(range(100)))
    with pytest.raises(ValueError, match=r"X should be 2-d; got shape \(100, 1, 1\)"):
        _validate_x(X, y, "X", "cpu")


def test_that__validate_x_returns_empty_tensor_when_given_none():
    y = torch.tensor(list(range(100)))
    X = _validate_x(None, y, "X example", "cpu")
    assert X.shape == (100, 0)


def test__validate_x_with_numpy_input():
    X = np.array(list(range(100))).reshape((100, 1))
    y = torch.tensor(list(range(100)))
    X_out = _validate_x(X, y, "X", "cpu")
    assert all(X[i, 0].item() == X_out[i, 0].item() for i in range(100))


def test__validate_x_with_tensor_input():
    X = torch.tensor([[i, i + 1] for i in range(100)])
    y = torch.tensor(list(range(100)))
    X_out = _validate_x(X, y, "X", "cpu")
    assert X_out is X


def test_that__validate_y_raises_error_with_wrong_type():
    with pytest.raises(TypeError, match="y should be Tensor or NumPy array"):
        _validate_y(None, task="regression", device="cpu")


def test_that__validate_y_raises_error_with_wrong_size():
    y = torch.tensor([])
    with pytest.raises(
        ValueError,
        match=r"y has a zero-sized dimension; got shape torch.Size\(\[0\]\)"
    ):
        _validate_y(y, task="classification", device="cpu")


def test_that__validate_y_raises_error_with_wrong_shape():
    y = torch.tensor(list(range(100))).reshape((25, 4))
    with pytest.raises(
        ValueError,
        match="for classification y must be 1-d or 2-d with one column"
    ):
        _validate_y(y, task="classification", device="cpu")

    y = torch.tensor(list(range(100))).reshape((100, 1, 1))
    with pytest.raises(
        ValueError,
        match=r"y has too many dimensions; got shape torch.Size\(\[100, 1, 1\]\)",
    ):
        _validate_y(y, task="regression", device="cpu")


def test__validate_y_with_numpy_input():
    y = np.array(list(range(100)), dtype="int").reshape((100, 1))
    y_out = _validate_y(y, task="regression", device="cpu")
    assert all(y[i, 0].item() == y_out[i, 0].item() for i in range(100))
    assert y_out.shape == (100, 1)
    y_out = _validate_y(y, task="classification", device="cpu")
    assert y_out.shape == (100,)


def test__validate_y_with_tensor_input():
    y = torch.tensor(list(range(100)))
    y_out = _validate_y(y, task="regression", device="cpu")
    assert all(y[i].item() == y_out[i, 0].item() for i in range(100))
    assert y_out.shape == (100, 1)
    y_out = _validate_y(y, task="classification", device="cpu")
    assert y_out is y

    y = torch.tensor(list(range(100))).reshape((100, 1))
    y_out = _validate_y(y, task="regression", device="cpu")
    assert y_out is y


def test_that_TabularDataset_raises_error_when_both_Xs_are_None():
    y = np.array([0, 1] * 10)
    with pytest.raises(TypeError, match="X_num and X_cat cannot both be None"):
        TabularDataset(None, None, y, task="regression")


def test_TabularDataset_with_numpy_input():
    X_num = np.array([[i - 1, i, i + 1] for i in range(20, 0, -1)])
    X_cat = np.array([[i + j for j in range(5)] for i in range(20)])
    y = np.array([0, 1] * 10)

    dataset = TabularDataset(X_num, X_cat, y, task="regression")
    assert len(dataset) == 20

    data_pt = dataset[16]
    assert isinstance(data_pt, tuple)
    assert len(data_pt) == 3
    assert data_pt[0].shape == (3,)
    assert data_pt[1].shape == (5,)
    assert data_pt[2].shape == (1,)
    assert all(x1 == x2 for x1, x2 in zip(data_pt[0], X_num[16]))
    assert all(x1 == x2 for x1, x2 in zip(data_pt[1], X_cat[16]))
    assert data_pt[2][0].item() == y[16].item()


def test_TabularDataset_with_tensor_input():
    X_num = torch.tensor([[i - 1, i, i + 1] for i in range(20, 0, -1)])
    y = torch.tensor([0, 1] * 10)

    dataset = TabularDataset(X_num, None, y, task="classification")
    assert len(dataset) == 20

    data_pt = dataset[16]
    assert isinstance(data_pt, tuple)
    assert len(data_pt) == 3
    assert data_pt[0].shape == (3,)
    assert data_pt[1].shape == (0,)
    assert data_pt[2].shape == ()
    assert all(x1 == x2 for x1, x2 in zip(data_pt[0], X_num[16]))
    assert data_pt[2].item() == y[16].item()

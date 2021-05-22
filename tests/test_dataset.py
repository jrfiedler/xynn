
import numpy as np
import torch
import pytest

from xynn.dataset import _validate_x, _validate_y, TabularDataLoader


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


def test_that_TabularDataLoader_raises_error_when_both_Xs_are_None():
    y = np.array([0, 1] * 10)
    with pytest.raises(TypeError, match="X_num and X_cat cannot both be None"):
        TabularDataLoader(task="regression", X_num=None, X_cat=None, y=y)


def test_TabularDataLoader_with_numpy_input():
    X_num = np.array([[i - 1, i, i + 1] for i in range(20, 0, -1)])
    X_cat = np.array([[i + j for j in range(5)] for i in range(20)])
    y = np.array([0, 1] * 10)

    loader = TabularDataLoader(task="regression", X_num=X_num, X_cat=X_cat, y=y)
    assert len(loader) == 1

    batch = next(iter(loader))
    assert isinstance(batch, tuple)
    assert len(batch) == 3
    # batch size is greater than 20
    assert torch.all(batch[0] == torch.from_numpy(X_num)).item()
    assert torch.all(batch[1] == torch.from_numpy(X_cat)).item()
    assert torch.all(batch[2] == torch.from_numpy(y).reshape((20, 1))).item()


def test_TabularDataLoader_with_shuffled_numpy_input():
    X_num = np.array([[i - 1, i, i + 1] for i in range(20, 0, -1)])
    X_cat = np.array([[i + j for j in range(5)] for i in range(20)])
    y = np.arange(20)

    loader = TabularDataLoader(
        task="regression", X_num=X_num, X_cat=X_cat, y=y, shuffle=True
    )
    assert len(loader) == 1

    batch = next(iter(loader))
    assert isinstance(batch, tuple)
    assert len(batch) == 3
    # batch size is greater than 20
    order = [int(x.item()) for x in batch[2].reshape((20,))]
    assert set(order) == set(y)
    assert order != list(y)
    assert torch.all(batch[0] == torch.from_numpy(X_num[order])).item()
    assert torch.all(batch[1] == torch.from_numpy(X_cat[order])).item()
    assert torch.all(batch[2] == torch.from_numpy(y[order]).reshape((20, 1))).item()


def test_TabularDataLoader_with_numpy_input_and_smaller_batches():
    X_num = np.array([[i - 1, i, i + 1] for i in range(20, 0, -1)])
    X_cat = np.array([[i + j for j in range(5)] for i in range(20)])
    y = np.arange(20)

    loader = TabularDataLoader(
        task="regression", X_num=X_num, X_cat=X_cat, y=y, shuffle=True, batch_size=10
    )
    assert len(loader) == 2

    batch_0, batch_1 = list(loader)
    assert all(len(t) == 10 for batch in (batch_0, batch_1) for t in batch)
    batch_cat = tuple(torch.cat([t_0, t_1], dim=0) for t_0, t_1 in zip(batch_0, batch_1))
    order = [int(x.item()) for x in batch_cat[2].reshape((20,))]
    assert set(order) == set(y)
    assert order != list(y)
    assert torch.all(batch_cat[0] == torch.from_numpy(X_num[order])).item()
    assert torch.all(batch_cat[1] == torch.from_numpy(X_cat[order])).item()
    assert torch.all(batch_cat[2] == torch.from_numpy(y[order]).reshape((20, 1))).item()


def test_TabularDataLoader_with_tensor_input():
    X_num = torch.tensor([[i - 1, i, i + 1] for i in range(20, 0, -1)])
    y = torch.tensor([0, 1] * 10)

    loader = TabularDataLoader(task="classification", X_num=X_num, X_cat=None, y=y)
    assert len(loader) == 1

    batch = next(iter(loader))
    assert isinstance(batch, tuple)
    assert len(batch) == 3
    # batch size is greater than 20
    assert torch.all(batch[0] == X_num).item()
    assert batch[1].shape == (20, 0)
    assert torch.all(batch[2] == y).item()

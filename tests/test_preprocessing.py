
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
import pytest

from xynn.preprocessing import IntegerEncoder, _ismissing, _columns, _isin


def test__ismissing_numpy():
    data = np.array([0, 1, np.nan, 3, 4, 5, np.nan, np.nan])
    missing = _ismissing(data)
    expected = np.array([False, False, True, False, False, False, True, True])
    assert np.all(missing == expected)

    data = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    missing = _ismissing(data)
    expected = np.array([False, False, False, False, False, False, False, False])
    assert np.all(missing == expected)


def test__ismissing_tensor():
    data = torch.tensor([0, 1, np.nan, 3, 4, 5, np.nan, np.nan])
    missing = _ismissing(data)
    expected = torch.tensor([False, False, True, False, False, False, True, True])
    assert torch.all(missing == expected).item()

    data = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
    missing = _ismissing(data)
    expected = torch.tensor([False, False, False, False, False, False, False, False])
    assert torch.all(missing == expected).item()


def test__columns_numpy():
    X = np.random.randint(-5, 5, (8, 6))
    assert len(list(_columns(X))) == 6
    assert all(
        isinstance(column, np.ndarray)
        and column.shape == (8,)
        and np.all(column == X[:, i])
        for i, column in enumerate(_columns(X))
    )


def test__columns_tensor():
    X = torch.randint(-5, 5, (8, 6))
    assert len(list(_columns(X))) == 6
    assert all(
        isinstance(column, torch.Tensor)
        and column.shape == (8,)
        and torch.all(column == X[:, i].reshape((-1,))).item()
        for i, column in enumerate(_columns(X))
    )


def test__isin_numpy():
    column = np.array([-5, -3, -1, 1, 3, 5])
    test_vals = np.array([-3, 1, 5])
    assert np.all(
        _isin(column, test_vals) == np.array([False, True, False, True, False, True])
    )


def test__isin_tensor():
    column = torch.tensor([-5, -3, -1, 1, 3, 5])
    test_vals = np.array([-3, 1, 5])
    assert torch.all(
        _isin(column, test_vals) == torch.tensor([False, True, False, True, False, True])
    ).item()


def test_that_integerencoder_must_be_fit_before_transforming():
    X = np.array(
        [
            [     1, -1,     -3, -4,     -3,      3],
            [     4, -3, np.nan,  0,      0,     -4],
        ]
    )
    encoder = IntegerEncoder()
    with pytest.raises(RuntimeError, match="encoder needs to be fit first"):
        encoder.transform(X)


def test_integerencoder_fit_numpy():
    X = np.array(
        [
            [     1, -1,     -3, -4,     -3,      3],
            [     4, -3, np.nan,  0,      0,     -4],
            [    -5,  3, np.nan, -1,     -3,      2],
            [     1,  4,     -5, -2, np.nan,      0],
            [    -1,  1,      2,  0,     -3,      2],
            [np.nan, -4,      2, -5,     -2,     -3],
            [     2,  2,      3,  0,      0, np.nan],
            [    -2, -1,     -2,  0,      3,      1],
        ]
    )
    encoder = IntegerEncoder()
    encoder.fit(X)
    expected_classes = [
        np.array([-5, -2, -1, 1, 2, 4]),
        np.array([-4, -3, -1, 1, 2, 3, 4]),
        np.array([-5, -3, -2, 2, 3]),
        np.array([-5, -4, -2, -1, 0]),
        np.array([-3, -2, 0, 3]),
        np.array([-4, -3, 0, 1, 2, 3]),
    ]
    assert len(encoder.encoders) == 6
    assert all(isinstance(skl_enc, LabelEncoder) for skl_enc in encoder.encoders)
    assert len(encoder.classes_) == 6
    for found, expected in zip(encoder.classes_, expected_classes):
        assert np.all(found == expected)
    assert encoder.nan_labels == [6, -1, 5, -1, 4, 6]
    assert encoder.num_classes == [7, 7, 6, 5, 5, 7]


def test_integerencoder_fit_tensor():
    X = torch.tensor(
        [
            [     1, -1,     -3, -4,     -3,      3],
            [     4, -3, np.nan,  0,      0,     -4],
            [    -5,  3, np.nan, -1,     -3,      2],
            [     1,  4,     -5, -2, np.nan,      0],
            [    -1,  1,      2,  0,     -3,      2],
            [np.nan, -4,      2, -5,     -2,     -3],
            [     2,  2,      3,  0,      0, np.nan],
            [    -2, -1,     -2,  0,      3,      1],
        ]
    )
    encoder = IntegerEncoder()
    encoder.fit(X)
    expected_classes = [
        np.array([-5, -2, -1, 1, 2, 4]),
        np.array([-4, -3, -1, 1, 2, 3, 4]),
        np.array([-5, -3, -2, 2, 3]),
        np.array([-5, -4, -2, -1, 0]),
        np.array([-3, -2, 0, 3]),
        np.array([-4, -3, 0, 1, 2, 3]),
    ]
    assert len(encoder.encoders) == 6
    assert all(isinstance(skl_enc, LabelEncoder) for skl_enc in encoder.encoders)
    assert len(encoder.classes_) == 6
    for found, expected in zip(encoder.classes_, expected_classes):
        assert np.all(found == expected)
    assert encoder.nan_labels == [6, -1, 5, -1, 4, 6]
    assert encoder.num_classes == [7, 7, 6, 5, 5, 7]


def test_that_integerencoder_must_be_fit_before_calling__unexpected():
    column = np.array([0, -1, 1, np.nan, 2, 6, np.nan])
    encoder = IntegerEncoder()
    with pytest.raises(RuntimeError, match="encoder needs to be fit first"):
        encoder._unexpected(column, 0)


def test_integerencoder__unexpected_numpy():
    X = np.array(
        [
            [     1, -1,     -3, -4,     -3,      3],
            [     4, -3, np.nan,  0,      0,     -4],
            [    -5,  3, np.nan, -1,     -3,      2],
            [     1,  4,     -5, -2, np.nan,      0],
            [    -1,  1,      2,  0,     -3,      2],
            [np.nan, -4,      2, -5,     -2,     -3],
            [     2,  2,      3,  0,      0, np.nan],
            [    -2, -1,     -2,  0,      3,      1],
        ]
    )
    encoder = IntegerEncoder().fit(X)

    column = np.array([0, -1, 1, np.nan, 2, 6, np.nan])

    compare = np.array([True, False, False, False, False, True, False])
    assert np.all(encoder._unexpected(column, 0) == compare)
    compare = np.array([True, False, False, True, False, True, True])
    assert np.all(encoder._unexpected(column, 1) == compare)
    compare = np.array([False, True, True, False, True, True, False])
    assert np.all(encoder._unexpected(column, 4) == compare)


def test_integerencoder__unexpected_tensor():
    X = torch.tensor(
        [
            [     1, -1,     -3, -4,     -3,      3],
            [     4, -3, np.nan,  0,      0,     -4],
            [    -5,  3, np.nan, -1,     -3,      2],
            [     1,  4,     -5, -2, np.nan,      0],
            [    -1,  1,      2,  0,     -3,      2],
            [np.nan, -4,      2, -5,     -2,     -3],
            [     2,  2,      3,  0,      0, np.nan],
            [    -2, -1,     -2,  0,      3,      1],
        ]
    )
    encoder = IntegerEncoder().fit(X)

    column = torch.tensor([0, -1, 1, np.nan, 2, 6, np.nan])

    compare = torch.tensor([True, False, False, False, False, True, False])
    assert torch.all(encoder._unexpected(column, 0) == compare).item()
    compare = torch.tensor([True, False, False, True, False, True, True])
    assert torch.all(encoder._unexpected(column, 1) == compare).item()
    compare = torch.tensor([False, True, True, False, True, True, False])
    assert torch.all(encoder._unexpected(column, 4) == compare).item()


def test_that_integerencoder_raises_error_with_wrong_size_input():
    X = np.array(
        [
            [     1, -1,     -3, -4,     -3,      3],
            [     4, -3, np.nan,  0,      0,     -4],
            [    -5,  3, np.nan, -1,     -3,      2],
            [     1,  4,     -5, -2, np.nan,      0],
        ]
    )
    encoder = IntegerEncoder()
    encoder.fit(X)
    with pytest.raises(
        ValueError, match="input has the wrong shape, expected 6 columns, got 4"
    ):
        encoder.transform(X[:, :4])


def test_integerencoder_transform_raises_error_for_unexpected_when_requested():
    X = np.array(
        [
            [     1, -1,     -3, -4,     -3,      3],
            [     4, -3, np.nan,  0,      0,     -4],
            [    -5,  3, np.nan, -1,     -3,      2],
            [     1,  4,     -5, -2, np.nan,      0],
        ]
    )
    encoder = IntegerEncoder(unexpected="raise").fit(X)
    X_test = np.array([[0, -1, 1, np.nan, 2, 6, np.nan]] * 6).T
    with pytest.raises(
        ValueError,
        match="unexpected values found in input: 0.0, -1.0, nan, ..."
    ):
        encoder.transform(X_test)


def test_integerencoder_transform_with_unexpected_numpy():
    X = np.array(
        [
            [     1, -1,     -3, -4,     -3,      3],
            [     4, -3, np.nan,  0,      0,     -4],
            [    -5,  3, np.nan, -1,     -3,      2],
            [     1,  4,     -5, -2, np.nan,      0],
            [    -1,  1,      2,  0,     -3,      2],
            [np.nan, -4,      2, -5,     -2,     -3],
            [     2,  2,      3,  0,      0, np.nan],
            [    -2, -1,     -2,  0,      3,      1],
        ]
    )
    encoder = IntegerEncoder().fit(X)

    X_test = np.array([[0, -1, 1, np.nan, 2, 6, np.nan]] * 6).T

    compare = torch.tensor(
        [
            [7, 2, 3, 6, 4, 7, 6],
            [7, 2, 3, 7, 4, 7, 7],
            [6, 6, 6, 5, 3, 6, 5],
            [4, 3, 5, 5, 5, 5, 5],
            [2, 5, 5, 4, 5, 5, 4],
            [2, 7, 3, 6, 4, 7, 6],
        ]
    ).T

    transformed = encoder.transform(X_test)
    assert torch.all(transformed == compare).item()
    assert transformed.dtype == torch.int64


def test_integerencoder_transform_with_unexpected_tensor():
    X = np.array(
        [
            [     1, -1,     -3, -4,     -3,      3],
            [     4, -3, np.nan,  0,      0,     -4],
            [    -5,  3, np.nan, -1,     -3,      2],
            [     1,  4,     -5, -2, np.nan,      0],
            [    -1,  1,      2,  0,     -3,      2],
            [np.nan, -4,      2, -5,     -2,     -3],
            [     2,  2,      3,  0,      0, np.nan],
            [    -2, -1,     -2,  0,      3,      1],
        ]
    )
    encoder = IntegerEncoder().fit(X)

    X_test = np.array([[0, -1, 1, np.nan, 2, 6, np.nan]] * 6).T

    compare = torch.tensor(
        [
            [7, 2, 3, 6, 4, 7, 6],
            [7, 2, 3, 7, 4, 7, 7],
            [6, 6, 6, 5, 3, 6, 5],
            [4, 3, 5, 5, 5, 5, 5],
            [2, 5, 5, 4, 5, 5, 4],
            [2, 7, 3, 6, 4, 7, 6],
        ]
    ).T

    transformed = encoder.transform(X_test)
    assert torch.all(transformed == compare).item()
    assert transformed.dtype == torch.int64


def test_integerencoder_fit_transform_numpy():
    X = np.array(
        [
            [     1, -1,     -3, -4,     -3,      3],
            [     4, -3, np.nan,  0,      0,     -4],
            [    -5,  3, np.nan, -1,     -3,      2],
            [     1,  4,     -5, -2, np.nan,      0],
            [    -1,  1,      2,  0,     -3,      2],
            [np.nan, -4,      2, -5,     -2,     -3],
            [     2,  2,      3,  0,      0, np.nan],
            [    -2, -1,     -2,  0,      3,      1],
        ]
    )
    encoder = IntegerEncoder()
    X_out = encoder.fit_transform(X)
    X_expected = torch.tensor(
        [
            [ 3, 2, 1, 1, 0, 5],
            [ 5, 1, 5, 4, 2, 0],
            [ 0, 5, 5, 3, 0, 4],
            [ 3, 6, 0, 2, 4, 2],
            [ 2, 3, 3, 4, 0, 4],
            [ 6, 0, 3, 0, 1, 1],
            [ 4, 4, 4, 4, 2, 6],
            [ 1, 2, 2, 4, 3, 3],
        ]
    )
    assert torch.all(X_out == X_expected).item()
    assert X_out.dtype == torch.int64


def test_integerencoder_fit_transform_tensor():
    X = torch.tensor(
        [
            [     1, -1,     -3, -4,     -3,      3],
            [     4, -3, np.nan,  0,      0,     -4],
            [    -5,  3, np.nan, -1,     -3,      2],
            [     1,  4,     -5, -2, np.nan,      0],
            [    -1,  1,      2,  0,     -3,      2],
            [np.nan, -4,      2, -5,     -2,     -3],
            [     2,  2,      3,  0,      0, np.nan],
            [    -2, -1,     -2,  0,      3,      1],
        ]
    )
    encoder = IntegerEncoder()
    X_out = encoder.fit_transform(X)
    X_expected = torch.tensor(
        [
            [ 3, 2, 1, 1, 0, 5],
            [ 5, 1, 5, 4, 2, 0],
            [ 0, 5, 5, 3, 0, 4],
            [ 3, 6, 0, 2, 4, 2],
            [ 2, 3, 3, 4, 0, 4],
            [ 6, 0, 3, 0, 1, 1],
            [ 4, 4, 4, 4, 2, 6],
            [ 1, 2, 2, 4, 3, 3],
        ]
    )
    assert torch.all(X_out == X_expected).item()
    assert X_out.dtype == torch.int64

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
import pytest

from xynn.base_classes.estimators import _set_seed
from xynn.embedding import BasicEmbedding, DefaultEmbedding
from ...common import simple_train_loop
from ..utils import example_data, Reshape, SimpleDataset


def test_that_basicembedding_must_be_fit():
    embedding = BasicEmbedding(embedding_size=2)
    data_test = pd.DataFrame(
        {
            "cat_a": [0, 1, 2, 3],
            "cat_b": [0, 1, 2, 0],
            "cat_c": [0, 1, np.nan, 0],
        }
    )
    msg = "need to call `fit` or `from_values` first"
    with pytest.raises(RuntimeError, match=msg):
        embedding(data_test.values)


def test_basicembedding_with_pandas_example():
    data_cat = example_data()[["cat_a", "cat_b", "cat_c"]]
    embedding = BasicEmbedding(embedding_size=2).fit(data_cat)
    data_test = pd.DataFrame(
        {
            "cat_a": [0, 1, 2, 3],
            "cat_b": [0, 1, 2, 0],
            "cat_c": [0, 1, np.nan, 0],
        }
    )
    output = embedding(data_test.values)
    weight = embedding.embedding.weight
    assert weight.shape == (10, 2)
    assert output.shape == (4, 3, 2)
    # test returned vectors vs weight matrix
    assert torch.all(output[:, 0, :] == weight[:4]).item()
    assert torch.all(output[:3, 1, :] == weight[4:7]).item()
    assert torch.all(
        torch.sort(output[:3, 2, :], dim=0)[0] == torch.sort(weight[7:], dim=0)[0]
    ).item()  # can't be sure of order when calculating `unique`


def test_basicembedding_with_numpy_example():
    data_cat = example_data()[["cat_a", "cat_b", "cat_c"]].values
    embedding = BasicEmbedding(embedding_size=2).fit(data_cat)
    data_test = pd.DataFrame(
        {
            "cat_a": [0, 1, 2, 3],
            "cat_b": [0, 1, 2, 0],
            "cat_c": [0, 1, np.nan, 0],
        }
    )
    output = embedding(data_test.values)
    weight = embedding.embedding.weight
    assert weight.shape == (10, 2)
    assert output.shape == (4, 3, 2)
    # test returned vectors vs weight matrix
    assert torch.all(output[:, 0, :] == weight[:4]).item()
    assert torch.all(output[:3, 1, :] == weight[4:7]).item()
    assert torch.all(
        torch.sort(output[:3, 2, :], dim=0)[0] == torch.sort(weight[7:], dim=0)[0]
    ).item()  # can't be sure of order when calculating `unique`


def test_basicembedding_with_tensor_example():
    data_cat = torch.from_numpy(
        example_data()[["cat_a", "cat_b", "cat_c"]].values
    )
    embedding = BasicEmbedding(embedding_size=2).fit(data_cat)
    data_test = pd.DataFrame(
        {
            "cat_a": [0, 1, 2, 3],
            "cat_b": [0, 1, 2, 0],
            "cat_c": [0, 1, np.nan, 0],
        }
    )
    output = embedding(data_test.values)
    weight = embedding.embedding.weight
    assert weight.shape == (10, 2)
    assert output.shape == (4, 3, 2)
    # test returned vectors vs weight matrix
    assert torch.all(output[:, 0, :] == weight[:4]).item()
    assert torch.all(output[:3, 1, :] == weight[4:7]).item()
    assert torch.all(output[:3, 2, :] == weight[7:]).item()


def test_basicembedding_with_dataloader_example():
    data_cat = torch.from_numpy(
        example_data()[["cat_a", "cat_b", "cat_c"]].values
    )
    dataset = SimpleDataset(data_cat)
    dataloader = DataLoader(dataset, batch_size=5)
    embedding = BasicEmbedding(embedding_size=2).fit(dataloader)
    data_test = pd.DataFrame(
        {
            "cat_a": [0, 1, 2, 3],
            "cat_b": [0, 1, 2, 0],
            "cat_c": [0, 1, np.nan, 0],
        }
    )
    output = embedding(data_test.values)
    weight = embedding.embedding.weight
    assert weight.shape == (10, 2)
    assert output.shape == (4, 3, 2)
    # test returned vectors vs weight matrix
    assert torch.all(output[:, 0, :] == weight[:4]).item()
    assert torch.all(output[:3, 1, :] == weight[4:7]).item()
    assert torch.all(output[:3, 2, :] == weight[7:]).item()


def test_basicembedding_weight_sum():
    data_cat = example_data()[["cat_a", "cat_b", "cat_c"]].values
    embedding = BasicEmbedding(embedding_size=4)
    assert embedding.weight_sum() == (0.0, 0.0)

    embedding.fit(data_cat)
    e1_sum, e2_sum = embedding.weight_sum()
    assert isinstance(e1_sum, float)
    assert isinstance(e2_sum, float)

    weight = embedding.embedding.weight
    embedding.embedding.weight = nn.Parameter(torch.ones(weight.shape))
    e1_sum, e2_sum = embedding.weight_sum()
    assert isinstance(e1_sum, float)
    assert isinstance(e2_sum, float)
    assert e1_sum == weight.shape[0] * weight.shape[1]
    assert e2_sum == weight.shape[0] * weight.shape[1]


def test_that_basicembedding_raises_error_with_wrong_fit_input():
    data_cat = example_data().to_dict()
    embedding = BasicEmbedding(embedding_size=2)
    msg = (
        "input X must be a PyTorch Tensor, PyTorch DataLoader, "
        "NumPy array, or Pandas DataFrame"
    )
    with pytest.raises(TypeError, match=msg):
        embedding.fit(data_cat)


def test_that_basicembedding_raises_error_with_bad_from_values_input():
    uniques = [["a", "b", "c", "d"], [0, 1, np.nan], [0, "a"]]
    has_nan = [False, True]
    embedding = BasicEmbedding(embedding_size=2)
    msg = "length of uniques and has_nan should be equal, got 3, 2"
    with pytest.raises(ValueError, match=msg):
        embedding.from_values(uniques, has_nan)


def test_that_basicembedding_learns():
    X = np.random.randint(0, 5, (100, 10))
    y = torch.rand((100, 3))
    embedding = BasicEmbedding(embedding_size=3).fit(X)
    model = nn.Sequential(embedding, Reshape(), nn.Linear(30, 3))
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(embedding.parameters(), lr=1e-1)  # not model.parameters()
    wt_before = torch.clone(embedding.embedding.weight)
    loss_vals = simple_train_loop(model, X, y, loss_func, optimizer, num_epochs=5)
    assert torch.all(embedding.embedding.weight != wt_before).item()
    assert loss_vals[0] > loss_vals[-1]


def test_that_defaultembedding_must_be_fit():
    embedding = DefaultEmbedding(embedding_size=2, alpha=2)
    data_test = pd.DataFrame(
        {
            "cat_a": [0, 1, 2, 3],
            "cat_b": [0, 1, 2, 0],
            "cat_c": [0, 1, np.nan, 0],
        }
    )
    msg = "need to call `fit` or `from_values` first"
    with pytest.raises(RuntimeError, match=msg):
        embedding(data_test.values)


def test_defaultembedding_with_pandas_example():
    data_cat = example_data()[["cat_a", "cat_b", "cat_c"]]
    embedding = DefaultEmbedding(embedding_size=3, alpha=2).fit(data_cat)
    data_test = pd.DataFrame(
        {
            "cat_a": [0, 1, 4],
            "cat_b": [0, 2, 3],
            "cat_c": [0, np.nan, 2],
        }
    )
    output = embedding(data_test.values)
    weight = embedding.embedding.weight
    assert weight.shape == (13, 3)
    assert output.shape == (3, 3, 3)
    # test returned vectors vs weight matrix
    # `any` is used because we can't be sure of the order
    assert any(
        torch.allclose(output[0, 0], (2 / 3) * weight[i] + (1 / 3) * weight[0])
        for i in (1, 2, 3, 4)
    )
    assert any(
        torch.allclose(output[0, 1], (2 / 3) * weight[i] + (1 / 3) * weight[5])
        for i in (6, 7, 8)
    )
    assert any(
        torch.allclose(output[0, 2], (3 / 5) * weight[i] + (2 / 5) * weight[9])
        for i in (10, 11, 12)
    )
    assert any(
        torch.allclose(output[1, 0], (3 / 5) * weight[i] + (2 / 5) * weight[0])
        for i in (1, 2, 3, 4)
    )
    assert any(
        torch.allclose(output[1, 1], (1 / 3) * weight[i] + (2 / 3) * weight[5])
        for i in (6, 7, 8)
    )
    assert any(
        torch.allclose(output[1, 2], (1 / 3) * weight[i] + (2 / 3) * weight[9])
        for i in (10, 11, 12)
    )
    assert torch.allclose(output[2, 0], weight[0])
    assert torch.allclose(output[2, 1], weight[5])
    assert torch.allclose(output[2, 2], weight[9])


def test_defaultembedding_with_numpy_example():
    data_cat = example_data()[["cat_a", "cat_b", "cat_c"]].values
    embedding = DefaultEmbedding(embedding_size=3, alpha=2).fit(data_cat)
    data_test = pd.DataFrame(
        {
            "cat_a": [0, 1, 4],
            "cat_b": [0, 2, 3],
            "cat_c": [0, np.nan, 2],
        }
    )
    output = embedding(data_test.values)
    weight = embedding.embedding.weight
    assert weight.shape == (13, 3)
    assert output.shape == (3, 3, 3)
    # test returned vectors vs weight matrix
    # `any` is used because we can't be sure of the order
    assert any(
        torch.allclose(output[0, 0], (2 / 3) * weight[i] + (1 / 3) * weight[0])
        for i in (1, 2, 3, 4)
    )
    assert any(
        torch.allclose(output[0, 1], (2 / 3) * weight[i] + (1 / 3) * weight[5])
        for i in (6, 7, 8)
    )
    assert any(
        torch.allclose(output[0, 2], (3 / 5) * weight[i] + (2 / 5) * weight[9])
        for i in (10, 11, 12)
    )
    assert any(
        torch.allclose(output[1, 0], (3 / 5) * weight[i] + (2 / 5) * weight[0])
        for i in (1, 2, 3, 4)
    )
    assert any(
        torch.allclose(output[1, 1], (1 / 3) * weight[i] + (2 / 3) * weight[5])
        for i in (6, 7, 8)
    )
    assert any(
        torch.allclose(output[1, 2], (1 / 3) * weight[i] + (2 / 3) * weight[9])
        for i in (10, 11, 12)
    )
    assert torch.allclose(output[2, 0], weight[0])
    assert torch.allclose(output[2, 1], weight[5])
    assert torch.allclose(output[2, 2], weight[9])


def test_defaultembedding_with_dataloader_example():
    _set_seed(68794)
    data_cat = example_data()[["cat_a", "cat_b", "cat_c"]].values
    dataset = SimpleDataset(data_cat)
    dataloader = DataLoader(dataset, batch_size=5)
    embedding = DefaultEmbedding(embedding_size=3, alpha=2).fit(dataloader)
    data_test = pd.DataFrame(
        {
            "cat_a": [0, 1, 4],
            "cat_b": [0, 2, 3],
            "cat_c": [0, np.nan, 2],
        }
    )
    output = embedding(data_test.values)
    weight = embedding.embedding.weight
    assert weight.shape == (13, 3)
    assert output.shape == (3, 3, 3)
    # test returned vectors vs weight matrix
    # `any` is used because we can't be sure of the order
    assert any(
        torch.allclose(output[0, 0], (2 / 3) * weight[i] + (1 / 3) * weight[0])
        for i in (1, 2, 3, 4)
    )
    assert any(
        torch.allclose(output[0, 1], (2 / 3) * weight[i] + (1 / 3) * weight[5])
        for i in (6, 7, 8)
    )
    assert any(
        torch.allclose(output[0, 2], (3 / 5) * weight[i] + (2 / 5) * weight[9])
        for i in (10, 11, 12)
    )
    assert any(
        torch.allclose(output[1, 0], (3 / 5) * weight[i] + (2 / 5) * weight[0])
        for i in (1, 2, 3, 4)
    )
    assert any(
        torch.allclose(output[1, 1], (1 / 3) * weight[i] + (2 / 3) * weight[5])
        for i in (6, 7, 8)
    )
    assert any(
        torch.allclose(output[1, 2], (1 / 3) * weight[i] + (2 / 3) * weight[9])
        for i in (10, 11, 12)
    )
    assert torch.allclose(output[2, 0], weight[0])
    assert torch.allclose(output[2, 1], weight[5])
    assert torch.allclose(output[2, 2], weight[9])


def test_that_defaultembedding_raises_error_with_bad_from_values_input():
    unique_counts = [{"a": 4, "b": 3, "c": 2, "d": 1}, {0: 6, 1: 3}, {0: 5, "a": 5}]
    nan_counts = [0, 1]
    embedding = DefaultEmbedding(embedding_size=2, alpha=2)
    msg = "length of unique_counts and nan_counts should be equal, got 3, 2"
    with pytest.raises(ValueError, match=msg):
        embedding.from_values(unique_counts, nan_counts)


def test_that_defaultembedding_learns():
    X = np.random.randint(0, 5, (100, 10))
    y = torch.rand((100, 3))
    embedding = DefaultEmbedding(embedding_size=3, alpha=5).fit(X)
    model = nn.Sequential(embedding, Reshape(), nn.Linear(30, 3))
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(embedding.parameters(), lr=1e-1)  # not model.parameters()
    wt_before = torch.clone(embedding.embedding.weight)
    loss_vals = simple_train_loop(model, X, y, loss_func, optimizer, num_epochs=5)
    assert torch.all(embedding.embedding.weight != wt_before).item()
    assert loss_vals[0] > loss_vals[-1]

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
import pytest

from xynn.embedding import LinearEmbedding
from ...common import simple_train_loop
from ..utils import example_data, Reshape, SimpleDataset


def test_that_linearembedding_must_be_fit():
    embedding = LinearEmbedding(embedding_size=2)
    data_test = pd.DataFrame(
        {
            "num_a": [1, 0, 0.5, 0, -1],
            "num_b": [1, 0.5, 0, 0, -1],
        }
    )
    msg = "need to call `fit` or `from_values` first"
    with pytest.raises(RuntimeError, match=msg):
        embedding(data_test.values)


def test_linearembedding_with_pandas_example():
    data_num = example_data()[["num_a", "num_b"]]
    embedding = LinearEmbedding(embedding_size=3).fit(data_num)
    data_test = pd.DataFrame(
        {
            "num_a": [1, 0, 0.5, 0, -1],
            "num_b": [1, 0.5, 0, 0, -1],
        }
    )
    weight = embedding.embedding.weight
    output = embedding(torch.from_numpy(data_test.values)).to(dtype=weight.dtype)
    assert weight.shape == (2, 3)
    assert output.shape == (5, 2, 3)
    # test returned vectors vs weight matrix
    assert torch.all(output[0] == weight).item()
    assert torch.all(output[1, 0] == 0).item()
    assert torch.allclose(output[1, 1], 0.5 * weight[1])
    assert torch.allclose(output[2, 0], 0.5 * weight[0])
    assert torch.all(output[2, 1] == 0).item()
    assert torch.all(output[3] == 0).item()
    assert torch.all(output[4] == -weight).item()


def test_linearembedding_with_tensor_example():
    data_num = torch.from_numpy(example_data()[["num_a", "num_b"]].values)
    embedding = LinearEmbedding(embedding_size=3).fit(data_num)
    data_test = pd.DataFrame(
        {
            "num_a": [1, 0, 0.5, 0, -1],
            "num_b": [1, 0.5, 0, 0, -1],
        }
    )
    weight = embedding.embedding.weight
    output = embedding(torch.from_numpy(data_test.values)).to(dtype=weight.dtype)
    assert weight.shape == (2, 3)
    assert output.shape == (5, 2, 3)
    # test returned vectors vs weight matrix
    assert torch.all(output[0] == weight).item()
    assert torch.all(output[1, 0] == 0).item()
    assert torch.allclose(output[1, 1], 0.5 * weight[1])
    assert torch.allclose(output[2, 0], 0.5 * weight[0])
    assert torch.all(output[2, 1] == 0).item()
    assert torch.all(output[3] == 0).item()
    assert torch.all(output[4] == -weight).item()


def test_linearembedding_with_dataloader():
    data_num = torch.from_numpy(example_data()[["num_a", "num_b"]].values)
    dataset = SimpleDataset(data_num)
    dataloader = DataLoader(dataset, batch_size=5)
    embedding = LinearEmbedding(embedding_size=3).fit(dataloader)
    data_test = pd.DataFrame(
        {
            "num_a": [1, 0, 0.5, 0, -1],
            "num_b": [1, 0.5, 0, 0, -1],
        }
    )
    weight = embedding.embedding.weight
    output = embedding(torch.from_numpy(data_test.values)).to(dtype=weight.dtype)
    assert weight.shape == (2, 3)
    assert output.shape == (5, 2, 3)
    # test returned vectors vs weight matrix
    assert torch.all(output[0] == weight).item()
    assert torch.all(output[1, 0] == 0).item()
    assert torch.allclose(output[1, 1], 0.5 * weight[1])
    assert torch.allclose(output[2, 0], 0.5 * weight[0])
    assert torch.all(output[2, 1] == 0).item()
    assert torch.all(output[3] == 0).item()
    assert torch.all(output[4] == -weight).item()


def test_that_linearembedding_learns():
    X = torch.rand((100, 10))
    y = torch.rand((100, 3))
    embedding = LinearEmbedding(embedding_size=3).fit(X)
    model = nn.Sequential(embedding, Reshape(), nn.Linear(30, 3))
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(embedding.parameters(), lr=1e-1)  # not model.parameters()
    wt_before = torch.clone(embedding.embedding.weight)
    loss_vals = simple_train_loop(model, X, y, loss_func, optimizer, num_epochs=5)
    assert torch.all(embedding.embedding.weight != wt_before).item()
    assert loss_vals[0] > loss_vals[-1]

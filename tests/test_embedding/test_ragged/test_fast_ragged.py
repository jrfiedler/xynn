import numpy as np
import pandas as pd
import torch
from torch import nn
import pytest

from xynn.embedding import FastRaggedEmbedding, FastRaggedDefaultEmbedding
from ...common import simple_train_loop
from ..utils import example_data


def test_that_fastraggedembedding_must_be_fit():
    embedding = FastRaggedEmbedding(embedding_size=[2, 3, 2])
    data_test = torch.tensor(
        [[0, 0, 0], [1, 1, 1], [2, 2, 0], [3, 0, 2]], dtype=torch.int64
    )
    msg = "need to call `fit` or `from_summary` first"
    with pytest.raises(RuntimeError, match=msg):
        embedding(data_test)


def test_fastraggedembedding_repr():
    embedding = FastRaggedEmbedding(embedding_size=[2, 3, 2])
    assert repr(embedding) == "FastRaggedEmbedding([2, 3, 2], 100, 'cpu')"
    embedding = FastRaggedEmbedding(max_size=200)
    assert repr(embedding) == "FastRaggedEmbedding('sqrt', 200, 'cpu')"


def test_fastraggedembedding_repr_after_fitting():
    data_cat = example_data()[["cat_a", "cat_b", "cat_c"]]
    data_cat["cat_c"] = data_cat.cat_c.fillna(2).astype("int")
    embedding = FastRaggedEmbedding(embedding_size=[2, 3, 2]).fit(data_cat)
    assert repr(embedding) == "FastRaggedEmbedding([2, 3, 2], 100, 'cpu')"


def test_fastraggedembedding_with_pandas_example():
    data_cat = example_data()[["cat_a", "cat_b", "cat_c"]]
    data_cat["cat_c"] = data_cat.cat_c.fillna(2).astype("int")
    embedding = FastRaggedEmbedding(embedding_size=[3, 2, 2]).fit(data_cat)
    data_test = torch.tensor(
        [[0, 0, 0], [1, 1, 1], [2, 2, 0], [3, 0, 2]], dtype=torch.int64
    )
    output = embedding(data_test)
    weight = [emb.weight for emb in embedding.embedding]
    assert [w.shape for w in weight] == [(4, 3), (3, 2), (3, 2)]
    assert output.shape == (4, 7)
    # test returned vectors vs weight matrix
    assert torch.all(output[:, :3] == weight[0]).item()
    assert torch.all(output[:3, 3:5] == weight[1]).item()
    assert torch.all(output[[0, 1, 3], 5:] == weight[2]).item()


def test_fastraggedembedding_with_numpy_raises_errors_with_nans():
    data_cat = example_data()[["cat_a", "cat_b", "cat_c"]]
    data_cat = data_cat.values
    with pytest.raises(ValueError, match="NaN found in categorical data"):
        FastRaggedEmbedding(embedding_size="sqrt").fit(data_cat)


def test_fastraggedembedding_with_numpy_example():
    data_cat = example_data()[["cat_a", "cat_b", "cat_c"]]
    data_cat["cat_c"] = data_cat.cat_c.fillna(2).astype("int")
    data_cat = data_cat.values
    embedding = FastRaggedEmbedding(embedding_size="sqrt").fit(data_cat)
    data_test = torch.tensor(
        [[0, 0, 0], [1, 1, 1], [2, 2, 0], [3, 0, 2]], dtype=torch.int64
    )
    output = embedding(data_test)
    weight = [emb.weight for emb in embedding.embedding]
    assert [w.shape for w in weight] == [(4, 2), (3, 2), (3, 2)]
    assert output.shape == (4, 6)
    # test returned vectors vs weight matrix
    assert torch.all(output[:, :2] == weight[0]).item()
    assert torch.all(output[:3, 2:4] == weight[1]).item()
    assert torch.all(output[[0, 1, 3], 4:] == weight[2]).item()


def test_fastraggedembedding_with_tensor_example():
    data_cat = example_data()[["cat_a", "cat_b", "cat_c"]]
    data_cat["cat_c"] = data_cat.cat_c.fillna(2).astype("int")
    data_cat = torch.from_numpy(data_cat.values)
    embedding = FastRaggedEmbedding(embedding_size="sqrt").fit(data_cat)
    data_test = torch.tensor(
        [[0, 0, 0], [1, 1, 1], [2, 2, 0], [3, 0, 2]], dtype=torch.int64
    )
    output = embedding(data_test)
    weight = [emb.weight for emb in embedding.embedding]
    assert [w.shape for w in weight] == [(4, 2), (3, 2), (3, 2)]
    assert output.shape == (4, 6)
    # test returned vectors vs weight matrix
    assert torch.all(output[:, :2] == weight[0]).item()
    assert torch.all(output[:3, 2:4] == weight[1]).item()
    assert torch.all(output[[0, 1, 3], 4:] == weight[2]).item()


def test_fastraggedembedding_weight_sum():
    data_cat = example_data()[["cat_a", "cat_b", "cat_c"]]
    data_cat["cat_c"] = data_cat.cat_c.fillna(2).astype("int")
    embedding = FastRaggedEmbedding(embedding_size="sqrt")
    assert embedding.weight_sum() == (0.0, 0.0)

    embedding.fit(data_cat)

    expected_e1_sum = 0.0
    expected_e2_sum = 0.0
    for layer in embedding.embedding:
        weight = layer.weight
        layer.weight = nn.Parameter(torch.ones(weight.shape))
        expected_e1_sum += weight.shape[0] * weight.shape[1]
        expected_e2_sum += weight.shape[0] * weight.shape[1]

    e1_sum, e2_sum = embedding.weight_sum()

    assert isinstance(e1_sum, torch.Tensor)
    assert isinstance(e2_sum, torch.Tensor)
    assert e1_sum.item() == expected_e1_sum
    assert e2_sum.item() == expected_e2_sum


def test_fastraggedembedding_initialization_with_from_summary():
    uniques = [10, 25, 5, 2]
    embedding = FastRaggedEmbedding(embedding_size="log")
    embedding.from_summary(uniques)
    weight = [emb.weight for emb in embedding.embedding]
    assert [w.shape for w in weight] == [(10, 3), (25, 4), (5, 2), (2, 1)]


def test_fastraggedembedding_raises_error_for_bad_embedding_size():
    msg = (
        "str embedding_size value must be one of {'sqrt', 'log', 'fastai'}; "
        "got 'surprise me'"
    )
    with pytest.raises(ValueError, match=msg):
        embedding = FastRaggedEmbedding(embedding_size="surprise me")

    msg = r"embedding_size \['sqrt', 'log', 2\] not understood"
    with pytest.raises(TypeError, match=msg):
        embedding = FastRaggedEmbedding(embedding_size=["sqrt", "log", 2])


def test_that_fastraggedembedding_learns():
    X = torch.randint(0, 5, (100, 10))
    y = torch.rand((100, 3))
    embedding = FastRaggedEmbedding(embedding_size="sqrt").fit(X)
    model = nn.Sequential(embedding, nn.Linear(30, 3))
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(embedding.parameters(), lr=1e-1)  # not model.parameters()
    wt_before = [torch.clone(layer.weight) for layer in embedding.embedding]
    loss_vals = simple_train_loop(model, X, y, loss_func, optimizer, num_epochs=5)
    for layer, wt in zip(embedding.embedding, wt_before):
        assert torch.all(layer.weight != wt).item()
    assert loss_vals[0] > loss_vals[-1]


def test_that_fastraggeddefaultembedding_must_be_fit():
    embedding = FastRaggedDefaultEmbedding(embedding_size=[2, 3, 2], alpha=2)
    data_test = torch.tensor(
        [[0, 0, 0], [1, 1, 1], [2, 2, 0], [3, 0, 2]], dtype=torch.int64
    )
    msg = "need to call `fit` or `from_summary` first"
    with pytest.raises(RuntimeError, match=msg):
        embedding(data_test.values)


def test_fastraggeddefaultembedding_repr():
    embedding = FastRaggedDefaultEmbedding(embedding_size=[2, 3, 2], alpha=2)
    assert repr(embedding) == "FastRaggedDefaultEmbedding([2, 3, 2], 100, 2, 'cpu')"
    embedding = FastRaggedDefaultEmbedding(max_size=200)
    assert repr(embedding) == "FastRaggedDefaultEmbedding('sqrt', 200, 20, 'cpu')"


def test_fastraggeddefaultembedding_repr_after_fitting():
    data_cat = example_data()[["cat_a", "cat_b", "cat_c"]]
    data_cat["cat_c"] = data_cat.cat_c.fillna(2).astype("int")
    embedding = FastRaggedDefaultEmbedding(embedding_size=[2, 3, 2], alpha=2).fit(data_cat)
    assert repr(embedding) == "FastRaggedDefaultEmbedding([2, 3, 2], 100, 2, 'cpu')"


def test_fastraggeddefaultembedding_with_pandas_example():
    data_cat = example_data()[["cat_a", "cat_b", "cat_c"]]
    data_cat["cat_c"] = data_cat.cat_c.fillna(2).astype("int")
    embedding = FastRaggedDefaultEmbedding(embedding_size=[2, 3, 2], alpha=2)
    embedding.fit(data_cat)
    data_test = torch.tensor(
        [[0, 0, 0], [1, 1, 1], [2, 2, 0], [3, 0, 2]], dtype=torch.int64
    )
    output = embedding(data_test)
    weight = [emb.weight for emb in embedding.embedding]
    assert [w.shape for w in weight] == [(5, 2), (4, 3), (4, 2)]
    assert output.shape == (4, 7)
    a_wt = torch.tensor([[[4 / 6], [3 / 5], [2 / 4], [1 / 3]]])
    expected = weight[0][:-1] * a_wt + weight[0][-1:] * (1 - a_wt)
    assert torch.allclose(output[:, :2], expected)
    b_wt = torch.tensor([[[4 / 6], [5 / 7], [1 / 3]]])
    expected = weight[1][:-1] * b_wt + weight[1][-1:] * (1 - b_wt)
    assert torch.allclose(output[:3, 2:5], expected)
    c_wt = torch.tensor([[[3 / 5], [6 / 8], [1 / 3]]])
    expected = weight[2][:-1] * c_wt + weight[2][-1:] * (1 - c_wt)
    assert torch.allclose(output[[0, 1, 3], 5:], expected)


def test_fastraggeddefaultembedding_with_tensor_example():
    data_cat = example_data()[["cat_a", "cat_b", "cat_c"]]
    data_cat["cat_c"] = data_cat.cat_c.fillna(2).astype("int")
    data_cat = torch.from_numpy(data_cat.values)
    embedding = FastRaggedDefaultEmbedding(embedding_size=[2, 3, 2], alpha=2)
    embedding.fit(data_cat)
    data_test = torch.tensor(
        [[0, 0, 0], [1, 1, 1], [2, 2, 0], [3, 0, 2]], dtype=torch.int64
    )
    output = embedding(data_test)
    weight = [emb.weight for emb in embedding.embedding]
    assert [w.shape for w in weight] == [(5, 2), (4, 3), (4, 2)]
    assert output.shape == (4, 7)
    a_wt = torch.tensor([[[4 / 6], [3 / 5], [2 / 4], [1 / 3]]])
    expected = weight[0][:-1] * a_wt + weight[0][-1:] * (1 - a_wt)
    assert torch.allclose(output[:, :2], expected)
    b_wt = torch.tensor([[[4 / 6], [5 / 7], [1 / 3]]])
    expected = weight[1][:-1] * b_wt + weight[1][-1:] * (1 - b_wt)
    assert torch.allclose(output[:3, 2:5], expected)
    c_wt = torch.tensor([[[3 / 5], [6 / 8], [1 / 3]]])
    expected = weight[2][:-1] * c_wt + weight[2][-1:] * (1 - c_wt)
    assert torch.allclose(output[[0, 1, 3], 5:], expected)


def test_fastraggeddefaultembedding_with_unexpected_values():
    data_cat = example_data()[["cat_a", "cat_b", "cat_c"]]
    data_cat["cat_c"] = data_cat.cat_c.fillna(2).astype("int")
    data_cat = data_cat.values
    embedding = FastRaggedDefaultEmbedding(embedding_size=[2, 3, 2], alpha=2)
    embedding.fit(data_cat)
    data_test = torch.tensor(
        [[0, 0, 0], [4, 3, 3]], dtype=torch.int64
    )
    output = embedding(data_test)
    weight = [emb.weight for emb in embedding.embedding]
    assert [w.shape for w in weight] == [(5, 2), (4, 3), (4, 2)]
    assert output.shape == (2, 7)
    a_wt = torch.tensor([[[4 / 6], [0]]])
    expected = weight[0][:2] * a_wt + weight[0][-1:] * (1 - a_wt)
    assert torch.allclose(output[:, 0:2], expected)
    b_wt = torch.tensor([[[4 / 6], [0]]])
    expected = weight[1][:2] * b_wt + weight[1][-1:] * (1 - b_wt)
    assert torch.allclose(output[:, 2:5], expected)
    c_wt = torch.tensor([[[3 / 5], [0]]])
    expected = weight[2][:2] * c_wt + weight[2][-1:] * (1 - c_wt)
    assert torch.allclose(output[:, 5:], expected)


def test_that_fastraggeddefaultembedding_learns():
    X = torch.randint(0, 5, (100, 10))
    y = torch.rand((100, 3))
    embedding = FastRaggedDefaultEmbedding(embedding_size="sqrt", alpha=5).fit(X)
    model = nn.Sequential(embedding, nn.Linear(30, 3))
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(embedding.parameters(), lr=1e-1)  # not model.parameters()
    wt_before = [torch.clone(layer.weight) for layer in embedding.embedding]
    loss_vals = simple_train_loop(model, X, y, loss_func, optimizer, num_epochs=5)
    for layer, wt in zip(embedding.embedding, wt_before):
        assert torch.all(layer.weight != wt).item()
    assert loss_vals[0] > loss_vals[-1]

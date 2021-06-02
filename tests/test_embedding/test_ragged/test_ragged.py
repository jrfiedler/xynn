import numpy as np
import pandas as pd
import torch
from torch import nn
import pytest

from xynn.embedding import RaggedEmbedding, RaggedDefaultEmbedding
from ...common import simple_train_loop
from ..utils import example_data


def test_that_raggedembedding_must_be_fit():
    embedding = RaggedEmbedding(embedding_size=[2, 3, 2])
    data_test = pd.DataFrame(
        {
            "cat_a": [0, 1, 2, 3],
            "cat_b": [0, 1, 2, 0],
            "cat_c": [0, 1, np.nan, 0],
        }
    )
    msg = "need to call `fit` or `from_summary` first"
    with pytest.raises(RuntimeError, match=msg):
        embedding(data_test.values)


def test_raggedembedding_repr():
    embedding = RaggedEmbedding(embedding_size=[2, 3, 2])
    assert repr(embedding) == "RaggedEmbedding([2, 3, 2], 100, 'cpu')"
    embedding = RaggedEmbedding(max_size=200)
    assert repr(embedding) == "RaggedEmbedding('sqrt', 200, 'cpu')"


def test_raggedembedding_repr_after_fitting():
    data_cat = example_data()[["cat_a", "cat_b", "cat_c"]]
    embedding = RaggedEmbedding(embedding_size=[2, 3, 2]).fit(data_cat)
    assert repr(embedding) == "RaggedEmbedding([2, 3, 2], 100, 'cpu')"


def test_raggedembedding_with_pandas_example():
    data_cat = example_data()[["cat_a", "cat_b", "cat_c"]]
    embedding = RaggedEmbedding(embedding_size=[3, 2, 2]).fit(data_cat)
    data_test = pd.DataFrame(
        {
            "cat_a": [0, 1],
            "cat_b": [0, 2],
            "cat_c": [0, np.nan],
        }
    )
    output = embedding(data_test.values)
    weight = [emb.weight for emb in embedding.embedding]
    assert [w.shape for w in weight] == [(4, 3), (3, 2), (3, 2)]
    assert output.shape == (2, 7)
    # test returned vectors vs weight matrix
    assert torch.all(output[0, :3] == weight[0][0]).item()
    assert torch.all(output[0, 3:5] == weight[1][0]).item()
    assert (  # can't be sure of the order of categories with NaN
        torch.all(output[0, 5:] == weight[2][0]).item()
        or torch.all(output[0, 5:] == weight[2][1]).item()
        or torch.all(output[0, 5:] == weight[2][2]).item()
    )
    assert torch.all(output[1, :3] == weight[0][1]).item()
    assert torch.all(output[1, 3:5] == weight[1][2]).item()
    assert (  # can't be sure of the order of categories with NaN
        torch.all(output[1, 5:] == weight[2][0]).item()
        or torch.all(output[1, 5:] == weight[2][1]).item()
        or torch.all(output[1, 5:] == weight[2][2]).item()
    )


def test_raggedembedding_with_numpy_example():
    data_cat = example_data()[["cat_a", "cat_b", "cat_c"]].values
    embedding = RaggedEmbedding(embedding_size="sqrt").fit(data_cat)
    data_test = pd.DataFrame(
        {
            "cat_a": [0, 1],
            "cat_b": [0, 2],
            "cat_c": [0, np.nan],
        }
    )
    output = embedding(data_test.values)
    weight = [emb.weight for emb in embedding.embedding]
    assert [w.shape for w in weight] == [(4, 2), (3, 2), (3, 2)]
    assert output.shape == (2, 6)
    # test returned vectors vs weight matrix
    assert torch.all(output[0, :2] == weight[0][0]).item()
    assert torch.all(output[0, 2:4] == weight[1][0]).item()
    assert (  # can't be sure of the order of categories with NaN
        torch.all(output[0, 4:] == weight[2][0]).item()
        or torch.all(output[0, 4:] == weight[2][1]).item()
        or torch.all(output[0, 4:] == weight[2][2]).item()
    )
    assert torch.all(output[1, :2] == weight[0][1]).item()
    assert torch.all(output[1, 2:4] == weight[1][2]).item()
    assert (  # can't be sure of the order of categories with NaN
        torch.all(output[1, 4:] == weight[2][0]).item()
        or torch.all(output[1, 4:] == weight[2][1]).item()
        or torch.all(output[1, 4:] == weight[2][2]).item()
    )


def test_raggedembedding_with_pytorch_example():
    data_cat = torch.from_numpy(example_data()[["cat_a", "cat_b", "cat_c"]].values)
    embedding = RaggedEmbedding(embedding_size="sqrt").fit(data_cat)
    data_test = pd.DataFrame(
        {
            "cat_a": [0, 1],
            "cat_b": [0, 2],
            "cat_c": [0, np.nan],
        }
    )
    output = embedding(torch.from_numpy(data_test.values))
    weight = [emb.weight for emb in embedding.embedding]
    assert [w.shape for w in weight] == [(4, 2), (3, 2), (3, 2)]
    assert output.shape == (2, 6)
    # test returned vectors vs weight matrix
    assert torch.all(output[0, :2] == weight[0][0]).item()
    assert torch.all(output[0, 2:4] == weight[1][0]).item()
    assert (  # can't be sure of the order of categories with NaN
        torch.all(output[0, 4:] == weight[2][0]).item()
        or torch.all(output[0, 4:] == weight[2][1]).item()
        or torch.all(output[0, 4:] == weight[2][2]).item()
    )
    assert torch.all(output[1, :2] == weight[0][1]).item()
    assert torch.all(output[1, 2:4] == weight[1][2]).item()
    assert (  # can't be sure of the order of categories with NaN
        torch.all(output[1, 4:] == weight[2][0]).item()
        or torch.all(output[1, 4:] == weight[2][1]).item()
        or torch.all(output[1, 4:] == weight[2][2]).item()
    )


def test_raggedembedding_weight_sum():
    data_cat = example_data()[["cat_a", "cat_b", "cat_c"]].values
    embedding = RaggedEmbedding(embedding_size="sqrt")
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


def test_raggedembedding_initialization_with_from_summary():
    uniques = [list(range(10)), list(range(25)), list(range(5)), list(range(2))]
    has_nan = [False, False, True, False]
    embedding = RaggedEmbedding(embedding_size="log")
    embedding.from_summary(uniques, has_nan)
    weight = [emb.weight for emb in embedding.embedding]
    assert [w.shape for w in weight] == [(10, 3), (25, 4), (6, 2), (2, 1)]


def test_raggedembedding_raises_error_for_bad_embedding_size():
    msg = (
        "str embedding_size value must be one of {'sqrt', 'log', 'fastai'}; "
        "got 'surprise me'"
    )
    with pytest.raises(ValueError, match=msg):
        embedding = RaggedEmbedding(embedding_size="surprise me")

    msg = r"embedding_size \['sqrt', 'log', 2\] not understood"
    with pytest.raises(TypeError, match=msg):
        embedding = RaggedEmbedding(embedding_size=["sqrt", "log", 2])

    embedding = RaggedEmbedding(embedding_size=[3, 2])
    uniques = [["a", "b", "c", "d"], [0, 1, np.nan], [0, "a", 2.5]]
    has_nan = [False, True, False]
    msg = "number of embeddings must match number of fields, got 2 sizes and 3 fields"
    with pytest.raises(ValueError, match=msg):
        embedding.from_summary(uniques, has_nan)


def test_that_raggedembedding_raises_error_with_bad_from_summary_input():
    uniques = [["a", "b", "c", "d"], [0, 1, np.nan], [0, "a"]]
    has_nan = [False, True]
    embedding = RaggedEmbedding()
    msg = "length of uniques and has_nan should be equal, got 3, 2"
    with pytest.raises(ValueError, match=msg):
        embedding.from_summary(uniques, has_nan)


def test_that_raggedembedding_learns():
    X = np.random.randint(0, 5, (100, 10))
    y = torch.rand((100, 3))
    embedding = RaggedEmbedding(embedding_size="sqrt").fit(X)
    model = nn.Sequential(embedding, nn.Linear(30, 3))
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(embedding.parameters(), lr=1e-1)  # not model.parameters()
    wt_before = [torch.clone(layer.weight) for layer in embedding.embedding]
    loss_vals = simple_train_loop(model, X, y, loss_func, optimizer, num_epochs=5)
    for layer, wt in zip(embedding.embedding, wt_before):
        assert torch.all(layer.weight != wt).item()
    assert loss_vals[0] > loss_vals[-1]


def test_that_raggeddefaultembedding_must_be_fit():
    embedding = RaggedDefaultEmbedding(embedding_size=[2, 3, 2], alpha=2)
    data_test = pd.DataFrame(
        {
            "cat_a": [0, 1, 2, 3],
            "cat_b": [0, 1, 2, 0],
            "cat_c": [0, 1, np.nan, 0],
        }
    )
    msg = "need to call `fit` or `from_summary` first"
    with pytest.raises(RuntimeError, match=msg):
        embedding(data_test.values)


def test_raggeddefaultembedding_repr():
    embedding = RaggedDefaultEmbedding(embedding_size=[2, 3, 2], alpha=2)
    assert repr(embedding) == "RaggedDefaultEmbedding([2, 3, 2], 100, 2, 'cpu')"
    embedding = RaggedDefaultEmbedding(max_size=200)
    assert repr(embedding) == "RaggedDefaultEmbedding('sqrt', 200, 20, 'cpu')"


def test_raggeddefaultembedding_repr_after_fitting():
    data_cat = example_data()[["cat_a", "cat_b", "cat_c"]]
    embedding = RaggedDefaultEmbedding(embedding_size=[2, 3, 2], alpha=2).fit(data_cat)
    assert repr(embedding) == "RaggedDefaultEmbedding([2, 3, 2], 100, 2, 'cpu')"


def test_raggeddefaultembedding_with_pandas_example():
    data_cat = example_data()[["cat_a", "cat_b", "cat_c"]]
    embedding = RaggedDefaultEmbedding(embedding_size=[2, 3, 2], alpha=2).fit(data_cat)
    data_test = pd.DataFrame(
        {
            "cat_a": [0, 1, 4],
            "cat_b": [0, 2, 3],
            "cat_c": [0, np.nan, 2],
        }
    )
    output = embedding(data_test.values)
    weight = [emb.weight for emb in embedding.embedding]
    assert [w.shape for w in weight] == [(5, 2), (4, 3), (4, 2)]
    assert output.shape == (3, 7)
    # test returned vectors vs weight matrix
    # `any` is used because we can't be sure of the order
    assert any(
        torch.allclose(output[0, 0:2], (2 / 3) * weight[0][i] + (1 / 3) * weight[0][0])
        for i in (1, 2, 3, 4)
    )
    assert any(
        torch.allclose(output[0, 2:5], (2 / 3) * weight[1][i] + (1 / 3) * weight[1][0])
        for i in (1, 2, 3)
    )
    assert any(
        torch.allclose(output[0, 5:7], (3 / 5) * weight[2][i] + (2 / 5) * weight[2][0])
        for i in (1, 2, 3)
    )
    assert any(
        torch.allclose(output[1, 0:2], (3 / 5) * weight[0][i] + (2 / 5) * weight[0][0])
        for i in (1, 2, 3, 4)
    )
    assert any(
        torch.allclose(output[1, 2:5], (1 / 3) * weight[1][i] + (2 / 3) * weight[1][0])
        for i in (1, 2, 3)
    )
    assert any(
        torch.allclose(output[1, 5:7], (1 / 3) * weight[2][i] + (2 / 3) * weight[2][0])
        for i in (1, 2, 3)
    )
    assert torch.allclose(output[2, 0:2], weight[0][0])
    assert torch.allclose(output[2, 2:5], weight[1][0])
    assert torch.allclose(output[2, 5:7], weight[2][0])


def test_raggeddefaultembedding_with_tensor_example():
    data_cat = torch.from_numpy(
        example_data()[["cat_a", "cat_b", "cat_c"]].values
    )
    embedding = RaggedDefaultEmbedding(embedding_size=[2, 3, 2], alpha=2).fit(data_cat)
    data_test = pd.DataFrame(
        {
            "cat_a": [0, 1, 5],
            "cat_b": [0, 2, 5],
            "cat_c": [0, np.nan, 2],
        }
    )
    output = embedding(torch.from_numpy(data_test.values))
    weight = [emb.weight for emb in embedding.embedding]
    assert [w.shape for w in weight] == [(5, 2), (4, 3), (4, 2)]
    assert output.shape == (3, 7)
    # test returned vectors vs weight matrix
    # `any` is used because we can't be sure of the order
    assert any(
        torch.allclose(output[0, 0:2], (2 / 3) * weight[0][i] + (1 / 3) * weight[0][0])
        for i in (1, 2, 3, 4)
    )
    assert any(
        torch.allclose(output[0, 2:5], (2 / 3) * weight[1][i] + (1 / 3) * weight[1][0])
        for i in (1, 2, 3)
    )
    assert any(
        torch.allclose(output[0, 5:7], (3 / 5) * weight[2][i] + (2 / 5) * weight[2][0])
        for i in (1, 2, 3)
    )
    assert any(
        torch.allclose(output[1, 0:2], (3 / 5) * weight[0][i] + (2 / 5) * weight[0][0])
        for i in (1, 2, 3, 4)
    )
    assert any(
        torch.allclose(output[1, 2:5], (1 / 3) * weight[1][i] + (2 / 3) * weight[1][0])
        for i in (1, 2, 3)
    )
    assert any(
        torch.allclose(output[1, 5:7], (1 / 3) * weight[2][i] + (2 / 3) * weight[2][0])
        for i in (1, 2, 3)
    )
    assert torch.allclose(output[2, 0:2], weight[0][0])
    assert torch.allclose(output[2, 2:5], weight[1][0])
    assert torch.allclose(output[2, 5:7], weight[2][0])


def test_that_raggeddefaultembedding_raises_error_with_bad_from_summary_input():
    unique_counts = [{"a": 4, "b": 3, "c": 2, "d": 1}, {0: 6, 1: 3}, {0: 5, "a": 5}]
    nan_counts = [0, 1]
    embedding = RaggedDefaultEmbedding()
    msg = "length of unique_counts and nan_count should be equal, got 3, 2"
    with pytest.raises(ValueError, match=msg):
        embedding.from_summary(unique_counts, nan_counts)


def test_that_raggeddefaultembedding_learns():
    X = np.random.randint(0, 5, (100, 10))
    y = torch.rand((100, 3))
    embedding = RaggedDefaultEmbedding(embedding_size="sqrt", alpha=5).fit(X)
    model = nn.Sequential(embedding, nn.Linear(30, 3))
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(embedding.parameters(), lr=1e-1)  # not model.parameters()
    wt_before = [torch.clone(layer.weight) for layer in embedding.embedding]
    loss_vals = simple_train_loop(model, X, y, loss_func, optimizer, num_epochs=5)
    for layer, wt in zip(embedding.embedding, wt_before):
        assert torch.all(layer.weight != wt).item()
    assert loss_vals[0] > loss_vals[-1]

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
import pytest
from sklearn.preprocessing import LabelEncoder

from xynn.base_classes.estimators import _set_seed
from xynn.embedding import FastBasicEmbedding, FastDefaultEmbedding
from xynn.preprocessing import IntegerEncoder
from ...common import simple_train_loop
from ..utils import example_data, Reshape, SimpleDataset


def test_that_fastbasicembedding_must_be_fit():
    embedding = FastBasicEmbedding(embedding_size=2)
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


def test_fastbasicembedding_repr():
    embedding = FastBasicEmbedding(embedding_size=2)
    assert repr(embedding) == "FastBasicEmbedding(2, 'cpu')"
    embedding = FastBasicEmbedding()
    assert repr(embedding) == "FastBasicEmbedding(10, 'cpu')"


def test_fastbasicembedding_with_pandas_example():
    data_cat = example_data()[["cat_a", "cat_b", "cat_c"]]
    data_cat["cat_c"] = data_cat.cat_c.fillna(2).astype("int")

    embedding = FastBasicEmbedding(embedding_size=2).fit(data_cat)
    data_test = torch.tensor(
        [[0, 0, 0], [1, 1, 1], [2, 2, 0], [3, 0, 2]], dtype=torch.int64
    )
    output = embedding(data_test)
    weight = embedding.embedding.weight
    assert weight.shape == (10, 2)
    assert output.shape == (4, 3, 2)
    # test returned vectors vs weight matrix
    assert torch.all(output[:, 0, :] == weight[:4]).item()
    assert torch.all(output[:3, 1, :] == weight[4:7]).item()
    assert torch.all(output[[0, 1, 3], 2, :] == weight[7:]).item()


def test_fastbasicembedding_with_numpy_raises_errors_with_nans():
    data_cat = example_data()[["cat_a", "cat_b", "cat_c"]]
    data_cat = data_cat.values
    with pytest.raises(ValueError, match="NaN found in categorical data"):
        FastBasicEmbedding(embedding_size=2).fit(data_cat)


def test_fastbasicembedding_with_numpy_example():
    data_cat = example_data()[["cat_a", "cat_b", "cat_c"]]
    data_cat["cat_c"] = data_cat.cat_c.fillna(2).astype("int")
    data_cat = data_cat.values

    embedding = FastBasicEmbedding(embedding_size=2).fit(data_cat)
    data_test = torch.tensor(
        [[0, 0, 0], [1, 1, 1], [2, 2, 0], [3, 0, 2]], dtype=torch.int64
    )
    output = embedding(data_test)
    weight = embedding.embedding.weight
    assert weight.shape == (10, 2)
    assert output.shape == (4, 3, 2)
    # test returned vectors vs weight matrix
    assert torch.all(output[:, 0, :] == weight[:4]).item()
    assert torch.all(output[:3, 1, :] == weight[4:7]).item()
    assert torch.all(output[[0, 1, 3], 2, :] == weight[7:]).item()


def test_fastbasicembedding_with_tensor_example():
    data_cat = example_data()[["cat_a", "cat_b", "cat_c"]]
    data_cat["cat_c"] = data_cat.cat_c.fillna(2).astype("int")
    data_cat = torch.from_numpy(data_cat.values)

    embedding = FastBasicEmbedding(embedding_size=2).fit(data_cat)
    data_test = torch.tensor(
        [[0, 0, 0], [1, 1, 1], [2, 2, 0], [3, 0, 2]], dtype=torch.int64
    )
    output = embedding(data_test)
    weight = embedding.embedding.weight
    assert weight.shape == (10, 2)
    assert output.shape == (4, 3, 2)
    # test returned vectors vs weight matrix
    assert torch.all(output[:, 0, :] == weight[:4]).item()
    assert torch.all(output[:3, 1, :] == weight[4:7]).item()
    assert torch.all(output[[0, 1, 3], 2, :] == weight[7:]).item()


def test_fastbasicembedding_with_dataloader_raises_errors_with_nans():
    data_cat = example_data()[["cat_a", "cat_b", "cat_c"]]
    data_cat = torch.from_numpy(data_cat.values)
    dataset = SimpleDataset(data_cat)
    dataloader = DataLoader(dataset, batch_size=5)
    with pytest.raises(ValueError, match="NaN found in categorical data"):
        FastBasicEmbedding(embedding_size=2).fit(dataloader)


def test_fastbasicembedding_with_dataloader_example():
    data_cat = example_data()[["cat_a", "cat_b", "cat_c"]]
    data_cat["cat_c"] = data_cat.cat_c.fillna(2).astype("int")
    data_cat = torch.from_numpy(data_cat.values)
    dataset = SimpleDataset(data_cat)
    dataloader = DataLoader(dataset, batch_size=5)
    embedding = FastBasicEmbedding(embedding_size=2).fit(dataloader)
    data_test = torch.tensor(
        [[0, 0, 0], [1, 1, 1], [2, 2, 0], [3, 0, 2]], dtype=torch.int64
    )
    output = embedding(data_test)
    weight = embedding.embedding.weight
    assert weight.shape == (10, 2)
    assert output.shape == (4, 3, 2)
    # test returned vectors vs weight matrix
    assert torch.all(output[:, 0, :] == weight[:4]).item()
    assert torch.all(output[:3, 1, :] == weight[4:7]).item()
    assert torch.all(output[[0, 1, 3], 2, :] == weight[7:]).item()


def test_that_fastbasicembedding_from_encoder_raises_error_for_bad_input():
    data_cat = example_data()[["cat_a", "cat_b", "cat_c"]]
    data_cat["cat_c"] = data_cat.cat_c.fillna(2).astype("int")
    data_cat = data_cat.values

    # not fit
    encoder = IntegerEncoder()
    with pytest.raises(ValueError, match="encoder needs to be fit"):
        FastBasicEmbedding(embedding_size=2).from_encoder(encoder)

    # wrong class
    encoder = LabelEncoder().fit(data_cat[:, 0])
    with pytest.raises(TypeError, match="encoder needs to be a fit IntegerEncoder"):
        FastBasicEmbedding(embedding_size=2).from_encoder(encoder)


def test_fastbasicembedding_from_encoder():
    data_cat = example_data()[["cat_a", "cat_b", "cat_c"]]
    data_cat["cat_c"] = data_cat.cat_c.fillna(2).astype("int")
    data_cat = data_cat.values

    encoder = IntegerEncoder().fit(data_cat)

    embedding = FastBasicEmbedding(embedding_size=2).from_encoder(encoder)
    data_test = torch.tensor(
        [[0, 0, 0], [1, 1, 1], [2, 2, 0], [3, 0, 2]], dtype=torch.int64
    )
    output = embedding(data_test)
    weight = embedding.embedding.weight
    assert weight.shape == (10, 2)
    assert output.shape == (4, 3, 2)
    # test returned vectors vs weight matrix
    assert torch.all(output[:, 0, :] == weight[:4]).item()
    assert torch.all(output[:3, 1, :] == weight[4:7]).item()
    assert torch.all(output[[0, 1, 3], 2, :] == weight[7:]).item()


def test_fastbasicembedding_weight_sum():
    data_cat = example_data()[["cat_a", "cat_b", "cat_c"]]
    data_cat["cat_c"] = data_cat.cat_c.fillna(2).astype("int")
    data_cat = data_cat.values
    embedding = FastBasicEmbedding(embedding_size=4)
    assert embedding.weight_sum() == (0.0, 0.0)

    embedding.fit(data_cat)

    weight = embedding.embedding.weight
    embedding.embedding.weight = nn.Parameter(torch.ones(weight.shape))

    e1_sum, e2_sum = embedding.weight_sum()

    assert isinstance(e1_sum, torch.Tensor)
    assert isinstance(e2_sum, torch.Tensor)
    assert e1_sum.item() == weight.shape[0] * weight.shape[1]
    assert e2_sum.item() == weight.shape[0] * weight.shape[1]


def test_that_fastbasicembedding_raises_error_with_wrong_fit_input():
    data_cat = example_data().to_dict()
    embedding = FastBasicEmbedding(embedding_size=2)
    msg = (
        "input X must be a PyTorch Tensor, PyTorch DataLoader, "
        "NumPy array, or Pandas DataFrame"
    )
    with pytest.raises(TypeError, match=msg):
        embedding.fit(data_cat)


def test_that_fastbasicembedding_learns():
    X = torch.randint(0, 5, (100, 10))
    y = torch.rand((100, 3))
    embedding = FastBasicEmbedding(embedding_size=3).fit(X)
    model = nn.Sequential(embedding, Reshape(), nn.Linear(30, 3))
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(embedding.parameters(), lr=1e-1)  # not model.parameters()
    wt_before = torch.clone(embedding.embedding.weight)
    loss_vals = simple_train_loop(model, X, y, loss_func, optimizer, num_epochs=5)
    assert torch.all(embedding.embedding.weight != wt_before).item()
    assert loss_vals[0] > loss_vals[-1]


def test_that_fastdefaultembedding_must_be_fit():
    embedding = FastDefaultEmbedding(embedding_size=2, alpha=2)
    data_test = torch.tensor(
        [[0, 0, 0], [1, 1, 1], [2, 2, 0], [3, 0, 2]], dtype=torch.int64
    )
    msg = "need to call `fit` or `from_summary` first"
    with pytest.raises(RuntimeError, match=msg):
        embedding(data_test.values)


def test_fastdefaultembedding_repr():
    embedding = FastDefaultEmbedding(embedding_size=2, alpha=2)
    assert repr(embedding) == "FastDefaultEmbedding(2, 2, 'cpu')"
    embedding = FastDefaultEmbedding()
    assert repr(embedding) == "FastDefaultEmbedding(10, 20, 'cpu')"


def test_fastdefaultembedding_with_pandas_example():
    data_cat = example_data()[["cat_a", "cat_b", "cat_c"]]
    data_cat["cat_c"] = data_cat.cat_c.fillna(2).astype("int")
    embedding = FastDefaultEmbedding(embedding_size=3, alpha=2).fit(data_cat)
    data_test = torch.tensor(
        [[0, 0, 0], [1, 1, 1], [2, 2, 0], [3, 0, 2]], dtype=torch.int64
    )
    output = embedding(data_test)
    weight = embedding.embedding.weight
    assert weight.shape == (13, 3)
    assert output.shape == (4, 3, 3)
    a_wt = torch.tensor([[[4 / 6], [3 / 5], [2 / 4], [1 / 3]]])
    expected = weight[:4] * a_wt + weight[10] * (1 - a_wt)
    assert torch.allclose(output[:, 0], expected)
    b_wt = torch.tensor([[[4 / 6], [5 / 7], [1 / 3]]])
    expected = weight[4:7] * b_wt + weight[11] * (1 - b_wt)
    assert torch.allclose(output[:3, 1], expected)
    c_wt = torch.tensor([[[3 / 5], [6 / 8], [1 / 3]]])
    expected = weight[7:10] * c_wt + weight[12] * (1 - c_wt)
    assert torch.allclose(output[[0, 1, 3], 2], expected)


def test_fastdefaultembedding_with_numpy_raises_errors_with_nans():
    data_cat = example_data()[["cat_a", "cat_b", "cat_c"]]
    data_cat = data_cat.values
    with pytest.raises(ValueError, match="NaN found in categorical data"):
        FastDefaultEmbedding(embedding_size=2).fit(data_cat)


def test_fastdefaultembedding_with_numpy_example():
    data_cat = example_data()[["cat_a", "cat_b", "cat_c"]]
    data_cat["cat_c"] = data_cat.cat_c.fillna(2).astype("int")
    data_cat = data_cat.values
    embedding = FastDefaultEmbedding(embedding_size=3, alpha=2).fit(data_cat)
    data_test = torch.tensor(
        [[0, 0, 0], [1, 1, 1], [2, 2, 0], [3, 0, 2]], dtype=torch.int64
    )
    output = embedding(data_test)
    weight = embedding.embedding.weight
    assert weight.shape == (13, 3)
    assert output.shape == (4, 3, 3)
    a_wt = torch.tensor([[[4 / 6], [3 / 5], [2 / 4], [1 / 3]]])
    expected = weight[:4] * a_wt + weight[10] * (1 - a_wt)
    assert torch.allclose(output[:, 0], expected)
    b_wt = torch.tensor([[[4 / 6], [5 / 7], [1 / 3]]])
    expected = weight[4:7] * b_wt + weight[11] * (1 - b_wt)
    assert torch.allclose(output[:3, 1], expected)
    c_wt = torch.tensor([[[3 / 5], [6 / 8], [1 / 3]]])
    expected = weight[7:10] * c_wt + weight[12] * (1 - c_wt)
    assert torch.allclose(output[[0, 1, 3], 2], expected)


def test_fastdefaultembedding_with_unexpected_values():
    data_cat = example_data()[["cat_a", "cat_b", "cat_c"]]
    data_cat["cat_c"] = data_cat.cat_c.fillna(2).astype("int")
    data_cat = data_cat.values
    embedding = FastDefaultEmbedding(embedding_size=3, alpha=2).fit(data_cat)
    data_test = torch.tensor(
        [[0, 0, 0], [4, 3, 3]], dtype=torch.int64
    )
    output = embedding(data_test)
    weight = embedding.embedding.weight
    assert weight.shape == (13, 3)
    assert output.shape == (2, 3, 3)
    a_wt = torch.tensor([[[4 / 6], [0]]])
    expected = weight[:2] * a_wt + weight[10] * (1 - a_wt)
    assert torch.allclose(output[:, 0], expected)
    b_wt = torch.tensor([[[4 / 6], [0]]])
    expected = weight[4:6] * b_wt + weight[11] * (1 - b_wt)
    assert torch.allclose(output[:, 1], expected)
    c_wt = torch.tensor([[[3 / 5], [0]]])
    expected = weight[7:9] * c_wt + weight[12] * (1 - c_wt)
    assert torch.allclose(output[:, 2], expected)


def test_fastdefaultembedding_with_dataloader_raises_errors_with_nans():
    data_cat = example_data()[["cat_a", "cat_b", "cat_c"]]
    data_cat = torch.from_numpy(data_cat.values)
    dataset = SimpleDataset(data_cat)
    dataloader = DataLoader(dataset, batch_size=5)
    with pytest.raises(ValueError, match="NaN found in categorical data"):
        FastDefaultEmbedding(embedding_size=2).fit(dataloader)


def test_fastdefaultembedding_with_dataloader_example():
    data_cat = example_data()[["cat_a", "cat_b", "cat_c"]]
    data_cat["cat_c"] = data_cat.cat_c.fillna(2).astype("int")
    data_cat = torch.from_numpy(data_cat.values)
    dataset = SimpleDataset(data_cat)
    dataloader = DataLoader(dataset, batch_size=5)
    embedding = FastDefaultEmbedding(embedding_size=3, alpha=2).fit(dataloader)
    data_test = torch.tensor(
        [[0, 0, 0], [1, 1, 1], [2, 2, 0], [3, 0, 2]], dtype=torch.int64
    )
    output = embedding(data_test)
    weight = embedding.embedding.weight
    assert weight.shape == (13, 3)
    assert output.shape == (4, 3, 3)
    a_wt = torch.tensor([[[4 / 6], [3 / 5], [2 / 4], [1 / 3]]])
    expected = weight[:4] * a_wt + weight[10] * (1 - a_wt)
    assert torch.allclose(output[:, 0], expected)
    b_wt = torch.tensor([[[4 / 6], [5 / 7], [1 / 3]]])
    expected = weight[4:7] * b_wt + weight[11] * (1 - b_wt)
    assert torch.allclose(output[:3, 1], expected)
    c_wt = torch.tensor([[[3 / 5], [6 / 8], [1 / 3]]])
    expected = weight[7:10] * c_wt + weight[12] * (1 - c_wt)
    assert torch.allclose(output[[0, 1, 3], 2], expected)


def test_that_fastdefaultembedding_from_encoder_raises_error_for_bad_input():
    data_cat = example_data()[["cat_a", "cat_b", "cat_c"]]
    data_cat["cat_c"] = data_cat.cat_c.fillna(2).astype("int")
    data_cat = data_cat.values

    # not fit
    encoder = IntegerEncoder()
    with pytest.raises(ValueError, match="encoder needs to be fit"):
        FastDefaultEmbedding(embedding_size=2).from_encoder(encoder)

    # wrong class
    encoder = LabelEncoder().fit(data_cat[:, 0])
    with pytest.raises(TypeError, match="encoder needs to be a fit IntegerEncoder"):
        FastDefaultEmbedding(embedding_size=2).from_encoder(encoder)


def test_fastdefaultembedding_from_encoder():
    data_cat = example_data()[["cat_a", "cat_b", "cat_c"]]
    data_cat["cat_c"] = data_cat.cat_c.fillna(2).astype("int")
    data_cat = data_cat.values

    encoder = IntegerEncoder().fit(data_cat)

    embedding = FastDefaultEmbedding(embedding_size=2, alpha=2).from_encoder(encoder)
    data_test = torch.tensor(
        [[0, 0, 0], [1, 1, 1], [2, 2, 0], [3, 0, 2]], dtype=torch.int64
    )
    output = embedding(data_test)
    weight = embedding.embedding.weight
    assert weight.shape == (13, 2)
    assert output.shape == (4, 3, 2)
    a_wt = torch.tensor([[[4 / 6], [3 / 5], [2 / 4], [1 / 3]]])
    expected = weight[:4] * a_wt + weight[10] * (1 - a_wt)
    assert torch.allclose(output[:, 0], expected)
    b_wt = torch.tensor([[[4 / 6], [5 / 7], [1 / 3]]])
    expected = weight[4:7] * b_wt + weight[11] * (1 - b_wt)
    assert torch.allclose(output[:3, 1], expected)
    c_wt = torch.tensor([[[3 / 5], [6 / 8], [1 / 3]]])
    expected = weight[7:10] * c_wt + weight[12] * (1 - c_wt)
    assert torch.allclose(output[[0, 1, 3], 2], expected)


def test_that_fastdefaultembedding_learns():
    X = torch.randint(0, 5, (100, 10))
    y = torch.rand((100, 3))
    embedding = FastDefaultEmbedding(embedding_size=3, alpha=5).fit(X)
    model = nn.Sequential(embedding, Reshape(), nn.Linear(30, 3))
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(embedding.parameters(), lr=1e-1)  # not model.parameters()
    wt_before = torch.clone(embedding.embedding.weight)
    loss_vals = simple_train_loop(model, X, y, loss_func, optimizer, num_epochs=5)
    assert torch.all(embedding.embedding.weight != wt_before).item()
    assert loss_vals[0] > loss_vals[-1]

import pytest
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from xynn.embedding.utils import _init_embed_info, _check_is_uniform
from xynn.embedding import fit_embeddings, check_uniform_embeddings
from xynn.embedding import LinearEmbedding, BasicEmbedding, DefaultEmbedding
from xynn.embedding import RaggedEmbedding, RaggedDefaultEmbedding
from xynn.dataset import TabularDataset
from .utils import example_data


def simple_dataloader():
    data = example_data()
    X_num = data[["num_a", "num_b"]].values
    X_cat = data[["cat_a", "cat_b", "cat_c"]].values
    y = np.array([0, 1, 1, 0, 1, 1, 0, 0, 1, 0])
    dataset = TabularDataset(X_num, X_cat, y, task="classification")
    dataloader = DataLoader(dataset, batch_size=2)
    return dataloader


def check_linear(embedding):
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


def test_fit_embeddings_with_both_none():
    dataloader = simple_dataloader()
    embedding_num, embedding_cat = fit_embeddings(dataloader, None, None)
    assert embedding_num is None
    assert embedding_cat is None


def test_fit_embeddings_with_None_and_basic():
    dataloader = simple_dataloader()
    embedding_cat = BasicEmbedding(embedding_size=2)
    embedding_num, embedding_cat = fit_embeddings(dataloader, None, embedding_cat)
    assert embedding_num is None
    data_test = pd.DataFrame(
        {
            "cat_a": [0, 1, 2, 3],
            "cat_b": [0, 1, 2, 0],
            "cat_c": [0, 1, np.nan, 0],
        }
    )
    output = embedding_cat(data_test.values)
    weight = embedding_cat.embedding.weight
    assert weight.shape == (10, 2)
    assert output.shape == (4, 3, 2)
    # test returned vectors vs weight matrix
    assert torch.all(output[:, 0, :] == weight[:4]).item()
    assert torch.all(output[:3, 1, :] == weight[4:7]).item()
    assert torch.all(output[:3, 2, :] == weight[7:]).item()


def test_fit_embeddings_with_linear_and_none():
    dataloader = simple_dataloader()
    embedding_num = LinearEmbedding(embedding_size=3)
    embedding_num, embedding_cat = fit_embeddings(dataloader, embedding_num, None)
    assert embedding_cat is None
    check_linear(embedding_num)


def test_fit_embeddings_with_linear_and_ragged_default():
    dataloader = simple_dataloader()
    embedding_num = LinearEmbedding(embedding_size=3)
    embedding_cat = RaggedDefaultEmbedding(embedding_size=[2, 3, 2], alpha=2)
    embedding_num, embedding_cat = fit_embeddings(dataloader, embedding_num, embedding_cat)

    # test linear embedding
    check_linear(embedding_num)

    # test categorical embedding
    data_test = pd.DataFrame(
        {
            "cat_a": [0, 1, 4],
            "cat_b": [0, 2, 3],
            "cat_c": [0, np.nan, 2],
        }
    )
    output = embedding_cat(data_test.values)
    weight = [emb.weight for emb in embedding_cat.embedding]
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


def test_that_check_uniform_raises_error_with_nonuniform_embedding():
    embedding_num = LinearEmbedding(embedding_size=10)
    embedding_cat = RaggedDefaultEmbedding()
    with pytest.raises(
        TypeError, 
        match=(
            "only \'uniform\' embeddings are allowed for this model; "
            "embedding_cat is not a uniform embedding"
        ),
    ):
        check_uniform_embeddings(embedding_num, embedding_cat)

    embedding_num = RaggedEmbedding()
    embedding_cat = BasicEmbedding(embedding_size=10)
    with pytest.raises(
        TypeError, 
        match=(
            "only \'uniform\' embeddings are allowed for this model; "
            "embedding_num is not a uniform embedding"
        ),
    ):
        check_uniform_embeddings(embedding_num, embedding_cat)


def test_that_check_uniform_raises_error_when_both_none():
    with pytest.raises(
        ValueError, match="embedding_num and embedding_cat cannot both be None"
    ):
        check_uniform_embeddings(None, None)


def test_that_check_uniform_raises_error_when_sizes_differ():
    embedding_num = LinearEmbedding(embedding_size=4)
    embedding_cat = BasicEmbedding(embedding_size=10)
    with pytest.raises(
        ValueError, 
        match="embedding sizes must be the same for numeric and catgorical; got 4 and 10",
    ):
        check_uniform_embeddings(embedding_num, embedding_cat)


def test_check_uniform_when_embedding_num_is_none():
    dataloader = simple_dataloader()
    embedding_num = None

    embedding_cat = DefaultEmbedding(embedding_size=10, alpha=2)
    embedding_num, embedding_cat = fit_embeddings(dataloader, embedding_num, embedding_cat)
    num_fields, emb_size, out_size = check_uniform_embeddings(embedding_num, embedding_cat)
    assert num_fields == 3
    assert emb_size == 10
    assert out_size == 30

    embedding_cat = BasicEmbedding(embedding_size=8)
    embedding_num, embedding_cat = fit_embeddings(dataloader, embedding_num, embedding_cat)
    num_fields, emb_size, out_size = check_uniform_embeddings(embedding_num, embedding_cat)
    assert num_fields == 3
    assert emb_size == 8
    assert out_size == 24


def test_check_uniform_when_embedding_cat_is_none():
    dataloader = simple_dataloader()
    embedding_num = LinearEmbedding(embedding_size=4)
    embedding_cat = None
    embedding_num, embedding_cat = fit_embeddings(dataloader, embedding_num, embedding_cat)
    num_fields, emb_size, out_size = check_uniform_embeddings(embedding_num, embedding_cat)
    assert num_fields == 2
    assert emb_size == 4
    assert out_size == 8


def test_check_uniform_when_both_not_none():
    dataloader = simple_dataloader()

    embedding_num = LinearEmbedding(embedding_size=10)
    embedding_cat = DefaultEmbedding(embedding_size=10, alpha=2)
    embedding_num, embedding_cat = fit_embeddings(dataloader, embedding_num, embedding_cat)
    num_fields, emb_size, out_size = check_uniform_embeddings(embedding_num, embedding_cat)
    assert num_fields == 5
    assert emb_size == 10
    assert out_size == 50

    embedding_num = LinearEmbedding(embedding_size=8)
    embedding_cat = BasicEmbedding(embedding_size=8)
    embedding_num, embedding_cat = fit_embeddings(dataloader, embedding_num, embedding_cat)
    num_fields, emb_size, out_size = check_uniform_embeddings(embedding_num, embedding_cat)
    assert num_fields == 5
    assert emb_size == 8
    assert out_size == 40

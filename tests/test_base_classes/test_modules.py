
import pytest
import torch
from torch import nn
from pytorch_lightning import Trainer

from xynn.embedding import RaggedEmbedding, LinearEmbedding
from ..common import SimpleMLP, simple_train_inputs, SimpleEmbedding


def test_that_basenn_raises_error_for_bad_task_value():
    with pytest.raises(
        ValueError,
        match=(
            "task classy-regression not recognized; "
            "should be 'regression' or 'classification'"
        )
    ):
        SimpleMLP(task="classy-regression")


def test_that_basenn_raises_error_when_configuring_optimizer_without_setting():
    mlp = SimpleMLP()
    with pytest.raises(
        RuntimeError,
        match=(
            "The optimizer and learning rate info needs to first be set "
            "with the `set_optimizer` method"
        ),
    ):
        mlp.configure_optimizers()


def test_num_parameters_against_known_value():
    mlp = SimpleMLP()
    assert mlp.num_parameters() == 11 * 7 + 7 + 7 * 3 + 3
    mlp.embedding_num = SimpleEmbedding(20, 3)
    assert mlp.num_parameters() == 20 * 3 + 11 * 7 + 7 + 7 * 3 + 3


def test_embedding_sum_against_known_value():
    mlp = SimpleMLP()
    mlp.embedding_num = SimpleEmbedding(20, 3)
    assert mlp.embedding_sum() == mlp.embedding_num.weight_sum()

    mlp.embedding_num.embedding.weight = nn.Parameter(
        torch.tensor([[-1, 0, 1]] * 20, dtype=torch.float32)
    )
    mlp.embedding_cat = SimpleEmbedding(10, 4)
    mlp.embedding_cat.embedding.weight = nn.Parameter(
        torch.tensor([[-1, 0, 1, 2]] * 10, dtype=torch.float32)
    )
    assert mlp.embedding_sum() == (80, 100)


def test_that_embed_raises_error_when_both_Xs_none():
    mlp = SimpleMLP()
    with pytest.raises(ValueError, match="X_num and X_cat cannot both be None"):
        mlp.embed(None, None)


def test_that_embed_raises_error_for_bad_num_dim():
    X_num = torch.tensor([[0, 5, 15], [1, 6, 16]])
    X_cat = torch.tensor([[0, 5], [1, 6]])
    mlp = SimpleMLP(
        embedding_num=SimpleEmbedding(20, 3),
        embedding_cat=SimpleEmbedding(10, 3),
    )

    with pytest.raises(ValueError, match="num_dim should be 2 or 3, got 4"):
        mlp.embed(X_num, X_cat, num_dim=4)

    with pytest.raises(ValueError, match="num_dim should be 2 or 3, got any"):
        mlp.embed(X_num, X_cat, num_dim="any")


def test_that_embed_raises_error_for_num_dim_3_with_ragged_embedding():
    X_num = torch.tensor([[0, 5, 15], [1, 6, 16]])
    X_cat = torch.tensor([[0, 5], [1, 6]])
    mlp = SimpleMLP(
        embedding_num=SimpleEmbedding(20, 3),
        embedding_cat=RaggedEmbedding(),
    )
    with pytest.raises(ValueError, match="cannot use num_dim=3 with ragged embeddings"):
        mlp.embed(X_num, X_cat, num_dim=3)


def test_embed():
    X_num = torch.tensor([[0, 5, 15], [1, 6, 16]])
    X_cat = torch.tensor([[0, 5], [1, 6]])
    mlp = SimpleMLP(
        embedding_num=SimpleEmbedding(20, 3),
        embedding_cat=SimpleEmbedding(10, 3),
    )
    X_num_emb, X_cat_emb = mlp.embed(X_num, X_cat, concat=False)
    assert X_num_emb.shape == (2, 3, 3)
    assert X_cat_emb.shape == (2, 2, 3)


def test_embed_when_numeric_data_has_zero_columns():
    mlp = SimpleMLP(
        embedding_num=SimpleEmbedding(20, 3),
        embedding_cat=SimpleEmbedding(10, 4),
    )
    X_cat = torch.tensor([[0, 5], [1, 6]])
    X_num = torch.empty(size=(2, 0))

    X_num_emb, X_cat_emb = mlp.embed(X_num, X_cat, concat=False)
    assert X_num_emb.shape == (2, 0, 4)
    assert X_cat_emb.shape == (2, 2, 4)

    embedded = mlp.embed(X_num, X_cat)
    assert embedded.shape == (2, 2, 4), str(embedded.shape)

    embedded = mlp.embed(X_num, X_cat, num_dim=2)
    assert embedded.shape == (2, 8), str(embedded.shape)

    # show that None works the same as the empty X_num above
    assert torch.all(mlp.embed(X_num, X_cat) == mlp.embed(None, X_cat)).item()


def test_embed_when_categorical_data_has_zero_columns():
    mlp = SimpleMLP(
        embedding_num=SimpleEmbedding(20, 3),
        embedding_cat=SimpleEmbedding(10, 4),
    )
    X_num = torch.tensor([[0, 5, 15], [1, 6, 16]])
    X_cat = torch.empty(size=(2, 0))

    X_num_emb, X_cat_emb = mlp.embed(X_num, X_cat, concat=False)
    assert X_num_emb.shape == (2, 3, 3)
    assert X_cat_emb.shape == (2, 0, 3)

    embedded = mlp.embed(X_num, X_cat)
    assert embedded.shape == (2, 3, 3), str(embedded.shape)

    embedded = mlp.embed(X_num, X_cat, num_dim=2)
    assert embedded.shape == (2, 9), str(embedded.shape)

    # show that None works the same as the empty X_cat above
    assert torch.all(mlp.embed(X_num, X_cat) == mlp.embed(X_num, None)).item()


def test_embed_results_without_numeric_embeddings():
    X_num = torch.tensor([[0, 5, 15], [1, 6, 16]])
    X_cat = torch.tensor([[0, 5], [1, 6]])
    mlp = SimpleMLP(
        embedding_num=None,
        embedding_cat=SimpleEmbedding(10, 4),
    )

    X_num_emb, X_cat_emb = mlp.embed(X_num, X_cat, concat=False)
    assert X_num_emb.shape == (2, 3, 1)
    assert X_cat_emb.shape == (2, 2, 4)

    X_num_emb, X_cat_emb = mlp.embed(X_num, X_cat, num_dim=2, concat=False)
    assert X_num_emb.shape == (2, 3)
    assert X_cat_emb.shape == (2, 8)

    embedded = mlp.embed(X_num, X_cat, num_dim=2)
    assert embedded.shape == (2, 11)

    # cannot concat with different numbers in dim 2
    with pytest.raises(RuntimeError):
        mlp.embed(X_num, X_cat)


def test_embed_results_without_categorical_embeddings():
    X_num = torch.tensor([[0, 5, 15], [1, 6, 16]])
    X_cat = torch.tensor([[0, 5], [1, 6]])
    mlp = SimpleMLP(
        embedding_num=SimpleEmbedding(20, 3),
        embedding_cat=None,
    )

    X_num_emb, X_cat_emb = mlp.embed(X_num, X_cat, concat=False)
    assert X_num_emb.shape == (2, 3, 3)
    assert X_cat_emb.shape == (2, 0, 3)

    embedded = mlp.embed(X_num, X_cat)
    assert embedded.shape == (2, 3, 3)

    embedded = mlp.embed(X_num, X_cat, num_dim=2)
    assert embedded.shape == (2, 9)


def test_embed_results_without_any_embeddings():
    X_num = torch.tensor([[0, 5, 15], [1, 6, 16]])
    X_cat = torch.tensor([[0, 5], [1, 6]])
    mlp = SimpleMLP(
        embedding_num=None,
        embedding_cat=None,
    )

    X_num_emb, X_cat_emb = mlp.embed(X_num, X_cat, concat=False)
    assert torch.all(X_num_emb == X_num.reshape((2, 3, 1))).item()
    assert X_cat_emb.shape == (2, 0, 1)

    embedded = mlp.embed(X_num, X_cat)
    assert torch.all(embedded == X_num.reshape((2, 3, 1))).item()

    embedded = mlp.embed(X_num, X_cat, num_dim=2)
    assert torch.all(embedded == X_num).item()


def test_embed_results_with_ragged_embedding():
    X_num = torch.tensor([[0, 5, 15], [1, 6, 16]])
    X_cat = torch.tensor([[0, 5], [1, 6]])

    mlp = SimpleMLP(
        embedding_num=None,
        embedding_cat=RaggedEmbedding(embedding_size=(3, 4)).fit(X_cat),
    )
    X_num_emb, X_cat_emb = mlp.embed(X_num, X_cat, num_dim=2, concat=False)
    assert X_num_emb.shape == (2, 3)
    assert X_cat_emb.shape == (2, 7)

    embedded = mlp.embed(X_num, X_cat, num_dim=2)
    assert embedded.shape == (2, 10)

    mlp = SimpleMLP(
        embedding_num=LinearEmbedding(embedding_size=3).fit(X_num),
        embedding_cat=RaggedEmbedding(embedding_size=(3, 4)).fit(X_cat),
    )
    X_num_emb, X_cat_emb = mlp.embed(X_num, X_cat, num_dim=2, concat=False)
    assert X_num_emb.shape == (2, 9)
    assert X_cat_emb.shape == (2, 7)

    embedded = mlp.embed(X_num, X_cat, num_dim=2)
    assert embedded.shape == (2, 16)


def test_that_pytorch_lightning_runs_without_error():
    model, train_dl, valid_dl = simple_train_inputs(configure=False)
    test_dl = valid_dl  # just to check that the code runs
    trainer = Trainer(max_epochs=5)
    trainer.fit(model, train_dl, valid_dl)
    trainer.test(model, test_dl)

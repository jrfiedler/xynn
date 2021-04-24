
import pytest
import torch
from torch import nn
from pytorch_lightning import Trainer

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


def test_embed():
    X_num = torch.tensor([[0, 5, 15], [1, 6, 16]])
    X_cat = torch.tensor([[0, 5], [1, 6]])
    mlp = SimpleMLP()

    mlp.embedding_num = SimpleEmbedding(20, 3)
    mlp.embedding_cat = SimpleEmbedding(10, 4)
    X_num_emb, X_cat_emb = mlp.embed(X_num, X_cat)
    assert X_num_emb.shape == (2, 3, 3)
    assert X_cat_emb.shape == (2, 2, 4)


def test_embed_when_data_has_zero_columns():
    mlp = SimpleMLP()
    mlp.embedding_num = SimpleEmbedding(20, 3)
    mlp.embedding_cat = SimpleEmbedding(10, 4)

    X_cat = torch.tensor([[0, 5], [1, 6]])
    X_num = torch.empty(size=(2, 0))
    X_num_emb, X_cat_emb = mlp.embed(X_num, X_cat)
    assert X_num_emb.shape == (2, 0, 4)
    assert X_cat_emb.shape == (2, 2, 4)

    X_num = torch.tensor([[0, 5, 15], [1, 6, 16]])
    X_cat = torch.empty(size=(2, 0))
    X_num_emb, X_cat_emb = mlp.embed(X_num, X_cat)
    assert X_num_emb.shape == (2, 3, 3)
    assert X_cat_emb.shape == (2, 0, 3)


def test_embed_results_without_one_or_both_embeddings():
    X_num = torch.tensor([[0, 5, 15], [1, 6, 16]])
    X_cat = torch.tensor([[0, 5], [1, 6]])
    mlp = SimpleMLP()

    mlp.embedding_num = None
    mlp.embedding_cat = SimpleEmbedding(10, 4)
    X_num_emb, X_cat_emb = mlp.embed(X_num, X_cat)
    assert X_num_emb.shape == (2, 0, 4)
    assert X_cat_emb.shape == (2, 2, 4)

    mlp.embedding_num = SimpleEmbedding(20, 3)
    mlp.embedding_cat = None
    X_num_emb, X_cat_emb = mlp.embed(X_num, X_cat)
    assert X_num_emb.shape == (2, 3, 3)
    assert X_cat_emb.shape == (2, 0, 3)

    mlp.embedding_num = None
    mlp.embedding_cat = None
    X_num_emb, X_cat_emb = mlp.embed(X_num, X_cat)
    assert X_num_emb is X_num
    assert X_cat_emb is X_cat


def test_that_pytorch_lightning_runs_without_error():
    model, train_dl, valid_dl = simple_train_inputs(configure=False)
    test_dl = valid_dl  # just to check that the code runs
    trainer = Trainer(max_epochs=5)
    trainer.fit(model, train_dl, valid_dl)
    trainer.test(model, test_dl)

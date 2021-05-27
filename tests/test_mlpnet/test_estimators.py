import json
import random
from tempfile import NamedTemporaryFile

import torch
from torch import nn
import numpy as np

from xynn.base_classes.estimators import _set_seed
from xynn.embedding import DenseEmbedding, RaggedEmbedding
from xynn.mlpnet import MLPRegressor, MLPClassifier

from ..common import check_estimator_learns


def test_that_basic_params_are_passed_to_mlpnet_module():
    X = torch.rand((100, 10)) - 0.5
    y = X[:, 0] - X[:, 1] + X[:, 2] - X[:, 4] + 2 * X[:, 6] - 2 * X[:, 8]
    estimator = MLPRegressor(
        embedding_cat=None,
        embedding_l1_reg=0.1,
        mlp_l2_reg=0.2,
    )
    estimator.fit(
        X_num=X,
        X_cat=None,
        y=y,
        optimizer=torch.optim.Adam,
        opt_kwargs={"lr": 1e-1},
        num_epochs=1,
    )

    model = estimator._model

    assert model.task == "regression"
    assert model.num_epochs == 1
    assert isinstance(model.loss_fn, nn.MSELoss)
    assert model.embedding_num is not None
    assert model.embedding_cat is None
    assert model.embedding_l1_reg == 0.1
    assert model.embedding_l2_reg == 0.0
    assert model.mlp_l1_reg == 0.0
    assert model.mlp_l2_reg == 0.2
    assert model.optimizer is not None
    assert model.optimizer_info != {}
    assert model.scheduler == {}
    assert model._device == torch.device("cpu")


def test_that_mlpregressor_learns():
    _set_seed(10101)
    X = torch.rand((100, 10)) - 0.5
    y = X[:, 0] - X[:, 1] + X[:, 2] - X[:, 4] + 2 * X[:, 6] - 2 * X[:, 8]
    estimator = MLPRegressor(
        mlp_hidden_sizes=[10, 8, 8, 6],
        mlp_use_bn=False,
        mlp_leaky_gate=False,
        mlp_use_skip=False,
    )
    check_estimator_learns(estimator, task="regression", data=(X, None, y))


def test_that_mlpclassifier_learns():
    _set_seed(10101)
    X = torch.rand((100, 10)) - 0.5
    y_cont = X[:, 0] - X[:, 1] + X[:, 2] - X[:, 4] + 2 * X[:, 6] - 2 * X[:, 8]
    y = (y_cont > 0)
    estimator = MLPClassifier(
        mlp_hidden_sizes=[10, 8, 8, 6],
        mlp_use_bn=False,
        mlp_leaky_gate=False,
        mlp_use_skip=False,
    )
    check_estimator_learns(estimator, task="regression", data=(X, None, y))


def test_that_mlpregressor_allows_dense_and_ragged_embeddings():
    _set_seed(10101)
    estimator = MLPClassifier(
        embedding_num=DenseEmbedding(),
        embedding_cat=RaggedEmbedding(),
        mlp_hidden_sizes=[10, 8, 8, 6],
        mlp_use_bn=False,
        mlp_leaky_gate=False,
        mlp_use_skip=False,
    )
    check_estimator_learns(estimator, task="regression")


def test_that_mlpclassifier_doesnt_require_numeric_embedding():
    _set_seed(10101)
    estimator = MLPClassifier(
        embedding_num=None,
        mlp_hidden_sizes=[10, 8, 8, 6],
        mlp_use_bn=False,
        mlp_leaky_gate=False,
        mlp_use_skip=False,
    )
    check_estimator_learns(estimator, task="regression")

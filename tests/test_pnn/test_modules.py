
import random

import torch
from torch import nn
import numpy as np
import pytest

from xynn.base_classes.estimators import _set_seed
from xynn.base_classes.modules import BaseNN
from xynn.pnn.modules import InnerProduct, OuterProduct, PNNCore, PNN, PNNPlus
from xynn.embedding import LinearEmbedding, BasicEmbedding
from xynn.mlp import LeakyGate, GhostBatchNorm

from ..common import simple_train_inputs, simple_model_train_loop


def test_inner_product():
    prod = InnerProduct(15, 10)
    assert isinstance(prod.weights, nn.Parameter)
    assert prod.weights.shape == (10, 15)
    X = torch.rand((30, 15, 8))
    assert prod(X).shape == (30, 10)


def test_outer_product():
    prod = OuterProduct(8, 10)
    assert isinstance(prod.weights, nn.Parameter)
    assert prod.weights.shape == (10, 8, 8)
    X = torch.rand((30, 15, 8))
    assert prod(X).shape == (30, 10)


def test_that_pnn_modules_raise_error_for_bad_product_name():
    X = torch.randint(0, 10, (100, 10))
    embedding_num = LinearEmbedding(embedding_size=3).fit(X)

    with pytest.raises(
        ValueError, match="pnn_product_type should be 'inner', 'outer', or 'both'"
    ):
        model = PNN(
            task="classification",
            output_size=3,
            embedding_num=embedding_num,
            embedding_cat=None,
            pnn_product_type="either",
        )

    with pytest.raises(
        ValueError, match="pnn_product_type should be 'inner', 'outer', or 'both'"
    ):
        model = PNNPlus(
            task="classification",
            output_size=3,
            embedding_num=embedding_num,
            embedding_cat=None,
            pnn_product_type="far_outer",
        )


def test_that_pnn_modules_subclass_basenn():
    assert issubclass(PNN, BaseNN)
    assert issubclass(PNNPlus, BaseNN)


def test_that_pnn_modules_use_basenn_init():
    X = torch.randint(0, 10, (100, 10))
    embedding_num = LinearEmbedding(embedding_size=3).fit(X)

    model = PNN(
        task="classification",
        output_size=3,
        embedding_num=embedding_num,
        embedding_cat=None,
        embedding_l2_reg=0.2,
        mlp_l1_reg=0.1
    )

    assert model.task == "classification"
    assert model.num_epochs == 0
    assert isinstance(model.loss_fn, nn.CrossEntropyLoss)
    assert model.embedding_num is embedding_num
    assert model.embedding_cat is None
    assert model.embedding_l1_reg == 0.0
    assert model.embedding_l2_reg == 0.2
    assert model.mlp_l1_reg == 0.1
    assert model.mlp_l2_reg == 0.0
    assert model.optimizer is None
    assert model.optimizer_info == {}
    assert model.scheduler == {}
    assert model._device == "cpu"

    model = PNNPlus(
        task="regression",
        output_size=3,
        embedding_num=embedding_num,
        embedding_cat=None,
        embedding_l2_reg=0.2,
        mlp_l1_reg=0.1
    )

    assert model.task == "regression"
    assert model.num_epochs == 0
    assert isinstance(model.loss_fn, nn.MSELoss)
    assert model.embedding_num is embedding_num
    assert model.embedding_cat is None
    assert model.embedding_l1_reg == 0.0
    assert model.embedding_l2_reg == 0.2
    assert model.mlp_l1_reg == 0.1
    assert model.mlp_l2_reg == 0.0
    assert model.optimizer is None
    assert model.optimizer_info == {}
    assert model.scheduler == {}
    assert model._device == "cpu"


def test_that_pnn_parameters_are_passed_to_submodules():
    X = torch.randint(0, 10, (100, 10))
    embedding_cat = BasicEmbedding(embedding_size=3).fit(X)
    model = PNN(
        task="classification",
        output_size=3,
        embedding_num=None,
        embedding_cat=embedding_cat,
        mlp_activation=nn.ReLU,
        mlp_hidden_sizes=(512, 128, 32),
        mlp_use_bn=False,
        mlp_use_skip=False,
        use_leaky_gate=False,
    )

    expected_classes = [
        nn.Linear,
        nn.ReLU,
        nn.Linear,
        nn.ReLU,
        nn.Linear,
        nn.ReLU,
        nn.Linear,
    ]
    mlp = model.pnn.mlp
    assert len(mlp.main_layers) == len(expected_classes)
    for layer, expected_class in zip(mlp.main_layers, expected_classes):
        assert isinstance(layer, expected_class)
    assert mlp.skip_layers is None
    assert model.pnn.product_type == "outer"
    assert model.pnn.inner is None
    assert isinstance(model.pnn.outer, OuterProduct)


def test_that_pnnplus_parameters_are_passed_to_submodules():
    X = torch.randint(0, 10, (100, 10))
    embedding_num = LinearEmbedding(embedding_size=3).fit(X)
    embedding_cat = BasicEmbedding(embedding_size=3).fit(X)
    model = PNNPlus(
        task="classification",
        output_size=3,
        embedding_num=embedding_num,
        embedding_cat=embedding_cat,
        pnn_product_type="both",
        mlp_hidden_sizes=(512, 64),
        mlp_use_bn=True,
        mlp_dropout=0.1,
        mlp_use_skip=True,
        use_leaky_gate=True,
    )

    expected_classes = [
        LeakyGate,
        nn.Dropout,
        nn.Linear,
        nn.BatchNorm1d,
        nn.LeakyReLU,
        nn.Dropout,
        nn.Linear,
        nn.BatchNorm1d,
        nn.LeakyReLU,
        nn.Dropout,
        nn.Linear,
    ]
    mlp = model.pnn.mlp
    assert len(mlp.main_layers) == len(expected_classes)
    for layer, expected_class in zip(mlp.main_layers, expected_classes):
        assert isinstance(layer, expected_class)

    expected_classes = [LeakyGate, nn.Linear]
    assert len(mlp.skip_layers) == len(expected_classes)
    for layer, expected_class in zip(mlp.skip_layers, expected_classes):
        assert isinstance(layer, expected_class)

    assert model.pnn.product_type == "both"
    assert isinstance(model.pnn.inner, InnerProduct)
    assert isinstance(model.pnn.outer, OuterProduct)


def test_pnnplus_parameters_with_ghost_batch():
    X = torch.randint(0, 10, (100, 10))
    embedding_num = LinearEmbedding(embedding_size=3).fit(X)
    embedding_cat = BasicEmbedding(embedding_size=3).fit(X)
    model = PNNPlus(
        task="classification",
        output_size=3,
        embedding_num=embedding_num,
        embedding_cat=embedding_cat,
        pnn_product_type="both",
        mlp_hidden_sizes=(512, 64),
        mlp_use_bn=True,
        mlp_ghost_batch=32,
        mlp_use_skip=False,
        use_leaky_gate=True,
    )

    expected_classes = [
        LeakyGate,
        nn.Linear,
        GhostBatchNorm,
        nn.LeakyReLU,
        nn.Linear,
        GhostBatchNorm,
        nn.LeakyReLU,
        nn.Linear,
    ]
    mlp = model.pnn.mlp
    assert len(mlp.main_layers) == len(expected_classes)
    for layer, expected_class in zip(mlp.main_layers, expected_classes):
        assert isinstance(layer, expected_class)
    assert mlp.skip_layers is None


def test_that_pnn_diagram_exists_and_prints_something(capsys):
    PNN.diagram()
    captured = capsys.readouterr()
    assert len(captured.out.split("\n")) > 5


def test_that_pnnplus_diagram_exists_and_prints_something(capsys):
    PNNPlus.diagram()
    captured = capsys.readouterr()
    assert len(captured.out.split("\n")) > 5


def test_pnn_mlp_weight():
    X = torch.randint(0, 10, (100, 10))
    embedding_num = LinearEmbedding(embedding_size=3).fit(X)

    model = PNN(
        task="regression",
        output_size=1,
        embedding_num=embedding_num,
        embedding_cat=None,
        mlp_use_bn=False,
        mlp_use_skip=False,
        use_leaky_gate=False,
    )
    mlp = model.pnn.mlp
    w1, w2 = model.mlp_weight_sum()
    exp_w1 = sum(l.weight.abs().sum().item() for l in mlp.main_layers[::2])
    exp_w2 = sum((l.weight ** 2).sum().item() for l in mlp.main_layers[::2])
    assert np.isclose(w1.item(), exp_w1)
    assert np.isclose(w2.item(), exp_w2)


def test_pnnplus_mlp_weight():
    X = torch.randint(0, 10, (100, 10))
    embedding_num = LinearEmbedding(embedding_size=3).fit(X)

    model = PNNPlus(
        task="regression",
        output_size=1,
        embedding_num=embedding_num,
        embedding_cat=None,
        mlp_use_bn=False,
        mlp_use_skip=False,
        use_leaky_gate=False,
    )
    mlp1 = model.pnn.mlp
    mlp2 = model.mlp
    w1, w2 = model.mlp_weight_sum()
    exp_w1 = sum(l.weight.abs().sum().item() for l in mlp1.main_layers[::2])
    exp_w2 = sum((l.weight ** 2).sum().item() for l in mlp1.main_layers[::2])
    exp_w1 += sum(l.weight.abs().sum().item() for l in mlp2.main_layers[::2])
    exp_w2 += sum((l.weight ** 2).sum().item() for l in mlp2.main_layers[::2])
    assert np.isclose(w1.item(), exp_w1)
    assert np.isclose(w2.item(), exp_w2)


def test_that_pnn_learns():
    _set_seed(10101)

    X = torch.randint(0, 10, (100, 10))
    y = torch.rand((100, 1)) * 6 - 3
    embedding_num = LinearEmbedding(embedding_size=3).fit(X)
    model = PNN(
        task="regression",
        output_size=1,
        embedding_num=embedding_num,
        embedding_cat=None,
        mlp_hidden_sizes=[10, 8, 6],
        mlp_use_bn=False,
        mlp_use_skip=False,
        use_leaky_gate=False,
    )
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)
    loss_vals = simple_model_train_loop(model, X, None, y, loss_func, optimizer, num_epochs=5)
    assert loss_vals[0] > loss_vals[-1]


def test_that_pnnplus_learns():
    _set_seed(10101)

    X = torch.randint(0, 10, (100, 10))
    y = torch.rand((100, 1)) * 6 - 3
    embedding_num = LinearEmbedding(embedding_size=3).fit(X)
    model = PNNPlus(
        task="regression",
        output_size=1,
        embedding_num=embedding_num,
        embedding_cat=None,
        mlp_hidden_sizes=[10, 8, 6],
        mlp_use_bn=False,
        mlp_use_skip=False,
        use_leaky_gate=False,
    )
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)
    loss_vals = simple_model_train_loop(model, X, None, y, loss_func, optimizer, num_epochs=5)
    assert loss_vals[0] > loss_vals[-1]

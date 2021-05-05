
import random

import torch
from torch import nn
import numpy as np

from xynn.base_classes.estimators import _set_seed
from xynn.base_classes.modules import BaseNN
from xynn.mlpnet.modules import MLPNet
from xynn.embedding import LinearEmbedding
from xynn.mlp import LeakyGate

from ..common import simple_train_inputs, simple_model_train_loop, SimpleEmbedding


def test_that_mlpnet_subclasses_basenn():
    assert issubclass(MLPNet, BaseNN)


def test_that_mlpnet_uses_basenn_init():
    embedding_num = SimpleEmbedding(20, 3)
    model = MLPNet(
        task="classification",
        output_size=3,
        embedding_num=embedding_num,
        embedding_cat=None,
        num_numeric_fields=20,
    )

    assert model.task == "classification"
    assert model.num_epochs == 0
    assert isinstance(model.loss_fn, nn.CrossEntropyLoss)
    assert model.embedding_num is embedding_num
    assert model.embedding_cat is None
    assert model.optimizer is None
    assert model.optimizer_info == {}
    assert model.scheduler == {}
    assert model._device == "cpu"


def test_that_activation_and_sizes_are_passed_to_mlp_module():
    embedding_num = SimpleEmbedding(20, 3)
    model = MLPNet(
        task="classification",
        output_size=3,
        embedding_num=embedding_num,
        embedding_cat=None,
        num_numeric_fields=20,
        mlp_activation=nn.ReLU,
        mlp_hidden_sizes=(512, 128, 32),
        mlp_use_bn=False,
        mlp_leaky_gate=False,
        mlp_use_skip=False,
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
    assert len(model.mlp.main_layers) == len(expected_classes)
    for layer, expected_class in zip(model.mlp.main_layers, expected_classes):
        assert isinstance(layer, expected_class)
    assert model.mlp.skip_layers is None


def test_that_more_parameters_are_passed_to_mlp_module():
    embedding_num = SimpleEmbedding(20, 3)
    model = MLPNet(
        task="classification",
        output_size=3,
        embedding_num=embedding_num,
        embedding_cat=None,
        num_numeric_fields=20,
        mlp_hidden_sizes=(512, 64),
        mlp_use_bn=True,
        mlp_dropout=0.1,
        mlp_leaky_gate=True,
        mlp_use_skip=True,
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
    assert len(model.mlp.main_layers) == len(expected_classes)
    for layer, expected_class in zip(model.mlp.main_layers, expected_classes):
        assert isinstance(layer, expected_class)

    expected_classes = [LeakyGate, nn.Linear]
    assert len(model.mlp.skip_layers) == len(expected_classes)
    for layer, expected_class in zip(model.mlp.skip_layers, expected_classes):
        assert isinstance(layer, expected_class)


def test_that_diagram_exists_and_prints_something(capsys):
    MLPNet.diagram()
    captured = capsys.readouterr()
    assert len(captured.out.split("\n")) > 5


def test_mlp_weight():
    model = MLPNet(
        task="regression",
        output_size=1,
        embedding_num=SimpleEmbedding(20, 3),
        embedding_cat=None,
        num_numeric_fields=3,
        mlp_use_bn=False,
        mlp_leaky_gate=False,
        mlp_use_skip=False,
    )
    mlp = model.mlp
    w1, w2 = model.mlp_weight_sum()
    exp_w1 = sum(l.weight.abs().sum().item() for l in mlp.main_layers[::2])
    exp_w2 = sum((l.weight ** 2).sum().item() for l in mlp.main_layers[::2])
    assert np.isclose(w1, exp_w1)
    assert np.isclose(w2, exp_w2)


def test_that_mlpnet_learns():
    _set_seed(10101)

    X = torch.randint(0, 10, (100, 10))
    y = torch.rand((100, 1)) * 6 - 3
    embedding_num = LinearEmbedding(embedding_size=3).fit(X)
    model = MLPNet(
        task="regression",
        output_size=1,
        embedding_num=embedding_num,
        embedding_cat=None,
        num_numeric_fields=10,
        mlp_hidden_sizes=[10, 8, 8, 6],
        mlp_use_bn=False,
        mlp_leaky_gate=False,
        mlp_use_skip=False,
    )
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)
    loss_vals = simple_model_train_loop(model, X, None, y, loss_func, optimizer, num_epochs=5)
    assert loss_vals[0] > loss_vals[-1]

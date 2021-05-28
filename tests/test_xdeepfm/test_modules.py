
import random

import torch
from torch import nn
import numpy as np
import pytest

from xynn.base_classes.estimators import _set_seed
from xynn.base_classes.modules import BaseNN
from xynn.xdeepfm.modules import CIN
from xynn.xdeepfm import XDeepFM
from xynn.embedding import LinearEmbedding, BasicEmbedding
from xynn.mlp import LeakyGate

from ..common import simple_train_inputs, simple_model_train_loop


def test_that_cin_raises_error_for_bad_layer_sizes_when_not_full_agg():
    with pytest.raises(
        ValueError,
        match="when using full_agg=False, all but the last layer size must be even"
    ):
        CIN(num_fields=5, layer_sizes=(127, 127), full_agg=False)


def test_cin_layers_without_activation_and_bn():
    cin = CIN(num_fields=5, use_bn=False)
    assert len(cin.convs) == 2
    assert len(cin.actns) == 2
    assert len(cin.norms) == 2
    assert all(isinstance(conv, nn.Conv1d) for conv in cin.convs)
    assert all(isinstance(actn, nn.Identity) for actn in cin.actns)
    assert all(isinstance(norm, nn.Identity) for norm in cin.norms)


def test_cin_layers_with_activation_and_bn():
    cin = CIN(num_fields=5, activation=nn.ReLU, use_bn=True)
    assert len(cin.convs) == 2
    assert len(cin.actns) == 2
    assert len(cin.norms) == 2
    assert all(isinstance(conv, nn.Conv1d) for conv in cin.convs)
    assert all(isinstance(actn, nn.ReLU) for actn in cin.actns)
    assert all(isinstance(norm, nn.BatchNorm1d) for norm in cin.norms)


def test_cin_shape_of_output():
    x = torch.rand((20, 5, 8))

    cin = CIN(num_fields=5, layer_sizes=(10,))
    out = cin(x)
    assert out.shape == (20, 10)

    cin = CIN(num_fields=5, layer_sizes=(10, 10))
    out = cin(x)
    assert out.shape == (20, 15)

    cin = CIN(num_fields=5, layer_sizes=(10, 10), full_agg=True)
    out = cin(x)
    assert out.shape == (20, 20)


def test_that_xdeepfm_module_subclasses_basenn():
    assert issubclass(XDeepFM, BaseNN)


def test_that_xdeepfm_uses_basenn_init():
    X = torch.randint(0, 10, (100, 10))
    embedding_num = LinearEmbedding(embedding_size=3).fit(X)

    model = XDeepFM(
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


def test_that_xdeepfm_parameters_are_passed_to_submodules():
    X = torch.randint(0, 10, (100, 10))
    embedding_cat = BasicEmbedding(embedding_size=3).fit(X)
    model = XDeepFM(
        task="classification",
        output_size=3,
        embedding_num=None,
        embedding_cat=embedding_cat,
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
    for mlp in (model.mlp, model.cin_final):
        assert len(mlp.main_layers) == len(expected_classes)
        for layer, expected_class in zip(mlp.main_layers, expected_classes):
            assert isinstance(layer, expected_class)
        assert mlp.skip_layers is None

    assert model.use_residual
    assert isinstance(model.cin, CIN) and len(model.cin.convs) == 2
    assert model.mix.requires_grad


def test_that_xdeepfm_parameters_are_passed_to_submodules_other_params():
    X = torch.randint(0, 10, (100, 10))
    embedding_cat = BasicEmbedding(embedding_size=3).fit(X)
    model = XDeepFM(
        task="classification",
        output_size=3,
        embedding_num=None,
        embedding_cat=embedding_cat,
        cin_layer_sizes=(20, 20, 20),
        cin_use_residual=False,
        mlp_use_skip=True,
    )

    expected_classes = [
        LeakyGate,
        nn.Linear,
        nn.BatchNorm1d,
        nn.LeakyReLU,
        nn.Linear,
        nn.BatchNorm1d,
        nn.LeakyReLU,
        nn.Linear,
        nn.BatchNorm1d,
        nn.LeakyReLU,
        nn.Linear,
        nn.BatchNorm1d,
        nn.LeakyReLU,
        nn.Linear,
    ]
    for mlp in (model.mlp, model.cin_final):
        assert len(mlp.main_layers) == len(expected_classes)
        for layer, expected_class in zip(mlp.main_layers, expected_classes):
            assert isinstance(layer, expected_class)
        assert mlp.skip_layers is not None

    assert not model.use_residual
    assert isinstance(model.cin, CIN) and len(model.cin.convs) == 3
    assert model.mix.requires_grad


def test_that_xdeepfm_diagram_exists_and_prints_something(capsys):
    XDeepFM.diagram()
    captured = capsys.readouterr()
    assert len(captured.out.split("\n")) > 5


def test_xdeepfm_mlp_weight():
    X = torch.randint(0, 10, (100, 10))
    embedding_num = LinearEmbedding(embedding_size=3).fit(X)

    # without Linear after CIN
    model = XDeepFM(
        task="regression",
        output_size=1,
        embedding_num=embedding_num,
        embedding_cat=None,
        cin_use_mlp=False,
        mlp_use_bn=False,
        mlp_leaky_gate=False,
        mlp_use_skip=False,
    )

    exp_w1 = 0
    exp_w2 = 0
    for mlp in (model.mlp, model.cin_final):
        exp_w1 += sum(l.weight.abs().sum().item() for l in mlp.main_layers[::2])
        exp_w2 += sum((l.weight ** 2).sum().item() for l in mlp.main_layers[::2])

    w1, w2 = model.mlp_weight_sum()
    assert np.isclose(w1.item(), exp_w1)
    assert np.isclose(w2.item(), exp_w2)

    # with MLP after CIN
    model = XDeepFM(
        task="regression",
        output_size=1,
        embedding_num=embedding_num,
        embedding_cat=None,
        cin_use_mlp=True,
        mlp_use_bn=False,
        mlp_leaky_gate=False,
        mlp_use_skip=False,
    )

    exp_w1 = 0
    exp_w2 = 0
    for mlp in (model.mlp, model.cin_final):
        exp_w1 += sum(l.weight.abs().sum().item() for l in mlp.main_layers[::2])
        exp_w2 += sum((l.weight ** 2).sum().item() for l in mlp.main_layers[::2])

    w1, w2 = model.mlp_weight_sum()
    assert np.isclose(w1.item(), exp_w1)
    assert np.isclose(w2.item(), exp_w2)


def test_that_xdeepfm_learns():
    _set_seed(10101)

    X_num = torch.randint(0, 10, (100, 10))
    X_cat = torch.randint(0, 5, (100, 1))
    y = (
        (X_cat - 2) + X_num[:, ::2].sum(dim=1) - X_num[:, 1::2].sum(dim=1)
    ).to(dtype=torch.float)
    
    model = XDeepFM(
        task="regression",
        output_size=1,
        embedding_num=LinearEmbedding(embedding_size=3).fit(X_num),
        embedding_cat=BasicEmbedding(embedding_size=3).fit(X_cat),
        mlp_hidden_sizes=[10, 8, 6],
        mlp_use_bn=False,
        mlp_leaky_gate=False,
        mlp_use_skip=False,
    )
    
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)
    loss_vals = simple_model_train_loop(
        model, X_num, X_cat, y, loss_func, optimizer, num_epochs=5
    )
    
    assert loss_vals[0] > loss_vals[-1]


def test_that_xdeepfm_learns_with_other_params():
    _set_seed(10101)

    X_num = torch.randint(0, 10, (100, 10))
    X_cat = torch.randint(0, 5, (100, 1))
    y = (
        (X_cat - 2) + X_num[:, ::2].sum(dim=1) - X_num[:, 1::2].sum(dim=1)
    ).to(dtype=torch.float)
    
    model = XDeepFM(
        task="regression",
        output_size=1,
        embedding_num=LinearEmbedding(embedding_size=3).fit(X_num),
        embedding_cat=BasicEmbedding(embedding_size=3).fit(X_cat),
        mlp_hidden_sizes=[],
        mlp_use_bn=False,
        mlp_leaky_gate=False,
        mlp_use_skip=False,
    )
    
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)
    loss_vals = simple_model_train_loop(
        model, X_num, X_cat, y, loss_func, optimizer, num_epochs=5
    )
    
    assert loss_vals[0] > loss_vals[-1]

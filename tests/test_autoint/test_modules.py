
import random

import torch
from torch import nn
import numpy as np
import pytest

from xynn.base_classes.estimators import _set_seed
from xynn.base_classes.modules import BaseNN
from xynn.autoint.modules import AttnInteractionLayer, AttnInteractionBlock
from xynn.autoint import AutoInt
from xynn.embedding import LinearEmbedding, BasicEmbedding
from xynn.mlp import LeakyGate, GhostBatchNorm

from ..common import simple_train_inputs, simple_model_train_loop


def test_attnlayer_basic_initialization():
    attn = AttnInteractionLayer(
        field_input_size=5, 
        use_residual=False,
        dropout=0.0,
        normalize=False,
    )
    assert attn.W_q.shape == (5, 8, 2)
    assert attn.W_k.shape == (5, 8, 2)
    assert attn.W_v.shape == (5, 8, 2)
    assert attn.W_r is None
    assert isinstance(attn.w_act, nn.Identity)
    assert isinstance(attn.dropout, nn.Identity)
    assert isinstance(attn.layer_norm, nn.Identity)


def test_attnlayer_intitialization_with_more_options():
    attn = AttnInteractionLayer(
        field_input_size=5, 
        field_output_size=10,
        activation=nn.ReLU,
    )
    assert attn.W_q.shape == (5, 10, 2)
    assert attn.W_k.shape == (5, 10, 2)
    assert attn.W_v.shape == (5, 10, 2)
    assert attn.W_r.shape == (5, 20)
    assert isinstance(attn.w_act, nn.ReLU)
    assert isinstance(attn.dropout, nn.Dropout)
    assert isinstance(attn.layer_norm, nn.LayerNorm)


def test_attnlayer_output_shape():
    x = torch.tensor([[[1, 0]]], dtype=torch.float)
    attn = AttnInteractionLayer(field_input_size=2, field_output_size=3)
    out = attn(x)
    assert out.shape == (1, 1, 6)


def test_that_autoint_module_subclasses_basenn():
    assert issubclass(AutoInt, BaseNN)


def test_that_autoint_uses_basenn_init():
    X = torch.randint(0, 10, (100, 10))
    embedding_num = LinearEmbedding(embedding_size=3).fit(X)

    model = AutoInt(
        task="classification",
        output_size=3,
        embedding_num=embedding_num,
        embedding_cat=None,
        embedding_l1_reg=0.1,
        mlp_l2_reg=0.2,
    )

    assert model.task == "classification"
    assert model.num_epochs == 0
    assert isinstance(model.loss_fn, nn.CrossEntropyLoss)
    assert model.embedding_num is embedding_num
    assert model.embedding_cat is None
    assert model.embedding_l1_reg == 0.1
    assert model.embedding_l2_reg == 0.0
    assert model.mlp_l1_reg == 0.0
    assert model.mlp_l2_reg == 0.2
    assert model.optimizer is None
    assert model.optimizer_info == {}
    assert model.scheduler == {}
    assert model._device == "cpu"


def test_that_autoint_parameters_are_passed_to_submodules():
    X = torch.randint(0, 10, (100, 10))
    embedding_cat = BasicEmbedding(embedding_size=3).fit(X)
    model = AutoInt(
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
    for mlp in (model.mlp, model.attn_final):
        assert len(mlp.main_layers) == len(expected_classes)
        for layer, expected_class in zip(mlp.main_layers, expected_classes):
            assert isinstance(layer, expected_class)
        assert mlp.skip_layers is None

    assert isinstance(model.attn_interact, AttnInteractionBlock)
    assert len(model.attn_interact.layers) == 3
    assert model.mix.requires_grad


def test_that_autoint_parameters_are_passed_to_submodules_other_params():
    X = torch.randint(0, 10, (100, 10))
    embedding_cat = BasicEmbedding(embedding_size=3).fit(X)
    model = AutoInt(
        task="classification",
        output_size=3,
        embedding_num=None,
        embedding_cat=embedding_cat,
        attn_num_layers=2,
        mlp_ghost_batch=8,
        mlp_use_skip=True,
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
        GhostBatchNorm,
        nn.LeakyReLU,
        nn.Linear,
        GhostBatchNorm,
        nn.LeakyReLU,
        nn.Linear,
    ]
    for mlp in (model.mlp, model.attn_final):
        assert len(mlp.main_layers) == len(expected_classes)
        for layer, expected_class in zip(mlp.main_layers, expected_classes):
            assert isinstance(layer, expected_class)
        assert mlp.skip_layers is not None

    assert isinstance(model.attn_interact, AttnInteractionBlock)
    assert len(model.attn_interact.layers) == 2
    assert model.mix.requires_grad


def test_that_autoint_diagram_exists_and_prints_something(capsys):
    AutoInt.diagram()
    captured = capsys.readouterr()
    assert len(captured.out.split("\n")) > 5


def test_autoint_mlp_weight():
    X = torch.randint(0, 10, (100, 10))
    embedding_num = LinearEmbedding(embedding_size=3).fit(X)

    # without Linear after CIN
    model = AutoInt(
        task="regression",
        output_size=1,
        embedding_num=embedding_num,
        embedding_cat=None,
        attn_use_mlp=False,
        mlp_use_bn=False,
        mlp_use_skip=False,
        use_leaky_gate=False,
    )

    exp_w1 = 0
    exp_w2 = 0
    for mlp in (model.mlp, model.attn_final):
        exp_w1 += sum(l.weight.abs().sum().item() for l in mlp.main_layers[::2])
        exp_w2 += sum((l.weight ** 2).sum().item() for l in mlp.main_layers[::2])

    w1, w2 = model.mlp_weight_sum()
    assert np.isclose(w1.item(), exp_w1)
    assert np.isclose(w2.item(), exp_w2)

    # with MLP after CIN
    model = AutoInt(
        task="regression",
        output_size=1,
        embedding_num=embedding_num,
        embedding_cat=None,
        attn_use_mlp=True,
        mlp_use_bn=False,
        mlp_use_skip=False,
        use_leaky_gate=False,
    )

    exp_w1 = 0
    exp_w2 = 0
    for mlp in (model.mlp, model.attn_final):
        exp_w1 += sum(l.weight.abs().sum().item() for l in mlp.main_layers[::2])
        exp_w2 += sum((l.weight ** 2).sum().item() for l in mlp.main_layers[::2])

    w1, w2 = model.mlp_weight_sum()
    assert np.isclose(w1.item(), exp_w1)
    assert np.isclose(w2.item(), exp_w2)


def test_that_autoint_learns():
    _set_seed(10101)

    X_num = torch.randint(0, 10, (100, 10))
    X_cat = torch.randint(0, 5, (100, 1))
    y = (
        (X_cat - 2) + X_num[:, ::2].sum(dim=1) - X_num[:, 1::2].sum(dim=1)
    ).to(dtype=torch.float)
    
    model = AutoInt(
        task="regression",
        output_size=1,
        embedding_num=LinearEmbedding(embedding_size=3).fit(X_num),
        embedding_cat=BasicEmbedding(embedding_size=3).fit(X_cat),
        mlp_hidden_sizes=[10, 8, 6],
    )
    
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)
    loss_vals = simple_model_train_loop(
        model, X_num, X_cat, y, loss_func, optimizer, num_epochs=5
    )
    
    assert loss_vals[0] > loss_vals[-1]


def test_that_autoint_learns_with_other_params():
    _set_seed(10101)

    X_num = torch.randint(0, 10, (100, 10))
    X_cat = torch.randint(0, 5, (100, 1))
    y = (
        (X_cat - 2) + X_num[:, ::2].sum(dim=1) - X_num[:, 1::2].sum(dim=1)
    ).to(dtype=torch.float)
    
    model = AutoInt(
        task="regression",
        output_size=1,
        embedding_num=LinearEmbedding(embedding_size=3).fit(X_num),
        embedding_cat=BasicEmbedding(embedding_size=3).fit(X_cat),
        mlp_hidden_sizes=[],
    )
    
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)
    loss_vals = simple_model_train_loop(
        model, X_num, X_cat, y, loss_func, optimizer, num_epochs=5
    )
    
    assert loss_vals[0] > loss_vals[-1]

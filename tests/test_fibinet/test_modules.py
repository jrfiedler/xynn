
import random

import torch
from torch import nn
import numpy as np
import pytest

from xynn.base_classes.estimators import _set_seed
from xynn.base_classes.modules import BaseNN
from xynn.fibinet.modules import SENET, Bilinear, Hadamard
from xynn.fibinet import FiBiNet
from xynn.embedding import LinearEmbedding, BasicEmbedding
from xynn.mlp import LeakyGate

from ..common import simple_train_inputs, simple_model_train_loop


def test_that_senet_raises_error_for_bad_reduction_ratio():
    with pytest.raises(
        ValueError,
        match="reduction_ratio should be a positive integer, got -1"
    ):
        SENET(num_fields=20, reduction_ratio=-1)


def test_senet():
    senet = SENET(num_fields=20, reduction_ratio=4)
    assert len(senet.layers) == 4
    assert isinstance(senet.layers[0], nn.Linear) and senet.layers[0].weight.shape == (5, 20)
    assert isinstance(senet.layers[1], nn.LeakyReLU)
    assert isinstance(senet.layers[2], nn.Linear) and senet.layers[2].weight.shape == (20, 5)
    assert isinstance(senet.layers[3], nn.LeakyReLU)

    senet = SENET(num_fields=20, reduction_ratio=4, activation=nn.Identity)
    x_in = torch.rand((15, 20, 8))
    x_out = senet(x_in)
    expected_out = x_in * senet.layers[2](senet.layers[0](x_in.mean(dim=2))).unsqueeze(dim=-1)

    assert x_out.shape == (15, 20, 8)
    assert torch.allclose(x_out, expected_out)


def test_that_bilinear_raises_error_for_bad_product_type():
    with pytest.raises(ValueError, match="unrecognized product type 'inner'"):
        Bilinear(num_fields=20, embedding_size=8, product_type="inner")
    with pytest.raises(ValueError, match="unrecognized product type 'all'"):
        Bilinear(num_fields=20, embedding_size=8, product_type="all")


def test_bilinear_number_and_size_of_layers():
    bilinear = Bilinear(num_fields=20, embedding_size=8, product_type="field-all")
    assert len(bilinear.layers) == 1
    assert bilinear.layers[0].weight.shape == (8, 8)
    bilinear = Bilinear(num_fields=20, embedding_size=8, product_type="sym-all")
    assert len(bilinear.layers) == 1
    assert bilinear.layers[0].weight.shape == (8, 8)

    bilinear = Bilinear(num_fields=20, embedding_size=6, product_type="field-each")
    assert len(bilinear.layers) == 20
    assert all(layer.weight.shape == (6, 6) for layer in bilinear.layers)
    bilinear = Bilinear(num_fields=20, embedding_size=6, product_type="sym-each")
    assert len(bilinear.layers) == 20
    assert all(layer.weight.shape == (6, 6) for layer in bilinear.layers)

    bilinear = Bilinear(num_fields=20, embedding_size=10, product_type="field-interaction")
    assert len(bilinear.layers) == 190
    assert all(layer.weight.shape == (10, 10) for layer in bilinear.layers)
    bilinear = Bilinear(num_fields=20, embedding_size=10, product_type="sym-interaction")
    assert len(bilinear.layers) == 190
    assert all(layer.weight.shape == (10, 10) for layer in bilinear.layers)


def test_bilinear_output_with_product_type_field_all():
    x_ones = torch.ones((10, 8, 4))
    bilinear = Bilinear(num_fields=10, embedding_size=4, product_type="field-all")
    layer = bilinear.layers[0]
    x_out = bilinear(x_ones)
    expected_out = layer.weight.sum(dim=1).unsqueeze(dim=0)
    assert x_out.shape == (10, 28, 4)
    assert all(torch.allclose(x_out[:, i], expected_out) for i in range(28))


def test_bilinear_output_with_product_type_sym_all():
    x_ones = torch.ones((10, 8, 4))
    bilinear = Bilinear(num_fields=10, embedding_size=4, product_type="sym-all")
    layer = bilinear.layers[0]
    x_out = bilinear(x_ones)
    expected_out = layer.weight.sum(dim=1).unsqueeze(dim=0)
    assert x_out.shape == (10, 28, 4)
    assert all(torch.allclose(x_out[:, i], expected_out) for i in range(28))


def test_bilinear_output_with_product_type_field_each():
    x_ones = torch.ones((10, 8, 4))
    bilinear = Bilinear(num_fields=10, embedding_size=4, product_type="field-each")
    expected_out = [layer.weight.sum(dim=1).unsqueeze(dim=0) for layer in bilinear.layers]
    x_out = bilinear(x_ones)
    assert x_out.shape == (10, 28, 4)
    output_num = 0
    for i in range(7):
        for j in range(i + 1, 8):
            assert torch.allclose(x_out[:, output_num], expected_out[i])
            output_num += 1


def test_bilinear_output_with_product_type_sym_each():
    x_ones = torch.ones((10, 8, 4))
    bilinear = Bilinear(num_fields=10, embedding_size=4, product_type="sym-each")
    expected_out = [layer.weight.sum(dim=1).unsqueeze(dim=0) for layer in bilinear.layers]
    x_out = bilinear(x_ones)
    assert x_out.shape == (10, 28, 4)
    output_num = 0
    for i in range(7):
        for j in range(i + 1, 8):
            assert torch.allclose(x_out[:, output_num], (expected_out[i] + expected_out[j]) / 2)
            output_num += 1


def test_bilinear_output_with_product_type_field_interaction():
    x_ones = torch.ones((10, 8, 4))
    bilinear = Bilinear(num_fields=10, embedding_size=4, product_type="field-interaction")
    x_out = bilinear(x_ones)
    expected_out = [layer.weight.sum(dim=1).unsqueeze(dim=0) for layer in bilinear.layers]
    assert x_out.shape == (10, 28, 4)
    output_num = 0
    for i in range(7):
        for j in range(i + 1, 8):
            assert torch.allclose(x_out[:, output_num], expected_out[output_num])
            output_num += 1


def test_bilinear_output_with_product_type_sym_interaction():
    x_ones = torch.ones((10, 8, 4))
    bilinear = Bilinear(num_fields=10, embedding_size=4, product_type="sym-interaction")
    x_out = bilinear(x_ones)
    expected_out = [layer.weight.sum(dim=1).unsqueeze(dim=0) for layer in bilinear.layers]
    assert x_out.shape == (10, 28, 4)
    output_num = 0
    for i in range(7):
        for j in range(i + 1, 8):
            assert torch.allclose(x_out[:, output_num], expected_out[output_num])
            output_num += 1


def test_hadamard_output():
    hadamard = Hadamard()

    x_ones = torch.ones((10, 8, 4))
    out = hadamard(x_ones)
    expected_out = torch.ones((10, 28, 4))
    assert torch.allclose(out, expected_out)

    x_rand = torch.rand((10, 8, 4))
    out = hadamard(x_rand)
    expected_out = torch.cat(
        [
            (x_rand[:, i] * x_rand[:, j]).reshape((10, 1, 4))
            for i in range(0, 7)
            for j in range(i + 1, 8)
        ],
        dim=1,
    )
    assert torch.allclose(out, expected_out)


def test_that_fibinet_module_subclasses_basenn():
    assert issubclass(FiBiNet, BaseNN)


def test_that_fibinet_uses_basenn_init():
    X = torch.randint(0, 10, (100, 10))
    embedding_num = LinearEmbedding(embedding_size=3).fit(X)

    model = FiBiNet(
        task="classification",
        output_size=3,
        embedding_num=embedding_num,
        embedding_cat=None,
        num_numeric_fields=10,
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


def test_that_fibinet_parameters_are_passed_to_submodules():
    X = torch.randint(0, 10, (100, 10))
    embedding_cat = BasicEmbedding(embedding_size=3).fit(X)
    model = FiBiNet(
        task="classification",
        output_size=3,
        embedding_num=None,
        embedding_cat=embedding_cat,
        num_numeric_fields=0,
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
    mlp = model.mlp
    assert len(mlp.main_layers) == len(expected_classes)
    for layer, expected_class in zip(mlp.main_layers, expected_classes):
        assert isinstance(layer, expected_class)
    assert mlp.skip_layers is None
    assert model.senet_skip
    assert isinstance(model.senet, SENET)
    assert isinstance(model.senet_product, Bilinear)
    assert isinstance(model.embed_product, Bilinear)
    assert model.senet_product is not model.embed_product


def test_that_fibinet_parameters_are_passed_to_submodules_other_params():
    X = torch.randint(0, 10, (100, 10))
    embedding_cat = BasicEmbedding(embedding_size=3).fit(X)
    model = FiBiNet(
        task="classification",
        output_size=3,
        embedding_num=None,
        embedding_cat=embedding_cat,
        num_numeric_fields=0,
        fibi_embed_product="shared",
        fibi_senet_skip=False,
        mlp_use_skip=False,
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
    mlp = model.mlp
    assert len(mlp.main_layers) == len(expected_classes)
    for layer, expected_class in zip(mlp.main_layers, expected_classes):
        assert isinstance(layer, expected_class)
    assert mlp.skip_layers is None
    assert not model.senet_skip
    assert isinstance(model.senet, SENET)
    assert isinstance(model.senet_product, Bilinear)
    assert model.senet_product is model.embed_product


def test_that_fibinet_diagram_exists_and_prints_something(capsys):
    FiBiNet.diagram()
    captured = capsys.readouterr()
    assert len(captured.out.split("\n")) > 5


def test_fibinet_mlp_weight():
    X = torch.randint(0, 10, (100, 10))
    embedding_num = LinearEmbedding(embedding_size=3).fit(X)

    model = FiBiNet(
        task="regression",
        output_size=1,
        embedding_num=embedding_num,
        embedding_cat=None,
        num_numeric_fields=10,
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


def test_that_fibinet_learns():
    _set_seed(10101)

    X_num = torch.randint(0, 10, (100, 10))
    X_cat = torch.randint(0, 5, (100, 1))
    y = (
        (X_cat - 2) + X_num[:, ::2].sum(dim=1) - X_num[:, 1::2].sum(dim=1)
    ).to(dtype=torch.float)
    
    model = FiBiNet(
        task="regression",
        output_size=1,
        embedding_num=LinearEmbedding(embedding_size=3).fit(X_num),
        embedding_cat=BasicEmbedding(embedding_size=3).fit(X_cat),
        num_numeric_fields=10,
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


def test_that_fibinet_learns_with_other_params():
    _set_seed(10101)

    X_num = torch.randint(0, 10, (100, 10))
    X_cat = torch.randint(0, 5, (100, 1))
    y = (
        (X_cat - 2) + X_num[:, ::2].sum(dim=1) - X_num[:, 1::2].sum(dim=1)
    ).to(dtype=torch.float)
    
    model = FiBiNet(
        task="regression",
        output_size=1,
        embedding_num=LinearEmbedding(embedding_size=3).fit(X_num),
        embedding_cat=BasicEmbedding(embedding_size=3).fit(X_cat),
        num_numeric_fields=10,
        fibi_senet_product="hadamard",
        fibi_embed_product="hadamard",
        fibi_senet_skip=False,
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

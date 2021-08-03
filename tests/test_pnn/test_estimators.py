import torch
from torch import nn

from xynn.pnn import PNNRegressor, PNNClassifier
from xynn.pnn import PNNPlusRegressor, PNNPlusClassifier
from xynn.embedding import LinearEmbedding, DefaultEmbedding

from ..common import check_estimator_learns, simple_data


def test_that_basic_params_are_passed_to_pnn_module():
    X_num, X_cat, y = simple_data(task="classification")
    estimator = PNNClassifier(
        embedding_l2_reg=0.2,
        mlp_l1_reg=0.1,
    )
    estimator.fit(
        X_num=X_num,
        X_cat=X_cat,
        y=y,
        optimizer=torch.optim.Adam,
        opt_kwargs={"lr": 1e-1},
        num_epochs=1,
    )

    model = estimator._model

    assert model.task == "classification"
    assert model.num_epochs == 1
    assert isinstance(model.loss_fn, nn.CrossEntropyLoss)
    assert model.embedding_num is not None
    assert model.embedding_cat is not None
    assert model.embedding_l1_reg == 0.0
    assert model.embedding_l2_reg == 0.2
    assert model.mlp_l1_reg == 0.1
    assert model.mlp_l2_reg == 0.0
    assert model.optimizer is not None
    assert model.optimizer_info != {}
    assert model.scheduler == {}
    assert model._device == torch.device("cpu")


def test_that_basic_params_are_passed_to_pnnplus_module():
    X_num, X_cat, y = simple_data(task="classification")
    estimator = PNNPlusClassifier(
        embedding_l2_reg=0.2,
        mlp_l1_reg=0.1,
    )
    estimator.fit(
        X_num=X_num,
        X_cat=X_cat,
        y=y,
        optimizer=torch.optim.Adam,
        opt_kwargs={"lr": 1e-1},
        num_epochs=1,
    )

    model = estimator._model

    assert model.task == "classification"
    assert model.num_epochs == 1
    assert isinstance(model.loss_fn, nn.CrossEntropyLoss)
    assert model.embedding_num is not None
    assert model.embedding_cat is not None
    assert model.embedding_l1_reg == 0.0
    assert model.embedding_l2_reg == 0.2
    assert model.mlp_l1_reg == 0.1
    assert model.mlp_l2_reg == 0.0
    assert model.optimizer is not None
    assert model.optimizer_info != {}
    assert model.scheduler == {}
    assert model._device == torch.device("cpu")


def test_that_pnnregressor_learns():
    estimator = PNNRegressor(
        mlp_hidden_sizes=[10, 8, 8, 6],
        mlp_use_bn=False,
        mlp_use_skip=False,
        use_leaky_gate=False,
    )
    check_estimator_learns(estimator, task="regression")
    assert estimator.init_parameters == {
        "embedding_num": "auto",
        "embedding_cat": "auto",
        "embedding_l1_reg": 0.0,
        "embedding_l2_reg": 0.0,
        "pnn_product_type": "outer",
        "pnn_product_size": 10,
        "mlp_hidden_sizes": [10, 8, 8, 6],
        "mlp_activation": nn.LeakyReLU,
        "mlp_use_bn": False,
        "mlp_bn_momentum": 0.1,
        "mlp_ghost_batch": None,
        "mlp_dropout": 0.0,
        "mlp_l1_reg": 0.0,
        "mlp_l2_reg": 0.0,
        "mlp_use_skip": False,
        "use_leaky_gate": False,
        "loss_fn": "auto",
        "seed": None,
        "device": "cpu",
    }


def test_that_pnnclassifier_learns():
    estimator = PNNClassifier(
        pnn_product_type="inner",
        mlp_hidden_sizes=[10, 8, 8, 6],
        mlp_use_bn=False,
        mlp_ghost_batch=4,
        mlp_use_skip=False,
        use_leaky_gate=False,
    )
    check_estimator_learns(estimator, task="classification")
    assert estimator.init_parameters == {
        "embedding_num": "auto",
        "embedding_cat": "auto",
        "embedding_l1_reg": 0.0,
        "embedding_l2_reg": 0.0,
        "pnn_product_type": "inner",
        "pnn_product_size": 10,
        "mlp_hidden_sizes": [10, 8, 8, 6],
        "mlp_activation": nn.LeakyReLU,
        "mlp_use_bn": False,
        "mlp_bn_momentum": 0.1,
        "mlp_ghost_batch": 4,
        "mlp_dropout": 0.0,
        "mlp_l1_reg": 0.0,
        "mlp_l2_reg": 0.0,
        "mlp_use_skip": False,
        "use_leaky_gate": False,
        "loss_fn": "auto",
        "seed": None,
        "device": "cpu",
    }



def test_that_pnnplusregressor_learns():
    estimator = PNNPlusRegressor(
        pnn_product_type="both",
        pnn_product_size=8,
        mlp_hidden_sizes=[10, 8, 8, 6],
        mlp_use_bn=False,
        mlp_ghost_batch=4,
        mlp_use_skip=False,
        use_leaky_gate=False,
    )
    check_estimator_learns(estimator, task="regression")
    assert estimator.init_parameters == {
        "embedding_num": "auto",
        "embedding_cat": "auto",
        "embedding_l1_reg": 0.0,
        "embedding_l2_reg": 0.0,
        "pnn_product_type": "both",
        "pnn_product_size": 8,
        "mlp_hidden_sizes": [10, 8, 8, 6],
        "mlp_activation": nn.LeakyReLU,
        "mlp_use_bn": False,
        "mlp_bn_momentum": 0.1,
        "mlp_ghost_batch": 4,
        "mlp_dropout": 0.0,
        "mlp_l1_reg": 0.0,
        "mlp_l2_reg": 0.0,
        "mlp_use_skip": False,
        "use_leaky_gate": False,
        "weighted_sum": True,
        "loss_fn": "auto",
        "seed": None,
        "device": "cpu",
    }


def test_that_pnnplusclassifier_learns():
    embed_num = LinearEmbedding(10)
    embed_cat = DefaultEmbedding(10)
    estimator = PNNPlusClassifier(
        embedding_num=embed_num,
        embedding_cat=embed_cat,
        mlp_hidden_sizes=[10, 8, 8, 6],
        mlp_use_bn=False,
        mlp_use_skip=False,
        use_leaky_gate=False,
    )
    check_estimator_learns(estimator, task="classification")
    assert estimator.init_parameters == {
        "embedding_num": embed_num,
        "embedding_cat": embed_cat,
        "embedding_l1_reg": 0.0,
        "embedding_l2_reg": 0.0,
        "pnn_product_type": "outer",
        "pnn_product_size": 10,
        "mlp_hidden_sizes": [10, 8, 8, 6],
        "mlp_activation": nn.LeakyReLU,
        "mlp_use_bn": False,
        "mlp_bn_momentum": 0.1,
        "mlp_ghost_batch": None,
        "mlp_dropout": 0.0,
        "mlp_l1_reg": 0.0,
        "mlp_l2_reg": 0.0,
        "mlp_use_skip": False,
        "use_leaky_gate": False,
        "weighted_sum": True,
        "loss_fn": "auto",
        "seed": None,
        "device": "cpu",
    }
    assert repr(estimator._model.embedding_num) == "LinearEmbedding(10, 'cpu')"
    assert repr(estimator._model.embedding_cat) == "DefaultEmbedding(10, 20, 'cpu')"

from torch import nn

from xynn.xdeepfm import XDeepFMRegressor, XDeepFMClassifier
from xynn.embedding import LinearEmbedding, DefaultEmbedding

from ..common import check_estimator_learns


def test_that_xdeepfmregressor_learns():
    estimator = XDeepFMRegressor(
        mlp_hidden_sizes=[10, 8, 8, 6],
        mlp_use_bn=False,
    )
    check_estimator_learns(estimator, task="regression")
    assert estimator.init_parameters == {
        "embedding_num": "auto",
        "embedding_cat": "auto",
        "embedding_l1_reg": 0.0,
        "embedding_l2_reg": 0.0,
        "cin_layer_sizes": (128, 128),
        "cin_activation": "Identity",
        "cin_full_agg": False,
        "cin_use_bn": True,
        "cin_bn_momentum": 0.1,
        "cin_use_residual": True,
        "cin_use_mlp": True,
        "mlp_hidden_sizes": [10, 8, 8, 6],
        "mlp_activation": "LeakyReLU",
        "mlp_use_bn": False,
        "mlp_bn_momentum": 0.1,
        "mlp_dropout": 0.0,
        "mlp_l1_reg": 0.0,
        "mlp_l2_reg": 0.0,
        "mlp_leaky_gate": True,
        "mlp_use_skip": True,
        "loss_fn": "auto",
        "seed": None,
        "device": "cpu",
    }


def test_that_xdeepfmclassifier_learns():
    estimator = XDeepFMClassifier(
        cin_layer_sizes=[64, 64],
        cin_activation=nn.ReLU,
        cin_full_agg=True,
        cin_use_bn=False,
        mlp_hidden_sizes=[10, 8, 8, 6],
        mlp_use_bn=False,
        mlp_leaky_gate=False,
        mlp_use_skip=False,
    )
    assert estimator
    check_estimator_learns(estimator, task="classification")
    assert estimator.init_parameters == {
        "embedding_num": "auto",
        "embedding_cat": "auto",
        "embedding_l1_reg": 0.0,
        "embedding_l2_reg": 0.0,
        "cin_layer_sizes": [64, 64],
        "cin_activation": "ReLU",
        "cin_full_agg": True,
        "cin_use_bn": False,
        "cin_bn_momentum": 0.1,
        "cin_use_residual": True,
        "cin_use_mlp": True,
        "mlp_hidden_sizes": [10, 8, 8, 6],
        "mlp_activation": "LeakyReLU",
        "mlp_use_bn": False,
        "mlp_bn_momentum": 0.1,
        "mlp_dropout": 0.0,
        "mlp_l1_reg": 0.0,
        "mlp_l2_reg": 0.0,
        "mlp_leaky_gate": False,
        "mlp_use_skip": False,
        "loss_fn": "auto",
        "seed": None,
        "device": "cpu",
    }

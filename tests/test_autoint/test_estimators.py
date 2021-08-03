from torch import nn

from xynn.autoint import AutoIntRegressor, AutoIntClassifier
from xynn.embedding import LinearEmbedding, DefaultEmbedding

from ..common import check_estimator_learns


def test_that_autointregressor_learns():
    estimator = AutoIntRegressor(
        mlp_hidden_sizes=[10, 8, 8, 6],
        mlp_use_bn=False,
    )
    check_estimator_learns(estimator, task="regression")
    assert estimator.init_parameters == {
        "embedding_num": "auto",
        "embedding_cat": "auto",
        "embedding_l1_reg": 0.0,
        "embedding_l2_reg": 0.0,
        "attn_embedding_size": 8,
        "attn_num_layers": 3,
        "attn_num_heads": 2,
        "attn_activation": None,
        "attn_use_residual": True,
        "attn_dropout": 0.1,
        "attn_normalize": True,
        "attn_use_mlp": True,
        "mlp_hidden_sizes": [10, 8, 8, 6],
        "mlp_activation": nn.LeakyReLU,
        "mlp_use_bn": False,
        "mlp_bn_momentum": 0.1,
        "mlp_ghost_batch": None,
        "mlp_dropout": 0.0,
        "mlp_l1_reg": 0.0,
        "mlp_l2_reg": 0.0,
        "mlp_use_skip": True,
        "use_leaky_gate": True,
        "weighted_sum": True,
        "loss_fn": "auto",
        "seed": None,
        "device": "cpu",
    }


def test_that_autointclassifier_learns():
    estimator = AutoIntClassifier(
        attn_embedding_size=12,
        attn_activation=nn.ReLU,
        attn_dropout=0.0,
        attn_use_mlp=False,
        mlp_hidden_sizes=[10, 8, 8, 6],
        mlp_use_bn=False,
        mlp_use_skip=False,
        use_leaky_gate=False,
    )
    assert estimator
    check_estimator_learns(estimator, task="classification")
    assert estimator.init_parameters == {
        "embedding_num": "auto",
        "embedding_cat": "auto",
        "embedding_l1_reg": 0.0,
        "embedding_l2_reg": 0.0,
        "attn_embedding_size": 12,
        "attn_num_layers": 3,
        "attn_num_heads": 2,
        "attn_activation": nn.ReLU,
        "attn_use_residual": True,
        "attn_dropout": 0.0,
        "attn_normalize": True,
        "attn_use_mlp": False,
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

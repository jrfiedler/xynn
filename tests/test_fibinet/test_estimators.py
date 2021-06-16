from torch import nn

from xynn.fibinet import FiBiNetRegressor, FiBiNetClassifier
from xynn.embedding import LinearEmbedding, DefaultEmbedding

from ..common import check_estimator_learns


def test_that_fibinetregressor_learns():
    estimator = FiBiNetRegressor(
        mlp_hidden_sizes=[10, 8, 8, 6],
        mlp_use_bn=False,
        mlp_leaky_gate=False,
        mlp_use_skip=False,
    )
    check_estimator_learns(estimator, task="regression")
    assert estimator.init_parameters == {
        "embedding_num": "auto",
        "embedding_cat": "auto",
        "embedding_l1_reg": 0.0,
        "embedding_l2_reg": 0.0,
        "fibi_reduction_ratio": 3,
        "fibi_activation": nn.LeakyReLU,
        "fibi_senet_product": "sym-interaction",
        "fibi_embed_product": "sym-interaction",
        "fibi_senet_skip": True,
        "mlp_hidden_sizes": [10, 8, 8, 6],
        "mlp_activation": nn.LeakyReLU,
        "mlp_use_bn": False,
        "mlp_bn_momentum": 0.1,
        "mlp_ghost_batch": None,
        "mlp_dropout": 0.0,
        "mlp_l1_reg": 0.0,
        "mlp_l2_reg": 0.0,
        "mlp_leaky_gate": False,
        "mlp_use_skip": False,
        "loss_fn": "auto",
        "seed": None,
        "device": "cpu",
    }


def test_that_fibinetclassifier_learns():
    estimator = FiBiNetClassifier(
        fibi_reduction_ratio=4,
        fibi_activation=nn.ReLU,
        fibi_senet_product="field-each",
        fibi_embed_product="shared",
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
        "fibi_reduction_ratio": 4,
        "fibi_activation": nn.ReLU,
        "fibi_senet_product": "field-each",
        "fibi_embed_product": "shared",
        "fibi_senet_skip": True,
        "mlp_hidden_sizes": [10, 8, 8, 6],
        "mlp_activation": nn.LeakyReLU,
        "mlp_use_bn": False,
        "mlp_bn_momentum": 0.1,
        "mlp_ghost_batch": None,
        "mlp_dropout": 0.0,
        "mlp_l1_reg": 0.0,
        "mlp_l2_reg": 0.0,
        "mlp_leaky_gate": False,
        "mlp_use_skip": False,
        "loss_fn": "auto",
        "seed": None,
        "device": "cpu",
    }


def test_that_fibinetclassifier_learns_with_hadamard_products():
    estimator = FiBiNetClassifier(
        fibi_reduction_ratio=4,
        fibi_activation=nn.ReLU,
        fibi_senet_product="hadamard",
        fibi_embed_product="shared",
        mlp_hidden_sizes=[10, 8, 8, 6],
        mlp_use_bn=False,
        mlp_leaky_gate=False,
        mlp_use_skip=False,
    )
    check_estimator_learns(estimator, task="classification")
    assert estimator.init_parameters == {
        "embedding_num": "auto",
        "embedding_cat": "auto",
        "embedding_l1_reg": 0.0,
        "embedding_l2_reg": 0.0,
        "fibi_reduction_ratio": 4,
        "fibi_activation": nn.ReLU,
        "fibi_senet_product": "hadamard",
        "fibi_embed_product": "shared",
        "fibi_senet_skip": True,
        "mlp_hidden_sizes": [10, 8, 8, 6],
        "mlp_activation": nn.LeakyReLU,
        "mlp_use_bn": False,
        "mlp_bn_momentum": 0.1,
        "mlp_ghost_batch": None,
        "mlp_dropout": 0.0,
        "mlp_l1_reg": 0.0,
        "mlp_l2_reg": 0.0,
        "mlp_leaky_gate": False,
        "mlp_use_skip": False,
        "loss_fn": "auto",
        "seed": None,
        "device": "cpu",
    }

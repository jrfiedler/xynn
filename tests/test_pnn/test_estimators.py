from xynn.pnn import PNNRegressor, PNNClassifier
from xynn.pnn import PNNPlusRegressor, PNNPlusClassifier
from xynn.embedding import LinearEmbedding, DefaultEmbedding

from ..common import check_estimator_learns


def test_that_pnnregressor_learns():
    estimator = PNNRegressor(
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
        "pnn_product_type": "outer",
        "pnn_product_size": 10,
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


def test_that_pnnclassifier_learns():
    estimator = PNNClassifier(
        pnn_product_type="inner",
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
        "pnn_product_type": "inner",
        "pnn_product_size": 10,
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



def test_that_pnnplusregressor_learns():
    estimator = PNNPlusRegressor(
        pnn_product_type="both",
        pnn_product_size=8,
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
        "pnn_product_type": "both",
        "pnn_product_size": 8,
        "mlp_hidden_sizes": [10, 8, 8, 6],
        "mlp_activation": "LeakyReLU",
        "mlp_use_bn": False,
        "mlp_bn_momentum": 0.1,
        "mlp_dropout": 0.0,
        "mlp_l1_reg": 0.0,
        "mlp_l2_reg": 0.0,
        "mlp_leaky_gate": False,
        "mlp_use_skip": False,
        "weighted_sum": True,
        "loss_fn": "auto",
        "seed": None,
        "device": "cpu",
    }


def test_that_pnnplusclassifier_learns():
    estimator = PNNPlusClassifier(
        embedding_num=LinearEmbedding(10),
        embedding_cat=DefaultEmbedding(10),
        mlp_hidden_sizes=[10, 8, 8, 6],
        mlp_use_bn=False,
        mlp_leaky_gate=False,
        mlp_use_skip=False,
    )
    check_estimator_learns(estimator, task="classification")
    assert estimator.init_parameters == {
        "embedding_num": "LinearEmbedding(10, 'cpu')",
        "embedding_cat": "DefaultEmbedding(10, 20, 'cpu')",
        "embedding_l1_reg": 0.0,
        "embedding_l2_reg": 0.0,
        "pnn_product_type": "outer",
        "pnn_product_size": 10,
        "mlp_hidden_sizes": [10, 8, 8, 6],
        "mlp_activation": "LeakyReLU",
        "mlp_use_bn": False,
        "mlp_bn_momentum": 0.1,
        "mlp_dropout": 0.0,
        "mlp_l1_reg": 0.0,
        "mlp_l2_reg": 0.0,
        "mlp_leaky_gate": False,
        "mlp_use_skip": False,
        "weighted_sum": True,
        "loss_fn": "auto",
        "seed": None,
        "device": "cpu",
    }

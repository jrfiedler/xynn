from xynn.pnn import PNNRegressor, PNNClassifier
from xynn.pnn import PNNPlusRegressor, PNNPlusClassifier

from ..common import check_estimator_learns


def test_that_pnnregressor_learns():
    estimator = PNNRegressor(
        mlp_hidden_sizes=[10, 8, 8, 6],
        mlp_use_bn=False,
        mlp_leaky_gate=False,
        mlp_use_skip=False,
    )
    check_estimator_learns(estimator, task="regression")


def test_that_pnnclassifier_learns():
    estimator = PNNClassifier(
        product_type="inner",
        mlp_hidden_sizes=[10, 8, 8, 6],
        mlp_use_bn=False,
        mlp_leaky_gate=False,
        mlp_use_skip=False,
    )
    check_estimator_learns(estimator, task="classification")


def test_that_pnnplusregressor_learns():
    estimator = PNNPlusRegressor(
        product_type="both",
        mlp_hidden_sizes=[10, 8, 8, 6],
        mlp_use_bn=False,
        mlp_leaky_gate=False,
        mlp_use_skip=False,
    )
    check_estimator_learns(estimator, task="regression")


def test_that_pnnplusclassifier_learns():
    estimator = PNNPlusClassifier(
        mlp_hidden_sizes=[10, 8, 8, 6],
        mlp_use_bn=False,
        mlp_leaky_gate=False,
        mlp_use_skip=False,
    )
    check_estimator_learns(estimator, task="classification")

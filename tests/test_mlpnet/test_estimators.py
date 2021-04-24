import json
import random
from tempfile import NamedTemporaryFile

import torch
from torch import nn
import numpy as np

from xynn.base_classes.estimators import _set_seed
from xynn.mlpnet import MLPRegressor, MLPClassifier

from ..common import check_estimator_learns


def test_that_mlpregressor_learns():
    _set_seed(10101)
    X = torch.rand((100, 10)) - 0.5
    y = X[:, 0] - X[:, 1] + X[:, 2] - X[:, 4] + 2 * X[:, 6] - 2 * X[:, 8]
    estimator = MLPRegressor(
        mlp_hidden_sizes=[10, 8, 8, 6],
        mlp_use_bn=False,
        mlp_leaky_gate=False,
        mlp_use_skip=False,
    )
    check_estimator_learns(estimator, task="regression", data=(X, None, y))


def test_that_mlpclassifier_learns():
    _set_seed(10101)
    X = torch.rand((100, 10)) - 0.5
    y_cont = X[:, 0] - X[:, 1] + X[:, 2] - X[:, 4] + 2 * X[:, 6] - 2 * X[:, 8]
    y = (y_cont > 0)
    estimator = MLPClassifier(
        mlp_hidden_sizes=[10, 8, 8, 6],
        mlp_use_bn=False,
        mlp_leaky_gate=False,
        mlp_use_skip=False,
    )
    check_estimator_learns(estimator, task="regression", data=(X, None, y))

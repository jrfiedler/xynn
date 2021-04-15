
import json
import datetime
from tempfile import NamedTemporaryFile

import pytest
import numpy as np
import torch
from torch import nn

from xynn.base_classes.estimators import _set_seed, BaseClassifier, BaseRegressor
from xynn.embedding import BasicEmbedding, LinearEmbedding, DefaultEmbedding
from ..common import SimpleMLP, simple_data


class SimpleMLPEstimator:

    def __init__(
        self,
        embedding_alpha=4,
        hidden_sizes=(7,),
        loss_fn="auto",
        device="cpu",
        seed=None,
    ):
        self.loss_fn = loss_fn
        self._model = None
        self.train_info = []
        self.seed = seed
        self.device = torch.device(device)
        self.init_parameters = {"loss_fn": loss_fn, "seed": seed, "device": device}
        self.embedding_size = 3
        self.embedding_alpha = embedding_alpha
        self.hidden_sizes = hidden_sizes
        self.model_kwargs = {}
        self._device = device
        if seed is not None:
            _set_seed(seed)

    def _create_model(self, embedding_num, embedding_cat):
        self._model = SimpleMLP(
            task=self.task,
            hidden_sizes=self.hidden_sizes,
            loss_fn=self.loss_fn,
        )
        self._model.embedding_num = embedding_num
        self._model.embedding_cat = embedding_cat


class SimpleMLPRegressor(SimpleMLPEstimator, BaseRegressor):

    def __init__(
        self,
        embedding_alpha=4,
        hidden_sizes=(7,),
        loss_fn="auto",
        device="cpu",
        seed=None,
    ):
        super().__init__(embedding_alpha, hidden_sizes, loss_fn, device, seed)
        self.task = "regression"
        self.num_targets = 0
        self.init_parameters["task"] = "regression"


class SimpleMLPClassifier(SimpleMLPEstimator, BaseClassifier):

    def __init__(
        self,
        embedding_alpha=4,
        hidden_sizes=(7,),
        loss_fn="auto",
        device="cpu",
        seed=None,
    ):
        super().__init__(embedding_alpha, hidden_sizes, loss_fn, device, seed)
        self.task = "classification"
        self.classes = {}
        self.init_parameters["task"] = "classification"


def test_some_values_before_nn_created():
    model = SimpleMLPRegressor()
    assert model.mlp_weight_sum() == (0.0, 0.0)
    assert model.embedding_sum() == (0.0, 0.0)
    assert model.num_parameters() == 0


def test__embeddings_with_both_X_num_and_X_cat():
    X_num, X_cat, _ = simple_data()
    model = SimpleMLPRegressor()

    model.embedding_alpha = 3
    embedding_num, embedding_cat = model._embeddings(X_num, X_cat)
    assert isinstance(embedding_num, LinearEmbedding)
    assert isinstance(embedding_cat, DefaultEmbedding)

    model.embedding_alpha = 0
    embedding_num, embedding_cat = model._embeddings(X_num, X_cat)
    assert isinstance(embedding_num, LinearEmbedding)
    assert isinstance(embedding_cat, BasicEmbedding)


def test__embeddings_without_X_num_or_X_cat():
    X_num, X_cat, _ = simple_data()
    model = SimpleMLPRegressor()
    model.embedding_alpha = 3

    embedding_num, embedding_cat = model._embeddings(
        torch.empty((X_cat.shape[0], 0)), X_cat
    )
    assert embedding_num is None
    assert isinstance(embedding_cat, DefaultEmbedding)

    embedding_num, embedding_cat = model._embeddings(
        X_num, torch.empty((X_cat.shape[0], 0))
    )
    assert isinstance(embedding_num, LinearEmbedding)
    assert embedding_cat is None


def test_that__convert_x_raises_error_when_both_X_num_and_X_cat_are_None():
    _, _, y = simple_data()
    model = SimpleMLPRegressor()
    with pytest.raises(TypeError, match="X_num and X_cat cannot both be None"):
        model._convert_x(None, None, y)


def test_that__convert_x_raises_error_with_shape_mismatch():
    X_num, X_cat, y = simple_data()
    model = SimpleMLPRegressor()
    with pytest.raises(
        ValueError,
        match=(
            r"mismatch in shapes for X_num torch.Size\(\[100, 10\]\), "
            r"X_cat torch.Size\(\[300, 1\]\), y torch.Size\(\[300, 3\]\)"
        )
    ):
        model._convert_x(X_num[:100], X_cat, y)


def test__convert_x_with_tensors_and_one_of_X_num_and_X_cat():
    X_num, X_cat, y = simple_data()
    model = SimpleMLPRegressor()

    X_num_out, X_cat_out = model._convert_x(X_num, None, y)
    assert X_num is X_num_out
    assert X_cat_out.shape == (300, 0)

    X_num_out, X_cat_out = model._convert_x(None, X_cat, y)
    assert X_num_out.shape == (300, 0)
    assert X_cat is X_cat_out


def test__convert_xy_with_tensors_and_both_X_num_and_X_cat():
    X_num, X_cat, y = simple_data()
    model = SimpleMLPRegressor()
    X_num_out, X_cat_out = model._convert_x(X_num, X_cat, y)
    assert X_num is X_num_out
    assert X_cat is X_cat_out


def test__convert_x_with_numpy_arrays_and_both_X_num_and_X_cat():
    model = SimpleMLPRegressor()

    X_num_orig, X_cat_orig, y_orig = simple_data()

    X_num = np.array(X_num_orig)
    X_cat = np.array(X_cat_orig)
    y = np.array(y_orig)

    X_num_out, X_cat_out = model._convert_x(X_num, X_cat, y)

    assert torch.all(X_num_orig == X_num_out).item()
    assert all(
        X_cat_orig[i, 0].item() == X_cat_out[i, 0].item()
        for i in range(X_cat_orig.shape[0])
    )


def test_regressor__convert_y_with_tensor():
    model = SimpleMLPRegressor()
    _, _, y = simple_data()
    y_out = model._convert_y(y)
    assert y is y_out


def test_regressor__convert_y_with_numpy_array():
    model = SimpleMLPRegressor()
    _, _, y_orig = simple_data()
    y = np.array(y_orig)
    y_out = model._convert_y(y)
    assert torch.all(y_orig == y_out).item()


def test_regressor__convert_y_with_1d_input():
    model = SimpleMLPRegressor()
    _, _, y_orig = simple_data()
    y = y_orig[:, 0]
    y_out = model._convert_y(y)
    assert y.shape == (300,)
    assert y_out.shape == (300, 1)
    assert torch.all(y == y_out.reshape((-1,))).item()


def test_regressor__fit_init():
    model = SimpleMLPRegressor()
    assert model.num_targets == 0
    assert model._model is None

    X_num, X_cat, y = simple_data()
    model._fit_init(X_num, X_cat, y)

    assert model.num_targets == 3
    assert isinstance(model._model, SimpleMLP)
    assert isinstance(model._model.embedding_num, LinearEmbedding)
    assert isinstance(model._model.embedding_cat, DefaultEmbedding)


def test_some_method_return_values_after_nn_created():
    model = SimpleMLPRegressor()
    X_num, X_cat, y = simple_data()
    model._fit_init(X_num, X_cat, y)
    w1, w2 = model.mlp_weight_sum()
    e1, e2 = model.embedding_sum()
    assert w1 > 0 and w2 > 0
    assert e1 > 0 and e2 > 0
    # embedding_num: 10 * 3
    # embedding_cat: 3 * 3
    # 1st layer w: 11 * 7, b: 7
    # 2nd layer w:  7 * 3, b: 3
    assert model.num_parameters() == 10 * 3 + 3 * 3 + 11 * 7 + 7 + 7 * 3 + 3


def test_classifier__convert_y_with_tensor():
    model = SimpleMLPClassifier()
    model.classes = {0: 0, 1: 1, 2: 2}
    _, _, y = simple_data(task="classification")
    y_out = model._convert_y(y)
    assert y.shape == (300,)
    assert y_out.shape == (300,)
    assert torch.all(y == y_out.reshape((-1,))).item()


def test_classifier__convert_y_with_numpy_array():
    model = SimpleMLPClassifier()
    model.classes = {0: 0, 1: 1, 2: 2}
    _, _, y_orig = simple_data(task="classification")
    y = np.array(y_orig)
    y_out = model._convert_y(y)
    assert y.shape == (300,)
    assert y_out.shape == (300,)
    assert torch.all(y_orig == y_out.reshape((-1,))).item()


def test_classifier__fit_init():
    model = SimpleMLPClassifier(embedding_alpha=0)
    assert model.classes == {}
    assert model._model is None

    X_num, X_cat, y = simple_data(task="classification")
    model._fit_init(X_num, X_cat, y)

    assert model.classes == {0: 0, 1: 1, 2: 2}
    assert isinstance(model._model, SimpleMLP)
    assert isinstance(model._model.embedding_num, LinearEmbedding)
    assert isinstance(model._model.embedding_cat, BasicEmbedding)


def test_regressor_fit():
    model = SimpleMLPRegressor()

    logfile = NamedTemporaryFile()

    X_num, X_cat, y = simple_data()
    model.fit(
        X_num,
        X_cat,
        y,
        optimizer=torch.optim.Adam,
        opt_kwargs={"lr": 1e-1},
        log_path=logfile.name,
    )

    with open(logfile.name, "r") as infile:
        contents = json.load(infile)

    assert contents["init_parameters"] == {
        "loss_fn": "auto",
        "device": "cpu",
        "task": "regression",
        "seed": None,
    }
    assert contents["fit_parameters"] == {
        "optimizer": "Adam",
        "opt_kwargs": {"lr": 0.1},
        "scheduler": None,
        "sch_kwargs": None,
        "num_epochs": 5,
        "batch_size": 128,
        "extra_metrics": None,
        "early_stopping_metric": "val_loss",
        "early_stopping_patience": float("inf"),
        "early_stopping_mode": "min",
        "shuffle": True
    }
    assert isinstance(contents["train_info"], list)
    assert len(contents["train_info"]) == 5
    assert all(isinstance(info, dict) for info in contents["train_info"])


def test_regressor_fit_with_param_path():
    model = SimpleMLPRegressor()

    param_file = NamedTemporaryFile()

    X_num, X_cat, y = simple_data()
    model.fit(
        X_num,
        X_cat,
        y,
        val_sets=[(X_num, X_cat, y)],
        optimizer=torch.optim.Adam,
        opt_kwargs={"lr": 1e-1},
        param_path=param_file.name,
    )

    model_state_dict = {
        key: torch.clone(tsr)
        for key, tsr in model._model.state_dict().items()
    }

    # check contents of saved file
    state_dict = torch.load(param_file.name)
    assert isinstance(state_dict, dict)  # actually OrderedDict
    assert set(state_dict) == set(model_state_dict)
    for key in state_dict:
        assert torch.all(state_dict[key] == model_state_dict[key]).item()


def test_classifier_fit_with_valid_set_and_warm_start():
    X_num, X_cat, y = simple_data(task="classification")
    X_num_train, X_num_valid = X_num[:220], X_num[220:]
    X_cat_train, X_cat_valid = X_cat[:220], X_cat[220:]
    y_train, y_valid = y[:220], y[220:]

    print(y.shape)

    logfile = NamedTemporaryFile()

    model = SimpleMLPClassifier()
    model.fit(
        X_num_train,
        X_cat_train,
        y_train,
        val_sets=[(X_num_valid, X_cat_valid, y_valid)],
        optimizer=torch.optim.Adam,
        opt_kwargs={"lr": 1e-1},
    )
    model.fit(
        X_num_train,
        X_cat_train,
        y_train,
        val_sets=[(X_num_valid, X_cat_valid, y_valid)],
        optimizer=torch.optim.Adam,
        opt_kwargs={"lr": 1e-1},
        warm_start=True,
        log_path=logfile.name,
    )

    with open(logfile.name, "r") as infile:
        contents = json.load(infile)

    assert contents["init_parameters"] == {
        "loss_fn": "auto",
        "device": "cpu",
        "task": "classification",
        "seed": None,
    }
    assert contents["fit_parameters"] == {
        "optimizer": "Adam",
        "opt_kwargs": {"lr": 0.1},
        "scheduler": None,
        "sch_kwargs": None,
        "num_epochs": 5,
        "batch_size": 128,
        "extra_metrics": None,
        "early_stopping_metric": "val_loss",
        "early_stopping_patience": float("inf"),
        "early_stopping_mode": "min",
        "shuffle": True
    }
    assert isinstance(contents["train_info"], list)
    assert len(contents["train_info"]) == 10
    assert all(isinstance(info, dict) for info in contents["train_info"])


def test_that_setting_seed_produces_same_output():
    logfile = NamedTemporaryFile()
    X_num, X_cat, y = simple_data()

    model = SimpleMLPRegressor(seed=34755)
    model.fit(
        X_num,
        X_cat,
        y,
        optimizer=torch.optim.Adam,
        opt_kwargs={"lr": 1e-1},
        log_path=logfile.name,
    )
    with open(logfile.name, "r") as infile:
        contents_1 = json.load(infile)

    model = SimpleMLPRegressor(seed=34755)
    model.fit(
        X_num,
        X_cat,
        y,
        optimizer=torch.optim.Adam,
        opt_kwargs={"lr": 1e-1},
        log_path=logfile.name,
    )
    with open(logfile.name, "r") as infile:
        contents_2 = json.load(infile)

    def compare_contents(contents_1, contents_2):
        for key, val_1 in contents_1.items():
            val_2 = contents_2[key]
            if key.startswith("time"):
                dt_1 = datetime.datetime.strptime(val_1, '%Y-%m-%d %H:%M:%S')
                dt_2 = datetime.datetime.strptime(val_2, '%Y-%m-%d %H:%M:%S')
                msg = f"time difference for key {key} is too large"
                assert dt_1 - dt_2 < datetime.timedelta(seconds=10), msg
            elif isinstance(val_1, dict):
                assert isinstance(val_2, dict)
                compare_contents(val_1, val_2)
            elif isinstance(val_1, list):
                assert isinstance(val_2, list)
                for dict_1, dict_2 in zip(val_1, val_2):
                    assert isinstance(dict_1, dict)
                    assert isinstance(dict_2, dict)
                    compare_contents(dict_1, dict_2)
            else:
                assert val_1 == val_2

    compare_contents(contents_1, contents_2)


def test_that_regressor_predict_raises_error_before_fitting():
    X_num, X_cat, y = simple_data()
    model = SimpleMLPRegressor(hidden_sizes=[])
    with pytest.raises(RuntimeError, match="you need to fit the model first"):
        model.predict(X_num, X_cat)


def test_regressor_predict():
    X_num, X_cat, y = simple_data()
    model = SimpleMLPRegressor(hidden_sizes=[])
    model.fit(X_num, X_cat, y, optimizer=torch.optim.Adam, opt_kwargs={"lr": 1e-1})
    model._model.train()

    # replace the embeddings and X_num and X_cat together be the identity,
    # to make the output easy to check
    model._model.embedding_num = nn.Identity()
    model._model.embedding_cat = nn.Identity()
    X_num = torch.tensor([[1.0 if row == col else 0.0 for col in range(7)] for row in range(11)])
    X_cat = torch.tensor([[1.0 if row == col else 0.0 for col in range(7, 11)] for row in range(11)])

    preds = model.predict(X_num, X_cat)

    assert preds.requires_grad == False
    assert model._model.training == False

    linear = model._model.layers[0]

    assert torch.all(torch.eq(preds, linear.weight.T + linear.bias)).item()


def test_that_classifier_predict_raises_error_before_fitting():
    X_num, X_cat, y = simple_data(task="classification")
    model = SimpleMLPClassifier(hidden_sizes=[])
    with pytest.raises(RuntimeError, match="you need to fit the model first"):
        model.predict(X_num, X_cat)


def test_that_classifier_predict_proba_raises_error_before_fitting():
    X_num, X_cat, y = simple_data(task="classification")
    model = SimpleMLPClassifier(hidden_sizes=[])
    with pytest.raises(RuntimeError, match="you need to fit the model first"):
        model.predict_proba(X_num, X_cat)


def test_classifier_predict():
    X_num, X_cat, y = simple_data(task="classification")
    model = SimpleMLPClassifier(hidden_sizes=[])
    model.fit(X_num, X_cat, y, optimizer=torch.optim.Adam, opt_kwargs={"lr": 1e-1})
    model._model.train()

    # replace the embeddings and X_num and X_cat together be the identity,
    # to make the output easy to check
    model._model.embedding_num = nn.Identity()
    model._model.embedding_cat = nn.Identity()
    X_num = torch.tensor([[1.0 if row == col else 0.0 for col in range(7)] for row in range(11)])
    X_cat = torch.tensor([[1.0 if row == col else 0.0 for col in range(7, 11)] for row in range(11)])

    preds = model.predict(X_num, X_cat)

    assert preds.requires_grad == False
    assert model._model.training == False

    linear = model._model.layers[0]
    expect = torch.argmax(linear.weight.T + linear.bias, dim=1)

    assert torch.all(torch.eq(preds, expect)).item()


def test_classifier_predict_proba():
    X_num, X_cat, y = simple_data(task="classification")
    model = SimpleMLPClassifier(hidden_sizes=[])
    model.fit(X_num, X_cat, y, optimizer=torch.optim.Adam, opt_kwargs={"lr": 1e-1})
    model._model.train()

    # replace the embeddings and X_num and X_cat together be the identity,
    # to make the output easy to check
    model._model.embedding_num = nn.Identity()
    model._model.embedding_cat = nn.Identity()
    X_num = torch.tensor([[1.0 if row == col else 0.0 for col in range(7)] for row in range(11)])
    X_cat = torch.tensor([[1.0 if row == col else 0.0 for col in range(7, 11)] for row in range(11)])

    probas = model.predict_proba(X_num, X_cat)

    assert probas.requires_grad == False
    assert model._model.training == False

    linear = model._model.layers[0]
    expect = nn.functional.softmax(linear.weight.T + linear.bias, dim=1)

    assert torch.all(torch.eq(probas, expect)).item()

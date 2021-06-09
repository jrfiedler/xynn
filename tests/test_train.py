import re
from tempfile import NamedTemporaryFile

import pytest
import numpy as np
import torch
from torch import nn

from xynn.train import _print_header, _evaluate, train
from .common import SimpleMLP, simple_train_inputs


def test__print_header_with_defaults(capsys):
    model = SimpleMLP()
    tmplt_main, tmplt_xtra, bar = _print_header(
        model, has_validation=False, extra_metrics=[]
    )
    captured = capsys.readouterr()
    assert tmplt_main == "{epoch:>5}  {lr:>#8.3g}  {train_loss:>#10.4g}"
    assert tmplt_xtra == "                           "
    assert bar == "─" * len("epoch  lrn rate  train loss")
    assert captured.out == (
        "epoch  lrn rate  train loss\n"
        "───────────────────────────\n"
    )


def test__print_header_with_model_with_mix_parameter(capsys):
    model = SimpleMLP(mix_value=0.03)
    tmplt_main, tmplt_xtra, bar = _print_header(
        model, has_validation=False, extra_metrics=[]
    )
    captured = capsys.readouterr()
    assert tmplt_main == "{epoch:>5}  {lr:>#8.3g}  {mix:>#7.2g}  {train_loss:>#10.4g}"
    assert tmplt_xtra == "                                    "
    assert bar == "─" * len("epoch  lrn rate  non-mlp  train loss")
    assert captured.out == (
        "epoch  lrn rate  non-mlp  train loss\n"
        "────────────────────────────────────\n"
    )


def test__print_header_with_validation(capsys):
    model = SimpleMLP()
    tmplt_main, tmplt_xtra, bar = _print_header(
        model, has_validation=True, extra_metrics=[]
    )
    captured = capsys.readouterr()
    assert tmplt_main == "{epoch:>5}  {lr:>#8.3g}  {train_loss:>#10.4g}  {val_loss:>#9.4g}"
    assert tmplt_xtra == "                             {val_loss:>#9.4g}"
    assert bar == "─" * len("epoch  lrn rate  train loss   val loss")
    assert captured.out == (
        "epoch  lrn rate  train loss   val loss\n"
        "──────────────────────────────────────\n"
    )


def test__print_header_with_validation_and_extra_metrics(capsys):
    model = SimpleMLP(mix_value=0.03)
    tmplt_main, tmplt_xtra, bar = _print_header(
        model, has_validation=True, extra_metrics=[("mock", lambda y_pred, y_true: 0)]
    )
    captured = capsys.readouterr()
    assert tmplt_main == (
        "{epoch:>5}  {lr:>#8.3g}  {mix:>#7.2g}  {train_loss:>#10.4g}"
        "  {val_loss:>#9.4g}  {mock:>#9.4g}"
    )
    assert tmplt_xtra == (
        "                                    "
        "  {val_loss:>#9.4g}  {mock:>#9.4g}"
    )
    assert bar == "─" * len("epoch  lrn rate  non-mlp  train loss   val loss       mock")
    assert captured.out == (
        "epoch  lrn rate  non-mlp  train loss   val loss       mock\n"
        "──────────────────────────────────────────────────────────\n"
    )


def test__print_header_with_extra_metrics_but_no_validation(capsys):
    # should be same as if no extra metrics
    model = SimpleMLP(mix_value=0.03)
    tmplt_main, tmplt_xtra, bar = _print_header(
        model, has_validation=False, extra_metrics=[("mock", lambda y_pred, y_true: 0)]
    )
    captured = capsys.readouterr()
    assert tmplt_main == "{epoch:>5}  {lr:>#8.3g}  {mix:>#7.2g}  {train_loss:>#10.4g}"
    assert tmplt_xtra == "                                    "
    assert bar == "─" * len("epoch  lrn rate  non-mlp  train loss")
    assert captured.out == (
        "epoch  lrn rate  non-mlp  train loss\n"
        "────────────────────────────────────\n"
    )


def test_that__evaluate_raises_error_with_missing_metric_value():
    with pytest.raises(
        IndexError,
        match="cannot find early_stopping_metric 'rmse' in validation info",
    ):
        _evaluate(
            model=None,
            metric="rmse",
            patience=5,
            mode="min",
            window=1,
            best=0.1,
            count=100,
            log_info=[{"mean_absolute_error": 0.05}],
            param_path=None,
        )


def test_that__evaluate_returns_early_without_info_and_either_patience_or_param_path():
    best, count = _evaluate(
        model=None,
        metric="rmse",
        patience=5,
        mode="min",
        window=1,
        best=0.1,
        count=100,
        log_info=[],
        param_path="path.pkl",
    )
    assert best == 0.1
    assert count == 100

    best, count = _evaluate(
        model=None,
        metric="rmse",
        patience=float("inf"),
        mode="min",
        window=1,
        best=0.1,
        count=100,
        log_info=[{"rmse": 0.05}],
        param_path=None,
    )
    assert best == 0.1
    assert count == 100


def test_that__evaluate_returns_early_with_too_few_epochs():
    best, count = _evaluate(
        model=None,
        metric="rmse",
        patience=5,
        mode="min",
        window=5,
        best=float("inf"),
        count=0,
        log_info=[{"rmse": 0.55}, {"rmse": 0.45}, {"rmse": 0.25}, {"rmse": 0.05}],
        param_path="path.pkl",
    )
    assert best == float("inf")
    assert count == 0


def test__evaluate_and_saved_state_dict_when_param_path_given():
    model = SimpleMLP(task="regression")
    state_dict_orig = {
        key: torch.clone(tsr)
        for key, tsr in model.state_dict().items()
    }
    param_file = NamedTemporaryFile()
    best, count = _evaluate(
        model=model,
        metric="rmse",
        patience=float("inf"),
        mode="min",
        window=1,
        best=0.1,
        count=100,
        log_info=[
            {"epoch": 0, "rmse": 0.15},
            {"epoch": 1, "rmse": 0.15},
            {"epoch": 2, "rmse": 0.10},
            {"epoch": 3, "rmse": 0.12},
            {"epoch": 4, "rmse": 0.05},
        ],
        param_path=param_file.name,
    )
    assert best == 0.05
    assert count == 0

    # check contents of saved file
    state_dict = torch.load(param_file.name)
    assert isinstance(state_dict, dict)  # actually OrderedDict
    assert set(state_dict) == set(state_dict_orig)
    for key in state_dict:
        assert torch.all(state_dict[key] == state_dict_orig[key]).item()


def test__evaluate_with_better_min_value():
    best, count = _evaluate(
        model=None,
        metric="rmse",
        patience=5,
        mode="min",
        window=1,
        best=0.1,
        count=2,
        log_info=[
            {"rmse": 0.3688},
            {"rmse": 0.1000},
            {"rmse": 0.1654},
            {"rmse": 0.1212},
            {"rmse": 0.0625},
        ],
        param_path=None,
    )
    assert best == 0.0625
    assert count == 0


def test__evaluate_with_better_min_value_and_window():
    best, count = _evaluate(
        model=None,
        metric="rmse",
        patience=5,
        mode="min",
        window=5,
        best=0.5984,
        count=1,
        log_info=[
            {"rmse": 7.160},
            {"rmse": 6.115},
            {"rmse": 5.688},
            {"rmse": 4.977},
            {"rmse": 4.240},
            {"rmse": 3.633},
            {"rmse": 2.511},
            {"rmse": 2.137},
            {"rmse": 1.944},
            {"rmse": 1.683},
            {"rmse": 2.051},
            {"rmse": 1.713},
            {"rmse": 1.701},
            {"rmse": 0.607},
            {"rmse": 1.114},
            {"rmse": 0.337},
            {"rmse": 0.461},
            {"rmse": 0.473},
            {"rmse": 0.669},
            {"rmse": 0.359},
        ],
        param_path=None,
    )
    assert np.isclose(best, 0.4598)
    assert count == 0


def test__evaluate_with_better_max_value():
    best, count = _evaluate(
        model=None,
        metric="accuracy",
        patience=5,
        mode="max",
        window=1,
        best=0.1,
        count=2,
        log_info=[
            {"accuracy": 0.01},
            {"accuracy": 0.01},
            {"accuracy": 0.02},
            {"accuracy": 0.09},
            {"accuracy": 0.10},
            {"accuracy": 0.50},
        ],
        param_path=None,
    )
    assert best == 0.5
    assert count == 0


def test__evaluate_with_better_max_value_and_window():
    best, count = _evaluate(
        model=None,
        metric="accuracy",
        patience=5,
        mode="max",
        window=3,
        best=79.36266667,
        count=0,
        log_info=[
            {"accuracy": 72.84},
            {"accuracy": 73.885},
            {"accuracy": 74.312},
            {"accuracy": 75.023},
            {"accuracy": 75.76},
            {"accuracy": 76.367},
            {"accuracy": 77.489},
            {"accuracy": 77.863},
            {"accuracy": 78.056},
            {"accuracy": 78.317},
            {"accuracy": 77.949},
            {"accuracy": 78.287},
            {"accuracy": 78.299},
            {"accuracy": 79.393},
            {"accuracy": 78.886},
            {"accuracy": 79.663},
            {"accuracy": 79.539},
            {"accuracy": 79.527},
        ],
        param_path=None,
    )
    assert np.isclose(best, 79.57633333)
    assert count == 0


def test__evaluate_with_worse_values():
    best, count = _evaluate(
        model=None,
        metric="accuracy",
        patience=5,
        mode="max",
        window=1,
        best=0.25,
        count=2,
        log_info=[
            {"accuracy": 0.125},
            {"accuracy": 0.250},
            {"accuracy": 0.125},
            {"accuracy": 0.200},
            {"accuracy": 0.125},
        ],
        param_path=None,
    )
    assert best == 0.25
    assert count == 3

    best, count = _evaluate(
        model=None,
        metric="rmse",
        patience=5,
        mode="min",
        window=1,
        best=0.125,
        count=2,
        log_info=[
            {"rmse": 0.275},
            {"rmse": 0.250},
            {"rmse": 0.225},
            {"rmse": 0.125},
            {"rmse": 0.165},
            {"rmse": 0.225},
            {"rmse": 0.250},
        ],
        param_path=None,
    )
    assert best == 0.125
    assert count == 3


def test__evaluate_with_worse_values_and_window():
    best, count = _evaluate(
        model=None,
        metric="accuracy",
        patience=5,
        mode="max",
        window=3,
        best=79.57633333,
        count=1,
        log_info=[
            {"accuracy": 72.84},
            {"accuracy": 73.885},
            {"accuracy": 74.312},
            {"accuracy": 75.023},
            {"accuracy": 75.76},
            {"accuracy": 76.367},
            {"accuracy": 77.489},
            {"accuracy": 77.863},
            {"accuracy": 78.056},
            {"accuracy": 78.317},
            {"accuracy": 77.949},
            {"accuracy": 78.287},
            {"accuracy": 78.299},
            {"accuracy": 79.393},
            {"accuracy": 78.886},
            {"accuracy": 79.663},
            {"accuracy": 79.539},
            {"accuracy": 79.527},
            {"accuracy": 79.331},
            {"accuracy": 79.641},
        ],
        param_path=None,
    )
    assert best == 79.57633333
    assert count == 2

    best, count = _evaluate(
        model=None,
        metric="rmse",
        patience=5,
        mode="min",
        window=3,
        best=0.42366667,
        count=1,
        log_info=[
            {"rmse": 7.16},
            {"rmse": 6.115},
            {"rmse": 5.688},
            {"rmse": 4.977},
            {"rmse": 4.24},
            {"rmse": 3.633},
            {"rmse": 2.511},
            {"rmse": 2.137},
            {"rmse": 1.944},
            {"rmse": 1.683},
            {"rmse": 2.051},
            {"rmse": 1.713},
            {"rmse": 1.701},
            {"rmse": 0.607},
            {"rmse": 1.114},
            {"rmse": 0.337},
            {"rmse": 0.461},
            {"rmse": 0.473},
            {"rmse": 0.669},
            {"rmse": 0.359},
        ],
        param_path=None,
    )
    assert best == 0.42366667
    assert count == 2


def test_train_early_stopping_patience_requires_val_data():
    model, train_dl, _ = simple_train_inputs()
    with pytest.raises(
        ValueError, match="early_stopping_patience given without validation sets"
    ):
        train(
            model,
            train_dl,
            val_data=None,
            num_epochs=500,
            early_stopping_patience=3,
        )


def test_train_stopping_metric_needs_to_be_in_validation_metrics():
    model, train_dl, valid_dl = simple_train_inputs()
    with pytest.raises(
        ValueError,
        match=(
            "early_stopping_metric 'no_such_metric' "
            "is not 'val_loss' and is not one of the extra_metrics"
        )
    ):
        train(
            model,
            train_dl,
            valid_dl,
            num_epochs=500,
            early_stopping_patience=3,
            early_stopping_metric="no_such_metric",
        )


def test_train_raises_error_for_bad_early_stopping_mode():
    model, train_dl, valid_dl = simple_train_inputs()
    with pytest.raises(
        ValueError,
        match=r"early_stopping_mode needs to be 'min' or 'max'; got 'best\?'"
    ):
        train(
            model,
            train_dl,
            valid_dl,
            num_epochs=500,
            early_stopping_patience=3,
            early_stopping_mode="best?",
        )


def test_train_early_stopping_window_must_be_positive_integer():
    model, train_dl, valid_dl = simple_train_inputs()
    with pytest.raises(
        ValueError, match="early_stopping_window needs to be a positive integer; got 0"
    ):
        train(
            model,
            train_dl,
            valid_dl,
            num_epochs=500,
            early_stopping_patience=3,
            early_stopping_window=0,
        )
    with pytest.raises(
        ValueError, match="early_stopping_window needs to be a positive integer; got 4.5"
    ):
        train(
            model,
            train_dl,
            valid_dl,
            num_epochs=500,
            early_stopping_patience=3,
            early_stopping_window=4.5,
        )


def test_train_requires_validation_data_with_scheduler_monitor():
    model, train_dl, valid_dl = simple_train_inputs(
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau
    )
    with pytest.raises(
        ValueError,
        match=(
            "the model's scheduler expected to monitor "
            "'val_loss', but there is no validation data"
        )
    ):
        train(model, train_dl, val_data=[])


def test_train_requires_scheduler_monitor_to_be_in_validation_metrics():
    model, train_dl, valid_dl = simple_train_inputs(
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
        sch_options={"monitor": "rmse"},
    )
    with pytest.raises(
        ValueError,
        match="scheduler monitor 'rmse' not found in validation metrics",
    ):
        train(model, train_dl, valid_dl, extra_metrics=[("mae", lambda y_pred, y_true: 0.0)])


def test_train_on_simple_example():
    model, train_dl, valid_dl = simple_train_inputs()
    log_info = train(model, train_dl, valid_dl)

    assert len(log_info) == 5
    assert all(
        set(info.keys()) == set(['epoch', 'time', 'lr', 'train_loss', 'val_loss'])
        for info in log_info
    )
    assert all(info["epoch"] == i for i, info in enumerate(log_info))
    assert all(info["lr"] == 0.01 for info in log_info)
    assert all(
        isinstance(info[key], float)
        for info in log_info
        for key in ['train_loss', 'val_loss']
    )
    assert log_info[-1]["val_loss"] < log_info[0]["val_loss"]


def test_train_with_callback(capsys):
    def callback(log_info):
        print(log_info[-1])

    model, train_dl, valid_dl = simple_train_inputs()
    log_info = train(model, train_dl, valid_dl, callback=callback)

    captured = capsys.readouterr()

    for i, line in enumerate(captured.out.split("\n")):
        if not line:
            break
        assert f"{{'epoch': {i}" in line
    assert i == 5


def test_train_on_simple_example_with_saved_parameters():
    model, train_dl, valid_dl = simple_train_inputs()
    state_dict_orig = {
        key: torch.clone(tsr)
        for key, tsr in model.state_dict().items()
    }
    param_file = NamedTemporaryFile()
    log_info = train(model, train_dl, valid_dl, param_path=param_file.name)

    # check that contents of saved file has right keys,
    # but tensores are not exactly the same as the initial state dict
    state_dict = torch.load(param_file.name)
    assert isinstance(state_dict, dict)  # actually OrderedDict
    assert set(state_dict) == set(state_dict_orig)
    print(log_info)
    assert not all(
        torch.all(state_dict[key] == state_dict_orig[key]).item()
        for key in state_dict
    )


def test_train_with_early_stopping():
    model, train_dl, valid_dl = simple_train_inputs()
    log_info = train(
        model,
        train_dl,
        valid_dl,
        num_epochs=500,
        early_stopping_patience=3,
    )

    assert len(log_info) < 450
    assert all(
        set(info.keys()) == set(['epoch', 'time', 'lr', 'train_loss', 'val_loss'])
        for info in log_info
    )
    assert all(info["epoch"] == i for i, info in enumerate(log_info))
    assert all(info["lr"] == 0.01 for info in log_info)
    assert all(
        isinstance(info[key], float)
        for info in log_info
        for key in ['train_loss', 'val_loss']
    )
    assert all(
        log_info[-5]["val_loss"] < info["val_loss"]
        for info in log_info[:-5] + log_info[-4:]
    )


def test_train_with_scheduler_and_monitor():
    model, train_dl, valid_dl = simple_train_inputs(
        opt_kwargs={"lr": 0.1},
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
        sch_kwargs={"patience": 2},
    )
    log_info = train(
        model,
        train_dl,
        valid_dl,
        num_epochs=20,
    )

    assert all(
        set(info.keys()) == set(['epoch', 'time', 'lr', 'train_loss', 'val_loss'])
        for info in log_info
    )
    assert all(info["epoch"] == i for i, info in enumerate(log_info))
    assert all(info["lr"] <= 0.1 for info in log_info)
    assert any(np.isclose(info["lr"], 0.01) for info in log_info)
    assert all(
        isinstance(info[key], float)
        for info in log_info
        for key in ['train_loss', 'val_loss']
    )


def test_train_with_scheduler_without_monitor():
    model, train_dl, valid_dl = simple_train_inputs(
        mix_value=-3.4760986898352733,
        scheduler=torch.optim.lr_scheduler.StepLR,
        sch_kwargs={"step_size": 2, "gamma": 0.1},
    )
    log_info = train(
        model,
        train_dl,
        valid_dl,
        num_epochs=10,
    )

    assert all(
        set(info.keys()) == set(['epoch', 'time', 'lr', 'mix', 'train_loss', 'val_loss'])
        for info in log_info
    )
    assert all(info["epoch"] == i for i, info in enumerate(log_info))
    assert all(
        np.isclose(info["lr"], 0.01 * (0.1 ** (i // 2)))
        for i, info in enumerate(log_info)
    )
    assert all(np.isclose(info["mix"], 0.03) for info in log_info)
    assert all(
        isinstance(info[key], float)
        for info in log_info
        for key in ['train_loss', 'val_loss']
    )


def test_train_with_early_stopping_and_second_validation_set_and_extra_metrics():
    def rmse(y_hat, y):
        return torch.sqrt(nn.functional.mse_loss(y_hat, y))

    model, train_dl, valid_dl = simple_train_inputs()
    log_info = train(
        model,
        train_dl,
        [valid_dl, valid_dl],
        num_epochs=500,
        early_stopping_metric="rmse",
        early_stopping_patience=3,
        extra_metrics=[("rmse", rmse)],
    )

    assert len(log_info) < 450
    assert all(
        set(info.keys()) == set(['epoch', 'time', 'lr', 'train_loss', 'val_loss', 'rmse'])
        for info in log_info[::2]
    )
    assert all(
        set(info.keys()) == set(['val_loss', 'rmse'])
        for info in log_info[1::2]
    )
    assert all(info["epoch"] == i for i, info in enumerate(log_info[::2]))
    assert all(info["lr"] == 0.01 for info in log_info[::2])
    assert all(
        isinstance(info[key], float)
        for info in log_info[::2]
        for key in ['train_loss', 'val_loss', 'rmse']
    )
    assert all(
        log_info[::2][-5]["rmse"] < info["rmse"]
        for info in log_info[::2][:-5] + log_info[::2][-4:]
    )


def test_train_with_early_stopping_and_verbose(capsys):
    model, train_dl, valid_dl = simple_train_inputs()
    log_info = train(
        model,
        train_dl,
        valid_dl,
        num_epochs=500,
        early_stopping_patience=3,
        verbose=True
    )

    captured = capsys.readouterr()
    print_info = []
    for line in captured.out.split("\n"):
        if not line:
            continue
        if "\r" in line:
            line = line[line.rfind("\r") + 2:]
        print_info.append(line)

    assert print_info[0] == "epoch  lrn rate  train loss   val loss"
    assert print_info[1] == "──────────────────────────────────────"
    assert all(
        re.match(r"[0-9](\.[0-9]+)?", text)
        for line in print_info[2:-1]
        for text in line.split()
    )
    assert print_info[-1].startswith("Stopping early. Best epoch:")


def test_train_with_verbose_and_without_validation_data(capsys):
    model, train_dl, _ = simple_train_inputs()
    log_info = train(
        model,
        train_dl,
        val_data=None,
        verbose=True,
    )

    captured = capsys.readouterr()
    print_info = []
    for line in captured.out.split("\n"):
        if not line:
            continue
        if "\r" in line:
            line = line[line.rfind("\r") + 2:]
        print_info.append(line)

    assert print_info[0] == "epoch  lrn rate  train loss"
    assert print_info[1] == "───────────────────────────"
    assert all(
        re.match(r"[0-9](\.[0-9]+)?", text)
        for line in print_info[2:]
        for text in line.split()
    )


def test_train_with_scheduler_step_batch_without_monitor():
    model, train_dl, valid_dl = simple_train_inputs(
        scheduler=torch.optim.lr_scheduler.StepLR,
        sch_kwargs={"step_size": 5, "gamma": 0.1},
    )
    log_info = train(
        model,
        train_dl,
        valid_dl,
        num_epochs=2,
        scheduler_step="batch",
    )
    assert np.isclose(log_info[0]["lr"], 1e-6)
    assert np.isclose(log_info[1]["lr"], 1e-10)


def test_train_with_different_loss_fn():
    model, train_dl, valid_dl = simple_train_inputs(loss_fn=nn.L1Loss())
    log_info = train(model, train_dl, valid_dl)

    assert len(log_info) == 5
    assert all(
        set(info.keys()) == set(['epoch', 'time', 'lr', 'train_loss', 'val_loss'])
        for info in log_info
    )
    assert all(info["epoch"] == i for i, info in enumerate(log_info))
    assert all(info["lr"] == 0.01 for info in log_info)
    assert all(
        isinstance(info[key], float)
        for info in log_info
        for key in ['train_loss', 'val_loss']
    )
    assert log_info[-1]["val_loss"] < log_info[0]["val_loss"]

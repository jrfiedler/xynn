"""
Functions for a simple train loop that can be used for the
models and experiments in this package.

"""

import sys
import time
import datetime
from collections import defaultdict
from typing import Tuple, Callable, Optional, Iterable, Union, Dict, List

import numpy as np
import torch
from torch import nn, Tensor
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.special import expit

from .base_classes.modules import BaseNN


LogInfo = Dict[str, Union[str, int, float, bool]]


def now() -> str:
    """
    Return string representing current time

    Returns
    -------
    string with format '%Y-%m-%d %H:%M:%S'

    """
    timestamp = time.time()
    value = datetime.datetime.fromtimestamp(timestamp)
    return value.strftime('%Y-%m-%d %H:%M:%S')


def _print_header(
    model: nn.Module,
    has_validation: bool,
    extra_metrics: Iterable[Tuple[str, Callable]],
) -> Tuple[str, str, str]:
    top = "epoch  lrn rate"
    bar = "───────────────"
    tmplt_main = "{epoch:>5}  {lr:>#8.3g}"
    tmplt_xtra = "               "

    if hasattr(model, "mix") and model.mix is not None:
        top += "  non-mlp"
        bar += "─────────"
        tmplt_main += "  {mix:>#7.2g}"
        tmplt_xtra += " " * 9

    top += "  train loss"
    bar += "────────────"
    tmplt_main += "  {train_loss:>#10.4g}"
    tmplt_xtra += " " * 12

    if has_validation:
        top += "   val loss"
        bar += "───────────"
        tmplt_main += "  {val_loss:>#9.4g}"
        tmplt_xtra += "  {val_loss:>#9.4g}"
        for name, _ in extra_metrics:
            width = max(len(name), 9)
            precision = width - 5
            fmt = f"  {{{name}:>#{width}.{precision}g}}"
            top += " " * (2 + width - len(name)) + name
            bar += "─" * (width + 2)
            tmplt_main += fmt
            tmplt_xtra += fmt

    print(f"{top}\n{bar}", flush=True)

    return tmplt_main, tmplt_xtra, bar


def _scheduler_step(model: BaseNN, log_info: List[LogInfo]):
    if not model.scheduler:
        return
    if "monitor" in model.scheduler:
        metric = log_info[0][model.scheduler["monitor"]]
        model.scheduler["scheduler"].step(metric)
    else:
        model.scheduler["scheduler"].step()


def _train_batch(
    model: BaseNN,
    batch: List[Tensor],
    batch_idx: int,
    max_grad_norm: float,
    scheduler_step: str,
) -> float:
    model.optimizer.zero_grad(set_to_none=True)
    info = model.training_step(batch, batch_idx)
    info["loss"].backward()
    if max_grad_norm != float("inf"):
        clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
    model.optimizer.step()
    if scheduler_step == "batch":
        _scheduler_step(model, [])
    return info["loss"].item()


def _val_epoch(
    model: BaseNN,
    loader: DataLoader,
    extra_metrics: Iterable[Tuple[str, Callable]],
) -> LogInfo:
    pbar = tqdm(
        enumerate(loader),
        leave=False,
        file=sys.stdout,
        total=len(loader),
    )
    pbar.set_description(f"Eval {model.num_epochs}")
    ypairs = []
    for batch_idx, batch in pbar:
        ypair = model.validation_step(batch, batch_idx)
        val_loss = model.loss_fn(ypair[0], ypair[1]).item()
        pbar.set_postfix({"Loss": f"{val_loss:#.2g}"})
        ypairs.append(ypair)

    val_info = {}
    metric_info = model.custom_val_epoch_end(ypairs, extra_metrics)
    for name, value in metric_info.items():
        if "_step" in name:
            name = name.replace("_step", "")
        val_info[name] = value

    return val_info


def _epoch_info(
    model: BaseNN,
    log_info: List[LogInfo],
    val_data: Optional[Iterable[DataLoader]],
    extra_metrics: Iterable[Tuple[str, Callable]],
    verbose: bool,
    tmplt_main: str,
    tmplt_xtra: str,
) -> List[LogInfo]:

    for param_group in model.optimizer.param_groups:
        log_info[0]["lr"] = param_group['lr']
        break

    if hasattr(model, "mix") and model.mix is not None:
        log_info[0]["mix"] = expit(model.mix.item())

    if val_data:
        model.eval()
        with torch.no_grad():
            for i, loader in enumerate(val_data):
                if i == 0:
                    tmplt = tmplt_main
                else:
                    tmplt = tmplt_xtra
                    log_info.append({})
                val_info = _val_epoch(model, loader, extra_metrics)
                log_info[-1].update(val_info)
                if verbose:
                    print(tmplt.format(**log_info[-1]), flush=True)
    elif verbose:
        print(tmplt_main.format(**log_info[-1]), flush=True)

    return log_info


def _epoch(
    model: BaseNN,
    train_data: DataLoader,
    val_data: Optional[Iterable[DataLoader]],
    max_grad_norm: float,
    extra_metrics: Iterable[Tuple[str, Callable]],
    scheduler_step: str,
    verbose: bool,
    tmplt_main: str,
    tmplt_xtra: str,
):
    model.train()

    log_info: List[LogInfo] = [{"epoch": model.num_epochs, "time": now()}]

    pbar = tqdm(
        enumerate(train_data),
        leave=False,
        file=sys.stdout,
        total=len(train_data),
    )
    pbar.set_description(f"Train {model.num_epochs}")
    for batch_idx, batch in pbar:
        loss = _train_batch(model, batch, batch_idx, max_grad_norm, scheduler_step)
        pbar.set_postfix({"Loss": f"{loss:#.2g}"})

    log_info[0]["train_loss"] = loss

    log_info = _epoch_info(
        model, log_info, val_data, extra_metrics, verbose, tmplt_main, tmplt_xtra
    )

    if scheduler_step == "epoch":
        _scheduler_step(model, log_info)

    model.num_epochs += 1

    return log_info


def _evaluate(model, metric, patience, mode, window, best, count, log_info, param_path):
    if (patience == float("inf") and not param_path) or not log_info:
        # either nothing requested or don't have the necessary information
        return best, count
    if len(log_info) < window:
        # not enough values to calculate best yet
        return best, count
    if metric not in log_info[0]:
        raise IndexError(f"cannot find early_stopping_metric '{metric}' in validation info")
    value = np.mean([info[metric] for info in log_info[-window:]])
    if (mode == "min" and value < best) or (mode == "max" and value > best):
        best = value
        count = 0
        if param_path:
            torch.save(model.state_dict(), param_path)
    else:
        count += 1
    return best, count


def train(
    model: BaseNN,
    train_data: DataLoader,
    val_data: Optional[Union[DataLoader, Iterable[DataLoader]]] = None,
    num_epochs: int = 5,
    max_grad_norm: float = float("inf"),
    extra_metrics: Optional[List[Tuple[str, Callable]]] = None,
    scheduler_step: str = "epoch",
    early_stopping_metric: str = "val_loss",
    early_stopping_patience: Union[int, float] = float("inf"),
    early_stopping_mode: str = "min",
    early_stopping_window: int = 1,
    param_path: Optional[str] = None,
    verbose: bool = False,
):
    """
    Train the given model.

    Optimizer and optional scheduler should be already set with
    `model.set_optimizer()` and initialized with `model.configure_optimizer`.

    Parameters
    ----------
    model : BaseNN
        any PyTorch model from this package
    train_data : PyTorch DataLoader
    val_data : PyTorch DataLoader, iterable of DataLoader, or None; optional
        default is None
    num_epochs : int, optional
        default is 5
    max_grad_norm : float, optional
        value to clip gradient norms to; default is float("inf") (no clipping)
    extra_metrics : list of (str, callable) tuples or None, optional
        default is None
    scheduler_step : {"epoch", "batch"}, optional
        whether the scheduler step should be called each epoch or each batch;
        if "batch", the scheduler won't have access to validation metrics;
        default is "epoch"
    early_stopping_metric : str, optional
        should be "val_loss" or one of the passed `extra_metrics`;
        default is "val_loss"
    early_stopping_patience : int, float; optional
        default is float("inf") (no early stopping)
    early_stopping_mode : {"min", "max"}, optional
        use "min" if smaller values are better; default is "min"
    early_stopping_window : int, optional
        number of consecutive epochs to average to determine best;
        default is 1
    param_path : str or None, optional
        specify this to have the best parameters reloaded at end of training;
        default is None
    verbose : boolean, optional
        default is False

    Return
    ------
    list of dictionaries, one dictionary for each epoch

    """

    if isinstance(val_data, DataLoader):
        val_data = [val_data]

    if extra_metrics is None:
        extra_metrics = []

    val_metric_names = ["val_loss"] + [name for name, _ in extra_metrics]

    # check early stopping values
    if early_stopping_patience < float("inf"):
        if not val_data:
            raise ValueError("early_stopping_patience given without validation sets")
        if early_stopping_metric not in val_metric_names:
            raise ValueError(
                f"early_stopping_metric {repr(early_stopping_metric)} "
                "is not 'val_loss' and is not one of the extra_metrics"
            )
        if early_stopping_mode not in ("min", "max"):
            raise ValueError(
                "early_stopping_mode needs to be 'min' or 'max'; "
                f"got {repr(early_stopping_mode)}"
            )
        if not isinstance(early_stopping_window, int) or early_stopping_window <= 0:
            raise ValueError(
                "early_stopping_window needs to be a positive integer; "
                f"got {repr(early_stopping_window)}"
            )

    # check if model's sheduler needs to monitor a validation metric,
    # and check if the metric is in the validation metrics
    if model.scheduler is not None and "monitor" in model.scheduler:
        if not val_data:
            raise ValueError(
                "the model's scheduler expected to monitor "
                f"\'{model.scheduler['monitor']}\', but there is no validation data"
            )
        if model.scheduler["monitor"] not in val_metric_names:
            raise ValueError(
                f"scheduler monitor \'{model.scheduler['monitor']}\' "
                "not found in validation metrics"
            )

    if verbose:
        tmplt_main, tmplt_xtra, _ = _print_header(
            model=model,
            has_validation=bool(val_data),
            extra_metrics=extra_metrics
        )
    else:
        tmplt_main, tmplt_xtra = "", ""

    log_info = []
    es_count = 0
    es_best = float("inf") if early_stopping_mode == "min" else float("-inf")
    for _ in range(num_epochs):
        epoch_log_info = _epoch(
            model=model,
            train_data=train_data,
            val_data=val_data,
            max_grad_norm=max_grad_norm,
            extra_metrics=extra_metrics,
            scheduler_step=scheduler_step,
            verbose=verbose,
            tmplt_main=tmplt_main,
            tmplt_xtra=tmplt_xtra,
        )
        log_info.extend(epoch_log_info)
        es_best, es_count = _evaluate(
            model=model,
            metric=early_stopping_metric,
            patience=early_stopping_patience,
            mode=early_stopping_mode,
            window=early_stopping_window,
            best=es_best,
            count=es_count,
            log_info=log_info,
            param_path=param_path,
        )
        if es_count >= early_stopping_patience + 1:
            if verbose:
                best_epoch = log_info[-1]['epoch'] - es_count - early_stopping_window // 2
                print(
                    "Stopping early. "
                    f"Best epoch: {best_epoch}. "
                    f"Best {early_stopping_metric}: {es_best:11.6g}"
                )
            break

    if param_path:
        model.load_state_dict(torch.load(param_path))

    return log_info

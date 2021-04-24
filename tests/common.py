import json
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from xynn.base_classes.estimators import _set_seed
from xynn.base_classes.modules import BaseNN


class SimpleEmbedding(nn.Module):

    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.num_fields = 1
        self.embedding_size = embedding_dim
        self.output_size = embedding_dim

    def weight_sum(self):
        w = self.embedding.weight
        return w.abs().sum().item(), (w ** 2).sum().item()

    def forward(self, x):
        return self.embedding(x)


def simple_train_loop(module, X, y, loss_func, optimizer, num_epochs):
    module.train()
    losses = []
    for e_ in range(num_epochs):
        optimizer.zero_grad()
        y_pred = module(X)
        loss = loss_func(y_pred, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


def simple_model_train_loop(model, X_num, X_cat, y, loss_func, optimizer, num_epochs):
    model.train()
    losses = []
    for e_ in range(num_epochs):
        optimizer.zero_grad()
        y_pred = model(X_num, X_cat)
        print(y_pred.shape, y.shape)
        loss = loss_func(y_pred, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


class Reshape(nn.Module):

    def forward(self, X):
        return X.reshape((X.shape[0], -1))


def example_data():
    data = pd.DataFrame(
        {
            "num_a": [i / 10 for i in range(10)],
            "num_b": range(10, 0, -1),
            "cat_a": list("abcdabcaba"),
            "cat_b": list("abbabacbab"),
            "cat_c": [1, 1, 0, 0, 1, 1, 0, np.nan, 1, 1],
            "cat_a_num": [0, 1, 2, 3, 0, 1, 2, 0, 1, 0],
            "cat_b_num": [0, 1, 1, 0, 1, 0, 2, 1, 0, 1],
        }
    )
    return data


class SimpleMLP(BaseNN):

    def __init__(
        self,
        task="regression",
        input_size=11,
        hidden_sizes=(7,),
        output_size=3,
        loss_fn="auto",
        mix_value=None,
    ):
        super().__init__(task, None, None, loss_fn)
        layers = []
        for size in hidden_sizes:
            layers.append(nn.Linear(input_size, size))
            input_size = size
            layers.append(nn.ReLU())
        layers.append(nn.Linear(input_size, output_size))
        self.layers = nn.Sequential(*layers)
        if mix_value is not None:
            self.mix = torch.tensor([mix_value])
        else:
            self.mix = None
        self._device = "cpu"
        self.to("cpu")

    def mlp_weight_sum(self):
        w1_sum = 0.0
        w2_sum = 0.0
        for layer in self.layers:
            if not isinstance(layer, nn.Linear):
                continue
            w1_sum += layer.weight.abs().sum().item()
            w2_sum += (layer.weight ** 2).sum().item()
        return w1_sum, w2_sum

    def forward(self, X_num, X_cat):
        x = torch.cat([X_num, X_cat], axis=1)
        return self.layers(x)


class SimpleDataset(Dataset):

    def __init__(self, X_num, X_cat, y):
        self.X_num = X_num
        self.X_cat = X_cat
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_num[idx], self.X_cat[idx], self.y[idx]


def simple_data(task="regression"):
    X_num = torch.randint(-2, 2, (300, 10), dtype=torch.float32)
    X_cat = torch.randint(0, 2, (300, 1), dtype=torch.float32)
    z = torch.rand(size=(300, 3), dtype=torch.float32) - 0.5
    y = torch.tensor(
        [
            [
                0.1 * num[0] - 0.2 * num[1] + 0.1 * num[2] * num[3] + cat[0],
                - 0.3 * num[4] * num[6] * num[7] + 0.1 * num[8] - num[9],
                - 0.2 * num[1] - 0.3 * num[4] * num[6] * num[7] + 0.1 * cat[0],
            ]
            for num, cat in zip(X_num, X_cat)
        ],
        dtype=torch.float32
    ) + z
    if task == "classification":
        y_sum = y.sum(dim=1)
        y_cuts = torch.quantile(y_sum, q=torch.tensor([1/3, 2/3]))
        y = (
            (y_sum > y_cuts[0]).to(dtype=torch.int)
            + (y_sum > y_cuts[1]).to(dtype=torch.int)
        )
    return X_num, X_cat, y


def simple_train_inputs(
    loss_fn="auto",
    mix_value=None,
    optimizer=torch.optim.Adam,
    opt_kwargs={"lr": 1e-2},
    scheduler=None,
    sch_kwargs=None,
    sch_options=None,
    configure=True,
):
    X_num, X_cat, y = simple_data()
    X_num_train, X_num_valid = X_num[:220], X_num[220:]
    X_cat_train, X_cat_valid = X_cat[:220], X_cat[220:]
    y_train, y_valid = y[:220], y[220:]

    model = SimpleMLP(task="regression", loss_fn=loss_fn, mix_value=mix_value)
    model.set_optimizer(
        optimizer=optimizer,
        opt_kwargs=opt_kwargs,
        scheduler=scheduler,
        sch_kwargs=sch_kwargs,
        sch_options=sch_options,
    )
    if configure:
        model.configure_optimizers()

    train_ds = SimpleDataset(X_num_train, X_cat_train, y_train)
    valid_ds = SimpleDataset(X_num_valid, X_cat_valid, y_valid)
    train_dl = DataLoader(train_ds, batch_size=10, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=10)

    return model, train_dl, valid_dl


def check_estimator_learns(estimator, task, data=None, seed=10101):
    _set_seed(seed)

    if data is None:
        X_num, X_cat, y = simple_data(task=task)
    else:
        X_num, X_cat, y = data

    logfile = NamedTemporaryFile()

    estimator.fit(
        X_num=X_num,
        X_cat=X_cat,
        y=y,
        optimizer=torch.optim.Adam,
        opt_kwargs={"lr": 1e-1},
        log_path=logfile.name,
    )

    with open(logfile.name, "r") as infile:
        train_info = json.load(infile)

    loss_vals = [epoch["train_loss"] for epoch in train_info["train_info"]]
    assert any(loss_vals[i] < loss_vals[0] for i in range(1, len(loss_vals)))

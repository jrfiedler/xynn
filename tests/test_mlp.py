import numpy as np
import torch
from torch import nn
import pytest

from xynn.mlp import MLP, LeakyGate
from .common import simple_train_loop


def test_leakygate_with_identity_activation_and_2d_input():
	gate = LeakyGate(10, activation=nn.Identity)
	X = torch.ones((1, 10))
	output = gate(X)
	assert output.shape == (1, 10)
	assert torch.all(output == gate.weight).item()


def test_leakygate_with_identity_activation_and_3d_input():
	gate = LeakyGate(10, activation=nn.Identity)
	X = torch.ones((1, 5, 2))
	output = gate(X)
	assert output.shape == (1, 5, 2)
	assert torch.all(output.reshape((1, 10)) == gate.weight).item()


def test_that_leakygate_learns_without_bias():
	X = torch.rand((1, 10)) * 2 - 1
	y = torch.rand((1, 10)) * 6 - 3
	leakygate = LeakyGate(10, bias=False)
	loss_func = nn.MSELoss()
	optimizer = torch.optim.Adam(leakygate.parameters(), lr=1e-1)
	wt_before = torch.clone(leakygate.weight)
	loss_vals = simple_train_loop(leakygate, X, y, loss_func, optimizer, num_epochs=5)
	assert torch.all(leakygate.weight != wt_before).item()
	assert torch.all(leakygate.bias == 0).item()
	assert loss_vals[0] > loss_vals[-1]


def test_that_leakygate_learns_with_bias():
	X = torch.rand((1, 10)) * 2 - 1
	y = torch.rand((1, 10)) * 6 - 3
	leakygate = LeakyGate(10)
	loss_func = nn.MSELoss()
	optimizer = torch.optim.Adam(leakygate.parameters(), lr=1e-1)
	wt_before = torch.clone(leakygate.weight)
	loss_vals = simple_train_loop(leakygate, X, y, loss_func, optimizer, num_epochs=5)
	assert torch.all(leakygate.weight != wt_before).item()
	assert torch.all(leakygate.bias != 0).item()
	assert loss_vals[0] > loss_vals[-1]


def test_simple_mlp_with_bias():
	mlp = MLP(
		task="regression",  # use bias in final layer
		input_size=10,
		hidden_sizes=[],
		output_size=6,
		dropout=0.0,
		leaky_gate=False,
		use_skip=False,
		use_bn=False,
	)
	X = torch.tensor(
		[
			[1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
			[0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
			[0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
			[0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
			[0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
		]
	)

	assert len(mlp.main_layers) == 1
	assert mlp.main_layers[0].weight.shape == (6, 10)
	assert mlp.skip_layers is None

	output = mlp(X)
	weight, bias = mlp.main_layers[0].weight, mlp.main_layers[0].bias
	expected = weight[:, ::2].T + bias
	assert output.shape == (5, 6)
	assert torch.all(output == expected).item()

	w1_sum, w2_sum = mlp.weight_sum()
	assert isinstance(w1_sum, torch.Tensor)
	assert isinstance(w2_sum, torch.Tensor)
	assert np.isclose(w1_sum.item(), weight.abs().sum().item())
	assert np.isclose(w2_sum.item(), (weight ** 2).sum().item())


def test_simple_mlp_without_bias():
	mlp = MLP(
		task="classification",  # no bias in final layer
		input_size=10,
		hidden_sizes=[],
		output_size=6,
		dropout=0.0,
		leaky_gate=False,
		use_skip=False,
		use_bn=False,
	)
	X = torch.tensor(
		[
			[1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
			[0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
			[0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
			[0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
			[0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
		]
	)

	assert len(mlp.main_layers) == 1
	assert mlp.main_layers[0].weight.shape == (6, 10)

	output = mlp(X)
	weight = mlp.main_layers[0].weight
	expected = weight[:, ::2].T
	assert output.shape == (5, 6)
	assert torch.all(output == expected).item()

	w1_sum, w2_sum = mlp.weight_sum()
	assert isinstance(w1_sum, torch.Tensor)
	assert isinstance(w2_sum, torch.Tensor)
	assert np.isclose(w1_sum.item(), weight.abs().sum().item())
	assert np.isclose(w2_sum.item(), (weight ** 2).sum().item())


def test_mlp_with_bigger_example():
	mlp = MLP(
		task="classification",  # no bias in final layer
		input_size=10,
		hidden_sizes=[10, 8, 8, 6],
		output_size=6,
		dropout=0.0,
		leaky_gate=False,
		use_skip=False,
		use_bn=False,
	)

	sizes = [10, 10, 8, 8, 6]
	assert len(mlp.main_layers) == 9
	for i in range(4):
		assert mlp.main_layers[2 * i].weight.shape == (sizes[i + 1], sizes[i])
		assert isinstance(mlp.main_layers[2 * i + 1], nn.LeakyReLU)
	assert mlp.main_layers[8].weight.shape == (6, 6)

	w1_sum, w2_sum = mlp.weight_sum()
	exp_w1_sum = sum(l.weight.abs().sum().item() for l in mlp.main_layers[::2])
	exp_w2_sum = sum((l.weight ** 2).sum().item() for l in mlp.main_layers[::2])
	assert isinstance(w1_sum, torch.Tensor)
	assert isinstance(w2_sum, torch.Tensor)
	assert np.isclose(w1_sum.item(), exp_w1_sum)
	assert np.isclose(w2_sum.item(), exp_w2_sum)


def test_mlp_with_bigger_example_with_leaky_gate_and_skip():
	mlp = MLP(
		task="classification",  # no bias in final layer
		input_size=10,
		hidden_sizes=[10, 8, 8, 6],
		output_size=6,
		dropout=0.0,
		leaky_gate=True,
		use_skip=True,
		use_bn=False,
	)

	sizes = [10, 10, 8, 8, 6]
	assert len(mlp.main_layers) == 10
	for i in range(4):
		assert mlp.main_layers[2 * i + 1].weight.shape == (sizes[i + 1], sizes[i])
		assert isinstance(mlp.main_layers[2 * i + 2], nn.LeakyReLU)
	assert mlp.main_layers[9].weight.shape == (6, 6)

	w1_sum, w2_sum = mlp.weight_sum()
	exp_w1_sum = sum(
		l.weight.abs().sum().item()
		for layer_group in (mlp.main_layers[1::2], mlp.skip_layers[1:])
		for l in layer_group
	)
	exp_w2_sum = sum(
		(l.weight ** 2).sum().item()
		for layer_group in (mlp.main_layers[1::2], mlp.skip_layers[1:])
		for l in layer_group
	)
	assert isinstance(w1_sum, torch.Tensor)
	assert isinstance(w2_sum, torch.Tensor)
	assert np.isclose(w1_sum.item(), exp_w1_sum)
	assert np.isclose(w2_sum.item(), exp_w2_sum)


def test_mlp_with_dropout():
	mlp = MLP(
		task="classification",  # no bias in final layer
		input_size=10,
		hidden_sizes=8,
		output_size=6,
		dropout=0.1,
		dropout_first=True,
		use_bn=False,
		leaky_gate=False,
		use_skip=False,
	)

	sizes = [10, 8, 6]
	assert len(mlp.main_layers) == 5
	assert isinstance(mlp.main_layers[0], nn.Dropout)
	assert isinstance(mlp.main_layers[1], nn.Linear)
	assert isinstance(mlp.main_layers[2], nn.LeakyReLU)
	assert isinstance(mlp.main_layers[3], nn.Dropout)
	assert isinstance(mlp.main_layers[4], nn.Linear)

	X = torch.tensor(
		[
			[1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
			[0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
			[0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
			[0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
			[0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
		]
	)
	output = mlp(X)
	assert output.shape == (5, 6)

	w1_sum, w2_sum = mlp.weight_sum()
	exp_w1_sum = sum(l.weight.abs().sum().item() for l in mlp.main_layers[1::3])
	exp_w2_sum = sum((l.weight ** 2).sum().item() for l in mlp.main_layers[1::3])
	assert isinstance(w1_sum, torch.Tensor)
	assert isinstance(w2_sum, torch.Tensor)
	assert np.isclose(w1_sum.item(), exp_w1_sum)
	assert np.isclose(w2_sum.item(), exp_w2_sum)


def test_that_mlp_raises_error_with_wrong_number_of_dropout_values():
	msg = (
        "expected a single dropout value or 2 values "
        "(one more than hidden_sizes)"
    )
	with pytest.raises(ValueError):
		mlp = MLP(
			task="classification",  # no bias in final layer
			input_size=10,
			hidden_sizes=8,
			output_size=6,
			dropout=[0.1, 0.1, 0.1],  # should be a single float or length 2
			dropout_first=True,
			use_bn=False,
		)


def test_mlp_with_various_options():
	mlp = MLP(
		task="classification",  # no bias in final layer
		input_size=10,
		hidden_sizes=8,
		output_size=6,
		dropout=0.0,
		use_bn=True,
		leaky_gate=True,
	)

	sizes = [10, 8, 6]
	assert len(mlp.main_layers) == 5
	assert isinstance(mlp.main_layers[0], LeakyGate)
	assert isinstance(mlp.main_layers[1], nn.Linear)
	assert isinstance(mlp.main_layers[2], nn.BatchNorm1d)
	assert isinstance(mlp.main_layers[3], nn.LeakyReLU)
	assert isinstance(mlp.main_layers[4], nn.Linear)

	X = torch.tensor(
		[
			[1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
			[0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
			[0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
			[0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
			[0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
		]
	)
	output = mlp(X)
	assert output.shape == (5, 6)

	w1_sum, w2_sum = mlp.weight_sum()
	exp_w1_sum = sum(
		l.weight.abs().sum().item()
		for layer_group in (mlp.main_layers[1::3], mlp.skip_layers[1:])
		for l in layer_group
	)
	exp_w2_sum = sum(
		(l.weight ** 2).sum().item()
		for layer_group in (mlp.main_layers[1::3], mlp.skip_layers[1:])
		for l in layer_group
	)
	assert isinstance(w1_sum, torch.Tensor)
	assert isinstance(w2_sum, torch.Tensor)
	assert np.isclose(w1_sum.item(), exp_w1_sum)
	assert np.isclose(w2_sum.item(), exp_w2_sum)


def test_that_mlp_learns():
	X = torch.rand((1, 10)) * 2 - 1
	y = torch.rand((1, 6)) * 6 - 3
	mlp = MLP(
		task="regression",  # has bias in final layer
		input_size=10,
		hidden_sizes=[10, 8, 8, 6],
		output_size=6,
		dropout=0.0,
		use_skip=False,
		use_bn=False,
	)
	loss_func = nn.MSELoss()
	optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-1)
	wt_before = [torch.clone(layer.weight) for layer in mlp.main_layers[1::2]]
	loss_vals = simple_train_loop(mlp, X, y, loss_func, optimizer, num_epochs=5)
	for wb, layer in zip(wt_before, mlp.main_layers[1::2]):
		assert torch.all(layer.weight != wb).item()
	assert loss_vals[0] > loss_vals[-1]

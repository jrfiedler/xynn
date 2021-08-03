import pytest
import torch
from torch import nn

from xynn.ghost_norm import GhostBatchNorm, GhostLayerNorm


def test_ghostbatchnorm():
    gbn = GhostBatchNorm(3, 4, 0.2)
    assert gbn.inner_norm.num_features == 3
    assert gbn.inner_norm.momentum == 0.2
    assert gbn.virtual_batch_size == 4

    x = torch.tensor(
        [
            [-1,  0,  3],
            [ 0,  2, -3],
            [ 1, -2,  0],
            [ 0,  0,  0],
            [-2,  0,  1],
            [ 0,  3, -1],
            [ 2, -3,  0],
            [ 0,  0,  0],
            [-3,  0,  2],
            [ 0,  1, -2],
            [ 3, -1,  0],
            [ 0,  0,  0],
        ],
        dtype=torch.float,
    )
    out = gbn(x)
    expected = torch.tensor(
        [
            [-1.4142,  0.0000,  1.4142],
            [ 0.0000,  1.4142, -1.4142],
            [ 1.4142, -1.4142,  0.0000],
            [ 0.0000,  0.0000,  0.0000],
            [-1.4142,  0.0000,  1.4142],
            [ 0.0000,  1.4142, -1.4142],
            [ 1.4142, -1.4142,  0.0000],
            [ 0.0000,  0.0000,  0.0000],
            [-1.4142,  0.0000,  1.4142],
            [ 0.0000,  1.4142, -1.4142],
            [ 1.4142, -1.4142,  0.0000],
            [ 0.0000,  0.0000,  0.0000],
        ]
    )
    assert torch.allclose(out, expected)


def test_ghostlayernorm():
    gbn = GhostLayerNorm(3, 4)
    assert gbn.inner_norm.normalized_shape == (3,)
    assert gbn.virtual_batch_size == 4

    x = torch.tensor(
        [
            [-1,  0,  3],
            [ 0,  2, -3],
            [ 1, -2,  0],
            [ 0,  0,  0],
            [-2,  0,  1],
            [ 0,  3, -1],
            [ 2, -3,  0],
            [ 0,  0,  0],
            [-3,  0,  2],
            [ 0,  1, -2],
            [ 3, -1,  0],
            [ 0,  0,  0],
        ],
        dtype=torch.float,
    )
    out = gbn(x)
    expected = torch.tensor(
        [
            [-0.9806, -0.3922,  1.3728],
            [ 0.1622,  1.1355, -1.2978],
            [ 1.0690, -1.3363,  0.2673],
            [ 0.0000,  0.0000,  0.0000],
            [-1.3363,  0.2673,  1.0690],
            [-0.3922,  1.3728, -0.9806],
            [ 1.1355, -1.2978,  0.1622],
            [ 0.0000,  0.0000,  0.0000],
            [-1.2978,  0.1622,  1.1355],
            [ 0.2673,  1.0690, -1.3363],
            [ 1.3728, -0.9806, -0.3922],
            [ 0.0000,  0.0000,  0.0000],
        ],
        dtype=torch.float,
    )
    print(out)
    assert torch.allclose(out, expected, rtol=1e-4, atol=1e-4)

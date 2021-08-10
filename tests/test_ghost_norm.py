import pytest
import torch
from torch import nn

from xynn.ghost_norm import GhostBatchNorm


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

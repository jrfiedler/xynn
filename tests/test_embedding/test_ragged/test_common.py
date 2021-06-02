import numpy as np
import pandas as pd
import torch
from torch import nn
import pytest

from xynn.embedding.ragged.common import _check_embedding_size, _parse_embedding_size


def test__check_embedding_size_raises_error_for_bad_embedding_size():
    # bad name
    with pytest.raises(
        ValueError,
        match=(
            "str embedding_size value must be one of {'sqrt', 'log', 'fastai'}; "
            "got 'fourth_rt'"
        ),
    ):
        _check_embedding_size("fourth_rt")

    # single int not allowed
    with pytest.raises(
        TypeError,
        match="embedding_size 5 not understood",
    ):
        _check_embedding_size(5)

    # float values not allowed
    with pytest.raises(
        TypeError,
        match="embedding_size \[5, 10, 15.0\] not understood",
    ):
        _check_embedding_size([5, 10, 15.0])

    # wrong number of ints
    with pytest.raises(
        ValueError,
        match="number of embeddings must match number of fields, got 3 sizes and 4 fields",
    ):
        _check_embedding_size([5, 10, 15], [10, 20, 30, 40])


def test__check_embedding_size_with_uppercase():
    assert _check_embedding_size("SQRT") == "sqrt"
    assert _check_embedding_size("Log") == "log"
    assert _check_embedding_size("FastAI") == "fastai"


def test__check_embedding_size_with_ints():
    assert _check_embedding_size([5, 10, 15]) == [5, 10, 15]
    assert _check_embedding_size((5, 10, 15)) == (5, 10, 15)


def test__parse_embedding_size_with_sqrt():
    output = _parse_embedding_size("sqrt", 20, [4, 25, 64, 196, 400, 625, 1600])
    assert output == [2, 5, 8, 14, 20, 20, 20]


def test__parse_embedding_size_with_log():
    output = _parse_embedding_size("log", 7, [4, 25, 64, 196, 400, 625, 1600])
    assert output == [2, 4, 5, 6, 6, 7, 7]


def test__parse_embedding_size_with_fastai():
    output = _parse_embedding_size("fastai", 50, [4, 25, 64, 196, 400, 625, 1600])
    assert output == [3, 10, 16, 31, 46, 50, 50]


def test__parse_embedding_size_with_ints():
    output = _parse_embedding_size(
        [5, 5, 5, 5, 5, 5, 5], 20, [4, 25, 64, 196, 400, 625, 1600]
    )
    assert output == [5] * 7

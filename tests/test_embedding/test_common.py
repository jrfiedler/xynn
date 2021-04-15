import numpy as np
import pandas as pd
import torch
import pytest

from xynn.embedding.common import _isnan, _isnan_index, _unique, _value_counts
from .utils import example_data


def test__isnan():
    assert _isnan(float("nan"))
    assert not _isnan("NaN")
    assert not _isnan(20.22)
    assert not _isnan(20122)


def test__isnan_index_with_simple_example():
    data = pd.Series([10, 8, 6, 4, 2], index=[0, np.nan, 1, 4, np.nan])
    assert np.all(_isnan_index(data) == [False, True, False, False, True])


def test_that__unique_raises_error_on_bad_input():
    msg = "input should be Pandas DataFrame, NumPy array, or PyTorch Tensor"
    with pytest.raises(TypeError, match=msg):
        _unique([10, 8, 6, 8, 4, 2, 0, 10, 2, 0, 2])


def test__unique_on_numpy_examples():
    data = example_data()[["cat_a", "cat_b", "cat_c"]]
    data["cat_d"] = ["a", "b", np.nan, "c", "a", np.nan, np.nan, "a", "b", "c"]
    data["cat_e"] = [1, 1, np.nan, 0, 0, 0, np.nan, 1, 1, 0]
    uniques, has_nan = _unique(data.values)
    assert [set(values) for values in uniques] == [
        set([0, 1, 2, 3]), set([0, 1, 2]), set([0, 1]), set("abc"), set([0, 1])
    ]
    assert has_nan == [False, False, True, True, True]

    data = example_data()[["cat_c"]]
    data["cat_e"] = [1, 1, np.nan, 0, 0, 0, np.nan, 1, 1, 0]
    uniques, has_nan = _unique(data.values)
    assert [set(values) for values in uniques] == [set([0, 1]), set([0, 1])]
    assert has_nan == [True, True]


def test__unique_on_pandas_examples():
    data = example_data()[["cat_a", "cat_b", "cat_c"]]
    data["cat_d"] = ["a", "b", np.nan, "c", "a", np.nan, np.nan, "a", "b", "c"]
    data["cat_e"] = [1, 1, np.nan, 0, 0, 0, np.nan, 1, 1, 0]
    uniques, has_nan = _unique(data)
    assert [set(values) for values in uniques] == [
        set([0, 1, 2, 3]), set([0, 1, 2]), set([0, 1]), set("abc"), set([0, 1])
    ]
    assert has_nan == [False, False, True, True, True]

    data = example_data()[["cat_c"]]
    data["cat_e"] = [1, 1, np.nan, 0, 0, 0, np.nan, 1, 1, 0]
    uniques, has_nan = _unique(data)
    assert [set(values) for values in uniques] == [set([0, 1]), set([0, 1])]
    assert has_nan == [True, True]


def test__unique_on_tensor_example():
    data = example_data()[["cat_a", "cat_b", "cat_c"]]
    data["cat_e"] = [1, 1, np.nan, 0, 0, 0, np.nan, 1, 1, 0]
    uniques, has_nan = _unique(torch.from_numpy(data.values))
    assert [set(values) for values in uniques] == [
        set([0, 1, 2, 3]), set([0, 1, 2]), set([0, 1]), set([0, 1])
    ]
    assert has_nan == [False, False, True, True]

    data = example_data()[["cat_c"]]
    data["cat_e"] = [1, 1, np.nan, 0, 0, 0, np.nan, 1, 1, 0]
    uniques, has_nan = _unique(torch.from_numpy(data.values))
    assert [set(values) for values in uniques] == [set([0, 1]), set([0, 1])]
    assert has_nan == [True, True]


def test_that__value_counts_raises_error_on_bad_input():
    msg = "input should be Pandas DataFrame, NumPy array, or PyTorch Tensor"
    with pytest.raises(TypeError, match=msg):
        _value_counts([10, 8, 6, 8, 4, 2, 0, 10, 2, 0, 2])


def test__value_counts_on_numpy_examples():
    data = example_data()[["cat_a", "cat_b", "cat_c"]]
    data["cat_d"] = ["a", "b", np.nan, "c", "a", np.nan, np.nan, "a", "b", "c"]
    data["cat_e"] = [1, 1, np.nan, 0, 0, 0, np.nan, 1, 1, 0]
    unique_counts, nan_counts = _value_counts(data.values)
    assert unique_counts == [
        {0: 4, 1: 3, 2: 2, 3: 1},
        {0: 4, 1: 5, 2: 1},
        {0: 3, 1: 6},
        {"a": 3, "b": 2, "c": 2},
        {0: 4, 1: 4},
    ]
    assert nan_counts == [0, 0, 1, 3, 2]

    data = example_data()[["cat_c"]]
    data["cat_e"] = [1, 1, np.nan, 0, 0, 0, np.nan, 1, 1, 0]
    unique_counts, nan_counts = _value_counts(data.values)
    assert unique_counts == [{0: 3, 1: 6}, {0: 4, 1: 4}]
    assert nan_counts == [1, 2]


def test__value_counts_on_pandas_examples():
    data = example_data()[["cat_a", "cat_b", "cat_c"]]
    data["cat_d"] = ["a", "b", np.nan, "c", "a", np.nan, np.nan, "a", "b", "c"]
    data["cat_e"] = [1, 1, np.nan, 0, 0, 0, np.nan, 1, 1, 0]
    unique_counts, nan_counts = _value_counts(data)
    assert unique_counts == [
        {0: 4, 1: 3, 2: 2, 3: 1},
        {0: 4, 1: 5, 2: 1},
        {0: 3, 1: 6},
        {"a": 3, "b": 2, "c": 2},
        {0: 4, 1: 4},
    ]
    assert nan_counts == [0, 0, 1, 3, 2]

    data = example_data()[["cat_c"]]
    data["cat_e"] = [1, 1, np.nan, 0, 0, 0, np.nan, 1, 1, 0]
    unique_counts, nan_counts = _value_counts(data.values)
    assert unique_counts == [{0: 3, 1: 6}, {0: 4, 1: 4}]
    assert nan_counts == [1, 2]


def test__value_counts_on_tensor_examples():
    data = example_data()[["cat_a", "cat_b", "cat_c"]]
    data["cat_e"] = [1, 1, np.nan, 0, 0, 0, np.nan, 1, 1, 0]
    unique_counts, nan_counts = _value_counts(torch.from_numpy(data.values))
    assert unique_counts == [
        {0: 4, 1: 3, 2: 2, 3: 1}, {0: 4, 1: 5, 2: 1}, {0: 3, 1: 6}, {0: 4, 1: 4}
    ]
    assert nan_counts == [0, 0, 1, 2]

    data = example_data()[["cat_c"]]
    data["cat_e"] = [1, 1, np.nan, 0, 0, 0, np.nan, 1, 1, 0]
    unique_counts, nan_counts = _value_counts(torch.from_numpy(data.values))
    assert unique_counts == [{0: 3, 1: 6}, {0: 4, 1: 4}]
    assert nan_counts == [1, 2]

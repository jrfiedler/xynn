
import numpy as np
import pandas as pd


def example_data():
    data = pd.DataFrame(
        {
            "num_a": [i / 10 for i in range(10)],
            "num_b": range(10, 0, -1),
            "cat_a": [0, 1, 2, 3, 0, 1, 2, 0, 1, 0],
            "cat_b": [0, 1, 1, 0, 1, 0, 2, 1, 0, 1],
            "cat_c": [1, 1, 0, 0, 1, 1, 0, np.nan, 1, 1],
        }
    )
    return data

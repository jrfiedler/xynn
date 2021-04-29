
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import Dataset


class Reshape(nn.Module):
    def forward(self, X):
        return X.reshape((X.shape[0], -1))


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


class SimpleDataset(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

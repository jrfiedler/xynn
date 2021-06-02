"""
Base class for ragged embeddings and helpers for parsing embedding size

FastAI embedding size: arXiv preprint arXiv:2002.04688

"""

from math import ceil, sqrt, log
from collections.abc import Iterable as IterableClass
from typing import Any, Union, List, Dict, Iterable, Optional, Tuple

import numpy as np
import torch
from torch import nn, Tensor

from ..common import EmbeddingBase


def _check_embedding_size(embedding_size, num_categories=None):
    """Check that given `embedding_size` makes sense for ragged embeddings"""
    if isinstance(embedding_size, str):
        embedding_size = embedding_size.lower()
        if embedding_size not in ("sqrt", "log", "fastai"):
            raise ValueError(
                "str embedding_size value must be one of {'sqrt', 'log', 'fastai'}; "
                f"got '{embedding_size}'"
            )
    elif not isinstance(embedding_size, IterableClass) or not all(
        isinstance(size, int) for size in embedding_size
    ):
        raise TypeError(f"embedding_size {repr(embedding_size)} not understood")
    elif num_categories is not None and len(embedding_size) != len(num_categories):
        raise ValueError(
            "number of embeddings must match number of fields, got "
            f"{len(embedding_size)} sizes and {len(num_categories)} fields"
        )
    return embedding_size


def _parse_embedding_size(embedding_size, max_size, num_categories) -> List[int]:
    """
    Parse given `embedding_size` into a list of individual sizes,
    for ragged embeddings
    """
    _check_embedding_size(embedding_size, num_categories)
    # calculate the individual values if "sqrt" or "log"
    if isinstance(embedding_size, str):
        num_categories = np.array(num_categories)
        if embedding_size == "sqrt":
            base_size = np.ceil(np.sqrt(num_categories))
        elif embedding_size == "log":
            base_size = np.ceil(np.log(num_categories))
        else:  # embedding_size == "fastai":
            base_size = (1.6 * num_categories ** 0.56).round()
        clipped_size = np.clip(1, max_size, base_size).astype("int")
        embedding_size = list(clipped_size)
    else:  # iterable of int
        pass
    return embedding_size


class RaggedBase(EmbeddingBase):
    """Base class for embeddings that allow a different vector size for each field"""

    def __init__(self):
        super().__init__()
        self.embedding: Optional[nn.ModuleList] = None

    def weight_sum(self) -> Tuple[Tensor, Tensor]:
        """
        Sum of absolute value and square of embedding weights

        Return
        ------
        e1_sum : sum of absolute value of embedding values
        e2_sum : sum of squared embedding values
        """
        if not self._isfit:
            return 0.0, 0.0
        e1_sum = 0.0
        e2_sum = 0.0
        for embedding in self.embedding:
            e1_sum += embedding.weight.abs().sum()
            e2_sum += (embedding.weight ** 2).sum()
        return e1_sum, e2_sum

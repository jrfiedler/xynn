"""
Base class for uniform embeddings, with fixed vector size across fields

"""

from typing import Optional, Tuple

from torch import nn

from ..common import EmbeddingBase


class UniformBase(EmbeddingBase):
    """Base class for embeddings that have a single vector size for all fields"""

    def weight_sum(self) -> Tuple[float, float]:
        """
        Sum of absolute value and square of embedding weights

        Return
        ------
        e1_sum : sum of absolute value of embedding values
        e2_sum : sum of squared embedding values
        """
        if not self._isfit:
            return 0.0, 0.0
        e1_sum = self.embedding.weight.abs().sum().item()
        e2_sum = (self.embedding.weight ** 2).sum().item()
        return e1_sum, e2_sum
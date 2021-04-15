"""
Functions for fitting embeddings or checking embeddings

"""
from typing import Tuple, Optional
from collections import namedtuple

from torch.utils.data import DataLoader

from .common import _linear_agg, _unique_agg, _value_counts_agg, EmbeddingBase
from .uniform import UniformBase, LinearEmbedding, BasicEmbedding
from .ragged import RaggedEmbedding


EmbeddingInfo = namedtuple("EmbeddingInfo", ["num_fields", "embedding_size", "output_size"])


def _init_embed_info(embedding):
    if isinstance(embedding, (LinearEmbedding)):
        info_1 = 0
        info_2 = None
        agg_fn = _linear_agg
    else:
        info_1 = []
        info_2 = []
        if isinstance(embedding, (BasicEmbedding, RaggedEmbedding)):
            agg_fn = _unique_agg
        else:
            agg_fn = _value_counts_agg
    return info_1, info_2, agg_fn


def fit_embeddings(
    data: DataLoader,
    embedding_num: Optional[EmbeddingBase],
    embedding_cat: Optional[EmbeddingBase],
) -> Tuple[Optional[EmbeddingBase], Optional[EmbeddingBase]]:
    """
    Create the internal embedding info for the given numerical and categorical
    embeddings.

    Note: the passed embeddings are modified in place

    Parameters
    ----------
    data : torch.utils.data.DataLoader
    embedding_num : initialized embedding or None
    embedding_cat : initialized embedding or None

    Returns
    -------
    embedding_num, embedding_cat inputs, modified in place

    """
    if embedding_num is None and embedding_cat is None:
        return None, None

    # get initial values and aggregation functions
    if embedding_num is not None:
        num_info_1, num_info_2, num_agg_fn = _init_embed_info(embedding_num)
    if embedding_cat is not None:
        cat_info_1, cat_info_2, cat_agg_fn = _init_embed_info(embedding_cat)

    # iterate and update the initial aggregation values/objects
    for batch in data:
        if embedding_num is not None:
            num_info_1, num_info_2 = num_agg_fn(num_info_1, num_info_2, batch[0])
        if embedding_cat is not None:
            cat_info_1, cat_info_2 = cat_agg_fn(cat_info_1, cat_info_2, batch[1])

    # use aggregated values to set the embeddings
    if embedding_num is None:
        pass
    elif isinstance(embedding_num, LinearEmbedding):
        embedding_num.from_values(num_info_1)
    else:
        embedding_num.from_values(num_info_1, num_info_2)

    if embedding_cat is not None:
        embedding_cat.from_values(cat_info_1, cat_info_2)

    return embedding_num, embedding_cat


def _check_is_uniform(embedding, name):
    if embedding is None:
        return
    if not isinstance(embedding, UniformBase):
        raise TypeError(
            "only 'uniform' embeddings are allowed for this model; "
            f"{name} is not a uniform embedding"
        )


def check_uniform_embeddings(
    embedding_num: Optional[EmbeddingBase],
    embedding_cat: Optional[EmbeddingBase],
) -> EmbeddingInfo:
    # check embedding sizes and get derived values
    if embedding_num is None and embedding_cat is None:
        raise ValueError("embedding_num and embedding_cat cannot both be None")

    _check_is_uniform(embedding_num, "embedding_num")
    _check_is_uniform(embedding_cat, "embedding_cat")

    if (
        embedding_num is not None
        and embedding_cat is not None
        and not embedding_num.embedding_size == embedding_cat.embedding_size
    ):
        raise ValueError(
            "embedding sizes must be the same for numeric and catgorical; got "
            f"{embedding_num.embedding_size} and {embedding_cat.embedding_size}"
        )

    num_fields = 0
    if embedding_num is not None:
        num_fields += embedding_num.num_fields
        embedding_size = embedding_num.embedding_size

    if embedding_cat is not None:
        num_fields += embedding_cat.num_fields
        embedding_size = embedding_cat.embedding_size

    return EmbeddingInfo(num_fields, embedding_size, num_fields * embedding_size)

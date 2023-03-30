from typing import Union, NamedTuple, Optional, List

import faiss
import autofaiss

from .index_factory import index_factory

def to_faiss_metric_type(metric_type: Union[str, int]) -> int:
    """convert metric_type string/enum to faiss enum of the distance metric"""

    if metric_type in ["ip", "IP", faiss.METRIC_INNER_PRODUCT]:
        return faiss.METRIC_INNER_PRODUCT
    elif metric_type in ["l2", "L2", faiss.METRIC_L2]:
        return faiss.METRIC_L2
    else:
        raise ValueError("Metric currently not supported")

def create_empty_index(vec_dim: int, index_key: str, metric_type: Union[str, int]) -> faiss.Index:
    """Create empty index"""

    # Convert metric_type to faiss type
    metric_type = to_faiss_metric_type(metric_type)

    # Instanciate the index
    return index_factory(vec_dim, index_key, metric_type)


def check_if_index_needs_training(index_key: str) -> bool:
    """
    Function that checks if the index needs to be trained
    """

    if "IVF" in index_key:
        return True
    elif "IMI" in index_key:
        return True
    else:
        return False


def get_best_index(nb_vectors, dim_vector, max_index_memory_usage='100G'):
    best_indexes = autofaiss.external.optimize.get_optimal_index_keys_v2(
        nb_vectors = nb_vectors,
        dim_vector = dim_vector,
        max_index_memory_usage = max_index_memory_usage,
        flat_threshold = 1000,
        quantization_threshold = 10000,
        force_pq = None,
        make_direct_map = False,
        should_be_memory_mappable = False,
        ivf_flat_threshold = 1_000_000,
        use_gpu = False,
    )

    return best_indexes[0]

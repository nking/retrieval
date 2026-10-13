
import scann
from tensorflow import Tensor as tf_Tensor
from tensorflow import shape as tf_shape
from tensorflow import math as tf_math
import numpy as np
from usearch.index import Index

def build_usearch_searcher(embeddings: np.ndarray, top_k: int,
        normalize:bool=False, quantization:str=None) -> Index:
    '''
    Builds a fast, parallelized USearch (HNSW) index for vector retrieval.
    :param embeddings:
    :param top_k:
    :param normalize:
    :return:
    '''
    metric = "IP" #inner product
    
    if isinstance(embeddings, tf_Tensor):
        embeddings = embeddings.numpy()
        
    # Cast to float32 for initial index construction
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)
        
    # L2 normalize embeddings for dot-product equivalence
    if normalize:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings = embeddings / norms
    
    n_embeddings, ndim = embeddings.shape
    
    # Tune HNSW graph construction hyper-parameters according to scale
    if n_embeddings < 20000:
        connectivity = 16
        expansion_add = 64
        expansion_search = max(32, top_k * 2)
        if quantization is None:
            quantization = 'f32'
    else:
        connectivity = 24  # Graph degree bound per node
        expansion_add = 128  # Build-time search depth (higher = higher graph quality)
        expansion_search = max(64, top_k * 4)  # Query-time search depth
        if quantization is None:
            quantization = 'f16'
    
    # Initialize USearch Index
    index = Index(
        ndim=ndim,
        metric=metric,
        dtype=quantization,
        connectivity=connectivity,
        expansion_add=expansion_add,
        expansion_search=expansion_search
    )
    
    # Pre-allocate graph capacity to avoid reallocation overhead
    #index.reserve(n_embeddings) #not avail in python
    
    # Batch add keys and vectors using all available CPU threads (threads=0)
    keys = np.arange(n_embeddings, dtype=np.int64)
    index.add(keys, embeddings, threads=0)
    
    return index
    
def build_scann_searcher(embeddings: tf_Tensor,
        top_k: int, normalize:bool=False) -> scann.scann_ops_pybind.ScannSearcher:
    """
    build an ScANN indexer initialized with embeddings, and top_k number of nearest neighbors,
    and the brute force algorithm.
    TODO: tune configuration for high performance and accuracy.

    Usage: neighbors, distances = searcher.search_batched(query_embedding)

    to use scann.
    # https://github.com/google-research/google-research/blob/master/scann/docs/example.ipynb
    # https://github.com/google-research/google-research/blob/master/scann/docs/algorithms.md
    """
    if normalize:
        embeddings = tf_math.l2_normalize(embeddings, axis=-1)
        
    # Extract exact integer dimension
    n_embeddings = int(embeddings.shape[0])
    
    builder = scann.scann_ops_pybind.builder(
        db=embeddings,
        num_neighbors=top_k,
        distance_measure="dot_product"
    )
    
    # Brute-force path for small collections (MovieLens 1M item tower is ~3.9k items)
    if n_embeddings < 20000:
        return builder.score_brute_force(quantize=False).build()
    
    elif n_embeddings < 1_000_000:
        # Rule of thumb: num_leaves ~= sqrt(N)
        n_leaves = int(n_embeddings ** 0.5)
        # Search 20-30% of leaves
        n_leaves_to_search = max(30, int(n_leaves * 0.25))
        reorder_target = max(100, top_k * 10)
    else:
        # Rule of thumb: num_leaves ~= sqrt(N)
        n_leaves = int(n_embeddings ** 0.5)
        # Search ~5-10% of leaves to achieve >98% recall
        n_leaves_to_search = max(20, int(n_leaves * 0.05))
        reorder_target = max(500, top_k * 30)
    
    builder = builder.tree(
        num_leaves=n_leaves,
        num_leaves_to_search=n_leaves_to_search,
        training_sample_size=min(n_embeddings, 250000)
    )
    
    # Asymmetric Hashing (AH) with anisotropic quantization optimized for Two-Tower MIPS.
    # This is worth doing for almost any dataset over 10k
    builder = builder.score_ah(
        dimensions_per_block=2,
        anisotropic_quantization_threshold=0.2
    )
    
    # Rescore with exact inner product over top candidate pool
    builder = builder.reorder(reordering_num_neighbors=reorder_target)
    
    return builder.build()
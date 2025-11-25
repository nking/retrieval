import platform
from typing import Union

import numpy as np
import datetime
is_linux = platform.system().lower() == "linux"

#the imports are now handled by pyproject.toml or setup.py
# if linux, ScaNN is used,
# else Faiss is used.  The CPU version of Faiss is installed, but could be changed to faiss-gpu
FAISS_INSTALLED = False
SCANN_INSTALLED = False
if is_linux:
  import scann
  SCANN_INSTALLED = True
else:
  import faiss
  FAISS_INSTALLED = True
  
#@keras.saving.register_keras_serializable(package="",name="build_scann_searcher")
def build_scann_searcher(embeddings: np.ndarray, top_k: int):
  '''
  build an ScANN indexer initialized with embeddings, and top_k number of nearest neighbors,
  and the brute force algorithm.
  TODO: tune configuration for high performance and accuracy.

  Usage: neighbors, distances = searcher.search_batched(query_embedding)

  to use scann.
  # https://github.com/google-research/google-research/blob/master/scann/docs/example.ipynb
  # https://github.com/google-research/google-research/blob/master/scann/docs/algorithms.md
  '''
  if embeddings.dtype != np.float32:
    raise Exception(f'embeddings must be dtype np.float32\n')
  n = len(embeddings)
  searcher = scann.scann_ops_pybind.builder(embeddings, top_k, "dot_product") \
    .score_brute_force(quantize=False).build()
  return searcher

#@keras.saving.register_keras_serializable(package="", name="build_faiss_index")
def build_faiss_index(embeddings: np.ndarray, dimension: int):  # , ids: np.ndarray, dimension: int):
  dimension = int(dimension)
  if dimension < 1:
    raise Exception(
      f'dimension must be an integer > 0. dimension={dimension}\n')
  if embeddings.dtype != np.float32:
    raise Exception(f'embeddings must be dtype np.float32\n')
  '''
  Usage:
  distances, top_ids = index.search(query, k)
  distances = distances.reshape(-1)
  top_ids = top_ids.reshape(-1)

  to speed up performance, try:
  nlist = 100  # Number of clusters
  quantizer = faiss.IndexFlatL2(dimension)
  index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
  index.train(normalized_vectors)
  index.add(normalized_vectors)

  '''
  index = faiss.IndexFlatIP(dimension)  # IP for inner product.  search is cosine similarity
  # \index = faiss.IndexIDMap2(index)
  # movie_embeddings = np.reshape(movie_embeddings, (-1, np.shape(movie_embeddings)[2]))
  # print(f'shape movie_embeddings = {np.shape(movie_embeddings)}\n')
  # movie_embeddings shape is (num_movies, embed_dim)
  # print(f'len of embeddings, ids = {len(movie_embeddings), len(_movie_ids)}\n')
  # index.add_with_ids(embeddings, ids)  # self.movie_ids)
  index.add(embeddings)  # self.movie_ids)
  return index

class Retrieval:
  def __init__(self, user_movie_saved_model_dir:str, metadata_saved_model_dir:str,
               movies,  users, k: int = 1000):
    pass
  
  def get_movies_given_user(self, users, top_k):
    pass
  def get_user_given_user(self, users, top_k):
    pass
  def get_movies_given_movie(self, movies, top_k):
    pass
  def get_user_given_movie(self, movies, top_k):
    pass
  
  def is_user_known(self, user_id):
    pass
  
  def get_cold_start_rankings(self, top_k):
    pass
  
  def get_predictions(self, users):
    pass


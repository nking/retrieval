import platform
from typing import Union, List, Dict, Tuple
from movie_lens_retrieval.misc.inferrence_data_prep import convert_dict_inputs_to_tfexample_ser
import tensorflow as tf
import numpy as np
from rbloom import Bloom
import polars as pl

from movie_lens_retrieval.misc.Bayesian import BayesianShrinkageEstimator
from absl import logging
logging.set_verbosity(logging.WARNING)
logging.set_stderrthreshold(logging.WARNING)

#the imports are now handled by pyproject.toml or setup.py
# if linux, ScaNN is used,
# else Faiss is used.  The CPU version of Faiss is installed, but could be changed to faiss-gpu

class RetrieverAndRanker:
  #static members:
  is_linux = platform.system().lower() == "linux"
  if is_linux:
    import scann
    
  def __init__(self, user_movie_saved_model_dir:str, metadata_saved_model_dir:str,
    movies_path: str,  ratings_paths:list[str], max_k: int = 1000,
    #dynamically_rank:bool=False
    ):
    """
    NOTE that data should only contain data up to and including training data and eval data. no test
    data should be included.
    TODO: add cloud config options as needed and adapt as needed for hosted models
    :param user_movie_saved_model_dir: path to the saved_model directory for the main user_movie mondel
    :param metadata_saved_model_dir: path to the saved_model directory for the metadata model
    :param movies_paths: list of glob file path pattern to the parquet files containing all movie_ids
    :param ratings_path: list of glob file path pattern to the joined ratings parquet files
    :param max_k: the largest number of movies or users to be returned for each query element
    in an approximate nearest neighbor search.
    """
    self.max_k = max_k
    
    self.movies_path = movies_path
    if '*' in movies_path:
      movies_pl = pl.read_parquet(movies_path, glob=True)
    else:
      movies_pl = pl.read_parquet(movies_path, glob=True)
    
    self.ratings_paths = ratings_paths
    ratings_pl = RetrieverAndRanker._read_ratings(ratings_paths)
    
    #self.dynamically_rank = dynamically_rank
    self.shift_bytes = 13
    self.user_bloom_filter, self.user_movie_bloom_filter = RetrieverAndRanker._init_rbloom(ratings_pl, self.shift_bytes)
    
    self.loaded_user_movie_model = tf.saved_model.load(user_movie_saved_model_dir)
    
    #create indexes using saved_models
    inputs_dict_np = RetrieverAndRanker._polars_to_numpy_dict(ratings_pl) #TODO: improve if possible
    examples_list = convert_dict_inputs_to_tfexample_ser(inputs_dict_np) #TODO: improve if possible
    # longest step in computations:
    self.user_indexers = RetrieverAndRanker._create_user_indexers(examples_list, self.loaded_user_movie_model, self.max_k)
    self.movie_indexers = RetrieverAndRanker._create_movie_indexers(examples_list, self.loaded_user_movie_model, self.max_k)
    self.user_ids = inputs_dict_np['user_id'].numpy().tolist()
    self.movie_ids = inputs_dict_np['movie_id'].numpy().tolist()
    
    #adds column "predicted_from_genres"
    predicted_mdm = RetrieverAndRanker._get_metadata_predictions(metadata_saved_model_dir, movies_pl)
    self.cold_start_rankings = self._prep_cold_start_rankings(ratings_pl, movies_pl, "predicted_from_genres")
    
  
  def _get_metadata_predictions(metadata_saved_model_dir:str, movies:pl.DataFrame) -> pl.DataFrame:
    #the model requires ratings_joined column,
    # but only useds movie_id and genres as inputs
    inputs_dict_np = RetrieverAndRanker._polars_to_numpy_dict(movies)
    del inputs_dict_np['title']
    #add fake data for the user_id,timestamp,gender,age,occupation
    RetrieverAndRanker.fill_missing_cols_with_fake(inputs_dict_np)
   
    examples_list = convert_dict_inputs_to_tfexample_ser(inputs_dict_np)
  
    movie_model = tf.saved_model.load(metadata_saved_model_dir)
    infer = movie_model.signatures["serving_default"]
    INPUT_KEY = list(infer.structured_input_signature[1].keys())[0]
    
    predicted = infer(**{INPUT_KEY: examples_list})['outputs'].numpy()
    movies = movies.with_columns(
      pl.Series(name="predicted_from_genres", values=predicted)
    )
    return movies # movie_id, title, genres, predicted_from_genres
    
  def _create_movie_indexers(inputs: Union[Dict[str, np.ndarray], List[bytes]], loaded_user_movie_model, max_k:int) -> np.ndarray:
    """
    note that the indexes are w.r.t the ordering given in ratings_pl
    :param ratings_pl:
    :return:
    """
    embeddings_np = RetrieverAndRanker._create_movie_embeddings(inputs, loaded_user_movie_model)
    if RetrieverAndRanker.is_linux:
      indexer = RetrieverAndRanker.build_scann_searcher(embeddings=embeddings_np, top_k=max_k)
    else:
      d = np.shape(embeddings_np)[1]
      indexer = RetrieverAndRanker.build_faiss_index(embeddings=embeddings_np, dimension=d)
    return indexer

  def _init_rbloom(ratings: pl.DataFrame, shift_bytes:int) -> Tuple[Bloom, Bloom]:
    """
    :param ratings: polars dataframe having columns "user_id" and "movie_id".  rating doesn't have to be prosent but the relationship
    implicitly indicates that there was a rating
    :param shift_bytes:
    :return: tuple of tuple of user bloom filter and user_ids for it, and a user-movie bloom filter
    """
    # 12 MB memory?
    users = ratings['user_id'].unique().to_list()
    u_bf = Bloom(10*len(users), 0.01)
    u_bf.update(users)
    # 17 MB memory?
    n_user_movies = len(ratings.count())
    um_bf = Bloom(10 * n_user_movies, 0.001)
    columns = ratings.columns
    user_idx = ratings.columns.index('user_id')
    movie_idx = ratings.columns.index('movie_id')
    for row_tuple in ratings.iter_rows():
      um_bf.add(row_tuple[user_idx] << shift_bytes + row_tuple[movie_idx])
    return u_bf, um_bf
 
  def _agg_movie_counts(ratings: pl.DataFrame, movies:pl.DataFrame) -> pl.DataFrame:
    pivoted = ratings.pivot(
      index="movie_id", columns="rating", values="rating", aggregate_function="count",
    ).fill_null(0).sort("movie_id")
    pivoted = pivoted.with_columns(
      pl.col(name).cast(pl.Int32) for name in
      [name for name in pivoted.columns if name != "movie_id"]
    )
    missing_df = movies.join(pivoted.select(pl.col("movie_id")),
      on="movie_id", how="anti").select(pl.col("movie_id"))
    rating_cols = [col for col in pivoted.columns if col != 'movie_id']
    missing_df = missing_df.with_columns(
      pl.lit(0).alias(col_name) for col_name in rating_cols
    )
    return pivoted.vstack(missing_df)
  
  def _prep_cold_start_rankings(ratings: pl.DataFrame, movies:pl.DataFrame, max_k:int, prior_rating_column_name:str=None):
    pivoted = RetrieverAndRanker._agg_movie_counts(ratings, movies)
    b = BayesianShrinkageEstimator(pivoted, prior_rating_column_name)
    return b.get_top(max_k)
    
  def get_cold_start_rankings(self, top_k:int):
      self.cold_start_rankings[:top_k].copy()
   
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
    import scann
    bind1 = scann.scann_ops_pybind.builder(db=embeddings, num_neighbors=top_k, distance_measure="dot_product")
    searcher = bind1.score_brute_force(quantize=False).build()
    return searcher
  
  def _polars_to_numpy_dict(df: pl.DataFrame) -> Dict[str, np.ndarray]:
    inp_dict = df.to_dict(as_series=False)
    for key in inp_dict.keys():
      if isinstance(inp_dict[key][0], str):
        arr_arr = [[bytes(item, 'utf-8')] for item in inp_dict[key]]
      else:
        arr_arr = [[item] for item in inp_dict[key]]
      inp_dict[key] = np.array(arr_arr)
    return inp_dict
  
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
    import faiss
    index = faiss.IndexFlatIP(dimension)  # IP for inner product.  search is cosine similarity
    # \index = faiss.IndexIDMap2(index)
    # movie_embeddings = np.reshape(movie_embeddings, (-1, np.shape(movie_embeddings)[2]))
    # print(f'shape movie_embeddings = {np.shape(movie_embeddings)}\n')
    # movie_embeddings shape is (num_movies, embed_dim)
    # print(f'len of embeddings, ids = {len(movie_embeddings), len(_movie_ids)}\n')
    # index.add_with_ids(embeddings, ids)  # self.movie_ids)
    index.add(embeddings)  # self.movie_ids)
    return index
    
  def _read_ratings(ratings_paths: List[str]) -> pl.DataFrame:
    combined = None
    if len(ratings_paths) > 2:
      for ratings_path in ratings_paths:
        if '*' in ratings_path:
          ratings_pl = pl.read_parquet(ratings_path, glob=True)
        else:
          ratings_pl = pl.read_parquet(ratings_path)
        if combined is None:
          combined = ratings_pl
        else:
          combined = combined.extend(ratings_pl)
    else:
      for ratings_path in ratings_paths:
        if '*' in ratings_path:
          ratings_pl = pl.read_parquet(ratings_path, glob=True)
        else:
          ratings_pl = pl.read_parquet(ratings_path)
        if combined is None:
          combined = ratings_pl
        else:
          combined = combined.vstack(ratings_pl)
    return combined
    
  def _create_user_embeddings(inputs: Union[Dict[str, np.ndarray], List[bytes]], loaded_user_movie_model) -> np.ndarray:
    """
    given inputs, use the query_candidate model to make embeddings.
    :param inputs: dictionary of inputs where keys must be all columns from the joined ratings file that the models
    were trained upon.  Note that for the user model, only usr_id and age are used, so the other items can be fake.
    :return: embeddings usable for the vector approx nearest neighbor searches
    """
    if isinstance(inputs, dict):
      examples_list = convert_dict_inputs_to_tfexample_ser(inputs)
    else:
      examples_list = inputs
    infer = loaded_user_movie_model.signatures["serving_query"]
    INPUT_KEY = list(infer.structured_input_signature[1].keys())[0]
    embeddings_list = infer(**{INPUT_KEY: examples_list})['outputs'] # k X embed_dim
    #print(f'np.shape(embeddings={np.shape(embeddings_list)})')
    return np.vstack(embeddings_list)
  
  def _create_movie_embeddings(inputs: Union[Dict[str, np.ndarray], List[bytes]], loaded_user_movie_model) -> np.ndarray:
    """
    given inputs, use the serving_candidate model to make embeddings.
    :param inputs: dictionary of inputs where keys must be all columns from the joined ratings file that the models
    were trained upon.  Note that for the movie model, nly ovie_id and genres are used, so the other items can be fake.
    :return: embeddings usable for the vector approx nearest neighbor searches
    """
    if isinstance(inputs, dict):
      examples_list = convert_dict_inputs_to_tfexample_ser(inputs)
    else:
      examples_list = inputs
    infer = loaded_user_movie_model.signatures["serving_candidate"]
    INPUT_KEY = list(infer.structured_input_signature[1].keys())[0]
    embeddings_list = infer(**{INPUT_KEY: examples_list})['outputs']  # k X embed_dim
    print(f'np.shape(embeddings={np.shape(embeddings_list)})')
    return np.vstack(embeddings_list)
  
  def _create_user_indexers(inputs: Union[Dict[str, np.ndarray], List[bytes]], loaded_user_movie_model, max_k:int) -> np.ndarray:
    """
    note that the indexes are w.r.t the ordering given in ratings_pl
    :param ratings_pl:
    :return:
    """
    embeddings_np = RetrieverAndRanker._create_user_embeddings(inputs, loaded_user_movie_model)
    if RetrieverAndRanker.is_linux:
      indexer = RetrieverAndRanker.build_scann_searcher(embeddings=embeddings_np, top_k=max_k)
    else:
      d = np.shape(embeddings_np)[1]
      indexer = RetrieverAndRanker.build_faiss_index(embeddings=embeddings_np, dimension=d)
    return indexer

  def is_user_known(self, user_id):
    return user_id in self.user_bloom_filter
  
  def has_seen_movie(self, user_id, movie_id):
    return (user_id << self.shift_bytes) + movie_id in self.user_movie_bloom_filter

  #def get_predictions(self, data):
  #  pass
  
  def fill_missing_cols_with_fake(inp_dict:Dict[str, Union[int, bytes]]):
    """populate an inputs dictionary for a model signature needing the joined raw file columns"""
    for key in inp_dict:
      n = len(inp_dict[key])
      break
    for key in ["movie_id", "user_id", "age", "occupation"]:
      if key not in inp_dict:
        inp_dict[key] = np.array([[0] for _ in range(n)])
    if "gender" not in inp_dict:
      inp_dict['gender'] = np.random.choice(np.array([b'M', b'F']), size=n, replace=True)
      inp_dict['gender'] = inp_dict['gender'].reshape(-1, 1)
    if "timestamp" not in inp_dict:
      inp_dict['timestamp'] = np.array([[966606623] for _ in range(n)])
    if "genres" not in inp_dict:
      inp_dict['genres'] = np.array([[b'Drama'] for _ in range(n)])

  def get_users_given_users(self, user_data_dict:Dict[str, Union[int, bytes]], top_k:int):
    #to find similar users requires all ratings_joined columns, but only the user_id and age are used for latest model.
    # TODO: the best hyperparams for a model need to be read by this class and used where needed.
    #vector search for similar users.  then find similar movies, then use ranking, sort and return top_k
    RetrieverAndRanker.fill_missing_cols_with_fake(user_data_dict)
    # TODO: create user embeddings
    # TODO: find top_k similar users for each row in user_data_dict
    user_embeddings = RetrieverAndRanker._create_user_embeddings(user_data_dict, self.loaded_user_movie_model)
    if RetrieverAndRanker.is_linux:
      #indexes are insert order indexes
      neighbor_idxs, distances = self.user_indexers.search_batched(user_embeddings, top_k=top_k)
    else:
      distances, neighbor_idxs = self.user_indexers.search(user_embeddings, top_k)
    nearest_user_ids = [[self.user_ids[i] for i in _list] for _list in neighbor_idxs]
    return nearest_user_ids
  
  def get_movies_given_movie(self, movie_data_dict:Dict[str, Union[int, bytes]], top_k:int):
    #to find similar movies requires all ratings_joined columns, but only the movie_id and genres are used for latest model.
    RetrieverAndRanker.fill_missing_cols_with_fake(movie_data_dict)
    # TODO: create user embeddings
    # TODO: find top_k similar users for each row in user_data_dict
    movie_embeddings = RetrieverAndRanker._create_movieembeddings(movie_data_dict, self.loaded_user_movie_model)
    if RetrieverAndRanker.is_linux:
      #indexes are insert order indexes
      neighbor_idxs, distances = self.movie_indexers.search_batched(movie_embeddings, top_k=top_k)
    else:
      distances, neighbor_idxs = self.movie_indexers.search(movie_embeddings, top_k)
    nearest_user_ids = [[self.movie_ids[i] for i in _list] for _list in neighbor_idxs]
    return nearest_user_ids
  
  def get_user_given_movie(self, movie_data_dict:Dict[str, Union[int, bytes]], top_k:int):
    RetrieverAndRanker.fill_missing_cols_with_fake(movie_data_dict)
    movie_embeddings = RetrieverAndRanker._create_movie_embeddings(movie_data_dict, self.loaded_user_movie_model)
    if RetrieverAndRanker.is_linux:
      #indexes are insert order indexes
      neighbor_idxs, distances = self.user_indexers.search_batched(movie_embeddings, top_k=top_k)
    else:
      distances, neighbor_idxs = self.user_indexers.search(movie_embeddings, top_k)
    nearest_user_ids = [[self.user_ids[i] for i in _list] for _list in neighbor_idxs]
    return nearest_user_ids

  def get_movies_given_user(self, user_data_dict:Dict[str, Union[int, bytes]], top_k:int):
    RetrieverAndRanker.fill_missing_cols_with_fake(user_data_dict)
    user_embeddings = RetrieverAndRanker._create_movieembeddings(user_data_dict, self.loaded_user_movie_model)
    if RetrieverAndRanker.is_linux:
      #indexes are insert order indexes
      neighbor_idxs, distances = self.movie_indexers.search_batched(user_embeddings, top_k=top_k)
    else:
      distances, neighbor_idxs = self.movie_indexers.search(user_embeddings, top_k)
    nearest_user_ids = [[self.movie_ids[i] for i in _list] for _list in neighbor_idxs]
    return nearest_user_ids
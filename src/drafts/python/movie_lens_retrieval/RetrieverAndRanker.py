from typing import Union, List, Dict, Tuple, Any
import tensorflow as tf
from rbloom import Bloom
from google.protobuf import text_format
import random
import glob

from absl import logging
logging.set_verbosity(logging.WARNING)
logging.set_stderrthreshold(logging.WARNING)

import scann

"""
This is a version containing a Bloom Filter.  It's not ideal as the Blloom Filter should be
a separate component outside of the Retrieval.  Kept it for an all-in-one toy, but the
use of it in whether a user has already seen a movie of not has to be further looked into
as the false positive rate is higher than expected..


NOTE that data should only contain data up to and including training data and eval data. no test
data should be included.

TODO: should use MLMD and model lineage for the saved_models and data and schema

TODO: add cloud config options as needed and adapt as needed for hosted models

For cloud based RetrieverAndRanker, can adapt for services for ScANN and bloom filters.
"""

class RetrieverAndRanker:
  
  def __init__(self, user_movie_saved_model_dir:str,
    movies_path: str,  users_path:str, ratings_paths:list[str],
    movies_pivot_path:str, max_k: int = 1000,
    movies_batch_size:int=256, users_batch_size:int=256, ratings_batch_size:int=256
    ):
    """
    param user_movie_saved_model_dir: path to the saved_model directory for the main user_movie model.
        the embeddings model signatures are in this.
    :param movies_path:
    :param users_path:
    :param ratings_paths:
    :param movie_pivot_path:
    :param movie_pivot_pred_col_name:
    :param max_k:
    :param movies_batch_size:
    :param ratings_batch_size:
    """
    
    self.max_k = max_k
    
    self.loaded_user_movie_model = tf.saved_model.load(user_movie_saved_model_dir)
    
    #  easier to hard code a dictionary for now than install and import tfx transform to read schema
    #string serialized examples => dict of inputs
    self.feature_spec = {"user_id": tf.io.FixedLenFeature([], tf.int64),
      "movie_id":tf.io.FixedLenFeature([], tf.int64),
      "rating" : tf.io.FixedLenFeature([], tf.int64),
      "timestamp": tf.io.FixedLenFeature([], tf.int64),
      "gender" : tf.io.FixedLenFeature([], tf.string),
      "age" : tf.io.FixedLenFeature([], tf.int64),
      "occupation" : tf.io.FixedLenFeature([], tf.int64),
      "genres" : tf.io.FixedLenFeature([], tf.string)}
    
    # create indexes using saved_models
    # longest step in computations because it reads out entire datasets.
    # the user_ids are needed to lookup the ids from indexes from search results, though, the movie_lens datasets
    # were created so that these inputs are sequential and start at 1, so the scann indexes returned are the true ids - 1,
    #  and so reading out the user_ids and movie_ids can be excluded to speed up RetrieverAndRanker construction.
    self.user_indexers, self.user_ids = RetrieverAndRanker._create_user_indexers(users_path,
      self.loaded_user_movie_model, self.feature_spec, self.max_k, batch_size=users_batch_size)
    self.movie_indexers, self.movie_ids = RetrieverAndRanker._create_movie_indexers(movies_path,
      self.loaded_user_movie_model, self.feature_spec, self.max_k, batch_size=movies_batch_size)
    
    pivot_feature_spec = {
      "movie_id":tf.io.FixedLenFeature([], tf.int64),
    }
    """
      "1" : tf.io.FixedLenFeature([], tf.int64),
      "2": tf.io.FixedLenFeature([], tf.int64),
      "3" : tf.io.FixedLenFeature([], tf.string),
      "4" : tf.io.FixedLenFeature([], tf.int64),
      "5" : tf.io.FixedLenFeature([], tf.int64)}
    """

    self.cold_start_rankings = RetrieverAndRanker._prep_cold_start_rankings(
      movies_pivot_path=movies_pivot_path,
      feature_spec=pivot_feature_spec,
      max_k=self.max_k)
    
    #the ratings_ds is largest to load.
    # it is used for the user_mode bloom filter to check whether user has not seen movie
    ratings_ds = RetrieverAndRanker._joined_ratings_tt_to_ds(ratings_paths, self.feature_spec,
      batch_size=2048)

    #using rbloom filters for less memory than tf.lookup.StaticHashTable.  would need to be reconsidered for cloud infrastruture
    self.shift_bits = 13
    self.user_bloom_filter, self.user_movie_bloom_filter = RetrieverAndRanker._init_rbloom(self.user_ids,
        ratings_ds, self.shift_bits)
        
  def _joined_ratings_tt_to_ds(file_path_globs:List[str], feature_spec:dict, batch_size:int=256):
    GLOB_PATTERNS = file_path_globs
    GZIP_PATTERN = r".*\.gz$"
    def load_tfrecords(filepath):
      """Creates a TFRecordDataset, setting compression based on the file path."""
      is_compressed = tf.strings.regex_full_match(filepath, GZIP_PATTERN)
      compression_type = tf.where(is_compressed, tf.constant("GZIP"), tf.constant(""))
      return tf.data.TFRecordDataset(filepath, compression_type=compression_type)
    
    files_dataset = tf.data.Dataset.list_files(GLOB_PATTERNS, shuffle=False)
    
    records_dataset = files_dataset.interleave(
      load_tfrecords, cycle_length=tf.data.AUTOTUNE,  block_length=1,
      num_parallel_calls=tf.data.AUTOTUNE)
    
    def parse_tf_example(example_proto, feature_spec):
      return tf.io.parse_single_example(example_proto, feature_spec)
    
    dataset = (records_dataset
      .map(lambda x: parse_tf_example(x, feature_spec)).cache()
      .batch(batch_size).prefetch(tf.data.AUTOTUNE))
    
    return dataset
  
  def _create_movie_indexers(movies_path: str,
    loaded_user_movie_model, feature_spec:dict, max_k:int, batch_size:int=256) -> Tuple[scann.scann_ops_pybind.ScannSearcher, List[int]]:
    """
    given the glob file pattern for the movies file path, create a ScANN searcher instance
    with embeddings from the model and input files.
    
    :param movies_path: the file path pattern to the TFRecords holding all movies in the format
      of the joined ratings tfrecords.  the movie_id and genres columns are the only used here though.
    :param loaded_user_movie_model: the saved_model that will be used to extract the "serving_query"
    signature
    :param feature_spec: the feature scpe needed to deserialize the tfrecord serialized strings into
    a dataset of dictionary of tensors.
    :param max_k: the maximum number of items that will be possible to return from a search of the
    resulting seracher.
    :return: an instance of ScANN searcher loaded with embedding made from the TFRecords at movies_path
    """
    
    _ct = "GZIP" if movies_path.endswith(".gz") else None
    file_paths = glob.glob(movies_path)
    ds_ser = tf.data.TFRecordDataset(file_paths, compression_type=_ct)
    
    candidate_model = loaded_user_movie_model.signatures["serving_candidate"]
    INPUT_KEY = list(candidate_model.structured_input_signature[1].keys())[0]
    embeddings = []
    for batch in ds_ser.batch(batch_size):
      emb = candidate_model(**{INPUT_KEY: batch})['outputs'] 
      embeddings.append(emb)
    embeddings = tf.concat(embeddings, 0)

    indexer = RetrieverAndRanker.build_scann_searcher(embeddings=embeddings, top_k=max_k)

    def parse_tf_example(example_proto, feature_spec):
      row = tf.io.parse_single_example(example_proto, feature_spec)
      return row['movie_id']
    ds_ids = ds_ser.map(lambda x: parse_tf_example(x, feature_spec))
    all_ids_tensor = RetrieverAndRanker._get_all_ids_as_tensor(ds_ids)
    ids = all_ids_tensor.numpy().tolist()

    return indexer, ids

  def _init_rbloom(user_ids: List[int], ratings_ds: tf.data.Dataset, bits_shift:int=13) -> Tuple[Bloom, Bloom]:
    
    # 12 MB memory?
    u_bf = Bloom(5*len(user_ids), 0.01)
    u_bf.update(user_ids)
      
    # 17 MB memory for 0.001?
    n_ratings = ratings_ds.reduce(0, lambda x, _: x + 1).numpy()
    
    um_bf = Bloom(2*n_ratings, 0.00001)
    #TODO: consider batching:
    for batch in ratings_ds:
      user_idx = batch['user_id'].numpy()
      movie_idx = batch['movie_id'].numpy()
      um_bf.update((user_idx << bits_shift) + movie_idx)
      
    return u_bf, um_bf
 
  def _prep_cold_start_rankings(movies_pivot_path:str, feature_spec:Dict[str, Any], max_k:int=1000,
    batch_size:int=256):
    
    _ct = "GZIP" if movies_pivot_path.endswith(".gz") else None
    file_paths = glob.glob(movies_pivot_path)
    pivot_ds_ser = tf.data.TFRecordDataset(file_paths, compression_type="GZIP")
    
    def parse_tf_example(example_proto, feature_spec):
      return tf.io.parse_single_example(example_proto, feature_spec)
    pivot_ds = pivot_ds_ser.map(lambda x: parse_tf_example(x, feature_spec))
    
    #the pivot_ds rows are already ordered by descending prior_rating_column_name
    movie_ids = []
    i = 0
    for x in pivot_ds.batch(batch_size):
      if i >= max_k:
        break
      m = x['movie_id'].numpy()
      movie_ids.extend(m)
      i += len(m)
    return movie_ids
  
  def _create_serialized_tfexample(inputs:Dict[str, Union[int, str]]) -> bytes:
    expected_keys = {'user_id':int, 'movie_id':int, 'rating':int, "timestamp":int,
      "gender":str, "age":int, "occupation":int, "genres":str}
    feature_map = {}
    try:
      for name, value in inputs.items():
        element_type = expected_keys[name]
        if element_type == float:
          f = tf.train.Feature(float_list=tf.train.FloatList(value=[float(value)]))
        elif element_type == int or element_type == bool:
          f = tf.train.Feature(int64_list=tf.train.Int64List(value=[int(value)]))
        elif element_type == str:
          f = tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode('utf-8')]))
        else:
          raise ValueError(f"element_type={element_type}, but only float, int, and str classes are handled.")
        feature_map[name] = f
    except Exception as ex:
      logging.error(f"ERROR: {ex}, name={name}, value={value}, element_type={element_type}")
      raise ex
    try:
      # add fake entries to make consistent with the joined ratings file columns
      for out_name, out_type in expected_keys.items():
        if out_name in feature_map:
          continue
        if out_type == float:
          f = tf.train.Feature(float_list=tf.train.FloatList(value=[0.0]))
        elif out_type == int or element_type == bool:
          if out_name == "timestamp":
            value = 956703932
          else:
            value = 0
          f = tf.train.Feature(
            int64_list=tf.train.Int64List(value=[value]))
        elif out_type == str:
          if out_name == "genres":
            value = b"Drama"
          elif out_name == "gender":
            value = random.choice([b"M", b"F"])
          else:
            value = b""
          f = tf.train.Feature( bytes_list=tf.train.BytesList(value=[value]))
        else:
          raise ValueError(
            f"out_type={out_type}, but only float, int, and str classes are handled.")
        feature_map[out_name] = f
      tf_example = tf.train.Example(features=tf.train.Features(feature=feature_map))
      return tf_example.SerializeToString()
    except Exception as ex:
      logging.error( f"ERROR: {ex}, out_name={out_name}, out_type={out_type}")
      raise ex
    
  #@keras.saving.register_keras_serializable(package="",name="build_scann_searcher")
  def build_scann_searcher(embeddings:tf.Tensor, top_k: int):
    '''
    build an ScANN indexer initialized with embeddings, and top_k number of nearest neighbors,
    and the brute force algorithm.
    TODO: tune configuration for high performance and accuracy.
  
    Usage: neighbors, distances = searcher.search_batched(query_embedding)
  
    to use scann.
    # https://github.com/google-research/google-research/blob/master/scann/docs/example.ipynb
    # https://github.com/google-research/google-research/blob/master/scann/docs/algorithms.md
    '''
    bind1 = scann.scann_ops_pybind.builder(db=embeddings, num_neighbors=top_k, distance_measure="dot_product")
    searcher = bind1.score_brute_force(quantize=False).build()
    return searcher
  
  def _create_user_embeddings(inputs: Union[Dict[str, Union[int, str]], List[Dict[str, Union[int, str]]]],
    loaded_user_movie_model) -> tf.Tensor:
    """
    given inputs, use the query_candidate model to make embeddings.
    :param inputs: dictionary of inputs where keys must be all columns from the joined ratings file that the models
    were trained upon.  Note that for the user model, only usr_id and age are used, so the other items can be fake.
    :return: embeddings usable for the vector approx nearest neighbor searches.
    output format is tensor of shape (len(inputs as a list),)
    """
    if not isinstance(inputs, list):
      inputs = [inputs]
    examples_list = []
    for inp_dict in inputs:
      if not isinstance(inp_dict, dict) or "user_id" not in inp_dict or "age" not in inp_dict:
        raise ValueError("expecting inputs  to be a dictionary that includes user_id and age or a list of dictionaries including those")
      examples_list.append(RetrieverAndRanker._create_serialized_tfexample(inp_dict))
    infer = loaded_user_movie_model.signatures["serving_query"]
    INPUT_KEY = list(infer.structured_input_signature[1].keys())[0]
    embeddings_list = infer(**{INPUT_KEY: examples_list})['outputs'] # k X embed_dim
    #embeddings_list is a single tensory with a 2D-array of embeddings
    return embeddings_list
  
  def _create_movie_embeddings(inputs: Union[Dict[str, Union[int, str]], List[Dict[str, Union[int, str]]]],
    loaded_user_movie_model) -> tf.Tensor:
    """
    given inputs, use the serving_candidate model to make embeddings.
    :param inputs: dictionary of inputs where keys must be all columns from the joined ratings file that the models
    were trained upon.  Note that for the movie model, only ovie_id and genres are used, so the other items can be fake.
    :return: embeddings usable for the vector approx nearest neighbor searches
    """
    if not isinstance(inputs, list):
      inputs = [inputs]
    examples_list = []
    for inp_dict in inputs:
      if not isinstance(inp_dict, dict) or "movie_id" not in inp_dict or "genres" not in inp_dict:
        raise ValueError("expecting inputs  to be a dictionary that includes movie_id and genres or a list of dictionaries including those")
      examples_list.append(RetrieverAndRanker._create_serialized_tfexample(inp_dict))
    infer = loaded_user_movie_model.signatures["serving_candidate"]
    INPUT_KEY = list(infer.structured_input_signature[1].keys())[0]
    embeddings_list = infer(**{INPUT_KEY: examples_list})['outputs']  # k X embed_dim
    # embeddings_list is a single tensory with a 2D-array of embeddings
    return embeddings_list
  
  @tf.function
  def _get_all_ids_as_tensor(ds):
    initial_state = tf.constant([], dtype=tf.int64)
    def reduce_fn(previous_tensor, current_element):
      return tf.concat([previous_tensor,
        tf.expand_dims(current_element, axis=0)], axis=0)
    all_ids_tensor = ds.reduce(initial_state=initial_state,
      reduce_func=reduce_fn)
    return all_ids_tensor
  
  def _create_user_indexers(users_path:str, loaded_user_movie_model,
    feature_spec:dict, max_k:int, batch_size:int=256):
    #-> Tuple[scann.ScannSearcher, List[int]]:

    _ct = "GZIP" if users_path.endswith(".gz") else None
    file_paths = glob.glob(users_path)
    ds_ser = tf.data.TFRecordDataset(file_paths, compression_type=_ct)
    query_model = loaded_user_movie_model.signatures["serving_query"]
    INPUT_KEY = list(query_model.structured_input_signature[1].keys())[0]
    embeddings = []
    for batch in ds_ser.batch(batch_size):
      emb = query_model(**{INPUT_KEY: batch})['outputs']  # k X embed_dim
      embeddings.append(emb)
    embeddings = tf.concat(embeddings, 0)

    indexer = RetrieverAndRanker.build_scann_searcher(embeddings=embeddings, top_k=max_k)
    
    def parse_tf_example(example_proto, feature_spec):
      row = tf.io.parse_single_example(example_proto, feature_spec)
      return row['user_id']
    ds_ids = ds_ser.map(lambda x: parse_tf_example(x, feature_spec))
    all_ids_tensor = RetrieverAndRanker._get_all_ids_as_tensor(ds_ids)
    ids = all_ids_tensor.numpy().tolist()

    return indexer, ids

  def is_user_known(self, user_id):
    return user_id in self.user_bloom_filter
  
  def has_seen_movie(self, user_id, movie_id):
    return ((user_id << self.shift_bits) + movie_id) in self.user_movie_bloom_filter

  def get_users_given_users(self, user_data_dict:Union[
    Dict[str, Union[int, str]], List[Dict[str, Union[int, str]]]], top_k:int):
    if top_k < 1:
      raise ValueError('top_k must be >= 1')
    if top_k > self.max_k:
      top_k = self.max_k
    embeddings_tensor = RetrieverAndRanker._create_user_embeddings(user_data_dict, self.loaded_user_movie_model)
    neighbor_idxs, distances = self.user_indexers.search_batched(embeddings_tensor, top_k)
    #returns numpy ndarrays
    nearest_user_ids = [[self.user_ids[i] for i in _list] for _list in neighbor_idxs]
    if not isinstance(user_data_dict, list):
      user_data_dict = [user_data_dict]
    for input_dict, movie_ids in zip(user_data_dict, nearest_user_ids):
      user_id = input_dict['user_id']
      if user_id in movie_ids:
        movie_ids.remove(user_id)
    return nearest_user_ids
  
  def get_movies_given_movies(self, movie_data_dict:Union[Dict[str, Union[int, str]], List[Dict[str, Union[int, str]]]],
    top_k:int):
    if top_k < 1:
      raise ValueError('top_k must be >= 1')
    if top_k > self.max_k:
      top_k = self.max_k
    #to find similar movies requires all ratings_joined columns, but only the movie_id and genres are used for latest model.
    
    movie_embeddings = RetrieverAndRanker._create_movie_embeddings(movie_data_dict, self.loaded_user_movie_model)
    neighbor_idxs, distances = self.movie_indexers.search_batched(movie_embeddings, top_k)
    nearest_movie_ids = [[self.movie_ids[i] for i in _list] for _list in neighbor_idxs]
    if not isinstance(movie_data_dict, list):
      movie_data_dict = [movie_data_dict]
    for input_dict, movie_ids in zip(movie_data_dict, nearest_movie_ids):
      movie_id = input_dict['movie_id']
      if movie_id in movie_ids:
        movie_ids.remove(movie_id)
    return nearest_movie_ids
  
  def get_users_given_movies(self, movie_data_dict:Union[Dict[str, Union[int, str]], List[Dict[str, Union[int, str]]]],
    top_k:int):
    if top_k < 1:
      raise ValueError('top_k must be >= 1')
    if top_k > self.max_k:
      top_k = self.max_k
    movie_embeddings = RetrieverAndRanker._create_movie_embeddings(movie_data_dict, self.loaded_user_movie_model)
    neighbor_idxs, distances = self.user_indexers.search_batched(movie_embeddings, top_k)
    nearest_user_ids = [[self.user_ids[i] for i in _list] for _list in neighbor_idxs]
    return nearest_user_ids

  def get_movies_given_users(self, user_data_dict:Union[Dict[str, Union[int, str]], List[Dict[str, Union[int, str]]]],
    top_k:int, remove_aready_seen:bool=True):
    if top_k < 1:
      raise ValueError('top_k must be >= 1')
    if top_k > self.max_k:
      top_k = self.max_k
    user_embeddings = RetrieverAndRanker._create_user_embeddings(user_data_dict, self.loaded_user_movie_model)
    #choose more than top_k because we will filter to remove already seen
    k = min(top_k*1000, self.max_k) if remove_aready_seen else top_k
    neighbor_idxs, distances = self.movie_indexers.search_batched(user_embeddings, k)
    nearest_movie_ids = [[self.movie_ids[i] for i in _list] for _list in neighbor_idxs]
    if remove_aready_seen:
      if not isinstance(user_data_dict, list):
        user_data_dict = [user_data_dict]
      tmp_list = []
      for input_dict, movie_ids in zip(user_data_dict, nearest_movie_ids):
        user_id = int(input_dict['user_id'])
        tmp_i = []
        for movie_id in movie_ids:
          if len(tmp_i) == top_k:
            break
          if not self.has_seen_movie(user_id, movie_id):
            tmp_i.append(movie_id)
        tmp_list.append(tmp_i)
      nearest_movie_ids = tmp_list
    return nearest_movie_ids
  
  @classmethod
  def _parse_pbtxt_file(cls, schema_uri, message):
    try:
      with tf.io.gfile.GFile(schema_uri, 'r') as f:
        contents = f.read()
    except tf.errors.NotFoundError:
      print(f"Error: File not found at {schema_uri}")
    except Exception as e:
      print(f"An error occurred: {e}")
    text_format.Parse(contents, message)
    return message

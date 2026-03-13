from typing import Union, List, Dict, Tuple, Any
import tensorflow as tf
#from google.protobuf import text_format
import random
import glob

from absl import logging
from tensorflow.python.ops.variables import moving_average_variables

logging.set_verbosity(logging.WARNING)
logging.set_stderrthreshold(logging.WARNING)

import scann

"""
NOTE that data should only contain data up to and including training data and eval data. no test
data should be included.

TODO: should use MLMD and model lineage for the saved_models and data and schema

TODO: add cloud config options as needed and adapt as needed for hosted models

For cloud based RetrieverAndRanker, can adapt for services for ScANN.

Bloom filters or efficient database and cache system can be used outside of this component to:
(1) check if user has already seen movies returned by this component.

"""

class RetrieverAndRanker:
  
  def __init__(self, user_movie_saved_model_dir:str,
    movies_path: str,  users_path:str,
    movies_pivot_path:str, max_k: int = 1000,
    movies_batch_size:int=256, users_batch_size:int=256
    ):
    """
    param user_movie_saved_model_dir: path to the saved_model directory for the main user_movie model.
        the embeddings model signatures are in this.
        
    :param movies_path: glob pattern to the TFRecords of the movies, where the columns are expected to be
    the same as the joined ratings column, even though only the movie_id and genres columns are used.
    
    :param users_path: glob pattern to the TFRecords of the users, where the columns are expected to be
    the same as the joined ratings column, even though only the user_id and age columns are used.
    
    :param movie_pivot_path: glob pattern to the TFRecord of the pivot file contianining the movie_ids ordered by
    the weighted Bayesian shrinkage estimates from using the metadata model as a prior.
    
    :param max_k: the maximum number of embeddings to return from a ScANN embedding search.  This should be higher
    than the top_k desired to account for later removing movies already seen.  The default is 1000.
    
    :param movies_batch_size: a batch size to use when creating embeddings when initializing the candidate ScANN searcher.
    The default is 256.
    
    :param ratings_batch_size: a batch size to use when creating embedding when initializing the query ScANN searcher.
    The default is 256.
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
    
    self.user_indexers, self.users_ht, self.user_age_ht = self._create_user_indexer(users_path,
      self.loaded_user_movie_model, self.max_k, batch_size=users_batch_size)
    self.movie_indexers, self.movies_ht, self.movie_genres_ht = self._create_movie_indexer(movies_path,
      self.loaded_user_movie_model, self.max_k, batch_size=movies_batch_size)
    
    pivot_feature_spec = {
      "movie_id":tf.io.FixedLenFeature([], tf.int64),
    }
    """
      "1" : tf.io.FixedLenFeature([], tf.int64),
      "2": tf.io.FixedLenFeature([], tf.int64),
      "3" : tf.io.FixedLenFeature([], tf.string),
      "4" : tf.io.FixedLenFeature([], tf.int64),
      "5" : tf.io.FixedLenFeature([], tf.int64)}
      'prediction_mm', 'total_votes', 'movie_ratings_mean', 'weighted_rating'
    """

    self.cold_start_rankings = RetrieverAndRanker._prep_cold_start_rankings(
      movies_pivot_path=movies_pivot_path,
      feature_spec=pivot_feature_spec,
      max_k=self.max_k)
    
  def _create_movie_indexer(self, movies_path: str,
    loaded_user_movie_model, max_k:int, batch_size:int=256) \
    -> Tuple[scann.scann_ops_pybind.ScannSearcher, tf.lookup.StaticHashTable]:
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

    ht, ht2 = self._create_movie_static_hashtables(ds_ser,  batch_size)
    
    return indexer, ht, ht2
  
  def _create_user_static_hashtables(self, ds_ser:tf.data.Dataset, batch_size:int=256) \
    -> Tuple[tf.lookup.StaticHashTable, tf.lookup.StaticHashTable]:
    
    feature_spec = {"user_id": tf.io.FixedLenFeature(shape=[], dtype=tf.int64, default_value=None),
      "age": tf.io.FixedLenFeature(shape=[], dtype=tf.int64, default_value=None)}
    def parse_tf_example(example_proto, feature_spec):
      return tf.io.parse_single_example(example_proto, feature_spec)
    ds = ds_ser.map(lambda x: parse_tf_example(x, feature_spec))
    ids = []
    i = 0
    ages = []
    for batch in ds.batch(batch_size):
      ids.append(batch["user_id"])
      ages.append(batch["age"])
      i += len(batch["user_id"])
    ids = tf.concat(ids, 0)
    ages = tf.concat(ages, 0)
    indexes = tf.constant([idx for idx in range(i)], dtype=tf.int64)
    ht = tf.lookup.StaticHashTable(
      tf.lookup.KeyValueTensorInitializer(indexes, ids),
      default_value=-1
    )
    ht2 = tf.lookup.StaticHashTable(
      tf.lookup.KeyValueTensorInitializer(ids, ages),
      default_value=-1
    )
    
    #print(f'[{key}] ht lookup [0,3]={ht.lookup(tf.constant([0,3], dtype=tf.int64))}')

    return ht, ht2
  
  def _create_movie_static_hashtables(self, ds_ser:tf.data.Dataset, batch_size:int=256) \
    -> Tuple[tf.lookup.StaticHashTable, tf.lookup.StaticHashTable]:
    
    feature_spec = {"movie_id": tf.io.FixedLenFeature(shape=[], dtype=tf.int64, default_value=None),
      "genres": tf.io.FixedLenFeature(shape=[], dtype=tf.string, default_value=None)}
    def parse_tf_example(example_proto, feature_spec):
      return tf.io.parse_single_example(example_proto, feature_spec)
    def clean_genres(features):
      features['genres'] = tf.strings.regex_replace(input=features['genres'],
        pattern="Children's",rewrite="Children")
      return features
    ds = ds_ser.map(lambda x: parse_tf_example(x, feature_spec)).batch(batch_size)
    #unfortunately, my data pre-processing is part of the TFX pipeline, and this is not the transformed data, its
    # the raw, so insert clean_genres here..  TODO: consider refactoring to use the model's transform signature here
    ds = ds.map(clean_genres, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    ids = []
    i = 0
    genres = []
    for batch in ds:
      ids.append(batch["movie_id"])
      genres.append(batch["genres"])
      i += len(batch["movie_id"])
    ids = tf.concat(ids, 0)
    genres = tf.concat(genres, 0)
    indexes = tf.constant([idx for idx in range(i)], dtype=tf.int64)
    ht = tf.lookup.StaticHashTable(
      tf.lookup.KeyValueTensorInitializer(indexes, ids),
      default_value=-1
    )
    ht2 = tf.lookup.StaticHashTable(
      tf.lookup.KeyValueTensorInitializer(ids, genres),
      default_value=""
    )
    
    #print(f'[{key}] ht lookup [0,3]={ht.lookup(tf.constant([0,3], dtype=tf.int64))}')

    return ht, ht2
  
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
      m = x['movie_id'].numpy().tolist()
      movie_ids.extend(m)
      i += len(m)
    return movie_ids
  
  def _create_dictionary_of_tensors(self, inputs:List[Dict[str, Union[int, str]]]):
    """
    inputs: list of dictionary where each dictionary has form:
    
    example inputs:{"user_id":1, "age":25, "movie_id":1, "genres":"Animation|Children's|Comedy"
    or
    example inputs:{"user_id":1, "age":25,
      "movie_id":[1, 3952],
      "genres":["Animation|Children's|Comedy", "Drama|Thriller"]
      
    or list of those
    
    :return:
    """
    reqs = set(["user_id", "movie_id", "age", "genres"])
    final_keys = {'user_id': tf.int64, 'movie_id': tf.int64,
      "timestamp": tf.int64, "gender": tf.string, "age": tf.int64,
      "occupation": tf.int64, "genres": tf.string}

    for inp_dict in inputs:
      for key in reqs:
        if key not in inp_dict:
          raise ValueError(f"missing required key: {key}")
    
    outp_dict = {key: [] for key in final_keys}
    for inp_dict in inputs:
      u_id = inp_dict['user_id']
      age = inp_dict['age']
      m_id = inp_dict['movie_id']
      genres = inp_dict['genres']
      if isinstance(m_id, list) and not isinstance(genres, list) \
        or isinstance(genres, list) and not isinstance(m_id, list):
          raise ValueError("genres and m_id must both be lists or scalars")
      if not isinstance(m_id, list):
        m_id = [m_id]
        genres = [genres]
      for m, g in zip(m_id, genres):
        outp_dict['user_id'].append([u_id])
        outp_dict['age'].append([age])
        outp_dict['movie_id'].append([m])
        outp_dict['genres'].append([g])
        for key in final_keys:
          if key not in reqs:
            if key in inp_dict:
              outp_dict[key].append([inp_dict[key]])
            elif key == "timestamp":
              outp_dict[key].append([975768870]) #latest time in training dataset
            elif key == "gender":
              outp_dict[key].append([random.choice(["M", "F"])])
            elif key == "occupation":
              outp_dict[key].append([random.randint(0, 21)])
    for key in outp_dict:
      outp_dict[key] = tf.constant(outp_dict[key], dtype=final_keys[key])
    return outp_dict
      
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
            value = 975768870
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
  
  def get_cold_start_movie_recommendations(self, top_k: int = 100):
    return self.cold_start_rankings[:top_k].copy()
  
  def _create_user_embeddings(inputs: Union[Dict[str, Union[int, str]], List[Dict[str, Union[int, str]]]],
    loaded_user_movie_model, user_age_hashtable: tf.lookup.StaticHashTable=None) -> tf.Tensor:
    """
    given inputs, use the query_candidate model to make embeddings.
    :param inputs: dictionary of inputs where keys must be all columns from the joined ratings file that the models
    were trained upon.  Note that for the user model, only user_id and age are used, so the other items can be fake.
    :return: embeddings usable for the vector approx nearest neighbor searches.
    output format is tensor of shape (len(inputs as a list),)
    """
    if not isinstance(inputs, list):
      inputs = [inputs]
    examples_list = []
    for inp_dict in inputs:
      if not isinstance(inp_dict, dict) or "user_id" not in inp_dict:
        raise ValueError("expecting inputs  to be a dictionary that includes user_id "
            "or a list of dictionaries containing user_id")
      user_id = inp_dict["user_id"]
      if user_age_hashtable and "age" not in inp_dict:
        age = user_age_hashtable.lookup(tf.constant(user_id, dtype=tf.int64))
        if age == -1:
          raise ValueError(f"user_id {user_id} is not registered, and 'age' is missing from inputs")
        inp_dict["age"] = age.numpy().item()
      elif "age" not in inp_dict:
        raise ValueError(f"'age' is missing from inputs")
      examples_list.append(RetrieverAndRanker._create_serialized_tfexample(inp_dict))
    infer = loaded_user_movie_model.signatures["serving_query"]
    INPUT_KEY = list(infer.structured_input_signature[1].keys())[0]
    embeddings_list = infer(**{INPUT_KEY: examples_list})['outputs'] # k X embed_dim
    #embeddings_list is a single tensory with a 2D-array of embeddings
    return embeddings_list
  
  def _create_movie_embeddings(inputs: Union[Dict[str, Union[int, str]], List[Dict[str, Union[int, str]]]],
    loaded_user_movie_model, movie_genres_hashtable: tf.lookup.StaticHashTable=None) -> tf.Tensor:
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
      if not isinstance(inp_dict, dict) or "movie_id" not in inp_dict:
        raise ValueError("expecting inputs  to be a dictionary that includes movie_id "
          "or a list of dictionaries containing movie_id")
      movie_id = inp_dict["movie_id"]
      if movie_genres_hashtable and "genres" not in inp_dict:
        genres = movie_genres_hashtable.lookup(tf.constant(movie_id, dtype=tf.int64))
        if genres == b"":
          raise ValueError(f"movie_id {movie_id} is not registered, and 'genres' is missing from inputs")
        inp_dict["genres"] = genres.numpy().decode('utf-8')
      elif "genres" not in inp_dict:
        raise ValueError(f"'genres' is missing from inputs")
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
  
  def _create_user_indexer(self, users_path:str, loaded_user_movie_model,
    max_k:int, batch_size:int=256) -> Tuple[scann.scann_ops_pybind.ScannSearcher, tf.lookup.StaticHashTable]:
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
    
    id_ht, age_ht = self._create_user_static_hashtables(ds_ser, batch_size)
    
    return indexer, id_ht, age_ht

  def get_users_given_users(self, user_data_dict:Union[
    Dict[str, Union[int, str]], List[Dict[str, Union[int, str]]]], top_k:int):
    if top_k < 1:
      raise ValueError('top_k must be >= 1')
    if top_k > self.max_k:
      top_k = self.max_k
    embeddings_tensor = RetrieverAndRanker._create_user_embeddings(user_data_dict, self.loaded_user_movie_model,
      self.user_age_ht)
    neighbor_idxs, distances = self.user_indexers.search_batched(embeddings_tensor, top_k)
    nearest_user_ids = [[int(idx) for idx in self.users_ht.lookup(tf.constant(_list, dtype=tf.int64)).numpy()] for _list in neighbor_idxs]
    if not isinstance(user_data_dict, list):
      user_data_dict = [user_data_dict]
    for input_dict, movie_ids in zip(user_data_dict, nearest_user_ids):
      user_id = input_dict['user_id']
      if user_id in movie_ids:
        movie_ids.remove(user_id)
    return nearest_user_ids
  
  def get_movies_given_movies(self, movie_data_dict:Union[Dict[str, Union[int, str]], List[Dict[str, Union[int, str]]]],
    top_k:int):
    #TODO: add ranking to the results
    if top_k < 1:
      raise ValueError('top_k must be >= 1')
    if top_k > self.max_k:
      top_k = self.max_k
    #to find similar movies requires all ratings_joined columns, but only the movie_id and genres are used for latest model.
    
    movie_embeddings = RetrieverAndRanker._create_movie_embeddings(movie_data_dict, self.loaded_user_movie_model,
      self.movie_genres_ht)
    neighbor_idxs, distances = self.movie_indexers.search_batched(movie_embeddings, top_k)
    nearest_movie_ids = [[int(idx) for idx in self.movies_ht.lookup(tf.constant(_list, dtype=tf.int64)).numpy()] for _list in neighbor_idxs]
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
    movie_embeddings = RetrieverAndRanker._create_movie_embeddings(movie_data_dict, self.loaded_user_movie_model,
      self.movie_genres_ht)
    neighbor_idxs, distances = self.user_indexers.search_batched(movie_embeddings, top_k)
    nearest_user_ids = [[int(idx) for idx in self.users_ht.lookup(tf.constant(_list, dtype=tf.int64)).numpy()] for _list in neighbor_idxs]
    return nearest_user_ids

  def get_movies_given_users(self, user_data_dict:Union[Dict[str, Union[int, str]], List[Dict[str, Union[int, str]]]],
    top_k:int, use_ranker:bool=True):
    if top_k < 1:
      raise ValueError('top_k must be >= 1')
    if top_k > self.max_k:
      top_k = self.max_k
    user_embeddings = RetrieverAndRanker._create_user_embeddings(user_data_dict, self.loaded_user_movie_model,
      self.user_age_ht)
    neighbor_idxs, distances = self.movie_indexers.search_batched(user_embeddings, top_k)
    nearest_movie_ids = [[int(idx) for idx in self.movies_ht.lookup(tf.constant(_list, dtype=tf.int64)).numpy()] for _list in neighbor_idxs]
    
    if use_ranker:
      #rank results by the predicted ratings for the user and movie combination
      if not isinstance(user_data_dict, list):
        user_data_dict = [user_data_dict]
      outputs = []
      for inp_dict, movie_ids in zip(user_data_dict, nearest_movie_ids):
        #reformat into:
        #user_inp = {'user_id': 635, 'age': 56,
        #         'movie_id': [1704, 1940],
        #         'genres': ['Drama', 'Drama']}
        #reformat into
        user_inp = {**inp_dict}
        user_inp['movie_id'] = movie_ids
        genres = self.movie_genres_ht.lookup(tf.constant(movie_ids, dtype=tf.int64))
        user_inp['genres'] = genres.numpy().tolist()
        preds = self.get_predictions(user_inp)
        sorted_comb = sorted(zip(preds, movie_ids))
        sorted_ratings, sorted_movies = zip(*sorted_comb)
        outputs.append(sorted_movies)
      return outputs
    
    return nearest_movie_ids
  
  def user_is_known(self, user_id) -> bool:
    id = self.users_ht.lookup(tf.constant(user_id, dtype=tf.int64))
    return False if id == -1 else True
  
  def get_predictions(self, user_data_dict: Union[
    Dict[str, Union[int, str]], List[Dict[str, Union[int, str]]]], as_tensor:bool=False):
    """
    given users and movies, return predictions.
    :param user_data_dict: a dictionary of list of dictionaries containing "user_id", "age",
    "movie_id" and "genres".  the movie_id and genres can be lists instead of scalars, and
    if one is a list, the other must be also.
    
    example input:{"user_id":1, "age":25, "movie_id":1, "genres":"Animation|Children's|Comedy"
    
    example input:{"user_id":1, "age":25,
      "movie_id":[1, 3952],
      "genres":["Animation|Children's|Comedy", "Drama|Thriller"]

    :return: list of predictions
    """
    if not isinstance(user_data_dict, list):
      user_data_dict = [user_data_dict]
    
    batch = self._create_dictionary_of_tensors(user_data_dict)
    
    infer_default_for_dict = self.loaded_user_movie_model.signatures["serving_default_dict"]
    
    predictions = infer_default_for_dict(
      age=batch['age'],
      gender=batch['gender'],
      genres=batch['genres'],
      movie_id=batch['movie_id'],
      occupation=batch['occupation'],
      timestamp=batch['timestamp'],
      user_id=batch['user_id'])
    tensor_pred = predictions['outputs']
    tensor_pred = tf.reshape(tensor_pred, -1)
    
    if as_tensor:
      return tensor_pred
    
    return tensor_pred.numpy().tolist()
    
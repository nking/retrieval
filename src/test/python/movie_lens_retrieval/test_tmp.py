import unittest

import tensorflow as tf
from helper import *

class TestBayesianShrinkageEstimator(unittest.TestCase):
  def setUp(self):
    self.joined_ratings_tf_records = [os.path.join(get_project_dir(),
      "src/test/resources/data/sorted_1/tfrecord-*.gz"),
      os.path.join(get_project_dir(),"src/test/resources/data/sorted_2/tfrecord-*.gz"),]
    saved_models_dir = os.path.join(get_project_dir(), "src/main/resources/serving_models")
    self.user_movie_models_dir = os.path.join(saved_models_dir, "user_movie_model")
    self.metadata_model_dir = os.path.join(saved_models_dir, "metadata_model")
    
  def test_indexer_tensors(self):
    
    loaded_model = tf.saved_model.load(self.user_movie_models_dir)
    query_model = loaded_model.signatures["serving_query"]
    INPUT_KEY = list(query_model.structured_input_signature[1].keys())[0]
    
    import os
    import re
    
    # 1. Define your glob patterns and helper functions
    GLOB_PATTERNS = self.joined_ratings_tf_records
    
    # Pattern to check for GZIP compression (used for the load function)
    GZIP_PATTERN = r".*\.gz$"
    
    def load_tfrecords(filepath):
      """Creates a TFRecordDataset, setting compression based on the file path."""
      
      # Use tf.strings.regex_full_match to determine compression type
      is_compressed = tf.strings.regex_full_match(filepath, GZIP_PATTERN)
      
      # Use tf.where to set the compression_type argument (TensorFlow control flow)
      compression_type = tf.where(is_compressed, tf.constant("GZIP"),
                                  tf.constant(""))
      
      # The tf.data.TFRecordDataset constructor is called with the determined compression_type Tensor
      return tf.data.TFRecordDataset(filepath,
                                     compression_type=compression_type)
    
    # 1. List files matching all glob patterns
    # tf.data.Dataset.list_files automatically accepts a list of patterns.
    files_dataset = tf.data.Dataset.list_files(
      GLOB_PATTERNS,
      shuffle=False
      # Shuffle the order of files for better dataset mixing
    )
    
    # 2. Use interleave to read from multiple files concurrently
    # This is the most efficient method for reading many files.
    records_dataset = files_dataset.interleave(
      load_tfrecords,
      cycle_length=tf.data.AUTOTUNE,
      block_length=1,
      num_parallel_calls=tf.data.AUTOTUNE
    )
    #use largest batch size possible while avoiding out-of-memory errors
    dataset = records_dataset.cache().batch(256).prefetch(tf.data.AUTOTUNE)
    
    def _parse_function(example_proto):
      row = tf.io.parse_single_example(example_proto, col_name_feature_types)
      # only keeping the id's
      return row
    
    embeddings = []
    ids = []
    for batch in dataset:
      emb = query_model(**{INPUT_KEY: batch})['outputs']  # k X embed_dim
      embeddings.append(emb)
    self.assertTrue(len(emb) > 0)
    test_emb = embeddings[0]
    embeddings = tf.concat(embeddings, 0)
    
    import scann
    bind1 = scann.scann_ops_pybind.builder(db=embeddings, num_neighbors=10,
      distance_measure="dot_product")
    searcher = bind1.score_brute_force(quantize=False).build()
    #showing that scann indexer works with tensors (documentation says have to build scann[tf], but that is not true now.
    
    neighbor_idxs, distances = searcher.search_batched(test_emb, 10)
    self.assertIsNotNone(neighbor_idxs)
    self.assertTrue(0 in neighbor_idxs)
    
    #checking user inputs ****
    tf_records = os.path.join(get_project_dir(),
      "src/test/resources/data/user_emb_inp/tfrecords*gz")
    dataset = tf.data.TFRecordDataset(tf_records, compression_type=".GZIP")
    dataset = records_dataset.cache().batch(256).prefetch(tf.data.AUTOTUNE)
    
    embeddings = []
    for batch in dataset:
      emb = query_model(**{INPUT_KEY: batch})['outputs']  # k X embed_dim
      embeddings.append(emb)
    self.assertTrue(len(emb) > 0)
    test_emb = embeddings[0]
    embeddings = tf.concat(embeddings, 0)
    
    import scann
    bind1 = scann.scann_ops_pybind.builder(db=embeddings, num_neighbors=10,
                                           distance_measure="dot_product")
    searcher = bind1.score_brute_force(quantize=False).build()
    neighbor_idxs, distances = searcher.search_batched(test_emb, 10)
    self.assertIsNotNone(neighbor_idxs)
    self.assertTrue(0 in neighbor_idxs)
    
    ## check movie inputs
    candidate_model = loaded_model.signatures["serving_candidate"]
    INPUT_KEY = list(candidate_model.structured_input_signature[1].keys())[0]
    
    movie_tf_records = os.path.join(get_project_dir(),
      "src/test/resources/data/movie_emb_inp/tfrecords*gz")
    dataset = tf.data.TFRecordDataset(movie_tf_records,
                                      compression_type=".GZIP")
    dataset = records_dataset.cache().batch(256).prefetch(tf.data.AUTOTUNE)
    
    embeddings = []
    for batch in dataset:
      emb = candidate_model(**{INPUT_KEY: batch})[
        'outputs']  # k X embed_dim
      embeddings.append(emb)
    self.assertTrue(len(emb) > 0)
    test_emb = embeddings[0]
    embeddings = tf.concat(embeddings, 0)
    
    import scann
    bind1 = scann.scann_ops_pybind.builder(db=embeddings,
                                           num_neighbors=10,
                                           distance_measure="dot_product")
    searcher = bind1.score_brute_force(quantize=False).build()
    neighbor_idxs, distances = searcher.search_batched(test_emb, 10)
    self.assertIsNotNone(neighbor_idxs)
    self.assertTrue(0 in neighbor_idxs)
    
  
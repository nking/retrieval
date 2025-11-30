import unittest

import tensorflow as tf
from helper import *

class TestBayesianShrinkageEstimator(unittest.TestCase):
  def setUp(self):
    self.movie_tf_records = os.path.join(get_project_dir(), "src/test/resources/data/movie/tfrecords*gz")
    #just 1 record for test:
    self.joined_ratings_tf_records = os.path.join(get_project_dir(),
        "src/test/resources/data/sorted_2/tfrecord-00000-of-00004.gz")
    saved_models_dir = os.path.join(get_project_dir(), "src/main/resources/serving_models")
    self.user_movie_models_dir = os.path.join(saved_models_dir, "user_movie_model")
    self.metadata_model_dir = os.path.join(saved_models_dir, "metadata_model")
    
  def test_indexer_tensors(self):
    
    loaded_model = tf.saved_model.load(self.user_movie_models_dir)
    query_model = loaded_model.signatures["serving_query"]
    INPUT_KEY = list(query_model.structured_input_signature[1].keys())[0]
    
    dataset = tf.data.TFRecordDataset(self.joined_ratings_tf_records, compression_type="GZIP")
    dataset = dataset.batch(32)
    
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
    #showing that scann indexer works with tensors (documentation says have to build scann[tf], but that is not true now.
    
    neighbor_idxs, distances = searcher.search_batched(test_emb, 10)
    self.assertIsNotNone(neighbor_idxs)
    self.assertTrue(0 in neighbor_idxs)
    
  
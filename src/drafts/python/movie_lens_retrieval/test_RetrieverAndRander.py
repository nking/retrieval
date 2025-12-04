import os.path
import unittest

# sys.path and imports might need to be adjusted for drafts.
#  this code has been placed here, but not tested in this directory

import tensorflow as tf
import sys
import os
sys.path.append(os.path.join(os.getcwd(), "src/drafts/python/movie_lens_retrieval/RetrieverAndRanker.py"))
import RetrieverAndRanker

from helper import *

class TestRetrievalAndRanker(unittest.TestCase):
  def setUp(self):
    
    self.joined_ratings_tf_records = [os.path.join(get_project_dir(),
      "src/test/resources/data/sorted_1/tfrecord-*.gz"),
      os.path.join(get_project_dir(),"src/test/resources/data/sorted_2/tfrecord*.gz"),]
    
    saved_models_dir = os.path.join(get_project_dir(), "src/main/resources/serving_models")
    self.user_movie_models_dir = os.path.join(saved_models_dir, "user_movie_model")
    
    self.movie_inputs = os.path.join(get_project_dir(),
      "src/test/resources/data/movie_emb_inp/tfrecord*.gz")
    self.user_inputs = os.path.join(get_project_dir(),
      "src/test/resources/data/user_emb_inp/tfrecord*.gz")
    self.movies_predictions_pivot = os.path.join(get_project_dir(),
      "src/test/resources/data/ratings_and_predictions_pivot/tfrecord*.gz")
    self.movies_predictions_pivot_prior_col_name = "weighted_rating"
    self.feature_spec = {"user_id": tf.io.FixedLenFeature([], tf.int64),
      "movie_id":tf.io.FixedLenFeature([], tf.int64),
      "rating" : tf.io.FixedLenFeature([], tf.int64),
      "timestamp": tf.io.FixedLenFeature([], tf.int64),
      "gender" : tf.io.FixedLenFeature([], tf.string),
      "age" : tf.io.FixedLenFeature([], tf.int64),
      "occupation" : tf.io.FixedLenFeature([], tf.int64),
      "genres" : tf.io.FixedLenFeature([], tf.string)}
    self.max_k = 10
      
  def _est_indexer_tensors(self):
   
    loaded_model = tf.saved_model.load(self.user_movie_models_dir)
    
    inputs1 = [{'user_id': 1, 'age': 10}, {'user_id': 2, 'age': 16}]
    inputs2 = [{'movie_id': 1, 'genres': "Animation|Children's|Comedy"},
      {'movie_id': 2, 'genres': "Adventure|Children's|Fantasy"}]
    
    for j in [0, 1]:
      if j == 0:
        inputs = inputs1
        embeddings_tensor = RetrieverAndRanker._create_user_embeddings(
          inputs, loaded_model)
      else:
        inputs = inputs2
        embeddings_tensor = RetrieverAndRanker._create_movie_embeddings(
          inputs, loaded_model)
        
      indexer = RetrieverAndRanker.build_scann_searcher(embeddings_tensor, top_k=2)
      neighbor_idxs, distances = indexer.search_batched(embeddings_tensor, 2)
      #results are both np.ndarrays
      self.assertEquals([0,1], neighbor_idxs[0].tolist())
      self.assertEquals([1,0], neighbor_idxs[1].tolist())
      a = set([i  for _list in neighbor_idxs for i in _list])
      self.assertTrue(0 in a)
      self.assertTrue(1 in a)
    
  def _est_bloom_filters(self):
    
    loaded_model = tf.saved_model.load(self.user_movie_models_dir)
    
    user_indexers, user_ids = RetrieverAndRanker._create_user_indexer(self.user_inputs,
                                                                      loaded_model, self.feature_spec, self.max_k)
    
    ratings_ds = RetrieverAndRanker._joined_ratings_tt_to_ds(self.joined_ratings_tf_records,
      self.feature_spec, batch_size=2048)
    
    shift_bits = 13
    ubf, umbf = RetrieverAndRanker._init_rbloom(user_ids, ratings_ds, bits_shift=shift_bits)
    
    self.assertTrue(1 in ubf)
    
    #known user, movie rating pair
    self.assertTrue((6040 << shift_bits)+858 in umbf)
    
  def test_retrieval(self):
    
    rr = RetrieverAndRanker(user_movie_saved_model_dir = self.user_movie_models_dir,
                            movies_path = self.movie_inputs, users_path=self.user_inputs,
                            ratings_paths = self.joined_ratings_tf_records,
                            movies_pivot_path=self.movies_predictions_pivot,
                            max_k= 1000, movies_batch_size=256, ratings_batch_size=256)
    
    self.assertTrue(rr.is_user_known(1))
    self.assertTrue(rr.has_seen_movie(6040, 858))
    
    #who are the users similar to user_id=1
    user_inp = {'user_id': 5077, 'age':25}
    sim_users = rr.get_users_given_users(user_inp, top_k=9)
    print(f'sim_users: {sim_users}')
    #1587, 2059, 5684, 1859, 4899, 5217, 3468, 2345, 3040
    
    sim_movies = rr.get_movies_given_users(user_inp, top_k=9, remove_aready_seen=True)
    print(f'sim_movies: {sim_movies}')
    #3089, 1572, 3030, 1068, 2731, 326, 1759, 3134, 2575, 2940
    
    movie_inp = {'movie_id': 1068, 'genres': 'Crime|Film-Noir'}
    sim_users = rr.get_users_given_movies(movie_inp, top_k=9)
    print(f'sim_users: {sim_users}')
    
    sim_movies = rr.get_movies_given_movies(movie_inp, top_k=9)
    print(f'sim_movies: {sim_movies}')
    
  if __name__ == '__main__':
    unittest.main()
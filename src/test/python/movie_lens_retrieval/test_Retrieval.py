import sys
from unittest import TestCase
import polars as pl
import tensorflow.saved_model as saved_model
from movie_lens_retrieval.Retrieval import Retrieval
from helper import *
import numpy as np
class TestRetrieval(TestCase):
  
  def test__read_ratings(self):
    file_paths = [os.path.join(get_project_dir(), "src/test/resources/data/ratings_sorted_1_joined*parquet")]
    ratings_pl = Retrieval._read_ratings(file_paths)
    self.assertIsNotNone(ratings_pl)
    n_rows = ratings_pl['movie_id'].count()
    self.assertTrue(n_rows > 100_000)
    self.assertTrue(len(ratings_pl.columns) > 3)
    self.assertTrue("user_id" in ratings_pl.columns)
    self.assertTrue("movie_id" in ratings_pl.columns)
    self.assertTrue("rating" in ratings_pl.columns)
    
    file_paths = [os.path.join(get_project_dir(),
      "src/test/resources/data/ratings_sorted_1_joined*parquet"),
      os.path.join(get_project_dir(),
      "src/test/resources/data/ratings_sorted_2_joined*parquet")]
    ratings_pl = Retrieval._read_ratings(file_paths)
    n_rows = ratings_pl['movie_id'].count()
    self.assertGreater(n_rows, 1_000_000)
    self.assertTrue(len(ratings_pl.columns) > 3)
    self.assertTrue("user_id" in ratings_pl.columns)
    self.assertTrue("movie_id" in ratings_pl.columns)
    self.assertTrue("rating" in ratings_pl.columns)
  
  def test_create_embeddings(self):
    #def _create_user_embeddings(inputs: Union[Dict[str, np.ndarray], List[bytes]]) -> np.ndarray:
    file_paths = [os.path.join(get_project_dir(),
      "src/test/resources/data/ratings_sorted_1_joined*parquet"),
      os.path.join(get_project_dir(),
      "src/test/resources/data/ratings_sorted_2_joined*parquet")]
    ratings_pl = Retrieval._read_ratings(file_paths)
    self.assertGreater(ratings_pl['user_id'].count(), 1_000_000)
    
    inputs_dict_np = Retrieval._polars_to_numpy_dict(ratings_pl.head(1000))
    self.assertIsNotNone(inputs_dict_np)
    self.assertEqual(len(ratings_pl.columns), len(inputs_dict_np.keys()))
    
    serving_dir = os.path.join(get_project_dir(), "src/main/resources/serving_models/user_movie_model")
    user_movie_model = saved_model.load(serving_dir)
    
    user_embeddings = Retrieval._create_user_embeddings(inputs_dict_np, user_movie_model)
    self.assertIsNotNone(user_embeddings)
    self.assertTrue(len(user_embeddings) > 0)
    
    movie_embeddings = Retrieval._create_movie_embeddings(inputs_dict_np, user_movie_model)
    self.assertIsNotNone(movie_embeddings)
    self.assertTrue(len(movie_embeddings) > 0)
    
  def test__create_user_indexers(self):
    file_paths = [os.path.join(get_project_dir(),
      "src/test/resources/data/ratings_sorted_1_joined*parquet")]
    ratings_pl = Retrieval._read_ratings(file_paths)
    inputs_dict_np = Retrieval._polars_to_numpy_dict(ratings_pl.head(1100))
    max_k = 10
    
    serving_dir = os.path.join(get_project_dir(),  "src/main/resources/serving_models/user_movie_model")
    user_movie_model = saved_model.load(serving_dir)
    
    test_inputs_dict_np = Retrieval._polars_to_numpy_dict(ratings_pl.head(2))
    test_embeddings = Retrieval._create_user_embeddings(test_inputs_dict_np, user_movie_model)
    if Retrieval.is_linux:
      #indexes are insert order indexes
      users_indexer = Retrieval._create_user_indexers(inputs_dict_np, user_movie_model, max_k = max_k)
      self.assertIsNotNone(users_indexer)
      neighbor_idxs, distances = users_indexer.search_batched(test_embeddings)
      self.assertTrue(0 in neighbor_idxs[0])
      self.assertTrue(1 in neighbor_idxs[1])
    #can test faiss on all major OSes:
    import faiss
    sav = Retrieval.is_linux
    Retrieval.is_linux = False
    users_indexer = Retrieval._create_user_indexers(inputs_dict_np, user_movie_model, max_k = max_k)
    self.assertIsNotNone(users_indexer)
    distances, neighbor_idxs = users_indexer.search(test_embeddings, max_k)
    self.assertTrue(0 in neighbor_idxs[0])
    self.assertTrue(1 in neighbor_idxs[1])
    Retrieval.is_linux = sav
  
  def test__agg_movie_counts(self):
    file_paths = [os.path.join(get_project_dir(),
      "src/test/resources/data/ratings_sorted_1_joined*parquet")]
    ratings_pl = Retrieval._read_ratings(file_paths)
    file_path = os.path.join(get_project_dir(),
      "src/test/resources/data/movies*parquet")
    movies_pl = pl.read_parquet(file_path, glob=True)
    pivoted = Retrieval._agg_movie_counts(ratings_pl, movies_pl)
    self.assertEqual(len(pivoted.columns), 6)
    for key in ["movie_id", "1", "2", "3", "4", "5"]:
      self.assertTrue(key in pivoted.columns)
    self.assertEqual(movies_pl["movie_id"].count(), pivoted["movie_id"].count())
  
    #test one of the cold start rankings
    max_k = 10
    ranked = Retrieval._prep_cold_start_rankings(ratings_pl, movies_pl, max_k)
    self.assertEqual(len(ranked), max_k)
  
  def test_get_metadata_predictions(self):
    metadata_saved_model_dir = os.path.join(get_project_dir(),
      "src/main/resources/serving_models/metadata_model")
    file_path = os.path.join(get_project_dir(),
      "src/test/resources/data/movies*parquet")
    movies_pl = pl.read_parquet(file_path, glob=True)
    #movie_id, title, genres, predicted_from_genres
    movies = Retrieval._get_metadata_predictions(metadata_saved_model_dir, movies_pl)
    self.assertEqual(len(movies), movies_pl["movie_id"].count())
    self.assertTrue("predicted_from_genres" in movies.columns)
    self.assertEqual(len(movies.columns), len(movies_pl.columns) + 1)
    
    max_k = 10
    file_paths = [os.path.join(get_project_dir(),
      "src/test/resources/data/ratings_sorted_1_joined*parquet"),
      os.path.join(get_project_dir(),
      "src/test/resources/data/ratings_sorted_2_joined*parquet")]
    ratings_pl = Retrieval._read_ratings(file_paths)
    ranked = Retrieval._prep_cold_start_rankings(ratings_pl, movies_pl,
      max_k, "predicted_from_genres")
    self.assertEqual(len(ranked), max_k)
    
  def test__init_rbloom(self):
    pass
 
  def test_get_cold_start_rankings(self):
    pass

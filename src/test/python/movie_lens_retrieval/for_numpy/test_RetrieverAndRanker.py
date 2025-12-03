import random
from unittest import TestCase
import polars as pl
import tensorflow.saved_model as saved_model
from movie_lens_retrieval.for_numpy.RetrieverAndRanker import RetrieverAndRanker
from helper import *
import numpy as np
class TestRetrieval(TestCase):
  
  def test__read_ratings(self):
    file_paths = [os.path.join(get_project_dir(),
      "src/test/resources/data/sorted_1/ratings_sorted_1_joined*parquet")]
    ratings_pl = RetrieverAndRanker._read_ratings(file_paths)
    self.assertIsNotNone(ratings_pl)
    n_rows = ratings_pl['movie_id'].count()
    self.assertTrue(n_rows > 100_000)
    self.assertTrue(len(ratings_pl.columns) > 3)
    self.assertTrue("user_id" in ratings_pl.columns)
    self.assertTrue("movie_id" in ratings_pl.columns)
    self.assertTrue("rating" in ratings_pl.columns)
    
    file_paths = [os.path.join(get_project_dir(),
      "src/test/resources/data/sorted_1/ratings_sorted_1_joined*parquet"),
      os.path.join(get_project_dir(),
      "src/test/resources/data/sorted_2/ratings_sorted_2_joined*parquet")]
    ratings_pl = RetrieverAndRanker._read_ratings(file_paths)
    n_rows = ratings_pl['movie_id'].count()
    self.assertGreater(n_rows, 1_000_000)
    self.assertTrue(len(ratings_pl.columns) > 3)
    self.assertTrue("user_id" in ratings_pl.columns)
    self.assertTrue("movie_id" in ratings_pl.columns)
    self.assertTrue("rating" in ratings_pl.columns)
  
  def test_create_embeddings(self):
    #def _create_user_embeddings(inputs: Union[Dict[str, np.ndarray], List[bytes]]) -> np.ndarray:
    file_paths = [os.path.join(get_project_dir(),
      "src/test/resources/data/sorted_1/ratings_sorted_1_joined*parquet"),
      os.path.join(get_project_dir(),
      "src/test/resources/data/sorted_2/ratings_sorted_2_joined*parquet")]
    ratings_pl = RetrieverAndRanker._read_ratings(file_paths)
    self.assertGreater(ratings_pl['user_id'].count(), 1_000_000)
    
    inputs_dict_np = RetrieverAndRanker._polars_to_numpy_dict(ratings_pl.head(1000))
    self.assertIsNotNone(inputs_dict_np)
    self.assertEqual(len(ratings_pl.columns), len(inputs_dict_np.keys()))
    
    user_movie_saved_model_dir = os.path.join(get_project_dir(), "src/main/resources/serving_models/user_movie_model")
    user_movie_model = saved_model.load(user_movie_saved_model_dir)
    
    user_embeddings = RetrieverAndRanker._create_user_embeddings(inputs_dict_np, user_movie_model)
    self.assertIsNotNone(user_embeddings)
    self.assertTrue(len(user_embeddings) > 0)
    
    movie_embeddings = RetrieverAndRanker._create_movie_embeddings(inputs_dict_np, user_movie_model)
    self.assertIsNotNone(movie_embeddings)
    self.assertTrue(len(movie_embeddings) > 0)
    
  def test__create_user_indexers(self):
    file_paths = [os.path.join(get_project_dir(),
      "src/test/resources/data/sorted_1/ratings_sorted_1_joined*parquet")]
    ratings_pl = RetrieverAndRanker._read_ratings(file_paths)
    inputs_dict_np = RetrieverAndRanker._polars_to_numpy_dict(ratings_pl.head(1100))
    max_k = 10
    
    serving_dir = os.path.join(get_project_dir(),  "src/main/resources/serving_models/user_movie_model")
    user_movie_model = saved_model.load(serving_dir)
    
    test_inputs_dict_np = RetrieverAndRanker._polars_to_numpy_dict(ratings_pl.head(2))
    test_embeddings = RetrieverAndRanker._create_user_embeddings(test_inputs_dict_np, user_movie_model)
    if RetrieverAndRanker.is_linux:
      #indexes are insert order indexes
      users_indexer = RetrieverAndRanker._create_user_indexers(inputs_dict_np, user_movie_model, max_k = max_k)
      self.assertIsNotNone(users_indexer)
      neighbor_idxs, distances = users_indexer.search_batched(test_embeddings)
      self.assertTrue(0 in neighbor_idxs[0])
      self.assertTrue(1 in neighbor_idxs[1])
    #can test faiss on all major OSes:
    sav = RetrieverAndRanker.is_linux
    RetrieverAndRanker.is_linux = False
    users_indexer = RetrieverAndRanker._create_user_indexers(inputs_dict_np, user_movie_model, max_k = max_k)
    self.assertIsNotNone(users_indexer)
    distances, neighbor_idxs = users_indexer.search(test_embeddings, max_k)
    self.assertTrue(0 in neighbor_idxs[0])
    self.assertTrue(1 in neighbor_idxs[1])
    RetrieverAndRanker.is_linux = sav
  
  def test__agg_movie_counts(self):
    file_paths = [os.path.join(get_project_dir(),
      "src/test/resources/data/sorted_1/ratings_sorted_1_joined*parquet")]
    ratings_pl = RetrieverAndRanker._read_ratings(file_paths)
    file_path = os.path.join(get_project_dir(),
      "src/test/resources/data/movies/movies*parquet")
    movies_pl = pl.read_parquet(file_path, glob=True)
    pivoted = RetrieverAndRanker._agg_movie_counts(ratings_pl, movies_pl)
    self.assertEqual(len(pivoted.columns), 6)
    for key in ["movie_id", "1", "2", "3", "4", "5"]:
      self.assertTrue(key in pivoted.columns)
    self.assertEqual(movies_pl["movie_id"].count(), pivoted["movie_id"].count())
  
    #test one of the cold start rankings
    max_k = 10
    ranked = RetrieverAndRanker._prep_cold_start_rankings(ratings_pl, movies_pl, max_k)
    self.assertEqual(len(ranked), max_k)
  
  def test_get_metadata_predictions(self):
    #for ranking
    metadata_saved_model_dir = os.path.join(get_project_dir(),
      "src/main/resources/serving_models/metadata_model")
    file_path = os.path.join(get_project_dir(),
      "src/test/resources/data/movies/movies*parquet")
    movies_pl = pl.read_parquet(file_path, glob=True)
    #movie_id, title, genres, predicted_from_genres
    movies = RetrieverAndRanker._get_metadata_predictions(metadata_saved_model_dir, movies_pl)
    self.assertEqual(len(movies), movies_pl["movie_id"].count())
    self.assertTrue("predicted_from_genres" in movies.columns)
    self.assertEqual(len(movies.columns), len(movies_pl.columns) + 1)
    
    max_k = 10
    file_paths = [os.path.join(get_project_dir(),
      "src/test/resources/data/sorted_1/ratings_sorted_1_joined*parquet"),
      os.path.join(get_project_dir(),
      "src/test/resources/data/sorted_2/ratings_sorted_2_joined*parquet")]
    ratings_pl = RetrieverAndRanker._read_ratings(file_paths)
    ranked = RetrieverAndRanker._prep_cold_start_rankings(ratings_pl, movies_pl,
                                                          max_k, "predicted_from_genres")
    self.assertEqual(len(ranked), max_k)
    
  def test__init_rbloom(self):
    shift_bits = 13
    file_paths = [os.path.join(get_project_dir(),
      "src/test/resources/data/sorted_1/sorted_1/ratings_sorted_1_joined*parquet"),
      os.path.join(get_project_dir(),
      "src/test/resources/data/sorted_2/sorted_2/ratings_sorted_2_joined*parquet")]
    ratings_pl_1 = RetrieverAndRanker._read_ratings([file_paths[0]])
    #ratings_pl_2 = Retrieval._read_ratings([file_paths[1]])
    
    max_user_id = ratings_pl_1["user_id"].max()
    
    user_bloom_filter, user_movie_bloom_filter = RetrieverAndRanker._init_rbloom(ratings_pl_1, shift_bits)
    
    #test that all out-of-vocabulary user_ids are not in bloom filter
    n_false_u = 0
    for _ in range(100_000):
      user_id = random.randint(max_user_id+1, max_user_id+10_000)
      if user_id in user_bloom_filter:
        n_false_u += 1
    fpr = n_false_u/100_000
    print(f'false positive rate for user bloom filter = {fpr}, expected = 0.01')
    self.assertLessEqual(fpr, 0.01)
    
    """
      GT       True   False
    PRED  TRUE  TP     FP
          FALSE FN     TN
                T      F
    """
    
    #TODO: test the errors for false positives of users and user_movie combinations for those that are in ratings_pl_1
    user_idx = ratings_pl_1.columns.index("user_id")
    movie_idx = ratings_pl_1.columns.index("movie_id")
    n = ratings_pl_1.shape[0]
    n_false_um = 0
    n_false_u = 0
    for _ in range(100_000):
      row = ratings_pl_1.row(np.random.randint(1, n))
      user_id = row[user_idx]
      movie_id = row[movie_idx]
      encoded = (user_id << shift_bits) + movie_id
      if encoded not in user_movie_bloom_filter:
        n_false_um += 1
      if user_id not in user_bloom_filter:
        n_false_u += 1
        
    fnr = n_false_u / 100_000
    fnr_um = n_false_um / 100_000
    #um 0.001
    #u 0.01
    print(f'false negative rate for user-movie bloom filter = {fnr_um}')
    print( f'false negative rate for user bloom filter = {fnr}')
    self.assertLessEqual(fnr, 0.01)
    self.assertLessEqual(fnr_um, 0.001)
 
  ##  ===================== INSTANCE METHODS ===========
  
  def test_get_cold_start_rankings(self):
    user_movie_saved_model_dir = os.path.join(get_project_dir(), "src/main/resources/serving_models/user_movie_model")
    metadata_saved_model_dir = os.path.join(get_project_dir(),
      "src/main/resources/serving_models/metadata_model")
    movies_path = os.path.join(get_project_dir(),
      "src/test/resources/data/movies/movies*parquet")
    ratings_paths = [os.path.join(get_project_dir(),
      "src/test/resources/data/sorted_1/ratings_sorted_1_joined*parquet"),
      os.path.join(get_project_dir(),
      "src/test/resources/data/sorted_2/ratings_sorted_2_joined*parquet")]
    max_k = 1_000
    
    r = RetrieverAndRanker(user_movie_saved_model_dir=user_movie_saved_model_dir,
      metadata_saved_model_dir=metadata_saved_model_dir,
      movies_path=movies_path, ratings_paths=ratings_paths, max_k=max_k)
    
    recommendations = r.get_cold_start_rankings(top_k=10)
    
    movies_pl = pl.read_parquet(movies_path, glob=True)
    
    recommended_movies = movies_pl.filter(pl.col('movie_id').is_in(recommendations['movie_id']))
    print(f'cold start movies = {recommended_movies}')
    
    pass

import os.path
import unittest

from helper import *
from movie_lens_retrieval.RetrieverAndRanker import RetrieverAndRanker

class TestRetrieverAndRanker(unittest.TestCase):
  def setUp(self):
    
    saved_models_dir = os.path.join(get_project_dir(), "src/main/resources/serving_models")
    self.user_movie_models_dir = os.path.join(saved_models_dir, "user_movie_model")
    
    self.movie_inputs = os.path.join(get_project_dir(),
      "src/test/resources/data/movie_emb_inp/tfrecord*.gz")
    self.user_inputs = os.path.join(get_project_dir(),
      "src/test/resources/data/user_emb_inp/tfrecord*.gz")
    self.movies_mean_ratings_pivot = os.path.join(get_project_dir(),
      "src/test/resources/data/ratings_and_predictions_pivot/mean_ratings_tfrecord*.gz")
    self.movies_predictions_pivot = os.path.join(get_project_dir(),
      "src/test/resources/data/ratings_and_predictions_pivot/mm_predictions_tfrecord*.gz")
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
      
  def test_indexer_tensors(self):
   
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
      
  def test_retrieval(self):
    
    rr = RetrieverAndRanker(user_movie_saved_model_dir = self.user_movie_models_dir,
                            movies_path = self.movie_inputs, users_path=self.user_inputs,
                            movies_pivot_path=self.movies_mean_ratings_pivot,
                            max_k= 1000, movies_batch_size=256)
    
    #who are the users similar to user_id=
    user_inp = {'user_id': 5077, 'age':25}
    sim_users = rr.get_users_given_users(user_inp, top_k=9)
    print(f'sim_users: {sim_users}')
    #1587, 2059, 5684, 1859, 4899, 5217, 3468, 2345, 3040
    
    sim_movies = rr.get_movies_given_users(user_inp, top_k=9)
    print(f'sim_movies: {sim_movies}')
    #3089, 1572, 3030, 1068, 2731, 326, 1759, 3134, 2575, 2940
    
    #test that age is retrieved when missing from inouts
    user_inp = [{'user_id': 5077}, {'user_id': 1}]
    sim_users = rr.get_users_given_users(user_inp, top_k=9)
    print(f'sim_users: {sim_users}')
    try:
      user_inp = [{'user_id': 1_000_000}]
      sim_users = rr.get_users_given_users(user_inp, top_k=9)
      self.fail("Should have thrown a ValueError")
    except ValueError:
      pass
    
    movie_inp = {'movie_id': 1068, 'genres': 'Crime|Film-Noir'}
    sim_users = rr.get_users_given_movies(movie_inp, top_k=9)
    print(f'sim_users: {sim_users}')
    
    movie_inp = [{'movie_id': 1068}, {'movie_id': 1}]
    sim_users = rr.get_users_given_movies(movie_inp, top_k=9)
    print(f'sim_users: {sim_users}')
    
    try:
      movie_inp = {'movie_id': 1_000_000}
      sim_users = rr.get_users_given_movies(movie_inp, top_k=9)
      self.fail("Should have thrown a ValueError")
    except ValueError:
      pass
    
    movie_inp = [{'movie_id': 1068}, {'movie_id': 1}]
    sim_movies = rr.get_movies_given_movies(movie_inp, top_k=9)
    print(f'sim_movies: {sim_movies}')
    
    cold_starts = rr.get_cold_start_movie_recommendations(10)
    print(f'cold_starts: {cold_starts}')
    
    print(f'is_user_known(1_000_000)={rr.user_is_known(1_000_000)}')
    print(f'is_user_known(1)={rr.user_is_known(1)}')
    
    #use test data to check recommendations.  these are movies the user loved.
    # the returned ratings shuld be high
    user_inp = {'user_id': 635, 'age': 56,
      'movie_id': [1704, 1940], 'genres': ['Drama', 'Drama']}
    preds = rr.get_predictions(user_inp)
    print(f'predictions: {preds}')
    
  def test_eval_single_genre(self):
    '''
    evaluate the test users who have only rated movies that have a single genre and that those movies they've rated
    all have the same single genre.
    '''
    import polars as pl
    
    rr = RetrieverAndRanker(
      user_movie_saved_model_dir=self.user_movie_models_dir,
      movies_path=self.movie_inputs, users_path=self.user_inputs,
      movies_pivot_path=self.movies_mean_ratings_pivot,
      max_k=1000, movies_batch_size=256)
    
    test_users_df = pl.read_parquet(os.path.join(get_project_dir(),
      "src/test/resources/data/single_genre/users_single_genre.parquet"))
    test_users_df = test_users_df.drop('zipcode')
    # columns=['user_id', 'gender', 'age', 'occupation', 'zipcode']
    
    ratings_seen_df = pl.read_parquet(os.path.join(get_project_dir(),
      "src/test/resources/data/sorted_1/ratings_sorted_1_joined-*.parquet"))
    ratings_seen_df = ratings_seen_df.filter(pl.col('user_id').is_in(test_users_df['user_id'].implode()))
    print(f'ratings seen {ratings_seen_df.count()}')
    
    ratings_unseen_df = pl.read_parquet(os.path.join(get_project_dir(),
      "src/test/resources/data/sorted_2/ratings_sorted_2_joined-*.parquet"))
    ratings_unseen_df = ratings_unseen_df.filter(
      pl.col('user_id').is_in(test_users_df['user_id'].implode()))
    print(f'ratings unseen {ratings_unseen_df.count()}')
    
    res_hg  = {}
    res_ndcg = {}
    res_mrr = {}
    res_recall = {}
    users_inp  = test_users_df.to_dicts()
    for k in [20, 50, 100, 200]:
      sim_movies = rr.get_movies_given_users(users_inp, top_k=k)
      for i in range(len(sim_movies)):
        recommended = set(sim_movies[i])
        user_inp = users_inp[i]
        seen = (
          ratings_seen_df.filter(pl.col('user_id') == user_inp['user_id'])
          .select('movie_id').to_series().to_list()
        )
        unseen = (
          ratings_unseen_df.filter(
            pl.col('user_id') == user_inp['user_id'])
            .select(['movie_id', 'rating'])
        )
        n_seen = len(seen)
        n_unseen = unseen['movie_id'].count()
        #predict the scores.  this can be done better in the re-ranker
        recommended = list(recommended - set(seen))
        inp = {**user_inp}
        inp['movie_id'] = recommended
        inp['genres'] = rr.movie_genres_ht.lookup(tf.constant(recommended, dtype=tf.int64)).numpy().tolist()
        preds = rr.get_predictions(inp)
        sorted_comb = sorted(zip(preds, recommended))
        sorted_ratings, sorted_movies = zip(*sorted_comb)
        
        #TODO: finish eval stats
        
  if __name__ == '__main__':
    unittest.main()
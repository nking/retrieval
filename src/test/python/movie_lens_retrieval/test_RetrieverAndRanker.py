import collections
import os.path
import unittest
import glob
from typing import Tuple
from scipy.stats import hypergeom, combine_pvalues
from sklearn.metrics import ndcg_score, average_precision_score
import numpy as np

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
    
    rating_limit = 4
    
    rr = RetrieverAndRanker(
      user_movie_saved_model_dir=self.user_movie_models_dir,
      movies_path=self.movie_inputs, users_path=self.user_inputs,
      movies_pivot_path=self.movies_mean_ratings_pivot,
      max_k=1000, movies_batch_size=256)
    
    test_users_df = pl.read_parquet(os.path.join(get_project_dir(),
      "src/test/resources/data/single_genre/users_single_genre.parquet"))
    test_users_df = test_users_df.drop('zipcode')
    # columns=['user_id', 'gender', 'age', 'occupation', 'zipcode']
    n_users = test_users_df['user_id'].count()
    
    #train dataset:
    ratings_seen_df = pl.read_parquet(os.path.join(get_project_dir(),
      "src/test/resources/data/sorted_1/ratings_sorted_1_joined-*.parquet"))
    ratings_seen_df = ratings_seen_df.filter(pl.col('user_id').is_in(test_users_df['user_id'].implode()))
    print(f'ratings seen {ratings_seen_df.count()}')
    
    #test dataset:
    ratings_unseen_df = pl.read_parquet(os.path.join(get_project_dir(),
      "src/test/resources/data/sorted_2/ratings_sorted_2_joined-*.parquet"))
    ratings_unseen_df = ratings_unseen_df.filter(
      pl.col('user_id').is_in(test_users_df['user_id'].implode()))
    print(f'ratings unseen {ratings_unseen_df.count()}')
    
    #dict of (key=genre, value=list of movie_ids), N = total number of movies
    g_m_ht, N = self.read_movies_file_into_genre_dict()
    
    #NOTE: below here are evaluation metrics that are best performed on the ranked, and re-ranked
    # recommendations, but we start with retrieval evals first:
    
    res_hg  = collections.defaultdict(list)
    res_ndcg = {}
    res_mrr = {}
    res_recall = {}
    hit_rate = collections.defaultdict(float)
    mean_ap = collections.defaultdict(float)
    users_inp  = test_users_df.to_dicts()
    for top_k in [20, 50, 100, 200]:
      sim_movies = rr.get_movies_given_users(users_inp, top_k=top_k)
      #the sim_movies are lists returned in same order of list of input users
      for i in range(len(sim_movies)):
        user_inp = users_inp[i]
        seen = (
          ratings_seen_df.filter(pl.col('user_id') == user_inp['user_id'])
          .select('movie_id').to_series().to_list()
        )
        test_data = (
          ratings_unseen_df.filter(
            pl.col('user_id') == user_inp['user_id'])
            .select(['movie_id', 'rating'])
        )
        genre = ratings_seen_df.filter(
          pl.col('user_id') == user_inp['user_id'],
          pl.col('rating') > rating_limit)
        if genre.is_empty():
          genre = ratings_unseen_df.filter(
            pl.col('user_id') == user_inp['user_id'],
            pl.col('rating') > rating_limit)
        print(genre)
        genre = genre.head(1)['genres'].to_numpy()[0]
        
        test_data_liked = test_data.filter(pl.col('rating') > 3)
        n_seen = len(seen)
        n_test = test_data['movie_id'].count()
        recommended = set(sim_movies[i])
        recommended = list(recommended - set(seen))
        inp = {**user_inp}
        inp['movie_id'] = recommended
        inp['genres'] = rr.movie_genres_ht.lookup(tf.constant(recommended, dtype=tf.int64)).numpy().tolist()
        preds = rr.get_predictions(inp)
        
        sorted_comb = sorted(zip(preds, recommended))
        sorted_ratings, sorted_movies = zip(*sorted_comb)
        
        ## === enrichment analysis ===
        
        # M = total number of movies in entire db minus already seen
        M = N - n_seen
        # n_successes is K_genre = total number of movies in db belonging to the user's single genre, excluding n_seen
        n_successes = len( set(g_m_ht[genre]) - set(seen))
        # N_draws  = number of recommendations generated for the user (top-k)
        N_draws = len(recommended)
        # k_observed = number of movies in the top-k recommendations that belong to that specific genre
        k_observed = inp['genres'].count(genre)
        p_value = hypergeom.sf(k_observed - 1, M, n_successes, N_draws)
        # high p_value when embedding finds good recommendations
        # low p_value suggests randomly choosing recommendations
        res_hg[top_k].append(p_value)
        
        k = 0
        ranks = [] # a list of positions of the k movies. e.g., if the 1st and 3rd recs were the right genre, ranks = [1, 3]
        y_genre = []
        y_pred = []
        for i, movie_id in enumerate(sorted_movies):
          g = rr.movie_genres_ht.lookup(movie_id)
          if g == genre:
            k += 1
            ranks.append(i+1)
            y_genre.append(1)
          else:
            y_genre.append(0)
          y_pred.append(1)
        recall_at_n = k / K
        mrr = 1.0/min(ranks) if len(ranks) else 0.0
        ndcg_at_n = ndcg_score([y_genre], [y_pred], k=len(y_genre))
        # ===== Learning to Rank evaluation =====
        # from perspective of test data acquired after train data
        #Hit Rate: at least one of the recommended movies contains at least one of the test movies
        test_in_recommended = test_data_liked.filter(pl.col('movie_id').is_in(recommended))
        if test_in_recommended['movie_id'].count() > 0:
          hit_rate[top_k] += 1.
        #TODO: add Expected Reciprocal Rank (ERR) from ranx
        #Mean Average Precision (MAP)
        # ground truth is test_data_liked
        # generate predictions for them and call them predicted_Scores
        # negatives: find the movies that are in recommendations and not in test_data_liked and give those value 0
        # y_true has all of test_data_liked then appends as 1's then appends negatives as 0's
        # y_score has the prediction scores for items in y_true
        inp2 = {**user_inp}
        inp2['movie_id'] = test_data_liked['movie_id'].to_numpy().tolist()
        inp2['genres'] = rr.movie_genres_ht.lookup(
          tf.constant(inp2['movie_id'], dtype=tf.int64)).numpy().tolist()
        preds2 = rr.get_predictions(inp2)
        y_scores = preds2.copy()
        y_true_binary = [1]*len(y_scores)
        #append recommendations that are not in test_data_liked
        for i in range(len(preds)):
          if not test_data_liked.filter(pl.col("movie_id") == recommended[i]).is_empty():
            y_true_binary.append(0)
            y_scores.append(preds[i])
        mean_ap[top_k] += average_precision_score(y_true_binary, y_scores)
      hit_rate[top_k] /= n_users
      mean_ap[top_k] /= n_users
    
    for top_k in res_hg.keys():
      statistic, global_p_value = combine_pvalues(res_hg[top_key], method='fisher')
      
    # TODO: for each stat overplot curves of stat vs top_k
      
  def read_movies_file_into_genre_dict(self, filter_for_single:bool=True) -> Tuple[collections.defaultdict(list), int]:
    _ct = "GZIP" if self.movie_inputs.endswith(".gz") else None
    file_paths = glob.glob(self.movie_inputs)
    ds_ser = tf.data.TFRecordDataset(file_paths, compression_type=_ct)
    feature_spec = {
      "movie_id": tf.io.FixedLenFeature(shape=[], dtype=tf.int64,
        default_value=None),
      "genres": tf.io.FixedLenFeature(shape=[], dtype=tf.string,
        default_value=None)}
    def parse_tf_example(example_proto, feature_spec):
      return tf.io.parse_single_example(example_proto, feature_spec)
    ds = ds_ser.map(lambda x: parse_tf_example(x, feature_spec))
    #dict with key=genre, value=movie_id
    genre_to_ids = collections.defaultdict(list)
    n_movies = 0
    for x in ds.as_numpy_iterator():
      n_movies += 1
      if filter_for_single:
        if x['genres'].find(b'|')>-1:
          continue
      genre_to_ids[x['genres']].append(x['movie_id'])
    return genre_to_ids, n_movies
    
  if __name__ == '__main__':
    unittest.main()
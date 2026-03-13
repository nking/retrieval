import collections
import os.path
import unittest
import glob
from typing import Tuple
from scipy.stats import hypergeom, combine_pvalues
from sklearn.metrics import ndcg_score, average_precision_score
import numpy as np
import plotly.express as px  #needs kaleido to write pngs
from plotly.subplots import make_subplots

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
      self.assertEqual([0,1], neighbor_idxs[0].tolist())
      self.assertEqual([1,0], neighbor_idxs[1].tolist())
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
    evaluate the test users who were derived from the first ratings partition filtered for rating > 5
    and when grouped by user, each further filtered user had a movie list with 1 unique genre.
    The resulting number of unique users is small, 19, but they have characteristics that are easier to
    predict for an enrichment evaluation using the hypergeometric survival function.
    The goal is to understand whether the embeddings find good recommendations for these easy to understand
    test users.
    
    standard Learning To Rank (LTR) information retrieval metrics are also calculated.
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
    print(f'test_users_df.count: {test_users_df.count()}')
    
    #train dataset:
    ratings_seen_df = pl.read_parquet(os.path.join(get_project_dir(),
      "src/test/resources/data/sorted_1/ratings_sorted_1_joined-*.parquet"))
    n_ratings_train = ratings_seen_df.shape[0]
    n_unique_genres_comb_train = ratings_seen_df['genres'].unique().shape[0]
    ratings_seen_df = ratings_seen_df.filter(pl.col('user_id').is_in(test_users_df['user_id'].implode()))
    print(f'ratings seen {ratings_seen_df.count()}')
    
    #test dataset:
    ratings_unseen_df = pl.read_parquet(os.path.join(get_project_dir(),
      "src/test/resources/data/sorted_2/ratings_sorted_2_joined-*.parquet"))
    n_unique_genres_comb_test = ratings_unseen_df['genres'].unique().shape[0]
    n_ratings_test = ratings_unseen_df.shape[0]
    ratings_unseen_df = ratings_unseen_df.filter(
      pl.col('user_id').is_in(test_users_df['user_id'].implode()))
    print(f'ratings unseen {ratings_unseen_df.count()}')
    print(f'number of unique genre combinations in train and test are = '
          f'{n_unique_genres_comb_train}, {n_unique_genres_comb_test} respectively '
          f'for numbers of ratings = {n_ratings_train}, {n_ratings_test}')
    
    #dict of (key=genre, value=list of movie_ids), N = total number of movies
    g_m_ht, N = self.read_movies_file_into_genre_dict()
    
    #NOTE: below here are evaluation metrics that are best performed on the ranked, and re-ranked
    # recommendations, but we start with retrieval evals first:
    
    res_hg  = collections.defaultdict(list)
    results_user_hg = collections.defaultdict(list)
    res_ndcg = collections.defaultdict(list)
    res_mrr = collections.defaultdict(list)
    res_mrr_d = collections.defaultdict(list)
    res_recall = collections.defaultdict(list)
    res_hit_rate = collections.defaultdict(float)
    res_avoidance_hit_rate = collections.defaultdict(float)
    res_mean_ap_per_user = collections.defaultdict(list)
    res_mean_ap = collections.defaultdict(float)
    res_rec_frac_of_neg_per_user = collections.defaultdict(list)
    res_rec_frac_of_neg = collections.defaultdict(float)

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
        if test_data.is_empty():
          continue
        genre = ratings_seen_df.filter(
          pl.col('user_id') == user_inp['user_id'],
          pl.col('rating') > rating_limit).head(1)['genres'].to_numpy()[0]
        genre = genre.encode('utf-8')
        print(f'user_id={user_inp["user_id"]}, genre={genre}, n_test={test_data.height}')
        
        #the users were derived from ratings > 4
        test_data_liked = test_data.filter(pl.col('rating') > 3)
        test_data_disliked = test_data.filter(pl.col('rating') < 3)
        
        recommended = list(set(sim_movies[i]) - set(seen))
        inp = {**user_inp}
        inp['movie_id'] = recommended
        inp['genres'] = rr.movie_genres_ht.lookup(tf.constant(recommended, dtype=tf.int64)).numpy().tolist()
        preds = rr.get_predictions(inp)
        
        ## === enrichment analysis ===
        
        # M = total number of movies in entire db minus already seen
        M = N - len(seen)
        # n_successes is K_genre = total number of movies in db belonging to the user's single genre, excluding n_seen
        n_successes = len(g_m_ht[genre]) - len(seen)
        # N_draws  = number of recommendations generated for the user (top-k)
        N_draws = len(recommended)
        # k_observed = number of movies in the top-k recommendations that belong to that specific genre
        k_observed = inp['genres'].count(genre)
        p_value = hypergeom.sf(k_observed - 1, M, n_successes, N_draws)
        # high p_value when embedding finds good recommendations
        # low p_value suggests randomly choosing recommendations
        res_hg[top_k].append(p_value)
        # store by user too to more easily see user based probabilities
        results_user_hg[user_inp['user_id']].append(p_value)
        
        ## ======= information retrieval metrics =====
        sorted_comb = sorted(zip(preds, recommended))
        sorted_ratings, sorted_recommended_movies = zip(*sorted_comb)
        
        k = 0
        ranks = [] # a list of positions of the k movies. e.g., if the 1st and 3rd recs were the right genre, ranks = [1, 3]
        negative_ranks = [] # rank of any recommendation that the user rated 1 or 2
        y_genre = [] # 1's for recommened and of expected genre, 0's for recommended and not of expected genre
        y_pred = [] # 1's for recommended movies
        for i, movie_id in enumerate(sorted_recommended_movies):
          g = rr.movie_genres_ht.lookup(tf.constant(movie_id, dtype=tf.int64))
          #if genre == g:  # loosened to count if any genre in list matches genre
          if tf.strings.regex_full_match(g, f".*{genre.decode()}.*").numpy():
            k += 1
            ranks.append(i+1)
            y_genre.append(1)
          else:
            y_genre.append(0)
          y_pred.append(1)
          if test_data_liked.select(pl.col('movie_id').is_in([movie_id]).any()).item() > 0:
            negative_ranks.append(i+1)
        res_recall[top_k].append(k / N_draws) #TP/len(ground_truth_positives)
        res_mrr[top_k].append(1.0/min(ranks) if len(ranks) else 0.0)
        res_mrr_d[top_k].append(1.0 / min(negative_ranks) if len(negative_ranks) else 0.0)
        res_ndcg[top_k].append(ndcg_score([y_genre], [y_pred], k=len(y_genre)))
        res_rec_frac_of_neg_per_user[top_k].append(len(negative_ranks)/len(y_pred))
        #NOTE: can use negative ranking to calculate a Rank-Biased Toxicity / Penalty
        
        # ===== Learning to Rank evaluation =====
        # from perspective of test data acquired after train data
        #Hit Rate: at least one of the recommended movies contains at least one of the test movies
        if test_data_liked.select(pl.col('movie_id').is_in(recommended).sum()).item() > 0:
          res_hit_rate[top_k] += 1.
        if test_data_disliked.select(pl.col('movie_id').is_in(recommended).sum()).item() > 0:
          res_avoidance_hit_rate[top_k] += 1
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
        res_mean_ap_per_user[top_k].append(average_precision_score(y_true_binary, y_scores))
        
      res_hit_rate[top_k] /= len(results_user_hg) # denom is number of users
      res_mean_ap[top_k] = np.mean(res_mean_ap_per_user[top_k])
      res_rec_frac_of_neg[top_k] = np.mean(res_rec_frac_of_neg_per_user[top_k])
    
    # NOTE: to compare models, use the means over users for these plots, overl plotting model A, B, C values to find which
    # has highest MAP with lowest rec fract negatives
    fig = make_subplots(rows=2, cols=2, subplot_titles=("TopK=20", "TopK=50", "TopK=100", "TopK=200"),
      x_title="MAP", y_title="% recs with negs")
    for i, top_k in enumerate([20,50, 100,200]):
      fig_i = px.scatter(x=res_mean_ap_per_user[top_k], y=res_rec_frac_of_neg_per_user[top_k])
      fig_i.update_layout(title=f'k={top_k}', xaxis_title='MAP', yaxis_title='% recs with negs')
      row_idx = i // 2
      col_idx = i % 2
      for trace in fig_i.data:
        fig.add_trace(trace, row=row_idx+1, col=col_idx+1)
    fig.write_image(os.path.join(get_bin_dir(), "map_vs_rec_negs.png"))
    #fig.show()
    del fig
    
    for top_k in res_hit_rate.keys():
      print(f'top_k={top_k}, hit_rates={res_hit_rate[top_k]}')
    for top_k in res_avoidance_hit_rate.keys():
      print(f'top_k={top_k}, avoidance_hit_rates={res_avoidance_hit_rate[top_k]}')
    for top_k in res_mean_ap.keys():
      print(f'top_k={top_k}, mean_ap={res_mean_ap[top_k]}')
    for top_k in res_rec_frac_of_neg.keys():
      print(f'top_k={top_k}, fraction of negatives in recommendations={res_rec_frac_of_neg[top_k]}')
    for top_k in res_ndcg.keys():
      print(f'top_k={top_k}, NDCG@K={res_ndcg[top_k]}')
    for top_k in res_mrr.keys():
      print(f'top_k={top_k}, MRR@K (1 is best)={res_mrr[top_k]}')
    for top_k in res_mrr_d.keys():
      print(f'top_k={top_k}, MRRD@K (dislikes in the recs. 0 is best)={res_mrr_d[top_k]}')
    for top_k in res_recall.keys():
      print(f'top_k={top_k}, recall@K={res_recall[top_k]}')
    
    for top_k in res_hg.keys():
      statistic, global_p_value = combine_pvalues(res_hg[top_k], method='fisher')
      print(f'top_k={top_k}, stat={statistic}, global hypergeom.sf p_value={global_p_value}')
  
  def test_eval_all(self):
    '''
    calculate and visualize for all test users the
    standard Learning To Rank (LTR) information retrieval metrics.
    '''
    pass
      
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
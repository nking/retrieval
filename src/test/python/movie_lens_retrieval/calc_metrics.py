import collections
import os.path
import unittest
import glob
from collections import defaultdict
from typing import Any, Dict, Union

import polars as pl
import numpy as np
import plotly.express as px  # needs kaleido to write pngs
from plotly.subplots import make_subplots

import msgpack
from array_record.python import array_record_module

from helper import *
from rich import print as rprint

from movie_lens_retrieval.Retriever import Retriever, EmbeddingType

'''
===========
Retrieval
-----------
ground truth:
- for each user in test dataset, make list of movies they rated > 3.
retrieval baseline:
- for each user:
  - score all movies in the catalog (all 3883)
  - take the top 20
  - calc ndcg@20 and other metrics

------------
Ranker Lift
------------
- for each user in test dataset
  - use Retriever to get top 200 candidates
  - use the Ranker to rank (sort) them 
  - take top 20
  - calc ndcg@20 and other metrics
   
------------
ReRanker
------------
- for each user in test dataset
  - use system through Ranker to get top 50
  - use the Re-Ranker to re-rank them 
  - take top 20
  - calc ndcg@20 and other metrics

-------------------
for these metrics, I'll use Polars for now, but consider scalable alternatives in future.
  - some future alternatives:
      - beam w/ custom pyfunc components
      - polars + fugue + pspark

data
  user_movie models uses tfrecords
  ranker model uses array_record
  reranker model uses parquet

'''

class CalcMetrics(unittest.TestCase):
    def setUp(self):
        saved_models_dir = os.path.join(get_project_dir(),
            "src/main/resources/serving_models")
        self.user_movie_models_dir = os.path.join(saved_models_dir,
            "user_movie_model")
        
        self.cold_start_path = os.path.join(get_project_dir(), "src/test/resources/data/cold_start_movies.txt")
        
        self.embed_dim = 16 #though this could be read and parsed from a single entry in embeddings
        self.movie_emb = os.path.join(get_project_dir(),
            "src/test/resources/data/movie_emb_inp/*tfrecord*.gz")
        self.user_emb = os.path.join(get_project_dir(),
            "src/test/resources/data/user_emb_inp/*tfrecord*.gz")
        
        self.users_path = os.path.join(get_project_dir(),
            "src/test/resources/data/users/users.parquet")
        self.movies_path = os.path.join(get_project_dir(),
            "src/test/resources/data/movies/movies.parquet")
        
        self.user_movie_hist_path_patterns = [os.path.join(get_project_dir(), "src/test/resources/data/ratings_train/ratings_train.array_record"),
            os.path.join(get_project_dir(),
                "src/test/resources/data/ratings_val/ratings_val.array_record")
            ]
        self.n_movies = 3883
        self.MOVIE_OFFSET = 6040 + 1
        
    def _construct_Retrieval(self) -> Retriever:
        return Retriever(user_movie_saved_model_dir=self.user_movie_models_dir,
            movie_id_offset = self.MOVIE_OFFSET,
            user_embed_path=self.user_emb,
            movie_embed_path=self.movie_emb,
            embed_dim=self.embed_dim,
            cold_start_movie_path=self.cold_start_path,
            users_path=self.users_path,
            movies_path=self.movies_path,
            user_movie_hist_path_patterns=self.user_movie_hist_path_patterns,
            max_k=self.n_movies)
        
    def _read_tfrecords(self, file_pattern:str, feature_spec:Dict[str, Any],
            return_as_ds:bool=False) -> Union[pl.DataFrame, tf.data.Dataset]:
        file_paths = glob.glob(file_pattern)
        if len(file_paths) == 0:
            raise FileNotFoundError(f'could not find {file_pattern}')
        _ct = "GZIP" if file_paths[0].endswith(".gz") else None
        ds_ser = tf.data.TFRecordDataset(file_paths, compression_type=_ct)
        def parse_tf_example(example_proto, fs):
            return tf.io.parse_single_example(example_proto, fs)
        ds = ds_ser.map(lambda x: parse_tf_example(x, feature_spec))
        if return_as_ds:
            return ds
        records = [{k: v.numpy() for k, v in record.items()} for record in ds]
        return pl.DataFrame(records)
    
    def _read_users_tfrecords(self, file_path:str) -> pl.DataFrame:
        feature_spec = {"movie_id": tf.io.FixedLenFeature([], tf.int64),
            "gender": tf.io.FixedLenFeature([], tf.string),
            "age": tf.io.FixedLenFeature([], tf.int64),
            "occupation": tf.io.FixedLenFeature([], tf.int64),
            } #not parsing zipcode
        return self._read_tfrecords(file_path, feature_spec)
    
    def _read_movies_tfrecords(self, file_path:str) -> pl.DataFrame:
        feature_spec = {"movie_id": tf.io.FixedLenFeature([], tf.int64),
            "genres": tf.io.FixedLenFeature([], tf.string)}
        return self._read_tfrecords(file_path, feature_spec)
    
    def _read_ratings_array_record(self, file_path:str, batch_size:int=2048) -> pl.DataFrame:
        if not os.path.exists(file_path):
            raise Exception(f'file not found: {file_path}')
        records = []
        reader = None
        try:
            reader = array_record_module.ArrayRecordReader(file_path)
            n = reader.num_records()
            for i in range(0, n, batch_size):
                i_end = i + batch_size
                if i_end >= n:
                    i_end = n
                batch_bytes = reader.read([x for x in range(i, i_end)]) # a single list of encodings, each being a list of 4 integers
                data = [msgpack.unpackb(b, use_list=False) for b in batch_bytes] # list of tuples of 4 integers
                for record in data: 
                    records.append({'user_id': int(record[0]), 'movie_id': int(record[1]),
                        'rating': int(record[2]), 'timestamp': int(record[3])})
        finally:
            if reader is not None:
                reader.close()        
        return pl.DataFrame(records)
   
    def _read_train_val_combined(self) -> pl.DataFrame:
        df_train_ratings = self._read_ratings_array_record(os.path.join(get_project_dir(),
            'src/test/resources/data/ratings_train/ratings_train.array_record'))
        df_val_ratings = self._read_ratings_array_record(os.path.join(get_project_dir(),
            'src/test/resources/data/ratings_val/ratings_val.array_record'))
        return pl.concat([df_train_ratings, df_val_ratings])
        
    def test_calc_metrics_full_catalog(self):
        
        '''
        (1) get test ratings data
        (2) get unique user ids of test ratings data into its own dataframe
        (3) construct rr = Retriever(...top_k=n_movies)
            note rr.max_hist
            top_k = n_movies - rr.max_hist
            (a) rr.create_dictionary_of_tensors(emb_type, inp)
                where arr is df['user_id'].numpy()
                inp = arr[:, np.newaxis]
            (b) rr.get_movies_given_users(user_test_dict, top_k=top_k, rm_hist=True)
                returns np.ndarray 2D of movie_ids
        ready for calculating metrics
        (4) consider above but adding test rating timestamp then calculating metrics
            up to that point in time...
        '''
        
        top_k = 20
        
        df_test_ratings = self._read_ratings_array_record(os.path.join(get_project_dir(),
            'src/test/resources/data/ratings_test_liked/ratings_test_liked.array_record'))

        #group test ratings by user_id, keeping list of movies and ratings and timestamps
        df_test_ratings_per_user = df_test_ratings.group_by('user_id').agg(
            [pl.col("movie_id").sort_by("rating", descending=True),
                pl.col('rating').sort_by("rating", descending=True),
                pl.col('timestamp').mean()])
        df_test_ratings_per_user = (df_test_ratings_per_user.select(
            pl.col('user_id'),
            pl.col('movie_id').alias('movie_ids'),
            pl.col('rating').alias('movie_ratings'),
            pl.col('timestamp').cast(pl.Int64)
        ))
        
        print(f'df_test_ratings_per_user={df_test_ratings_per_user}')
        print(f'count: {df_test_ratings_per_user.count()}', flush=True)
        
        arrow_table = df_test_ratings_per_user.select(['user_id', 'timestamp']).to_arrow()
        inp_tensor_dict = {
            name: tf.convert_to_tensor(arrow_table.column(name).to_numpy())
            for name in arrow_table.column_names
        }
        
        inp_tensor_dict['user_id'] = inp_tensor_dict['user_id'][:, tf.newaxis]
        inp_tensor_dict['timestamp'] = inp_tensor_dict['timestamp'][:, tf.newaxis]

        rr = self._construct_Retrieval()
        
        #enrich the inputs with user data:
        inp_tensor_dict = rr.create_dictionary_of_tensors(EmbeddingType.USER, inp_tensor_dict)
        
        rec_movies = rr.get_movies_given_users(inp_tensor_dict, top_k=top_k, rm_hist=True)
      
        # NDCG@K, MRR@K, Recall@K, hit_rate
        
        user_ground_truth_set = {}
        user_relevance_map = {}
        for row in df_test_ratings_per_user.to_dicts():
            u_id = row['user_id']
            m_ids = row['movie_ids']
            ratings = row['movie_ratings'] #ratings are in range 1 to 5, icnlusive
            user_ground_truth_set[u_id] = set(m_ids)
            user_relevance_map[u_id] = dict(zip(m_ids, ratings))
        # Extract the user_ids in the same order as your np_recommendations
        ordered_user_ids = inp_tensor_dict['user_id'].numpy()
        
        #calc metrics using vectorized ops:
        results = self._calculate_metrics(rec_movies, ordered_user_ids,
            user_ground_truth_set, user_relevance_map, k=20)
        rprint(results)
        '''
        {
            'hit_rate_20': 0.49663299663299665,
            'recall_20': 0.03736773380777989,
            'mrr_20': 0.16019396576232348,
            'ndcg_20': 0.06371890045798831
        }'''
        
    def _calculate_metrics(self, recommendations:np.ndarray, user_ids:np.ndarray,
        users_ground_truth_set:dict, relevance_map:dict, k:int=20) -> Dict[str, float]:
        
        hits_list = []
        recalls = []
        mrrs = []
        ndcgs = []
    
        for i, u_id in enumerate(user_ids):
            # Get the top K recommendations for this user
            top_k_rec = recommendations[i][:k]
            
            u_id = u_id[0]
            
            # Get the ground truth (what they actually watched/rated)
            truth_set = users_ground_truth_set.get(u_id, set())
            if not truth_set:
                continue # Skip users with no test data
                
            # Identify which recs were actually hits (1 if hit, 0 otherwise)
            hits = [1 if m_id in truth_set else 0 for m_id in top_k_rec]
            
            # --- HIT RATE & RECALL ---
            num_hits = sum(hits)
            hits_list.append(1 if num_hits > 0 else 0)
            recalls.append(num_hits / len(truth_set))
            
            # --- MRR (Mean Reciprocal Rank) ---
            # Find the index of the first '1' in hits
            try:
                first_hit_idx = hits.index(1)
                mrrs.append(1.0 / (first_hit_idx + 1))
            except ValueError:
                mrrs.append(0.0)
                
            # --- NDCG (Normalized Discounted Cumulative Gain) ---
            # Gain: Using the actual ratings for relevance
            dcg = 0.0
            for idx, m_id in enumerate(top_k_rec):
                if m_id in truth_set:
                    rel = relevance_map[u_id][m_id]
                    dcg += (2**rel - 1) / np.log2(idx + 2)
            
            # Ideal DCG: Sort the user's actual ratings descending and take top K
            actual_ratings = sorted(relevance_map[u_id].values(), reverse=True)[:k]
            idcg = sum((2**r - 1) / np.log2(idx + 2) for idx, r in enumerate(actual_ratings))
            
            ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
    
        return {
            f"hit_rate_{k}": np.mean(hits_list).item(),
            f"recall_{k}": np.mean(recalls).item(),
            f"mrr_{k}": np.mean(mrrs).item(),
            f"ndcg_{k}": np.mean(ndcgs).item()
        }

    def test_eval_single_genre(self):
        """
        evaluate the test users who were derived from the first ratings partition filtered for rating > 3
        and when grouped by user, each further filtered user had a movie list with 1 unique genre.
        The resulting number of unique users is small, 17ish, but they have characteristics that are easier to
        predict for an enrichment evaluation using the hypergeometric survival function.
        The goal is to understand whether the embeddings find good recommendations for these easier to understand
        "single genres" test users.  Note that as the process increases from Retrieval, to Ranker, to Re-Ranker,
        this narrow group  will receive broadened recommendations that may be more diverse.
        """
        
        top_ks = [20]#, 50, 100, 200, 1000]
        
        # Note: SciPy uses different labels:
        # M = Total population
        # n = Total successes in population
        # N = Number of draws
        # hypergeom.sf(k, M, n, N)
        
        ## ===== read in test users  and filter the test ratings data to keep only those users ====
        df_test_users = pl.read_parquet(os.path.join(get_project_dir(),
            "src/test/resources/data/single_genre/users_single_genre.parquet"))
        df_test_users = df_test_users.drop('zipcode')
        # columns=['user_id', 'gender', 'age', 'occupation']
        print(f'users_single_genre count: {df_test_users.count()}')
        
        df_test_ratings = self._read_ratings_array_record(
            os.path.join(get_project_dir(),
                'src/test/resources/data/ratings_test_liked/ratings_test_liked.array_record'))
        df_test_ratings = df_test_ratings.join(df_test_users, on='user_id', how='semi')
        
        # ======== get recommendations for those single genres users ======
        df_test_ratings_per_user = df_test_ratings.group_by('user_id').agg(
            [pl.col("movie_id").sort_by("rating", descending=True),
                pl.col('rating').sort_by("rating", descending=True),
                pl.col('timestamp').mean()])
        df_test_ratings_per_user = (df_test_ratings_per_user.select(
            pl.col('user_id'),
            pl.col('movie_id').alias('movie_ids'),
            pl.col('rating').alias('movie_ratings'),
            pl.col('timestamp').cast(pl.Int64)
        ))
        print(f'df_test_ratings_per_user count: {df_test_ratings_per_user.count()}')
        
        arrow_table = df_test_ratings_per_user.select(['user_id', 'timestamp']).to_arrow()
        inp_tensor_dict = {
            name: tf.convert_to_tensor(arrow_table.column(name).to_numpy())
            for name in arrow_table.column_names
        }
        
        inp_tensor_dict['user_id'] = inp_tensor_dict['user_id'][:, tf.newaxis]
        inp_tensor_dict['timestamp'] = inp_tensor_dict['timestamp'][:, tf.newaxis]
        
        rr = self._construct_Retrieval()
        
        # enrich the inputs with user data:
        inp_tensor_dict = rr.create_dictionary_of_tensors(EmbeddingType.USER,
            inp_tensor_dict)
        
        rec_movies = rr.get_movies_given_users(inp_tensor_dict, top_k=top_ks[-1],
            rm_hist=True)
        
        #add column for number of liked ratings for each user
        df_test_ratings_per_user = df_test_ratings_per_user.with_columns(
            N = pl.col("movie_ids").list.len()
        )
        
        ## ====== get the single genres for these users by joining users and movies =====
        df_movies = pl.read_parquet(self.movies_path)
        inp_hg_df = df_test_ratings.join(df_movies, on='movie_id', how='left')
        inp_hg_df = inp_hg_df.unique(subset=["user_id"], keep="first")
        inp_hg_df = inp_hg_df.select(['user_id', 'genres'])
        df_movies = df_movies.group_by('genres').agg(
            pl.col('movie_id').alias('all_genre_movies'))
        
        inp_hg_df = inp_hg_df.join(df_movies, on='genres', how='left') #[user_id, genres, all_genre_movies
        del df_movies
        
        #join user history of movies
        M_df = pl.DataFrame(
            [(k, list(v), self.n_movies - len(v)) for k, v in rr.user_history_dict.items()],
            schema=["user_id", "watched_movies", "M"],
            orient="row"
        )
        inp_hg_df = inp_hg_df.join(M_df, on='user_id', how='left') #[user_id, genres, all_genre_movies, watched_movies, M
        del M_df
        
        inp_hg_df = inp_hg_df.with_columns(
            avail_movies=pl.col("all_genre_movies").list.set_difference(
                pl.col("watched_movies"))
        )
        inp_hg_df = inp_hg_df.with_columns(
            n = pl.col("avail_movies").list.len()
        )
        #[user_id, genres, all_genre_movies, watched_movies, M, avail_movies, n,
        
        inp_hg_df = (inp_hg_df.join(
            df_test_ratings_per_user.select(["user_id", "N"]), on="user_id", how='left'
        ))
        # [user_id, genres, all_genre_movies, watched_movies, M, avail_movies, n, N
        
        # Note: SciPy uses different labels:
        # M = Total population
        # n = Total successes in population
        # N = Number of draws
        # hypergeom.sf(k, M, n, N)
        
        # We want the probability of getting MORE THAN the observed successes
        # M = self.n_movies - len(user_hist_dict[user_id])
        # n = number of the single genre movies avail to watch (specific to user)
        # N = number of ratings_test_liked for user
        # k = intersection of top_k recommendations with ratings_test_liked for user
        # prob_more_than_k = hypergeom.sf(k, M, n, N)
        
        print(f'inp_hg_df={inp_hg_df}')
        
        user_ids = inp_tensor_dict['user_id'].numpy()
        for top_k in top_ks:
            recommended = rec_movies[:, :top_k]
            rec_df = pl.DataFrame(
                [(u[0], r) for u, r in zip(user_ids, recommended)],
                schema=["user_id", "recommended"],
                orient="row"
            )
            #intersection of 'movie_ids' and 'recommended'
            df2 = df_test_ratings_per_user.join(rec_df, on='user_id', how='left')
            df2 = df2.with_columns(
                k_obs=pl.col("movie_ids").list.set_intersection(pl.col("recommended")).list.len().fill_null(0)
            )
            df2 = df2.select(['user_id', 'k_obs'])
            df2 = df2.join(inp_hg_df, on='user_id', how='left')
            df2 = df2.select(['user_id', 'k_obs', 'M', 'n', 'N'])
            # [user_id, M, n, N, k_obs
            
            k_obs_tensor = tf.constant(df2['k_obs'].to_numpy(), dtype=tf.float32)
            M_tensor = tf.constant(df2['M'].to_numpy(), dtype=tf.float32)
            n_tensor = tf.constant(df2['n'].to_numpy(), dtype=tf.float32)
            N_tensor = tf.constant(df2['N'].to_numpy(), dtype=tf.float32)
            
            sf_results = self.vectorized_hypergeom_sf(k_obs_tensor, M_tensor, n_tensor, N_tensor)
            
            name = f"survival_prob_{top_k}"
            df2 = df2.with_columns(
                pl.Series(name, sf_results.numpy())
            )
            df2 = df2.select(['user_id', name])
            df2 = df2.with_columns(
                (pl.col(name) < 0.05).alias(f'is_sig_{top_k}')
            )
            inp_hg_df = inp_hg_df.join(df2, on='user_id', how='left')
        
        #average over all results
        inp_hg_df = inp_hg_df.drop(['all_genre_movies', 'watched_movies', 'avail_movies'])
        print(f'results: {inp_hg_df}')
        inp_hg_df.write_csv(
            os.path.join(get_bin_dir(), "hypergeom_sf_single_genre_users.dat"),
            include_header=True, separator=","
        )
        
    def vectorized_hypergeom_pmf_log(self, k, M, n, N):
        """Log-PMF: log(comb(n, k)) + log(comb(M-n, N-k)) - log(comb(M, N))"""
        def log_comb(n, k):
            return (tf.math.lgamma(n + 1) -
                    tf.math.lgamma(k + 1) -
                    tf.math.lgamma(n - k + 1))
        
        return log_comb(n, k) + log_comb(M - n, N - k) - log_comb(M, N)
    
    def vectorized_hypergeom_sf(self, k_obs, M, n, N):
        """
        Computes P(X > k_obs) by summing PMFs.
        Expects k_obs, M, n, N to be tensors of the same shape.
        """
        # Create a grid for the summation: from max(k_obs)+1 up to max(N)
        # This is slightly complex to vectorize perfectly across different N,
        # but for small sample sizes, we can sum up to the global max N.
        max_sample = tf.cast(tf.reduce_max(N), tf.int32)
        k_range = tf.range(1, max_sample + 1, dtype=tf.float32)  # [1, 2, ..., max_N]
        
        # Expand dims for broadcasting: (1, max_N) and (batch_size, 1)
        k_grid = k_range[tf.newaxis, :]
        k_obs_col = k_obs[:, tf.newaxis]
        
        # Mask to only sum where k > k_obs and k <= N
        mask = (k_grid > k_obs_col) & (k_grid <= N[:, tf.newaxis])
        
        # Compute all possible log_pmfs in the range
        log_pmfs = self.vectorized_hypergeom_pmf_log(k_grid,
            M[:, tf.newaxis],
            n[:, tf.newaxis],
            N[:, tf.newaxis])
        
        # Convert back from log space, apply mask, and sum
        pmfs = tf.exp(log_pmfs)
        masked_pmfs = tf.where(mask, pmfs, tf.zeros_like(pmfs))
        
        return tf.reduce_sum(masked_pmfs, axis=1)
    
    
    def deserialize_fn(serialized_data):
        return msgpack.unpackb(serialized_data, raw=False)
    
    def save_retrieval_to_arrayrecord(self,
            writer: array_record_module.ArrayRecordWriter,
            user_id: int, recommended_movies: List[int]):
        record = msgpack.packb([user_id, recommended_movies],
            use_bin_type=True)
        writer.write(record)
    
   
    @staticmethod
    def make_ndcg_histogram(TOP_KS: list, res_ndcg_per_user,
            filename: str, show: bool = False):
        fig = make_subplots(rows=2, cols=2,
            subplot_titles=("TopK=20", "TopK=50", "TopK=100",
                "TopK=200"),
            x_title="NDCG@K", y_title="count")
        for i, top_k in enumerate(TOP_KS):
            fig_i = px.histogram(res_ndcg_per_user[top_k], nbins=50,
                title=f"NDCG@{top_k} Histogram")
            row_idx = i // 2
            col_idx = i % 2
            for trace in fig_i.data:
                fig.add_trace(trace, row=row_idx + 1, col=col_idx + 1)
        fig.write_image(os.path.join(get_bin_dir(), filename))
        if show:
            fig.show()
        del fig
        
    if __name__ == '__main__':
        unittest.main()

"""
scripts to analysis the trained models from the recommender_systems project.

some of the analysis is in rust in the ranker project directory inference_src and will be ported
to python here to have it all in one place.

NOTE: nothing in here uses the src/test/resources/data embeddings or recommendations.
"""
from pathlib import Path

'''
example usage from CLI:

export PIPELINE_DATA_DIR="../TMP10/bin/rs_pipeline"
export OUTPUT_BASE_DIR="../TMP10"
python3 -m unittest src.test.python.movie_lens_retrieval.post_training_analysis.TestAnalysis

or
PIPELINE_DATA_DIR="../TMP10/bin/rs_pipeline" \
OUTPUT_BASE_DIR="../TMP10" \
python3 -m unittest src.test.python.movie_lens_retrieval.post_training_analysis.TestAnalysis

metrics calculated:
- NDCG@k: DCG is a sum of the relevance of results, discounted by their ranks.
          NDCG is the normalization of DCG by the maximum possible DCG of the results set when
          ranked from highest to lowest gain.
- recall@k:  TP/(TP + FN) = TPR = sensitivity = hitrate
              relevant retrieved instances / all relevant instances
- precision@k:  TP/ (TP + FP)
              relevant retrieved instances / all retrieved instances
- MRR@k:  sum of the inverse of the rank of the results.  consider that each should be first, but
          was found at rank r.  MRR = (1/n_queries)*sum_{i=1,n_queries)( 1/ r_i )
- intra-list diversity
- inter-list diversity
- coverage
- embedding hubness
- popularity bias
'''
import json
import os
import shutil
import unittest
from typing import Any, Dict, Union, Tuple, List, Optional
import polars as pl
import msgpack
from array_record.python import array_record_module
from tensorflow.data import TFRecordDataset
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import tensorflow as tf

# 0 = all logs, 1 = no info, 2 = no info/warn, 3 = no info/warn/error
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="umap")

import sys
is_cli = __name__ == '__main__' or any('unittest' in arg for arg in sys.argv)
if is_cli:
    print('** running from CLI **', flush=True)
    sys.path.insert(0,
        os.path.join(os.getcwd(), "src/test/python/movie_lens_retrieval"))
    sys.path.insert(0,
        os.path.join(os.getcwd(), "src/main/python/movie_lens_retrieval"))

from helper import get_project_dir, get_bin_dir, \
    get_random_user_and_first_timestamp_from_ratings, \
    get_stratified_user_and_first_timestamp_from_ratings, \
    get_user_and_first_timestamp_from_ratings

from find_pushed_trainer_uri import get_pushed_trainer_uri

from movie_lens_retrieval.MovieData import MovieData, get_movie_tiers_df
from movie_lens_retrieval.UserData import UserData, get_user_tiers_df
from movie_lens_retrieval.ANNSearcher import *
from scann.scann_ops.py.scann_ops_pybind import ScannSearcher

class TestAnalysis(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        
        cls.AnnSearchType = Index
        #cls.AnnSearchType = scann.scann_ops_pybind.ScannSearcher
        
        cls.default_top_k = 100
        
        #defaults to the last run model in the recommender_systems bin directory
        pipeline_dir = os.environ.get("PIPELINE_DATA_DIR",
            os.path.join(get_project_dir(), "../recommender_systems/bin/rs_pipeline"))
        if not os.path.exists(pipeline_dir):
            raise ValueError(f"'{pipeline_dir}' does not exist")
        
        if not os.environ.get("PUSHER_URI", None):
            pusher_uri, trainer_uri = get_pushed_trainer_uri()
            saved_model_dir = pusher_uri
        else:
            saved_model_dir = os.environ.get("PUSHER_URI")
        if not os.path.exists(saved_model_dir):
            raise ValueError(f"{saved_model_dir} does not exist")
        
        raw_output_base = os.environ.get("OUTPUT_BASE_DIR", "").strip()
        output_base_dir = raw_output_base if raw_output_base else get_bin_dir()

        cls.summary_output_dir = os.path.join(output_base_dir, "post_training_analysis")
        if os.path.basename(cls.summary_output_dir) != "post_training_analysis":
            raise ValueError(f"Unsafe summary_output_dir: {cls.summary_output_dir}")

        shutil.rmtree(cls.summary_output_dir, ignore_errors=True)
        os.makedirs(cls.summary_output_dir, exist_ok=True)
        cls.summary_output_conclusions_dict = dict()
        cls.summary_output_metrics_dict = dict()
        
        #directories to write the parquet files to for pairwise user metrics comparisons
        cls.parquet_output_dir = os.path.join(cls.summary_output_dir, "parquet_metrics")
        os.makedirs(cls.parquet_output_dir, exist_ok=True)
        
        cls.loaded_user_movie_model = tf.saved_model.load(saved_model_dir)
        
        test_res_dir = os.path.join(get_project_dir(), "src/test/resources/data")
        
        #cold_start_path = os.path.join(test_res_dir, "cold_start_movies.txt")
        
        users_path = os.path.join(test_res_dir, "users/users.parquet")
        cls.movies_path = os.path.join(test_res_dir, "movies/movies.parquet")
        
        cls.model_dict = read_model_assets_hparams(saved_model_dir)
        cls.MOVIE_OFFSET: int = cls.model_dict['n_users'] + 1
        embed_dim = json.loads(cls.model_dict['layer_sizes'])[-1]
        cls.user_id_range_incl = [1, cls.model_dict['n_users']]
        cls.movie_id_range_incl = [cls.MOVIE_OFFSET, cls.MOVIE_OFFSET + cls.model_dict['n_movies']]
        
        cls.ratings_dict = {
            "full_history" : [
                    os.path.join(test_res_dir, "ratings_train.array_record"),
                    os.path.join(test_res_dir, "ratings_val.array_record")
                ],
            "positive_history" : [
                    os.path.join(test_res_dir, "ratings_train_liked.array_record"),
                    os.path.join(test_res_dir, "ratings_val_liked.array_record")
                ],
            "positive_train": os.path.join(test_res_dir, "ratings_train_liked.array_record"),
            "positive_val": os.path.join(test_res_dir, "ratings_val_liked.array_record"),
            "positive_test": os.path.join(test_res_dir, "ratings_test_liked.array_record"),
            "train": os.path.join(test_res_dir, "ratings_train.array_record"),
            "val": os.path.join(test_res_dir, "ratings_val.array_record"),
            "test": os.path.join(test_res_dir, "ratings_test.array_record"),
        }
        
        emb_movie_feature_spec = {
            "movie_id": tf.io.FixedLenFeature(shape=[], dtype=tf.int64,
                default_value=None),
            "embedding": tf.io.FixedLenFeature(shape=[embed_dim], dtype=tf.float32)
        }
        
        emb_user_feature_spec = {
            "user_id": tf.io.FixedLenFeature(shape=[], dtype=tf.int64,
                default_value=None),
            "embedding": tf.io.FixedLenFeature(shape=[embed_dim], dtype=tf.float32)
        }
        
        user_feature_spec = {
            "user_id": tf.io.FixedLenFeature([], tf.int64),
            "gender": tf.io.FixedLenFeature([], tf.string),
            "age": tf.io.FixedLenFeature([], tf.int64),
            "occupation": tf.io.FixedLenFeature([], tf.int64),
            "timestamp": tf.io.FixedLenFeature([], tf.int64),
        }
        
        movie_feature_spec = {
            "movie_id": tf.io.FixedLenFeature([], tf.int64),
            "genres": tf.io.FixedLenFeature([], tf.string)}
        
        cls.user_data = UserData(users_path)
        cls.movie_data = MovieData(cls.movies_path, cls.MOVIE_OFFSET)
        
        #to make it easier to point script to a new model, will generate the movie embeddings live
        # tf.Tensors:
        cls.catalog_movie_ids, cls.movie_catalog_embeddings = create_movie_catalog_embeddings()
        
        cls.train_history_df = read_ratings_to_df(cls.ratings_dict['positive_train'])
        
        cls.movie_tiers_df = get_movie_tiers_df(cls.train_history_df, pl.read_parquet(cls.movies_path))
        cls.user_tiers_df = get_user_tiers_df(cls.train_history_df, pl.read_parquet(users_path))
        
        cls.movie_catalog_emb_indexer : ScannSearcher = cls.build_searcher(embeddings=cls.movie_catalog_embeddings,
            top_k=TestAnalysis.default_top_k)
    
    @classmethod
    def tearDownClass(cls):
        out_file_path = os.path.join(cls.summary_output_dir, "summary_conclusions.json")
        with open(out_file_path, "w") as f:
            json.dump(cls.summary_output_conclusions_dict, f, indent=4)
        
        out_file_path = os.path.join(cls.summary_output_dir,"summary_metrics.json")
        with open(out_file_path, "w") as f:
            json.dump(cls.summary_output_metrics_dict, f, indent=4)
        print(f'wrote to ==> {cls.summary_output_dir}')

    @staticmethod
    def build_searcher(embeddings: Union[tf.Tensor, np.ndarray], top_k: int, normalize:bool=False):
        if issubclass(TestAnalysis.AnnSearchType, Index):
            return build_usearch_searcher(embeddings, top_k, normalize)
        else:
            return build_scann_searcher(embeddings, top_k, normalize)
    
    @staticmethod
    def search_batch(indexer: Union[Index, ScannSearcher], queries: Union[tf.Tensor, np.ndarray],
        final_num_neighbors:int):
        if issubclass(TestAnalysis.AnnSearchType, Index):
            if isinstance(queries, tf.Tensor):
                queries = queries.numpy()
            matches = indexer.search(queries, final_num_neighbors)
            return matches.keys, matches.distances
        else:
            return indexer.search_batched(queries=queries, final_num_neighbors=final_num_neighbors)
        
    
    def test_data_shifts(self):
        output_file_path = os.path.join(TestAnalysis.summary_output_dir, "data_shifts.json")
        
        res = dict()
        
        # a look at P(X,Y) where X = features, Y = label
        ## and its factors  P(X), P(X|Y), P(Y), P(Y|X)
        
        #movie_tiers_df is a dataframe of the entire movie_catalog with columns movie_id, movie_tier where movie_tier=0,1 or 2
        movie_counts = [TestAnalysis.movie_tiers_df.select(pl.col("movie_tier").eq(t).sum()).item() for t in range(3)]
        movie_fracs = self.normalize(movie_counts).tolist()
        for movie_tier in (0, 1, 2):
            res[f"movie_tier_{movie_tier}_catalog_counts"] = movie_counts[movie_tier]
    
        res["movie_tier_catalog_distribution"] = movie_fracs
        
        ## calc P(X) : for each full train, val, test count total pos+neg for each movie_tier and normalize, then calc emds
        ## calc P(Y|X) : for each full train, val, test count for each movie_tier (#pos)/(#pos + #neg), normalize then calc emds
        ## calc P(Y) : for each full train, val, test count total positives/total ratings
        ## calc P(X|Y) : only calculating for Y=1 which are the positives.  for each positive train, val, test count
        ##                movie_tier and normalize, then calc EMDs
        for name in ["train", "val", "test"]:
            # Full ratings split (pos + neg)
            full_df = read_ratings_to_df(TestAnalysis.ratings_dict[f'{name}'])
            full_df = full_df.join(TestAnalysis.movie_tiers_df, on="movie_id", how="left")
            full_movie_counts = [full_df.select(pl.col("movie_tier").eq(t).sum()).item() for t in range(3)]
            full_movie_fracs = self.normalize(full_movie_counts).tolist()
            res[f"P_X_{name}"] = full_movie_fracs
            
            # Ground-truth positive ratings split (Y=1)
            df = read_ratings_to_df(TestAnalysis.ratings_dict[f'positive_{name}'])
            df = df.join(TestAnalysis.movie_tiers_df, on="movie_id", how="left")
            pos_movie_counts_by_tier = [df.select(pl.col("movie_tier").eq(t).sum()).item() for t in range(3)]
            
            # Conditional probability P(Y=1 | X=tier t) = #(Y=1, X=t) / #(X=t)
            hit_rates = [
                p / m if m > 0 else 0.0
                for p, m in zip(pos_movie_counts_by_tier, full_movie_counts)
            ]
            res[f"P_Y_given_X_raw_{name}"] = hit_rates
            norm_hit_rates = self.normalize(hit_rates).tolist()
            res[f"P_Y|X_{name}"] = norm_hit_rates
            
            # Prior P(Y=1) = total positives / total ratings
            ratings_counts = max(1, full_df.height)
            res[f"P_Y_{name}"] = float(df.height) / float(ratings_counts)
            
            # P(X|Y=1) stratified by movie_tier
            movie_counts = pos_movie_counts_by_tier
            movie_fracs = self.normalize(movie_counts).tolist()
            for movie_tier in (0, 1, 2):
                res[f"movie_tier_{movie_tier}_{name}_counts"] = movie_counts[movie_tier]
            res[f"P_X|Y1_{name}"] = movie_fracs
        
            # #P(X|Y=1) stratified by user_tier
            #user_tiers_df is a polars dataframe of the full user catalog.  it has columns "user_id", "user_tier"
            df = df.join(TestAnalysis.user_tiers_df, on="user_id", how="left")
            user_counts = [df.select(pl.col("user_tier").eq(t).sum()).item() for t in range(3)]
            user_fracs = self.normalize(user_counts).tolist()
            for user_tier in (0, 1, 2):
                res[f"user_tier_{user_tier}_{name}_counts"] = user_counts[user_tier]
            
            res[f"movie_tier_{name}_distribution"] = movie_fracs
            res[f"user_tier_{name}_distribution"] = user_fracs
        
        # compare the data distributions.
        # because the tiers are ordinal, we can use the Wasserstein distance (a.k.a. Earth Mover's Distance)
        # EMDs will be in range [0,1]
        #    where 0 is no difference
        #          1 is polar opposite piling of mass in the distributions
        
        movie_pairs = [
            ("train", "val"),
            ("val", "test"),
            ("train", "test"),
        ]
        for name_a, name_b in movie_pairs:
            p = res[f"P_X_{name_a}"]
            q = res[f"P_X_{name_b}"]
            res[f"P_X_emd_{name_a}_to_{name_b}"] = self.calc_earth_mover_distances(p, q)
            p = res[f"P_Y|X_{name_a}"]
            q = res[f"P_Y|X_{name_b}"]
            res[f"P_Y|X_emd_{name_a}_to_{name_b}"] = self.calc_earth_mover_distances(p, q)
       
        # Pairwise Movie Tier EMDs
        movie_pairs = [
            ("train", "val"),
            ("val", "test"),
            ("train", "test"),
            ("catalog", "train"),
        ]
        for name_a, name_b in movie_pairs:
            p = res[f"movie_tier_{name_a}_distribution"]
            q = res[f"movie_tier_{name_b}_distribution"]
            res[f"movie_tier_emd_{name_a}_to_{name_b}"] = self.calc_earth_mover_distances(p, q)
        
        user_pairs = [("train", "val"), ("val", "test"), ("train", "test")]
        
        for name_a, name_b in user_pairs:
            p = res[f"user_tier_{name_a}_distribution"]
            q = res[f"user_tier_{name_b}_distribution"]
            res[f"user_tier_emd_{name_a}_to_{name_b}"] = self.calc_earth_mover_distances(p, q)
        
        conclusions = []
        m_train_to_val = res[f'movie_tier_emd_train_to_val']
        m_train_to_test = res[f'movie_tier_emd_train_to_test']
        u_train_to_val = res[f'user_tier_emd_train_to_val']
        u_train_to_test = res[f'user_tier_emd_train_to_test']
        
        # Rule 1: Movie Tier Split Drift (Train vs Val/Test)
        max_m_split_drift = max(m_train_to_val, m_train_to_test)
        
        if max_m_split_drift < 0.015:
            conclusions.append(
                f"MOVIE TIER STABILITY EXCELLENT: Minimal drift across dataset splits (Train-Test EMD: "
                f"{m_train_to_test:.4f}). Popularity profile is well-balanced across splits."
            )
        elif max_m_split_drift < 0.04:
            conclusions.append(
                f"MODERATE MOVIE TIER DRIFT: Slight popularity shift across splits (Train-Test EMD: "
                f"{m_train_to_test:.4f}), typical for temporal evaluation splits."
            )
        else:
            conclusions.append(
                f"HIGH MOVIE TIER DRIFT WARNING: Significant item popularity shift between Train and Test splits (Train-Test EMD: "
                f"{m_train_to_test:.4f}). Evaluation may be biased."
            )
        
        # Rule 2: User Tier Split Drift (Train vs Val/Test)
        max_u_split_drift = max(u_train_to_val, u_train_to_test)
        if max_u_split_drift < 0.015:
            conclusions.append(
                f"USER TIER STABILITY EXCELLENT: Activity profile distribution remains consistent across splits (Train-Test EMD: {u_train_to_test:.4f})."
            )
        elif max_u_split_drift < 0.04:
            conclusions.append(
                f"MODERATE USER TIER DRIFT: Mild activity profile shift between Train and Test (Train-Test EMD: {u_train_to_test:.4f})."
            )
        else:
            conclusions.append(
                f"HIGH USER TIER DRIFT WARNING: Power vs. light user ratio changes significantly in evaluation splits (Train-Test EMD: {u_train_to_test:.4f})."
            )
        
        # Rule 3: Catalog Interaction Bias (Catalog vs Train)
        cat_tr_emd = res[f'movie_tier_emd_catalog_to_train']
        if cat_tr_emd > 0.025:
            conclusions.append(
                f"EXPECTED INTERACTION POPULARITY BIAS: Catalog vs. Train EMD is {cat_tr_emd:.4f}. Training interactions heavily concentrate on popular head (Tier 0) items relative to catalog availability."
            )
        else:
            conclusions.append(
                f"UNUSUAL UNIFORM CATALOG COVERAGE: Catalog vs. Train EMD is low ({cat_tr_emd:.4f}), indicating interactions are surprisingly evenly spread across all item tiers."
            )
        
        ## ===========================
        p_x_train_to_val = res[f'P_X_emd_train_to_val']
        p_x_train_to_test = res[f'P_X_emd_train_to_test']
        p_y_x_train_to_val = res[f'P_Y|X_emd_train_to_val']
        p_y_x_train_to_test = res[f'P_Y|X_emd_train_to_test']
        p_y_train_to_val = abs(res['P_Y_train'] - res['P_Y_val'])
        p_y_train_to_test = abs(res['P_Y_train'] - res['P_Y_test'])
        p_x_y_train_to_val = res[f'movie_tier_emd_train_to_val']
        p_x_y_train_to_test = res[f'movie_tier_emd_train_to_test']
        
        shift_specs = [
            ("P(X) [Marginal Item Exposure]", p_x_train_to_val, p_x_train_to_test, "EMD"),
            ("P(Y=1|X) [Conditional Hit-Rate Profile]", p_y_x_train_to_val, p_y_x_train_to_test, "EMD"),
            ("P(Y=1) [Global Positive Rating Rate]", p_y_train_to_val, p_y_train_to_test, "AbsDiff"),
            ("P(X|Y=1) [Positive Item Tier Distribution]", p_x_y_train_to_val, p_x_y_train_to_test, "EMD"),
        ]
        for label_desc, tv, tt, metric_unit in shift_specs:
            max_split_drift = max(tv, tt)
            if max_split_drift < 0.015:
                conclusions.append(
                    f"{label_desc} STABILITY EXCELLENT: Distribution remains consistent across splits (Train-Test {metric_unit}: {tt:.4f})."
                )
            elif max_split_drift < 0.04:
                conclusions.append(
                    f"{label_desc} MODERATE DRIFT: Mild temporal shift across splits (Train-Test {metric_unit}: {tt:.4f})."
                )
            else:
                conclusions.append(
                    f"{label_desc} HIGH DRIFT WARNING: Significant shift between Train and Test splits (Train-Test {metric_unit}: {tt:.4f})."
                )
        
        # covariate shift P(X)_changed * P(Y|X)_unchanged
        # manifestation shift P(Y)_unchanged * P(X|Y)_changed
        # label shift P(Y)_changed * P(X|Y)_unchanged
        # concept shift covariate shift P(X)_unchanged * P(Y|X)_changed
        covariate_shift = False
        label_shift = False
        if max(p_x_train_to_val, p_x_train_to_test) >= 0.015 and max(p_y_x_train_to_val, p_y_x_train_to_test) < 0.015:
            conclusions.append(
                f"Covariate shift between Train and Test splits."
            )
            covariate_shift = True
        if max(p_y_train_to_val, p_y_train_to_test) >= 0.015 and max(p_x_y_train_to_val, p_x_y_train_to_test) < 0.015:
            conclusions.append(
                f"Label shift between Train and Test splits."
            )
            label_shift = True
        if not label_shift and max(p_x_train_to_val, p_x_train_to_test) < 0.015 and max(p_y_x_train_to_val, p_y_x_train_to_test) >= 0.015:
            conclusions.append(
                f"Concept shift between Train and Test splits."
            )
        if not covariate_shift and max(p_y_train_to_val, p_y_train_to_test) < 0.015 and max(p_x_y_train_to_val, p_x_y_train_to_test) >= 0.015:
            conclusions.append(
                f"Manifestation shift between Train and Test splits."
            )
            
        res["automated_conclusions"] = conclusions
        
        TestAnalysis.summary_output_conclusions_dict['data_shifts'] = conclusions
        
        #res = self.convert_to_native_types(res)
        
        print(f"Tier EMD Analysis:\n{json.dumps(res, indent=4)}")
        
        TestAnalysis.summary_output_metrics_dict.update(**res)
        del TestAnalysis.summary_output_metrics_dict['automated_conclusions']
        
        with open(output_file_path, "w") as f:
            json.dump(res, f, indent=4)
    
    def test_stratified_metrics(self):
        
        output_file_path = os.path.join(TestAnalysis.summary_output_dir,
            "stratified_metrics.json")
        
        # pos_test_df is ground truth positives test dataset w/ columns: user_id, movie_id, rating, timestamp
        pos_test_df = read_ratings_to_df(
            TestAnalysis.ratings_dict["positive_test"])
        user_ids, timestamps = get_user_and_first_timestamp_from_ratings(
            pos_test_df)
        user_ids = np.expand_dims(user_ids, axis=1)
        timestamps = np.expand_dims(timestamps, axis=1)
        user_embeddings = self.create_user_embeddings_batch(user_ids,
            timestamps)
        
        pos_test_tiered = pos_test_df.join(TestAnalysis.movie_tiers_df,
            on="movie_id", how="left")
        # Calculate ground-truth positive counts per user: overall and per tier
        user_gt_counts = pos_test_tiered.group_by("user_id").agg(
            pl.len().alias("total_positives"),
            (pl.col("movie_tier") == 0).sum().alias("gt_pos_tier_0"),
            (pl.col("movie_tier") == 1).sum().alias("gt_pos_tier_1"),
            (pl.col("movie_tier") == 2).sum().alias("gt_pos_tier_2"),
        )
        
        agg_res = dict()
        
        for top_k in (20, TestAnalysis.default_top_k):
            
            catalog_size = TestAnalysis.model_dict['n_movies']
            
            expected_random_recall = float(top_k) / float(catalog_size)
            
            # expected DCG@K = (N/C) * sum_{r=1,K}(1/log_2(r+1))
            expected_random_dcg_part1 = sum(
                [(1. / np.log2(r + 1.)) for r in range(1, top_k + 1)]) / float(
                catalog_size)
            
            # np.ndarrays of shape (n_users, top_k)
            neighbors, distances = TestAnalysis.search_batch(indexer=TestAnalysis.movie_catalog_emb_indexer,
                queries=user_embeddings, final_num_neighbors=top_k)
                
            neighbors += TestAnalysis.MOVIE_OFFSET
            
            retrieval_df = pl.DataFrame({
                "user_id": user_ids.squeeze(),
                "movie_id": neighbors
            }).explode("movie_id")
            
            # Rank the movie_ids for each user:
            retrieval_df = retrieval_df.with_columns(
                pl.int_range(1, pl.len() + 1).over("user_id").alias("rank")
            )
            
            user_pos_counts = pos_test_df.group_by("user_id").agg(
                pl.len().alias("total_positives")
            )
            
            retrieval_df = retrieval_df.join(pos_test_df,
                on=["user_id", "movie_id"], how="left")
            retrieval_df = retrieval_df.join(TestAnalysis.movie_tiers_df,
                on="movie_id", how="left")
            
            metrics_df = retrieval_df.group_by("user_id").agg(
                pl.col("rating").is_not_null().sum().alias("hits_at_k"),
                (1.0 / (pl.col("rank").filter(
                    pl.col("rating").is_not_null()) + 1.0).log(2)).sum().alias(
                    "dcg_at_k")
            )
            metrics_df = metrics_df.join(TestAnalysis.user_tiers_df,
                on="user_id", how="left")
            
            # Calculate Native IDCG, NDCG, Precision, and Baselines
            metrics_df = metrics_df.join(user_pos_counts, on="user_id",
                how="inner").with_columns(
                pl.int_ranges(1, pl.min_horizontal(pl.col("total_positives"),
                    pl.lit(top_k)) + 1)
                .list.eval(1.0 / (pl.element() + 1.0).log(2))
                .list.sum()
                .alias("idcg_at_k")
            ).with_columns(
                # Model Metrics
                (pl.col("hits_at_k") / pl.col("total_positives")).alias(
                    "recall_at_k"),
                (pl.col("hits_at_k") / float(top_k)).alias("precision_at_k"),
                (pl.col("dcg_at_k") / pl.col("idcg_at_k")).fill_nan(0.0).alias(
                    "ndcg_at_k"),
                
                # Expected Random Baselines
                pl.lit(expected_random_recall).alias(
                    "expected_random_recall_at_k"),
                (pl.col("total_positives") / float(catalog_size)).alias(
                    "expected_random_precision_at_k"),
                (pl.col(
                    "total_positives") * expected_random_dcg_part1 / pl.col(
                    "idcg_at_k")).alias("expected_random_ndcg_at_k")
            )
            
            #write to parquet for pair-wise comparison later between models
            parquet_file_path = os.path.join(TestAnalysis.parquet_output_dir,
                f"stratified_metrics_top_{top_k}.parquet")
            metrics_df.write_parquet(parquet_file_path)
            
            res = {
                f"n_samples_global": metrics_df.height,
                f"recall_at_{top_k}_mean": metrics_df.select(
                    pl.col("recall_at_k")).mean().item(),
                f"recall_at_{top_k}_std": metrics_df.select(
                    pl.col("recall_at_k")).std().item(),
                f"precision_at_{top_k}_mean": metrics_df.select(
                    pl.col("precision_at_k")).mean().item(),
                f"precision_at_{top_k}_std": metrics_df.select(
                    pl.col("precision_at_k")).std().item(),
                f"ndcg_at_{top_k}_mean": metrics_df.select(
                    pl.col("ndcg_at_k")).mean().item(),
                f"recall_at_{top_k}_mean_random": metrics_df.select(
                    pl.col("expected_random_recall_at_k")).mean().item(),
                f"precision_at_{top_k}_mean_random": metrics_df.select(
                    pl.col("expected_random_precision_at_k")).mean().item(),
                f"ndcg_at_{top_k}_mean_random": metrics_df.select(
                    pl.col("expected_random_ndcg_at_k")).mean().item()
            }
            
            agg_res = agg_res | res
            
            for user_tier in (0, 1, 2):
                df = metrics_df.filter(pl.col("user_tier") == user_tier)
                res = {
                    f"n_samples_user_tier_{user_tier}": df.height,
                    f"recall_at_{top_k}_mean_user_tier_{user_tier}": df.select(
                        pl.col("recall_at_k")).mean().item(),
                    f"recall_at_{top_k}_std_user_tier_{user_tier}": df.select(
                        pl.col("recall_at_k")).std().item(),
                    f"precision_at_{top_k}_mean_user_tier_{user_tier}": df.select(
                        pl.col("precision_at_k")).mean().item(),
                    f"precision_at_{top_k}_std_user_tier_{user_tier}": df.select(
                        pl.col("precision_at_k")).std().item(),
                    f"ndcg_at_{top_k}_mean_user_tier_{user_tier}": df.select(
                        pl.col("ndcg_at_k")).mean().item(),
                    f"ndcg_at_{top_k}_std_user_tier_{user_tier}": df.select(
                        pl.col("ndcg_at_k")).std().item(),
                }
                agg_res = agg_res | res
            
            # ====== conclusions ==========
            
            metrics_df = retrieval_df.group_by("user_id").agg(
                pl.col("rating").is_not_null().sum().alias("hits_global"),
                (1.0 / (pl.col("rank").filter(
                    pl.col("rating").is_not_null()) + 1.0).log(2)).sum().alias(
                    "dcg_global"),
                *[(pl.col("rating").is_not_null() & (
                        pl.col("movie_tier") == tier)).sum().alias(
                    f"hits_tier_{tier}") for tier in (0, 1, 2)],
                *[(1.0 / (pl.col("rank").filter(
                    pl.col("rating").is_not_null() & (
                            pl.col("movie_tier") == tier)) + 1.0).log(
                    2)).sum().alias(f"dcg_tier_{tier}") for tier in (0, 1, 2)]
            )
            metrics_df = metrics_df.join(user_gt_counts, on="user_id",
                how="inner")
            
            metrics_df = metrics_df.with_columns(
                pl.int_ranges(1, pl.min_horizontal(pl.col("total_positives"),
                    pl.lit(top_k)) + 1)
                .list.eval(1.0 / (pl.element() + 1.0).log(2))
                .list.sum()
                .alias("idcg_global"),
                *[
                    pl.when(pl.col(f"gt_pos_tier_{tier}") > 0)
                .then(
                        pl.int_ranges(1,
                            pl.min_horizontal(pl.col(f"gt_pos_tier_{tier}"),
                                pl.lit(top_k)) + 1)
                        .list.eval(1.0 / (pl.element() + 1.0).log(2))
                        .list.sum()
                    )
                .otherwise(None)
                .alias(f"idcg_tier_{tier}")
                    for tier in (0, 1, 2)
                ]
            )
            
            metrics_df = metrics_df.with_columns(
                # Global metrics
                (pl.col("hits_global") / pl.col("total_positives")).alias(
                    "recall_global"),
                (pl.col("hits_global") / float(top_k)).alias(
                    "precision_global"),
                (pl.col("dcg_global") / pl.col("idcg_global")).fill_nan(
                    0.0).alias("ndcg_global"),
                
                # Tier Recall & Precision
                *[
                    pl.when(pl.col(f"gt_pos_tier_{tier}") > 0)
                    .then(pl.col(f"hits_tier_{tier}") / pl.col(
                        f"gt_pos_tier_{tier}"))
                    .otherwise(None)
                    .alias(f"recall_tier_{tier}")
                    for tier in (0, 1, 2)
                ],
                *[
                    pl.when(pl.col(f"gt_pos_tier_{tier}") > 0)
                    .then(pl.col(f"hits_tier_{tier}") / float(top_k))
                    .otherwise(None)
                    .alias(f"precision_tier_{tier}")
                    for tier in (0, 1, 2)
                ],
                
                # Tier NDCG
                *[
                    pl.when(pl.col(f"idcg_tier_{tier}").is_not_null())
                    .then(pl.col(f"dcg_tier_{tier}") / pl.col(
                        f"idcg_tier_{tier}"))
                    .otherwise(None)
                    .alias(f"ndcg_tier_{tier}")
                    for tier in (0, 1, 2)
                ],
                
                # Random baselines
                pl.lit(expected_random_recall).alias("expected_random_recall"),
                (pl.col("total_positives") / float(catalog_size)).alias(
                    "expected_random_precision"),
                (pl.col(
                    "total_positives") * expected_random_dcg_part1 / pl.col(
                    "idcg_global")).alias("expected_random_ndcg")
            )
            
            res = {
                f"n_samples_global": metrics_df.height,
                f"recall_at_{top_k}_mean": metrics_df.select(
                    pl.col("recall_global")).mean().item(),
                f"recall_at_{top_k}_std": metrics_df.select(
                    pl.col("recall_global")).std().item(),
                f"precision_at_{top_k}_mean": metrics_df.select(
                    pl.col("precision_global")).mean().item(),
                f"precision_at_{top_k}_std": metrics_df.select(
                    pl.col("precision_global")).std().item(),
                f"ndcg_at_{top_k}_mean": metrics_df.select(
                    pl.col("ndcg_global")).mean().item(),
                f"ndcg_at_{top_k}_std": metrics_df.select(
                    pl.col("ndcg_global")).std().item(),
                f"recall_at_{top_k}_mean_random": metrics_df.select(
                    pl.col("expected_random_recall")).mean().item(),
                f"precision_at_{top_k}_mean_random": metrics_df.select(
                    pl.col("expected_random_precision")).mean().item(),
                f"ndcg_at_{top_k}_mean_random": metrics_df.select(
                    pl.col("expected_random_ndcg")).mean().item(),
                f"ndcg_at_{top_k}_std_random": metrics_df.select(
                    pl.col("expected_random_ndcg")).std().item(),
                
                # Dynamic Sample sizes for Movie Tiers based on non-null metrics
                **{f"n_samples_movie_tier_{tier}": metrics_df.select(
                    pl.col(f"recall_tier_{tier}").is_not_null().sum()).item()
                    for tier in (0, 1, 2)},
                
                **{
                    f"recall_at_{top_k}_mean_movie_tier_{tier}": metrics_df.select(
                        pl.col(f"recall_tier_{tier}")).mean().item() for tier
                    in
                    (0, 1, 2)},
                **{
                    f"precision_at_{top_k}_mean_movie_tier_{tier}": metrics_df.select(
                        pl.col(f"precision_tier_{tier}")).mean().item() for
                    tier in (0, 1, 2)},
                **{
                    f"ndcg_at_{top_k}_mean_movie_tier_{tier}": metrics_df.select(
                        pl.col(f"ndcg_tier_{tier}")).mean().item() for tier in
                    (0, 1, 2)}
            }
            agg_res = agg_res | res
            
            conclusions = []
            
            global_recall = agg_res[f"recall_at_{top_k}_mean"]
            random_recall = agg_res[f"recall_at_{top_k}_mean_random"]
            global_precision = agg_res[f"precision_at_{top_k}_mean"]
            random_precision = agg_res[f"precision_at_{top_k}_mean_random"]
            
            # Baseline Performance Lift
            if global_recall > random_recall * 3:
                conclusions.append(
                    f"STRONG RECALL LIFT: Global Recall@{top_k} ({global_recall:.1%}) heavily outperforms the random expected baseline ({random_recall:.1%}). The candidate generator is extracting meaningful semantic signal.")
            elif global_recall > random_recall:
                conclusions.append(
                    f"MODERATE RECALL LIFT: Global Recall@{top_k} ({global_recall:.1%}) beats the random baseline ({random_recall:.1%}), but there is room for improvement.")
            else:
                conclusions.append(
                    "RECALL FAILURE: The model fails to outperform a random candidate generator on Recall.")
            
            # Precision Lift
            if global_precision > random_precision * 3:
                conclusions.append(
                    f"STRONG PRECISION LIFT: Global Precision@{top_k} ({global_precision:.2%}) heavily outperforms the random expected precision ({random_precision:.2%}), indicating high density of relevant items in the retrieved slates.")
            elif global_precision > random_precision:
                conclusions.append(
                    f"MODERATE PRECISION LIFT: Global Precision@{top_k} ({global_precision:.2%}) is better than random ({random_precision:.2%}), suggesting basic relevance filtering is functioning.")
            else:
                conclusions.append(
                    "PRECISION FAILURE: The model retrieves slates with the same or worse precision than a completely random draw.")
            
            # Cohort History Utilization (Recall Trend: Tier 0 = Power Users, Tier 2 = Light Users)
            recall_power = agg_res[f"recall_at_{top_k}_mean_user_tier_0"]
            recall_light = agg_res[f"recall_at_{top_k}_mean_user_tier_2"]
            
            if recall_power > recall_light * 1.5:
                conclusions.append(
                    f"EXCELLENT HISTORY UTILIZATION: Power users (Tier 0 Recall: {recall_power:.1%}) perform massively better than light users (Tier 2 Recall: {recall_light:.1%}). The model successfully translates rich interaction histories into highly relevant slates.")
            elif recall_power > recall_light:
                conclusions.append(
                    f"MODERATE HISTORY UTILIZATION: Power users (Tier 0: {recall_power:.1%}) see better recall than light users (Tier 2: {recall_light:.1%}).")
            else:
                conclusions.append(
                    f"STRONG LIGHT-USER GENERALIZATION: Light/niche users (Tier 2 Recall: {recall_light:.1%}) achieve recall comparable to or exceeding power users (Tier 0 Recall: {recall_power:.1%}).")
            
            # NDCG (Ranking Quality: Tier 0 = Power Users, Tier 2 = Light Users)
            ndcg_power = agg_res[f"ndcg_at_{top_k}_mean_user_tier_0"]
            ndcg_light = agg_res[f"ndcg_at_{top_k}_mean_user_tier_2"]
            
            if ndcg_power > ndcg_light:
                conclusions.append(
                    f"POWER USER RANKING ADVANTAGE: NDCG is higher for power users (Tier 0: {ndcg_power:.2%} vs Tier 2: {ndcg_light:.2%}).")
            else:
                conclusions.append(
                    f"STRONG NICHE USER RANKING: Light/niche users achieve strong top-K ranking quality (Tier 2 NDCG: {ndcg_light:.2%} vs Tier 0: {ndcg_power:.2%}).")
            
            # Movie Tier Performance (Tier 0 = Head/Popular, Tier 2 = Tail/Niche)
            ndcg_head = agg_res.get(f"ndcg_at_{top_k}_mean_movie_tier_0", 0.0)
            ndcg_tail = agg_res.get(f"ndcg_at_{top_k}_mean_movie_tier_2", 0.0)
            
            if ndcg_head > ndcg_tail * 1.5:
                conclusions.append(
                    f"POPULARITY BIAS DETECTED: Ranking quality heavily favors head/popular movies (Tier 0 NDCG: {ndcg_head:.2%}) compared to long-tail/niche items (Tier 2 NDCG: {ndcg_tail:.2%}).")
            elif ndcg_head > ndcg_tail * 1.15:
                conclusions.append(
                    f"MODERATE HEAD FAVORITISM: Head movies rank higher (Tier 0 NDCG: {ndcg_head:.2%}) than long-tail items (Tier 2 NDCG: {ndcg_tail:.2%}), though tail items still receive meaningful top-{top_k} placement.")
            elif ndcg_tail > ndcg_head * 1.05:
                conclusions.append(
                    f"STRONG LONG-TAIL PERFORMANCE: Model surfaces and ranks niche/long-tail items (Tier 2 NDCG: {ndcg_tail:.2%}) higher than popular head items (Tier 0 NDCG: {ndcg_head:.2%}).")
            else:
                conclusions.append(
                    f"BALANCED ITEM TIER RANKING: NDCG performance remains consistent across movie tiers (Head Tier 0: {ndcg_head:.2%}, Tail Tier 2: {ndcg_tail:.2%}), indicating stable retrieval across popular and long-tail catalogs.")
            
            agg_res["automated_conclusions"] = conclusions
            
            TestAnalysis.summary_output_metrics_dict.update(**agg_res)
            del TestAnalysis.summary_output_metrics_dict[
                'automated_conclusions']
            
            TestAnalysis.summary_output_conclusions_dict[
                f'stratified_metrics@{top_k}'] = conclusions
        
        print(f'metrics\n={json.dumps(agg_res, indent=4)}')
        
        with open(output_file_path, "w") as f:
            json.dump(agg_res, f, indent=4)
    
    def test_coverage(self):
        """
        calculating item coverage as the number of unique movie ids recommended to users / size of movie catalog,
        then stratifying the results by movie_tier and user_tier.
        :return:
        """
        out_dir = os.path.join(TestAnalysis.summary_output_dir, "coverage")
        shutil.rmtree(out_dir, ignore_errors=True)
        os.makedirs(out_dir, exist_ok=True)
        
        output_file_path = os.path.join(out_dir, "coverage.json")
        
        agg_res = dict()
        
        pos_test_df = read_ratings_to_df(TestAnalysis.ratings_dict["positive_test"])
        
        user_ids, timestamps = get_user_and_first_timestamp_from_ratings(pos_test_df)
        user_ids = np.expand_dims(user_ids, axis=1)
        timestamps = np.expand_dims(timestamps, axis=1)
        #tf.Tensor
        user_embeddings = self.create_user_embeddings_batch(user_ids, timestamps)
        
        for top_k in [TestAnalysis.default_top_k, 20]:
            
            if top_k == TestAnalysis.default_top_k:
                indexer = TestAnalysis.movie_catalog_emb_indexer
            else:
                indexer = TestAnalysis.build_searcher(embeddings=TestAnalysis.movie_catalog_embeddings, top_k=top_k)
            
            res = dict()
            neighbors, distances = TestAnalysis.search_batch(indexer=indexer,
                queries=user_embeddings, final_num_neighbors=top_k)
            
            neighbors += TestAnalysis.MOVIE_OFFSET
            
            retrieval_df = pl.DataFrame({
                "user_id": user_ids.squeeze(),
                "movie_id": neighbors  #shape (n_users, top_k)
            }).explode("movie_id")
            
            count = retrieval_df["movie_id"].n_unique()
            cat_count = TestAnalysis.model_dict['n_movies']
            res[f"coverage_at_{top_k}_full"] = float(count)/float(cat_count)
            
            # count by movie_tier
            retrieval_df = retrieval_df.join(TestAnalysis.movie_tiers_df, on="movie_id", how="left")  #adds column "movie_tier"
            for movie_tier in range(0, 3):
                count = retrieval_df.filter(pl.col("movie_tier") == movie_tier)["movie_id"].n_unique()
                cat_count = TestAnalysis.movie_tiers_df.filter(pl.col("movie_tier") == movie_tier)["movie_id"].count()
                res[f"coverage_at_{top_k}_movie_tier_{movie_tier}"] = float(count) / float(cat_count)
            
            #count by user_tier
            retrieval_df = retrieval_df.join(TestAnalysis.user_tiers_df, on="user_id", how="left") #adds column "user_tier"
            cat_count = TestAnalysis.model_dict['n_movies']
            for user_tier in range(0, 3):
                count = retrieval_df.filter(pl.col("user_tier")==user_tier)["movie_id"].n_unique()
                res[f"coverage_at_{top_k}_user_tier_{user_tier}"] = float(count) / (float(cat_count))
                
            #count by user_tier and movie_tier
            for user_tier in range(0, 3):
                df = retrieval_df.filter(pl.col("user_tier") == user_tier)
                for movie_tier in range(0, 3):
                    df2 = df.filter(pl.col("movie_tier") == movie_tier)
                    count = df2["movie_id"].n_unique()
                    cat_count = TestAnalysis.movie_tiers_df.filter(pl.col("movie_tier") == movie_tier)["movie_id"].count()
                    res[f"coverage_at_{top_k}_user_tier_{user_tier}_movie_tier_{movie_tier}"] = float(count) / float(cat_count)
                    
            agg_res = agg_res | res
            
            retrieval_parquet_path = os.path.join(
                TestAnalysis.parquet_output_dir, f"user_retrievals_top_{top_k}.parquet")
            retrieval_df.write_parquet(retrieval_parquet_path)
            
            if True:
                # calc Gini coeff & Lorenz distribution
                res = dict()
                # ------------------------------------------------------------------
                # GINI COEFFICIENT & LORENZ DISTRIBUTION CALCULATION
                # Gini coefficient measures statistical dispersion of item exposure.
                # Rank-based formulation for a discrete population sorted ascending (y_1 <= ... <= y_n):
                #    G_pop = (2 * sum_{i=1}^n (i * y_i)) / (n * sum_{i=1}^n y_i) - ((n + 1) / n)
                #    G_unbiased = (n / (n - 1)) * G_pop
                # Lower Gini = more equitable catalog exposure; Higher Gini = popularity concentration.
                # ------------------------------------------------------------------
                
                # Must include catalog items with 0 retrievals to avoid underestimating inequality
                freq_df = (
                    TestAnalysis.movie_tiers_df.select("movie_id", "movie_tier")
                    .join(
                        retrieval_df.group_by("movie_id").agg(
                            pl.len().alias("retrieval_count")),
                        on="movie_id",
                        how="left"
                    )
                    .with_columns(pl.col("retrieval_count").fill_null(0))
                    .sort(["retrieval_count", "movie_id"]) # Deterministic sort for Lorenz & Gini math
                )
                
                freq_parquet_path = os.path.join(TestAnalysis.parquet_output_dir, f"item_frequencies_top_{top_k}.parquet")
                freq_df.write_parquet(freq_parquet_path)
                
                n_catalog = freq_df.height
                total_recs = freq_df["retrieval_count"].sum()
                
                if total_recs > 0:
                    # --- Full Catalog Gini (Unbiased [0, 1] scaled) ---
                    full_gini = self.calculate_gini(freq_df["retrieval_count"], unbiased=True)
                    
                    # --- Normalized Shannon Entropy & Effective Catalog Size ---
                    probs = freq_df.filter(pl.col("retrieval_count") > 0)["retrieval_count"].to_numpy() / float(total_recs)
                    shannon_entropy = float(-np.sum(probs * np.log(probs)))
                    norm_entropy = shannon_entropy / float(np.log(n_catalog)) if n_catalog > 1 else 0.0
                    effective_catalog_size = float(np.exp(shannon_entropy))
                    
                    # --- Lorenz Curve Calculation & Export ---
                    lorenz_df = freq_df.with_columns(
                        cum_items_pct=pl.int_range(1, n_catalog + 1) / n_catalog,
                        cum_recs_pct=pl.col("retrieval_count").cum_sum() / total_recs
                    )
                    
                    # Extract specific analytical points
                    bottom_80_share = float(
                        lorenz_df.filter(pl.col("cum_items_pct") <= 0.80)["cum_recs_pct"].max() or 0.0
                    )
                    top_10_share = float(
                        1.0 - (lorenz_df.filter(pl.col("cum_items_pct") <= 0.90)["cum_recs_pct"].max() or 0.0)
                    )
                    
                    # Export a clean, exact 100-point interpolated Lorenz Curve for plotting/JSON
                    cum_items_np = lorenz_df["cum_items_pct"].to_numpy()
                    cum_recs_np = lorenz_df["cum_recs_pct"].to_numpy()
                    percentiles = np.linspace(0.01, 1.00, 100)
                    lorenz_curve_array = np.interp(percentiles, cum_items_np, cum_recs_np).tolist()
                    
                    # --- Gini by Movie Tier (Within-Tier Exposure Inequality) ---
                    for movie_tier in range(0, 3):
                        tier_freq_df = freq_df.filter(
                            pl.col("movie_tier") == movie_tier
                        ).sort(["retrieval_count", "movie_id"])
                        tier_gini = self.calculate_gini(tier_freq_df["retrieval_count"], unbiased=True)
                        res[f"gini_at_{top_k}_movie_tier_{movie_tier}"] = float(tier_gini)
                    
                    # --- Gini by User Tier (Raw + Sample-Size Equalized to Prevent N-Cohort Bias) ---
                    tier_user_counts = [
                        retrieval_df.filter(pl.col("user_tier") == t)["user_id"].n_unique()
                        for t in range(0, 3)
                    ]
                    min_tier_users = min([c for c in tier_user_counts if c > 0], default=0)
                    
                    for user_tier in range(0, 3):
                        # Isolate retrievals generated ONLY by this user_tier
                        tier_retrieval_df = retrieval_df.filter(pl.col("user_tier") == user_tier)
                        
                        # Map those isolated retrievals onto the FULL movie catalog
                        user_tier_freq_df = (
                            TestAnalysis.movie_tiers_df.select("movie_id")
                            .join(
                                tier_retrieval_df.group_by("movie_id").agg(
                                    pl.len().alias("retrieval_count")),
                                on="movie_id",
                                how="left"
                            )
                            .with_columns(pl.col("retrieval_count").fill_null(0))
                            .sort(["retrieval_count", "movie_id"])
                        )
                        
                        u_tier_gini = self.calculate_gini(user_tier_freq_df["retrieval_count"], unbiased=True)
                        res[f"gini_at_{top_k}_user_tier_{user_tier}"] = float(u_tier_gini)
                        
                        # Equalized cohort sample size Gini (removes sample-size distortion across tiers)
                        if min_tier_users > 0:
                            sampled_users = (
                                tier_retrieval_df.select("user_id").unique()
                                .sample(n=min_tier_users, seed=42)
                            )
                            eq_tier_ret_df = tier_retrieval_df.join(sampled_users, on="user_id", how="inner")
                            eq_freq_df = (
                                TestAnalysis.movie_tiers_df.select("movie_id")
                                .join(
                                    eq_tier_ret_df.group_by("movie_id").agg(pl.len().alias("retrieval_count")),
                                    on="movie_id",
                                    how="left"
                                )
                                .with_columns(pl.col("retrieval_count").fill_null(0))
                            )
                            res[f"gini_equalized_at_{top_k}_user_tier_{user_tier}"] = float(
                                self.calculate_gini(eq_freq_df["retrieval_count"], unbiased=True)
                            )
                
                else:
                    full_gini = 0.0
                    norm_entropy = 0.0
                    effective_catalog_size = 0.0
                    bottom_80_share = 0.0
                    top_10_share = 0.0
                    lorenz_curve_array = []
                
                res[f"gini_at_{top_k}_full"] = float(full_gini)
                res[f"uniformity_entropy_at_{top_k}_full"] = float(norm_entropy)
                res[f"effective_catalog_size_at_{top_k}_full"] = float(effective_catalog_size)
                res[f"lorenz_at_{top_k}_bottom_80_share"] = float(bottom_80_share)
                res[f"lorenz_at_{top_k}_top_10_share"] = float(top_10_share)
                res[f"lorenz_at_{top_k}_curve_array"] = lorenz_curve_array  # List of 100 floats for JSON export
                
                if lorenz_curve_array:
                    output_lorenz_file_path = os.path.join(out_dir, f"lorenz_curve_k_{top_k}.png")
                    self.plot_lorenz_curve(lorenz_curve_array, top_k, full_gini, output_lorenz_file_path)
                
                # ------------------------------------------------------------------
                # AUTOMATED INSIGHTS & CONCLUSIONS
                # ------------------------------------------------------------------
                conclusions = []
                
                if total_recs > 0:
                    # Insight 1: Overall Catalog Concentration (Full Gini)
                    if full_gini > 0.90:
                        conclusions.append(
                            f"SEVERE POPULARITY CONCENTRATION at_{top_k}: Gini is {full_gini:.2f} (Effective Catalog: {effective_catalog_size:.0f} items). "
                            f"The model is acting as a popularity echo chamber, collapsing onto blockbuster items.")
                    elif full_gini < 0.45:
                        conclusions.append(
                            f"HIGHLY UNIFORM EXPOSURE at_{top_k}: Gini is {full_gini:.2f} (Effective Catalog: {effective_catalog_size:.0f} items). "
                            f"Recommendations are spread very broadly across the catalog.")
                    else:
                        conclusions.append(
                            f"HEALTHY EXPOSURE BALANCE at_{top_k}: Gini is {full_gini:.2f} (Effective Catalog: {effective_catalog_size:.0f} items, "
                            f"Normalized Entropy: {norm_entropy:.3f}). The model successfully balances mainstream relevance with catalog exploration.")
                    
                    # Insight 2: Long-Tail Health (Bottom 80% Share)
                    if bottom_80_share < 0.05:
                        conclusions.append(
                            f"DEAD TAIL at_{top_k}: The bottom 80% of the catalog receives only {bottom_80_share:.1%} of recommendations. Niche items are effectively invisible.")
                    elif bottom_80_share > 0.15:
                        conclusions.append(
                            f"STRONG TAIL at_{top_k}: The bottom 80% captures {bottom_80_share:.1%} of traffic, indicating excellent long-tail surfacing capability.")
                    else:
                        conclusions.append(
                            f"MODERATE TAIL at_{top_k}: The bottom 80% captures {bottom_80_share:.1%} of traffic.")
                    
                    # Insight 3: Head Concentration (Top 10% Share)
                    if top_10_share > 0.75:
                        conclusions.append(
                            f"HEAD HEAVY at_{top_k}: The top 10% of items consume {top_10_share:.1%} of all recommendation slots.")
                    else:
                        conclusions.append(
                            f"DIVERSE HEAD at_{top_k}: The top 10% consume {top_10_share:.1%} of slots, leaving plenty of room for the torso/tail.")
                    
                    # Insight 4: User Cohort Behavior (Tier 0 = Power Users, Tier 2 = Light Users)
                    gini_power = res.get(f"gini_equalized_at_{top_k}_user_tier_0", res.get(f"gini_at_{top_k}_user_tier_0", 1.0))
                    gini_light = res.get(f"gini_equalized_at_{top_k}_user_tier_2", res.get(f"gini_at_{top_k}_user_tier_2", 1.0))
                    
                    if gini_power < gini_light - 0.02:  # 0.02 buffer for noise
                        conclusions.append(
                            f"POWER USER DIVERSIFICATION at_{top_k}: Power users (Tier 0 Equalized Gini: {gini_power:.3f}) exhibit lower catalog concentration "
                            f"than light users (Tier 2 Equalized Gini: {gini_light:.3f}), leveraging rich interaction histories to explore broader inventory.")
                    elif gini_power > gini_light + 0.02:
                        conclusions.append(
                            f"LIGHT/NICHE USER DIVERSIFICATION at_{top_k}: Light/niche users (Tier 2 Equalized Gini: {gini_light:.3f}) receive more broadly distributed "
                            f"catalog exposure than power users (Tier 0 Equalized Gini: {gini_power:.3f}).")
                    else:
                        conclusions.append(
                            f"UNIFORM COHORT CONCENTRATION at_{top_k}: Power (Tier 0: {gini_power:.3f}) and Light (Tier 2: {gini_light:.3f}) users experience roughly the same level of catalog concentration.")
                
                res["automated_conclusions"] = conclusions
                
                TestAnalysis.summary_output_conclusions_dict[f'coverage@{top_k}'] = conclusions
                
                agg_res = agg_res | res
        
        TestAnalysis.summary_output_metrics_dict.update(**agg_res)
        TestAnalysis.summary_output_metrics_dict.pop('automated_conclusions', None)
        
        print(f'\n', json.dumps(agg_res, indent=4))
        with open(output_file_path, "w") as f:
            json.dump(agg_res, f, indent=4)
            
    def test_popularity_bias(self):
        ## a.k.a. Macroscopic Amplification
        
        output_file_path = os.path.join(TestAnalysis.summary_output_dir, "popularity_bias.json")
        
        agg_res = dict()
        
        # ======= stratified by user_tier ====================
        pos_test_df = read_ratings_to_df(TestAnalysis.ratings_dict["positive_test"])
        pos_test_df = pos_test_df.join(TestAnalysis.user_tiers_df, on="user_id", how="left")
        
        history_df = self.get_positive_ratings_history()
        history_df = history_df.join(TestAnalysis.user_tiers_df, on="user_id", how="left")
        
        stratification_key = "user_tier"
        stratification_values = [0,1,2]
        
        tier_user_ids_timestamps = get_stratified_user_and_first_timestamp_from_ratings(
            ratings_df=pos_test_df,
            stratification_key=stratification_key,
            stratification_values=stratification_values,
        )
        
        for tier in stratification_values:
            user_ids, timestamps = tier_user_ids_timestamps[tier]
            user_ids = np.expand_dims(user_ids, axis=1)
            timestamps = np.expand_dims(timestamps, axis=1)
            user_embeddings = self.create_user_embeddings_batch(user_ids, timestamps)
        
            # neighbors shape is (n_samples, top_k).  these are both np.ndarray
            neighbors, distances = TestAnalysis.search_batch(indexer=TestAnalysis.movie_catalog_emb_indexer,
                queries=user_embeddings, final_num_neighbors=TestAnalysis.default_top_k)
            #chk = full_movie_ids[neighbors]
            neighbors += TestAnalysis.MOVIE_OFFSET
            #are_equal = np.array_equal(chk, neighbors)
            
            tier_ground_truth_df = pos_test_df.filter(pl.col(stratification_key)==tier)
            tier_history_df = history_df.filter(pl.col(stratification_key)==tier)
            tag = f"{stratification_key}_{tier}"
            res = self.evaluate_popularity_bias(
                user_ids=user_ids.squeeze(),
                neighbors=neighbors,  # shape: (n_users, top_k)
                ground_truth_df = tier_ground_truth_df,  # Test set (positives only)
                train_history_df = tier_history_df,  # Train set (positives only)
                top_k = TestAnalysis.default_top_k,
                tag=tag
            )
        
            agg_res = agg_res | res
            
            TestAnalysis.summary_output_metrics_dict.update(**agg_res)
            del  TestAnalysis.summary_output_metrics_dict[f'{tag}_automated_conclusions']
            
        print("popularity bias\n", json.dumps(agg_res, indent=4))
        
        with open(output_file_path, "w") as f:
            json.dump(agg_res, f, indent=4)
    
    def test_embedding_hubness(self):
        """
        Evaluates Hypersphere Geometry (Wang & Isola Uniformity & Alignment) and
        Intrinsic k-NN Hubness (Radovanovic et al. k-occurrence skewness & anti-hubs)
        for both Movie (Candidate) and User (Query) embeddings.
        Exports per-movie and per-user Parquet files for paired statistical comparison.
        """
        from scipy.stats import skew
        output_file_path = os.path.join(TestAnalysis.summary_output_dir, "embedding_hubness.json")
        
        agg_res = dict()
        t_param = 2.0
        knn_k = 20
        
        def calc_uniformity_and_potentials(embs_np: np.ndarray) -> Tuple[float, np.ndarray]:
            """
            Computes global Wang & Isola log-uniformity and per-entity log radial potential:
                u_i = log( (1 / (N - 1)) * sum_{j != i} exp(-t * ||x_i - x_j||^2) )
            Lower (more negative) values indicate better spread across the unit hypersphere.
            """
            n = embs_np.shape[0]
            if n <= 1:
                return 0.0, np.zeros(n, dtype=np.float64)
            sim_matrix = np.clip(np.dot(embs_np, embs_np.T), -1.0, 1.0)
            dist_sq = np.maximum(0.0, 2.0 - 2.0 * sim_matrix)
            kernel_mat = np.exp(-t_param * dist_sq)
            np.fill_diagonal(kernel_mat, 0.0)
            row_means = np.maximum(np.sum(kernel_mat, axis=1) / float(n - 1), 1e-12)
            pointwise_log_potential = np.log(row_means)
            global_uniformity = float(np.log(np.maximum(np.mean(row_means), 1e-12)))
            return global_uniformity, pointwise_log_potential

        def calc_knn_hubness(embs_np: np.ndarray, k: int = 20) -> Tuple[np.ndarray, float, float, int]:
            """
            Computes the intrinsic k-nearest neighbor occurrence count N_k(i) for each item
            in the embedding space (excluding self), returning:
                - k_occurrences: array of shape (N,) with N_k(i) counts
                - skewness: standardized 3rd moment S_{N_k} (lower = less hubness concentration)
                - antihub_pct: fraction of items with N_k(i) == 0 (orphaned items)
                - max_hub_count: max_i N_k(i)
            """
            n = embs_np.shape[0]
            if n <= k + 1:
                return np.zeros(n, dtype=np.int32), 0.0, 0.0, 0
            sim_matrix = np.dot(embs_np, embs_np.T)
            np.fill_diagonal(sim_matrix, -np.inf)
            # Top-k indices per row using argpartition in O(N^2)
            top_k_indices = np.argpartition(sim_matrix, -k, axis=1)[:, -k:]
            k_occurrences = np.bincount(top_k_indices.ravel(), minlength=n)
            hub_skewness = float(skew(k_occurrences))
            antihub_pct = float(np.mean(k_occurrences == 0))
            max_hub_count = int(np.max(k_occurrences))
            return k_occurrences, hub_skewness, antihub_pct, max_hub_count

        # ------------------------------------------------------------------
        # 1. MOVIE (CANDIDATE) EMBEDDING UNIFORMITY & INTRINSIC HUBNESS
        # ------------------------------------------------------------------
        movie_embs_np = TestAnalysis.movie_catalog_embeddings.numpy()
        catalog_movie_ids_np = TestAnalysis.catalog_movie_ids.numpy().squeeze()
        
        global_item_uniformity, movie_potentials = calc_uniformity_and_potentials(movie_embs_np)
        movie_k_occ, item_hub_skew, item_antihub_pct, item_max_hub = calc_knn_hubness(movie_embs_np, k=knn_k)
        
        agg_res["uniformity_global_item"] = float(global_item_uniformity)
        agg_res[f"hubness_item_k{knn_k}_skewness"] = float(item_hub_skew)
        agg_res[f"hubness_item_k{knn_k}_antihub_pct"] = float(item_antihub_pct)
        agg_res[f"hubness_item_k{knn_k}_max_count"] = int(item_max_hub)
        
        movie_Geo_df = pl.DataFrame({
            "movie_id": catalog_movie_ids_np,
            "movie_uniformity_potential": movie_potentials,
            f"movie_knn_hubness_k{knn_k}": movie_k_occ
        }).join(TestAnalysis.movie_tiers_df, on="movie_id", how="left")
        
        for m_tier in [0, 1, 2]:
            tier_mask = (movie_Geo_df["movie_tier"] == m_tier).to_numpy()
            if np.any(tier_mask):
                tier_m_embs = movie_embs_np[tier_mask]
                tier_m_uni, _ = calc_uniformity_and_potentials(tier_m_embs)
                agg_res[f"uniformity_movie_tier_{m_tier}"] = float(tier_m_uni)
                agg_res[f"movie_potential_mean_tier_{m_tier}"] = float(np.mean(movie_potentials[tier_mask]))
                agg_res[f"hubness_item_k{knn_k}_mean_movie_tier_{m_tier}"] = float(np.mean(movie_k_occ[tier_mask]))
            else:
                agg_res[f"uniformity_movie_tier_{m_tier}"] = 0.0
                agg_res[f"movie_potential_mean_tier_{m_tier}"] = 0.0
                agg_res[f"hubness_item_k{knn_k}_mean_movie_tier_{m_tier}"] = 0.0

        # ------------------------------------------------------------------
        # 2. USER (QUERY) UNIFORMITY & POSITIVE PAIR ALIGNMENT (USER & MOVIE TIERS)
        # ------------------------------------------------------------------
        pos_test_df = read_ratings_to_df(TestAnalysis.ratings_dict["positive_test"])
        pos_test_df = pos_test_df.join(TestAnalysis.user_tiers_df, on="user_id", how="left")
        pos_test_df = pos_test_df.join(TestAnalysis.movie_tiers_df, on="movie_id", how="left")
        
        # Compute unique user embeddings once (using each user's first test timestamp)
        # so multi-interaction power users don't inject duplicate self-vectors into Uniformity
        uniq_u_ids_1d, uniq_ts_1d = get_user_and_first_timestamp_from_ratings(pos_test_df)
        uniq_u_embs = self.create_user_embeddings_batch(
            np.expand_dims(uniq_u_ids_1d, axis=1),
            np.expand_dims(uniq_ts_1d, axis=1)
        ).numpy()
        
        global_user_uniformity, user_potentials = calc_uniformity_and_potentials(uniq_u_embs)
        agg_res["uniformity_global_user"] = float(global_user_uniformity)
        
        user_geo_df = pl.DataFrame({
            "user_id": uniq_u_ids_1d,
            "user_uniformity_potential": user_potentials
        }).join(TestAnalysis.user_tiers_df, on="user_id", how="left")
        
        # Positive pair alignment across all positive test interactions
        all_u_ids = np.expand_dims(pos_test_df["user_id"].to_numpy(), axis=1)
        all_ts = np.expand_dims(pos_test_df["timestamp"].to_numpy(), axis=1)
        all_m_ids = pos_test_df["movie_id"].to_numpy()
        
        all_u_np = self.create_user_embeddings_batch(all_u_ids, all_ts).numpy()
        all_m_np = movie_embs_np[all_m_ids - TestAnalysis.MOVIE_OFFSET]
        pos_align_dists = np.maximum(0.0, 2.0 - 2.0 * np.sum(all_u_np * all_m_np, axis=1))
        
        pos_align_df = pos_test_df.with_columns(
            pl.Series("pos_alignment_dist_sq", pos_align_dists)
        )
        agg_res["alignment_global"] = float(np.mean(pos_align_dists))
        
        # Per-user tier Uniformity and Alignment
        for user_tier in [0, 1, 2]:
            # Deduplicated user uniformity for this user_tier
            u_tier_mask = (user_geo_df["user_tier"] == user_tier).to_numpy()
            if np.any(u_tier_mask):
                tier_u_uni, _ = calc_uniformity_and_potentials(uniq_u_embs[u_tier_mask])
            else:
                tier_u_uni = 0.0
            
            # Positive alignment for this user_tier
            tier_align_series = pos_align_df.filter(pl.col("user_tier") == user_tier)["pos_alignment_dist_sq"]
            tier_alignment = float(tier_align_series.mean()) if tier_align_series.len() > 0 else 0.0
            
            agg_res[f"uniformity_user_tier_{user_tier}"] = float(tier_u_uni)
            agg_res[f"alignment_user_tier_{user_tier}"] = float(tier_alignment)
            
        # Per-movie tier Positive Alignment
        for movie_tier in [0, 1, 2]:
            m_tier_align_series = pos_align_df.filter(pl.col("movie_tier") == movie_tier)["pos_alignment_dist_sq"]
            m_tier_alignment = float(m_tier_align_series.mean()) if m_tier_align_series.len() > 0 else 0.0
            agg_res[f"alignment_movie_tier_{movie_tier}"] = float(m_tier_alignment)

        # Export per-user and per-movie hubness/geometry Parquet files for paired testing
        per_user_align_df = pos_align_df.group_by("user_id").agg(
            pl.col("pos_alignment_dist_sq").mean().alias("user_pos_alignment")
        )
        user_geo_df = user_geo_df.join(per_user_align_df, on="user_id", how="left")
        user_geo_df.write_parquet(
            os.path.join(TestAnalysis.parquet_output_dir, "user_embedding_hubness.parquet")
        )
        
        per_movie_align_df = pos_align_df.group_by("movie_id").agg(
            pl.col("pos_alignment_dist_sq").mean().alias("movie_pos_alignment")
        )
        movie_Geo_df = movie_Geo_df.join(per_movie_align_df, on="movie_id", how="left")
        movie_Geo_df.write_parquet(
            os.path.join(TestAnalysis.parquet_output_dir, "movie_embedding_hubness.parquet")
        )

        # ------------------------------------------------------------------
        # RULE-BASED CONCLUSIONS
        # ------------------------------------------------------------------
        conclusions = []
        
        # Rule 1: Global Item Uniformity (Are items collapsed?)
        if global_item_uniformity > -1.0:
            conclusions.append(f"ITEM COLLAPSE WARNING: Global Item Uniformity is high ({global_item_uniformity:.2f}). The movie catalog is densely clumped together, likely leading to low diversity.")
        elif global_item_uniformity < -2.5:
            conclusions.append(f"HEALTHY ITEM DISPERSION: Global Item Uniformity is excellent ({global_item_uniformity:.2f}). Movies are well-distributed across the hypersphere.")
        else:
            conclusions.append(f"MODERATE ITEM UNIFORMITY: Items are adequately distributed ({global_item_uniformity:.2f}).")
            
        # Rule 2: Intrinsic Movie Hubness Skewness (k-NN Hubness)
        if item_hub_skew > 2.0:
            conclusions.append(
                f"HIGH MOVIE HUBNESS WARNING: k-NN occurrence skewness is {item_hub_skew:.2f} (Anti-hubs: {item_antihub_pct:.1%}, Max Hub: {item_max_hub}). "
                f"A small set of hub movies dominates nearest-neighbor structure."
            )
        else:
            conclusions.append(
                f"LOW MOVIE HUBNESS: k-NN occurrence skewness is well-controlled ({item_hub_skew:.2f}, Anti-hubs: {item_antihub_pct:.1%})."
            )
    
        # Rule 3: Global & Cohort User Uniformity (Tier 0 = Power Users, Tier 2 = Light Users)
        u_uni_power = agg_res["uniformity_user_tier_0"]
        u_uni_light = agg_res["uniformity_user_tier_2"]
        
        if global_user_uniformity > -2.0:
            conclusions.append(
                f"USER MANIFOLD COLLAPSE WARNING: Global User Uniformity is high ({global_user_uniformity:.2f}). "
                f"User query vectors are clumping into restricted islands on the hypersphere."
            )
        elif u_uni_power > u_uni_light + 0.35:
            conclusions.append(
                f"POWER USER COLLAPSE: Power users (Tier 0 uniformity: {u_uni_power:.2f}) are significantly more clumped than "
                f"Light users (Tier 2 uniformity: {u_uni_light:.2f}). Their rich histories are collapsing into dense 'average' vectors."
            )
        else:
            conclusions.append(
                f"USER GEOMETRY HEALTHY: Global User Uniformity ({global_user_uniformity:.2f}), Power users (Tier 0: {u_uni_power:.2f}), "
                f"and Light users (Tier 2: {u_uni_light:.2f}) are well-spread across the hypersphere without island collapse."
            )
    
        # Rule 4: Alignment Quality (Squared Euclidean distance 2 - 2*<u, m>; 2.0 = orthogonal/random, 0.0 = identical)
        # Note: Because a single static user vector must align with multiple diverse ground-truth movies simultaneously,
        # healthy multi-interest alignment typically sits around 1.25 - 1.65 (cosine similarity ~0.18 - 0.38).
        avg_alignment = np.mean([agg_res[f"alignment_user_tier_{i}"] for i in range(3)])
        if avg_alignment > 1.65:
            conclusions.append(
                f"WEAK ALIGNMENT WARNING: Average positive pair squared distance is {avg_alignment:.2f} (Orthogonal/Random is 2.0, Max is 4.0). "
                f"User embeddings are struggling to align with their ground-truth test items."
            )
        elif avg_alignment < 0.70:
            conclusions.append(
                f"OVERFIT / NARROW CONE WARNING: Average positive pair squared distance is {avg_alignment:.2f}. "
                f"Alignment is unusually tight, which may indicate memorization of narrow item cliques."
            )
        else:
            conclusions.append(
                f"HEALTHY MULTI-INTEREST ALIGNMENT: Average positive pair squared distance is {avg_alignment:.2f} "
                f"(Tier 0: {agg_res['alignment_user_tier_0']:.2f}, Tier 1: {agg_res['alignment_user_tier_1']:.2f}, Tier 2: {agg_res['alignment_user_tier_2']:.2f}). "
                f"Model balances proximity to positive items across multiple user interests."
            )
    
        agg_res["automated_conclusions"] = conclusions
        
        TestAnalysis.summary_output_conclusions_dict["embedding_hubness"] = conclusions
        
        agg_res = self.convert_to_native_types(agg_res)
        print(f"Embedding Hubness:\n", json.dumps(agg_res, indent=4))
        
        TestAnalysis.summary_output_metrics_dict.update(**agg_res)
        del TestAnalysis.summary_output_metrics_dict["automated_conclusions"]
        
        with open(output_file_path, "w") as f:
            json.dump(agg_res, f, indent=4)
    
    def test_plot_tsne_umap_movie_embeddings(self):
        
        outdir = os.path.join(TestAnalysis.summary_output_dir, "movie_embedding_plots")
        shutil.rmtree(outdir, ignore_errors=True)
        os.makedirs(outdir, exist_ok=True)
       
        #emb_movies_df = pl.DataFrame(
        #    [(m_id, emb) for m_id, emb in zip(TestAnalysis.catalog_movie_ids.numpy().squeeze(), TestAnalysis.movie_catalog_embeddings.numpy())],
        #    schema=["movie_id", "embedding"],
        #    orient="row"
        #)
        emb_movies_df = pl.DataFrame({
            "movie_id": TestAnalysis.catalog_movie_ids.numpy().squeeze(),
            "embedding": TestAnalysis.movie_catalog_embeddings.numpy()
        })
        emb_movies_df = emb_movies_df.join(TestAnalysis.movie_tiers_df, on="movie_id", how="left")
        
        self.plot_embeddings_umap_tsne(emb_movies_df, outdir, "all_movies",
            stratified_key="movie_tier")
    
    def test_plot_tsne_umap_user_embeddings(self):
        
        outdir = os.path.join(TestAnalysis.summary_output_dir,
            "user_embedding_plots")
        shutil.rmtree(outdir, ignore_errors=True)
        os.makedirs(outdir, exist_ok=True)
        
        # Use positive_test users and their first test interaction timestamp (consistent with all other evaluation tests)
        pos_ratings_df = read_ratings_to_df(TestAnalysis.ratings_dict["positive_test"])
        user_ids0, timestamps = get_user_and_first_timestamp_from_ratings(pos_ratings_df)
        user_ids = np.expand_dims(user_ids0, axis=1)
        timestamps = np.expand_dims(timestamps, axis=1)
        user_embeddings : tf.Tensor = self.create_user_embeddings_batch(user_ids, timestamps)
        
        emb_df = pl.DataFrame({
            "user_id": user_ids0,
            "embedding": user_embeddings.numpy()
        })
        
        emb_df = emb_df.join(TestAnalysis.user_tiers_df, on="user_id", how="left")
        
        self.plot_embeddings_umap_tsne(emb_df, outdir, "all_users",
            stratified_key="user_tier")
    
    def test_inter_list_diversity(self):
        
        output_file_path = os.path.join(TestAnalysis.summary_output_dir,
            "interlist_diversity.json")
        # Ensure you have a parquet output directory defined in your class
        parquet_dir = TestAnalysis.parquet_output_dir
        
        agg_res = dict()
        n_samples = 500
        num_catalog_movies = TestAnalysis.model_dict['n_movies']
        
        # ========= random sample of all users ==========================
        
        pos_test_df = read_ratings_to_df(
            TestAnalysis.ratings_dict["positive_test"])
        pos_test_df = pos_test_df.join(TestAnalysis.user_tiers_df,
            on="user_id", how="left")
        np.random.seed(42)
        (user_ids,
            timestamps) = get_random_user_and_first_timestamp_from_ratings(
            pos_test_df, n_samples)
        user_ids = np.expand_dims(user_ids, axis=1)
        timestamps = np.expand_dims(timestamps, axis=1)
        
        user_embeddings = self.create_user_embeddings_batch(user_ids,
            timestamps)
        
        neighbors, distances = TestAnalysis.search_batch(indexer=TestAnalysis.movie_catalog_emb_indexer,
            queries=user_embeddings, final_num_neighbors=TestAnalysis.default_top_k)
        
        inter_user_diversity, mean_jaccard, user_iud_array = self.calculate_exact_inter_user_diversity(
            neighbors, num_catalog_movies
        )
        
        # Write Parquet File
        pl.DataFrame({
            "user_id": user_ids.flatten(),
            "inter_list_diversity": user_iud_array
        }).write_parquet(
            os.path.join(parquet_dir,
                f"inter_list_diversity_all_users_top_{TestAnalysis.default_top_k}.parquet")
        )
        
        res = self.analyze_inter_user_diversity(
            mean_jaccard, num_catalog_movies, TestAnalysis.default_top_k,
            "all_users", baseline_jaccard=None
        )
        agg_res = agg_res | res
        
        # ======= stratified by user_tier ====================
        stratification_key = "user_tier"
        stratification_values: List = [0, 1, 2]
        
        np.random.seed(42)
        tier_dict: Dict[int, tuple[
            np.ndarray, np.ndarray]] = get_stratified_user_and_first_timestamp_from_ratings(
            ratings_df=pos_test_df,
            sample_size=n_samples,
            stratification_key=stratification_key,
            stratification_values=stratification_values,
        )
        
        for tier in stratification_values:
            t_user_ids, t_timestamps = tier_dict[tier]
            t_user_ids = np.expand_dims(t_user_ids, axis=1)
            t_timestamps = np.expand_dims(t_timestamps, axis=1)
            user_embeddings = self.create_user_embeddings_batch(t_user_ids,
                t_timestamps)
            
            neighbors, distances = TestAnalysis.search_batch(indexer=TestAnalysis.movie_catalog_emb_indexer,
                queries=user_embeddings, final_num_neighbors=TestAnalysis.default_top_k)
            inter_user_diversity, mean_jaccard, user_iud_array = self.calculate_exact_inter_user_diversity(
                neighbors, num_catalog_movies
            )
            
            # Write Parquet File
            pl.DataFrame({
                "user_id": t_user_ids.flatten(),
                "inter_list_diversity": user_iud_array
            }).write_parquet(
                os.path.join(parquet_dir,
                    f"inter_list_diversity_user_tier_{tier}_top_{TestAnalysis.default_top_k}.parquet")
            )
            
            res = self.analyze_inter_user_diversity(
                mean_jaccard, num_catalog_movies, TestAnalysis.default_top_k,
                f"user_tier_{tier}", baseline_jaccard=None
            )
            agg_res = agg_res | res
        
        # ===== cold start movies ====================
        (new_movie_ids, new_movie_embeddings) = self.get_cold_start_movies(
            pos_test_df)
        #concatenate the new embeddings onto the end of the original
        full_movie_embeddings = tf.concat(
            [TestAnalysis.movie_catalog_embeddings, new_movie_embeddings],
            axis=0)
        indexer = TestAnalysis.build_searcher(
            embeddings=full_movie_embeddings, top_k=TestAnalysis.default_top_k)
        
        num_catalog_movies += len(new_movie_ids)

        np.random.seed(42)
        (cs_user_ids,
            cs_timestamps) = get_random_user_and_first_timestamp_from_ratings(
            pos_test_df, n_samples)
        cs_user_ids = np.expand_dims(cs_user_ids, axis=1)
        cs_timestamps = np.expand_dims(cs_timestamps, axis=1)
        
        user_embeddings = self.create_user_embeddings_batch(cs_user_ids,
            cs_timestamps)
        
        neighbors, distances = TestAnalysis.search_batch(indexer=indexer,
                queries=user_embeddings, final_num_neighbors=TestAnalysis.default_top_k)
        
        inter_user_diversity, mean_jaccard, user_iud_array = self.calculate_exact_inter_user_diversity(
            neighbors, num_catalog_movies
        )
        
        # Write Parquet File
        pl.DataFrame({
            "user_id": cs_user_ids.flatten(),
            "inter_list_diversity": user_iud_array
        }).write_parquet(
            os.path.join(parquet_dir,
                f"inter_list_diversity_cold_start_top_{TestAnalysis.default_top_k}.parquet")
        )
        
        res = self.analyze_inter_user_diversity(
            mean_jaccard, num_catalog_movies, TestAnalysis.default_top_k,
            "all_users_and_catalog_has_cold_start_movies",
            baseline_jaccard=None
        )
        agg_res = agg_res | res
        
        print("inter_user_diversity\n", json.dumps(agg_res, indent=4))
        
        with open(output_file_path, "w") as f:
            json.dump(agg_res, f, indent=4)
    
    def calculate_exact_inter_user_diversity(self,
            neighbors: np.ndarray,
            num_catalog_movies: int
    ) -> tuple[float, float, np.ndarray]:
        """
        Computes exact Inter-User Diversity (1 - Jaccard Similarity) across user slates.

        Args:
            neighbors: (n_samples, top_k) array of retrieved movie IDs.
            num_catalog_movies: Total number of movies in the catalog.

        Returns:
            inter_user_diversity (float): Global average inter-user diversity.
            mean_jaccard_similarity (float): Global average Jaccard similarity.
            user_iud_array (np.ndarray): Per-user inter-list diversity scores (shape: n_samples,).
        """
        n_samples, top_k = neighbors.shape
        
        A = np.zeros((n_samples, num_catalog_movies), dtype=np.float32)
        
        row_indices = np.arange(n_samples)[:, None]
        A[row_indices, neighbors] = 1.0
        
        intersections = A @ A.T
        unions = (2 * top_k) - intersections
        
        jaccard_matrix = intersections / (unions + 1e-9)
        
        # --- NEW: Calculate Per-User Inter-List Diversity ---
        # Sum each row, subtract 1.0 (the diagonal where the user intersects 100% with themselves),
        # and divide by the number of other users to get the user's specific mean overlap.
        user_mean_jaccard = (np.sum(jaccard_matrix, axis=1) - np.diag(
            jaccard_matrix)) / (n_samples - 1)
        user_iud_array = 1.0 - user_mean_jaccard
        
        # --- Calculate Global Metrics ---
        total_jaccard = np.sum(jaccard_matrix).item() - np.trace(jaccard_matrix).item()
        num_pairs = n_samples * (n_samples - 1)
        
        mean_jaccard = float(total_jaccard / num_pairs)
        inter_user_diversity = 1.0 - mean_jaccard
        
        return inter_user_diversity, mean_jaccard, user_iud_array
    
    def test_intra_list_diversity(self):
        
        output_file_path = os.path.join(TestAnalysis.summary_output_dir, "intralist_diversity.json")
        
        #pos_test_df has columns user_id, movie_id, rating, timestamp
        pos_test_df = read_ratings_to_df(TestAnalysis.ratings_dict["positive_test"])
        first_interactions_df = pos_test_df.group_by("user_id").agg(pl.col("timestamp").min())
        user_ids = first_interactions_df["user_id"].to_numpy()
        print(f'n unique users in test ds={len(user_ids)}')
        user_ids = np.expand_dims(user_ids, axis=1)
        timestamps = first_interactions_df["timestamp"].to_numpy()
        timestamps = np.expand_dims(timestamps, axis=1)
        #user_embeddings is a tf.Tensor with shape (n_users, 32)
        user_embeddings = self.create_user_embeddings_batch(user_ids, timestamps)
        
        #neighbors and distances are the nearest neighbor mobie_ids and distances for each user embeedding in user_Embeddings
        #neighbors in an np.ndarray of shap (n_samples, top_k)
        neighbors, distances = TestAnalysis.search_batch(indexer=TestAnalysis.movie_catalog_emb_indexer,
            queries=user_embeddings, final_num_neighbors=TestAnalysis.default_top_k)
        if isinstance(neighbors, np.ndarray):
            neighbors = tf.convert_to_tensor(neighbors, dtype=tf.int32)
        
        res = self.calculate_batched_intra_list_diversity(neighbors, TestAnalysis.movie_catalog_embeddings)
        
        # EXTRACT PER-USER DATA AND EXPORT TO PARQUET
        user_ild_array = res.pop("user_ild_array")
        user_ild_df = pl.DataFrame({
            "user_id": user_ids.flatten(),  # Flatten to match the 1D ILD array
            "intra_list_diversity": user_ild_array
        })
        parquet_file_path = os.path.join(
            TestAnalysis.parquet_output_dir,
            f"user_ild_top_{TestAnalysis.default_top_k}.parquet"
        )
        user_ild_df.write_parquet(parquet_file_path)
        
        TestAnalysis.summary_output_metrics_dict.update(**res)
        del TestAnalysis.summary_output_metrics_dict["automated_conclusions"]
        
        print(f"Average Intra-List Diversity @ {TestAnalysis.default_top_k}\n: {json.dumps(res, indent=4)}")
        
        with open(output_file_path, "w") as f:
            json.dump(res, f, indent=4)
    
    def test_simulate_cold_start_embeddings(self):
        '''
        
        :return:
        '''
        """
        from the positive test dataset,
           candidate_pool: select unique movie ids for which there are at least 5-10 unique user_ids
           original_movie_ids : from candidate_pool_movies_users, choose 167 randomly from the movie_tier=0 partition, movie_tier=1 partition and movie_tier=2 partition.
           original_genres : the genres from original_movie_ids
           new_movie_ids : start numbering at self.movie_id_range_incl[-1] + 1
           original_embeddings: get the embeddings for original_movie_ids
           new_embeddings: create movie embeddings for (new_movie_ids, original_genres)
           original_scann_indexer : create ScANN index for all movies, including original_embeddings
           new_scann_indexer : create ScANN index for all movies, excluding original_embeddings, including new_embeddings.
           
           from users in candidate_pool_movies_users:
               - original_retrieval: get retrieval from original_scann_indexer.
                   are the original_movie_ids found?
                   - calc recall@k for each movie tier
               - new_retrieval: get retrieval from new_scann_indexer.
                   are the new_movie_ids found in same fractional amount as original_retrieval?
                   - calc recall@k for each movie tier
                   
          If your Recall@K plummets for these dropped-out items, your model relies too heavily
          on interaction IDs and has weak content representations.
        """
        output_file_path = os.path.join(TestAnalysis.summary_output_dir, "cold-start-recalls.json")
        
        pos_test_df = read_ratings_to_df(TestAnalysis.ratings_dict["positive_test"])
        #[user_id, movie_id, rating, timestamp, movie_tier]
        pos_test_df = self.join_df_to_movie_tiers(pos_test_df)
        
        movie_test_counts = pos_test_df.group_by("movie_id").agg(pl.len().alias("test_interaction_count"))
        candidate_pool = movie_test_counts.filter(pl.col("test_interaction_count") >= 5)
        
        pos_test_df = candidate_pool.join(pos_test_df, on="movie_id", how="left")
        
        num_per_tier = 167 #total = 501
        candidate_pool_df = (pos_test_df
            .sample(fraction=1.0, seed=42, shuffle=True)
            # Group by tier and take the top N from each group
            .group_by("movie_tier").head(num_per_tier)
        )
        #height=351
        
        #len=290
        original_movie_ids = candidate_pool_df["movie_id"].unique().to_numpy()
        
        #extract the unique users and their first timestamps. needed for user_embeddings input query
        first_interactions_df = candidate_pool_df.group_by("user_id").agg(pl.col("timestamp").min())
        user_ids = first_interactions_df["user_id"].to_numpy()
        user_ids = np.expand_dims(user_ids, axis=1)
        timestamps = first_interactions_df["timestamp"].to_numpy()
        timestamps = np.expand_dims(timestamps, axis=1)
        
        id0 = TestAnalysis.movie_id_range_incl[-1] + 1
        new_movie_ids = np.array([i for i in range(id0, id0 + len(original_movie_ids))])
        
        original_movie_ids = np.expand_dims(original_movie_ids, axis=1)
        new_movie_ids = np.expand_dims(new_movie_ids, axis=1)
        
        original_inputs = TestAnalysis.movie_data.get_movie(original_movie_ids)
        new_inputs = original_inputs.copy()
        new_inputs["movie_id"] = tf.constant(new_movie_ids)
        
        #original_embeddings : tf.Tensor = _create_movie_embeddings_batch(original_inputs)
        new_embeddings : tf.Tensor = _create_movie_embeddings_batch(new_inputs)
        new_embeddings = new_embeddings.numpy()
        
        full_movie_embeddings = TestAnalysis.movie_catalog_embeddings.numpy().copy()
        
        #for each new_movie_ids, replace with embedding for new_movie_ids.
        #this not only subtracts the old and inserts the new, but gives them the same index so that they
        # are findable when compared to the ground truth ratings.
        for i, m_id in enumerate(original_movie_ids):
            idx = m_id[0] - TestAnalysis.MOVIE_OFFSET
            assert(TestAnalysis.catalog_movie_ids[idx].numpy().item() == m_id[0])
            full_movie_embeddings[idx] = new_embeddings[i]
            
        top_k = TestAnalysis.default_top_k
        
        new_indexer = TestAnalysis.build_searcher(embeddings=full_movie_embeddings, top_k=top_k)
    
        #retrieve
        user_embeddings = self.create_user_embeddings_batch(user_ids, timestamps)
        
        orig_neighbors, orig_distances = TestAnalysis.search_batch(indexer=TestAnalysis.movie_catalog_emb_indexer,
                queries=user_embeddings, final_num_neighbors=TestAnalysis.default_top_k)
        new_neighbors, new_distances = TestAnalysis.search_batch(indexer=new_indexer,
                queries=user_embeddings, final_num_neighbors=TestAnalysis.default_top_k)
        
        res_original = self.evaluate_retrieval_with_tier_share(user_ids, orig_neighbors, TestAnalysis.MOVIE_OFFSET, candidate_pool_df, top_k=top_k)
        res_new = self.evaluate_retrieval_with_tier_share(user_ids, new_neighbors, TestAnalysis.MOVIE_OFFSET, candidate_pool_df, top_k=top_k)

        print("results on test set:\n", json.dumps(res_original, indent=4))
        print("results on test set cold start:\n", json.dumps(res_new, indent=4))
        
        res = self.compare_retrieval_runs(res_original, res_new, top_k, TestAnalysis.model_dict["n_movies"], "test set", "cold-start set")
        print("comparisons:\n", json.dumps(res, indent=4))
        
        TestAnalysis.summary_output_metrics_dict.update(**res)
        del TestAnalysis.summary_output_metrics_dict["automated_conclusions"]
        
        with open(output_file_path, "w") as f:
            json.dump(res, f, indent=4)
        
    def convert_to_native_types(self, obj):
        """Recursively converts NumPy types to native Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {str(k): self.convert_to_native_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_to_native_types(v) for v in obj]
        elif isinstance(obj, (np.float32, np.float64, np.floating)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64, np.integer)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    @staticmethod
    def calculate_gini(counts: Union[pl.Series, np.ndarray], unbiased: bool = True) -> float:
        """
        Computes the discrete Gini coefficient of item exposure / retrieval counts.
        
        Formula (sorted ascending y_1 <= y_2 <= ... <= y_n, 1-indexed i = 1..n):
            G_pop = (2 * sum_{i=1}^n (i * y_i)) / (n * sum_{i=1}^n y_i) - (n + 1) / n
        If unbiased=True, scales by n / (n - 1) so that a single item receiving 100%
        of all recommendations yields G = 1.0 regardless of catalog size n.
        """
        if isinstance(counts, pl.Series):
            y = np.sort(counts.to_numpy().astype(np.float64))
        else:
            y = np.sort(np.asarray(counts, dtype=np.float64))
            
        n = y.size
        total = np.sum(y)
        if n <= 1 or total <= 0.0:
            return 0.0
            
        ranks = np.arange(1, n + 1, dtype=np.float64)
        gini_pop = (2.0 * np.sum(ranks * y)) / (n * total) - ((n + 1.0) / n)
        if unbiased and n > 1:
            return float(np.clip(gini_pop * (n / (n - 1.0)), 0.0, 1.0))
        return float(np.clip(gini_pop, 0.0, 1.0))

    def calc_earth_mover_distances(self, distr_0, distr_1):
        """
        calculate normalized earth moviers distance between these 2 1D, length 3 distributions.
        Values returend wll be in range [0,1], inclusive
        :param distr_0:
        :param distr_1:
        :return: the normalized earth moviers distance between distr_0 and distr_1
        """
        if np.shape(distr_0) != (3,) or np.shape(distr_1) != (3,):
            raise ValueError(f"distr_0 and distr_1 must have shape (3,).  found {np.shape(distr_0)} and {np.shape(distr_1)}")
        """
        # for 1D and 3 bins, the EMD simplifies to the sum ob absolute differences of their CDFs
        # EMD(P,Q) = sum_{i=0, K-1}( |CDF_P(i) - CDF_Q(i)| )
        """
        cdf_0 = np.cumsum(distr_0)
        cdf_1 = np.cumsum(distr_1)
        max_emd = 2.0
        return float(np.sum(np.abs(cdf_0 - cdf_1)))/max_emd
    
    def evaluate_popularity_bias(self,
            user_ids: np.ndarray,
            neighbors: np.ndarray,  # shape: (n_users, top_k)
            ground_truth_df: pl.DataFrame,  # Test set (positives only)
            train_history_df: pl.DataFrame,  # Train set (positives only)
            top_k: int = 100,
            tag:str="all"
    ) -> Dict[str, Any]:
        """
        Measures Macroscopic Amplification (Popularity Bias) by comparing the
        popularity of retrieved slates vs. what the user naturally consumes.
        """
        # movie_count_map: calc movie_id count on history positive datasets.
        #     movie popularity follows a power-law (Zipfian) distribution.
        #     A blockbuster might have 5,000 positive interactions, while a niche movie has 5.
        #     So, the popularity counts are log transformed before averaging:
        #        log_pop(i) = log_2(1 + count(i))
        # for each user
        #      retrieved list:
        #          user_retr_avg = sum (movie_count_map[retr_movie_id]) / len(retr_movie_ids)
        #      test positive list:
        #          user_gt_avg = sum (movie_count_map[gt_movie_id]) / len(gt_movie_ids)
        #      popularity bias : if user_retr_avg is consistently > user_gt_avg
        # Build the Log-Popularity Map from Historical Training Data
        # Calculate log2(1 + count) for every movie
        pop_df = (TestAnalysis.train_history_df.group_by("movie_id")
            .agg(pl.len().alias("raw_count"))
            .with_columns(
                pl.col("raw_count").map_elements(lambda x: np.log2(1 + x),
                    return_dtype=pl.Float64).alias("log_pop")
            )
        )
        
        # Create a fast dictionary mapping movie_id -> log_pop
        # Default to 0.0 for items with no historical positives (pure cold-start)
        pop_map = dict(pop_df.select(["movie_id", "log_pop"]).iter_rows())
        
        n_users = neighbors.shape[0]
        user_retr_pop = []
        # Calculate Average Log-Popularity for Retrieved Slates
        for i in range(n_users):
            slate = neighbors[i, :top_k]
            slate_pops = [pop_map.get(m_id, 0.0) for m_id in slate]
            user_retr_pop.append(np.mean(slate_pops))
        user_retr_pop = np.array(user_retr_pop)
        
        retr_df = pl.DataFrame({
            "user_id" : user_ids,
            "retrieved_log_pop": user_retr_pop
        })
        # Calculate Average Log-Popularity for Ground Truth Test Items
        # Group the test DataFrame by user_id
        gt_grouped = (
            ground_truth_df
            .group_by("user_id")
            .agg(pl.col("movie_id").alias("gt_movies"))
            .with_columns(
                pl.col("gt_movies").map_elements(
                    lambda movies: float(np.mean([pop_map.get(m_id, 0.0) for m_id in movies])),
                    return_dtype=pl.Float64
                ).alias("ground_truth_log_pop")
            )
        )
        
        #write to parquet
        user_bias_df = (
            retr_df.join(gt_grouped.select(["user_id", "ground_truth_log_pop"]), 
                on="user_id", how="inner")
            .with_columns((pl.col("retrieved_log_pop") 
                - pl.col("ground_truth_log_pop")).alias("delta_log_pop"))
        )
        bias_parquet_path = os.path.join(TestAnalysis.parquet_output_dir,
            f"popularity_bias_{tag}.parquet")
        user_bias_df.write_parquet(bias_parquet_path)
        
        # Compute Bias Metrics
        mean_retr_pop = float(user_bias_df["retrieved_log_pop"].mean())
        mean_gt_pop = float(user_bias_df["ground_truth_log_pop"].mean())
        delta_pop = float(user_bias_df["delta_log_pop"].mean())
        std_delta_pop = float(user_bias_df["delta_log_pop"].std())
        
        # Amplification Ratio (Retrieved / Ground Truth)
        amplification_ratio = mean_retr_pop / max(1e-9, mean_gt_pop)
        
        # Automated Text Interpretation
        conclusions = []
        if amplification_ratio > 1.15:
            conclusions.append(
                f"[HIGH POPULARITY BIAS]: Model amplifies popularity by {amplification_ratio:.2f}x (Delta LogPop: {delta_pop:+.2f}). "
                f"Retrieved slates (LogPop: {mean_retr_pop:.2f}) are significantly more popular "
                f"than the users' natural test consumption (LogPop: {mean_gt_pop:.2f})."
            )
        elif amplification_ratio < 0.55 or delta_pop < -3.0:
            conclusions.append(
                f"[AGGRESSIVE POPULARITY SUPPRESSION]: Model suppresses popularity to {amplification_ratio:.2f}x (Delta LogPop: {delta_pop:+.2f}). "
                f"Retrieved slates (LogPop: {mean_retr_pop:.2f}) push heavily into obscure items relative to natural consumption ({mean_gt_pop:.2f})."
            )
        elif amplification_ratio < 0.85:
            conclusions.append(
                f"[LOG-Q TAIL EXPLORATION]: Retrieved top-{top_k} slates reach deeper into torso/tail inventory "
                f"(LogPop: {mean_retr_pop:.2f}, Ratio: {amplification_ratio:.2f}x, Delta: {delta_pop:+.2f}) than historical test positives ({mean_gt_pop:.2f}), "
                f"counteracting training popularity bias."
            )
        else:
            conclusions.append(
                f"[NEUTRAL POPULARITY]: Model preserves user consumption habits (Ratio: {amplification_ratio:.2f}x, Delta LogPop: {delta_pop:+.2f}). "
                f"Retrieved item popularity aligns closely with ground truth behavior."
            )
            
        TestAnalysis.summary_output_conclusions_dict[f"{tag}_popularity_bias"] = conclusions
        
        return {
            f"{tag}_metrics": {
                "mean_retrieved_log_pop": round(mean_retr_pop, 4),
                "mean_ground_truth_log_pop": round(mean_gt_pop, 4),
                "delta_log_pop": round(delta_pop, 4),
                "std_delta_log_pop": round(std_delta_pop, 4),
                "amplification_ratio": round(amplification_ratio, 4)
            },
            f"{tag}_automated_conclusions": conclusions
        }
    
    def normalize(self, a: list):
        return np.array(a) / float(sum(a))
    
    def analyze_inter_user_diversity(self,
            mean_jaccard: float,
            num_catalog_movies: int,
            top_k: int,
            dict_tag: str,
            baseline_jaccard: Optional[float] = None,
            baseline_name: str = "Standard Test",
            exp_name: str = "Cold Start"
    ) -> Dict[str, Any]:
        """
        Computes analytical random Jaccard floor, evaluates personalization/homogenization,
        and generates automated diagnostic text conclusions.

        Args:
            mean_jaccard: Mean pairwise Jaccard similarity across users (0.0 to 1.0).
            num_catalog_movies: Total unique movies in catalog (M).
            top_k: Candidate slate size evaluated (K).
            baseline_jaccard: Optional Jaccard score from standard test run for comparison.
        """
        inter_user_diversity = 1.0 - mean_jaccard
        
        ## interpret the results
        # mean_jaccard J: measures the overlap across different users.
        #    a higher value means that every user gets the same recommendation list
        #    which means there is a high popularity bias
        # inter_user_diversity UID = 1 - J: measures how unique recommendations are
        #    across candidates.  Max is 1.0.  Higher values indicate more personalization.
        # baseline:  a random baseline for analysis uses
        #    top_k items picked uniformly randomly from num_catalog_movies.
        #    then the expected jaccard similarity between any 2 users is
        #    modeled using item retrieval process as a probability problem using the Hypergeometric
        #    distribution and then applying a first-order approximation.
        #    Let M = num_catalog_movies
        #    Let A and B be the candidate slates for two different users,
        #    where both slates have exactly top_k randomly chosen items.
        #    J = |intersection| / |union|
        #       rewrite using exclusion inclusion principle
        #       |A union B| = |A| + |B| - |A intersect B|
        #          and since these are sizes, we know |A| and |B| are each top_k in size
        #       |A union B| = 2*top_k - |A intersect B|
        #    lex X = |A intersect B|
        #    then J = X / (2 * top_k - X)
        #
        #        A and B are each drawn without replacement from the whole catalog.
        #        X follows a hypergeometric distribution.
        #        consider A tagged, then we want to know how many of B are tagged.
        #        M = catalog
        #        K = size of A's slate
        #        n = number of draws of B
        #        E[X] = mean = n * K / M = K * K / M
        #
        #        E[J_random] = E[X / (2*top_k - X)]
        #           the variance of the hypergeometric is about equal to the mean when the catalog is much larger than the slate size
        #           so then we can simplify
        #        E[J_random] = E[X] / (2 * top_k - E[X])
        #        We can simplify the E[J(X)] = J(E[X]) because of Taylor series expansion and nearly negligible 2nd order terms.
        #             To find the expectation of a non-linear function f(X) which is J(x) here,
        #             we expand it around its mean mu = E[X] using a Taylor series:
        #               f(X) ~ f(mu) - (X-mu)*f(mu)' + (1/2)*(X-mu)^2*f(mu)''
        #             take expected value of both sides
        #               E[f(X)] ~ E[f(mu) - (X-mu)*f(mu)' + (1/2)*(X-mu)^2*f(mu)'']
        #             by linearity of expectation, we have that the expected value of a sum is the sum of the expected values
        #                E[f(X)] ~ E[f(mu)] - E[(X-mu)*f(mu)'] + E[(1/2)*(X-mu)^2*f(mu)'']
        #             because mu = E[X] and that is a constant scalar, not a random variable, we can take it outside of the Expectation.
        #                E[f(X)] ~ f(mu) - E[(X-mu)]*f(mu)' + E[(X-mu)^2]*(1/2)*f(mu)''
        #                E[f(X)] ~ f(mu) - (E[X]-mu)*f(mu)' + E[(X-mu)^2]*(1/2)*f(mu)''
        #                E[f(X)] ~ f(mu) - 0 + E[(X-mu)^2]*(1/2)*f(mu)''
        #                   by definition Var(X) is E[(X-mu)^2]
        #                E[f(X)] ~ f(mu) + Var(X)*(1/2)*f(mu)''
        #                   because the 2nd term is very small we have
        #                E[f(X)] ~ f(mu)
        #                E[J(X)] ~ J(E[X]) + error
        #    J = X / (2 * top_k - X)
        #    E[J_random] = top_k / (2 * num_catalog_movies - top_k)
        
        # 1. Analytical Expected Random Jaccard Floor: E[J] = K / (2M - K)
        expected_random_jaccard = top_k / max(1,(2 * num_catalog_movies) - top_k)
        expected_random_iud = 1.0 - expected_random_jaccard
        
        # 2. Overlap Multiplier (How many times more overlapping than random?)
        jaccard_multiplier = mean_jaccard / max(1e-9, expected_random_jaccard)
        
        # --- 3. Build Metrics Summary ---
        metrics = {
            "mean_jaccard_similarity": round(mean_jaccard, 6),
            "inter_user_diversity": round(inter_user_diversity, 6),
            "expected_random_jaccard": round(expected_random_jaccard, 6),
            "expected_random_iud": round(expected_random_iud, 6),
            "jaccard_multiplier_vs_random": round(jaccard_multiplier, 2)
        }
        
        if baseline_jaccard is not None:
            jaccard_abs_change = mean_jaccard - baseline_jaccard
            jaccard_rel_change = ((mean_jaccard - baseline_jaccard) / baseline_jaccard * 100) if baseline_jaccard != 0 else 0.0
            metrics[f"{baseline_name}_jaccard"] = round(baseline_jaccard, 6)
            metrics["jaccard_abs_change"] = round(jaccard_abs_change, 6)
            metrics["jaccard_rel_change_pct"] = f"{jaccard_rel_change:+.2f}%"
        
        # --- 4. Automated Text Conclusions ---
        conclusions = []
        
        # Absolute Personalization Health Checks
        if jaccard_multiplier >= 15.0:
            conclusions.append(
                f"[HIGH CATALOG HOMOGENIZATION]: Mean Jaccard overlap ({mean_jaccard * 100:.2f}%) is {jaccard_multiplier:.1f}x higher "
                f"than random chance ({expected_random_jaccard * 100:.2f}%). The query tower is over-indexing on popular items "
                f"and serving near-identical candidate slates across different users."
            )
        elif jaccard_multiplier >= 4.0:
            conclusions.append(
                f"[MODERATE PERSONALIZATION]: Mean Jaccard overlap ({mean_jaccard * 100:.2f}%) is {jaccard_multiplier:.1f}x random chance "
                f"({expected_random_jaccard * 100:.2f}%). Candidate slates share core popular hubs while retaining user-specific targeting."
            )
        else:
            conclusions.append(
                f"[HYPER-PERSONALIZED SLATES]: Mean Jaccard overlap ({mean_jaccard * 100:.2f}%) aligns closely with random chance "
                f"({expected_random_jaccard * 100:.2f}%). Candidate slates are highly individualized across users."
            )
        
        # Run Comparison (e.g., Cold Start vs Standard Test)
        if baseline_jaccard is not None:
            j_shift = ((mean_jaccard - baseline_jaccard) / baseline_jaccard) * 100
            if j_shift >= 25.0:
                conclusions.append(
                    f"[COLD-START POPULARITY RETREAT]: Jaccard slate overlap increased by {j_shift:+.1f}% under {exp_name} "
                    f"({baseline_jaccard * 100:.2f}% -> {mean_jaccard * 100:.2f}%). Without user interaction history, "
                    f"the query tower retreats to retrieving generic popular blockbusters."
                )
            elif j_shift <= -25.0:
                conclusions.append(
                    f"[COLD-START DIVERGENT SPRAY]: Jaccard slate overlap dropped by {abs(j_shift):.1f}% under {exp_name} "
                    f"({baseline_jaccard * 100:.2f}% -> {mean_jaccard * 100:.2f}%). Without interaction history, "
                    f"metadata embeddings scatter users across ungrounded regions of the catalog."
                )
            else:
                conclusions.append(
                    f"[STABLE INTER-USER SEPARATION]: {exp_name} maintains consistent cross-user slate overlap relative to {baseline_name} "
                    f"({baseline_jaccard * 100:.2f}% vs {mean_jaccard * 100:.2f}%)."
                )
        
        TestAnalysis.summary_output_conclusions_dict[f"{dict_tag}_inter_user_diversity_conclusions"] = conclusions

        TestAnalysis.summary_output_metrics_dict[f"{dict_tag}_inter_user_diversity_metrics"] = metrics.copy()
        
        return {
            f"{dict_tag}_inter_user_diversity_metrics": metrics,
            f"{dict_tag}_inter_user_diversity_conclusions": conclusions
        }
    
    def calculate_minhash_inter_user_diversity(self,
            neighbors: np.ndarray,
            num_hashes: int = 150
    ) -> tuple[float, float]:
        """
        Approximates Inter-User Diversity using MinHash signatures (MMDS approach).
        good to use when neighbors[-1] > 100_000
        """
        n_samples, top_k = neighbors.shape
        
        # Large prime number for the hash function: h(x) = (ax + b) % c
        # Usually choose a prime just larger than the max item ID which
        # is 3883 in this case, but if the ids were transformed to include
        #  self.MOVIES_OFFSET, would need to use another Merseinne prime: 16383
        prime = 8191
        
        # Generate random coefficients for the hash functions
        # Shapes: (num_hashes, 1, 1) for broadcasting
        a = np.random.randint(1, prime, size=(num_hashes, 1, 1))
        b = np.random.randint(0, prime, size=(num_hashes, 1, 1))
        
        # 1. Compute hash values for every item in every user's slate
        # neighbors shape broadcasted to (1, n_samples, top_k)
        # Output shape: (num_hashes, n_samples, top_k)
        hashed_values = (a * neighbors[None, :, :] + b) % prime
        
        # 2. Create MinHash Signatures
        # Take the minimum hash value across the 'top_k' items for each user
        # Output shape: (num_hashes, n_samples)
        signatures = np.min(hashed_values, axis=2)
        
        # 3. Compare signatures pairwise
        # Two signatures match with probability == Jaccard Similarity
        # Broadcasting magic: (num_hashes, n_samples, 1) == (num_hashes, 1, n_samples)
        matches = (signatures[:, :, None] == signatures[:, None, :])
        
        # Calculate estimated Jaccard by averaging matches across hash functions
        # jaccard_est shape: (n_samples, n_samples)
        jaccard_est = np.mean(matches, axis=0)
        
        # Average across all user pairs, ignoring the diagonal
        total_jaccard = np.sum(jaccard_est) - np.trace(jaccard_est)
        num_pairs = n_samples * (n_samples - 1)
        
        mean_jaccard = float(total_jaccard / num_pairs)
        inter_user_diversity = 1.0 - mean_jaccard
        
        return inter_user_diversity, mean_jaccard
    
    def calculate_batched_intra_list_diversity(
            self,
            neighbors: tf.Tensor,  # shape: (num_users, top_k)
            movie_embeddings: tf.Tensor,
            # shape: (num_catalog_movies, 32) (unit-normalized)
            num_random_samples: int = 200
            # Number of random slates to draw for baseline
    ) -> Dict[str, Any]:
        """
        Computes average Intra-List Diversity (ILD) across a batch of users,
        estimates the random catalog ILD baseline via Monte Carlo sampling,
        and returns an automated text analysis along with per-user ILD arrays.
        """
        num_users = tf.shape(neighbors)[0]
        top_k = tf.shape(neighbors)[1]
        num_catalog_movies = tf.shape(movie_embeddings)[0]
        
        # --- Helper: Vectorized ILD calculation for any batch of slates ---
        def _compute_ild(slate_indices: tf.Tensor) -> tf.Tensor:
            slate_embeddings = tf.gather(movie_embeddings, slate_indices)
            sim_matrices = tf.matmul(slate_embeddings, slate_embeddings, transpose_b=True)
            dist_matrices = 1.0 - sim_matrices
            
            n_slates = tf.shape(slate_indices)[0]
            k = tf.shape(slate_indices)[1]
            zeros_diagonal = tf.zeros((n_slates, k), dtype=dist_matrices.dtype)
            dist_matrices = tf.linalg.set_diag(dist_matrices, zeros_diagonal)
            
            sum_dists_per_slate = tf.reduce_sum(dist_matrices, axis=[1, 2])
            num_pairs = tf.cast(k * (k - 1), dtype=dist_matrices.dtype)
            
            # RETURN THE PER-SLATE TENSOR INSTEAD OF THE BATCH MEAN
            return sum_dists_per_slate / num_pairs
        
        # Compute Model ILD
        model_ild_per_slate = _compute_ild(neighbors)
        model_ild = float(tf.reduce_mean(model_ild_per_slate).numpy())
        
        # Estimate Empirical Random Baseline ILD
        random_indices = tf.random.uniform(
            shape=(num_random_samples, top_k),
            minval=0,
            maxval=num_catalog_movies,
            dtype=tf.int32
        )
        random_ild_per_slate = _compute_ild(random_indices)
        random_ild = float(tf.reduce_mean(random_ild_per_slate).numpy())
        
        # Compute Diversity Ratio
        diversity_ratio = model_ild / random_ild if random_ild > 0 else 0.0
        
        # Automated Text Analysis
        conclusions = []
        if diversity_ratio < 0.20:
            conclusions.append(
                f"[EXTREME CLUSTERING]: Model ILD ({model_ild:.4f}) retains only {diversity_ratio * 100:.1f}% of random catalog diversity. "
                f"Risk of severe candidate bottlenecking into a single sub-genre."
            )
        elif diversity_ratio < 0.50:
            conclusions.append(
                f"[FOCUSED CANDIDATE POOL]: Model ILD ({model_ild:.4f}) retains {diversity_ratio * 100:.1f}% of random catalog diversity. "
                f"This indicates strong, coherent cluster targeting around user preferences."
            )
        else:
            conclusions.append(
                f"[BROAD CANDIDATE POOL]: Model ILD ({model_ild:.4f}) retains {diversity_ratio * 100:.1f}% of random catalog diversity. "
                f"Candidates span multiple distinct semantic regions in the embedding space."
            )
            
        TestAnalysis.summary_output_conclusions_dict["intra-list divserity"] = conclusions
        
        return {
            "ild_movie": round(model_ild, 6),
            "ild_random": round(random_ild, 6),
            "diversity_ratio": round(diversity_ratio, 4),
            "automated_conclusions": conclusions,
            # PASS THE PER-USER ARRAY FOR PARQUET EXPORT
            "user_ild_array": model_ild_per_slate.numpy()
        }
    
    
    def get_cold_start_movies(self, pos_test_df: pl.DataFrame) -> Tuple[tf.Tensor, tf.Tensor]:
        
        pos_test_df = self.join_df_to_movie_tiers(pos_test_df)
        movie_test_counts = pos_test_df.group_by("movie_id").agg(pl.len().alias("test_interaction_count"))
        candidate_pool = movie_test_counts.filter(pl.col("test_interaction_count") >= 5)
        
        pos_test_df = candidate_pool.join(pos_test_df, on="movie_id", how="left")
        
        num_per_tier = 167  # total = 501
        candidate_pool_df = (pos_test_df.sample(fraction=1.0, seed=42, shuffle=True)
            .group_by("movie_tier").head(num_per_tier))
            
        original_movie_ids = candidate_pool_df["movie_id"].unique().to_numpy()
        
        id0 = TestAnalysis.movie_id_range_incl[-1] + 1
        new_movie_ids = np.array([i for i in range(id0, id0 + len(original_movie_ids))])
        
        n_movie_ids = len(new_movie_ids)
        zeroes = np.zeros(n_movie_ids, dtype=np.int64)
        
        original_movie_ids = np.expand_dims(original_movie_ids, axis=1)
        new_movie_ids = np.expand_dims(new_movie_ids, axis=1)
        zeroes = np.expand_dims(zeroes, axis=1)
        
        original_inputs = TestAnalysis.movie_data.get_movie(original_movie_ids)
        new_inputs = original_inputs.copy()
        # Use new_movie_ids (> max catalog movie_id) so MovieModel clips them to the untrained OOV slot (n_movies - 1),
        # rather than 0 which clips to index 0 (movie_id = MOVIE_OFFSET = 6041)
        new_inputs["movie_id"] = tf.constant(new_movie_ids, dtype=tf.int64)
        
        # original_embeddings : tf.Tensor = _create_movie_embeddings_batch(original_inputs)
        new_embeddings: tf.Tensor = _create_movie_embeddings_batch(new_inputs)
        
        new_movie_ids = tf.convert_to_tensor(new_movie_ids, dtype=tf.int32)
        
        return (new_movie_ids, new_embeddings)
    
    def compare_retrieval_runs(
            self,
            baseline: Dict[str, Any],
            experiment: Dict[str, Any],
            top_k: int,
            num_catalog_movies: int,
            baseline_name: str = "test_retrieval",
            exp_name: str = "cold_start_retrieval"
    ) -> Dict[str, Any]:
        
        metrics_to_compare = [
            f"recall@{top_k}_overall",
            f"recall_movie_tier0@{top_k}",
            f"recall_movie_tier1@{top_k}",
            f"recall_movie_tier2@{top_k}",
            f"retrieved_share_tier0@{top_k}",
            f"retrieved_share_tier1@{top_k}",
            f"retrieved_share_tier2@{top_k}"
        ]
        
        ##### compare to random too #####
        # N = num_catalog_movies
        # K = top_k
        # G = number of postive ground truth for a given user in the dataset.
        # we draw K items uniformly at random from N without replacement
        # X = number of hits = size of intersection of K with G
        # X ~ Hypergeometric}(N, G, K)
        #  E[X] = K * G / N
        #  recall@K = X / G
        # E[recall@K] = E[X/G] = E[X] / G = K / N
        expected_random_recall = min(1.0, top_k / max(1, num_catalog_movies))
        # Add a 5% relative buffer to account for statistical noise / float math
        random_threshold = expected_random_recall * 1.05
        
        diffs = {}
        for metric in metrics_to_compare:
            base_val = baseline.get(metric, 0.0)
            exp_val = experiment.get(metric, 0.0)
            
            abs_diff = exp_val - base_val
            rel_diff_pct = ((exp_val - base_val) / base_val * 100) if base_val != 0 else (
                100.0 if exp_val > 0 else 0.0)
            
            diffs[metric] = {
                f"{baseline_name}": round(base_val, 6),
                f"{exp_name}": round(exp_val, 6),
                "abs_change": round(abs_diff, 6),
                "rel_change_pct": f"{rel_diff_pct:+.2f}%"
            }
        
        # --- Automated Conclusion Generator Rules ---
        conclusions = []
        
        base_recall = baseline.get(f"recall@{top_k}_overall", 0.0)
        exp_recall = experiment.get(f"recall@{top_k}_overall", 0.0)
        
        # 1. Baseline Sanity Checks
        if base_recall <= random_threshold:
            conclusions.append(
                f"[INVALID BASELINE]: {baseline_name} overall recall ({base_recall * 100:.2f}%) is statistically indistinguishable "
                f"from random chance ({expected_random_recall * 100:.2f}%). Check model training or evaluation logic."
            )
        elif base_recall < 0.05:
            conclusions.append(
                f"[LOW BASELINE RECALL]: Baseline recall ({base_recall * 100:.2f}%) is low at K={top_k}. "
                f"Consider evaluating at a larger K (e.g., K=100 or K=200) to avoid candidate bottlenecking."
            )
        
        drop_pct = None
        # 2. Experiment Random Check & Overall Drop
        if exp_recall > 0 and exp_recall <= random_threshold:
            conclusions.append(
                f"[CRITICAL FAILURE - {exp_name}]: Overall recall ({exp_recall * 100:.2f}%) has degraded entirely to "
                f"random chance ({expected_random_recall * 100:.2f}%). The model lacks any predictive signal for these users/items."
            )
        elif base_recall > 0 and exp_recall < base_recall:
            drop_pct = ((base_recall - exp_recall) / base_recall) * 100
            if drop_pct >= 50.0:
                conclusions.append(
                    f"[SEVERE COLD-START PENALTY]: {exp_name} overall recall dropped by {drop_pct:.1f}% relative "
                    f"to {baseline_name} ({base_recall * 100:.2f}% -> {exp_recall * 100:.2f}%). "
                    f"The item tower overfits to interaction IDs rather than generalizing via metadata."
                )
            else:
                conclusions.append(
                    f"[PERFORMANCE DROP]: {exp_name} overall recall dropped by {drop_pct:.1f}% relative to {baseline_name}. "
                    f"Metadata partially compensates for missing interactions, but performance still degrades."
                )
        
        # 3. Per-Tier Recall & Precision Deficit Checks
        for tier in [0, 1, 2]:
            t_base = baseline.get(f"recall_movie_tier{tier}@{top_k}", 0.0)
            t_exp = experiment.get(f"recall_movie_tier{tier}@{top_k}", 0.0)
            t_share = experiment.get(f"retrieved_share_tier{tier}@{top_k}",
                0.0)
            
            is_exp_random = t_exp <= random_threshold
            
            if t_base > random_threshold and is_exp_random:
                conclusions.append(
                    f"[TIER {tier} COLLAPSE]: Tier {tier} recall dropped from {t_base * 100:.2f}% to {t_exp * 100:.2f}%. "
                    f"Without interaction history, the model's ability to target Tier {tier} falls to random chance."
                )
            elif t_base == 0.0 and t_exp == 0.0:
                conclusions.append(
                    f"[TIER {tier} MISALIGNMENT]: Tier {tier} recall is 0.0% in both runs. "
                    f"The model entirely fails to surface relevant Tier {tier} items into top-{top_k} candidates."
                )
            
            if t_share >= 0.15 and is_exp_random:
                conclusions.append(
                    f"[PRECISION DEFICIT - TIER {tier}]: Model allocates {t_share * 100:.1f}% of top-{top_k} candidate slots "
                    f"to Tier {tier} items, yet recall ({t_exp * 100:.2f}%) is at/below random chance. It retrieves the WRONG items."
                )
        
        # 4. Catalog Slate Composition Summary
        t0_share = experiment.get(f"retrieved_share_tier0@{top_k}", 0.0)
        t1_share = experiment.get(f"retrieved_share_tier1@{top_k}", 0.0)
        t2_share = experiment.get(f"retrieved_share_tier2@{top_k}", 0.0)
        
        conclusions.append(
            f"[SLATE COMPOSITION - {exp_name}]: "
            f"Top-{top_k} candidate slates consist of {t0_share * 100:.1f}% Tier 0, {t1_share * 100:.1f}% Tier 1, and {t2_share * 100:.1f}% Tier 2 items."
        )
        
        TestAnalysis.summary_output_conclusions_dict['cold_start_embeddings'] = conclusions
        
        return {
            f"expected_random_recall@{top_k}": round(expected_random_recall, 6),
            "comparison_metrics": diffs,
            f"recall_cold_start_percnt_drop" : drop_pct, #can be None
            "automated_conclusions": conclusions
        }
    
    def evaluate_retrieval_with_tier_share(self,
            user_ids: np.ndarray,  # shape: (325,)
            neighbors: np.ndarray,  # shape: (325, top_k)
            movie_offset: int,  # TestAnalysis.MOVIE_OFFSET
            ground_truth_df: pl.DataFrame, # columns: user_id, movie_id, rating, timestamp, tier
            top_k: int = 20
    ) -> dict:
        
        user_ids = user_ids.squeeze()
        
        adjusted_neighbors = neighbors + movie_offset  # shape: (325, top_k)
        
        # Map catalog movie_id -> tier for checking retrieved candidate distribution
        catalog_tier_map = dict(
            TestAnalysis.movie_tiers_df.select(["movie_id", "movie_tier"]).rows()
        )
        
        # Group ground truth items and their tiers per user
        gt_grouped = (ground_truth_df.group_by("user_id")
            .agg([
                pl.col("movie_id").alias("gt_movies"),
                pl.col("movie_tier").alias("gt_tiers")
            ])
        )
        # there's only one movie in gt_movies for almost all users, except a dozen or so have 2 movies in them
        
        gt_user_data = {
            row[0]: dict(zip(row[1], row[2])) for row in gt_grouped.select(["user_id", "gt_movies", "gt_tiers"]).rows()
        }
        
        tier_recalls = {0: [], 1: [], 2: []}
        retrieved_tier_counts = {0: 0, 1: 0, 2: 0}
        total_retrieved_items = 0
        all_recalls = []
        
        for i, u_id in enumerate(user_ids):
            retrieved_list = adjusted_neighbors[i, :top_k]
            retrieved_set = set(retrieved_list)
            
            # Track Retrieved Candidate Composition (What tiers were retrieved?)
            for m_id in retrieved_list:
                m_tier = catalog_tier_map.get(m_id, 2) #default to 2 for cold-start movies
                if m_tier in retrieved_tier_counts:
                    retrieved_tier_counts[m_tier] += 1
                total_retrieved_items += 1
            
            # Track Recall if user has ground truth in test set
            if u_id not in gt_user_data:
                continue
            
            movie_tier_dict = gt_user_data[u_id]
            if not movie_tier_dict:
                continue
            
            gt_set = set(movie_tier_dict.keys())
            all_recalls.append(
                len(retrieved_set.intersection(gt_set)) / len(gt_set))
            
            # Bin ground truth items by tier
            user_gt_by_tier = {0: set(), 1: set(), 2: set()}
            for m_id, tier in movie_tier_dict.items():
                if tier in user_gt_by_tier:
                    user_gt_by_tier[tier].add(m_id)
            
            # Calculate tier recall
            for tier, tier_gt_set in user_gt_by_tier.items():
                if len(tier_gt_set) == 0:
                    continue
                hits = len(retrieved_set.intersection(tier_gt_set))
                tier_recalls[tier].append(hits / len(tier_gt_set))
        
        # Aggregate metrics
        results = {
            f"recall@{top_k}_overall": float(
                np.mean(all_recalls)) if all_recalls else 0.0,
            f"recall_movie_tier0@{top_k}": float(np.mean(tier_recalls[0])) if
            tier_recalls[0] else 0.0,
            f"recall_movie_tier1@{top_k}": float(np.mean(tier_recalls[1])) if
            tier_recalls[1] else 0.0,
            f"recall_movie_tier2@{top_k}": float(np.mean(tier_recalls[2])) if
            tier_recalls[2] else 0.0,
            # Proportion of retrieved top-K candidates that belong to each tier
            f"retrieved_share_tier0@{top_k}": retrieved_tier_counts[0] / max(1,
                total_retrieved_items),
            f"retrieved_share_tier1@{top_k}": retrieved_tier_counts[1] / max(1,
                total_retrieved_items),
            f"retrieved_share_tier2@{top_k}": retrieved_tier_counts[2] / max(1,
                total_retrieved_items),
        }
        
        return results
    
    def create_movie_embeddings_batch(self, movie_ids: Union[tf.Tensor, np.ndarray]):
        
        """
        inputs = \
            {'movie_id': tf.constant([[6041], [6042], [6043]], dtype=tf.int64),
                'genres': tf.constant([["Animation|Children's|Comedy"], ["Adventure|Children's|Fantasy"], ["Comedy|Romance"]], dtype=tf.string),
        
        """
        inputs = TestAnalysis.movie_data.get_movie(movie_ids)
        return _create_movie_embeddings_batch(inputs)
    
    def create_user_embeddings_batch(self, user_ids: Union[tf.Tensor, np.ndarray],
            timestamps:Union[tf.Tensor, np.ndarray]) -> tf.Tensor:
        """
        inputs = \
            {'user_id': tf.constant([[1], [2], [3]], dtype=tf.int64),
                'gender': tf.constant([["F"], ["M"], ["M"]], dtype=tf.string),
                'age': tf.constant([[1], [56], [25]], dtype=tf.int64),
                'occupation': tf.constant([[10], [16], [15]], dtype=tf.int64),
                'timestamp': tf.constant([[ts], [ts], [ts]], dtype=tf.int64),
            }
        """
        inputs = TestAnalysis.user_data.get_user(user_id = user_ids, timestamp = timestamps)
        infer_for_dict = TestAnalysis.loaded_user_movie_model.signatures["serving_query_dict"]
        output_keyword = list(infer_for_dict.structured_outputs.keys())[0]
        embeddings_list = infer_for_dict(
            age=inputs['age'],
            gender=inputs['gender'],
            occupation=inputs['occupation'],
            timestamp=inputs['timestamp'],
            user_id=inputs['user_id']
        )[output_keyword]
        return embeddings_list
    
    def _parse_tfrecord(self, proto, em_feature_spec):
        parsed = tf.io.parse_single_example(proto, em_feature_spec)
        #return parsed['movie_id'], parsed['embedding']
        return parsed
    
    def read_embeddings_to_tfds(self, file_path:str, em_feature_spec:Dict[str, tf.io.FixedLenFeature]) -> TFRecordDataset:
        _ct = "GZIP" if file_path.endswith(".gz") else None
        dataset = tf.data.TFRecordDataset(file_path, compression_type=_ct)
        dataset = dataset.map(lambda proto : self._parse_tfrecord(proto, em_feature_spec))
        return dataset
    
    def read_embeddings_to_polars(self, file_path:str, em_feature_spec:Dict[str, tf.io.FixedLenFeature]) -> pl.DataFrame:
        dataset = self.read_embeddings_to_tfds(file_path, em_feature_spec)
        emb_df = pl.from_dicts(list(dataset.as_numpy_iterator()))
        return emb_df
    
    def join_df_to_movie_tiers(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.join(TestAnalysis.movie_tiers_df, on='movie_id', how='left')
        
    def plot_embeddings_umap_tsne(self, joined_df, outdir: str, file_tag: str,
        stratified_key: str = "tier"):
        
        # Convert to NumPy arrays for UMAP
        X = np.array(joined_df.get_column("embedding").to_list())
        y = joined_df.get_column(stratified_key).to_numpy()
        
        print(f'length of {file_tag} is {len(X)}')
        
        # Apply UMAP and plot
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42, n_jobs=1)
        embedding_2d = reducer.fit_transform(X)
        
        plt.figure(figsize=(10, 8))
        sns.scatterplot(
            x=embedding_2d[:, 0],
            y=embedding_2d[:, 1],
            hue=y,
            palette="tab10",
            s=15,
            alpha=0.8
        )
        plt.title(f"Embedding UMAP Projection by {stratified_key} ({file_tag})")
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        plt.legend(title=f"{stratified_key}", bbox_to_anchor=(1.05, 1), loc=2,
            borderaxespad=0.)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"umap_{file_tag}_{stratified_key}.png"),
            dpi=300, bbox_inches="tight")
        # plt.show()
        plt.close()
        
        # t-SNE plot
        reducer_tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embedding_tsne = reducer_tsne.fit_transform(X)
        
        plt.figure(figsize=(10, 8))
        sns.scatterplot(
            x=embedding_tsne[:, 0],
            y=embedding_tsne[:, 1],
            hue=y,
            palette="tab10",
            s=15,
            alpha=0.8
        )
        plt.title(f"Embedding t-SNE Projection by {stratified_key} ({file_tag})")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.legend(title=f"{stratified_key}", bbox_to_anchor=(1.05, 1), loc=2,
            borderaxespad=0.)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"tsne_{file_tag}_{stratified_key}.png"),
            dpi=300, bbox_inches="tight")
        # plt.show()
        plt.close()

    def get_positive_ratings_history(self) -> pl.DataFrame:
        r = []
        for file_path in TestAnalysis.ratings_dict["positive_history"]:
            ratings_df = read_ratings_to_df(file_path)
            r.append(ratings_df)
        return pl.concat(r)
    
    def plot_lorenz_curve(self, lorenz_array, top_k, gini_score, out_file_path):
        """Plots the Lorenz curve of recommendation distribution."""
        if not lorenz_array:
            print("No Lorenz data to plot.")
            return
        
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(8, 6))
        
        # Anchor the curve at (0,0)
        x_vals = [0] + list(range(1, 101))
        y_vals = [0.0] + lorenz_array
        
        # Line of Perfect Equality (y = x)
        perfect_equality = [x / 100.0 for x in x_vals]
        plt.plot(x_vals, perfect_equality, linestyle='--', color='gray',
            label='Perfect Equality (Gini=0.0)')
        
        # Model's Lorenz Curve
        plt.plot(x_vals, y_vals, color='darkblue', linewidth=2.5,
            label=f'Model @ k={top_k} (Gini={gini_score:.2f})')
        
        # Shade the Gini Area
        plt.fill_between(x_vals, perfect_equality, y_vals, color='darkblue',
            alpha=0.1)
        
        plt.title(f"Catalog Recommendation Inequality (k={top_k})",
            fontsize=14, pad=15)
        plt.xlabel(                                                                                                                                       "Cumulative % of Movie Catalog (Least to Most Popular)",
            fontsize=11)
        plt.ylabel("Cumulative % of Recommendation Slots", fontsize=11)
        plt.xlim(0, 100)
        plt.ylim(0, 1.0)
        plt.legend(loc="upper left")
        plt.tight_layout()
        
        plt.savefig(out_file_path, dpi=300, bbox_inches="tight")
        plt.close()
    
def get_largest_subdir_version(parent_dir: str) -> str:
    parent_dir = Path(parent_dir)
    numeric_dirs = [d for d in parent_dir.iterdir() if
        d.is_dir() and d.name.isdigit()]
    if numeric_dirs:
        largest_dir = max(numeric_dirs, key=lambda d: int(d.name))
        return largest_dir.name
    raise ValueError("No numeric subdirectories found.")

def read_ratings_to_df(file_path:str, batch_size:int=2048) -> pl.DataFrame:
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
    
def create_movie_catalog_embeddings() -> Tuple[tf.Tensor, tf.Tensor]:
    
    df = pl.read_parquet(TestAnalysis.movies_path)
    df = df.sort('movie_id')
    movie_ids = np.expand_dims(df['movie_id'].to_numpy(), axis=1)
    movie_ids = tf.constant(movie_ids, name='movie_id', dtype=tf.int64)
    
    inputs = TestAnalysis.movie_data.get_movie(movie_ids)
    embeddings: tf.Tensor = _create_movie_embeddings_batch(inputs)
    
    return movie_ids, embeddings

def read_model_assets_hparams(a_user_movie_models_dir) -> dict[str, Any]:
    file_path = os.path.join(a_user_movie_models_dir,
        "assets.extra/hyperparameters.json")
    with open(file_path, 'r', encoding='utf-8') as f:
        hyperparameters = json.load(f)['values']
        return hyperparameters
    
def _create_movie_embeddings_batch(inputs: Dict) -> tf.Tensor:
    """
    inputs = \
        {'movie_id': tf.constant([[6041], [6042], [6043]], dtype=tf.int64),
            'genres': tf.constant([["Animation|Children's|Comedy"], ["Adventure|Children's|Fantasy"], ["Comedy|Romance"]], dtype=tf.string),

    """
    infer_for_dict = TestAnalysis.loaded_user_movie_model.signatures[
        "serving_candidate_dict"]
    output_keyword = list(infer_for_dict.structured_outputs.keys())[0]
    embeddings_list = infer_for_dict(
        movie_id=inputs['movie_id'],
        genres=inputs['genres'])
    embeddings_list = embeddings_list[output_keyword]
    return embeddings_list

if __name__ == '__main__':
    unittest.main()

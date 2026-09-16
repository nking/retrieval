"""
scripts to analysis the trained models from the recommender_systems project.

some of the analysis is in rust in the ranker project directory inference_src and will be ported
to python here to have it all in one place
"""
import json
import os
import unittest
from typing import Any, Dict, Union, Tuple, List
import polars as pl
import msgpack
from array_record.python import array_record_module
from gto.constants import fullname
from tensorflow.data import TFRecordDataset
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import tensorflow as tf
import numpy as np
import json
from datetime import datetime

from helper import get_project_dir, get_bin_dir
from movie_lens_retrieval.MovieData import MovieData
from movie_lens_retrieval.Retriever import Retriever
from movie_lens_retrieval.UserData import UserData
from scann.scann_ops.py.scann_ops_pybind import ScannSearcher

class TestAnalysis(unittest.TestCase):
    def setUp(self):
        saved_models_dir = os.path.join(get_project_dir(),
            "src/main/resources/serving_models")
        self.user_movie_models_dir = os.path.join(saved_models_dir,
            "user_movie_model")
        
        test_res_dir = os.path.join(get_project_dir(), "src/test/resources/data")
        
        self.cold_start_path = os.path.join(test_res_dir, "cold_start_movies.txt")
        
        self.movie_emb_path = os.path.join(test_res_dir, "movie_emb_inp/movie_emb-00000-of-00001.tfrecord.gz")
        self.user_emb_path = os.path.join(test_res_dir, "user_emb_inp/user_emb-00000-of-00001.tfrecord.gz")
        
        self.users_path = os.path.join(test_res_dir, "users/users.parquet")
        self.movies_path = os.path.join(test_res_dir, "movies/movies.parquet")
        
        movie_tiers_path = os.path.join(test_res_dir, "movie_tiers.json")
        self.movie_tiers_df = pl.read_ndjson(movie_tiers_path)
        
        self.model_dict = self.read_model_assets_hparams(
            self.user_movie_models_dir)
        self.MOVIE_OFFSET : int = self.model_dict['n_users'] + 1
        self.embed_dim = json.loads(self.model_dict['layer_sizes'])[-1]
        self.user_id_range_incl = [1, self.model_dict['n_users']]
        self.movie_id_range_incl = [self.MOVIE_OFFSET, self.MOVIE_OFFSET + self.model_dict['n_movies']]
        
        self.ratings_dict = {
            "full_history" : [
                    os.path.join(test_res_dir, "ratings_train.array_record"),
                    os.path.join(test_res_dir, "ratings_val.array_record")
                ],
            "positive_history" : [
                    os.path.join(test_res_dir, "ratings_train_liked.array_record"),
                    os.path.join(test_res_dir, "ratings_val_liked.array_record")
                ],
            "positive_test": os.path.join(test_res_dir, "ratings_test_liked.array_record"),
        }
        
        self.emb_movie_feature_spec = {
            "movie_id": tf.io.FixedLenFeature(shape=[], dtype=tf.int64,
                default_value=None),
            "embedding": tf.io.FixedLenFeature(shape=[self.embed_dim],
                dtype=tf.float32)
        }
        
        self.emb_user_feature_spec = {
            "user_id": tf.io.FixedLenFeature(shape=[], dtype=tf.int64,
                default_value=None),
            "embedding": tf.io.FixedLenFeature(shape=[self.embed_dim],
                dtype=tf.float32)
        }
        
        self.user_feature_spec = {
            "user_id": tf.io.FixedLenFeature([], tf.int64),
            "gender": tf.io.FixedLenFeature([], tf.string),
            "age": tf.io.FixedLenFeature([], tf.int64),
            "occupation": tf.io.FixedLenFeature([], tf.int64),
            "timestamp": tf.io.FixedLenFeature([], tf.int64),
        }
        
        self.movie_feature_spec = {
            "movie_id": tf.io.FixedLenFeature([], tf.int64),
            "genres": tf.io.FixedLenFeature([], tf.string)}
        
        self.user_data = UserData(self.users_path)
        self.movie_data = MovieData(self.movies_path, self.MOVIE_OFFSET)
    
        self.max_k = 20
    
    def test_plot_movie_embeddings(self):
        
        emb_movies_df = self.read_embeddings_to_polars(self.movie_emb_path, self.emb_movie_feature_spec)
        emb_movies_df = self.join_df_to_movie_tiers(emb_movies_df)
        
        outdir = os.path.join(get_bin_dir(), "embedding_plots")
        os.makedirs(outdir, exist_ok=True)
        
        self.plot_embeddings_umap_tsne(emb_movies_df, outdir, "all_movies")
        
    def test_cold_start_distance(self):
        self.simulate_cold_start_embeddings()
        
    def test_intra_list_diversity(self):
        
        output_file_path = os.path.join(get_bin_dir(), "intralist_diversity.json")

        pos_test_df = self.read_ratings_to_df(self.ratings_dict["positive_test"])
        first_interactions_df = pos_test_df.group_by("user_id").agg(pl.col("timestamp").min())
        user_ids = first_interactions_df["user_id"].to_numpy()
        user_ids = np.expand_dims(user_ids, axis=1)
        timestamps = first_interactions_df["timestamp"].to_numpy()
        timestamps = np.expand_dims(timestamps, axis=1)
        user_embeddings = self.create_user_embeddings_batch(user_ids, timestamps) #tf.Tensor shape (5096, 32)
        
        #make indexer for movie embeddings
        top_k = 100
        #(tf.Tensor, ScannSearcher)
        (movie_embeddings, indexer)  = self.create_movie_indexer(top_k)
        
        neighbors, distances = indexer.search_batched(user_embeddings)
        #neighbors += self.MOVIE_OFFSET
        if isinstance(neighbors, np.ndarray):
            neighbors = tf.convert_to_tensor(neighbors, dtype=tf.int32)
        # --- SAFETY CHECK: Check bounds ---
        max_index = tf.reduce_max(neighbors)
        min_index = tf.reduce_min(neighbors)
        catalog_size = tf.shape(movie_embeddings)[0]
        
        print(f"Catalog size: {catalog_size.numpy()}")
        print(f"Max index in neighbors: {max_index.numpy()}")
        print(f"Min index in neighbors: {min_index.numpy()}")
        
        res = self.calculate_batched_intra_list_diversity(neighbors, movie_embeddings)
        print(f"Average Intra-List Diversity @ {top_k}\n: {json.dumps(res, indent=4)}")
        
        with open(output_file_path, "w") as f:
            json.dump(res, f, indent=4)
            
    
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
        and returns an automated text analysis.

        Args:
            neighbors: 2D tensor of retrieved item indices from ScaNN.
            movie_embeddings: 2D tensor of all item embeddings.
            num_random_samples: Number of random user slates used to compute random ILD.

        Returns:
            Dict with keys: 'model_ild', 'random_ild', 'diversity_ratio', and 'analysis'.
        """
        num_users = tf.shape(neighbors)[0]
        top_k = tf.shape(neighbors)[1]
        num_catalog_movies = tf.shape(movie_embeddings)[0]
        
        # --- Helper: Vectorized ILD calculation for any batch of slates ---
        def _compute_ild(slate_indices: tf.Tensor) -> tf.Tensor:
            # Fetch embeddings: (batch_size, top_k, 32)
            slate_embeddings = tf.gather(movie_embeddings, slate_indices)
            
            # Batch Matrix Multiplication: (batch_size, top_k, top_k)
            sim_matrices = tf.matmul(slate_embeddings, slate_embeddings, transpose_b=True)
            dist_matrices = 1.0 - sim_matrices
            
            # Zero out self-distances on the diagonal
            n_slates = tf.shape(slate_indices)[0]
            k = tf.shape(slate_indices)[1]
            zeros_diagonal = tf.zeros((n_slates, k), dtype=dist_matrices.dtype)
            dist_matrices = tf.linalg.set_diag(dist_matrices, zeros_diagonal)
            
            # Pairwise distance sum / total possible pairs
            sum_dists_per_slate = tf.reduce_sum(dist_matrices, axis=[1, 2])
            num_pairs = tf.cast(k * (k - 1), dtype=dist_matrices.dtype)
            return tf.reduce_mean(sum_dists_per_slate / num_pairs)
        
        # Compute Model ILD
        model_ild_tf = _compute_ild(neighbors)
        model_ild = float(model_ild_tf.numpy())
        
        # Estimate Empirical Random Baseline ILD
        random_indices = tf.random.uniform(
            shape=(num_random_samples, top_k),
            minval=0,
            maxval=num_catalog_movies,
            dtype=tf.int32
        )
        random_ild_tf = _compute_ild(random_indices)
        random_ild = float(random_ild_tf.numpy())
        
        # Compute Diversity Ratio (% of random catalog diversity retained)
        diversity_ratio = model_ild / random_ild if random_ild > 0 else 0.0
        
        # Automated Text Analysis
        analysis = []
        if diversity_ratio < 0.20:
            analysis.append(
                f"[EXTREME CLUSTERING]: Model ILD ({model_ild:.4f}) retains only {diversity_ratio * 100:.1f}% of random catalog diversity. "
                f"Risk of severe candidate bottlenecking into a single sub-genre."
            )
        elif diversity_ratio < 0.50:
            analysis.append(
                f"[FOCUSED CANDIDATE POOL]: Model ILD ({model_ild:.4f}) retains {diversity_ratio * 100:.1f}% of random catalog diversity. "
                f"This indicates strong, coherent cluster targeting around user preferences."
            )
        else:
            analysis.append(
                f"[BROAD CANDIDATE POOL]: Model ILD ({model_ild:.4f}) retains {diversity_ratio * 100:.1f}% of random catalog diversity. "
                f"Candidates span multiple distinct semantic regions in the embedding space."
            )
        
        return {
            "model_ild": round(model_ild, 6),
            "random_ild": round(random_ild, 6),
            "diversity_ratio": round(diversity_ratio, 4),
            "analysis": analysis
        }
    
    def simulate_cold_start_embeddings(self):
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
        output_file_path = os.path.join(get_bin_dir(), "cold-start-recalls.json")
        
        pos_test_df = self.read_ratings_to_df(self.ratings_dict["positive_test"])
        #[user_id, movie_id, rating, timestamp, tier]
        pos_test_df = self.join_df_to_movie_tiers(pos_test_df)
        
        movie_test_counts = pos_test_df.group_by("movie_id").agg(pl.len().alias("test_interaction_count"))
        candidate_pool = movie_test_counts.filter(pl.col("test_interaction_count") >= 5)
        
        pos_test_df = candidate_pool.join(pos_test_df, on="movie_id", how="left")
        
        num_per_tier = 167 #total = 501
        candidate_pool_df = (pos_test_df
            .sample(fraction=1.0, seed=42, shuffle=True)
            # Group by tier and take the top N from each group
            .group_by("tier").head(num_per_tier)
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
        
        id0 = self.movie_id_range_incl[-1] + 1
        new_movie_ids = np.array([i for i in range(id0, id0 + len(original_movie_ids))])
        
        original_movie_ids = np.expand_dims(original_movie_ids, axis=1)
        new_movie_ids = np.expand_dims(new_movie_ids, axis=1)
        
        original_inputs = self.movie_data.get_movie(original_movie_ids)
        new_inputs = original_inputs.copy()
        new_inputs["movie_id"] = tf.constant(new_movie_ids)
        
        original_embeddings : tf.Tensor = self._create_movie_embeddings_batch(original_inputs)
        new_embeddings : tf.Tensor = self._create_movie_embeddings_batch(new_inputs)
        
        full_movie_emb_ds : TFRecordDataset = self.read_embeddings_to_tfds(file_path=self.movie_emb_path,
            em_feature_spec=self.emb_movie_feature_spec)
        full_movie_ids = []
        full_movie_embeddings = []
        for batch in full_movie_emb_ds.batch(1024).as_numpy_iterator():
            full_movie_ids.append(batch['movie_id'])  #a numpy array
            full_movie_embeddings.append(batch['embedding'])  # a numpy array
        full_movie_ids = np.concatenate(full_movie_ids, axis=0)
        full_movie_embeddings = np.concatenate(full_movie_embeddings, axis=0)
       
        top_k = 100
        #build indexes
        orig_indexer = Retriever.build_scann_searcher(embeddings=full_movie_embeddings, top_k=top_k)
        
        #for each new_movie_ids, replace with embedding for new_movie_ids
        for i, m_id in enumerate(original_movie_ids):
            idx = m_id - self.MOVIE_OFFSET
            assert(full_movie_ids[idx] == m_id)
            full_movie_embeddings[idx] = new_embeddings[i]
        
        new_indexer = Retriever.build_scann_searcher(embeddings=full_movie_embeddings, top_k=top_k)
    
        #retrieve
        user_embeddings = self.create_user_embeddings_batch(user_ids, timestamps)
        
        orig_neighbors, orig_distances = orig_indexer.search_batched(user_embeddings)
        new_neighbors, new_distances = new_indexer.search_batched(user_embeddings)
        
        orig_results = self.evaluate_retrieval_with_tier_share(user_ids, orig_neighbors, self.MOVIE_OFFSET, candidate_pool_df, self.movie_tiers_df, top_k=top_k)
        new_results = self.evaluate_retrieval_with_tier_share(user_ids, new_neighbors, self.MOVIE_OFFSET, candidate_pool_df, self.movie_tiers_df, top_k=top_k)

        print("results on test set:\n", json.dumps(orig_results, indent=4))
        print("results on test set cold start:\n", json.dumps(new_results, indent=4))
        
        res = self.compare_retrieval_runs(orig_results, new_results, top_k, self.model_dict["n_movies"], "test set", "cold-start set")
        print("comparisons:\n", json.dumps(res, indent=4))
        
        with open(output_file_path, "w") as f:
            json.dump(res, f, indent=4)
    
    def create_movie_indexer(self, top_k: int) -> Tuple[tf.Tensor, ScannSearcher]:
        full_movie_emb_ds: TFRecordDataset = self.read_embeddings_to_tfds(
            file_path=self.movie_emb_path, em_feature_spec=self.emb_movie_feature_spec)
        
        full_movie_ids = []
        full_movie_embeddings = []
        for batch in full_movie_emb_ds.batch(1024):
            full_movie_ids.append(batch['movie_id'])  # a numpy array
            full_movie_embeddings.append(batch['embedding'])  # a numpy array
        full_movie_ids = tf.concat(full_movie_ids, axis=0)
        full_movie_embeddings = tf.concat(full_movie_embeddings, axis=0)
        
        #just in case they are no longer in order of increasing movie_id
        sorted_indices = tf.argsort(full_movie_ids, direction='ASCENDING')
        full_movie_ids = tf.gather(full_movie_ids, sorted_indices)
        full_movie_embeddings = tf.gather(full_movie_embeddings, sorted_indices)
        
        indexer = Retriever.build_scann_searcher(embeddings=full_movie_embeddings, top_k=top_k)
        return (full_movie_embeddings, indexer)
        
    def compare_retrieval_runs(
            self,
            baseline: Dict[str, Any],
            experiment: Dict[str, Any],
            top_k: int,
            num_catalog_movies: int,
            baseline_name: str = "Standard Test",
            exp_name: str = "Cold Start"
    ) -> Dict[str, Any]:
        
        metrics_to_compare = [
            f"overall_recall@{top_k}",
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
        
        base_recall = baseline.get(f"overall_recall@{top_k}", 0.0)
        exp_recall = experiment.get(f"overall_recall@{top_k}", 0.0)
        
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
        
        return {
            "expected_random_recall": round(expected_random_recall, 6),
            "comparison_metrics": diffs,
            "automated_conclusions": conclusions
        }
    
    def evaluate_retrieval_with_tier_share(self,
            user_ids: np.ndarray,  # shape: (325,)
            neighbors: np.ndarray,  # shape: (325, top_k)
            movie_offset: int,  # self.MOVIE_OFFSET
            ground_truth_df: pl.DataFrame, # columns: user_id, movie_id, rating, timestamp, tier
            movie_tiers_df: pl.DataFrame, # columns: movie_id, tier (for catalog-wide lookup)
            top_k: int = 20
    ) -> dict:
        
        user_ids = user_ids.squeeze()
        
        adjusted_neighbors = neighbors + movie_offset  # shape: (325, top_k)
        
        # Map catalog movie_id -> tier for checking retrieved candidate distribution
        catalog_tier_map = dict(
            movie_tiers_df.select(["movie_id", "tier"]).rows()
        )
        
        # Group ground truth items and their tiers per user
        gt_grouped = (
            ground_truth_df
            .group_by("user_id")
            .agg([
                pl.col("movie_id").alias("gt_movies"),
                pl.col("tier").alias("gt_tiers")
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
                m_tier = catalog_tier_map.get(m_id)
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
            f"overall_recall@{top_k}": float(
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
    
    def _create_movie_embeddings_batch(self, inputs : Dict) -> tf.Tensor:
        
        """
        inputs = \
            {'movie_id': tf.constant([[6041], [6042], [6043]], dtype=tf.int64),
                'genres': tf.constant([["Animation|Children's|Comedy"], ["Adventure|Children's|Fantasy"], ["Comedy|Romance"]], dtype=tf.string),

        """
        self.loaded_user_movie_model = tf.saved_model.load(self.user_movie_models_dir)
        infer_for_dict = self.loaded_user_movie_model.signatures["serving_candidate_dict"]
        output_keyword = list(infer_for_dict.structured_outputs.keys())[0]
        embeddings_list = infer_for_dict(
            movie_id=inputs['movie_id'],
            genres=inputs['genres'])
        embeddings_list = embeddings_list[output_keyword]
        return embeddings_list
    
    def create_movie_embeddings_batch(self, movie_ids: Union[tf.Tensor, np.ndarray]):
        
        """
        inputs = \
            {'movie_id': tf.constant([[6041], [6042], [6043]], dtype=tf.int64),
                'genres': tf.constant([["Animation|Children's|Comedy"], ["Adventure|Children's|Fantasy"], ["Comedy|Romance"]], dtype=tf.string),
        
        """
        #TODO: may need to reformat movie_ids into [[1],[2],...]
        inputs = self.movie_data.get_movie(movie_ids)
        return self._create_movie_embeddings_batch(inputs)
    
    def create_user_embeddings_batch(self, user_ids: Union[tf.Tensor, np.ndarray], timestamps:Union[tf.Tensor, np.ndarray]):
        """
        inputs = \
            {'user_id': tf.constant([[1], [2], [3]], dtype=tf.int64),
                'gender': tf.constant([["F"], ["M"], ["M"]], dtype=tf.string),
                'age': tf.constant([[1], [56], [25]], dtype=tf.int64),
                'occupation': tf.constant([[10], [16], [15]], dtype=tf.int64),
                'timestamp': tf.constant([[ts], [ts], [ts]], dtype=tf.int64),
            }
        """
        inputs = self.user_data.get_user(user_id = user_ids, timestamp = timestamps)
        self.loaded_user_movie_model = tf.saved_model.load(self.user_movie_models_dir)
        infer_for_dict = self.loaded_user_movie_model.signatures["serving_query_dict"]
        output_keyword = list(infer_for_dict.structured_outputs.keys())[0]
        embeddings_list = infer_for_dict(
            age=inputs['age'],
            gender=inputs['gender'],
            occupation=inputs['occupation'],
            timestamp=inputs['timestamp'],
            user_id=inputs['user_id']
        )[output_keyword]
        return embeddings_list
        
    def read_model_assets_hparams(self, user_movie_models_dir) -> dict[str, Any]:
        file_path = os.path.join(user_movie_models_dir, "assets.extra/hyperparameters.json")
        with open(file_path, 'r', encoding='utf-8') as f:
            hyperparameters = json.load(f)['values']
            return hyperparameters
        
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
        return df.join(self.movie_tiers_df, on='movie_id', how='left')
        
    def read_ratings_to_df(self, file_path:str, batch_size:int=2048) -> pl.DataFrame:
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
    
    def plot_embeddings_umap_tsne(self, joined_df, outdir: str, file_tag: str):
        # Convert to NumPy arrays for UMAP
        X = np.array(joined_df.get_column("embedding").to_list())
        y = joined_df.get_column("tier").to_numpy()
        
        print(f'length of {file_tag} is {len(X)}')
        
        # Apply UMAP and plot
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
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
        plt.title(f"Movie Embedding UMAP Projection by Tier ({file_tag})")
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        plt.legend(title="Tier", bbox_to_anchor=(1.05, 1), loc=2,
            borderaxespad=0.)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"umap_{file_tag}_tiers.png"),
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
        plt.title(f"Movie Embedding t-SNE Projection by Tier ({file_tag})")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.legend(title="Tier", bbox_to_anchor=(1.05, 1), loc=2,
            borderaxespad=0.)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"tsne_{file_tag}_tiers.png"),
            dpi=300, bbox_inches="tight")
        # plt.show()
        plt.close()
    
if __name__ == '__main__':
    unittest.main()

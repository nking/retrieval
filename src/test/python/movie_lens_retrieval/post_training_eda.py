"""
scripts to analysis the trained models from the recommender_systems project.

some of the analysis is in rust in the ranker project directory inference_src and will be ported
to python here to have it all in one place
"""
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
import numpy as np
import json
from datetime import datetime
from itertools import chain

from helper import get_project_dir, get_bin_dir, \
    get_random_user_and_first_timestamp_from_ratings, \
    get_stratified_user_and_first_timestamp_from_ratings, \
    get_user_and_first_timestamp_from_ratings

from movie_lens_retrieval.MovieData import MovieData, get_movie_tiers_df
from movie_lens_retrieval.Retriever import Retriever
from movie_lens_retrieval.UserData import UserData, get_user_tiers_df, get_user_tiers_from_df
from scann.scann_ops.py.scann_ops_pybind import ScannSearcher

class TestAnalysis(unittest.TestCase):
    def setUp(self):
        
        self.top_k = 100
        
        saved_models_dir = os.path.join(get_project_dir(),
            "src/main/resources/serving_models")
        self.user_movie_models_dir = os.path.join(saved_models_dir, "user_movie_model")
        
        test_res_dir = os.path.join(get_project_dir(), "src/test/resources/data")
        
        self.cold_start_path = os.path.join(test_res_dir, "cold_start_movies.txt")
        
        self.users_path = os.path.join(test_res_dir, "users/users.parquet")
        self.movies_path = os.path.join(test_res_dir, "movies/movies.parquet")
        
        self.model_dict = self.read_model_assets_hparams(
            self.user_movie_models_dir)
        self.MOVIE_OFFSET: int = self.model_dict['n_users'] + 1
        self.embed_dim = json.loads(self.model_dict['layer_sizes'])[-1]
        self.user_id_range_incl = [1, self.model_dict['n_users']]
        self.movie_id_range_incl = [self.MOVIE_OFFSET,
            self.MOVIE_OFFSET + self.model_dict['n_movies']]
        
        self.ratings_dict = {
            "full_history" : [
                    os.path.join(test_res_dir, "ratings_train.array_record"),
                    os.path.join(test_res_dir, "ratings_val.array_record")
                ],
            "positive_history" : [
                    os.path.join(test_res_dir, "ratings_train_liked.array_record"),
                    os.path.join(test_res_dir, "ratings_val_liked.array_record")
                ],
            "positive_train": os.path.join(test_res_dir, "ratings_train_liked.array_record"),
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
        
        #to make it easier to point script to a new model, will generate the movie embeddings live
        # tf.Tensors:
        self.movie_ids, self.movie_catalog_embeddings = self.create_movie_catalog_embeddings()
        
        train_history_df = self.read_ratings_to_df(self.ratings_dict['positive_train'])
        
        self.movie_tiers_df = get_movie_tiers_df(train_history_df)
        self.user_tiers_df = get_user_tiers_df(train_history_df)
        
        self.movie_indexer : ScannSearcher = Retriever.build_scann_searcher(embeddings=self.movie_catalog_embeddings,
            top_k=self.top_k)

    
    def test_coverage(self):
        """
        calculating item coverage as the number of unique movie ids recommended to users / size of movie catalog,
        then stratifying the results by movie_tier and user_tier.
        :return:
        """
        out_dir = os.path.join(get_bin_dir(), "coverage")
        shutil.rmtree(out_dir, ignore_errors=True)
        os.makedirs(out_dir, exist_ok=True)
        
        output_file_path = os.path.join(out_dir, "coverage.json")
        
        agg_res = dict()
        
        pos_test_df = self.read_ratings_to_df(self.ratings_dict["positive_test"])
        
        user_ids, timestamps = get_user_and_first_timestamp_from_ratings(pos_test_df)
        user_ids = np.expand_dims(user_ids, axis=1)
        timestamps = np.expand_dims(timestamps, axis=1)
        user_embeddings = self.create_user_embeddings_batch(user_ids, timestamps)
        
        for top_k in [self.top_k, 20]:
            
            if top_k == self.top_k:
                indexer = self.movie_indexer
            else:
                indexer = Retriever.build_scann_searcher(embeddings=self.movie_catalog_embeddings, top_k=top_k)
            
            res = dict()
            neighbors, distances = indexer.search_batched(user_embeddings)
            neighbors += self.MOVIE_OFFSET
            
            retrieval_df = pl.DataFrame({
                "user_id": user_ids.squeeze(),
                "movie_id": neighbors  #shape (n_users, top_k)
            }).explode("movie_id")
            
            count = retrieval_df["movie_id"].n_unique()
            cat_count = self.model_dict['n_movies']
            res["full_coverage"] = float(count)/float(cat_count)
            
            # count by movie_tier
            retrieval_df = retrieval_df.join(self.movie_tiers_df, on="movie_id", how="left")  #adds column "movie_tier"
            for movie_tier in range(0, 3):
                count = retrieval_df.filter(pl.col("movie_tier") == movie_tier)["movie_id"].n_unique()
                cat_count = self.movie_tiers_df.filter(pl.col("movie_tier") == movie_tier)["movie_id"].count()
                res[f"movie_tier_{movie_tier}_coverage"] = float(count) / float(cat_count)
            
            #count by user_tier
            retrieval_df = retrieval_df.join(self.user_tiers_df, on="user_id", how="left") #adds column "user_tier"
            cat_count = self.model_dict['n_movies']
            for user_tier in range(0, 3):
                count = retrieval_df.filter(pl.col("user_tier")==user_tier)["movie_id"].n_unique()
                res[f"user_tier_{user_tier}_coverage"] = float(count) / (float(cat_count))
                
            #count by user_tier and movie_tier
            for user_tier in range(0, 3):
                df = retrieval_df.filter(pl.col("user_tier") == user_tier)
                for movie_tier in range(0, 3):
                    df2 = df.filter(pl.col("movie_tier") == movie_tier)
                    count = df2["movie_id"].n_unique()
                    cat_count = self.movie_tiers_df.filter(pl.col("movie_tier") == movie_tier)["movie_id"].count()
                    res[f"user_tier_{user_tier}_movie_tier_{movie_tier}_coverage"] = float(count) / float(cat_count)
                    
            agg_res[f"coverage_k_{top_k}"] = res
            
            if top_k != self.top_k:
                #calc Gini coeff
                res = dict()
                # ------------------------------------------------------------------
                # GINI COEFFICIENT & LORENZ DISTRIBUTION CALCULATION
                #  gini coeff is a measure of statistical dispersion
                # this is the rank-based formulation for a discrete population.
                #    G = (2 * sum_{i=1,n}( i * y_i) / (n * sum_{i=1,n}( i * y_i))
                #          - ((n+1)/n)
                #     where y_i is the retrieval count of the movie
                #           i is the rank of the movie when sorted in ascending order from 1 to n
                #           n is the total number of iterm in the catalog
                #     the runtime complexity is limited by the sorting O(N * log(N))
                # ------------------------------------------------------------------
                
                # Must include catalog items with 0 retrievals to avoid underestimating inequality)
                freq_df = (
                    self.movie_tiers_df.select("movie_id", "movie_tier")
                    .join(
                        retrieval_df.group_by("movie_id").agg(
                            pl.len().alias("retrieval_count")),
                        on="movie_id",
                        how="left"
                    )
                    .with_columns(pl.col("retrieval_count").fill_null(0))
                    .sort("retrieval_count") # Must be sorted ascending for Lorenz & Gini math
                )
                
                n_catalog = freq_df.height
                total_recs = freq_df["retrieval_count"].sum()
                
                if total_recs > 0:
                    # --- Full Catalog Gini ---
                    full_gini = freq_df.select(
                        ((2.0 * (pl.col("retrieval_count") * pl.int_range(1,
                            n_catalog + 1)).sum()) /
                         (n_catalog * total_recs)) - (
                                    (n_catalog + 1.0) / n_catalog)
                    ).item()
                    
                    # --- Lorenz Curve Calculation & Export ---
                    lorenz_df = freq_df.with_columns(
                        cum_items_pct=pl.int_range(1,
                            n_catalog + 1) / n_catalog,
                        cum_recs_pct=pl.col(
                            "retrieval_count").cum_sum() / total_recs
                    )
                    
                    # Extract specific analytical points
                    bottom_80_share = lorenz_df.filter(pl.col("cum_items_pct") <= 0.80)[
                        "cum_recs_pct"].max()
                    top_10_share = 1.0 - (lorenz_df.filter(
                        pl.col("cum_items_pct") <= 0.90)[
                                              "cum_recs_pct"].max() or 0.0)
                    
                    # Export a clean 100-point Lorenz Curve for automated analysis/plotting
                    # Groups into 1% to 100% buckets and takes the max cumulative share for each
                    lorenz_curve_array = (
                        lorenz_df.with_columns(
                            (pl.col("cum_items_pct") * 100).ceil().cast(
                                pl.Int32).alias("percentile"))
                        .group_by("percentile").agg(
                            pl.col("cum_recs_pct").max())
                        .sort("percentile")["cum_recs_pct"].to_list()
                    )
                    
                    # --- Gini by Movie Tier ---
                    for movie_tier in range(0, 3):
                        tier_freq_df = freq_df.filter(
                            pl.col("movie_tier") == movie_tier).sort(
                            "retrieval_count")
                        n_tier = tier_freq_df.height
                        tier_recs = tier_freq_df["retrieval_count"].sum()
                        
                        tier_gini = 0.0
                        if tier_recs > 0:
                            tier_gini = tier_freq_df.select(
                                ((2.0 * (pl.col(
                                    "retrieval_count") * pl.int_range(1,
                                    n_tier + 1)).sum()) /
                                 (n_tier * tier_recs)) - (
                                            (n_tier + 1.0) / n_tier)
                            ).item()
                        res[f"movie_tier_{movie_tier}_gini"] = float(tier_gini)
                    
                    # --- Gini by User Tier ---
                    for user_tier in range(0, 3):
                        # Isolate retrievals generated ONLY by this user_tier
                        tier_retrieval_df = retrieval_df.filter(pl.col("user_tier") == user_tier)
                        
                        # Map those isolated retrievals onto the FULL movie catalog
                        user_tier_freq_df = (
                            self.movie_tiers_df.select("movie_id")
                            .join(
                                tier_retrieval_df.group_by("movie_id").agg(
                                    pl.len().alias("retrieval_count")),
                                on="movie_id",
                                how="left"
                            )
                            .with_columns(
                                pl.col("retrieval_count").fill_null(0))
                            .sort("retrieval_count")
                        )
                        
                        u_tier_recs = user_tier_freq_df[
                            "retrieval_count"].sum()
                        u_tier_gini = 0.0
                        
                        if u_tier_recs > 0:
                            u_tier_gini = user_tier_freq_df.select(
                                ((2.0 * (pl.col(
                                    "retrieval_count") * pl.int_range(1,
                                    n_catalog + 1)).sum()) /
                                 (n_catalog * u_tier_recs)) - (
                                            (n_catalog + 1.0) / n_catalog)
                            ).item()
                        res[f"user_tier_{user_tier}_gini"] = float(u_tier_gini)
                
                else:
                    full_gini = 0.0
                    bottom_80_share = 0.0
                    top_10_share = 0.0
                    lorenz_curve_array = []
                
                res["full_gini"] = float(full_gini)
                res["lorenz_bottom_80_share"] = float(bottom_80_share)
                res["lorenz_top_10_share"] = float(top_10_share)
                res["lorenz_curve_array"] = lorenz_curve_array  # List of 100 floats for JSON export
                
                if lorenz_curve_array:
                    output_lorenz_file_path = os.path.join(out_dir, f"lorenz_curve_k_{top_k}.png")
                    self.plot_lorenz_curve(lorenz_curve_array, top_k, full_gini, output_lorenz_file_path)
                
                # ------------------------------------------------------------------
                # AUTOMATED INSIGHTS & CONCLUSIONS
                # ------------------------------------------------------------------
                conclusions = []
                
                if total_recs > 0:
                    # Insight 1: Overall Popularity Bias (Full Gini)
                    if full_gini > 0.90:
                        conclusions.append(
                            f"SEVERE POPULARITY BIAS: Gini is {full_gini:.2f}. The model is acting as a popularity echo chamber, collapsing onto blockbuster items.")
                    elif full_gini < 0.45:
                        conclusions.append(
                            f"SUSPICIOUSLY UNIFORM: Gini is {full_gini:.2f}. The model may be overly random or popularity suppression (Log-Q/Temperature) is too aggressive.")
                    else:
                        conclusions.append(
                            f"HEALTHY BIAS: Gini is {full_gini:.2f}. The model successfully balances mainstream relevance with catalog exploration.")
                    
                    # Insight 2: Long-Tail Health (Bottom 80% Share)
                    if bottom_80_share < 0.05:
                        conclusions.append(
                            f"DEAD TAIL: The bottom 80% of the catalog receives only {bottom_80_share:.1%} of recommendations. Niche items are effectively invisible.")
                    elif bottom_80_share > 0.15:
                        conclusions.append(
                            f"STRONG TAIL: The bottom 80% captures {bottom_80_share:.1%} of traffic, indicating excellent long-tail surfacing capability.")
                    else:
                        conclusions.append(
                            f"MODERATE TAIL: The bottom 80% captures {bottom_80_share:.1%} of traffic.")
                    
                    # Insight 3: Head Concentration (Top 10% Share)
                    if top_10_share > 0.75:
                        conclusions.append(
                            f"HEAD HEAVY: The top 10% of items consume {top_10_share:.1%} of all recommendation slots.")
                    else:
                        conclusions.append(
                            f"DIVERSE HEAD: The top 10% consume {top_10_share:.1%} of slots, leaving plenty of room for the torso/tail.")
                    
                    # Insight 4: User Cohort Behavior
                    gini_power = res.get("user_tier_2_gini", 1.0)
                    gini_light = res.get("user_tier_0_gini", 1.0)
                    
                    if gini_power < gini_light - 0.02:  # 0.02 buffer for noise
                        conclusions.append(
                            "USER PERSONALIZATION: Power users exhibit lower Gini (more diverse slates) than light users, successfully leveraging rich interaction histories.")
                    elif gini_power > gini_light + 0.02:
                        conclusions.append(
                            "WARNING (COHORT COLLAPSE): Power users have higher concentration (Gini) than light users. The model may be pulling rich histories into dense popularity traps.")
                    else:
                        conclusions.append(
                            "UNIFORM COHORTS: Light and Power users experience roughly the same level of catalog concentration.")
                
                res["automated_conclusions"] = conclusions
                
                agg_res[f"eval_k_{top_k}"] = res
                
        print(f'\n', json.dumps(agg_res, indent=4))
        with open(output_file_path, "w") as f:
            json.dump(agg_res, f, indent=4)
            
        pass
    
    def test_popularity_bias(self):
        ## a.k.a. Macroscopic Amplification
        
        output_file_path = os.path.join(get_bin_dir(), "popularity_bias.json")
        
        agg_res = dict()
        
        # ======= stratified by user_tier ====================
        pos_test_df = self.read_ratings_to_df(self.ratings_dict["positive_test"])
        pos_test_df = pos_test_df.join(self.user_tiers_df, on="user_id", how="left")
        
        history_df = self.get_positive_ratings_history()
        history_df = history_df.join(self.user_tiers_df, on="user_id", how="left")
        
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
            neighbors, distances = self.movie_indexer.search_batched(user_embeddings)
            #chk = full_movie_ids[neighbors]
            neighbors += self.MOVIE_OFFSET
            #are_equal = np.array_equal(chk, neighbors)
            
            tier_ground_truth_df = pos_test_df.filter(pl.col(stratification_key)==tier)
            tier_history_df = history_df.filter(pl.col(stratification_key)==tier)
            
            res = self.evaluate_popularity_bias(
                neighbors=neighbors,  # shape: (n_users, top_k)
                ground_truth_df = tier_ground_truth_df,  # Test set (positives only)
                train_history_df = tier_history_df,  # Train set (positives only)
                top_k = self.top_k,
                tag=f"{stratification_key}_{tier}"
            )
        
            agg_res = agg_res | res
            
        print("popularity bias\n", json.dumps(agg_res, indent=4))
        
        with open(output_file_path, "w") as f:
            json.dump(agg_res, f, indent=4)
    
    def test_embedding_hubness(self):
        
        output_file_path = os.path.join(get_bin_dir(), "embedding_hubness.json")
        
        agg_res = dict()
        
        # GLOBAL ITEM UNIFORMITY (Calculated Once) ---
        # Evaluates how well the full catalog is distributed across the hypersphere.
        # We extract this into a helper lambda since we'll reuse the exact math for users.
        t_param = 2.0
        
        def calc_uniformity(embs_np):
            n = embs_np.shape[0]
            if n <= 1: return 0.0
            # Since ||u|| = 1, dist^2 = 2 - 2*(u dot v)
            dist_sq = 2.0 - 2.0 * np.dot(embs_np, embs_np.T)
            # Uniformity is log expected value of exp(-t * dist^2)
            # We subtract 'n' to remove the self-pair diagonals (where exp(0) = 1)
            sum_off_diag = np.sum(np.exp(-t_param * dist_sq)) - n
            return np.log(sum_off_diag / (n * (n - 1)))
        
        global_item_uniformity = calc_uniformity(self.movie_catalog_embeddings.numpy())
        
        agg_res[f"Global Item Uniformity"] = global_item_uniformity
        
        #this is the test dataset of positive ratings
        pos_test_df = self.read_ratings_to_df(self.ratings_dict["positive_test"])
        pos_test_df = pos_test_df.join(self.user_tiers_df, on="user_id", how="left")
        #has "user_id", "movie_id", "rating", "timestamp", "user_tier"
        
        for user_tier in [0, 1, 2]:
            
            #  Isolate the ground-truth positive pairs for this tier
            tier_pos_df = pos_test_df.filter(pl.col("user_tier") == user_tier)
            
            user_ids = np.expand_dims(tier_pos_df["user_id"].to_numpy(), axis=1)
            timestamps = np.expand_dims(tier_pos_df["timestamp"].to_numpy(), axis=1)
            pos_movie_ids = tier_pos_df["movie_id"].to_numpy()  # 1D array for indexing
            
            # Generate User Embeddings for these specific rows
            user_embeddings = self.create_user_embeddings_batch(user_ids, timestamps)
            u_np = user_embeddings.numpy()  # Shape: (N, emb_dim)
            
            # Fast Lookup of corresponding Movie Embeddings
            # (Assuming your movie_catalog_embeddings index perfectly matches movie_id)
            m_np = self.movie_catalog_embeddings.numpy()[pos_movie_ids - self.MOVIE_OFFSET]  # Shape: (N, emb_dim)
            
            # 5User Uniformity
            tier_user_uniformity = calc_uniformity(u_np)
            
            # Row-wise Dot Product for exact positive pair alignment
            # Element-wise multiply -> Sum across the embedding dimension
            pos_dot_products = np.sum(u_np * m_np, axis=1)
            tier_alignment = np.mean(2.0 - 2.0 * pos_dot_products)
            
            agg_res[f"user_tier_{user_tier}"] = {
                "alignment": float(tier_alignment),
                "user_uniformity": float(tier_user_uniformity)
            }
            
        # ------------------------------------------------------------------
        # RULE-BASED CONCLUSIONS
        # ------------------------------------------------------------------
        conclusions = []
        
        # Rule 1: Global Item Uniformity (Are items collapsed?)
        # Healthy distributions typically score between -1.5 and -3.5.
        # Closer to 0 means severe collapse.
        if global_item_uniformity > -1.0:
            conclusions.append(f"ITEM COLLAPSE WARNING: Global Item Uniformity is high ({global_item_uniformity:.2f}). The movie catalog is densely clumped together, likely leading to low diversity.")
        elif global_item_uniformity < -2.5:
            conclusions.append(f"HEALTHY ITEM DISPERSION: Global Item Uniformity is excellent ({global_item_uniformity:.2f}). Movies are well-distributed across the hypersphere.")
        else:
            conclusions.append(f"MODERATE ITEM UNIFORMITY: Items are adequately distributed ({global_item_uniformity:.2f}).")
    
        # Rule 2: User Cohort Collapse (Are power users clumping?)
        u_uni_light = agg_res["user_tier_0"]["user_uniformity"]
        u_uni_power = agg_res["user_tier_2"]["user_uniformity"]
        
        if u_uni_power > u_uni_light + 0.5:
            conclusions.append(f"POWER USER COLLAPSE: Power users (Tier 2 uniformity: {u_uni_power:.2f}) are significantly more clumped than Light users (Tier 0 uniformity: {u_uni_light:.2f}). Their rich histories are collapsing into dense 'average' vectors.")
        else:
            conclusions.append("USER GEOMETRY HEALTHY: Power users maintain distinct spatial representations without collapsing into a dense cluster.")
    
        # Rule 3: Alignment Quality (Are users near their ground truth items?)
        # Distance of 2.0 is perfectly orthogonal (random). < 1.0 is good alignment.
        avg_alignment = np.mean([agg_res[f"user_tier_{i}"]["alignment"] for i in range(3)])
        if avg_alignment > 1.5:
            conclusions.append(f"POOR ALIGNMENT: Average positive pair distance is {avg_alignment:.2f} (Max is 4.0). User embeddings are struggling to map closely to their interacted items.")
        elif avg_alignment < 0.5:
            conclusions.append(f"OVERFIT WARNING: Average positive pair distance is {avg_alignment:.2f}. Alignment is extremely tight, which may indicate the model is memorizing exact pairs rather than generalizing.")
        else:
            conclusions.append(f"HEALTHY ALIGNMENT: Average positive pair distance is {avg_alignment:.2f}. Model balances proximity to positive items while maintaining generalization space.")
    
        agg_res["automated_conclusions"] = conclusions
        
        agg_res = self.convert_to_native_types(agg_res)
        print(f"Embedding Hubness:\n", json.dumps(agg_res, indent=4))
        
        with open(output_file_path, "w") as f:
            json.dump(agg_res, f, indent=4)
    
    def test_plot_tsne_umap_movie_embeddings(self):
        
        outdir = os.path.join(get_bin_dir(), "embedding_plots")
        shutil.rmtree(outdir, ignore_errors=True)
        os.makedirs(outdir, exist_ok=True)
       
        #emb_movies_df = pl.DataFrame(
        #    [(m_id, emb) for m_id, emb in zip(self.movie_ids.numpy().squeeze(), self.movie_catalog_embeddings.numpy())],
        #    schema=["movie_id", "embedding"],
        #    orient="row"
        #)
        emb_movies_df = pl.DataFrame({
            "movie_id": self.movie_ids.numpy().squeeze(),
            "embedding": self.movie_catalog_embeddings.numpy()
        })
        emb_movies_df = emb_movies_df.join(self.movie_tiers_df, on="movie_id", how="left")
        
        self.plot_embeddings_umap_tsne(emb_movies_df, outdir, "all_movies",
            stratified_key="movie_tier")
        
    def test_cold_start_distance(self):
        self.simulate_cold_start_embeddings()
        
    def test_inter_list_diversity(self):
        
        output_file_path = os.path.join(get_bin_dir(), "interlist_diversity.json")
        
        agg_res = dict()
        
        # random sample of all users
        # random sample of stratified tier users
        # random sample of all users but catalog expanded to include cold-start metrics
        
        n_samples = 500
        
        num_catalog_movies = self.model_dict['n_movies']
        
        # ========= random sample of all users ==========================
        
        pos_test_df = self.read_ratings_to_df(self.ratings_dict["positive_test"])
        pos_test_df = pos_test_df.join(self.user_tiers_df, on="user_id", how="left")
        (user_ids, timestamps) = get_random_user_and_first_timestamp_from_ratings(pos_test_df, n_samples)
        user_ids = np.expand_dims(user_ids, axis=1)
        timestamps = np.expand_dims(timestamps, axis=1)
        
        user_embeddings = self.create_user_embeddings_batch(user_ids, timestamps) #tf.Tensor shape (5096, 32)
       
        #neighbors shape is (n_samples, top_k)
        neighbors, distances = self.movie_indexer.search_batched(user_embeddings)
        
        ## if the dataset samle were > 100_000, we could MinHash to conserve memory
        ## instead of the fast vectorized matrices with BLAS optimization that
        ## we use here to calc Jaccard similarity
        inter_user_diversity, mean_jaccard = self.calculate_exact_inter_user_diversity(
            neighbors, num_catalog_movies
        )
        
        res = self.analyze_inter_user_diversity(mean_jaccard, num_catalog_movies,
            self.top_k, "all_users",
            baseline_jaccard = None
        )
        
        agg_res = agg_res | res
        
        # ======= stratified by user_tier ====================
        stratification_key = "user_tier"
        stratification_values: List = [0, 1, 2]
        
        tier_dict : Dict[int, tuple[np.ndarray, np.ndarray]] = get_stratified_user_and_first_timestamp_from_ratings(
            ratings_df=pos_test_df,
            sample_size = n_samples,
            stratification_key = stratification_key,
            stratification_values = stratification_values,
        )
        
        for tier in stratification_values:
            user_ids, timestamps = tier_dict[tier]
            user_ids = np.expand_dims(user_ids, axis=1)
            timestamps = np.expand_dims(timestamps, axis=1)
            user_embeddings = self.create_user_embeddings_batch(user_ids, timestamps)
            # neighbors shape is (n_samples, top_k)
            neighbors, distances = self.movie_indexer.search_batched(user_embeddings)
            inter_user_diversity, mean_jaccard = self.calculate_exact_inter_user_diversity(
                neighbors, num_catalog_movies
            )
            res = self.analyze_inter_user_diversity(mean_jaccard,
                num_catalog_movies, self.top_k, f"tier_{tier}",
                baseline_jaccard=None
            )
            
            agg_res = agg_res | res
            
        # ===== cold start movies, adding 501 movies to the movie catalog (501 because its between 10-15% of catalog size and is 167 per movie tier) =====
        (new_movie_ids, new_movie_embeddings) = self.get_cold_start_movies(pos_test_df)
        full_movie_embeddings = tf.concat([self.movie_catalog_embeddings, new_movie_embeddings], axis=0)
        indexer = Retriever.build_scann_searcher(embeddings=full_movie_embeddings, top_k=self.top_k)
        
        num_catalog_movies += len(new_movie_ids)
        
        (user_ids, timestamps) = get_random_user_and_first_timestamp_from_ratings(pos_test_df, n_samples)
        user_ids = np.expand_dims(user_ids, axis=1)
        timestamps = np.expand_dims(timestamps, axis=1)
        
        user_embeddings = self.create_user_embeddings_batch(user_ids, timestamps)  # tf.Tensor shape (5096, 32)
        
        # neighbors shape is (n_samples, top_k)
        neighbors, distances = indexer.search_batched(user_embeddings)
        
        inter_user_diversity, mean_jaccard = self.calculate_exact_inter_user_diversity(
            neighbors, num_catalog_movies
        )
        
        res = self.analyze_inter_user_diversity(mean_jaccard,
            num_catalog_movies, self.top_k, "all_users_but_catalog_has_cold_start_movies",
            baseline_jaccard=None
        )
        
        agg_res = agg_res | res
        
        print("inter_user_diversity\n", json.dumps(agg_res, indent=4))
        
        with open(output_file_path, "w") as f:
            json.dump(agg_res, f, indent=4)
    
    def test_intra_list_diversity(self):
        
        output_file_path = os.path.join(get_bin_dir(), "intralist_diversity.json")

        agg_res = dict()
        
        pos_test_df = self.read_ratings_to_df(self.ratings_dict["positive_test"])
        first_interactions_df = pos_test_df.group_by("user_id").agg(pl.col("timestamp").min())
        user_ids = first_interactions_df["user_id"].to_numpy()
        print(f'n unique users in test ds={len(user_ids)}')
        user_ids = np.expand_dims(user_ids, axis=1)
        timestamps = first_interactions_df["timestamp"].to_numpy()
        timestamps = np.expand_dims(timestamps, axis=1)
        user_embeddings = self.create_user_embeddings_batch(user_ids, timestamps) #tf.Tensor shape (5096, 32)
        
        neighbors, distances = self.movie_indexer.search_batched(user_embeddings)
        if isinstance(neighbors, np.ndarray):
            neighbors = tf.convert_to_tensor(neighbors, dtype=tf.int32)
        
        res = self.calculate_batched_intra_list_diversity(neighbors, self.movie_catalog_embeddings)
        
        print(f"Average Intra-List Diversity @ {self.top_k}\n: {json.dumps(res, indent=4)}")
        
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
    
    def evaluate_popularity_bias(self,
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
        pop_df = (train_history_df.group_by("movie_id")
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
        
        # Calculate Average Log-Popularity for Ground Truth Test Items
        # Group the test DataFrame by user_id
        gt_grouped = (
            ground_truth_df
            .group_by("user_id")
            .agg(pl.col("movie_id").alias("gt_movies"))
        )
        
        user_gt_pop = []
        for row in gt_grouped.iter_rows():
            gt_movies = row[1]
            gt_pops = [pop_map.get(m_id, 0.0) for m_id in gt_movies]
            user_gt_pop.append(np.mean(gt_pops))
        user_gt_pop = np.array(user_gt_pop)
        
        # Compute Bias Metrics
        mean_retr_pop = float(np.mean(user_retr_pop))
        mean_gt_pop = float(np.mean(user_gt_pop))
        
        delta_pop = float(np.mean(user_retr_pop - user_gt_pop))
        std_delta_pop = float(np.std(user_retr_pop - user_gt_pop))
        
        # Amplification Ratio (Retrieved / Ground Truth)
        amplification_ratio = mean_retr_pop / max(1e-9, mean_gt_pop)
        
        # Automated Text Interpretation
        conclusions = []
        if amplification_ratio > 1.15:
            conclusions.append(
                f"[HIGH POPULARITY BIAS]: Model amplifies popularity by {amplification_ratio:.2f}x. "
                f"Retrieved slates (LogPop: {mean_retr_pop:.2f}) are significantly more popular "
                f"than the users' natural test consumption (LogPop: {mean_gt_pop:.2f})."
            )
        elif amplification_ratio < 0.85:
            conclusions.append(
                f"[NICHE BIAS]: Model suppresses popularity by {amplification_ratio:.2f}x. "
                f"Retrieved slates are pushing much more obscure items than the user normally consumes."
            )
        else:
            conclusions.append(
                f"[NEUTRAL POPULARITY]: Model preserves user consumption habits (Ratio: {amplification_ratio:.2f}x). "
                f"Retrieved item popularity aligns closely with ground truth behavior."
            )
        
        return {
            f"{tag}_metrics": {
                "mean_retrieved_log_pop": round(mean_retr_pop, 4),
                "mean_ground_truth_log_pop": round(mean_gt_pop, 4),
                "delta_log_pop": round(delta_pop, 4),
                "std_delta_log_pop": round(std_delta_pop, 4),
                "amplification_ratio": round(amplification_ratio, 4)
            },
            f"{tag}_analysis": conclusions
        }
        
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
        
        return {
            f"{dict_tag}_inter_user_diversity_metrics": metrics,
            f"{dict_tag}_inter_user_diversity_conclusions": conclusions
        }
    
    def calculate_exact_inter_user_diversity(self,
            neighbors: np.ndarray,
            num_catalog_movies: int
    ) -> tuple[float, float]:
        """
        Computes exact Inter-User Diversity (1 - Jaccard Similarity) across user slates.

        Args:
            neighbors: (n_samples, top_k) array of retrieved movie IDs.
            num_catalog_movies: Total number of movies in the catalog.

        Returns:
            inter_user_diversity (float), mean_jaccard_similarity (float)
        """
        n_samples, top_k = neighbors.shape
        
        # |A union B| = |A| + |B| - |A intersect B| = 2*top_k - |A intersect B|
        
        # Create a dense matrix of user-item interactions
        # Shape: (n_samples, num_catalog_movies). We use float32 for fast BLAS matmul.
        # if num_catalog_movies were > 100_000 we would use MinHash instead of this method
        A = np.zeros((n_samples, num_catalog_movies), dtype=np.float32)
        
        # Advanced indexing to populate the retrieved items instantly
        row_indices = np.arange(n_samples)[:, None]
        A[row_indices, neighbors] = 1.0
        #for each row in A, the ones are indictors of the neighbors indices
        #so now it contains indicators for B
        
        # Matrix Multiplication to find all pairwise intersections
        # A @ A.T yields a matrix where element (i,j) is the number of shared items
        intersections = A @ A.T
        
        # Calculate Unions
        unions = (2 * top_k) - intersections
        
        # Calculate pairwise Jaccard Similarity
        # Add a small epsilon to prevent division by zero in extreme edge cases
        jaccard_matrix = intersections / (unions + 1e-9)
        
        # Extract the average (ignoring the diagonal where users compare to themselves)
        total_jaccard = np.sum(jaccard_matrix) - np.trace(jaccard_matrix)
        num_pairs = n_samples * (n_samples - 1)
        
        mean_jaccard = float(total_jaccard / num_pairs)
        
        # Inter-User Diversity is the complement of Jaccard Similarity
        inter_user_diversity = 1.0 - mean_jaccard
        
        return inter_user_diversity, mean_jaccard
    
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
    
    def get_cold_start_movies(self, pos_test_df: pl.DataFrame) -> Tuple[tf.Tensor, tf.Tensor]:
        
        pos_test_df = self.join_df_to_movie_tiers(pos_test_df)
        movie_test_counts = pos_test_df.group_by("movie_id").agg(pl.len().alias("test_interaction_count"))
        candidate_pool = movie_test_counts.filter(pl.col("test_interaction_count") >= 5)
        
        pos_test_df = candidate_pool.join(pos_test_df, on="movie_id", how="left")
        
        num_per_tier = 167  # total = 501
        candidate_pool_df = (pos_test_df.sample(fraction=1.0, seed=42, shuffle=True)
            .group_by("movie_tier").head(num_per_tier))
            
        original_movie_ids = candidate_pool_df["movie_id"].unique().to_numpy()
        
        id0 = self.movie_id_range_incl[-1] + 1
        new_movie_ids = np.array([i for i in range(id0, id0 + len(original_movie_ids))])
        
        original_movie_ids = np.expand_dims(original_movie_ids, axis=1)
        new_movie_ids = np.expand_dims(new_movie_ids, axis=1)
        
        original_inputs = self.movie_data.get_movie(original_movie_ids)
        new_inputs = original_inputs.copy()
        new_inputs["movie_id"] = tf.constant(new_movie_ids)
        
        # original_embeddings : tf.Tensor = self._create_movie_embeddings_batch(original_inputs)
        new_embeddings: tf.Tensor = self._create_movie_embeddings_batch(new_inputs)
        
        new_movie_ids = tf.convert_to_tensor(new_movie_ids, dtype=tf.int32)
        
        return (new_movie_ids, new_embeddings)
    
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
        
        id0 = self.movie_id_range_incl[-1] + 1
        new_movie_ids = np.array([i for i in range(id0, id0 + len(original_movie_ids))])
        
        original_movie_ids = np.expand_dims(original_movie_ids, axis=1)
        new_movie_ids = np.expand_dims(new_movie_ids, axis=1)
        
        original_inputs = self.movie_data.get_movie(original_movie_ids)
        new_inputs = original_inputs.copy()
        new_inputs["movie_id"] = tf.constant(new_movie_ids)
        
        #original_embeddings : tf.Tensor = self._create_movie_embeddings_batch(original_inputs)
        new_embeddings : tf.Tensor = self._create_movie_embeddings_batch(new_inputs)
        new_embeddings = new_embeddings.numpy()
        
        full_movie_embeddings = self.movie_catalog_embeddings.numpy()
        
        #for each new_movie_ids, replace with embedding for new_movie_ids.
        #this not only subtracts the old and inserts the new, but gives them the same index so that they
        # are findable when compared to the ground truth ratings.
        for i, m_id in enumerate(original_movie_ids):
            idx = m_id[0] - self.MOVIE_OFFSET
            assert(self.movie_ids[idx].numpy().item() == m_id[0])
            full_movie_embeddings[idx] = new_embeddings[i]
            
        top_k = self.top_k
        
        new_indexer = Retriever.build_scann_searcher(embeddings=full_movie_embeddings, top_k=top_k)
    
        #retrieve
        user_embeddings = self.create_user_embeddings_batch(user_ids, timestamps)
        
        orig_neighbors, orig_distances = self.movie_indexer.search_batched(user_embeddings)
        new_neighbors, new_distances = new_indexer.search_batched(user_embeddings)
        
        orig_results = self.evaluate_retrieval_with_tier_share(user_ids, orig_neighbors, self.MOVIE_OFFSET, candidate_pool_df, self.movie_tiers_df, top_k=top_k)
        new_results = self.evaluate_retrieval_with_tier_share(user_ids, new_neighbors, self.MOVIE_OFFSET, candidate_pool_df, self.movie_tiers_df, top_k=top_k)

        print("results on test set:\n", json.dumps(orig_results, indent=4))
        print("results on test set cold start:\n", json.dumps(new_results, indent=4))
        
        res = self.compare_retrieval_runs(orig_results, new_results, top_k, self.model_dict["n_movies"], "test set", "cold-start set")
        print("comparisons:\n", json.dumps(res, indent=4))
        
        with open(output_file_path, "w") as f:
            json.dump(res, f, indent=4)
        
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
            movie_tiers_df.select(["movie_id", "movie_tier"]).rows()
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
        inputs = self.movie_data.get_movie(movie_ids)
        return self._create_movie_embeddings_batch(inputs)
    
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
    
    def plot_embeddings_umap_tsne(self, joined_df, outdir: str, file_tag: str,
        stratified_key: str = "tier"):
        
        # Convert to NumPy arrays for UMAP
        X = np.array(joined_df.get_column("embedding").to_list())
        y = joined_df.get_column(stratified_key).to_numpy()
        
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
        plt.title(f"Movie Embedding UMAP Projection by {stratified_key} ({file_tag})")
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
        plt.title(f"Movie Embedding t-SNE Projection by {stratified_key} ({file_tag})")
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
        for file_path in self.ratings_dict["positive_history"]:
            ratings_df = self.read_ratings_to_df(file_path)
            r.append(ratings_df)
        return pl.concat(r)
    
    def create_movie_catalog_embeddings(self) -> Tuple[tf.Tensor,  tf.Tensor]:
        
        df = pl.read_parquet(self.movies_path)
        df = df.sort('movie_id')
        movie_ids = np.expand_dims(df['movie_id'].to_numpy(), axis=1)
        movie_ids = tf.constant(movie_ids, name='movie_id', dtype=tf.int64)
        
        inputs = self.movie_data.get_movie(movie_ids)
        embeddings: tf.Tensor = self._create_movie_embeddings_batch(inputs)
        
        return movie_ids, embeddings
    
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

if __name__ == '__main__':
    unittest.main()

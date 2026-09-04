import collections
import json
import os.path
import unittest
import glob
from collections import defaultdict
from typing import Any, Dict, Union
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

import polars as pl
import numpy as np
import plotly.express as px  # needs kaleido to write pngs
from plotly.subplots import make_subplots

import msgpack
from array_record.python import array_record_module

from helper import *
from rich import print as rprint

from movie_lens_retrieval.Retriever import Retriever, EmbeddingType

class PlotEmbeddingsClusters(unittest.TestCase):
    def setUp(self):
        
        self.movie_emb = os.path.join(get_project_dir(),
            "src/test/resources/data/movie_emb_inp/*tfrecord*.gz")
        self.n_movies = 3883
        self.MOVIE_OFFSET = 6040 + 1
        
        _ct = "GZIP" if self.movie_emb.endswith(".gz") else None
        file_paths = glob.glob(self.movie_emb)
        if len(file_paths) == 0:
            raise FileNotFoundError(self.movie_emb)
        # read the associated metadata file
        metadata_path = Retriever.get_parent_directory(file_paths[0])
        metadata_path = f"{metadata_path}/emb_metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        self.embed_dim = metadata['embed_dim']
        
        self.em_feature_spec = {
            "movie_id": tf.io.FixedLenFeature(shape=[], dtype=tf.int64,
                default_value=None),
            "embedding": tf.io.FixedLenFeature(shape=[self.embed_dim],
                dtype=tf.float32)
        }
        
        movie_tiers_path = os.path.join(get_project_dir(),
            "src/test/resources/data/movie_tiers.json")
        self.movie_tiers_df = pl.read_ndjson(movie_tiers_path)
        
        dataset = tf.data.TFRecordDataset(os.path.join(get_project_dir(),
            "src/test/resources/data/movie_emb_inp/movie_emb-00000-of-00001.tfrecord.gz"),
            compression_type="GZIP")
        dataset = dataset.map(self.parse_tfrecord)
        
        emb_df = pl.from_dicts(list(dataset.as_numpy_iterator()))
        self.emb_df = emb_df.join(self.movie_tiers_df, on='movie_id', how='left')
        
    def parse_tfrecord(self, proto):
        parsed = tf.io.parse_single_example(proto, self.em_feature_spec)
        #return parsed['movie_id'], parsed['embedding']
        return parsed
    
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
    
    def test_plot_train_val_movie_embeddings(self):
        outdir = os.path.join(get_bin_dir(), "emb_tiers")
        os.makedirs(outdir, exist_ok=True)
        
        df_train_ratings = self._read_ratings_array_record(
            os.path.join(get_project_dir(),
                'src/test/resources/data/ratings_train_liked.array_record'))
        df_val_ratings = self._read_ratings_array_record(
            os.path.join(get_project_dir(),
                'src/test/resources/data/ratings_val_liked.array_record'))
        
        unique_movie_ids = (
            df_train_ratings.select("movie_id").unique().get_column("movie_id")
        )
        movie_tiers_filtered = self.emb_df.filter(
            pl.col("movie_id").is_in(unique_movie_ids)
        )
        self.plot_joined_df(movie_tiers_filtered, outdir, "train_movies")
        
        unique_movie_ids = (
            df_val_ratings.select("movie_id").unique().get_column("movie_id")
        )
        movie_tiers_filtered = self.emb_df.filter(
            pl.col("movie_id").is_in(unique_movie_ids)
        )
        self.plot_joined_df(movie_tiers_filtered, outdir, "val_movies")
        
        #intersection by movie between datasets
        common_ids = (
            df_train_ratings.select("movie_id")
            .unique()
            .join(df_val_ratings.select("movie_id").unique(), on="movie_id", how="inner")
            .get_column("movie_id")
        )
        movie_tiers_filtered = self.emb_df.filter(
            pl.col("movie_id").is_in(common_ids)
        )
        self.plot_joined_df(movie_tiers_filtered, outdir, "intersect_movies")
        # a quick look at the counts in train ratings files
        df_filtered = df_train_ratings.filter(
            pl.col("movie_id").is_in(common_ids)
        )
        df_filtered = df_filtered.join(movie_tiers_filtered, on="movie_id", how="left")
        print(f"train intersection movies all 3 tiers: {len(df_filtered['movie_id'])}")
        df_filtered = df_filtered.filter(
            pl.col("tier")==2
        )
        print(f"*train intersection movies tier=2: {len(df_filtered['movie_id'])}")
        # a quick look at the counts in val ratings files
        df_filtered = df_val_ratings.filter(pl.col("movie_id").is_in(common_ids))
        df_filtered = df_filtered.join(movie_tiers_filtered, on="movie_id",how="left")
        print(f"val intersection movies all 3 tiers: {len(df_filtered['movie_id'])}")
        df_filtered = df_filtered.filter(
            pl.col("tier") == 2
        )
        print(f"*val intersection movies tier=2: {len(df_filtered['movie_id'])}")
        
        #ntersection by user between datasets
        common_user_ids = (
            df_train_ratings.select("user_id")
            .unique()
            .join(df_val_ratings.select("user_id").unique(), on="user_id",
                how="inner")
            .get_column("user_id")
        )
        df1_filtered = df_train_ratings.filter(pl.col("user_id").is_in(common_user_ids))
        df2_filtered = df_val_ratings.filter(pl.col("user_id").is_in(common_user_ids))
        common_ids = (
            df1_filtered.select("movie_id")
            .unique()
            .join(df2_filtered.select("movie_id").unique(), on="movie_id",
                how="inner")
            .get_column("movie_id")
        )
        movie_tiers_filtered = self.emb_df.filter(
            pl.col("movie_id").is_in(common_ids)
        )
        if (movie_tiers_filtered['movie_id'].count() > 0):
            self.plot_joined_df(movie_tiers_filtered, outdir, "intersect_users_then_movies")
        
        # a quick look at the counts
        df_filtered = df1_filtered.join(movie_tiers_filtered, on="movie_id", how="left")
        df_filtered = df_filtered.filter(pl.col("tier") == 2)
        print( f"*train intersection users, then movies, tier=2: {len(df_filtered['movie_id'])}")
        df_filtered = df2_filtered.join(movie_tiers_filtered, on="movie_id", how="left")
        df_filtered = df_filtered.filter(pl.col("tier") == 2)
        print(f"*val intersection users, then movies, tier=2 {len(df_filtered['movie_id'])}")
    
    def test_plot_all_movie_embeddings(self):
        
        outdir = os.path.join(get_bin_dir(), "emb_tiers")
        os.makedirs(outdir, exist_ok=True)
        
        #Left join the dataset with movie_tiers_df on movie_id
        #  and use UMPor t-SNE to show the embedding distances as a function of "tier"
        
        # movie_tiers_df has columns "movie_id" and "tier"
        
        self.plot_joined_df(self.emb_df, outdir, "all_movies")
        
    def plot_joined_df(self, joined_df, outdir:str, file_tag:str):
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
        plt.savefig(os.path.join(outdir, f"umap_{file_tag}_tiers.png"), dpi=300, bbox_inches="tight")
        #plt.show()
        plt.close()
        
        #t-SNE plot
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
        plt.savefig(os.path.join(outdir, f"tsne_{file_tag}_tiers.png"), dpi=300, bbox_inches="tight")
        #plt.show()
        plt.close()
        
    if __name__ == '__main__':
        unittest.main()

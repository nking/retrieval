from typing import Union

import tensorflow as tf
import polars as pl
from numpy import ndarray as ndarray

class MovieData(object):
    def __init__(self, movie_path:str, offset:int=6041):
        """
        given path to movie file, creates a datastructure for movie_id access
        :param movie_path: path to the users.dat file containing fields movie_id, title, genrese.
        For now, provide a parquet file.  Also note that the user_ids must be ordered from 1 to N.
        :param offset: these movie_ids already in the movies.dat file begin with offset and end with offset + n_movies.
        """
        if not movie_path.endswith(".parquet"):
            print(f'WARNING: expecting input file to be a parquet file')
        self.offset = tf.constant(offset, dtype=tf.int64)
        df = pl.read_parquet(movie_path)
        
        df = df.sort('movie_id')
        self.title = tf.constant(df['title'].to_numpy(), name='movie_title', dtype=tf.string)
        self.genres = tf.constant(df['genres'].to_numpy(), name='movie_genres', dtype=tf.string)
        self.num_movies = len(df)
        del df
    
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, 1], dtype=tf.int64),
    ])
    def get_movie(self, movie_id: Union[tf.Tensor, ndarray]):
        """
        get a dictionary of inputs usable for the Candidate model dictionary signature.
        
        :param movie_id: a tensor of an array of integer movie_ids.
           example usage: user_data.get_user(user_id=tf.constant([6041]), timestamp=tf.constant([-1]))
        :param timestamp: timestamp associated with the user_id request. if the value
           is -1, it gets reset to tf.timestamp().
        :return:
        """
        if len(tf.shape(movie_id)) == 1:
            movie_id = tf.expand_dims(movie_id, 1)
            
        idx = tf.subtract(movie_id, self.offset)
        return {
            'movie_id': movie_id,
            'genres': tf.gather(self.genres, idx)
        }


def get_movie_tiers_df(ratings_df: pl.DataFrame) -> pl.DataFrame:
    """
    Given a Polars DataFrame with ['movie_id', ...],
    returns a DataFrame with 'movie_id', 'movie_tier' where tier is 0, 1, or 2 for
         head, torso, and tail of the distribution of the number of users ratings.
    """
    # Count history length per user
    counts = ratings_df.group_by("movie_id").agg(
        pl.len().alias("movie_counts")
    )
    
    # Find the exact cutoff lengths based on quantiles
    tail_cutoff_val = counts["movie_counts"].quantile(0.20,
        interpolation="nearest")
    head_cutoff_val = counts["movie_counts"].quantile(0.80,
        interpolation="nearest")
    
    # Map to tiers based on the cutoffs
    movie_tiers_df = counts.with_columns(
        pl.when(pl.col("movie_counts") <= tail_cutoff_val)
        .then(2)  # Tail
        .when(pl.col("movie_counts") >= head_cutoff_val)
        .then(0)  # Head
        .otherwise(1)  # Torso
        .alias("movie_tier")
    ).select(["movie_id", "movie_tier"])
    
    return movie_tiers_df

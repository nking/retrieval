from typing import Dict, Union

import tensorflow as tf
import polars as pl
from numpy import ndarray as ndarray

class UserData(object):
    def __init__(self, users_path:str):
        """
        given path to user file, creates a datastructure for movie_id access
        :param users_path: path to the users.dat file containing fields movie_id, gender, age, occupation, zipcode.
        For now, provide a parquet file.  Also note that the user_ids must be ordered from 1 to N.
        """
        if not users_path.endswith(".array_record"):
            print(f'WARNING: expecting input file to be a array_record file')
            
        df = pl.read_parquet(users_path)
        df = df.sort('user_id')
        self.gender = tf.constant(df['gender'].to_numpy(), name='user_gender', dtype=tf.string)
        self.age = tf.constant(df['age'].to_numpy(), name='user_age', dtype=tf.int64)
        self.occupation = tf.constant(df['occupation'].to_numpy(),  name='user_occupation', dtype=tf.int64)
        self.num_users = len(df)
        del df
    
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, 1], dtype=tf.int64),
        tf.TensorSpec(shape=[None, 1], dtype=tf.int64)
    ])
    def get_user(self, user_id: Union[tf.Tensor, ndarray], timestamp: Union[tf.Tensor, ndarray]) -> Dict[str, tf.Tensor]:
        """
        get a dictionary of inputs usable for the Query model dictionary signature.
        
        :param user_id: a tensor of an array of integer user_ids.
           example usage: user_data.get_user(movie_id=tf.constant([123]), timestamp=tf.constant([-1]))
        :param timestamp: timestamp associated with the movie_id request. if the value
           is -1, it gets reset to tf.timestamp().
        :return:
        """
        #tf.debugging.assert_equal(tf.shape(user_id), tf.shape(timestamp),
        #    message="User ID and Timestamp batches must be the same size")
        if len(tf.shape(user_id)) == 1:
            user_id = tf.expand_dims(user_id, 1)
        if len(tf.shape(timestamp)) == 1:
            timestamp = tf.expand_dims(timestamp, 1)
            
        idx = user_id - tf.constant(1, dtype=tf.int64)
        now = tf.cast(tf.timestamp(), tf.int64)
        resolved_ts = tf.where(
            tf.equal(timestamp, -1), now, timestamp
        )
        
        return {
            'user_id': user_id,
            'gender': tf.gather(self.gender, idx),
            'age': tf.gather(self.age, idx),
            'occupation': tf.gather(self.occupation, idx),
            'timestamp': resolved_ts,
        }
    
    @tf.function
    def users_exist(self, user_ids: tf.Tensor) -> tf.Tensor:
        # This checks a million IDs as fast as it checks one.
        lower_bound = tf.greater_equal(user_ids, 1)
        upper_bound = tf.less_equal(user_ids, self.num_users)
        return tf.logical_and(lower_bound, upper_bound)
    
    def user_exists(self, user_id: int) -> bool:
        return 1 <= user_id <= self.num_users


def get_user_tiers_df(ratings_df: pl.DataFrame) -> pl.DataFrame:
    """
    Given a Polars DataFrame with ['user_id', ...],
    returns DataFrame with columns 'user_id', 'user_tier' where tier is 0, 1, or 2 for
         head, torso, and tail of the distribution of the number of users ratings.
    """
    # Count history length per user
    user_counts = ratings_df.group_by("user_id").agg(
        pl.len().alias("history_length")
    )
    
    # Find the exact cutoff lengths based on quantiles
    tail_cutoff_val = user_counts["history_length"].quantile(0.20, interpolation="nearest")
    head_cutoff_val = user_counts["history_length"].quantile(0.80, interpolation="nearest")
    
    # Map to tiers based on the cutoffs
    user_tiers_df = user_counts.with_columns(
        pl.when(pl.col("history_length") <= tail_cutoff_val)
        .then(2)  # Tail
        .when(pl.col("history_length") >= head_cutoff_val)
        .then(0)  # Head
        .otherwise(1)  # Torso
        .alias("user_tier")
    ).select(["user_id", "user_tier"])
    
    return user_tiers_df


def get_user_tiers_from_df(ratings_df: pl.DataFrame) -> Dict[int, int]:
    """
       Given a Polars DataFrame with ['user_id', ...],
       returns a HashMap with key='user_id', value=tier where tier is 0, 1, or 2 for
            head, torso, and tail of the distribution of the number of users ratings.
       """
    user_tiers_df = get_user_tiers_df(ratings_df)
    return dict(user_tiers_df.iter_rows())

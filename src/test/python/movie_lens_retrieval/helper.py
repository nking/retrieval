import os
from typing import List, Dict
import tensorflow as tf
import polars as pl
from numpy import ndarray

def get_kaggle() -> bool:
  cwd = os.getcwd()
  if "kaggle" in cwd:
    kaggle = True
  else:
    kaggle = False
  return kaggle

def get_project_dir() -> str:
  cwd = os.getcwd()
  head = cwd
  proj_dir = ""
  while head and head != os.sep:
    head, tail = os.path.split(head)
    if tail:  # Add only if not an empty string (e.g., from root or multiple separators)
      if tail == "retrieval":
        proj_dir = os.path.join(head, tail)
        break
  return proj_dir

def get_bin_dir() -> str:
  return os.path.join(get_project_dir(), "bin")

def load_movies_into_polars() -> pl.DataFrame:
    ds = load_movies_into_tfdatadset()
    df = pl.from_dicts(list(ds.as_numpy_iterator()))
    df = df.with_columns([
        pl.col("title").cast(pl.String),
        pl.col("genres").cast(pl.String)
    ])
    return df

def load_movies_into_tfdatadset() -> tf.data.Dataset:
    feature_spec = {
        "movie_id": tf.io.FixedLenFeature([], tf.int64),
        "title": tf.io.FixedLenFeature([], tf.string),
        "genres": tf.io.FixedLenFeature([], tf.string),
    }
    
    def _parse_function(example_proto):
        return tf.io.parse_single_example(example_proto, feature_spec)
    file_paths = [os.path.join(get_project_dir(),
        'src/test/resources/data/movies/movies-00000-of-00001.tfrecord')]
    dataset = tf.data.TFRecordDataset(file_paths)
    parsed_dataset = dataset.map(_parse_function)
    
    return parsed_dataset
    
def load_list_of_globs_into_tfrecords(list_of_globs:List[str], batch_size:int=256):
  
  def load_tfrecords(filepath):
    """Creates a TFRecordDataset, setting compression based on the file path."""
    is_compressed = tf.strings.regex_full_match(filepath, r".*\.gz$")
    compression_type = tf.where(is_compressed, tf.constant("GZIP"),tf.constant(""))
    return tf.data.TFRecordDataset(filepath,  compression_type=compression_type)
    
  files_dataset = tf.data.Dataset.list_files(list_of_globs, shuffle=False                                             )
  
  # 2. Use interleave to read from multiple files concurrently
  # This is the most efficient method for reading many files.
  records_dataset = files_dataset.interleave(
    load_tfrecords,
    cycle_length=tf.data.AUTOTUNE,
    block_length=1,
    num_parallel_calls=tf.data.AUTOTUNE
  )
  #use largest batch size possible while avoiding out-of-memory errors
  dataset = records_dataset.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)
  return dataset

def get_user_tier_stratified_user_and_first_timestamp_from_ratings(
        ratings_df: pl.DataFrame,
        user_tier_map : Dict[int, int],
        sample_size: int = None,
        seed: int = 42
) -> Dict[int, tuple[ndarray, ndarray]]:
    """
    given a ratings_df and a user_tier_map, return a tuple for each user_tier.  The tuple is
    an array of user_ids and a parallel array of the first timestamps of that user in ratings_df.
    If sample_size is not None, then a random sample of those is returned for each tier.
    :param ratings_df: a polars DataFram with columns "user_id", "movie_id",  "rating" and "timestamp"
    :param user_tier_map: a dictionary with key=user_id, value=tier where tier is 0, 1, or 2 for
    the head, torso or tail of the distribution for the number of ratings per user.
    :param sample_size: if given, a random sample of the data are returned per tier
    :param seed: a random number generator seed
    :return: a dictionary with key=tier, value=(user_ids, timestamps) which are np.ndarrays
    """
    out = {}
    for tier in range(0, 3):
        user_set = {u_id for u_id, t in user_tier_map.items() if t == tier}
        unique_users_df = (
            ratings_df
            .filter(pl.col("user_id").is_in(user_set))
            .group_by("user_id")
            .agg(pl.col("timestamp").min().alias("timestamp"))
        )
        if sample_size is not None:
            actual_sample_size = min(sample_size, len(unique_users_df))
            unique_users_df = unique_users_df.sample(n=actual_sample_size,
                seed=seed,
                shuffle=True)
        out[tier] = (unique_users_df["user_id"].to_numpy(), unique_users_df["timestamp"].to_numpy())
    return out
    
def get_random_user_and_first_timestamp_from_ratings(
        ratings_df: pl.DataFrame,
        sample_size: int = 500,
        seed: int = 42
) -> tuple[ndarray, ndarray]:
    """
    Extracts a random sample of unique users and their first timestamp from a Polars DataFrame.
    """
    unique_users_df = (
        ratings_df
        .group_by("user_id")
        .agg(pl.col("timestamp").min().alias("timestamp"))
    )
    
    actual_sample_size = min(sample_size, len(unique_users_df))
    
    sampled_df = unique_users_df.sample(n=actual_sample_size, seed=seed,
        shuffle=True)
    
    user_ids = sampled_df["user_id"].to_numpy()
    timestamps = sampled_df["timestamp"].to_numpy()
    
    return user_ids, timestamps


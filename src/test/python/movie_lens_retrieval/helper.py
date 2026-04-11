import os
from typing import List
import tensorflow as tf
import polars as pl

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


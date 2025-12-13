import polars as pl
import random
import numpy as np
from helper import *
import os

"""
formatting the data into lists for inputs for training a r-ranker with listwise ranking loss.

Hard negative mining is followed to remove ratings "3" and
to partition the rest into positive for ratings > 3 and
negative for ratings < 3 for each user.

Then 80:20 splits are performed upon each user so that they are
in both the train set and the validation set, but the movies they
rated are split between train and validation.

then the data are formatted for candidate inputs for the listwise
re-ranker.
for each of train, validation:
  for each user, 
    for each positive rating:
      NUM_CANDIDATES_PER_LIST-1 are chosen randomly from the
      negative samples.
      a row is written to the output dataframe for the
      the sampled list and their genres and ratings.

NOTE:
- this could be made scalable by using Polars with lazy fram API and streaming,
  and adding Fugue and Pyspark to the pipeline.   Polars is useful for designing
  the vectorized operations, among many things.
- this could be made scalable by rewriting the transforms in TFX and beam.  The
  advantage to this approach would be seamless integration with an existing MLOps
  pipeline including archiving, resource tracking, lineage, etc.

below, can set NUM_CANDIDATES_PER_LIST and NUM_SAMPLES_PER_POS
"""

NUM_CANDIDATES_PER_LIST = 5
NUM_SAMPLES_PER_POS = 2 #for each positive movie rating for a user, make this many lists of length NUM_CANDIDATES_PER_LIST

in_file_pattern = os.path.join(get_project_dir(),
  "src/test/resources/data/sorted_1/ratings_sorted_1_joined*parquet")

df = pl.read_parquet(in_file_pattern)

filtered = df.group_by("user_id").agg(pl.len().alias("rating_count")).filter(pl.col("rating_count") >= 2*NUM_CANDIDATES_PER_LIST).select("user_id").join(df, on="user_id", how="inner")

def agg_columns(filtered0, ratings:List[int]=[4,5]):
  _df = filtered0.filter(pl.col("rating").is_in(ratings))
  df0 = _df.group_by("user_id").agg(pl.col("movie_id"))
  df1 = _df.group_by("user_id").agg(pl.col("genres"))
  df2 = _df.group_by("user_id").agg(pl.col("rating"))
  df3 = _df.select(["user_id", "age"]).unique()
  _user_df = df0.join(df1, on="user_id", how="inner")
  _user_df = _user_df.join(df2, on="user_id", how="inner")
  _user_df = _user_df.join(df3, on="user_id", how="inner")
  #remove any with only 1 movie. later need to split between train, val
  _user_df = _user_df.with_columns([
    pl.col("movie_id").list.len().alias("n_r"),
  ])
  _user_df = _user_df.filter(pl.col("n_r") > 1)
  return _user_df

pos_user_df = agg_columns(filtered, [4,5])
neg_user_df = agg_columns(filtered, [1,2])

#remove users not in both pos and neg. 
# A Left Anti Join (often just "Anti Join")
set_pos = set(pos_user_df["user_id"]) 
set_neg = set(neg_user_df["user_id"])
symm_diff = set_pos ^ set_neg
pos_user_df = pos_user_df.filter(~pl.col("user_id").is_in(symm_diff))
neg_user_df = neg_user_df.filter(~pl.col("user_id").is_in(symm_diff))

def split(df2):
  """
  an 80:20 split for training:validation
  n_r
  """
  df2 = df2.with_columns([
    pl.max_horizontal(
    (pl.col("n_r") * 0.2).cast(pl.Int64),pl.lit(1)
    ).alias("n_eval_r"),
  ])
  df2 = df2.with_columns([
    (pl.col("n_r") - pl.col("n_eval_r")).alias("n_train_r"),
  ])
  # Slice train and validation sets
  train_df = df2.select([
    "user_id", "age",
    pl.col("movie_id").list.slice(0, pl.col("n_train_r")).alias("movie_id"),
    pl.col("genres").list.slice(0, pl.col("n_train_r")).alias("genres"),
    pl.col("rating").list.slice(0, pl.col("n_train_r")).alias("rating"),
  ])
  val_df = df2.select([
    "user_id", "age",
    pl.col("movie_id").list.slice(
      pl.col("n_train_r"), pl.col("n_eval_r"))
      .alias("movie_id"),
    pl.col("genres").list.slice(
      pl.col("n_train_r"), pl.col("n_eval_r"))
      .alias("genres"),
    pl.col("rating").list.slice(
      pl.col("n_train_r"), pl.col("n_eval_r"))
      .alias("rating"),
  ])
  return train_df, val_df 

def lengths_are_same(train_df, val_df):
  _df = train_df.filter(
    (pl.col("movie_id").list.len() != pl.col("rating").list.len())
    | (pl.col("movie_id").list.len() != pl.col("genres").list.len()))
  print(f'N rows in error in train ={_df.count()}')
  _df = val_df.filter(
    (pl.col("movie_id").list.len() != pl.col("rating").list.len())
    | (pl.col("movie_id").list.len() != pl.col("genres").list.len())
  )
  print(f'N rows in error in val ={_df.count()} out of {val_df.count()}')

def rename_cols(df2, prefix:str="pos"):
  df2 = df2.rename({
    "movie_id": f"movie_id_{prefix}",
    "genres": f"genres_{prefix}",
    "rating": f"rating_{prefix}",
  })
  return df2

pos_train_df, pos_val_df = split(pos_user_df)
neg_train_df, neg_val_df = split(neg_user_df)

def join_pos_neg(pos_df2, neg_df2):
  pos_df2 = rename_cols(pos_df2, "pos")
  neg_df2 = rename_cols(neg_df2, "neg")
  joined = pos_df2.join(neg_df2, on="user_id", how="inner")
  joined = joined.drop(['age_right'])
  return joined

#merge back together with new column names
# ['user_id', 'age', 
#  'movie_id_pos', 'genres_pos', 'rating_pos', 
#  'movie_id_neg', 'genres_neg', 'rating_neg']
train_df = join_pos_neg(pos_train_df, neg_train_df)
val_df = join_pos_neg(pos_val_df, neg_val_df)

train_df2 = train_df.explode(['movie_id_pos', 'genres_pos', 'rating_pos'])
val_df2 = val_df.explode(['movie_id_pos', 'genres_pos', 'rating_pos'])
# for each pos_[train or val]_df, 
#  we choose NUM_CANDIDATES_PER_LIST-1 from neg_[train or val]_df
#  and put them in [train or val] df

K = NUM_CANDIDATES_PER_LIST - 1
struct_dtype = pl.Struct([
  pl.Field("movies", pl.List(pl.List(pl.Int64))),
  pl.Field("ratings", pl.List(pl.List(pl.Int64))),
  pl.Field("genres", pl.List(pl.List(pl.String)))
])

def create_samples(s: pl.Series) -> pl.Series:
  #the pos are scalars, the neg are arrays
  pos_id_series = s.struct.field("movie_id_pos")
  pos_rating_series = s.struct.field("rating_pos")
  pos_genres_series = s.struct.field("genres_pos")
  neg_ids_series = s.struct.field("movie_id_neg")
  neg_ratings_series = s.struct.field("rating_neg")
  neg_genres_series = s.struct.field("genres_neg")
  #output number of rows needs to match input, so we
  #expand on inner dimension.  in other words if we have 10 rows, and K=3, and N=2, the output shape is [10, 3, 2]
  output_movies_list = None
  output_ratings_list = None
  output_genres_list = None
  for i_sample in range(NUM_SAMPLES_PER_POS):
    output_movies = []
    output_ratings = []
    output_genres = []
    for pos_id, pos_rating, pos_genres, neg_ids, neg_ratings,\
      neg_genres in zip(pos_id_series, pos_rating_series,\
      pos_genres_series,  neg_ids_series, neg_ratings_series,\
      neg_genres_series):
      if len(neg_ids) < K:
        output_movies.append((K+1)*[-1])
        output_ratings.append((K+1)*[-1])
        output_genres.append((K+1)*[-1])
        continue
      else:
        chosen_indices = random.sample(range(len(neg_ids)), K)
      movie_ids = [pos_id] + [neg_ids[i] for i in chosen_indices]
      ratings = [pos_rating] + [neg_ratings[i] for i in chosen_indices]
      genres = [pos_genres] + [neg_genres[i] for i in chosen_indices]
      output_movies.append(movie_ids)
      output_ratings.append(ratings)
      output_genres.append(genres)
    output_movies = np.expand_dims(output_movies, axis=1)
    output_ratings = np.expand_dims(output_ratings, axis=1)
    output_genres = np.expand_dims(output_genres, axis=1)
    #print(f'i={i_sample}: shape={np.shape(output_movies)}')
    if output_movies_list is None:
      output_movies_list = output_movies.copy()
      output_ratings_list = output_ratings.copy()
      output_genres_list = output_genres.copy()
    else:
      output_movies_list = np.concatenate([output_movies_list, output_movies], axis=1)
      output_ratings_list = np.concatenate([output_ratings_list, output_ratings], axis=1)
      output_genres_list = np.concatenate([output_genres_list, output_genres], axis=1)
  #print(f'shapes={np.shape(output_movies_list)}, {np.shape(output_ratings_list)}, {np.shape(output_genres_list)}')
  return pl.Series(
    values=[
      {'movies': m, 'ratings': r, 'genres':g}
        for m, r, g in zip(output_movies_list.tolist(), output_ratings_list.tolist(), output_genres_list.tolist())
     ])

def make_samples(df2):
  samples = df2.with_columns(
    pl.struct([
      pl.col("movie_id_pos"), pl.col('rating_pos'),
      pl.col('genres_pos'),
      pl.col("movie_id_neg"), pl.col('rating_neg'),
      pl.col("genres_neg") 
    ])
    .map_batches(create_samples, return_dtype=struct_dtype)
    .alias("sample_struct")
  ).unnest("sample_struct")
  samples = (
    samples.explode(['movies', 'ratings', 'genres'])
  )
  samples = samples.select(['user_id', 'age', 'movies', 'ratings', 'genres'])
  samples = samples.filter(
    pl.col("movies").list.eval(pl.element() != -1).list.any()
  )
  return samples

print(f'writing train samples')
train_samples = make_samples(train_df2)
print(f'writing validation samples')
val_samples = make_samples(val_df2)

#use py_arrow=True for huggingface datasets
train_samples.write_parquet(file=os.path.join(get_bin_dir(), "train-00000-of-00001.parquet"), use_pyarrow=True)
val_samples.write_parquet(file=os.path.join(get_bin_dir(),"validation-00000-of-00001.parquet"), use_pyarrow=True)

#['user_id', 'age', 'movies', 'ratings', 'genres']
print(f'wrote parquet files to {get_bin_dir()}')
print(f'columns are {train_samples.columns}')

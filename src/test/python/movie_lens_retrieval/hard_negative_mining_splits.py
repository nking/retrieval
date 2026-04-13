from typing import Tuple, Dict

import polars as pl
import random
from helper import *
from movie_lens_retrieval.Retriever import Retriever
import os

"""
TODO:  correct to add to the exact hard negatives,
a random selection from "all movies - recommended movies - seen in train - the positive example"

formatting the data into lists for inputs for training a re-ranker with listwise ranking loss.

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
- this could be made scalable by using Polars with lazy frame API and streaming,
  and adding Fugue and Pyspark to the pipeline.   Polars is useful for designing
  the vectorized operations, among many things.
- this could be made scalable by rewriting the transforms in TFX and beam.  The
  advantage to this approach would be seamless integration with an existing MLOps
  pipeline including archiving, resource tracking, lineage, etc.

below, can set NUM_CANDIDATES_PER_LIST and NUM_SAMPLES_PER_POS
"""

NUM_CANDIDATES_PER_LIST = 5
NUM_SAMPLES_PER_POS = 2 #for each positive movie rating for a user, make this many lists of length NUM_CANDIDATES_PER_LIST
max_top_k = 500
#there are pos, neg counts=(1548, 1548) when max_top_k=1000,
# that is, 25% of users could use re-ranked recommendations
#the pos, neg counts=(4041, 4041) when max_top_k=10000, which is 67% of users
#when max_top_k=100, 32/800187 is negligible % of users needing re-ranker

df = (pl.concat([
        pl.read_parquet(os.path.join(get_project_dir(),
        "src/test/resources/data/train/ratings_joined*.parquet")),
        pl.read_parquet(os.path.join(get_project_dir(),
        "src/test/resources/data/val/ratings_joined*.parquet"))])
      )
print(f'count={df["movie_id"].count()}')

filtered = df.group_by("movie_id").agg(pl.len().alias("rating_count")).filter(pl.col("rating_count") >= 2*NUM_CANDIDATES_PER_LIST).select("movie_id").join(df, on="movie_id", how="inner")

print(f'count after filter for too few ratings ={filtered["movie_id"].count()}')

def agg_columns(filtered0, ratings:List[int]=[4,5]):
  _df = filtered0.filter(pl.col("rating").is_in(ratings))
  df0 = _df.group_by("movie_id").agg([pl.col("movie_id"), pl.col("genres"), pl.col("rating")])
  df1 = _df.select(["movie_id", "age"]).unique()
  _user_df = df0.join(df1, on="movie_id", how="inner")
  _user_df = _user_df.with_columns([
    pl.col("movie_id").list.len().alias("n_r"),
  ])
  _user_df = _user_df.filter(pl.col("n_r") > 1)
  return _user_df

pos_user_df = agg_columns(filtered, [4,5])
neg_user_df = agg_columns(filtered, [1,2])

def create_retrievalreranker():
  user_movie_models_dir = os.path.join(os.path.join(get_project_dir(),
    "src/main/resources/serving_models/user_movie_model"))
  movie_inputs = os.path.join(get_project_dir(),
    "src/test/resources/data/movie_emb_inp/tfrecord*.gz")
  user_inputs = os.path.join(get_project_dir(),
    "src/test/resources/data/user_emb_inp/tfrecord*.gz")
  movies_mean_ratings_pivot = os.path.join(get_project_dir(),
    "src/test/resources/data/ratings_bayesian_shrinkage/mean_ratings_tfrecord*.gz")
  movies_predictions_pivot = os.path.join(get_project_dir(),
    "src/test/resources/data/ratings_bayesian_shrinkage/mm_predictions_tfrecord*.gz")
  rr = Retriever(user_movie_saved_model_dir = user_movie_models_dir,
    movies_path = movie_inputs, users_path=user_inputs,
    movies_pivot_path=movies_mean_ratings_pivot,
    max_k= max_top_k, movies_batch_size=256)
  return rr
  
rr = create_retrievalreranker()

def calculate_hard_negative_movies(row) -> Dict[str, List[int] | List[str]]:
  top_k = max_top_k #3 * row['n_r']
  sim_movies = rr.get_movies_given_users(
    {'movie_id': row['movie_id'], 'age': row['age'] },
    top_k = top_k)[0]
  m_dict = {m_id: idx for idx, m_id in enumerate(row['movie_id'])}
  movies = []
  ratings = []
  genres = []
  for m_id in sim_movies:
    if len(movies) == top_k:
      break
    if m_id in m_dict:
      idx = m_dict[m_id]
      movies.append(row['movie_id'][idx])
      ratings.append(row['rating'][idx])
      genres.append(row['genres'][idx])
  return {'m_list': movies, 'r_list': ratings, 'g_list': genres}

#columns: ['movie_id', 'movie_id', 'genres', 'rating', 'age', 'n_r', 'movie_id_hard']
neg_user_df = neg_user_df.with_columns(
  pl.struct(['movie_id', 'age', 'movie_id', 'rating', 'genres'])
  .map_elements(calculate_hard_negative_movies,
    return_dtype=pl.Struct({
        "m_list": pl.List(pl.Int64),
        "r_list": pl.List(pl.Int64),
        "g_list": pl.List(pl.String)
    })
  ).alias('negatives')
)
neg_user_df = neg_user_df.unnest("negatives")
neg_user_df = neg_user_df.drop(['movie_id', 'rating', 'genres'])
neg_user_df = neg_user_df.rename({'m_list': 'movie_id', 'r_list': 'rating', 'g_list': 'genres'})
neg_user_df = neg_user_df.with_columns([
  pl.col("movie_id").list.len().alias("n_r"),
])
neg_user_df = neg_user_df.filter(pl.col("n_r") >= (NUM_CANDIDATES_PER_LIST-1))

#remove users not in both pos and neg.
# A Left Anti Join (often just "Anti Join")
set_pos = set(pos_user_df["movie_id"])
set_neg = set(neg_user_df["movie_id"])
symm_diff = set_pos ^ set_neg
pos_user_df = pos_user_df.filter(~pl.col("movie_id").is_in(symm_diff))
neg_user_df = neg_user_df.filter(~pl.col("movie_id").is_in(symm_diff))
#columns: ['movie_id', 'movie_id', 'genres', 'rating', 'age', 'n_r']
print(f'pos, neg counts={pos_user_df["movie_id"].count(), neg_user_df["movie_id"].count()}')

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
    "movie_id", "age",
    pl.col("movie_id").list.slice(0, pl.col("n_train_r")).alias("movie_id"),
    pl.col("genres").list.slice(0, pl.col("n_train_r")).alias("genres"),
    pl.col("rating").list.slice(0, pl.col("n_train_r")).alias("rating"),
  ])
  val_df = df2.select([
    "movie_id", "age",
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

print(f'train: pos, neg counts={pos_train_df["movie_id"].count(), neg_train_df["movie_id"].count()}')
print(f'validation: pos, neg counts={pos_val_df["movie_id"].count(), neg_val_df["movie_id"].count()}')

def join_pos_neg(pos_df2, neg_df2):
  pos_df2 = rename_cols(pos_df2, "pos")
  neg_df2 = rename_cols(neg_df2, "neg")
  joined = pos_df2.join(neg_df2, on="movie_id", how="inner")
  joined = joined.drop(['age_right'])
  return joined

#merge back together with new column names
# ['movie_id', 'age',
#  'movie_id_pos', 'genres_pos', 'rating_pos', 
#  'movie_id_neg', 'genres_neg', 'rating_neg']
train_df = join_pos_neg(pos_train_df, neg_train_df)
val_df = join_pos_neg(pos_val_df, neg_val_df)

print(f'train: counts={train_df["movie_id"].count()}')
print(f'validation: counts={val_df["movie_id"].count()}')

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

def create_samples(row) -> Dict[str, List[int] | List[str]]:
  output_movies_list = []
  output_ratings_list = []
  output_genres_list = []
  for i_sample in range(NUM_SAMPLES_PER_POS):
    if len(row['movie_id_neg']) < K:
      output_movies = (K + 1) * [-1]
      output_ratings = (K + 1) * [-1]
      output_genres = (K + 1) * [""]
    else:
      chosen_indices = random.sample(range(len(row['movie_id_neg'])), K)
      output_movies = [row['movie_id_pos']] + [row['movie_id_neg'][i] for i in chosen_indices]
      output_ratings = [row['rating_pos']] + [row['rating_neg'][i] for i in chosen_indices]
      output_genres = [row['genres_pos']] + [row['genres_neg'][i] for i in chosen_indices]
    output_movies_list.append(output_movies)
    output_ratings_list.append(output_ratings)
    output_genres_list.append(output_genres)
  return {'movies': output_movies_list, 'ratings': output_ratings_list,
        'genres': output_genres_list}

def make_samples(df2):
  samples = df2.with_columns(
    pl.struct(["movie_id_pos", 'rating_pos', 'genres_pos',
      "movie_id_neg", 'rating_neg', "genres_neg"
    ])
    .map_elements(create_samples, return_dtype=struct_dtype)
    .alias("sample_struct")
  ).unnest("sample_struct") #unnest makes the struct into new columns
  #print(f'SHAPE of unested: {np.shape(samples["movies"][0])}')
  samples = (
    samples.explode(['movies', 'ratings', 'genres']) #explode expands the columns into new rows
  )
  #print(f'SHAPE of unested: {np.shape(samples["movies"][0])}')
  samples = samples.select(['movie_id', 'age', 'movies', 'ratings', 'genres'])
  samples = samples.filter(
    pl.col("movies").list.eval(pl.element() != -1).list.any()
  )
  return samples

print(f'sampling train n_rows={train_df2["movie_id"].count()}')
train_samples = make_samples(train_df2)
print(f'=> train_samples n_rows={train_samples["movies"].count()}')

print(f'sampling validation n_rows={val_df2["movie_id"].count()}')
val_samples = make_samples(val_df2)
print(f'=> val_samples n_rows={val_samples["movies"].count()}')

#use py_arrow=True for huggingface datasets
train_samples.write_parquet(file=os.path.join(get_bin_dir(), "train-00000-of-00001.parquet"), use_pyarrow=True)
val_samples.write_parquet(file=os.path.join(get_bin_dir(),"validation-00000-of-00001.parquet"), use_pyarrow=True)

#write train and val having same ids in both:
set_tr = set(train_samples["movie_id"])
set_val = set(val_samples["movie_id"])
print(f'n_rows w/ same users in both train and val ={len(set_tr & set_val)}')

symm_diff = set_tr ^ set_val
train_samples_2 = train_samples.filter(~pl.col("movie_id").is_in(symm_diff))
val_samples_2 = val_samples.filter(~pl.col("movie_id").is_in(symm_diff))
train_samples_2.write_parquet(file=os.path.join(get_bin_dir(), "train2-00000-of-00001.parquet"), use_pyarrow=True)
val_samples_2.write_parquet(file=os.path.join(get_bin_dir(),"validation2-00000-of-00001.parquet"), use_pyarrow=True)

print(f'train_samples_2, val_samples_2 counts={train_samples_2["movie_id"].count(), val_samples_2["movie_id"].count()}')

#making a small sample for tests
user_ids = train_samples_2["movie_id"].unique()
n = min(100, user_ids.count())
print(f'=> n={n} for testsmall')

user_ids = user_ids.sample(n=n, seed=0)
user_ids = user_ids.implode()
train_samples_2 = train_samples_2.filter(pl.col("movie_id").is_in(user_ids))
train_samples_2 = train_samples_2.sample(n=n, seed=0)
val_samples_2 = val_samples_2.filter(pl.col("movie_id").is_in(user_ids))
val_samples_2 = val_samples_2.sample(n=n, seed=0)
train_samples_2.write_parquet(file=os.path.join(get_bin_dir(), "trainsmall-00000-of-00001.parquet"), use_pyarrow=True)
val_samples_2.write_parquet(file=os.path.join(get_bin_dir(),"validationsmall-00000-of-00001.parquet"), use_pyarrow=True)

## ================ do same except no splits for the test data ======
in_file_pattern = os.path.join(get_project_dir(),
  "src/test/resources/data/test/ratings_joined*parquet")
df = pl.read_parquet(in_file_pattern)
filtered = (df.group_by("movie_id").agg(pl.len().alias("rating_count"))
  .filter(pl.col("rating_count") >= 2*NUM_CANDIDATES_PER_LIST).select("movie_id")
  .join(df, on="movie_id", how="inner"))
pos_user_df = agg_columns(filtered, [4,5])
neg_user_df = agg_columns(filtered, [1,2])
set_pos = set(pos_user_df["movie_id"])
set_neg = set(neg_user_df["movie_id"])
symm_diff = set_pos ^ set_neg
pos_user_df = pos_user_df.filter(~pl.col("movie_id").is_in(symm_diff))
neg_user_df = neg_user_df.filter(~pl.col("movie_id").is_in(symm_diff))
test_df = join_pos_neg(pos_train_df, neg_train_df)
test_df2 = test_df.explode(['movie_id_pos', 'genres_pos', 'rating_pos'])
test_samples = make_samples(test_df2)
print(f'writing test samples')
test_samples.write_parquet(file=os.path.join(get_bin_dir(), "test-00000-of-00001.parquet"), use_pyarrow=True)
test_samples_2 = test_samples.filter(pl.col("movie_id").is_in(user_ids))
test_samples_2 = test_samples_2.sample(n=n, seed=0)
test_samples_2.write_parquet(file=os.path.join(get_bin_dir(), "testsmall-00000-of-00001.parquet"), use_pyarrow=True)

#['movie_id', 'age', 'movies', 'ratings', 'genres']
print(f'wrote train and validatin parquet files to {get_bin_dir()}')
print(f'columns are {train_samples.columns}')

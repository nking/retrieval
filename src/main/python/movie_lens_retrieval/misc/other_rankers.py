import numpy as np
import scipy.stats as st
from scipy.stats import dirichlet
import polars as pl
import pandas as pd

z_95 = st.norm.ppf(1 - (1 - 0.95) / 2)

def wilson_score(n_pos, n, confidence=0.95):
  if not isinstance(n_pos, int):
    raise TypeError("n_pos must be an int")
  if n == 0: return 0
  z = z_95 if confidence==0.95 else st.norm.ppf(1 - (1 - confidence) / 2)
  phat = n_pos / n
  numerator = phat + z * z / (2 * n) - z * np.sqrt(
    (phat * (1 - phat) + z * z / (4 * n)) / n)
  denominator = 1 + z * z / n
  return numerator / denominator
 
def wilson_score_df(df, n_pos_col_name:str, n_col_name:str, confidence:float=0.95):
  """
  the bernoulli standard, wilson score,
  Used by: Reddit (historically), Hacker News, Yelp.
  :param df:
  :param n_pos_col_name: column name for total count of positive votes
  :param n_col_name: column name for total count of all votes, that is positive + negative
  :param confidence: confidence interval to use, else 0.95 by default
  :return:
  """
  if not isinstance(df, pl.DataFrame) and not isinstance(df, pd.DataFrame):
    raise TypeError("df type should be polars or pandas DataFrame")
  
  z = z_95 if confidence == 0.95 else st.norm.ppf(1 - (1 - confidence) / 2)
  if isinstance(df, pd.DataFrame):
    #using Laplace smoothing to avoid 0 probabilities and divide by 0s.  add 1 to numer and 5 to denom
    df['p_hat'] = (
      (df[n_pos_col_name] +1) / (df[n_col_name]+5)
    )
    df['numerator'] = (
      df['p_hat'] + z * z / (2 * df[n_col_name])
      - z * np.sqrt((df['p_hat'] * (1 - df['p_hat']) + z * z / (4 * df[n_col_name])) / df[n_col_name])
    )
    df['denominator'] = (
      1 + z * z / df[n_col_name]
    )
    df['wilson_score'] = (
      df['denominator'] / df['numerator']
    )
    df.drop(columns=['numerator', 'denominator', 'p_hat'], inplace=True)
    df_sorted = df.sort_values('wilson_score', ascending=False)
    df.drop(columns=['wilson_score'], inplace=True)
    return df_sorted
  elif isinstance(df, pl.DataFrame):
    # using Laplace smoothing to avoid 0 probabilities and divide by 0s.  add 1 to numer and 5 to denom
    df = df.with_columns(
      ((pl.col(n_pos_col_name)+1)/ (pl.col(n_col_name)+5)).alias("p_hat")
    )
    #df = df.with_columns(
    #  pl.when(pl.col(n_col_name) == 0).then(0.0)
    #    .otherwise(pl.col('p_hat')).alias("p_hat")
    #)
    df = df.with_columns(
      (pl.col('p_hat') + z * z / (2 * pl.col(n_col_name))
      - z * ((pl.col('p_hat') * (1 - pl.col('p_hat')) + z * z / (4 * pl.col(n_col_name))) / pl.col(n_col_name)).sqrt()
      ).alias("numerator")
    )
    df = df.with_columns(
      (1 + z * z / pl.col(n_col_name)
       ).alias("denominator")
    )
    df = df.with_columns(
      (pl.col('denominator')/pl.col('numerator')).alias("wilson_score")
    )
    df = df.with_columns(
      pl.when(pl.col(n_col_name) == 0).then(0.0)
      .otherwise(pl.col('wilson_score')).alias("wilson_score")
    )
    df = df.select(
      pl.all().exclude('numerator', 'denominator', 'p_hat')
    )
    df_sorted = df.sort('wilson_score', descending=True)
    return df_sorted
  
def thompson_sampling_with_dirichlet_prior(df, seed=None):
  """
  performs a random sample for each movie using a dirchlet prior, so that the
  returned sorted ratings occassionally rank the lower frequency movies high enough to
  be visible within top.
  
  used by Netflix, Amazon New Product Recommendations) and in feeds like TikTok/Reels.
  :param df: dataframe with columns "1", "2", "3", "4", "5" for the total counts of each rating category
    where each row is for a single movie.
       e.g.
        df = pl.DataFrame({
          'title': ['popular', 'high_but_few_ratings', 'loved_and_hated', 'hated', 'new_unrated'],
          'movie_id': [1, 2, 3, 4, 5],
          '1': [500,   2,  4000, 800, 0],
          '2': [100,   1,  1000, 100, 0],
          '3': [200,   0,  500,   50, 0],
          '4': [3000,  5,  1000,  20, 0],
          '5': [8000, 40,  4000,  10, 1]
        })
  :param seed:
  :return:
  """
  if not isinstance(df, pl.DataFrame) and not isinstance(df, pd.DataFrame):
    raise TypeError("df type should be polars or pandas DataFrame")
  
  if seed:
    np.random.seed(seed)
  
  if isinstance(df, pd.DataFrame):
    laplace_smoothing = np.array([1, 1, 1, 1, 1])
    stars = np.array([1, 2, 3, 4, 5])
    def get_thompson_sample(row, prior):
      counts = row[['1', '2', '3', '4', '5']].values.astype(float)
      posterior_alphas = counts + laplace_smoothing
      sample_probs = dirichlet.rvs(posterior_alphas, size=1)[0]
      random_score = np.dot(sample_probs, stars)
      return random_score
    df['ranking_score'] = df.apply(get_thompson_sample, args=(laplace_smoothing,), axis=1)
    df_sorted = df.sort_values('ranking_score', ascending=False)
    df.drop(columns=['ranking_score'], inplace=True)
    return df_sorted
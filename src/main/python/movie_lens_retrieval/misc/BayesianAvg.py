from typing import Union

import polars as pl
import pandas as pd

"""
Note, Google's Gemini was used to find and supplement information.
"""

#TODO: consider adding impl for tf tensors

class BayesianShrinkageEstimator:
  """
  an implementation of the IMDB ratings as reference by Gemini.
  see "How do you calculate the IMDb rating displayed on a title page?"
  on https://help.imdb.com/article/imdb/track-movies-tv/ratings-faq/G67Y87TFYYP6TWAV#
  
  
  posterior mean theta_est is best guess for the rating.
  bayesian shrinkage estimator is linear combination of observed average x_bar and the
  prior mean mu_0.
  theta_est = w * x_bar + (1 - w) * mu_0.
  where
  x_bar is average of the movie (likelihood)
  mu_0 is the global average (prior)          <=== C
  w = weight, or trust in the movie's data.
    = 1 / variance
    = (sigma_0**2)/( sigma_0**2 + (sigma**2/n) )
  where
    data variance is sigma_0**2 = precision of the data = v
    and prior variance is sigma**2/n = m
    w = v/(v + m)
  Then theta_est = (v/(v + m)) * x_bar + (1 - (v/(v + m))) * mu_0.
  
  The IMDB formula is:
    weighted rating (WR) = (v / (v + m)) x R + (m / (v + m)) x C
      Where:
      R = average for the title (mean) = (rating)
      v = number of ratings for the title = (ratings)
      m = minimum ratings required to be listed in the Top Rated 250 chart (currently 25,000)
      C = the mean rating across the whole report
      
  (1 - (v/(v + m))) * mu_0 =  (m / (v + m)) x C
  C = ((v+m)/m - (v/m)) * mu_0
    = (m/m) * mu_0
    = mu_0
    
  Simplified:
     theta_est = WR = ((v * x_bar) + m * mu_0)/(v + m)
     
  solves the Bias-Variance Tradeoff:
    unweighted Average: Unbiased, but Massive Variance (scores jump wildly).
    Bayesian Estimate: Slightly Biased (towards the mean), but Low Variance (scores are stable).
  """
  def __init__(self, data: Union[pl.DataFrame, pd.DataFrame], m=20):
    if isinstance(data, pl.DataFrame):
      type = "polars"
    elif isinstance(data, pd.DataFrame):
      type = "pandas"
    else:
      raise TypeError("data type should be polars or pandas DataFrame")
    
    eps = 1E-9
    df = data
    if type == "pandas":
      df['total_votes'] = df[['1', '2', '3', '4', '5']].sum(axis=1).astype('float64')
      df.loc[df['total_votes'] == 0, 'total_votes'] = eps
      df['movie_ratings_mean'] = (
        (df['1'] * 1 + df['2'] * 2 + df['3'] * 3 + df['4'] * 4 + df[
          '5'] * 5) / df['total_votes']
      )
      C = df['movie_ratings_mean'].mean() #prior
      m = df['total_votes'].quantile(0.75)
      print(f"C={C}, m={m}")
      v = df['total_votes']
      R = df['movie_ratings_mean'] #likelihood
      df["weighted_rating"] = (
        (v / (v + m) * R) + (m / (v + m) * C)
      )
      self.df_sorted = df.sort_values('weighted_rating', ascending=False)
    else:
      df = df.with_columns(
        pl.sum_horizontal("1", "2", "3", "4", "5").cast(pl.Float64).alias("total_votes")
      )
      df = df.with_columns(
        pl.when(pl.col("total_votes") == 0.).then(eps)
        .otherwise(pl.col("total_votes"))
      )
      df = df.with_columns(
        ((pl.col('1') * 1 + pl.col('2') * 2 + pl.col('3') * 3 + pl.col('4') * 4 + pl.col('5') * 5)
          / pl.col('total_votes')).alias("movie_ratings_mean")
      )
      df = df.with_columns(
        (pl.col("movie_ratings_mean").fill_nan(0.))
      )
      C = df['movie_ratings_mean'].mean()  # prior
      m = df['total_votes'].quantile(0.75, interpolation='linear')
      print(f"C={C}, m={m}")
      #v = df['total_votes']
      #R = df['movie_ratings_mean']  # likelihood
      df=df.with_columns(
        (
          (pl.col('total_votes') / (pl.col('total_votes') + m) * pl.col('movie_ratings_mean'))
          + (m / (pl.col('total_votes') + m) * C)
        ).alias("weighted_rating")
      )
      self.df_sorted = df.sort('weighted_rating', descending=True)
      
  def get_top(self, top:int=10):
    return self.df_sorted.head(top)
  
class BayesianAvg:
  def __init__(self, data:Union[pl.DataFrame, pd.DataFrame], m=20):
    """
    
    :param data: polars dataframe in format having columns movie_id, and "1","2","3","4","5"
    where "1","2","3","4","5" are the numbers of ratings for that movie in that star
    category.
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
    """
    if isinstance(data, pl.DataFrame):
      type = "polars"
    elif isinstance(data, pd.DataFrame):
      type = "pandas"
    else:
      raise TypeError("data type should be polars or pandas DataFrame")
    
    eps = 1E-9
    df = data
    if type == "pandas":
      df['total_votes'] = df[ ['1', '2', '3', '4', '5']].sum(axis=1).astype('float64')
      df.loc[df['total_votes'] == 0, 'total_votes'] = eps
      # Calculate raw arithmetic average.  this==expectation for Normal, Poisson, and Bernoulli
      df['likelihood'] = (
        (df['1'] * 1 + df['2'] * 2 + df['3'] * 3 + df['4'] * 4 + df['5'] * 5) / df['total_votes']
      )
      #sum over all movies, the number of a rating category
      # length of number of ratings categories
      global_counts = df[['1', '2', '3', '4', '5']].sum()
      # scalar:
      total_global_votes = global_counts.sum()
      # probablity density as a normalized histogram:
      global_distribution = global_counts / total_global_votes
      alpha_vector = global_distribution * m
      a1, a2, a3, a4, a5 = alpha_vector
      df["numerator"] = (
        (df['1'] + a1) * 1 + (df['2'] + a2) * 2 + (df['3'] + a3) * 3 +
        (df['4'] + a4) * 4 + (df['5'] + a5) * 5
      )
      df["denominator"] = (
        (df['1'] + df['2'] + df['3'] + df['4'] + df['5']) * m
      )
      df["dirichlet_rating"] = (df['numerator'] / df["denominator"])
      df.loc[df['dirichlet_rating'] == float('inf'), 'dirichlet_rating'] = 0.
      df.drop(columns=['numerator', 'denominator'], inplace=True)
      self.df_sorted = df.sort_values('dirichlet_rating', ascending=False)
    else:
      df = df.with_columns(
        pl.sum_horizontal("1", "2", "3", "4", "5").cast(pl.Float64).alias("total_votes")
      )
      df = df.with_columns(
        pl.when(pl.col("total_votes") == 0.).then(eps)
        .otherwise(pl.col("total_votes"))
      )
      df = df.with_columns(
        ((pl.col('1') * 1 + pl.col('2') * 2 + pl.col('3') * 3 + pl.col('4') * 4 + pl.col('5') * 5)
          / pl.col('total_votes')).alias("likelihood")
      )
      df = df.with_columns(
        (pl.col("likelihood").fill_nan(0.))
      )
      
      global_counts = df.select(pl.sum("1"), pl.sum("2"), pl.sum("3"), pl.sum("4"), pl.sum("5")).select(pl.concat_list(pl.all())).item()
      total_global_votes = global_counts.sum()
      global_distribution = global_counts / total_global_votes
      alpha_vector = global_distribution * m
      a1, a2, a3, a4, a5 = alpha_vector
      df = df.with_columns(
        ((pl.col('1') +a1) * 1 + (pl.col('2') +a2) * 2 +
        (pl.col('3') +a3) * 3 + (pl.col('4') +a4) * 4 + (pl.col('5') +a5) * 5)
        .alias("numerator")
      )
      df = df.with_columns(
        ((pl.col('1') + pl.col('2') + pl.col('3') + pl.col('4') + pl.col('5'))*m)
        .alias("denominator")
      )
      df = df.with_columns(
        (pl.col('numerator')/pl.col("denominator")).alias("dirichlet_rating")
      )
      df = df.select(
        pl.all().exclude('numerator', 'denominator')
      )
      df = df.with_columns(
        pl.when(pl.col("dirichlet_rating").is_infinite())
          .then(0.0)
          .otherwise(pl.col("dirichlet_rating"))
          .alias("dirichlet_rating")
      )
      self.df_sorted = df.sort('dirichlet_rating', descending=True)
  
  def get_top(self, top:int=10):
    return self.df_sorted.head(top)
  
  
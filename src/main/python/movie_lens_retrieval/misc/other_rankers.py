import numpy as np
import scipy.stats as st
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
 
def wilson_score_df(df, n_pos_col_name, n_col_name, confidence=0.95):
  z = z_95 if confidence == 0.95 else st.norm.ppf(1 - (1 - confidence) / 2)
  if isinstance(df, pd.DataFrame):
    df['p_hat'] = (
      df[n_pos_col_name] / df[n_col_name]
    )
    condition = (df[n_col_name] == 0)
    df.loc[condition, 'p_hat'] = 0.0
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
    df = df.with_columns(
      (pl.col(n_pos_col_name)/ pl.col(n_col_name)).alias("p_hat")
    )
    df = df.with_columns(
      pl.when(pl.col(n_col_name) == 0).then(0.0)
        .otherwise(pl.col('p_hat')).alias("p_hat")
    )
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
  
  

import unittest

from movie_lens_retrieval.misc.other_rankers import *
import polars as pl
import pandas as pd
from helper import *
from array_record.python import array_record_module
import msgpack

pl.Config.set_tbl_width_chars(200) # Increases total "paper width" of the table
pl.Config.set_tbl_cols(-1)        # Always show all columns

class BayesianAvgTest(unittest.TestCase):
  
  def setUp(self):
    self.data = {
      'title': ['loved_many', 'loved_few', 'loved_and_hated', 'hated', 'new_unrated',  'hated_few'],
      'movie_id': [1, 2, 3, 4, 5, 6],
      '1': [500,   2,  4000, 800, 0, 40],
      '2': [100,   1,  1000, 100, 0, 5],
      '3': [200,   0,  500,   50, 0, 0],
      '4': [3000,  5,  1000,  20, 0, 0],
      '5': [8000, 40,  4000,  10, 0, 0]
    }
    self.movie_reader = array_record_module.ArrayRecordReader(
        os.path.join(get_project_dir(),
            "src/test/resources/data/movie_ratings_pivot_table/movie_ratings_pivot_table.array_record"))
    movies_df = load_movies_into_polars()
    # print(movies_df)
    batch_size = 1024
    batch_bytes = self.movie_reader.read([x for x in range(0, batch_size)])
    data = [msgpack.unpackb(b, use_list=False) for b in batch_bytes]
    pivot_df = pl.from_dicts(data)
    self.movie_pivot_df = pivot_df.join(movies_df.select(["movie_id", "title"]),
        on="movie_id", how="left"
    )
      
  def tearDown(self):
      self.movie_reader.close()
      
  def test_wilson_score(self):
    print('\nwilson score')
    df = pl.DataFrame(self.data)
    df = df.with_columns(
      pl.sum_horizontal("1", "2", "3", "4", "5").cast(pl.Float64).alias(
        "total_votes")
    )
    df = df.with_columns(
      pl.sum_horizontal("4","5").cast(pl.Float64).alias(
        "total_pos_votes")
    )
    df_ranked = wilson_score_df(df, 'total_pos_votes', 'total_votes')
    print(df_ranked)
    
    s1 = wilson_score(5, 5)
    s2 = wilson_score(95, 100)
    self.assertLess(s1, s2)
    
    # calc manually to check results
    z = 1.96 # 95% confidence level
    
    df_result = (
        df
        .with_columns(
            # 1. Total ratings
            n=pl.sum_horizontal(['1', '2', '3', '4', '5'])
        )
        .with_columns(
            # 2. Normalized average rating (p)
            p=(
                      ((1 * pl.col('1') + 2 * pl.col('2') + 3 * pl.col(
                          '3') + 4 * pl.col('4') + 5 * pl.col('5'))
                       / pl.col('n')) - 1
              ) / 4
        )
        .with_columns(
            # 3. Wilson Score formula
            wilson_score=pl.when(pl.col('n') == 0)
            .then(0.0)
            .otherwise(
                (pl.col('p') + (z ** 2) / (2 * pl.col('n')) - z * (
                            (pl.col('p') * (1 - pl.col('p')) / pl.col('n')) + (
                                z ** 2) / (4 * pl.col('n') ** 2)).sqrt())
                / (1 + (z ** 2) / pl.col('n'))
            )
        )
    )
    
    print(df_result.select(['title', 'n', 'wilson_score']).sort('wilson_score',
        descending=True))
  
  def test_wilson_score_movies_pivot(self):
      print('\nwilson score on full movies pivot')
      df = self.movie_pivot_df
      df = df.with_columns(
          pl.sum_horizontal("1", "2", "3", "4", "5").cast(pl.Float64).alias(
              "total_votes")
      )
      df = df.with_columns(
          pl.sum_horizontal("4", "5").cast(pl.Float64).alias(
              "total_pos_votes")
      )
      df_ranked = wilson_score_df(df, 'total_pos_votes', 'total_votes')
      print(df_ranked)
      
      s1 = wilson_score(5, 5)
      s2 = wilson_score(95, 100)
      self.assertLess(s1, s2)
      
      # calc manually to check results
      z = 1.96  # 95% confidence level
      
      df_result = (
          df
          .with_columns(
              # 1. Total ratings
              n=pl.sum_horizontal(['1', '2', '3', '4', '5'])
          )
          .with_columns(
              # 2. Normalized average rating (p)
              p=(
                        ((1 * pl.col('1') + 2 * pl.col('2') + 3 * pl.col(
                            '3') + 4 * pl.col('4') + 5 * pl.col('5'))
                         / pl.col('n')) - 1
                ) / 4
          )
          .with_columns(
              # 3. Wilson Score formula
              wilson_score=pl.when(pl.col('n') == 0)
              .then(0.0)
              .otherwise(
                  (pl.col('p') + (z ** 2) / (2 * pl.col('n')) - z * (
                          (pl.col('p') * (1 - pl.col('p')) / pl.col('n')) + (
                          z ** 2) / (4 * pl.col('n') ** 2)).sqrt())
                  / (1 + (z ** 2) / pl.col('n'))
              )
          )
      )
      
      print(
          df_result.select(['title', 'n', 'wilson_score']).sort('wilson_score',
              descending=True))
    
  
  def test_thomspon(self):
    print('\nthompson')
    df = pd.DataFrame(self.data)
    df_ranked = thompson_sampling_with_dirichlet_prior(df, seed=None)
    print(df_ranked)
    df_ranked = thompson_sampling_with_dirichlet_prior(df, seed=None)
    print(df_ranked)
    
    
if __name__ == '__main__':
  unittest.main()

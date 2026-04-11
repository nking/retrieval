import os
import unittest

from helper import *
from movie_lens_retrieval.misc.Bayesian import BayesianAvg, BayesianShrinkageEstimator
import polars as pl
import pandas as pd
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
      
  def tearDown(self):
      self.movie_reader.close()
    
  def testUnweighted(self):
    m = 1
    b = BayesianAvg(pd.DataFrame(self.data), m=m)
    t = b.get_top(10)
    print(f"\nm={m} unweighted pandas:\n{t}")
    b = BayesianAvg(pl.DataFrame(self.data), m=m)
    t = b.get_top(10)
    print(f"\nm={m} unweighted polars:\n{t}")
    
  def testWeighted(self):
    b = BayesianShrinkageEstimator(pd.DataFrame(self.data))
    t = b.get_top(10)
    print(f"\nweighted pandas:\n{t}")
    b = BayesianShrinkageEstimator(pl.DataFrame(self.data))
    t = b.get_top(10)
    print(f"\nweighted polars:\n{t}")
    
  def test_movies_rating_pivot(self):
      print(f'test with ML pivot')
      
      batch_size=1024
      movies_df = load_movies_into_polars()
      #print(movies_df)
      batch_bytes = self.movie_reader.read([x for x in range(0, batch_size)])
      data = [msgpack.unpackb(b, use_list=False) for b in batch_bytes]

      pivot_df = pl.from_dicts(data)
      #print(pivot_df)
          
      final_df = pivot_df.join(movies_df.select(["movie_id", "title"]),
          on="movie_id",how="left"
      )
      #print(final_df)
      
      m = 1
      b = BayesianAvg(final_df, m=m)
      t = b.get_top(10)
      print(f"\nm={m} unweighted BayesianAvg on full movies pivot:\n{t}")
      m = 2 #not a good setting for this algorithm
      b = BayesianAvg(final_df, m=m)
      t = b.get_top(10)
      print(f"\nm={m} unweighted BayesianAvg on full movies pivot:\n{t}")
      
      b = BayesianShrinkageEstimator(final_df)
      t = b.get_top(10)
      print(f"\nweighted, BayesianShrinkageEstimator on full movies pivot:\n{t}")
      order_df = pl.DataFrame({"movie_id": t})
      filtered = order_df.join(final_df, on="movie_id", how="left")
      print(filtered)
      

if __name__ == '__main__':
  unittest.main()

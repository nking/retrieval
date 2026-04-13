import os.path
import unittest

import polars as pl

import msgpack
from movie_lens_retrieval.misc.Bayesian import BayesianAvg

from array_record.python import array_record_module
from helper import *

class TestWrapperForColdStartMovies(unittest.TestCase):
    def setUp(self):
       pass
 
    def test0(self):
        movie_reader = None
        try:
            movie_reader = array_record_module.ArrayRecordReader(
                os.path.join(get_project_dir(),
                "src/test/resources/data/movie_ratings_pivot_table/movie_ratings_pivot_table.array_record"))
            count = movie_reader.num_records()
            data_bytes = movie_reader.read_all()
            data = [msgpack.unpackb(b, use_list=False) for b in data_bytes]
        finally:
            if movie_reader is not None:
                movie_reader.close()
    
        pivot_df = pl.from_dicts(data)
        
        print(f"count pivot: {pivot_df.count()}", flush=True)
        
        n_movies = pivot_df['movie_id'].count()
        m = 1
        b = BayesianAvg(pivot_df, m=m)
        top_df = b.get_top(n_movies) #catalog has 3883 movies

        outfile_path = os.path.join(get_bin_dir(),"cold_start_movies.txt")
        top_df.select("movie_id").write_csv(outfile_path, include_header=False)

        print(f'wrote {n_movies} movies to {outfile_path}') 

    
    if __name__ == '__main__':
        unittest.main()

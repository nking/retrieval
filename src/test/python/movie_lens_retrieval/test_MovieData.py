import os.path
import unittest

from helper import *
from movie_lens_retrieval.MovieData import MovieData
from movie_lens_retrieval.UserData import UserData

class TestUserData(unittest.TestCase):
    def setUp(self):
        self.movies_path = os.path.join(get_project_dir(),
            "src/test/resources/data/movies/movies.parquet")
    
    def test_get_movie(self):
        data = MovieData(self.movies_path)
        '''
        6047::Sabrina (1995)::Comedy|Romance
        6071::Dangerous Minds (1995)::Drama
        6135::In the Bleak Midwinter (1995)::Comedy
        '''
        movie_ids = tf.constant([[6047], [6071], [6135]], dtype=tf.int64)
        expected = {
            'movie_id' : tf.identity(movie_ids),
            'genres' : tf.constant([["Comedy|Romance"], ["Drama"], ["Comedy"]], dtype=tf.string)
        }
        
        d = data.get_movie(movie_ids)
        
        self.assertEqual(expected.keys(), d.keys())
        
        for key in expected:
            self.assertTrue(tf.reduce_all(tf.equal(expected[key], d[key])))
            
    if __name__ == '__main__':
        unittest.main()

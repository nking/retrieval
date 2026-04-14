import tensorflow as tf
import polars as pl

class MovieData(object):
    def __init__(self, movie_path:str, offset:int=6041):
        """
        given path to movie file, creates a datastructure for movie_id access
        :param movie_path: path to the users.dat file containing fields movie_id, title, genrese.
        For now, provide a parquet file.  Also note that the user_ids must be ordered from 1 to N.
        :param offset: these movie_ids already in the movies.dat file begin with offset and end with offset + n_movies.
        """
        if not movie_path.endswith(".parquet"):
            print(f'WARNING: expecting input file to be a parquet file')
        self.offset = tf.constant(offset, dtype=tf.int64)
        df = pl.read_parquet(movie_path)
        
        df = df.sort('movie_id')
        self.title = tf.constant(df['title'].to_numpy(), name='movie_title', dtype=tf.string)
        self.genres = tf.constant(df['genres'].to_numpy(), name='movie_genres', dtype=tf.string)
        self.num_users = len(df)
        del df
    
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, 1], dtype=tf.int64),
    ])
    def get_movie(self, movie_id: tf.Tensor):
        """
        get a dictionary of inputs usable for the Candidate model dictionary signature.
        
        :param movie_id: a tensor of an array of integer movie_ids.
           example usage: user_data.get_user(user_id=tf.constant([6041]), timestamp=tf.constant([-1]))
        :param timestamp: timestamp associated with the user_id request. if the value
           is -1, it gets reset to tf.timestamp().
        :return:
        """
        idx = tf.subtract(movie_id, self.offset)
        return {
            'movie_id': movie_id,
            'genres': tf.gather(self.genres, idx)
        }
    
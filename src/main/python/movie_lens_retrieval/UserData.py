import tensorflow as tf
import polars as pl

class UserData(object):
    def __init__(self, users_path:str):
        """
        given path ot user file, creates a datastructure for movie_id access
        :param users_path: path to the users.dat file containing fields movie_id, gender, age, occupation, zipcode.
        For now, provide a parquet file.  Also note that the user_ids must be ordered from 1 to N.
        """
        if not users_path.endswith(".parquet"):
            print(f'WARNING: expecting input file to be a parquet file')
            
        df = pl.read_parquet(users_path)
        df = df.sort('user_id')
        self.gender = tf.constant(df['gender'].to_numpy(), name='user_gender', dtype=tf.string)
        self.age = tf.constant(df['age'].to_numpy(), name='user_age', dtype=tf.int64)
        self.occupation = tf.constant(df['occupation'].to_numpy(),  name='user_occupation', dtype=tf.int64)
        self.num_users = len(df)
        del df
    
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, 1], dtype=tf.int64),
        tf.TensorSpec(shape=[None, 1], dtype=tf.int64)
    ])
    def get_user(self, user_id: tf.Tensor, timestamp: tf.Tensor):
        """
        get a dictionary of inputs usable for the Query model dictionary signature.
        
        :param user_id: a tensor of an array of integer user_ids.
           example usage: user_data.get_user(movie_id=tf.constant([123]), timestamp=tf.constant([-1]))
        :param timestamp: timestamp associated with the movie_id request. if the value
           is -1, it gets reset to tf.timestamp().
        :return:
        """
        #tf.debugging.assert_equal(tf.shape(user_id), tf.shape(timestamp),
        #    message="User ID and Timestamp batches must be the same size")
        
        idx = user_id - 1
        now = tf.cast(tf.timestamp(), tf.int64)
        resolved_ts = tf.where(
            tf.equal(timestamp, -1),
            now,
            timestamp
        )
        
        return {
            'user_id': user_id,
            'gender': tf.gather(self.gender, idx),
            'age': tf.gather(self.age, idx),
            'occupation': tf.gather(self.occupation, idx),
            'timestamp': resolved_ts,
        }
    
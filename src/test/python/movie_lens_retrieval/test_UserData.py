import os.path
import unittest

from helper import *
from movie_lens_retrieval.UserData import UserData

class TestUserData(unittest.TestCase):
    def setUp(self):
        self.users_path = os.path.join(get_project_dir(),
            "src/test/resources/data/users/users.parquet")
    
    def test_get_user(self):
        userData = UserData(self.users_path)
        '''
        7::M::35::1::06810
        35::M::45::1::02482
        114::F::25::2::83712
        '''
        user_ids = tf.constant([[7], [35], [114]], dtype=tf.int64)
        timestaps = tf.constant([[-1] for _ in range(len(user_ids))], dtype=tf.int64)
        expected = {
            'user_id' : tf.identity(user_ids),
            'gender' : tf.constant([["M"], ["M"], ["F"]], dtype=tf.string),
            'age': tf.constant([[35], [45], [25]], dtype=tf.int64),
            'occupation': tf.constant([[1], [1], [2]], dtype=tf.int64),
            'timestamp': tf.identity(timestaps),
        }
        
        d = userData.get_user(user_ids, timestaps)
        
        self.assertEqual(expected.keys(), d.keys())
        
        for key in expected:
            if key != "timestamp":
                self.assertTrue(tf.reduce_all(tf.equal(expected[key], d[key])))
            else:
                self.assertTrue(tf.reduce_any(tf.not_equal(expected[key], d[key])))
        
    if __name__ == '__main__':
        unittest.main()

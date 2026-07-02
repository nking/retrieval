import os.path
import unittest
from typing import Dict
import numpy as np
from array_record.python import array_record_module
import msgpack
import pyarrow as pa
import pyarrow.parquet as pq

from helper import *
from movie_lens_retrieval.Retriever import Retriever, EmbeddingType

class TestRetrieval(unittest.TestCase):
    def setUp(self):
        
        saved_models_dir = os.path.join(get_project_dir(),
            "src/main/resources/serving_models")
        self.user_movie_models_dir = os.path.join(saved_models_dir,
            "user_movie_model")
        
        self.cold_start_path = os.path.join(get_project_dir(), "src/test/resources/data/cold_start_movies.txt")
        
        self.embed_dim = 16 #though this could be read and parsed from a single entry in embeddings
        self.movie_emb = os.path.join(get_project_dir(),
            "src/test/resources/data/movie_emb_inp/*tfrecord*.gz")
        self.user_emb = os.path.join(get_project_dir(),
            "src/test/resources/data/user_emb_inp/*tfrecord*.gz")
        
        self.users_path = os.path.join(get_project_dir(),
            "src/test/resources/data/users/users.parquet")
        self.movies_path = os.path.join(get_project_dir(),
            "src/test/resources/data/movies/movies.parquet")
        
        self.user_movie_hist_path_patterns = [os.path.join(get_project_dir(),
            "src/test/resources/data/ratings_train/ratings_train.array_record"),
            os.path.join(get_project_dir(),
                "src/test/resources/data/ratings_val/ratings_val.array_record")
            ]
        self.max_k = 10
        self.MOVIE_OFFSET = 6040 + 1
        
    def test_read_cold_start_movies(self):
        cold_start_movie_list = Retriever._read_cold_start(self.cold_start_path)
        self.assertTrue(len(cold_start_movie_list) > 3000)
        self.assertTrue(isinstance(cold_start_movie_list[0], int))
        
    def _construct_Retrieval_using_train_val(self, max_k) -> Retriever:
        return Retriever(user_movie_saved_model_dir=self.user_movie_models_dir,
            movie_id_offset = self.MOVIE_OFFSET,
            user_embed_path=self.user_emb,
            movie_embed_path=self.movie_emb,
            embed_dim=self.embed_dim,
            cold_start_movie_path=self.cold_start_path,
            users_path=self.users_path,
            movies_path=self.movies_path,
            user_movie_hist_path_patterns=self.user_movie_hist_path_patterns,
            max_k=max_k)

    def test_retrieval(self):
        '''
        def __init__(self, user_movie_saved_model_dir: str,
                cold_start_movie_list: List[int],
                user_embed_path: str,
                movie_embed_path: str,
                max_k: int = 1000,
                embed_dim: int = 16
        ):
        :return:
        '''
        
        rr = self._construct_Retrieval_using_train_val(max_k=3883)
        self.assertTrue(rr.max_hist > 200)
        
        # first timestamp from test is 978133414
        ts = 978133414
        n_users = len(rr.user_data.gender)
        user_inp_dict = {
            'user_id': tf.constant([[i] for i in range(1, n_users + 1)], dtype=tf.int64),
            'gender': rr.user_data.gender[:, tf.newaxis],
            'age': rr.user_data.age[:, tf.newaxis],
            'occupation': rr.user_data.occupation[:, tf.newaxis],
            'timestamp': tf.constant([[ts] for _ in range(n_users)], dtype=tf.int64),
        }
        n_movies = rr.movie_data.num_movies
        top_k = n_movies - rr.max_hist
        
        #np.ndarray:
        recommended_movies = rr.get_movies_given_users(user_inp_dict, top_k=top_k, rm_hist=True)
        self.assertTrue(recommended_movies.shape == (n_users, top_k))
        
        #write to array_records
        outfile = os.path.join(get_bin_dir(), "recommended_movies_minus_train_val.array_record")
        writer = None
        try:
            writer = array_record_module.ArrayRecordWriter(outfile, 'group_size:1')
            for user_id, movie_ids in zip(user_inp_dict['user_id'].numpy(), recommended_movies):
                user_id = user_id[0].item()
                movie_ids = movie_ids.tolist()
                writer.write(msgpack.packb((user_id, movie_ids)))
        finally:
            if writer is not None:
                writer.close()
       
        #assert can read file
        reader = None
        try:
            reader = array_record_module.ArrayRecordReader(outfile)
            count = reader.num_records()
            batch_bytes = reader.read( [x for x in range(0, count)])
            records = [msgpack.unpackb(b, use_list=False) for b in batch_bytes]
            
            self.assertTrue(count == len(records))
            self.assertTrue(count == len(recommended_movies))
            for i, record in enumerate(records):
                self.assertTrue(isinstance(record[0], int))
                self.assertTrue(isinstance(record[1], tuple))
                self.assertTrue(isinstance(record[1][0], int))
                self.assertEqual(user_inp_dict['user_id'][i].numpy().item(), record[0])
                self.assertEqual(recommended_movies[i][0].item(), record[1][0])
                if i > 5:
                    break
        finally:
            if reader is not None:
                reader.close()
        
    def _test_write_negatives(self):
        """
        The types of negatives needed for listwise contrastive learning are listed and an example is
        given for 1 user.
        
        Here is example for 1 user's data:
            watched history = A,B,C,D,G
            disliked = A,B,C,D
            recommended by retrieval = B,D,F,G,H
            entire movie catalog is A,B,C,D,E,F,G,H,I,J
        
        1) "hard negatives" = recommended intersection with user's disliked.
            These are "False positives".
            intersect({B,D,F,G,H}, {A,B,C,D}) = B,D
        2) "implicit hard negatives" = recommended minus users watch history
            subtract({B,D,F,G,H}, {A,B,C,D,G}) = F,H
        3) "out of distr negatives" = disliked - recommended.
            subtract({A,B,C,D}, {B,D,F,G,H}) = A,C
        4) "easy negatives" = movie catalog - watch history
            subtract({A,B,C,D,E,F,G,H,I,J}, {A,B,C,D,G}) = E,F,H,I,J

        Because downstream models in the system have to use watch_history and diskliked with timestamps < target timestamps,
        the fixed lists are not written here anymore.
        """

    def test_write_recommendations_and_timestamps(self):
        """
        1) create recommended movies for each user, but do not subtract watched from them.
        2) load the train, val, test histories
        3) for each recommendation, store an array of timestamps
        with default timestamp of 2050 for all movies,
        unless the movies is in train, val, or test in which case it gets that timestamp.
        
        writes those recommednations and timestamps to 2 array_record files  and to 2 parquet files
        """
        num_movies = 3883
        rr = self._construct_Retrieval_using_train_val(max_k=num_movies)
        self.assertTrue(rr.max_hist > 200)
        
        # first timestamp from test is 978133414
        ts = 978133414
        n_users = len(rr.user_data.gender)
        user_inp_dict = {
            'user_id': tf.constant([[i] for i in range(1, n_users + 1)],
                dtype=tf.int64),
            'gender': rr.user_data.gender[:, tf.newaxis],
            'age': rr.user_data.age[:, tf.newaxis],
            'occupation': rr.user_data.occupation[:, tf.newaxis],
            'timestamp': tf.constant([[ts] for _ in range(n_users)],
                dtype=tf.int64),
        }
        top_k = num_movies
        
        # (1)  np.ndarray:
        recommended_movies = rr.get_movies_given_users(user_inp_dict,
            top_k=top_k, rm_hist=False)
        self.assertTrue(recommended_movies.shape == (n_users, top_k))
        
        ts_2050 = 2524608000
        
        '''
        for each user, for each recommended movie, there is either a timestamp from train, val, or test
           else ts_2050 is used
        '''
        
        #key  user_id, value=dict with key=movie_id, value=timestamp
        history_dict = self._read_all_ratings_into_dict()
        
        movies_schema = pa.schema([
            pa.field("user_id", pa.int32()),
            pa.field("movie_ids", pa.list_(pa.int32(), num_movies))
        ])
        
        ts_schema = pa.schema([
            pa.field("user_id", pa.int32()),
            pa.field("timestamps", pa.list_(pa.int64(), num_movies))
        ])
        
        # write the full recommenations w/o removal to array_record and assert can read it
        # write to array_records
        outfile = os.path.join(get_bin_dir(), "recommended_movies.array_record")
        outfile2 = os.path.join(get_bin_dir(), "recommended_movies_timestamps.array_record")
        pa_outfile = os.path.join(get_bin_dir(), "recommended_movies.parquet")
        pa_outfile2 = os.path.join(get_bin_dir(), "recommended_movies_timestamps.parquet")
        writer = None
        writer2 = None
        pa_movie_writer = None
        pa_timestamp_writer = None
        try:
            writer = array_record_module.ArrayRecordWriter(outfile, 'group_size:1')
            writer2 = array_record_module.ArrayRecordWriter(outfile2, 'group_size:1')
            
            pa_movie_writer = pq.ParquetWriter(pa_outfile, movies_schema)
            pa_timestamp_writer = pq.ParquetWriter(pa_outfile2, ts_schema)
            
            for user_id, movie_ids in zip(user_inp_dict['user_id'].numpy(), recommended_movies):
                user_id = user_id[0].item()
                movie_ids = movie_ids.tolist()
                self.assertEqual(top_k, len(movie_ids))
                writer.write(msgpack.packb((user_id, movie_ids)))
                
                movie_batch = pa.RecordBatch.from_arrays([
                    pa.array([user_id], type=pa.int32()),
                    pa.array([movie_ids], type=pa.list_(pa.int32(), num_movies))
                ], schema=movies_schema)
                pa_movie_writer.write_batch(movie_batch)
                
                
                #create the timestamps
                timestamps = []
                for movie_id in movie_ids:
                    if user_id in history_dict and movie_id in history_dict[user_id]:
                        timestamps.append(history_dict[user_id][movie_id])
                    else:
                        timestamps.append(ts_2050)
                writer2.write(msgpack.packb((user_id, timestamps)))
                
                ts_batch = pa.RecordBatch.from_arrays([
                    pa.array([user_id], type=pa.int32()),
                    pa.array([timestamps], type=pa.list_(pa.int64(), num_movies))
                ], schema=ts_schema)
                pa_timestamp_writer.write_batch(ts_batch)
                
        finally:
            if writer is not None:
                writer.close()
            if writer2 is not None:
                writer2.close()
            if pa_movie_writer is not None:
                pa_movie_writer.close()
            if pa_timestamp_writer is not None:
                pa_timestamp_writer.close()
        
        # assert can read file
        reader = None
        try:
            reader = array_record_module.ArrayRecordReader(outfile)
            count = reader.num_records()
            batch_bytes = reader.read([x for x in range(0, count)])
            records = [msgpack.unpackb(b, use_list=False) for b in batch_bytes]
            
            self.assertTrue(count == len(records))
            self.assertTrue(count == len(recommended_movies))
            for i, record in enumerate(records):
                self.assertTrue(isinstance(record[0], int))
                self.assertTrue(isinstance(record[1], tuple))
                self.assertTrue(isinstance(record[1][0], int))
                self.assertEqual(user_inp_dict['user_id'][i].numpy().item(),
                    record[0])
                self.assertEqual(recommended_movies[i][0].item(), record[1][0])
                if i > 5:
                    break
        finally:
            if reader is not None:
                reader.close()
                
        # assert can read file
        reader = None
        try:
            reader = array_record_module.ArrayRecordReader(outfile2)
            count = reader.num_records()
            batch_bytes = reader.read([x for x in range(0, count)])
            records = [msgpack.unpackb(b, use_list=False) for b in  batch_bytes]
            
            self.assertTrue(count == len(records))
            self.assertTrue(count == len(recommended_movies))
            for i, record in enumerate(records):
                self.assertTrue(isinstance(record[0], int))
                self.assertTrue(isinstance(record[1], tuple))
                self.assertTrue(isinstance(record[1][0], int))
                if i > 5:
                    break
        finally:
            if reader is not None:
                reader.close()
        # =======
        
    
    def _read_ratings_array_record(self, file_path:str, batch_size:int=2048) -> pl.DataFrame:
        if not os.path.exists(file_path):
            raise Exception(f'file not found: {file_path}')
        records = []
        reader = None
        try:
            reader = array_record_module.ArrayRecordReader(file_path)
            n = reader.num_records()
            for i in range(0, n, batch_size):
                i_end = i + batch_size
                if i_end >= n:
                    i_end = n
                batch_bytes = reader.read([x for x in range(i, i_end)]) # a single list of encodings, each being a list of 4 integers
                data = [msgpack.unpackb(b, use_list=False) for b in batch_bytes] # list of tuples of 4 integers
                for record in data:
                    records.append({'user_id': int(record[0]), 'movie_id': int(record[1]),
                        'rating': int(record[2]), 'timestamp': int(record[3])})
        finally:
            if reader is not None:
                reader.close()
        return pl.DataFrame(records)
    
    if __name__ == '__main__':
        unittest.main()
    
    def _read_all_ratings_into_dict(self):
        """
        create dictionary of user_id : {movie_id: timestamp}
        :return:
        """
        output = {}
        for file_path in [os.path.join(get_project_dir(),
            "src/test/resources/data/ratings_train/ratings_train.array_record"),
            os.path.join(get_project_dir(),
                "src/test/resources/data/ratings_val/ratings_val.array_record"),
            os.path.join(get_project_dir(),
                "src/test/resources/data/ratings_test/ratings_test.array_record")
            ]:
            reader = None
            try:
                reader = array_record_module.ArrayRecordReader(file_path)
                n = reader.num_records()
                batch_bytes = reader.read([x for x in range(n)])  # a single list of encodings, each being a list of 4 integers
                data = [msgpack.unpackb(b, use_list=False) for b in batch_bytes]  # list of tuples of 4 integers
                for record in data:
                    user_id = record[0]
                    if user_id not in output:
                        output[user_id] = {}
                    output[user_id][record[1]] = record[3]
            finally:
                if reader is not None:
                    reader.close()
        return output

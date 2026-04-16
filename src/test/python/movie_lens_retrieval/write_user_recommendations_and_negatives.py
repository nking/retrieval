import os.path
import unittest
from typing import Dict
import numpy as np
from array_record.python import array_record_module
import msgpack

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
        
        self.user_movie_hist_path_patterns = [os.path.join(get_project_dir(), "src/test/resources/data/ratings_train/ratings_train.array_record"),
            os.path.join(get_project_dir(),
                "src/test/resources/data/ratings_val/ratings_val.array_record")
            ]
        self.max_k = 10
        self.MOVIE_OFFSET = 6040 + 1
        
    def test_read_cold_start_movies(self):
        cold_start_movie_list = Retriever._read_cold_start(self.cold_start_path)
        self.assertTrue(len(cold_start_movie_list) > 3000)
        self.assertTrue(isinstance(cold_start_movie_list[0], int))
        
    def _construct_Retrieval(self, max_k) -> Retriever:
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
        
        rr = self._construct_Retrieval(max_k=3883)
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
        outfile = os.path.join(get_bin_dir(), "recommended_movies.array_record")
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
        
    def test_write_negatives(self):
        """
        1) create recommended movies for each user, but do not subtract watched from them.
        2) load the train and val movies disliked by user
        3) find the intersection of (1) and (2) and  as negatives file
        4) append to 3, any items from (2) that aren't already in (3)
        writes those negatives to array_record format file
        """
        
        rr = self._construct_Retrieval(max_k=3883)
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
        n_movies = rr.movie_data.num_movies
        top_k = n_movies
        
        #(1)  np.ndarray:
        recommended_movies = rr.get_movies_given_users(user_inp_dict, top_k=top_k, rm_hist=False)
        self.assertTrue(recommended_movies.shape == (n_users, top_k))
        #put into polars dataframe
        rec_df = pl.DataFrame(
            [(u[0], r) for u, r in zip(user_inp_dict['user_id'].numpy(), recommended_movies)],
            schema=["user_id", "recommended"],
            orient="row"
        )
        
        #(2) read the disliked from train and val, separately
        ratings_train_disliked_df = self._read_ratings_array_record(
            os.path.join(get_project_dir(),
                "src/test/resources/data/ratings_train_disliked/ratings_train_disliked.array_record")
        )
        ratings_train_disliked_df = ratings_train_disliked_df.group_by('user_id').agg(
            [pl.col("movie_id").sort_by("rating", descending=True)])
    
        ratings_val_disliked_df = self._read_ratings_array_record(
            os.path.join(get_project_dir(),
                "src/test/resources/data/ratings_val_disliked/ratings_val_disliked.array_record")
        )
        ratings_val_disliked_df = ratings_val_disliked_df.group_by(
            'user_id').agg([pl.col("movie_id").sort_by("rating", descending=True)])
        
        #intersection of train disliked with rec_df
        inter_train_df = ratings_train_disliked_df.join(rec_df, on='user_id', how='left')
        inter_train_df = inter_train_df.with_columns(
            hard_neg=pl.col("movie_id").list.set_intersection(pl.col("recommended"))
        )
        inter_train_df = inter_train_df.with_columns(
            easy_neg=pl.col("movie_id").list.set_difference(pl.col("hard_neg"))
        )
        inter_train_df = inter_train_df.with_columns(
            negatives=pl.col("hard_neg").list.concat(pl.col("easy_neg"))
        )
        
        #intersection of val disliked with rec_df
        inter_val_df = ratings_val_disliked_df.join(rec_df, on='user_id', how='left')
        inter_val_df = inter_val_df.with_columns(
            hard_neg=pl.col("movie_id").list.set_intersection(pl.col("recommended"))
        )
        inter_val_df = inter_val_df.with_columns(
            easy_neg=pl.col("movie_id").list.set_difference(pl.col("hard_neg"))
        )
        inter_val_df = inter_val_df.with_columns(
            negatives=pl.col("hard_neg").list.concat(pl.col("easy_neg"))
        )
        
        #write user_id, negatives to array_record
        for outfile, df in zip([os.path.join(get_bin_dir(), "train_negatives.array_record"),
            os.path.join(get_bin_dir(), "val_negatives.array_record")], [inter_train_df, inter_val_df]):
            data_to_write = df.select(["user_id", "negatives"])
            writer = None
            try:
                writer = array_record_module.ArrayRecordWriter(outfile, 'group_size:1')
                for row in data_to_write.iter_rows(named=False):
                    serialized_bytes = msgpack.packb(row, use_bin_type=True)
                    writer.write(serialized_bytes)
            finally:
                if writer is not None:
                    writer.close()
                    
        #assert can read the filtes
        for outfile in [os.path.join(get_bin_dir(), "train_negatives.array_record"),
            os.path.join(get_bin_dir(), "val_negatives.array_record")]:
            reader = None
            try:
                reader = array_record_module.ArrayRecordReader(outfile)
                count = reader.num_records()
                print(f'reading {count} records from {outfile}')
                batch_bytes = reader.read([x for x in range(0, count)])
                records = [msgpack.unpackb(b, use_list=False) for b in batch_bytes]
                
                self.assertTrue(count == len(records))
                for i, record in enumerate(records):
                    self.assertTrue(isinstance(record[0], int))
                    self.assertTrue(isinstance(record[1], tuple))
                    self.assertTrue(isinstance(record[1][0], int))
                    if i > 5:
                        break
            finally:
                if reader is not None:
                    reader.close()
        
        
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

import collections
import os.path
import unittest
import glob
from collections import defaultdict
from typing import Any

from scipy.stats import hypergeom, combine_pvalues
from six import moves
from sklearn.metrics import ndcg_score, average_precision_score
import polars as pl
import numpy as np
import plotly.express as px  # needs kaleido to write pngs
from plotly.subplots import make_subplots

import msgpack
import msgpack_numpy as m  # Optional: helps if you have raw numpy arrays

from movie_lens_retrieval.misc.Bayesian import BayesianAvg

m.patch()  # Makes msgpack understand numpy types automatically
from array_record.python import array_record_module

from helper import *
from movie_lens_retrieval.Retriever import Retriever, EmbeddingType


def create_cold_start_list() -> List[int]:
    batch_size = 1024
    # print(movies_df)
    movie_reader = None
    data = None
    try:
        movie_reader = array_record_module.ArrayRecordReader(
            os.path.join(get_project_dir(),
                "src/test/resources/data/movie_ratings_pivot_table/movie_ratings_pivot_table.array_record"))
        batch_bytes = movie_reader.read([x for x in range(0, batch_size)])
        data = [msgpack.unpackb(b, use_list=False) for b in batch_bytes]
    finally:
        if movie_reader is not None:
            movie_reader.close()
    
    pivot_df = pl.from_dicts(data)
    
    m = 1
    b = BayesianAvg(pivot_df, m=m)
    top_df = b.get_top(3883) #catalog has 3883 movies
    return top_df['movie_id'].to_list()
    
class TestRetrieverAndRanker(unittest.TestCase):
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
        
        self.movies_path = os.path.join(get_project_dir(),
            "src/test/resources/data/users/users.parquet")
        self.users_path = os.path.join(get_project_dir(),
            "src/test/resources/data/movies/movies.parquet")
        
        self.max_k = 10
        
    def test_read_cold_start_movies(self):
        cold_start_movie_list = Retriever._read_cold_start(self.cold_start_path)
        self.assertTrue(len(cold_start_movie_list) > 3000)
        self.assertTrue(isinstance(cold_start_movie_list[0]), int)
        
    def _construct_Retrieval(self, max_k) -> Retriever:
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
        return Retriever(user_movie_saved_model_dir=self.user_movie_models_dir,
            user_embed_path=self.user_emb,
            movie_embed_path=self.movie_emb,
            embed_dim=self.embed_dim,
            cold_start_movie_path=self.cold_start_path,
            users_path=self.users_path,
            movies_path=self.movies_path,
            max_k=max_k)
    
    def test_indexer_tensors(self):
        
        loaded_model = tf.saved_model.load(self.user_movie_models_dir)
        
        '''
        ways to provide inputs to the Retriever:
        1) user embedding mode:
           the following data structures must have a user_id and can optionally have these
           remaining keys where noted: gender, age, occupation, timestamp.   Missing keys will be fetched from
           user data and returned in the resulting tensors.
           
           datastructures accepted:
           - tf.Tensor of user_ids
           - list of user_ids
           - dictionary of user_ids and optional keys with values as scalars
           - dictionary of user_ids and optional keys with values as tensors of arrays
           - list of dictionary of user_ids and optional keys
        2) movie embedding mode:
           the following data structures must have a movie_id and can optionally have these
           remaining keys where noted: genres.   Missing keys will be fetched from
           movie data and returned in the resulting tensors.
           
           datastructures accepted:
           - tf.Tensor of movie_ids
           - list of movie_ids
           - dictionary of movie_ids and optional keys as scalars
           - dictionary of movie_ids and optional keys with values as tensors of arrays
           - list of dictionary of movie_ids pluands optional keys
        '''
        
        rr = self._construct_Retrieval(max_k=100)
        
        editing for refactored signatures
        
        inputs1 = [{'user_id': 1, 'age': 10}, {'user_id': 2, 'age': 16}]
        inputs1 = self._create_dictionary_of_tensors(inputs1)
        inputs2 = [{'movie_id': 1+6040, 'genres': "Animation|Children's|Comedy"},
            {'movie_id': 2+6040, 'genres': "Adventure|Children's|Fantasy"}]
        inputs2 = self._create_dictionary_of_tensors(inputs2)
        
        inputs3 = [{'movie_id': 1}, {'movie_id': 2}]
        inputs4 = [{'movie_id': 1+6040}, {'movie_id': 2+6040}]
        
       
        
        for j in range(4):
            if j == 0:
                inputs = inputs1
                embeddings_tensor = rr._create_embeddings(EmbeddingType.USER, inputs)
            elif j == 1:
                inputs = inputs2
                embeddings_tensor = rr._create_embeddings(EmbeddingType.MOVIE, inputs)
            if j == 2:
                inputs = inputs3
                embeddings_tensor = rr._create_embeddings(EmbeddingType.USER, inputs)
            elif j == 3:
                inputs = inputs4
                embeddings_tensor = rr._create_embeddings(EmbeddingType.MOVIE, inputs)
            
            indexer = Retriever.build_scann_searcher(embeddings_tensor, top_k=2)
            neighbor_idxs, distances = indexer.search_batched( embeddings_tensor, 2)
            # results are both np.ndarrays
            self.assertEqual([0, 1], neighbor_idxs[0].tolist())
            self.assertEqual([1, 0], neighbor_idxs[1].tolist())
            a = set([i for _list in neighbor_idxs for i in _list])
            self.assertTrue(0 in a)
            self.assertTrue(1 in a)
           
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
        rr = self._construct_Retrieval(max_k=1000)
        
        # who are the users similar to movie_id=
        user_inp = {'movie_id': 5077, 'age': 25}
        user_inp = Retriever._create_embeddings(EmbeddingType.USER, user_inp)
        sim_users = rr.get_users_given_users(user_inp, top_k=9)
        print(f'sim_users: {sim_users}')
        # 1587, 2059, 5684, 1859, 4899, 5217, 3468, 2345, 3040
        
        sim_movies = rr.get_movies_given_users(user_inp, top_k=9)
        print(f'sim_movies: {sim_movies}')
        # 3089, 1572, 3030, 1068, 2731, 326, 1759, 3134, 2575, 2940
        
        # test that age is retrieved when missing from inouts
        user_inp = [{'movie_id': 5077}, {'movie_id': 1}]
        sim_users = rr.get_users_given_users(user_inp, top_k=9)
        print(f'sim_users: {sim_users}')
        try:
            user_inp = [{'movie_id': 1_000_000}]
            sim_users = rr.get_users_given_users(user_inp, top_k=9)
            self.fail("Should have thrown a ValueError")
        except ValueError:
            pass
        
        movie_inp = {'movie_id': 1068 + 6040, 'genres': 'Crime|Film-Noir'}
        sim_users = rr.get_users_given_movies(movie_inp, top_k=9)
        print(f'sim_users: {sim_users}')
        
        movie_inp = [{'movie_id': 1068 + 6040}, {'movie_id': 1 + 6040}]
        sim_users = rr.get_users_given_movies(movie_inp, top_k=9)
        print(f'sim_users: {sim_users}')
        
        try:
            movie_inp = {'movie_id': 1_000_000}
            sim_users = rr.get_users_given_movies(movie_inp, top_k=9)
            self.fail("Should have thrown a ValueError")
        except ValueError:
            pass
        
        movie_inp = [{'movie_id': 1068 + 6040}, {'movie_id': 1 + 6040}]
        sim_movies = rr.get_movies_given_movies(movie_inp, top_k=9)
        print(f'sim_movies: {sim_movies}')
        
        cold_starts = rr.get_cold_start_movie_recommendations(10)
        print(f'cold_starts: {cold_starts}')
        
        print(f'is_user_known(1_000_000)={rr.user_is_known(1_000_000)}')
        print(f'is_user_known(1)={rr.user_is_known(1)}')
        
        # use test data to check recommendations.  these are movies the user loved.
        # the returned ratings shuld be high
        user_inp = {'movie_id': 635, 'age': 56,
            'movie_id': [1704 + 6040, 1940 + 6040],
            'genres': ['Drama', 'Drama']}
        preds = rr.get_predictions(user_inp)
        print(f'predictions: {preds}')
    
    def test_write_user_embeddings(self):
        loaded_user_movie_model = tf.saved_model.load(
            self.user_movie_models_dir)
        batch_size = 256
        users_path = self.user_emb
        output_uri = os.path.join(get_bin_dir(),
            "user_embeddings.array_record")
        
        _ct = "GZIP" if users_path.endswith(".gz") else None
        file_paths = glob.glob(users_path)
        ds_ser = tf.data.TFRecordDataset(file_paths, compression_type=_ct)
        query_model = loaded_user_movie_model.signatures["serving_query"]
        INPUT_KEY = \
            list(query_model.structured_input_signature[1].keys())[0]
        
        feature_spec = {"movie_id": tf.io.FixedLenFeature([], tf.int64),
            "movie_id": tf.io.FixedLenFeature([], tf.int64),
            "rating": tf.io.FixedLenFeature([], tf.int64),
            "timestamp": tf.io.FixedLenFeature([], tf.int64),
            "gender": tf.io.FixedLenFeature([], tf.string),
            "age": tf.io.FixedLenFeature([], tf.int64),
            "occupation": tf.io.FixedLenFeature([], tf.int64),
            "genres": tf.io.FixedLenFeature([], tf.string)}
        
        def parse_tf_example(example_proto, feature_spec):
            return tf.io.parse_single_example(example_proto,
                feature_spec)
        
        embeddings = []
        user_ids = []
        for batch in ds_ser.batch(batch_size):
            emb = query_model(**{INPUT_KEY: batch})[
                'outputs']  # batch_size x emb_dim, e.g. 256 X 32
            embeddings.extend(emb.numpy().tolist())
        ds = ds_ser.map(lambda x: parse_tf_example(x, feature_spec))
        for batch in ds.batch(batch_size):
            # NOTE: you can write all features out if needed for a different use:
            user_ids.extend(batch["movie_id"].numpy().tolist())
        assert (len(embeddings) == len(user_ids))
        
        writer = None
        try:
            writer = array_record_module.ArrayRecordWriter(output_uri,
                "group_size:1")
            for user_id, emb in zip(user_ids, embeddings):
                writer.write(
                    msgpack.packb([user_id, emb], use_bin_type=True))
        finally:
            if writer is not None:
                writer.close()
        
        reader = None
        try:
            reader = array_record_module.ArrayRecordReader(output_uri)
            record = msgpack.unpackb(reader.read(), use_list=True)
            self.assertEquals(2, len(record))
            self.assertTrue(isinstance(record[0], int))
            self.assertTrue(isinstance(record[1], list))
        
        finally:
            if reader is not None:
                reader.close()
    
    def test_write_movie_embeddings(self):
        loaded_user_movie_model = tf.saved_model.load(
            self.user_movie_models_dir)
        batch_size = 256
        movies_path = self.movie_emb
        output_uri = os.path.join(get_bin_dir(),
            "movie_embeddings.array_record")
        
        _ct = "GZIP" if movies_path.endswith(".gz") else None
        file_paths = glob.glob(movies_path)
        ds_ser = tf.data.TFRecordDataset(file_paths, compression_type=_ct)
        query_model = loaded_user_movie_model.signatures["serving_candidate"]
        INPUT_KEY = list(query_model.structured_input_signature[1].keys())[0]
        
        feature_spec = {
            "movie_id": tf.io.FixedLenFeature(shape=[], dtype=tf.int64,
                default_value=None),
            "genres": tf.io.FixedLenFeature(shape=[], dtype=tf.string,
                default_value=None)}
        
        def parse_tf_example(example_proto, feature_spec):
            return tf.io.parse_single_example(example_proto,
                feature_spec)
        
        embeddings = []
        movie_ids = []
        for batch in ds_ser.batch(batch_size):
            emb = query_model(**{INPUT_KEY: batch})[
                'outputs']  # batch_size x emb_dim, e.g. 256 X 32
            embeddings.extend(emb.numpy().tolist())
        ds = ds_ser.map(lambda x: parse_tf_example(x, feature_spec))
        for batch in ds.batch(batch_size):
            # NOTE: you can write all features out if needed for a different use:
            movie_ids.extend(batch["movie_id"].numpy().tolist())
        assert (len(embeddings) == len(movie_ids))
        
        writer = None
        try:
            writer = array_record_module.ArrayRecordWriter(output_uri,
                "group_size:1")
            for id, emb in zip(movie_ids, embeddings):
                writer.write(
                    msgpack.packb([id, emb], use_bin_type=True))
        finally:
            if writer is not None:
                writer.close()
        
        reader = None
        try:
            reader = array_record_module.ArrayRecordReader(output_uri)
            record = msgpack.unpackb(reader.read(), use_list=True)
            self.assertEquals(2, len(record))
            self.assertTrue(isinstance(record[0], int))
            self.assertTrue(isinstance(record[1], list))
            self.assertNotEqual(record[0], record[1])
        
        finally:
            if reader is not None:
                reader.close()
    
    def read_movies_file_into_genre_dict(self,
            filter_for_single: bool = True) -> \
            tuple[defaultdict[Any, list], int]:
        _ct = "GZIP" if self.movie_emb.endswith(".gz") else None
        file_paths = glob.glob(self.movie_emb)
        ds_ser = tf.data.TFRecordDataset(file_paths, compression_type=_ct)
        feature_spec2 = {
            "movie_id": tf.io.FixedLenFeature(shape=[], dtype=tf.int64,
                default_value=None),
            "genres": tf.io.FixedLenFeature(shape=[], dtype=tf.string,
                default_value=None)}
        
        def parse_tf_example(example_proto):
            return tf.io.parse_single_example(example_proto, feature_spec2)
        
        ds = ds_ser.map(lambda z: parse_tf_example(z))
        # dict with key=genre, value=movie_id
        genre_to_ids = collections.defaultdict(list)
        n_movies = 0
        for x in ds.as_numpy_iterator():
            n_movies += 1
            if filter_for_single:
                if x['genres'].find(b'|') > -1:
                    continue
            genre_to_ids[x['genres']].append(x['movie_id'])
        return genre_to_ids, n_movies
    
    if __name__ == '__main__':
        unittest.main()

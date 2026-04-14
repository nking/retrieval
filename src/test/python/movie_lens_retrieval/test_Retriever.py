import collections
import os.path
import time
import unittest
import glob
from collections import defaultdict
from typing import Any, Dict

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

MOVIE_OFFSET = 6040 + 1

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
        
        self.max_k = 10
        
    def test_read_cold_start_movies(self):
        cold_start_movie_list = Retriever._read_cold_start(self.cold_start_path)
        self.assertTrue(len(cold_start_movie_list) > 3000)
        self.assertTrue(isinstance(cold_start_movie_list[0], int))
        
    def _construct_Retrieval(self, max_k) -> Retriever:
        return Retriever(user_movie_saved_model_dir=self.user_movie_models_dir,
            movie_id_offset = MOVIE_OFFSET,
            user_embed_path=self.user_emb,
            movie_embed_path=self.movie_emb,
            embed_dim=self.embed_dim,
            cold_start_movie_path=self.cold_start_path,
            users_path=self.users_path,
            movies_path=self.movies_path,
            max_k=max_k)
    
    def test_format_construct_inputs_only_to_dict_tensors(self):
    
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
        '''
        1::F::1::10::48067
        2::M::56::16::70072
        3::M::25::15::55117
        '''
        rr = self._construct_Retrieval(max_k=100)
        
        key = 'user_id'
        
        test_inps = []
        
        ts = int(time.time())
        
        outp_expected_all = \
            {'user_id': tf.constant([[1], [2], [3]], dtype=tf.int64),
                'gender': tf.constant([["F"], ["M"], ["M"]], dtype=tf.string),
                'age': tf.constant([[1], [56], [25]], dtype=tf.int64),
                'occupation': tf.constant([[10], [16], [15]], dtype=tf.int64),
                'timestamp': tf.constant([[ts], [ts], [ts]], dtype=tf.int64),
            }
        outp_expected_0 = \
            {'user_id': tf.constant([[1]], dtype=tf.int64),
                'gender': tf.constant([["F"]], dtype=tf.string),
                'age': tf.constant([[1]], dtype=tf.int64),
                'occupation': tf.constant([[10]], dtype=tf.int64),
                'timestamp': tf.constant([[ts]], dtype=tf.int64),
            }
        
        ## === user id inputs ===
        #0: tf.Tensor,
        test_inps.append(tf.constant([[1], [2], [3]], dtype=tf.int64))
        
        #1:  List[int]
        test_inps.append([[1], [2], [3]])
        
        #2:  Dict[str, tf.Tensor]
        test_inps.append({key :tf.constant([[1], [2], [3]], dtype=tf.int64)})
        
        #3:  Dict[str, Union[int, str]]
        test_inps.append({key : 1})
        
        #4:  List[Dict[str, Union[int, str]]]
        test_inps.append([{key : 1}, {key : 2}, {key : 3}])
        
        #test_inps.append([{'user_id': 1, 'timestamp': time.time()},
        #    {'user_id': 2, 'timestamp': time.time()},
        #    {'user_id': 3, 'timestamp': time.time()}])
        
        emb_type = EmbeddingType.USER
        
        for i, inp in enumerate(test_inps):
            outp: Dict[str, tf.Tensor] = Retriever._format_inputs_only_to_dict_tensors( emb_type, inp)
            self.assertTrue(isinstance(outp, dict))
            self.assertTrue(key in outp)
            match i:
                case 0:
                    self.assertTrue(tf.reduce_all(tf.equal(outp[key], inp)))
                    # break no fall-through behavior in python
                case 1:
                    self.assertTrue(tf.reduce_all(tf.equal(outp[key], test_inps[0])))
                case 2:
                    self.assertTrue(tf.reduce_all(tf.equal(outp[key], test_inps[0])))
                case 3:
                    self.assertTrue(
                        tf.equal(outp[key], tf.constant([[1]], dtype=tf.int64)))
                case 4:
                    self.assertTrue(tf.reduce_all(tf.equal(outp[key], test_inps[0])))
                
        for i, inp in enumerate(test_inps):
            outp: Dict[str, tf.Tensor] = rr.create_dictionary_of_tensors(emb_type, inp)
            self.assertTrue(isinstance(outp, dict))
            for k in outp_expected_all.keys():
                self.assertTrue(k in outp)
            match i:
                case 0:
                    for k, v in outp_expected_all.items():
                        if k != 'timestamp':
                            self.assertTrue(tf.reduce_all(tf.equal(outp[k], v)))
                case 1:
                    for k, v in outp_expected_all.items():
                        if k != 'timestamp':
                            self.assertTrue(tf.reduce_all(tf.equal(outp[k], v)))
                case 2:
                    for k, v in outp_expected_all.items():
                        if k != 'timestamp':
                            self.assertTrue(tf.reduce_all(tf.equal(outp[k], v)))
                case 3:
                    for k, v in outp_expected_0.items():
                        if k != 'timestamp':
                            self.assertTrue(tf.reduce_all(tf.equal(outp[k], v)))
                case 4:
                    for k, v in outp_expected_all.items():
                        if k != 'timestamp':
                            self.assertTrue(tf.reduce_all(tf.equal(outp[k], v)))
        
        ##==== movie id inputs ====
        '''
        6041::Toy Story (1995)::Animation|Children's|Comedy
        6042::Jumanji (1995)::Adventure|Children's|Fantasy
        6043::Grumpier Old Men (1995)::Comedy|Romance
        '''
        key = 'movie_id'
        test_inps = []
        
        ## === user id inputs ===
        # 0: tf.Tensor,
        test_inps.append(tf.constant([[6041], [6042], [6043]], dtype=tf.int64))
        
        # 1:  List[int]
        test_inps.append([[6041], [6042], [6043]])
        
        # 2:  Dict[str, tf.Tensor]
        test_inps.append(
            {key: tf.constant([[6041], [6042], [6043]], dtype=tf.int64)})
        
        # 3:  Dict[str, Union[int, str]]
        test_inps.append({key: 6041})
        
        # 4:  List[Dict[str, Union[int, str]]]
        test_inps.append([{key: 6041}, {key: 6042}, {key: 6043}])
        
        outp_expected_all = \
            {'movie_id': tf.constant([[6041], [6042], [6043]], dtype=tf.int64),
                'genres': tf.constant([["Animation|Children's|Comedy"], ["Adventure|Children's|Fantasy"], ["Comedy|Romance"]], dtype=tf.string),
            }
        outp_expected_0 = \
            {'movie_id': tf.constant([[6041]], dtype=tf.int64),
                'genres': tf.constant([["Animation|Children's|Comedy"]], dtype=tf.string)
            }
        
        emb_type = EmbeddingType.MOVIE
        
        for i, inp in enumerate(test_inps):
            print(f'Test2 {i}')
            outp:Dict[str, tf.Tensor] = rr.create_dictionary_of_tensors(emb_type, inp)
            self.assertTrue(isinstance(outp, dict))
            self.assertTrue(key in outp)
            match i:
                case 0:
                    self.assertTrue(tf.reduce_all(tf.equal(outp[key], inp)))
                    #break no fall-through behavior in python
                case 1:
                    self.assertTrue(tf.reduce_all(tf.equal(outp[key], test_inps[0])))
                case 2:
                    self.assertTrue(tf.reduce_all(tf.equal(outp[key], test_inps[0])))
                case 3:
                    self.assertTrue(tf.reduce_all(tf.equal(outp[key], tf.constant([[6041]], dtype=tf.int64))))
                case 4:
                    self.assertTrue(tf.reduce_all(tf.equal(outp[key], test_inps[0])))
                
        for i, inp in enumerate(test_inps):
            print(f'Test3 {i}')
            outp: Dict[str, tf.Tensor] = rr.create_dictionary_of_tensors(emb_type, inp)
            self.assertTrue(isinstance(outp, dict))
            for k in outp_expected_all.keys():
                self.assertTrue(k in outp)
            match i:
                case 0:
                    for k, v in outp_expected_all.items():
                        self.assertTrue(tf.reduce_all(tf.equal(outp[k], v)))
                case 1:
                    for k, v in outp_expected_all.items():
                        self.assertTrue(tf.reduce_all(tf.equal(outp[k], v)))
                case 2:
                    for k, v in outp_expected_all.items():
                        self.assertTrue(tf.reduce_all(tf.equal(outp[k], v)))
                case 3:
                    for k, v in outp_expected_0.items():
                        self.assertTrue(tf.reduce_all(tf.equal(outp[k], v)))
                case 4:
                    for k, v in outp_expected_all.items():
                        self.assertTrue(tf.reduce_all(tf.equal(outp[k], v)))
             
    def test_create_embeddings(self):
        rr = self._construct_Retrieval(max_k=100)
        
        ts = int(time.time())
        
        user_inp = \
            {'user_id': tf.constant([[1], [2], [3]], dtype=tf.int64),
                'gender': tf.constant([["F"], ["M"], ["M"]], dtype=tf.string),
                'age': tf.constant([[1], [56], [25]], dtype=tf.int64),
                'occupation': tf.constant([[10], [16], [15]], dtype=tf.int64),
                'timestamp': tf.constant([[ts], [ts], [ts]], dtype=tf.int64),
            }
        
        user_emb = rr._create_embeddings(EmbeddingType.USER, user_inp)
        self.assertTrue(isinstance(user_emb, tf.Tensor))
        self.assertEqual(3, len(user_inp['user_id']))
        self.assertEqual(rr.embed_dim, len(user_emb[0]))
        
        movie_inp = \
            {'movie_id': tf.constant([[6041], [6042], [6043]], dtype=tf.int64),
                'genres': tf.constant([["Animation|Children's|Comedy"],
                    ["Adventure|Children's|Fantasy"], ["Comedy|Romance"]],
                    dtype=tf.string),
            }
        
        movie_emb = rr._create_embeddings(EmbeddingType.USER, user_inp)
        self.assertTrue(isinstance(movie_emb, tf.Tensor))
        self.assertEqual(3, len(movie_inp['movie_id']))
        self.assertEqual(rr.embed_dim, len(movie_emb[0]))
        
    def test_build_scann_searcher_small(self):
        
        # 0 and 1 are similar,
        # 2 and 3 are similar
        embeddings_tensor = tf.constant(
            [
                [0.26162238, 0.52324476, 0.81102938], # a / np.linalg.norm(a)
                [0.25441093, 0.55970404, 0.78867387],
                [0.42985807, 0.08597161, 0.89879415],
                [0.42955643, 0.0937214 , 0.89816344]
            ],
            dtype=tf.float32
        )
        
        indexer = Retriever.build_scann_searcher(embeddings_tensor, top_k=2)
        neighbor_idxs, distances = indexer.search_batched(embeddings_tensor, 2)
        # results are both np.ndarrays
        self.assertEqual([0, 1], neighbor_idxs[0].tolist())
        self.assertEqual([1, 0], neighbor_idxs[1].tolist())
       
        self.assertEqual([2, 3], neighbor_idxs[2].tolist())
        self.assertEqual([3, 2], neighbor_idxs[3].tolist())
        
    def test_build_scann_searcher_large(self):
        pass
    
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
        #TODO: finish these
       
    if __name__ == '__main__':
        unittest.main()

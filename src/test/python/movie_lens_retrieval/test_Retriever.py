import os.path
import time
import unittest
from typing import Dict
import numpy as np

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
    
    def test_format_construct_inputs(self):
    
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
    
    def test_load_histories(self):
        
        user_hist_dict, max_hist = Retriever._read_user_ratings_histories(self.user_movie_hist_path_patterns)
        self.assertTrue(len(user_hist_dict) > 2000)
        self.assertTrue(max_hist > 20)
        
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
        self.assertTrue(rr.max_hist > 200)
        
        ts = int(time.time())
        
        '''
        635::M::56::17::33785
        1875::M::35::12::94107
        6040::M::25::6::11106
        '''
        
        user_test_dict = \
            {'user_id': tf.constant([[635], [1875], [6040]], dtype=tf.int64),
                'gender': tf.constant([['M'], ['M'], ['M']], dtype=tf.string),
                'age': tf.constant([[56], [35], [25]], dtype=tf.int64),
                'occupation': tf.constant([[17], [12], [6]], dtype=tf.int64),
                'timestamp': tf.constant([[ts], [ts], [ts]], dtype=tf.int64),
            }
        
        movie_test_dict = \
            {'movie_id': tf.constant([[6041], [6042]], dtype=tf.int64),
                'genres': tf.constant([["Animation|Children's|Comedy"],
                    ["Adventure|Children's|Fantasy"]],
                    dtype=tf.string),
            }
        
        n_users = len(user_test_dict['user_id'])
        n_movies = len(movie_test_dict['movie_id'])
        
        user_histories = rr.get_user_history(user_test_dict['user_id'])
        self.assertEqual(len(user_histories), n_users)
        for hist in user_histories:
            self.assertTrue(len(hist) > 0)
        
        top_k = 2
        
        sim_users_q0_0 = rr.get_users_given_users(user_test_dict, top_k=top_k)
        self.assertTrue(isinstance(sim_users_q0_0, np.ndarray))
        self.assertTrue(sim_users_q0_0.shape == (n_users, top_k))
        
        sim_movies_q0_1 = rr.get_movies_given_users(user_test_dict, top_k=top_k, rm_hist=True)
        self.assertTrue(sim_movies_q0_1.shape == (n_users, top_k))
        
        sim_movies_q0_2 = rr.get_movies_given_users(user_test_dict, top_k=top_k, rm_hist=False)
        self.assertTrue(sim_movies_q0_2.shape == (n_users, top_k))
        
        sim_movies_q1_0 = rr.get_movies_given_movies(movie_test_dict, top_k=top_k)
        self.assertTrue(sim_movies_q1_0.shape == (n_movies, top_k))
        
        sim_users_q1_1 = rr.get_users_given_movies(movie_test_dict, top_k=top_k)
        self.assertTrue(sim_users_q1_1.shape == (n_movies, top_k))
        
    if __name__ == '__main__':
        unittest.main()

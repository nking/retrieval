from typing import Union, List, Dict, Tuple, Set

import numpy as np
import tensorflow as tf
# from google.protobuf import text_format
import glob
from enum import Enum
from absl import logging
from collections import defaultdict
from array_record.python import array_record_module
import msgpack

from movie_lens_retrieval.MovieData import MovieData
from movie_lens_retrieval.UserData import UserData

logging.set_verbosity(logging.WARNING)
logging.set_stderrthreshold(logging.WARNING)

import scann

"""
NOTE that data should only contain data up to and including training data and eval data. no test
data should be included.

For cloud based Retriever, can adapt for services for ScANN.

"""

class EmbeddingType(int, Enum):
    USER = 0
    MOVIE = 1

class Retriever:
    #  easier to hard code a dictionary for now than install and import tfx transform to read schema
    # string serialized examples => dict of inputs
    feature_spec = {
        "user_id": tf.io.FixedLenFeature([], tf.int64),
        "movie_id": tf.io.FixedLenFeature([], tf.int64),
        #"rating": tf.io.FixedLenFeature([], tf.int64),
        "timestamp": tf.io.FixedLenFeature([], tf.int64),
        "gender": tf.io.FixedLenFeature([], tf.string),
        "age": tf.io.FixedLenFeature([], tf.int64),
        "occupation": tf.io.FixedLenFeature([], tf.int64),
        "genres": tf.io.FixedLenFeature([], tf.string)}
    
    feature_spec_movie = {
        "movie_id": tf.io.FixedLenFeature([], tf.int64),
        "genres": tf.io.FixedLenFeature([], tf.string)}
    
    feature_spec_user = {
        "user_id": tf.io.FixedLenFeature([], tf.int64),
        "gender": tf.io.FixedLenFeature([], tf.string),
        "age": tf.io.FixedLenFeature([], tf.int64),
        "occupation": tf.io.FixedLenFeature([], tf.int64),
        "timestamp": tf.io.FixedLenFeature([], tf.int64),
        }
    
    def __init__(self,
            user_movie_saved_model_dir: str,
            movie_id_offset:int,
            user_embed_path: str,
            movie_embed_path: str,
            embed_dim: int,
            cold_start_movie_path: str,
            users_path: str,
            movies_path: str,
            user_movie_hist_path_patterns: List[str],
            max_k: int = 1000,
    ):
        """
        :param user_movie_saved_model_dir: path to the saved_model directory 
           for the  user_movie bi-encoder model.   the query and candidate embeddings model signatures are in this.
            
        :param movies_path: path to the TFRecords of the movies having columns
           movie_id, title, genres,
        
        :param users_path: path to the TFRecords of all users. has columns movie_id, age, gender, occupation,

        :param user_embed_path: path to TFRecords of all user embeddings made
           from latest trained Query model.

        :param movie_embed_path: path to TFRecords of all movie embeddings made
           from latest trained Candidate model.

        :param max_k: the maximum number of embeddings to return from a 
        ScANN embedding search.  This should be higher
        than the top_k desired to account for later removing movies already seen.  
        The default is 1000.
        
        :param embed_dim: the dimensionality of the embeddings
        """
        self.movie_id_offset = movie_id_offset
        
        self.user_data = UserData(users_path)
        self.movie_data = MovieData(movies_path, movie_id_offset)
        
        self.max_k = max_k
        
        self.cold_start_movie_list = Retriever._read_cold_start(cold_start_movie_path)
        
        self.loaded_user_movie_model = tf.saved_model.load(user_movie_saved_model_dir)
        
        self.embed_dim = embed_dim
        
        self.user_indexers, self.user_indexers_ids = Retriever._create_user_indexer(
            user_embed_path, self.max_k, self.embed_dim)
        
        self.movie_indexers, self.movie_indexers_ids = Retriever._create_movie_indexer(
            movie_embed_path, self.max_k, self.embed_dim)
        
        self.user_history_dict, self.max_hist = Retriever._read_user_ratings_histories(
            user_movie_hist_path_patterns)
    
    @staticmethod
    def _create_indexer_and_tables(embedding_type: EmbeddingType,
            embed_file_path: str, max_k: int, embed_dim: int = 16) -> Tuple[scann.scann_ops_pybind.ScannSearcher, tf.Tensor]:
        
        # read tfrecords
        _ct = "GZIP" if embed_file_path.endswith(".gz") else None
        file_paths = glob.glob(embed_file_path)
        if len(file_paths) == 0:
            raise FileNotFoundError(embed_file_path)
        
        embed_ds_ser = tf.data.TFRecordDataset(file_paths, compression_type=_ct)
        
        is_not_empty = embed_ds_ser.reduce(False, lambda x, _: True)
        #tf.debugging.assert_equal(is_not_empty, True,
        #    message="TFRecordDataset is empty")
        if not is_not_empty:
            raise ValueError(f"the files is empty at {embed_file_path}")
        
        key = "user_id" if embedding_type == EmbeddingType.USER else "movie_id"
        # ids is a Tensor of ints
        # embeddings is a tensor of float arrays of length embed_dim
        ids, embeddings = Retriever._parse_emb_tfrecord_ser_into_lists(
            embed_ds_ser, key, embed_dim)
    
        indexer = Retriever.build_scann_searcher(embeddings=embeddings,
            top_k=max_k)
        
        return indexer, ids
    
    @staticmethod
    def _create_movie_indexer(movie_embed_path: str, max_k: int,
            embed_dim: int = 16) -> Tuple[
        scann.scann_ops_pybind.ScannSearcher, tf.Tensor]:
        
        return Retriever._create_indexer_and_tables(EmbeddingType.MOVIE,
            movie_embed_path, max_k, embed_dim)
    
    @staticmethod
    def _create_user_indexer(user_embed_path: str, max_k: int,
            embed_dim: int = 16) -> Tuple[scann.scann_ops_pybind.ScannSearcher, tf.Tensor]:
        
        return Retriever._create_indexer_and_tables(EmbeddingType.USER,
            user_embed_path, max_k, embed_dim)
    
    @staticmethod
    def _parse_emb_tfrecord_ser_into_lists(ds_ser: tf.data.TFRecordDataset,
            id_key: str, embed_dim: int = 16) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        given a tfrecorddataset of ids and embeddings, parse them to return 3 datasetructures:
        a tensor of the ids in the dataset,
        a tensor of the embeddings (which are float arrays of length embed_dim),
        
        :param ds_ser: tfrecorddataset holding string serialized examples
        :param id_key:  the id_key used when writing the tfrecords.  e.g. "movie_id" or "movie_id"
        :param embed_dim: the length of the embeddings in ds_ser
        :return: a tensor of the ids in the dataset,
        a tensor of the embeddings (which are float arrays of length embed_dim),
        """
        em_feature_spec = {
            id_key: tf.io.FixedLenFeature(shape=[], dtype=tf.int64,
                default_value=None),
            "embedding": tf.io.FixedLenFeature(shape=[embed_dim], dtype=tf.float32)
            #"embedding": tf.io.VarLenFeature(tf.float32)
        }
        
        def parse_tf_example(example_proto, spec, key:str):
            d = tf.io.parse_single_example(example_proto, spec)
            # d={'embedding': <tf.Tensor 'ParseSingleExample/ParseExample/ParseExampleV2:0' shape=(16,) dtype=float32>,
            # 'movie_id': <tf.Tensor 'ParseSingleExample/ParseExample/ParseExampleV2:1' shape=() dtype=int64>}
            return {
                "id": d[key],
                "embedding": d["embedding"],
            }
        
        ds = ds_ser.map(lambda x: parse_tf_example(x, em_feature_spec, id_key), num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        
        #def assert_not_empty(dataset):
        #    is_not_empty = dataset.reduce(False, lambda x, _: True)
        #    tf.debugging.assert_equal(is_not_empty, True, "ds contains 0 elements")
        #assert_not_empty(ds.take(1))
        
        ids = []
        embeddings = []
        for batch in ds.batch(1024):
            ids.append(batch['id'])  #Tensor shape (batch_size, 1)
            embeddings.append(batch['embedding'])  #Tensor shape (batch_size, 16)
        ids = tf.concat(ids, axis=0)
        embeddings = tf.concat(embeddings, axis=0)
        return ids, embeddings
    
    @tf.function
    def _validate_tensor_and_int(self, t):
        # Dtype Check: Static (Traced once, zero runtime cost)
        if t.dtype != tf.int64:
            raise TypeError(f"Expected dtype tf.int64 but got {t.dtype}")
        
        # Use t.shape.rank (Python property) to avoid IndexErrors during tracing
        static_rank = t.shape.rank
        dynamic_rank = tf.rank(t)
        dynamic_shape = tf.shape(t)
        
        # Condition: (Rank 0) OR (Rank 2 AND Width 1)
        is_scalar = tf.equal(dynamic_rank, 0)
        
        # We only safely look at shape[-1] if we know it's not a scalar.
        # We use a simple 0 for the scalar case to avoid the IndexError.
        last_dim = dynamic_shape[-1] if static_rank != 0 else 0
        is_column = tf.logical_and(tf.equal(dynamic_rank, 2),
            tf.equal(last_dim, 1))
        
        # Execution-time Assertion
        tf.Assert(tf.logical_or(is_scalar, is_column),
            [t, "Shape must be () or (None, 1)"])
        
    def _check_inputs_for_id(self, inputs_type:EmbeddingType,
        inputs: Union[
        tf.Tensor,
        List[List[int]],
        Dict[str, tf.Tensor],
        Dict[str, Union[int, str]],
        List[Dict[str, Union[int, str]]]]):
        
        if inputs_type is None:
            raise ValueError("inputs_type cannot be None")
        
        if isinstance(inputs, tf.Tensor): #or tf.is_tensor(inputs)
            #can be scalar with shape=() or array with shape=(None,1)
            self._validate_tensor_and_int(inputs)
        elif isinstance(inputs, list):
            #list of integers or list of dictionaries
            if len(inputs) == 0:
                raise ValueError(f"expected at least 1 input for {inputs_type}")
            if not isinstance(inputs[0], list):
                if not isinstance(inputs[0], dict):
                    raise TypeError(f"inputs list must contain integers or dictionaries, got {type(inputs[0])}")
                #assert dictionaries have expected ids
                for d in inputs:
                    if inputs_type == EmbeddingType.USER and 'user_id' not in d:
                        raise TypeError(f"dictionaries must all contain at least 'user_id' key")
                    elif inputs_type == EmbeddingType.MOVIE and 'movie_id' not in d:
                        raise TypeError(f"dictionaries must all contain at least 'movie_id' key")
        elif isinstance(inputs, dict):
            #dictionary of tensors or dictionary of scalars, either way, needs id key
            if inputs_type == EmbeddingType.USER and 'user_id' not in inputs:
                raise TypeError(
                    f"dictionary input must all contain at least 'user_id' key")
            elif inputs_type == EmbeddingType.MOVIE and 'movie_id' not in inputs:
                raise TypeError(
                    f"dictionary input must all contain at least 'movie_id' key")
        else:
            raise TypeError(f"type {type(inputs)} not supported")
        pass
    
    @staticmethod
    def _format_inputs_only_to_dict_tensors(inputs_type:EmbeddingType,
        inputs: Union[
        tf.Tensor,
        List[List[int]],
        Dict[str, tf.Tensor],
        Dict[str, Union[int, str]],
        List[Dict[str, Union[int, str]]]]) -> Dict[str, tf.Tensor]:
        """
        note that inputs should have already been checked with self._check_inputs_for_id(inputs_type, inputs)
        :param inputs_type:
        :param inputs:
        :return:
        """
        
        if isinstance(inputs, tf.Tensor):
            #tensor of shape () or shape (None, 1)
            key = 'user_id' if inputs_type == EmbeddingType.USER else 'movie_id'
            if tf.rank(inputs) == 0:
                return {key: tf.constant([inputs], dtype=tf.int64)}
            else:
                return {key: inputs}
        
        if isinstance(inputs, list):
            #list of integers or list of dictionaries
            if isinstance(inputs[0], list):
                key = 'user_id' if inputs_type == EmbeddingType.USER else 'movie_id'
                return {key: tf.constant(inputs, dtype=tf.int64)}
            #else is list of dictionaries of scalars
            inter = Retriever.feature_spec.keys() & inputs[0].keys()
            outp_dict0 = {key: [] for key in inter}
            for d in inputs:
                for k,v in d.items():
                    outp_dict0[k].append([v])
            outp_dict = {}
            for k in outp_dict0.keys():
                outp_dict[k] = tf.constant(outp_dict0[k], dtype=Retriever.feature_spec[k].dtype)
            return outp_dict
        
        #else is dictionary of scalars or dictionary of tensors
        key = 'user_id' if inputs_type == EmbeddingType.USER else 'movie_id'
        if isinstance(inputs[key], tf.Tensor):
            #is already formatted as tensors
            return inputs
        
        #format input for the items that are within inputs.items()
        inter = Retriever.feature_spec.keys() & inputs.keys()
        outp_dict0 = {key: [] for key in inter}
        for k, v in inputs.items():
            outp_dict0[k].append([v])
        outp_dict = {}
        for k in outp_dict0.keys():
            outp_dict[k] = tf.constant(outp_dict0[k], dtype=Retriever.feature_spec[k].dtype)
        return outp_dict
        
    def create_dictionary_of_tensors(self, inputs_type:EmbeddingType,
        inputs: Union[
        tf.Tensor,
        List[List[int]],
        Dict[str, tf.Tensor],
        Dict[str, Union[int, str]],
        List[Dict[str, Union[int, str]]]]) -> Dict[str, tf.Tensor]:
        """
        given inputs_type and inputs, create a dictionary of tensors usable for model inputs.
        
        :param inputs_type: the type of the inputs for the model
        
        :param inputs:  the following datastructures are accepted.
            - tf.Tensor of ids
           - list of ids
           - dictionary of ids and optional keys with values as scalars
           - dictionary of ids and optional keys with values as tensors of arrays
           - list of dictionary of ids and optional keys
           where for EmbeddingType.USER, the ids are 'user_id' and optional keys are 'gender', 'age', 'occupation', 'timestamp'
           and for EmbeddingType.MOVIE, the ids are 'movie_id' and optional keys are 'genres'.
        :return: a dictionary of tensors where each key's value is a tensor holding a an array of single item lists.
        e.g 'age': <tf.Tensor: shape=(2, 1), dtype=int64, numpy=array([[18], [1]])>
        """
        if inputs_type == EmbeddingType.USER:
            #check that user_id or movie_id exists
            self._check_inputs_for_id(inputs_type, inputs)
            
        #create dictionary of tensors from inputs first
        inp_dict = Retriever._format_inputs_only_to_dict_tensors(inputs_type, inputs)
        
        if inputs_type == EmbeddingType.MOVIE:
            diff = Retriever.feature_spec_movie.keys() - inp_dict.keys()
            if len(diff) > 0:
                movie_data = self.movie_data.get_movie(inp_dict['movie_id'])
                for k in diff:
                    inp_dict[k] = movie_data[k]
        else:
            diff = Retriever.feature_spec_user.keys() - inp_dict.keys()
            if len(diff) > 0:
                if 'timestamp' not in inp_dict:
                    inp_dict['timestamp'] = tf.constant([[-1] for _ in range(len(inp_dict['user_id']))], dtype=tf.int64)
                user_data = self.user_data.get_user(inp_dict['user_id'], inp_dict['timestamp'])
                for k in diff:
                    inp_dict[k] = user_data[k]
        
        return inp_dict
   
    @staticmethod
    def build_scann_searcher(embeddings: tf.Tensor, top_k: int):
        """
        build an ScANN indexer initialized with embeddings, and top_k number of nearest neighbors,
        and the brute force algorithm.
        TODO: tune configuration for high performance and accuracy.
      
        Usage: neighbors, distances = searcher.search_batched(query_embedding)
      
        to use scann.
        # https://github.com/google-research/google-research/blob/master/scann/docs/example.ipynb
        # https://github.com/google-research/google-research/blob/master/scann/docs/algorithms.md
        """
        builder = scann.scann_ops_pybind.builder(db=embeddings,
            num_neighbors=top_k, distance_measure="dot_product")
        
        n_embeddings = tf.shape(embeddings)[0]
        if n_embeddings < 20000:
            return builder.score_brute_force(quantize=False).build()
        
        # Rule of thumb: num_leaves should be roughly sqrt(N)
        # We cap it at 100,000 for extremely large datasets
        n_leaves = int(n_embeddings**0.5)
        
        # We search ~5-10% of leaves to maintain high recall
        n_leaves_to_search = max(20, n_leaves // 20)
        
        builder = builder.tree(
            num_leaves=n_leaves,
            num_leaves_to_search=n_leaves_to_search,
            training_sample_size=min(n_embeddings, 250000)
        )
        
        # Add Asymmetric Hashing (AH)
        # This is worth doing for almost any dataset over 10k
        builder = builder.score_ah(
            dimensions_per_block=2,
            anisotropic_quantization_threshold=0.2
        )
        
        # Rescore (Reordering)
        # We re-rank a small fraction of the top AH results with brute-force math
        # for 100% precision on the final top_k.
        builder = builder.reorder(reordering_num_neighbors=top_k * 10)
        
        return builder.build()
    
    def get_cold_start_movie_recommendations(self, top_k: int = 100):
        return self.cold_start_movie_list[:top_k].copy()
    
    def _create_embeddings(self, inp_data_type:EmbeddingType, inp_data:Dict[str, tf.Tensor]) -> tf.Tensor:
        if inp_data_type == EmbeddingType.USER:
            infer_for_dict = self.loaded_user_movie_model.signatures[
                "serving_query_dict"]
            embeddings_list = infer_for_dict(
                age=inp_data['age'],
                gender=inp_data['gender'],
                occupation=inp_data['occupation'],
                timestamp=inp_data['timestamp'],
                user_id=inp_data['user_id'])
        else:
            infer_for_dict = self.loaded_user_movie_model.signatures[
                "serving_candidate_dict"]
            embeddings_list = infer_for_dict(
                genres=inp_data['genres'],
                movie_id=inp_data['movie_id'])
        # k X embed_dim
        # embeddings_list is a single tensor with a 2D-array of embeddings
        output_keyword = list(infer_for_dict.structured_outputs.keys())[0]
        return embeddings_list[output_keyword]
    
    def get_users_given_users(self, inputs: Dict[str, tf.Tensor], top_k: int) -> np.ndarray:
        """
        given user query data, get nearest users
        :param inputs: dictionary of tensors
        :param top_k:
        :return: list of lists of top_k user_ids similar to those in user_data.
        """
        neear_ids = self._get_ann_ids(inp_data_type=EmbeddingType.USER,
            lookup_type=EmbeddingType.USER, inp_data=inputs, top_k=top_k, rm_hist = False)
        return neear_ids
    
    def _get_ann_ids(self, inp_data_type: EmbeddingType,
            lookup_type: EmbeddingType, inp_data: Dict[str, tf.Tensor], top_k: int, rm_hist:bool=False) -> np.ndarray:
        """
        get the nearest neighbor embeddings of lookup_type to the embedding to be made for inp_data of inp_data_type
        :param inp_data: dictionary of tensors where keys are for the inp_data_type model.
           example for EmbeddingType.USER:
               'user_id': tf.constant([[1], [2], [3]], dtype=tf.int64),
                'gender': tf.constant([["F"], ["M"], ["M"]], dtype=tf.string),
                'age': tf.constant([[1], [56], [25]], dtype=tf.int64),
                'occupation': tf.constant([[10], [16], [15]], dtype=tf.int64),
                'timestamp': tf.constant([[ts], [ts], [ts]], dtype=tf.int64),
            }
            example of EmbeddingType.MOVIE:
               {'movie_id': tf.constant([[6041], [6042], [6043]], dtype=tf.int64),
                'genres': tf.constant([["Animation|Children's|Comedy"], ["Adventure|Children's|Fantasy"], ["Comedy|Romance"]], dtype=tf.string),
            }
        :param top_k:
        :return: list of lists of top_k user or movie ids depending upon lookup_type, similar to those in inp_data.
        """
        if top_k < 1:
            raise ValueError('top_k must be >= 1')
        if top_k > self.max_k:
            top_k = self.max_k
            
        embeddings_tensor = self._create_embeddings(inp_data_type, inp_data)
        if lookup_type == EmbeddingType.USER:
            neighbor_idxs, cos_sim = self.user_indexers.search_batched(embeddings_tensor, top_k)
            #neighbor_idxs are user_id - 1 so add 1
            #neighbor idxs are 2D numpy arrays
            nearest_ids = neighbor_idxs + 1
        else:
            k = top_k + self.max_hist if rm_hist else top_k
            neighbor_idxs, cos_sim = self.movie_indexers.search_batched(embeddings_tensor, k)
            # neighbor_idxs are 0 to n_movies-1 so add 1 then add movie_id_offset
            # neighbor idxs are 2D numpy arrays
            nearest_ids = neighbor_idxs + self.movie_id_offset
            if rm_hist:
                nearest_ids = self.filter_watched(inp_data['user_id'], nearest_ids, top_k)
                nearest_ids = np.array(nearest_ids)
        return nearest_ids
    
    def get_user_history(self, user_ids: tf.Tensor) -> List[Set[int]]:
        out = []
        for i, user_id in enumerate(user_ids.numpy()):
            history = self.user_history_dict.get(user_id[0], set())
            out.append(history)
        return out
    
    def filter_watched(self, user_ids: tf.Tensor, nearest_movie_ids: np.ndarray,
        top_k:int=10) -> List[List[int]]:
        filtered_results = []
        # Convert TF tensor to numpy for the loop if it isn't already
        u_ids_np = user_ids.numpy()
        for i, user_id in enumerate(u_ids_np):
            history = self.user_history_dict.get(user_id[0], set())
            clean_row = [m_id for m_id in nearest_movie_ids[i] if m_id not in history][:top_k]
            filtered_results.append(clean_row)
        return filtered_results
    
    def get_movies_given_movies(self, movie_data: Dict[str, tf.Tensor], top_k: int) -> np.ndarray:
        """
        get nearest movies to given movie data
        :param movie_data: dictionary or list of dictionaries of inputs where keys must be all columns from the joined ratings file that the models
        were trained upon, OR, the dictionary must contain only the movie_id or movie_data can be  a list of movie_ids.
        :param top_k:
        :return: list of lists of top_k movie ids similar to those in movie_data.
        """
        return self._get_ann_ids(inp_data_type=EmbeddingType.MOVIE,
            lookup_type=EmbeddingType.MOVIE, inp_data=movie_data, top_k=top_k, rm_hist=False)
    
    def get_users_given_movies(self, movie_data: Dict[str, tf.Tensor], top_k: int) -> np.ndarray:
        """
        get nearest users to given movie data
        :param movie_data: dictionary or list of dictionaries of inputs where keys must be all columns from the joined ratings file that the models
        were trained upon, OR, the dictionary must contain only the movie_id or movie_data can be  a list of movie_ids.
        :param top_k:
        :return: list of lists of top_k user ids similar to those in movie_data.
        """
        return self._get_ann_ids(inp_data_type=EmbeddingType.MOVIE,
            lookup_type=EmbeddingType.USER, inp_data=movie_data, top_k=top_k, rm_hist=False)
    
    def get_movies_given_users(self, user_data: Dict[str, tf.Tensor], top_k: int, rm_hist:bool=True) -> np.ndarray:
        """
        get nearest movies to given user data
        :param user_data: dictionary or list of dictionaries of inputs where keys must be all columns from the joined ratings file that the models
        were trained upon, OR, the dictionary must contain only the movie_id or user_data can be  a list of user_ids.
        :param top_k:
        :return: list of lists of top_k user ids similar to those in movie_data.
        """
        return self._get_ann_ids(inp_data_type=EmbeddingType.USER,
            lookup_type=EmbeddingType.MOVIE, inp_data=user_data, top_k=top_k, rm_hist=rm_hist)
    
    def user_is_known(self, user_id) -> bool:
        return self.user_data.user_exists(user_id)
    
    def users_are_known(self, user_ids:tf.Tensor) -> tf.Tensor:
        return self.user_data.users_exist(user_ids)
    
    def get_cos_sim_score(self, user_movie_data_dict: Dict[str, tf.Tensor]):
        
        infer_default_for_dict = self.loaded_user_movie_model.signatures[
            "serving_default_dict"]
        
        predictions = infer_default_for_dict(
            age=user_movie_data_dict['age'],
            gender=user_movie_data_dict['gender'],
            genres=user_movie_data_dict['genres'],
            movie_id=user_movie_data_dict['movie_id'],
            occupation=user_movie_data_dict['occupation'],
            timestamp=user_movie_data_dict['timestamp'],
            user_id=user_movie_data_dict['user_id'])
        tensor_pred = predictions['outputs']
        tensor_pred = tf.reshape(tensor_pred, -1)
        
        return tensor_pred.numpy().tolist()
    
    @staticmethod
    def _read_cold_start(cold_start_movie_path:str) -> List[int]:
        with open(cold_start_movie_path, 'r') as f:
            # f is an iterator; this is highly optimized in CPython
            numbers = [int(line) for line in f]
        return numbers
    
    @staticmethod
    def _read_user_ratings_histories(path_patterns: List[str], batch_size:int=2048) -> Tuple[defaultdict, int]:
        """
        read in array_records
        :param path_patterns:
        :return:
        """
        d = defaultdict(set)
        file_paths = []
        for path_pattern in path_patterns:
            _paths = glob.glob(path_pattern)
            if len(_paths) == 0:
                raise ValueError(f"no files found at: {path_pattern}")
            file_paths.extend(_paths)
        for file_path in file_paths:
            reader = None
            try:
                reader = array_record_module.ArrayRecordReader(file_path)
                n = reader.num_records()
                for i in range(0, n, batch_size):
                    i_end = i + batch_size
                    if i_end >= n:
                        i_end = n
                    batch_bytes = reader.read([x for x in range(i, i_end)])
                    data = [msgpack.unpackb(b, use_list=False) for b in
                        batch_bytes]  # list of tuples of 4 integers
                    for record in data:
                        d[int(record[0])].add(int(record[1]))
            finally:
                if reader is not None:
                    reader.close()
        max_hist = 0
        for k, v in d.items():
            max_hist = max(max_hist, len(v))
        return d, max_hist
    
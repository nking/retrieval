from typing import Union, List, Dict, Tuple
import tensorflow as tf
# from google.protobuf import text_format
import random
import glob

from enum import Enum
from absl import logging

from movie_lens_retrieval.MovieData import MovieData
from movie_lens_retrieval.UserData import UserData

logging.set_verbosity(logging.WARNING)
logging.set_stderrthreshold(logging.WARNING)

import scann

"""
NOTE that data should only contain data up to and including training data and eval data. no test
data should be included.  The train, val, and test splits formed in the recommender_systems project
are first split into (train + val) and (test) by timestamp with test containing later timestamps,
then (train) and (val) are split into disjoint users.  The final partitions are roughly 80:10:10 percentages
in length.

For cloud based Retriever, can adapt for services for ScANN.

Bloom filters or efficient database and cache system can be used outside of this component to:
(1) check if user has already seen movies returned by this component.

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
    
    def __init__(self,
            user_movie_saved_model_dir: str,
            movie_id_offset:int,
            user_embed_path: str,
            movie_embed_path: str,
            embed_dim: int,
            cold_start_movie_path: str,
            users_path: str,
            movies_path: str,
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
        
        key = "movie_id" if embedding_type == EmbeddingType.USER else "movie_id"
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
        
        def parse_tf_example(example_proto, spec, id_key):
            d = tf.io.parse_single_example(example_proto, spec)
            #print(f'd={d}', flush=True)
            # d={'embedding': <tf.Tensor 'ParseSingleExample/ParseExample/ParseExampleV2:0' shape=(16,) dtype=float32>,
            # 'movie_id': <tf.Tensor 'ParseSingleExample/ParseExample/ParseExampleV2:1' shape=() dtype=int64>}
            return {
                "id": d[id_key],
                "embedding": d["embedding"],
            }
        
        ds = ds_ser.map(lambda x: parse_tf_example(x, em_feature_spec, id_key), num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        
        element = next(iter(ds.take(1)), None)
        if element is None:
            raise ValueError(f"could not parse embedding file.  expecting features: {id_key} and embedding and that embedding dim={embed_dim}")
        
        ids = []
        embeddings = []
        for batch in ds.batch(1024):
            ids.extend(batch['id'])
            embeddings.extend(batch['embedding'])
        ids = tf.stack(ids, axis=0)
        embeddings = tf.stack(embeddings, axis=0)
        return ids, embeddings
    
    @tf.function
    def validate_tensor_and_int(self, t):
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
        List[int],
        Dict[str, tf.Tensor],
        Dict[str, Union[int, str]],
        List[Dict[str, Union[int, str]]]]):
        
        if inputs_type is None:
            raise ValueError("inputs_type cannot be None")
        
        if isinstance(inputs, tf.Tensor): #or tf.is_tensor(inputs)
            #can be scalar with shape=() or array with shape=(None,1)
            self.validate_tensor_and_int(inputs)
        elif isinstance(inputs, list):
            #list of integers or list of dictionaries
            if len(inputs) == 0:
                raise ValueError(f"expected at least 1 input for {inputs_type}")
            if not isinstance(inputs[0], int):
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
    
    def _format_inputs_only_to_dict_tensors(self, inputs_type:EmbeddingType,
        inputs: Union[
        tf.Tensor,
        List[int],
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
            if isinstance(inputs[0], int):
                key = 'user_id' if inputs_type == EmbeddingType.USER else 'movie_id'
                return {key: tf.constant(inputs, dtype=tf.int64)}
            #else is list of dictionaries of scalars
            outp_dict = {key: [] for key in Retriever.feature_spec.keys()}
            for d in inputs:
                for k,v in d.items():
                    outp_dict[k].append(v)
            for k in outp_dict.keys():
                outp_dict[k] = tf.constant(outp_dict[k], dtype=Retriever.feature_spec[k].dtype)
            return outp_dict
        
        #else is dictionary of scalars or dictionary of tensors
        key = 'user_id' if inputs_type == EmbeddingType.USER else 'movie_id'
        if isinstance(inputs[key], tf.Tensor):
            #is already formatted as tensors
            return inputs
        
        outp_dict = {key: [] for key in Retriever.feature_spec.keys()}
        for k, v in inputs.items():
            outp_dict[k].append(v)
        for k in outp_dict.keys():
            outp_dict[k] = tf.constant(outp_dict[k], dtype=Retriever.feature_spec[k].dtype)
        return outp_dict
        
    def _create_dictionary_of_tensors(self, inputs_type:EmbeddingType,
        inputs: Union[
        tf.Tensor,
        List[int],
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
        inp_dict = self._format_inputs_only_to_dict_tensors(inputs_type, inputs)
        
        editing for refactored signatures
        
        #lookup mising information and supply timestamp if value is -1
        
        
        return outp_dict
   
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
        bind1 = scann.scann_ops_pybind.builder(db=embeddings,
            num_neighbors=top_k, distance_measure="dot_product")
        searcher = bind1.score_brute_force(quantize=False).build()
        return searcher
    
    def get_cold_start_movie_recommendations(self, top_k: int = 100):
        return self.cold_start_movie_list[:top_k].copy()
    
    def _create_embeddings(self, embedding_type: EmbeddingType, inputs: Union[
        Dict[str, Union[int, str]], List[Dict[str, Union[int, str]]]]) -> tf.Tensor:
        """
        given inputs, use the query model to make embeddings.
        :param inputs: dictionary of inputs where keys must be all columns from the joined ratings file that the models
        were trained upon, OR, the dictionary must contain only the movie_id.
        :return: embeddings usable for the vector approx nearest neighbor searches.
        output format is tensor of shape (len(inputs as a list
        """
        if not isinstance(inputs, list):
            inputs = [inputs]
        use_ser = False
        key = "movie_id" if embedding_type == EmbeddingType.USER else "movie_id"
        for inp_dict in inputs:
            if not isinstance(inp_dict, dict) or "movie_id" not in inp_dict:
                raise ValueError(
                    "expecting inputs  to be a dictionary that includes movie_id "
                    "or a list of dictionaries containing movie_id")
            id1 = inp_dict[key]
            if len(inp_dict) == 1:
                use_ser = True
                break
            if Retriever.feature_spec.keys() != inp_dict.keys():
                raise ValueError(
                    f"expected inputs keys to be: {Retriever.feature_spec.keys()}")
            break
        if use_ser:
            self._create_embeddings_given_ids(embedding_type, inputs)
        return self._create_embeddings_given_dicts(embedding_type, inputs)
    
    def _create_embeddings_given_ids(self, embedding_type: EmbeddingType,
            inputs: Union[Dict[str, int], List[Dict[str, int]], List[
                int], tf.Tensor]) -> tf.Tensor:
        """
        given inputs, use the query_candidate model to make embeddings.
        :param inputs: dictionary or list of dictionaries of inputs containing only the movie_id.
        :return: embeddings usable for the vector approx nearest neighbor searches.
        output format is tensor of shape (len(inputs as a list
        """
        editing for refactored signatures

        key = "movie_id" if embedding_type == EmbeddingType.USER else "movie_id"
        if isinstance(inputs, list):
            if isinstance(inputs[0], int):
                inputs = tf.constant(inputs, dtype=tf.int64)
            elif isinstance(inputs[0], tf.Tensor):
                inputs = tf.stack(inputs, axis=0)
            elif isinstance(inputs[0], dict):
                tmp = [d[key] for d in inputs]
                inputs = tf.constant(tmp, dtype=tf.int64)
        elif isinstance(inputs, dict):
            tmp = [inputs[key]]
            inputs = tf.constant(tmp, dtype=tf.int64)
        if not isinstance(inputs, tf.Tensor):
            raise ValueError(
                f"expecting inputs to be a tensor at this stage.  type={inputs}")
        if embedding_type == EmbeddingType.USER:
            examples_ser_list = self.user_id_to_ser_ht.lookup(inputs)
            infer = self.loaded_user_movie_model.signatures["serving_query"]
        else:
            examples_ser_list = self.movie_id_to_ser_ht.lookup(inputs)
            infer = self.loaded_user_movie_model.signatures["serving_candidate"]
            
        INPUT_KEY = list(infer.structured_input_signature[1].keys())[0]
        output_keyword = list(infer.structured_outputs.keys())[0]
        #TODO: add try/except error handling
        embeddings_list = infer(**{INPUT_KEY: examples_ser_list})[output_keyword]
        
        # embeddings_list is a single tensory with a 2D-array of embeddings
        return embeddings_list
    
    def _create_embeddings_given_dicts(self, embedding_type: EmbeddingType,
            inputs: Union[Dict[str, Union[int, str]], List[
                Dict[str, Union[int, str]]]]) -> tf.Tensor:
        """
        given inputs, use the query_candidate model to make embeddings.
        :param inputs: dictionary or list of dictionary of inputs where keys must be all columns from the joined ratings file that the models
        were trained upon.
        :return: embeddings usable for the vector approx nearest neighbor searches.
        output format is tensor of shape (len(inputs as a list
        """
        editing for refactored signatures

        n = len(Retriever.feature_spec)
        if not isinstance(inputs, list):
            inputs = [inputs]
        #batch format is a dictionary of the feature keys where each dictionary value is a Tensor of the list of values for the key
        if (len(inputs) == 1 and inputs[0].keys() == Retriever.feature_spec.keys()
                and sum([int(isinstance(inputs[0][key], tf.Tensor)) for key in Retriever.feature_spec.keys()]) == n):
            batch = inputs[0]
        else:
            batch = {key: [] for key in Retriever.feature_spec.keys()}
            for inp_dict in inputs:
                if not isinstance(inp_dict, dict) or "movie_id" not in inp_dict:
                    raise ValueError(
                        "expecting inputs  to be a dictionary that includes movie_id "
                        "or a list of dictionaries containing movie_id")
                if Retriever.feature_spec.keys() != inp_dict.keys():
                    raise ValueError(
                        f"expected inputs keys to be: {Retriever.feature_spec.keys()}")
                # fake_dict = Retriever._create_serialized_tfexample(inp_dict))
                for key in Retriever.feature_spec.keys():
                    batch[key].append(inp_dict[key])
        if embedding_type == EmbeddingType.USER:
            infer_for_dict = self.loaded_user_movie_model.signatures[
                "serving_query_dict"]
        else:
            infer_for_dict = self.loaded_user_movie_model.signatures[
                "serving_candidate_dict"]
        embeddings_list = infer_for_dict(
            age=batch['age'],
            gender=batch['gender'],
            genres=batch['genres'],
            movie_id=batch['movie_id'],
            occupation=batch['occupation'],
            timestamp=batch['timestamp'],
            user_id=batch['movie_id'])
        # k X embed_dim
        # embeddings_list is a single tensory with a 2D-array of embeddings
        output_keyword = list(infer_for_dict.structured_outputs.keys())[0]
        return embeddings_list[output_keyword]
    
    def get_users_given_users(self, user_data: Union[
        Dict[str, Union[int, str]], List[Dict[str, Union[int, str]]], List[
            int]], top_k: int):
        """
        given user query data, get nearest users
        :param user_data: dictionary or list of dictionaries of inputs where keys must be all columns from the joined ratings file that the models
        were trained upon, OR, the dictionary must contain only the movie_id or user_data can be  a list of user_ids.
        :param top_k:
        :return: list of lists of top_k user_ids similar to those in user_data.
        """
        return self._get_ann_ids(inp_data_type=EmbeddingType.USER,
            lookup_type=EmbeddingType.USER, inp_data=user_data, top_k=top_k)
    
    def _get_ann_ids(self, inp_data_type: EmbeddingType,
            lookup_type: EmbeddingType, inp_data: Union[
                Dict[str, Union[int, str]], List[Dict[str, Union[int, str]]],
                List[int]], top_k: int):
        """
        get the nearest neighbor embeddings of lookup_type to the embedding to be made for inp_data of inp_data_type
        :param inp_data: dictionary or list of dictionaries of inputs where keys must be all columns from the joined ratings file that the models
        were trained upon, OR, the dictionary must contain only the id or inp_data can be  a list of ids.
        :param top_k:
        :return: list of lists of top_k user or movie ids depending upon lookup_type, similar to those in inp_data.
        """
        if top_k < 1:
            raise ValueError('top_k must be >= 1')
        if top_k > self.max_k:
            top_k = self.max_k
        embeddings_tensor = self._create_embeddings(inp_data_type, inp_data)
        if lookup_type == EmbeddingType.USER:
            neighbor_idxs, distances = self.user_indexers.search_batched(
                embeddings_tensor, top_k)
            nearest_ids = [[int(idx) for idx in self.id_to_user_id_ht.lookup(
                tf.constant(_list, dtype=tf.int64)).numpy()] for _list in
                neighbor_idxs]
        else:
            neighbor_idxs, distances = self.movie_indexers.search_batched(
                embeddings_tensor, top_k)
            nearest_ids = [[int(idx) for idx in self.id_to_movie_id_ht.lookup(
                tf.constant(_list, dtype=tf.int64)).numpy()] for _list in
                neighbor_idxs]
        
        if not isinstance(inp_data, list):
            inp_data = [inp_data]
        for inp, ids in zip(inp_data, nearest_ids):
            id0 = inp
            if isinstance(id0, dict):
                id0 = inp['movie_id'] if inp_data_type == EmbeddingType.USER else inp['movie_id']
            if isinstance(id0, tf.Tensor):
                id0 = id0.numpy().item()
            if id0 in nearest_ids:
                nearest_ids.remove(id0)
        return nearest_ids
    
    def get_movies_given_movies(self, movie_data: Union[
        Dict[str, Union[int, str]], List[Dict[str, Union[int, str]]], List[
            int]], top_k: int):
        """
        get nearest movies to given movie data
        :param movie_data: dictionary or list of dictionaries of inputs where keys must be all columns from the joined ratings file that the models
        were trained upon, OR, the dictionary must contain only the movie_id or movie_data can be  a list of movie_ids.
        :param top_k:
        :return: list of lists of top_k movie ids similar to those in movie_data.
        """
        return self._get_ann_ids(inp_data_type=EmbeddingType.MOVIE,
            lookup_type=EmbeddingType.MOVIE, inp_data=movie_data, top_k=top_k)
    
    def get_users_given_movies(self, movie_data: Union[
        Dict[str, Union[int, str]], List[Dict[str, Union[int, str]]], List[
            int]], top_k: int):
        """
        get nearest users to given movie data
        :param movie_data: dictionary or list of dictionaries of inputs where keys must be all columns from the joined ratings file that the models
        were trained upon, OR, the dictionary must contain only the movie_id or movie_data can be  a list of movie_ids.
        :param top_k:
        :return: list of lists of top_k user ids similar to those in movie_data.
        """
        return self._get_ann_ids(inp_data_type=EmbeddingType.MOVIE,
            lookup_type=EmbeddingType.USER, inp_data=movie_data, top_k=top_k)
    
    def get_movies_given_users(self, user_data: Union[
        Dict[str, Union[int, str]], List[Dict[str, Union[int, str]]], List[
            int]], top_k: int):
        """
        get nearest movies to given user data
        :param user_data: dictionary or list of dictionaries of inputs where keys must be all columns from the joined ratings file that the models
        were trained upon, OR, the dictionary must contain only the movie_id or user_data can be  a list of user_ids.
        :param top_k:
        :return: list of lists of top_k user ids similar to those in movie_data.
        """
        return self._get_ann_ids(inp_data_type=EmbeddingType.USER,
            lookup_type=EmbeddingType.MOVIE, inp_data=user_data, top_k=top_k)
    
    def user_is_known(self, user_id) -> bool:
        return self.user_data.user_exists(user_id)
    
    def users_are_known(self, user_ids:tf.Tensor) -> tf.Tensor:
        return self.user_data.users_exist(user_ids)
    
    def get_cos_sim_score(self, user_movie_data_dict: Union[
        Dict[str, Union[int, str]], List[Dict[str, Union[int, str]]]],
            as_tensor: bool = False):
        """
        given users and movies, return the cosine similarity score from the bi-encoder.
        :param as_tensor:
        :param user_movie_data_dict: a dictionary or list of dictionaries containing "movie_id", "age",
        "movie_id" and "genres".  the movie_id and genres can be lists instead of scalars, and
        if one is a list, the other must be also.
        
        example input:{"movie_id":1, "age":25, "movie_id":1, "genres":"Animation|Children's|Comedy"...}
        
        example input:{"movie_id":1, "age":25,
          "movie_id":[1, 3952],
          "genres":["Animation|Children's|Comedy", "Drama|Thriller"]...}
    
        :return: list of cosine similarity scores for the paired user and movie data
        """
        if not isinstance(user_movie_data_dict, list):
            user_movie_data_dict = [user_movie_data_dict]
        
        batch = Retriever._create_dictionary_of_tensors(user_movie_data_dict)
        
        infer_default_for_dict = self.loaded_user_movie_model.signatures[
            "serving_default_dict"]
        
        predictions = infer_default_for_dict(
            age=batch['age'],
            gender=batch['gender'],
            genres=batch['genres'],
            movie_id=batch['movie_id'],
            occupation=batch['occupation'],
            timestamp=batch['timestamp'],
            user_id=batch['movie_id'])
        tensor_pred = predictions['outputs']
        tensor_pred = tf.reshape(tensor_pred, -1)
        
        if as_tensor:
            return tensor_pred
        
        return tensor_pred.numpy().tolist()
    
    @staticmethod
    def _read_cold_start(cold_start_movie_path:str) -> List[int]:
        with open(cold_start_movie_path, 'r') as f:
            # f is an iterator; this is highly optimized in CPython
            numbers = [int(line) for line in f]
        return numbers
    
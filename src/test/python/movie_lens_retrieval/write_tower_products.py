"""
scripts to use the trained models from the recommender_systems project to
write user embeddings, movie embeddings and recommended movies.
These tower products are used in training and inference in the ranker project.

example usage from CLI:

export PIPELINE_DATA_DIR="../TMP10/bin/rs_pipeline"
export OUTPUT_BASE_DIR="../TMP10"
python3 -m unittest src.test.python.movie_lens_retrieval.write_tower_products.WriteTowerProducts

"""
from movie_lens_retrieval.MovieData import MovieData
from movie_lens_retrieval.UserData import UserData

from array_record.python import array_record_module
import pyarrow as pa
import pyarrow.parquet as pq

import json
import os
import shutil
import unittest
from typing import Any, Union, Tuple
import polars as pl
import msgpack
from array_record.python import array_record_module
import tensorflow as tf
import hashlib
from pathlib import Path
import re

# 0 = all logs, 1 = no info, 2 = no info/warn, 3 = no info/warn/error
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="umap")

import sys
is_cli = __name__ == '__main__' or any('unittest' in arg for arg in sys.argv)
if is_cli:
    print('** running from CLI **', flush=True)
    sys.path.insert(0,
        os.path.join(os.getcwd(), "src/test/python/movie_lens_retrieval"))
    sys.path.insert(0,
        os.path.join(os.getcwd(), "src/main/python/movie_lens_retrieval"))

from helper import get_project_dir, get_bin_dir

from movie_lens_retrieval.ANNSearcher import *

class WriteTowerProducts(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        
        cls.AnnSearchType = Index
        #cls.AnnSearchType = scann.scann_ops_pybind.ScannSearcher
        
        cls.default_top_k = 100
        
        pipeline_dir = os.environ.get("PIPELINE_DATA_DIR", None)
        if pipeline_dir:
            if not os.path.exists(pipeline_dir):
                raise ValueError(f"{pipeline_dir} does not exist")
            pushed_dir = os.path.join(pipeline_dir, "Pusher/pushed_model")
            if not os.path.exists(pushed_dir):
                raise ValueError(f"{pushed_dir} does not exist")
            push_version = get_largest_subdir_version(pushed_dir)
            cls.user_movie_models_dir = os.path.join(pushed_dir, push_version)
        else:
            saved_models_dir = os.path.join(get_project_dir(), "src/main/resources/serving_models")
            cls.user_movie_models_dir = os.path.join(saved_models_dir, "user_movie_model")
        
        output_base_dir = os.environ.get("OUTPUT_BASE_DIR", get_bin_dir())
        cls.output_dir = os.path.join(output_base_dir, "tower_products")
        shutil.rmtree(cls.output_dir, ignore_errors=True)
        os.makedirs(cls.output_dir, exist_ok=True)
        
        if not os.path.exists(WriteTowerProducts.user_movie_models_dir):
            raise ValueError(f'directory does not exist: {WriteTowerProducts.user_movie_models_dir}')
            
        cls.loaded_user_movie_model = tf.saved_model.load(WriteTowerProducts.user_movie_models_dir)
        
        test_res_dir = os.path.join(get_project_dir(), "src/test/resources/data")
        
        cls.users_path = os.path.join(test_res_dir, "users/users.parquet")
        cls.movies_path = os.path.join(test_res_dir, "movies/movies.parquet")
        
        cls.model_dict = read_model_assets_hparams(WriteTowerProducts.user_movie_models_dir)
        cls.MOVIE_OFFSET: int = cls.model_dict['n_users'] + 1
        embed_dim = json.loads(cls.model_dict['layer_sizes'])[-1]
        cls.user_id_range_incl = [1, cls.model_dict['n_users']]
        cls.movie_id_range_incl = [cls.MOVIE_OFFSET, cls.MOVIE_OFFSET + cls.model_dict['n_movies']]
        cls.recommender_systems_git_hash = cls.model_dict["git_hash"]
        
        cls.ratings_dict = {
            "full_history" : [
                    os.path.join(test_res_dir, "ratings_train.array_record"),
                    os.path.join(test_res_dir, "ratings_val.array_record")
                ],
            "positive_history" : [
                    os.path.join(test_res_dir, "ratings_train_liked.array_record"),
                    os.path.join(test_res_dir, "ratings_val_liked.array_record")
                ],
            "positive_train": os.path.join(test_res_dir, "ratings_train_liked.array_record"),
            "positive_val": os.path.join(test_res_dir, "ratings_val_liked.array_record"),
            "positive_test": os.path.join(test_res_dir, "ratings_test_liked.array_record"),
            "train": os.path.join(test_res_dir, "ratings_train.array_record"),
            "val": os.path.join(test_res_dir, "ratings_val.array_record"),
            "test": os.path.join(test_res_dir, "ratings_test.array_record"),
        }
        
        cls.user_data = UserData(cls.users_path)
        cls.movie_data = MovieData(cls.movies_path, cls.MOVIE_OFFSET)
        
        cls.train_history_df = read_ratings_to_df(cls.ratings_dict['positive_train'])

        cls.wrote_files = []
        
    @classmethod
    def tearDownClass(cls):
        for outfile in cls.wrote_files:
            print(f'wrote to {outfile}')

    def test_write(self):
        
        # the Query and Candidate models produce normalized embedding vectors
        
        # Tuple[tf.Tensor, tf.Tensor]
        catalog_movie_ids, movie_catalog_embeddings = create_movie_catalog_embeddings()
        
        write_embeddings("movie_emb", catalog_movie_ids, movie_catalog_embeddings)
        
        #Tuple[tf.Tensor, tf.Tensor]
        catalog_user_ids, timestamps, user_catalog_embeddings = create_user_catalog_embeddings()

        write_embeddings("user_emb", catalog_user_ids, user_catalog_embeddings)

        ## ===   write recommendations  =====
        n_search = int(tf.shape(movie_catalog_embeddings)[0])
        MOVIE_OFFSET = int(tf.shape(catalog_user_ids)[0]) + 1
        ann_searcher = build_usearch_searcher(movie_catalog_embeddings.numpy(), n_search, normalize=False)
        
        matches = ann_searcher.search(user_catalog_embeddings.numpy(), n_search)
        neighbors, distances = matches.keys, matches.distances
        neighbors += MOVIE_OFFSET
        
        write_recommendations(catalog_user_ids, neighbors, distances)
        
    @staticmethod
    def build_searcher(embeddings: Union[tf.Tensor, np.ndarray], top_k: int, normalize:bool=False):
        if isinstance(WriteTowerProducts.AnnSearchType, Index):
            return build_usearch_searcher(embeddings, top_k, normalize)
        else:
            return build_scann_searcher(embeddings, top_k, normalize)
    
def write_embeddings(outfile_root_name, ids: tf.Tensor, embeddings: tf.Tensor):
    
    embed_len = int(tf.shape(embeddings)[1])
    n_ids = int(tf.shape(ids)[0])
    
    outfile_metadata = os.path.join(WriteTowerProducts.output_dir, f"{outfile_root_name}_metadata.json")
    outfile_array_record = os.path.join(WriteTowerProducts.output_dir, f"{outfile_root_name}.array_record")
    outfile_parquet = os.path.join(WriteTowerProducts.output_dir, f"{outfile_root_name}.parquet")
    
    pa_schema = pa.schema([
        pa.field("id", pa.int64()),
        pa.field("embedding", pa.list_(pa.float32(), embed_len)),
    ])
   
    writer_a = None
    writer_p = None
    
    try:
        writer_a = array_record_module.ArrayRecordWriter(
            outfile_array_record, 'group_size:1')
        writer_p = pq.ParquetWriter(outfile_parquet, pa_schema)
        
        #TODO: consider improving these to write more efficient blocks
        for (id, embedding) in zip(ids.numpy(), embeddings.numpy()):
            id = id.item()
            embedding = embedding.tolist()
            writer_a.write(msgpack.packb((id, embedding)))
            pa_batch = pa.RecordBatch.from_arrays([
                pa.array([id], type=pa.int32()),
                pa.array([embedding], type=pa.list_(pa.int32(), embed_len))
            ], schema=pa_schema)
            writer_p.write_batch(pa_batch)
    
    finally:
        if writer_a is not None:
            writer_a.close()
        if writer_p is not None:
            writer_p.close()
            
    #spot assert that can read files
    reader = None
    try:
        reader = array_record_module.ArrayRecordReader(outfile_array_record)
        count = reader.num_records()
        batch_bytes = reader.read([x for x in range(0, count)])
        records = [msgpack.unpackb(b, use_list=True) for b in batch_bytes]
        
        assert(count == len(records))
        assert(count == len(ids))
        for i, record in enumerate(records):
            assert(isinstance(record[0], int))
            assert(isinstance(record[1], list))
            assert (isinstance(record[1][0], float))
            if i > 5:
                break
    finally:
        if reader is not None:
            reader.close()
    
    ## asserrt the parquet file
    table = pq.read_table(outfile_parquet)
    actual_data = table.to_pylist()
    assert(len(actual_data) == count)
    assert(isinstance(actual_data[0]['id'], int))
    assert(isinstance(actual_data[0]['embedding'], list))
    assert (isinstance(actual_data[0]['embedding'][0], float))
    
    # ==== write metadata ====
    md5sum_array_records = get_combined_md5(WriteTowerProducts.output_dir,
        fr'.*{outfile_root_name}\.array_record$')
    md5sum_parquets = get_combined_md5(WriteTowerProducts.output_dir,
        fr'.*{outfile_root_name}\.parquet')
    ##NOTE: to verify the md5sums fromcommand line, use:
    #    find . -type f -name "*.log" | sort | xargs cat | md5sum
    
    metadata = {
        "num_records": n_ids,
        "embed_dim": embed_len,
        "recommender_systems_git_hash": WriteTowerProducts.recommender_systems_git_hash,
        "md5sum_array_records" : md5sum_array_records,
        "md5sum_parquets" : md5sum_parquets
    }
    with open(outfile_metadata, "w") as f:
        json.dump(metadata, f)
    
    ## assert metadata file
    with open(outfile_metadata, "r") as f:
        metatadata_read = json.load(f)
        assert(metatadata_read == metadata)

    WriteTowerProducts.wrote_files.append(outfile_metadata)
    WriteTowerProducts.wrote_files.append(outfile_array_record)
    WriteTowerProducts.wrote_files.append(outfile_parquet)
    
def read_model_assets_hparams(a_user_movie_models_dir) -> dict[str, Any]:
    file_path = os.path.join(a_user_movie_models_dir,
        "assets.extra/hyperparameters.json")
    with open(file_path, 'r', encoding='utf-8') as f:
        hyperparameters = json.load(f)['values']
        return hyperparameters

def create_movie_catalog_embeddings() -> Tuple[tf.Tensor, tf.Tensor]:
    df = pl.read_parquet(WriteTowerProducts.movies_path)
    df = df.sort('movie_id')
    movie_ids = np.expand_dims(df['movie_id'].to_numpy(), axis=1)
    movie_ids = tf.constant(movie_ids, name='movie_id', dtype=tf.int64)
    
    inputs = WriteTowerProducts.movie_data.get_movie(movie_ids)
    
    infer_for_dict = WriteTowerProducts.loaded_user_movie_model.signatures[
        "serving_candidate_dict"]
    output_keyword = list(infer_for_dict.structured_outputs.keys())[0]
    embeddings_list = infer_for_dict(
        movie_id=inputs['movie_id'],
        genres=inputs['genres'])
    embeddings: tf.Tensor = embeddings_list[output_keyword]
    
    return movie_ids, embeddings

def create_user_catalog_embeddings() -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    
    ## write user embeddings given user_id and first timestamp from train, val history.  if no history is present,
    ## use the mean timestamp of history timestamps
    
    mean_timestamp = WriteTowerProducts.train_history_df.select(
        pl.col("timestamp").mean()
    ).item()
    
    #read user catalog:
    users_df = pl.read_parquet(WriteTowerProducts.users_path)
    users_df = users_df.select(pl.col("user_id"))
    
    #read train, val positive history dataframe and keep each unique user and their first timestamp
    users_2_df = (
        WriteTowerProducts.train_history_df
        .group_by("user_id")
        .agg(pl.col("timestamp").min().alias("timestamp"))
    )
    # join with all user_ids
    users_2_df = (
        users_df.join(users_2_df, on="user_id", how="left")
        .with_columns(pl.col("timestamp").fill_null(mean_timestamp))
    )
    users_2_df = users_2_df.sort('user_id')
    
    #extract user_ids and timestamps and format into tensors
    user_ids = np.expand_dims(users_2_df['user_id'].to_numpy(), axis=1)
    user_ids = tf.constant(user_ids, name='user_id', dtype=tf.int64)
    
    timestamps = np.expand_dims(users_2_df['timestamp'].to_numpy(), axis=1)
    timestamps = tf.constant(timestamps, name='timestamp', dtype=tf.int64)
    
    inputs = WriteTowerProducts.user_data.get_user(user_ids, timestamps)
    
    infer_for_dict = WriteTowerProducts.loaded_user_movie_model.signatures[
        "serving_query_dict"]
    output_keyword = list(infer_for_dict.structured_outputs.keys())[0]
    embeddings_list = infer_for_dict(
        age=inputs['age'],
        gender=inputs['gender'],
        occupation=inputs['occupation'],
        timestamp=inputs['timestamp'],
        user_id=inputs['user_id']
    )
    embeddings: tf.Tensor = embeddings_list[output_keyword]
    
    return user_ids, timestamps, embeddings


def read_ratings_to_df(file_path: str, batch_size: int = 2048) -> pl.DataFrame:
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
            batch_bytes = reader.read([x for x in range(i,
                i_end)])  # a single list of encodings, each being a list of 4 integers
            data = [msgpack.unpackb(b, use_list=False) for b in
                batch_bytes]  # list of tuples of 4 integers
            for record in data:
                records.append(
                    {'user_id': int(record[0]), 'movie_id': int(record[1]),
                        'rating': int(record[2]), 'timestamp': int(record[3])})
    finally:
        if reader is not None:
            reader.close()
    return pl.DataFrame(records)


def write_recommendations(catalog_user_ids, neighbors, distances):
    '''
    for each user, for each recommended movie, there is either a timestamp from train, or val
       else ts_2050 is used
    '''
    
    ts_2050 = 2524608000
    
    n_ids = int(tf.shape(catalog_user_ids)[0])
    num_movies = int(tf.shape(neighbors)[1])
    
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
    outfile = os.path.join(WriteTowerProducts.output_dir,
        "recommended_movies.array_record")
    outfile2 = os.path.join(WriteTowerProducts.output_dir,
        "recommended_movies_timestamps.array_record")
    pa_outfile = os.path.join(WriteTowerProducts.output_dir,
        "recommended_movies.parquet")
    pa_outfile2 = os.path.join(WriteTowerProducts.output_dir,
        "recommended_movies_timestamps.parquet")
    outfile_metadata = os.path.join(WriteTowerProducts.output_dir,
        "recommended_movies_metadata.json")
    
    # key  user_id, value={key=movie_id, value=timestamp}
    data = WriteTowerProducts.train_history_df.select(
        ["user_id", "movie_id", "timestamp"]
    ).to_dict(as_series=False)
    history_dict = {}
    for u, m, t in zip(data["user_id"], data["movie_id"], data["timestamp"]):
        if u not in history_dict:
            history_dict[u] = {}
        history_dict[u][m] = t
    
    # write array records and parquet records
    writer = None
    writer2 = None
    pa_movie_writer = None
    pa_timestamp_writer = None
    
    try:
        writer = array_record_module.ArrayRecordWriter(outfile, 'group_size:1')
        writer2 = array_record_module.ArrayRecordWriter(outfile2, 'group_size:1')
        
        pa_movie_writer = pq.ParquetWriter(pa_outfile, movies_schema)
        pa_timestamp_writer = pq.ParquetWriter(pa_outfile2, ts_schema)
        
        for user_id, movie_ids in zip(catalog_user_ids.numpy(), neighbors):
            user_id = user_id[0].item()
            movie_ids = movie_ids.tolist()
            assert num_movies == len(movie_ids)
            writer.write(msgpack.packb((user_id, movie_ids)))
            
            movie_batch = pa.RecordBatch.from_arrays([
                pa.array([user_id], type=pa.int32()),
                pa.array([movie_ids], type=pa.list_(pa.int32(), num_movies))
            ], schema=movies_schema)
            pa_movie_writer.write_batch(movie_batch)
            
            # create the timestamps
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
        
        assert(count == n_ids)
        for i, record in enumerate(records):
            assert(isinstance(record[0], int))
            assert(isinstance(record[1], tuple))
            assert(isinstance(record[1][0], int))
            assert(catalog_user_ids.numpy()[i].item() == record[0])
            assert(neighbors[i][0].item() == record[1][0])
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
        records = [msgpack.unpackb(b, use_list=False) for b in batch_bytes]
        
        assert (count == n_ids)
        for i, record in enumerate(records):
            assert (isinstance(record[0], int))
            assert (isinstance(record[1], tuple))
            assert (isinstance(record[1][0], int))
            assert (catalog_user_ids.numpy()[i].item() == record[0])
            if i > 5:
                break
    finally:
        if reader is not None:
            reader.close()
    
    md5sum_array_records = get_combined_md5(WriteTowerProducts.output_dir,
        r'.*recommended_movies.*\.array_record$')
    md5sum_parquets = get_combined_md5(WriteTowerProducts.output_dir,
        r'.*recommended_movies.*\.parquet')
    ##NOTE: to verify the md5sums fromcommand line, use:
    #    find . -type f -name "*.log" | sort | xargs cat | md5sum
    # write metadata
    
    metadata = {
        "num_records": n_ids,
        "num_recommendations": num_movies,
        "recommender_systems_git_hash": WriteTowerProducts.recommender_systems_git_hash,
        "md5sum_array_records" : md5sum_array_records,
        "md5sum_parquets" : md5sum_parquets
    }
    with open(outfile_metadata, "w") as f:
        json.dump(metadata, f)
    ## assert metadata file
    with open(outfile_metadata, "r") as f:
        metatadata_read = json.load(f)
        assert (metatadata_read == metadata)
    
    WriteTowerProducts.wrote_files.append(outfile)
    WriteTowerProducts.wrote_files.append(outfile2)
    WriteTowerProducts.wrote_files.append(pa_outfile)
    WriteTowerProducts.wrote_files.append(pa_outfile2)
    WriteTowerProducts.wrote_files.append(outfile_metadata)

def get_combined_md5(directory, pattern):
    hasher = hashlib.md5()
    # Find and sort files alphabetically for consistency
    files = sorted([p for p in Path(directory).rglob('*') if
        p.is_file() and re.match(pattern, str(p))])
    
    for filepath in files:
        with open(filepath, 'rb') as f:
            # Read in chunks to handle large files memory-efficiently
            for chunk in iter(lambda: f.read(65536), b''):
                hasher.update(chunk)
    
    return hasher.hexdigest()

def get_largest_subdir_version(parent_dir: str) -> str:
    parent_dir = Path(parent_dir)
    numeric_dirs = [d for d in parent_dir.iterdir() if
        d.is_dir() and d.name.isdigit()]
    if numeric_dirs:
        largest_dir = max(numeric_dirs, key=lambda d: int(d.name))
        return largest_dir.name
    raise ValueError("No numeric subdirectories found.")

if __name__ == '__main__':
    unittest.main()

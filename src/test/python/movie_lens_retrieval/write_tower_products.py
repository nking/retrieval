"""
scripts to use the trained models from the recommender_systems project to
write user embeddings, movie embeddings and recommended movies.
These tower products are used in training and inference in the ranker project.

example usage from CLI:

export PIPELINE_DATA_DIR="../TMP10/bin/rs_pipeline"
export OUTPUT_BASE_DIR="../TMP10"
python3 -m unittest src.test.python.movie_lens_retrieval.write_tower_products.WriteTowerProducts

NOTE: nothing in here uses the src/test/resources/data embeddings or recommendations.
"""
import os
import sys

is_cli = __name__ == '__main__' or any('unittest' in arg for arg in sys.argv)
if is_cli:
    print('** running from CLI **', flush=True)
    sys.path.insert(0,
        os.path.join(os.getcwd(), "src/test/python/movie_lens_retrieval"))
    sys.path.insert(0,
        os.path.join(os.getcwd(), "src/main/python/movie_lens_retrieval"))

from find_pushed_trainer_uri import get_pushed_trainer_uri

from movie_lens_retrieval.MovieData import MovieData
from movie_lens_retrieval.UserData import UserData

from array_record.python import array_record_module
import pyarrow as pa
import pyarrow.parquet as pq

import json
import shutil
import unittest
from typing import Any, Union, Tuple, Dict, List
import polars as pl
import msgpack
import tensorflow as tf
import numpy as np
import hashlib
from pathlib import Path
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# 0 = all logs, 1 = no info, 2 = no info/warn, 3 = no info/warn/error
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="umap")

from helper import get_project_dir, get_bin_dir
from movie_lens_retrieval.ANNSearcher import *
from scann.scann_ops.py.scann_ops_pybind import ScannSearcher

class WriteTowerProducts(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.AnnSearchType = Index
        cls.default_top_k = 100
        
        pipeline_dir = os.environ.get("PIPELINE_DATA_DIR", None)
        pusher_dir = os.environ.get("PUSHER_URI", None)
        saved_model_uri = None
        
        if pipeline_dir or pusher_dir:
            if not pusher_dir:
                pushed_uris = get_pushed_trainer_uri()
                if pushed_uris and pushed_uris[0]:
                    saved_model_uri = pushed_uris[0]
                else:
                    logger.warning(
                        "Could not resolve pushed trainer URI from MLMD metadata. Falling back to default.")
            else:
                saved_model_uri = pusher_dir
        
        if not saved_model_uri or not os.path.exists(saved_model_uri):
            saved_model_uri = os.path.join(get_project_dir(),
                "src/main/resources/serving_models", "user_movie_model")
        
        if not os.path.exists(saved_model_uri):
            raise FileNotFoundError(
                f'Critical Error: Saved model directory does not exist: {saved_model_uri}')
        
        output_base_dir = os.environ.get("OUTPUT_BASE_DIR", get_bin_dir())
        if not output_base_dir:
            output_base_dir = os.getcwd()
        
        cls.output_dir = os.path.join(output_base_dir, "tower_products")
        try:
            shutil.rmtree(cls.output_dir, ignore_errors=True)
            os.makedirs(cls.output_dir, exist_ok=True)
        except OSError as e:
            raise OSError(
                f"Failed to create or clean output directory {cls.output_dir}: {e}")
        
        try:
            cls.loaded_user_movie_model = tf.saved_model.load(saved_model_uri)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load TensorFlow SavedModel from {saved_model_uri}: {e}")
        
        test_res_dir = os.path.join(get_project_dir(),
            "src/test/resources/data")
        cls.users_path = os.path.join(test_res_dir, "users/users.parquet")
        cls.movies_path = os.path.join(test_res_dir, "movies/movies.parquet")
        
        for path_check, name in [(cls.users_path, "Users Parquet"),
            (cls.movies_path, "Movies Parquet")]:
            if not os.path.exists(path_check):
                raise FileNotFoundError(
                    f"Required {name} not found at: {path_check}")
        
        try:
            cls.model_dict = read_model_assets_hparams(saved_model_uri)
        except Exception as e:
            raise RuntimeError(f"Failed to read model hyperparameters: {e}")
        
        cls.MOVIE_OFFSET: int = cls.model_dict['n_users'] + 1
        try:
            embed_dim = json.loads(cls.model_dict['layer_sizes'])[-1]
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            raise ValueError(
                f"Invalid 'layer_sizes' format in hyperparameters: {e}")
        
        cls.user_id_range_incl = [1, cls.model_dict['n_users']]
        cls.movie_id_range_incl = [cls.MOVIE_OFFSET,
            cls.MOVIE_OFFSET + cls.model_dict['n_movies']]
        cls.recommender_systems_git_hash = cls.model_dict.get("git_hash",
            "unknown")
        
        cls.ratings_dict = {
            "full_history": [
                os.path.join(test_res_dir, "ratings_train.array_record"),
                os.path.join(test_res_dir, "ratings_val.array_record")
            ],
            "positive_history": [
                os.path.join(test_res_dir, "ratings_train_liked.array_record"),
                os.path.join(test_res_dir, "ratings_val_liked.array_record")
            ],
            "positive_train": os.path.join(test_res_dir,
                "ratings_train_liked.array_record"),
            "positive_val": os.path.join(test_res_dir,
                "ratings_val_liked.array_record"),
            "positive_test": os.path.join(test_res_dir,
                "ratings_test_liked.array_record"),
            "train": os.path.join(test_res_dir, "ratings_train.array_record"),
            "val": os.path.join(test_res_dir, "ratings_val.array_record"),
            "test": os.path.join(test_res_dir, "ratings_test.array_record"),
        }
        
        for k, p in cls.ratings_dict.items():
            if isinstance(p, list):
                for sub_p in p:
                    if not os.path.exists(sub_p):
                        logger.warning(
                            f"Ratings file [{k}] list member not found: {sub_p}")
            elif not os.path.exists(p):
                logger.warning(f"Ratings file [{k}] not found: {p}")
        
        cls.user_data = UserData(cls.users_path)
        cls.movie_data = MovieData(cls.movies_path, cls.MOVIE_OFFSET)
        
        if os.path.exists(cls.ratings_dict['positive_train']):
            cls.train_history_df = read_ratings_to_df(
                cls.ratings_dict['positive_train'])
        else:
            cls.train_history_df = pl.DataFrame(
                schema={"user_id": pl.Int64, "movie_id": pl.Int64,
                    "rating": pl.Int64, "timestamp": pl.Int64})
            logger.warning(
                "positive_train rating file missing; initialized empty train_history_df.")
        
        cls.wrote_files = []
    
    @classmethod
    def tearDownClass(cls):
        for outfile in cls.wrote_files:
            if os.path.exists(outfile):
                logger.info(f'wrote to {outfile}')
    
    def test_write(self):
        try:
            catalog_movie_ids, movie_catalog_embeddings = create_movie_catalog_embeddings()
            write_embeddings("movie_emb", catalog_movie_ids,
                movie_catalog_embeddings, key="movie")
            
            catalog_user_ids, timestamps, user_catalog_embeddings = create_user_catalog_embeddings()
            write_embeddings("user_emb", catalog_user_ids,
                user_catalog_embeddings, key="user")
            
            n_search = int(tf.shape(movie_catalog_embeddings)[0])
            if n_search == 0:
                raise ValueError(
                    "Movie catalog embeddings tensor is empty. Cannot build searcher.")
            
            MOVIE_OFFSET = int(tf.shape(catalog_user_ids)[0]) + 1
            ann_searcher = self.build_searcher(movie_catalog_embeddings, n_search, normalize=False)
            
            neighbors, distances = self.search_batch(ann_searcher, user_catalog_embeddings, n_search)
            neighbors += MOVIE_OFFSET
            
            write_recommendations(catalog_user_ids, neighbors, distances)
        except Exception as e:
            self.fail(f"test_write failed with exception: {e}")
    
    @staticmethod
    def build_searcher(embeddings: Union[tf.Tensor, np.ndarray], top_k: int, normalize: bool = False):
        if issubclass(WriteTowerProducts.AnnSearchType, Index):
            if isinstance(embeddings, tf.Tensor):
                embeddings = embeddings.numpy()
            return build_usearch_searcher(embeddings, top_k, normalize)
        else:
            return build_scann_searcher(embeddings, top_k, normalize)
    
    def search_batch(self, indexer: Union[Index, ScannSearcher],
            queries: Union[tf.Tensor, np.ndarray], final_num_neighbors: int):
        if isinstance(indexer, Index):
            if isinstance(queries, tf.Tensor):
                queries = queries.numpy()
            matches = indexer.search(vectors=queries, count=final_num_neighbors)
            return matches.keys, matches.distances
        else:
            return indexer.search_batched(queries=queries,
                final_num_neighbors=final_num_neighbors)

def write_embeddings(outfile_root_name, ids: tf.Tensor, embeddings: tf.Tensor,
        key: str = "user"):
    embed_len = int(tf.shape(embeddings)[1])
    n_ids = int(tf.shape(ids)[0])
    key_id = f"{key}_id"
    
    if n_ids == 0:
        raise ValueError(
            f"Cannot write embeddings for {outfile_root_name}: IDs tensor is empty.")
    
    outfile_metadata = os.path.join(WriteTowerProducts.output_dir,
        f"{outfile_root_name}_metadata.json")
    outfile_array_record = os.path.join(WriteTowerProducts.output_dir,
        f"{outfile_root_name}-00000-of-00001.array_record")
    outfile_parquet = os.path.join(WriteTowerProducts.output_dir,
        f"{outfile_root_name}-00000-of-00001.parquet")
    outfile_tfrecord = os.path.join(WriteTowerProducts.output_dir,
        f"{outfile_root_name}-00000-of-00001.tfrecord.gz")
    
    pa_schema = pa.schema([
        pa.field("id", pa.int64()),
        pa.field("embedding", pa.list_(pa.float32(), embed_len)),
    ])
    
    options = tf.io.TFRecordOptions(compression_type='GZIP')
    
    writer_a = None
    writer_p = None
    writer_t = None
    try:
        writer_a = array_record_module.ArrayRecordWriter(outfile_array_record,
            'group_size:1')
        writer_p = pq.ParquetWriter(outfile_parquet, pa_schema)
        writer_t = tf.io.TFRecordWriter(outfile_tfrecord, options=options)
        
        for (uid, embedding) in zip(ids.numpy(), embeddings.numpy()):
            uid_val = int(uid.item())
            emb_list = [float(x) for x in embedding.tolist()]
            
            writer_a.write(msgpack.packb((uid_val, emb_list)))
            
            pa_batch = pa.RecordBatch.from_arrays([
                pa.array([uid_val], type=pa.int64()),
                pa.array([emb_list], type=pa.list_(pa.float32(), embed_len))
            ], schema=pa_schema)
            writer_p.write_batch(pa_batch)
            
            feature = {
                key_id: tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[uid_val])),
                'embedding': tf.train.Feature(
                    float_list=tf.train.FloatList(value=emb_list))
            }
            example_proto = tf.train.Example(
                features=tf.train.Features(feature=feature))
            writer_t.write(example_proto.SerializeToString())
    except Exception as e:
        raise RuntimeError(
            f"Error writing embedding files for {outfile_root_name}: {e}")
    finally:
        if writer_a is not None:
            writer_a.close()
        if writer_p is not None:
            writer_p.close()
        if writer_t is not None:
            writer_t.close()
    
    # Spot check array record reader
    reader = None
    try:
        reader = array_record_module.ArrayRecordReader(outfile_array_record)
        count = reader.num_records()
        if count > 0:
            sample_indices = list(range(0, min(count, 5)))
            batch_bytes = reader.read(sample_indices)
            records = [msgpack.unpackb(b, use_list=True) for b in batch_bytes]
            assert count == n_ids, f"Record count mismatch: expected {n_ids}, got {count}"
            assert len(records) == len(sample_indices)
    except Exception as e:
        raise AssertionError(
            f"ArrayRecord verification failed for {outfile_root_name}: {e}")
    finally:
        if reader is not None:
            reader.close()
    
    # Assert parquet file
    try:
        table = pq.read_table(outfile_parquet)
        actual_data = table.to_pylist()
        assert len(
            actual_data) == n_ids, f"Parquet row count mismatch: expected {n_ids}, got {len(actual_data)}"
        assert isinstance(actual_data[0]['id'], int)
        assert isinstance(actual_data[0]['embedding'], list)
    except Exception as e:
        raise AssertionError(
            f"Parquet verification failed for {outfile_root_name}: {e}")
    
    # Assert tfrecord file
    try:
        feature_description = {
            key_id: tf.io.FixedLenFeature([], tf.int64),
            'embedding': tf.io.VarLenFeature(tf.float32)
        }
        dataset = tf.data.TFRecordDataset(outfile_tfrecord,
            compression_type="GZIP")
        for raw_record in dataset.take(5):
            parsed = tf.io.parse_single_example(raw_record,
                feature_description)
            parsed_id = parsed[key_id].numpy().item()
            parsed_emb = tf.sparse.to_dense(
                parsed['embedding']).numpy().tolist()
            assert isinstance(parsed_id, int)
            assert isinstance(parsed_emb[0], float)
    except Exception as e:
        raise AssertionError(
            f"TFRecord verification failed for {outfile_root_name}: {e}")
    
    md5sum_array_records = get_combined_md5(WriteTowerProducts.output_dir,
        fr'.*{outfile_root_name}\.array_record$')
    md5sum_parquets = get_combined_md5(WriteTowerProducts.output_dir,
        fr'.*{outfile_root_name}\.parquet')
    
    metadata = {
        "num_records": n_ids,
        "embed_dim": embed_len,
        "recommender_systems_git_hash": WriteTowerProducts.recommender_systems_git_hash,
        "md5sum_array_records": md5sum_array_records,
        "md5sum_parquets": md5sum_parquets
    }
    
    try:
        with open(outfile_metadata, "w", encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        with open(outfile_metadata, "r", encoding='utf-8') as f:
            metadata_read = json.load(f)
            assert metadata_read == metadata
    except Exception as e:
        raise IOError(
            f"Failed to write or verify metadata file {outfile_metadata}: {e}")
    
    WriteTowerProducts.wrote_files.extend(
        [outfile_metadata, outfile_array_record, outfile_parquet,
            outfile_tfrecord])


def read_model_assets_hparams(a_user_movie_models_dir) -> Dict[str, Any]:
    file_path = os.path.join(a_user_movie_models_dir,
        "assets.extra/hyperparameters.json")
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Hyperparameters file not found at: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if 'values' not in data:
            raise KeyError(
                f"Expected 'values' key in hyperparameters JSON at {file_path}")
        return data['values']


def create_movie_catalog_embeddings() -> Tuple[tf.Tensor, tf.Tensor]:
    if not os.path.exists(WriteTowerProducts.movies_path):
        raise FileNotFoundError(
            f"Movies parquet not found at: {WriteTowerProducts.movies_path}")
    
    df = pl.read_parquet(WriteTowerProducts.movies_path)
    if 'movie_id' not in df.columns:
        raise KeyError("Column 'movie_id' missing from movies DataFrame.")
    
    df = df.sort('movie_id')
    movie_ids = np.expand_dims(df['movie_id'].to_numpy(), axis=1)
    movie_ids_tf = tf.constant(movie_ids, name='movie_id', dtype=tf.int64)
    
    inputs = WriteTowerProducts.movie_data.get_movie(movie_ids_tf)
    
    signature_key = "serving_candidate_dict"
    if signature_key not in WriteTowerProducts.loaded_user_movie_model.signatures:
        raise KeyError(
            f"Signature '{signature_key}' not found in loaded TF model signatures.")
    
    infer_for_dict = WriteTowerProducts.loaded_user_movie_model.signatures[
        signature_key]
    output_outputs = list(infer_for_dict.structured_outputs.keys())
    if not output_outputs:
        raise ValueError(
            f"No structured outputs found for signature '{signature_key}'.")
    output_keyword = output_outputs[0]
    
    try:
        embeddings_list = infer_for_dict(
            movie_id=inputs['movie_id'],
            genres=inputs['genres'])
        embeddings: tf.Tensor = embeddings_list[output_keyword]
    except Exception as e:
        raise RuntimeError(f"Error during candidate signature inference: {e}")
    
    return movie_ids_tf, embeddings


def create_user_catalog_embeddings() -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    if not os.path.exists(WriteTowerProducts.users_path):
        raise FileNotFoundError(
            f"Users parquet not found at: {WriteTowerProducts.users_path}")
    
    if not WriteTowerProducts.train_history_df.is_empty() and "timestamp" in WriteTowerProducts.train_history_df.columns:
        mean_timestamp = WriteTowerProducts.train_history_df.select(
            pl.col("timestamp").mean()
        ).item()
        if mean_timestamp is None:
            mean_timestamp = 0
    else:
        mean_timestamp = 0
    
    users_df = pl.read_parquet(WriteTowerProducts.users_path)
    if 'user_id' not in users_df.columns:
        raise KeyError("Column 'user_id' missing from users DataFrame.")
    users_df = users_df.select(pl.col("user_id"))
    
    if not WriteTowerProducts.train_history_df.is_empty():
        users_2_df = (
            WriteTowerProducts.train_history_df
            .group_by("user_id")
            .agg(pl.col("timestamp").min().alias("timestamp"))
        )
        users_2_df = (
            users_df.join(users_2_df, on="user_id", how="left")
            .with_columns(pl.col("timestamp").fill_null(mean_timestamp))
        )
    else:
        users_2_df = users_df.with_columns(
            pl.lit(mean_timestamp).alias("timestamp"))
    
    users_2_df = users_2_df.sort('user_id')
    
    user_ids = np.expand_dims(users_2_df['user_id'].to_numpy(), axis=1)
    user_ids_tf = tf.constant(user_ids, name='user_id', dtype=tf.int64)
    
    timestamps = np.expand_dims(users_2_df['timestamp'].to_numpy(), axis=1)
    timestamps_tf = tf.constant(timestamps, name='timestamp', dtype=tf.int64)
    
    inputs = WriteTowerProducts.user_data.get_user(user_ids_tf, timestamps_tf)
    
    signature_key = "serving_query_dict"
    if signature_key not in WriteTowerProducts.loaded_user_movie_model.signatures:
        raise KeyError(
            f"Signature '{signature_key}' not found in loaded TF model signatures.")
    
    infer_for_dict = WriteTowerProducts.loaded_user_movie_model.signatures[
        signature_key]
    output_outputs = list(infer_for_dict.structured_outputs.keys())
    if not output_outputs:
        raise ValueError(
            f"No structured outputs found for signature '{signature_key}'.")
    output_keyword = output_outputs[0]
    
    try:
        embeddings_list = infer_for_dict(
            age=inputs['age'],
            gender=inputs['gender'],
            occupation=inputs['occupation'],
            timestamp=inputs['timestamp'],
            user_id=inputs['user_id']
        )
        embeddings: tf.Tensor = embeddings_list[output_keyword]
    except Exception as e:
        raise RuntimeError(f"Error during query signature inference: {e}")
    
    return user_ids_tf, timestamps_tf, embeddings


def read_ratings_to_df(file_path: str, batch_size: int = 2048) -> pl.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'Ratings file not found: {file_path}')
    records = []
    reader = None
    try:
        reader = array_record_module.ArrayRecordReader(file_path)
        n = reader.num_records()
        for i in range(0, n, batch_size):
            i_end = min(i + batch_size, n)
            batch_indices = list(range(i, i_end))
            batch_bytes = reader.read(batch_indices)
            data = [msgpack.unpackb(b, use_list=False) for b in batch_bytes]
            for record in data:
                if len(record) >= 4:
                    records.append({
                        'user_id': int(record[0]),
                        'movie_id': int(record[1]),
                        'rating': int(record[2]),
                        'timestamp': int(record[3])
                    })
    except Exception as e:
        raise RuntimeError(
            f"Error reading array record ratings from {file_path}: {e}")
    finally:
        if reader is not None:
            reader.close()
    return pl.DataFrame(records)


def write_recommendations(catalog_user_ids, neighbors, distances):
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
    
    history_dict = {}
    if not WriteTowerProducts.train_history_df.is_empty():
        data = WriteTowerProducts.train_history_df.select(
            ["user_id", "movie_id", "timestamp"]
        ).to_dict(as_series=False)
        for u, m, t in zip(data["user_id"], data["movie_id"],
                data["timestamp"]):
            if u not in history_dict:
                history_dict[u] = {}
            history_dict[u][m] = t
    
    writer = None
    writer2 = None
    pa_movie_writer = None
    pa_timestamp_writer = None
    
    try:
        writer = array_record_module.ArrayRecordWriter(outfile, 'group_size:1')
        writer2 = array_record_module.ArrayRecordWriter(outfile2,
            'group_size:1')
        
        pa_movie_writer = pq.ParquetWriter(pa_outfile, movies_schema)
        pa_timestamp_writer = pq.ParquetWriter(pa_outfile2, ts_schema)
        
        for user_id, movie_ids in zip(catalog_user_ids.numpy(), neighbors):
            u_id_val = int(user_id[0].item())
            m_ids_list = [int(x) for x in movie_ids.tolist()]
            assert num_movies == len(m_ids_list)
            
            writer.write(msgpack.packb((u_id_val, m_ids_list)))
            
            movie_batch = pa.RecordBatch.from_arrays([
                pa.array([u_id_val], type=pa.int32()),
                pa.array([m_ids_list], type=pa.list_(pa.int32(), num_movies))
            ], schema=movies_schema)
            pa_movie_writer.write_batch(movie_batch)
            
            timestamps = []
            for movie_id in m_ids_list:
                if u_id_val in history_dict and movie_id in history_dict[
                    u_id_val]:
                    timestamps.append(history_dict[u_id_val][movie_id])
                else:
                    timestamps.append(ts_2050)
            
            writer2.write(msgpack.packb((u_id_val, timestamps)))
            
            ts_batch = pa.RecordBatch.from_arrays([
                pa.array([u_id_val], type=pa.int32()),
                pa.array([timestamps], type=pa.list_(pa.int64(), num_movies))
            ], schema=ts_schema)
            pa_timestamp_writer.write_batch(ts_batch)
    except Exception as e:
        raise RuntimeError(f"Error writing recommendations: {e}")
    finally:
        if writer is not None:
            writer.close()
        if writer2 is not None:
            writer2.close()
        if pa_movie_writer is not None:
            pa_movie_writer.close()
        if pa_timestamp_writer is not None:
            pa_timestamp_writer.close()
    
    # Spot-check reader validation
    for target_file in [outfile, outfile2]:
        reader = None
        try:
            reader = array_record_module.ArrayRecordReader(target_file)
            count = reader.num_records()
            assert count == n_ids, f"Count mismatch for {target_file}: expected {n_ids}, got {count}"
        except Exception as e:
            raise AssertionError(
                f"Recommendation verification failed on {target_file}: {e}")
        finally:
            if reader is not None:
                reader.close()
    
    md5sum_array_records = get_combined_md5(WriteTowerProducts.output_dir,
        r'.*recommended_movies.*\.array_record$')
    md5sum_parquets = get_combined_md5(WriteTowerProducts.output_dir,
        r'.*recommended_movies.*\.parquet')
    
    metadata = {
        "num_records": n_ids,
        "num_recommendations": num_movies,
        "recommender_systems_git_hash": WriteTowerProducts.recommender_systems_git_hash,
        "md5sum_array_records": md5sum_array_records,
        "md5sum_parquets": md5sum_parquets
    }
    
    try:
        with open(outfile_metadata, "w", encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        with open(outfile_metadata, "r", encoding='utf-8') as f:
            metadata_read = json.load(f)
            assert metadata_read == metadata
    except Exception as e:
        raise IOError(
            f"Failed to write or verify recommendation metadata: {e}")
    
    WriteTowerProducts.wrote_files.extend(
        [outfile, outfile2, pa_outfile, pa_outfile2, outfile_metadata])


def get_combined_md5(directory, pattern):
    hasher = hashlib.md5()
    if not os.path.exists(directory):
        return ""
    files = sorted([p for p in Path(directory).rglob('*') if
        p.is_file() and re.match(pattern, str(p))])
    for filepath in files:
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b''):
                hasher.update(chunk)
    return hasher.hexdigest()


def get_largest_subdir_version(parent_dir: str) -> str:
    parent_dir = Path(parent_dir)
    if not parent_dir.exists():
        raise FileNotFoundError(f"Parent directory not found: {parent_dir}")
    numeric_dirs = [d for d in parent_dir.iterdir() if
        d.is_dir() and d.name.isdigit()]
    if numeric_dirs:
        largest_dir = max(numeric_dirs, key=lambda d: int(d.name))
        return largest_dir.name
    raise ValueError("No numeric subdirectories found.")


if __name__ == '__main__':
    unittest.main()
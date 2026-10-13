import os.path
import time
import unittest
import numpy as np
import tensorflow as tf

from helper import *
from movie_lens_retrieval.ANNSearcher import *

class TestAnnSearcher(unittest.TestCase):
    def setUp(self):
        pass
    
    def test_recall(self):
        # Simulate Two-Tower embeddings (e.g., 100,000 items, 64 dimensions)
        num_items = 100_000
        embedding_dim = 32
        
        raw_embeddings = np.random.randn(num_items, embedding_dim).astype(
            np.float32)
        tf_embeddings = tf.convert_to_tensor(raw_embeddings)
        
        for top_k in (1, 20, 50, 100):
        
            # Build ScaNN Index (TensorFlow)
            scann_index = build_scann_searcher(tf_embeddings, top_k=top_k)
            
            # Query ScaNN
            query_vector = tf.math.l2_normalize(tf_embeddings[:5], axis=-1)
            #np.ndarrays of shape (n_queries, top_k)
            scann_ids, scann_distances = scann_index.search_batched(query_vector,
                final_num_neighbors=top_k)
            
            # Build USearch Index (NumPy)
            usearch_index = build_usearch_searcher(raw_embeddings, top_k=top_k)
            
            # Query USearch
            query_np = raw_embeddings[:5]
            query_np = query_np / np.linalg.norm(query_np, axis=1, keepdims=True)
            
            # Parallel multi-query search
            matches = usearch_index.search(query_np, top_k, threads=0)
            usearch_ids = matches.keys
            # np.ndarrays of shape (n_queries, top_k)
            usearch_distances = matches.distances
            
            self.evaluate_ground_truth_recall(query_np,
                raw_embeddings, scann_index, usearch_index, top_k=top_k)
         
    def test_latency(self):
        num_items = 100_000
        dim = 64
        top_k = 20
        num_queries = 1_000
        
        # Generate and normalize random embeddings
        items = np.random.randn(num_items, dim).astype(np.float32)
        items /= np.linalg.norm(items, axis=1, keepdims=True)
        
        queries = np.random.randn(num_queries, dim).astype(np.float32)
        queries /= np.linalg.norm(queries, axis=1, keepdims=True)
        
        items_tf = tf.convert_to_tensor(items)
        
        print("Building ScaNN index...")
        scann_idx = build_scann_searcher(items_tf, top_k=top_k)
        
        print("Building USearch index...")
        usearch_idx = build_usearch_searcher(items, top_k=top_k)
        
        self.benchmark_latency_and_qps(scann_idx, usearch_idx, queries, top_k=top_k)
        
        
    def evaluate_ground_truth_recall(self,
            query_vectors: np.ndarray,
            doc_embeddings: np.ndarray,
            scann_index,
            usearch_index,
            top_k: int = 50
    ):
        """
        Evaluates Recall@K and Score Ratio for ScaNN and USearch against Exact Brute-Force.
        """
        # 1. Compute Exact Ground Truth via Brute-Force Dot Product
        # Shape: (num_queries, num_docs)
        exact_scores = np.dot(query_vectors, doc_embeddings.T)
        
        scann_recalls, usearch_recalls = [], []
        scann_ratios, usearch_ratios = [], []
        
        for i, q in enumerate(query_vectors):
            # Ground Truth Top-K
            gt_indices = set(np.argsort(-exact_scores[i])[:top_k])
            gt_max_score = exact_scores[i, list(gt_indices)[0]]
            
            # ScaNN Search
            q_tf = tf.convert_to_tensor([q], dtype=tf.float32)
            s_ids, s_scores = scann_index.search_batched(q_tf,
                final_num_neighbors=top_k)
            s_ids = set(s_ids[0])
            
            # USearch Search
            u_matches = usearch_index.search(q, top_k)
            u_ids = set(u_matches.keys)
            
            # Calculate Recall@K against Exact Ground Truth
            scann_recalls.append(len(gt_indices & s_ids) / top_k)
            usearch_recalls.append(len(gt_indices & u_ids) / top_k)
        
        print(f"--- Benchmark Results (Top-K = {top_k}) ---")
        print(
            f"ScaNN Mean Recall@{top_k} vs Exact GT  : {np.mean(scann_recalls):.4f} ± {np.std(scann_recalls):.4f}")
        print(
            f"USearch Mean Recall@{top_k} vs Exact GT: {np.mean(usearch_recalls):.4f} ± {np.std(usearch_recalls):.4f}")
    
    def benchmark_latency_and_qps(self,
            scann_index,
            usearch_index,
            queries_np: np.ndarray,
            top_k: int = 20,
            warmup_runs: int = 50
    ):
        queries_tf = tf.convert_to_tensor(queries_np, dtype=tf.float32)
        num_queries = len(queries_np)
        
        print(f"--- Benchmark Setup ---")
        print(
            f"Dataset Items: Index built | Benchmark Queries: {num_queries} | Top-K: {top_k}\n")
        
        # --------------------------------------
        # A. ScaNN Benchmarks
        # --------------------------------------
        # Warmup
        for i in range(warmup_runs):
            _ = scann_index.search_batched(queries_tf[i:i + 1],
                final_num_neighbors=top_k)
        
        # 1. Single-Query Latency
        scann_latencies_ms = []
        for i in range(num_queries):
            q = queries_tf[i:i + 1]
            t0 = time.perf_counter()
            _ = scann_index.search_batched(q, final_num_neighbors=top_k)
            t1 = time.perf_counter()
            scann_latencies_ms.append((t1 - t0) * 1000.0)
        
        # 2. Batched QPS (Throughput)
        t0 = time.perf_counter()
        _ = scann_index.search_batched(queries_tf, final_num_neighbors=top_k)
        t1 = time.perf_counter()
        scann_batch_time = t1 - t0
        scann_qps = num_queries / scann_batch_time
        
        # --------------------------------------
        # B. USearch Benchmarks
        # --------------------------------------
        # Warmup
        for i in range(warmup_runs):
            _ = usearch_index.search(queries_np[i], top_k, threads=1)
        
        # 1. Single-Query Latency (Single Threaded)
        usearch_latencies_ms = []
        for i in range(num_queries):
            q = queries_np[i]
            t0 = time.perf_counter()
            _ = usearch_index.search(q, top_k, threads=1)
            t1 = time.perf_counter()
            usearch_latencies_ms.append((t1 - t0) * 1000.0)
        
        # 2. Batched QPS (Multi-Threaded parallel search across all cores)
        t0 = time.perf_counter()
        _ = usearch_index.search(queries_np, top_k, threads=0)
        t1 = time.perf_counter()
        usearch_batch_time = t1 - t0
        usearch_qps = num_queries / usearch_batch_time
        
        # --------------------------------------
        # C. Print Metrics
        # --------------------------------------
        print(
            "==========================================================================")
        print(
            f"{'Metric':<30} | {'ScaNN (IVF-AH)':<18} | {'USearch (HNSW f16)':<18}")
        print(
            "==========================================================================")
        print(
            f"{'Mean Latency (ms)':<30} | {np.mean(scann_latencies_ms):<18.4f} | {np.mean(usearch_latencies_ms):<18.4f}")
        print(
            f"{'P50 Latency (ms)':<30} | {np.percentile(scann_latencies_ms, 50):<18.4f} | {np.percentile(usearch_latencies_ms, 50):<18.4f}")
        print(
            f"{'P95 Latency (ms)':<30} | {np.percentile(scann_latencies_ms, 95):<18.4f} | {np.percentile(usearch_latencies_ms, 95):<18.4f}")
        print(
            f"{'P99 Latency (ms)':<30} | {np.percentile(scann_latencies_ms, 99):<18.4f} | {np.percentile(usearch_latencies_ms, 99):<18.4f}")
        print(
            "--------------------------------------------------------------------------")
        print(
            f"{'Batch Throughput (QPS)':<30} | {scann_qps:<18.1f} | {usearch_qps:<18.1f}")
        print(
            "==========================================================================")
    
    if __name__ == '__main__':
        unittest.main()

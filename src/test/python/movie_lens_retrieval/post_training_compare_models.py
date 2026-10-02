"""
scripts to analysis the trained models from the recommender_systems project.

some of the analysis is in rust in the ranker project directory inference_src and will be ported
to python here to have it all in one place
"""
import json
import os
import glob
import shutil
import subprocess
import argparse
from typing import Dict, Any, Tuple
from scipy import stats
import polars as pl

# 0 = all logs, 1 = no info, 2 = no info/warn, 3 = no info/warn/error
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="umap")

import sys
is_cli = __name__ == '__main__' or any('unittest' in arg for arg in sys.argv)
if is_cli:
    #print('** running from CLI **', flush=True)
    sys.path.insert(0,
        os.path.join(os.getcwd(), "src/test/python/movie_lens_retrieval"))
    sys.path.insert(0,
        os.path.join(os.getcwd(), "src/main/python/movie_lens_retrieval"))

'''
example usages from CLI
 Use either --analyze_and_compare or --compare
 
python3 src/test/python/movie_lens_retrieval/post_training_compare_models.py \
--analyze_and_compare \
--SAVED_MODEL_DIR_1=../TMP10/bin/rs_pipeline/Pusher/pushed_model \
--MODEL_VERSION_1="21" \
--OUTPUT_BASE_DIR_1="../TMP10" \
--SAVED_MODEL_DIR_2=../TMP11/bin/rs_pipeline/Pusher/pushed_model \
--MODEL_VERSION_2="21" \
--OUTPUT_BASE_DIR_2="../TMP11" \
--TAG_1="high-t-model" \
--TAG_2="low-t-model" \
--OUTPUT_COMPARE_DIR="../TMP10_TMP11_Compare" \
--quiet

python -m debugpy --listen 5678 --wait-for-client \
src/test/python/movie_lens_retrieval/post_training_compare_models.py \
--analyze_and_compare \
--SAVED_MODEL_DIR_1=../TMP10/bin/rs_pipeline/Pusher/pushed_model \
--MODEL_VERSION_1="21" \
--OUTPUT_BASE_DIR_1="../TMP10" \
--SAVED_MODEL_DIR_2=../TMP11/bin/rs_pipeline/Pusher/pushed_model \
--MODEL_VERSION_2="21" \
--OUTPUT_BASE_DIR_2="../TMP11" \
--TAG_1="high-t-model" \
--TAG_2="low-t-model" \
--OUTPUT_COMPARE_DIR="../TMP10_TMP11_Compare" \
--quiet

or just compare:

python3 src/test/python/movie_lens_retrieval/post_training_compare_models.py \
--compare \
--INPUT_BASE_DIR_1="../TMP10/post_training_analysis" \
--INPUT_BASE_DIR_2="../TMP11/post_training_analysis" \
--TAG_1="high-t-model" \
--TAG_2="low-t-model" \
--OUTPUT_COMPARE_DIR="../TMP10_TMP11_Compare"
'''

from helper import get_project_dir, get_bin_dir

def run_analysis(saved_model_dir: str, model_version: str, output_dir: str,
        quiet: bool = False) -> str:
    script_path = os.path.join(get_project_dir(),
        "src/test/python/movie_lens_retrieval/post_training_analysis.py")
    # export SAVED_MODEL_DIR="../TMP10/bin/rs_pipeline/Pusher/pushed_model"
    # export MODEL_VERSION="21"
    # export OUTPUT_BASE_DIR="../TMP10"
    # python3 -m unittest src.test.python.movie_lens_retrieval.post_training_analysis.TestAnalysis
    env = os.environ.copy()
    env["SAVED_MODEL_DIR"] = saved_model_dir
    env["MODEL_VERSION"] = model_version
    env["OUTPUT_BASE_DIR"] = output_dir
        
    stdout_setting = subprocess.DEVNULL if args.quiet else None
    
    result = subprocess.run(["python3", script_path], env=env,
        stdout=stdout_setting)
    if result.returncode == 0:
        print("model analyzed")
    else:
        print("model analysis failed.")
    
    return os.path.join(output_dir, "post_training_analysis")

def welch_ttest_from_summary(
    mean1: float, std1: float, n1: int,
    mean2: float, std2: float, n2: int
) -> dict:
    """Calculates Welch's t-test given summary statistics from two runs."""
    t_stat, p_val = stats.ttest_ind_from_stats(
        mean1=mean1, std1=std1, nobs1=n1,
        mean2=mean2, std2=std2, nobs2=n2,
        equal_var=False  # Triggers Welch's t-test (unequal variances)
    )
    return {
        "t_statistic": t_stat,
        "p_value": p_val,
        "significant_005": p_val < 0.05
    }


import os
import polars as pl
import numpy as np
from scipy import stats
from typing import List, Dict, Any

def _compare_model_metrics(
        analysis_dir_1: str,
        analysis_dir_2: str,
        tag_1: str,
        tag_2: str,
        top_k: int = 100
) -> pl.DataFrame:
    """
    Loads user-level and item-level parquet metrics from two model directories,
    joins them on their respective IDs, and performs paired t-tests and
    Wilcoxon signed-rank tests to evaluate statistical significance.
    """
   
    results: List[Dict[str, Any]] = []
    
    def evaluate_metric(file_name: str, join_key: str, metric_col: str,
            display_name: str):
        path_1 = os.path.join(analysis_dir_1, file_name)
        path_2 = os.path.join(analysis_dir_2, file_name)
        
        if not (os.path.exists(path_1) and os.path.exists(path_2)):
            print(f"Skipping {display_name}: Missing file(s) in one or both directories.")
            return
        
        # Load only the join key and the specific metric column to minimize memory overhead
        df1 = pl.read_parquet(path_1).select([join_key, metric_col])
        df2 = pl.read_parquet(path_2).select([join_key, metric_col])
        
        # Rename metrics to prevent collisions and identify the model
        col_1 = f"{metric_col}_{tag_1}"
        col_2 = f"{metric_col}_{tag_2}"
        df1 = df1.rename({metric_col: col_1})
        df2 = df2.rename({metric_col: col_2})
        
        # Inner join to ensure we only compare perfectly paired data
        merged_df = df1.join(df2, on=join_key, how="inner").drop_nulls()
        
        arr_1 = merged_df[col_1].to_numpy()
        arr_2 = merged_df[col_2].to_numpy()
        
        if len(arr_1) == 0:
            print(f"Skipping {display_name}: No overlapping entities found.")
            return
        
        # Paired T-Test
        t_stat, t_pval = stats.ttest_rel(arr_1, arr_2)
        
        # Wilcoxon Signed-Rank Test
        try:
            # zero_method='wilcox' discards differences of zero.
            # 'pratt' is more robust if there are many zero-difference pairs (e.g., exact same recall)
            w_stat, w_pval = stats.wilcoxon(arr_1, arr_2, zero_method='pratt')
        except ValueError:
            # Triggers if all differences are exactly zero
            w_stat, w_pval = np.nan, np.nan
        
        results.append({
            "Metric": display_name,
            "N": len(arr_1),
            f"Mean ({tag_1})": np.mean(arr_1),
            f"Mean ({tag_2})": np.mean(arr_2),
            "T-Statistic": t_stat,
            "T-PValue": t_pval,
            "Wilcoxon-W": w_stat,
            "Wilcoxon-PValue": w_pval
        })
    
    # Stratified Metrics (Joined on user_id)
    # ['user_id', 'hits_at_k', 'dcg_at_k', 'user_tier', 'total_positives', 'idcg_at_k', 'recall_at_k',
    #    'precision_at_k', 'ndcg_at_k', 'expected_random_recall_at_k', 'expected_random_precision_at_k',
    #    'expected_random_ndcg_at_k']
    stratified_file = f"stratified_metrics_top_{top_k}.parquet"
    evaluate_metric(stratified_file, "user_id", "ndcg_at_k", f"NDCG@{top_k}")
    evaluate_metric(stratified_file, "user_id", "recall_at_k",
        f"Recall@{top_k}")
    evaluate_metric(stratified_file, "user_id", "precision_at_k",
        f"Precision@{top_k}")
    
    if top_k != 20:
        # Intra-List Diversity (Joined on user_id)
        #['user_id', 'intra_list_diversity']
        ild_file = f"user_ild_top_{top_k}.parquet"
        evaluate_metric(ild_file, "user_id", "intra_list_diversity",
            f"ILD@{top_k}")
        
        #inter_list_diviersity
        evaluate_metric(
            f"inter_list_diversity_all_users_top_{top_k}.parquet",
            "user_id", "inter_list_diversity",
            f"Inter-List Diversity (All Users)@{top_k}"
        )
        evaluate_metric(
            f"inter_list_diversity_cold_start_top_{top_k}.parquet",
            "user_id", "inter_list_diversity",
            f"Inter-List Diversity (Cold Start)@{top_k}"
        )
        
        for tier in [0, 1, 2]:
            evaluate_metric(
                f"inter_list_diversity_user_tier_{tier}_top_{top_k}.parquet",
                "user_id", "inter_list_diversity",
                f"Inter-List Diversity (User Tier {tier})@{top_k}"
            )
    
    # Popularity Bias (Joined on user_id)
    #['user_id', 'retrieved_log_pop', 'ground_truth_log_pop', 'delta_log_pop']
    for user_tier in [0, 1, 2]:
        pop_file = f"popularity_bias_user_tier_{user_tier}.parquet"
        evaluate_metric(pop_file, "user_id", "delta_log_pop",
            f"Delta Log Pop (User Tier {user_tier})")
    
    # Item Frequencies (Joined on movie_id)
    #['movie_id', 'movie_tier', 'retrieval_count']
    freq_file = f"item_frequencies_top_{top_k}.parquet"
    evaluate_metric(freq_file, "movie_id", "retrieval_count",
        f"Item Retrieval Count@{top_k}")
    
    # Return results as a formatted Polars DataFrame
    return pl.DataFrame(results)


def run_compare(analysis_dir_1: str, analysis_dir_2: str,
        output_compare_dir: str,
        tag_1: str, tag_2: str, num_catalog_movies: int,
        quiet: bool = False) -> str:
    shutil.rmtree(output_compare_dir, ignore_errors=True)
    os.makedirs(output_compare_dir, exist_ok=True)
    
    parquet_dir_1 = os.path.join(analysis_dir_1, "parquet_metrics")
    parquet_dir_2 = os.path.join(analysis_dir_2, "parquet_metrics")
    
    agg_res_metrics = dict()
    agg_res_conclusions = []
    
    # Updated to include Inter-List Diversity metadata
    metric_meaning_dict = {
        "NDCG@": {"meaning": "ranking relevant results at higher positions",
            "higher_is_better": True},
        "Recall@": {
            "meaning": "retrieving a high volume of relevant ground truth positives",
            "higher_is_better": True},
        "Precision@": {
            "meaning": "maintaining a high density of relevant results within the slate",
            "higher_is_better": True},
        "ILD@": {
            "meaning": "recommending a wide, semantically diverse range of movie genres",
            "higher_is_better": True},
        "Delta Log Pop (User Tier 0)": {
            "meaning": "surfacing niche, long-tail movies to mainstream users",
            "higher_is_better": False},
        "Delta Log Pop (User Tier 1)": {
            "meaning": "surfacing niche, long-tail movies to mid-tail users",
            "higher_is_better": False},
        "Delta Log Pop (User Tier 2)": {
            "meaning": "surfacing niche, long-tail movies to niche users",
            "higher_is_better": False},
        "Item Retrieval Count@": {
            "meaning": "distributing recommendation slots differently across the catalog",
            "higher_is_better": None},
        "Inter-List Diversity (All Users)@": {
            "meaning": "providing personalized, unique recommendation slates across the general user base",
            "higher_is_better": True},
        "Inter-List Diversity (Cold Start)@": {
            "meaning": "maintaining slate personalization when surfacing cold-start items",
            "higher_is_better": True},
        "Inter-List Diversity (User Tier 0)@": {
            "meaning": "providing highly individualized slates to mainstream users",
            "higher_is_better": True},
        "Inter-List Diversity (User Tier 1)@": {
            "meaning": "providing highly individualized slates to mid-tail users",
            "higher_is_better": True},
        "Inter-List Diversity (User Tier 2)@": {
            "meaning": "providing highly individualized slates to niche users",
            "higher_is_better": True}
    }
    
    def get_metric_meta(metric: str) -> Dict[str, Any]:
        for key, meta in metric_meaning_dict.items():
            if metric.startswith(key):
                return meta
        return {"meaning": "affecting an unknown metric",
            "higher_is_better": True}
    
    # Tracking for overall trends
    trend_tracker = {"accuracy_wins": 0, "accuracy_losses": 0,
        "diversity_wins": 0, "diversity_losses": 0}
    
    for top_k in (20, 100):
        df_results = _compare_model_metrics(
            analysis_dir_1=parquet_dir_1, analysis_dir_2=parquet_dir_2,
            tag_1=tag_1, tag_2=tag_2, top_k=top_k
        )
        
        if not quiet:
            print(df_results)
        
        key_col = "Metric"
        res = {
            row[key_col]: {k: v for k, v in row.items() if k != key_col}
            for row in df_results.to_dicts()
        }
        agg_res_metrics.update(res)
        
        conclusions = [f"\n--- Top {top_k} Analysis ---"]
        
        for metric, stats_dict in res.items():
            meta = get_metric_meta(metric)
            meaning = meta["meaning"]
            
            # Handle potential key differences from earlier script outputs
            c1 = stats_dict.get("T-PValue", 1.0)
            c2 = stats_dict.get("Wilcoxon-PVal",
                stats_dict.get("Wilcoxon-PValue", 1.0))
            
            mean_1 = stats_dict.get(f"Mean ({tag_1})", 0.0)
            mean_2 = stats_dict.get(f"Mean ({tag_2})", 0.0)
            
            is_significant = (c1 < 0.05) and (c2 < 0.05)
            
            if is_significant:
                if meta["higher_is_better"] is None:
                    # Special case for Item Retrieval Count variance
                    conclusions.append(
                        f"[*] {tag_1} is significantly {meaning} compared to {tag_2} (P < 0.05).")
                else:
                    tag_1_is_better = (mean_1 > mean_2) if meta[
                        "higher_is_better"] else (mean_1 < mean_2)
                    
                    if tag_1_is_better:
                        conclusions.append(
                            f"[+] {tag_1} is significantly BETTER than {tag_2} at {meaning}.")
                        if "NDCG" in metric or "Recall" in metric or "Precision" in metric:
                            trend_tracker["accuracy_wins"] += 1
                        # Track diversity wins for both ILD and the new Inter-List Diversity
                        if "ILD" in metric or "Delta" in metric or "Inter-List Diversity" in metric:
                            trend_tracker["diversity_wins"] += 1
                    else:
                        conclusions.append(
                            f"[-] {tag_1} is significantly WORSE than {tag_2} at {meaning}.")
                        if "NDCG" in metric or "Recall" in metric or "Precision" in metric:
                            trend_tracker["accuracy_losses"] += 1
                        # Track diversity losses for both ILD and the new Inter-List Diversity
                        if "ILD" in metric or "Delta" in metric or "Inter-List Diversity" in metric:
                            trend_tracker["diversity_losses"] += 1
            else:
                conclusions.append(
                    f"[=] {tag_1} is not significantly different from {tag_2} at {meaning}.")
        
        agg_res_conclusions.extend(conclusions)
        
        # PAIRED BOOTSTRAP RESAMPLING FOR GLOBAL METRICS
        bootstrap_results, bootstrap_conclusions = _bootstrap_global_metrics(
            parquet_dir_1, parquet_dir_2, tag_1, tag_2, top_k,
            num_catalog_movies
        )
        agg_res_metrics.update(bootstrap_results)
        agg_res_conclusions.extend(bootstrap_conclusions)
    
    # ADD OVERALL TRENDS CONCLUSIONS
    agg_res_conclusions.append("\n=== OVERALL MACRO TRENDS ===")
    
    if trend_tracker["diversity_wins"] > 0 and trend_tracker[
        "accuracy_losses"] == 0:
        agg_res_conclusions.append(
            f"-> PARETO IMPROVEMENT: {tag_1} significantly improved diversity and exploration without sacrificing ranking accuracy.")
    elif trend_tracker["diversity_wins"] > 0 and trend_tracker[
        "accuracy_losses"] > 0:
        agg_res_conclusions.append(
            f"-> TRADE-OFF: {tag_1} increases diversity and long-tail exploration, but at the cost of top-tier precision/accuracy.")
    elif trend_tracker["accuracy_wins"] > 0 and trend_tracker[
        "diversity_losses"] == 0:
        agg_res_conclusions.append(
            f"-> EXPLOITATION WIN: {tag_1} improved core accuracy/ranking metrics without harming existing catalog diversity.")
    elif trend_tracker["accuracy_losses"] == 0 and trend_tracker[
        "diversity_losses"] == 0 and trend_tracker["accuracy_wins"] == 0 and \
            trend_tracker["diversity_wins"] == 0:
        agg_res_conclusions.append(
            f"-> STATIC: No globally significant behavioral shifts were detected between {tag_1} and {tag_2}.")
    else:
        agg_res_conclusions.append(
            f"-> MIXED SHIFT: {tag_1} exhibits complex shifts. See detailed metric logs for tier-specific degradation or gains.")
    
    if not quiet:
        for c in agg_res_conclusions:
            print(c)
    
    with open(os.path.join(output_compare_dir, "summary_compare_metrics.json"),
            "w") as f:
        json.dump(agg_res_metrics, f, indent=4)
    
    with open(os.path.join(output_compare_dir,
            "summary_compare_conclusions.json"), "w") as f:
        json.dump(agg_res_conclusions, f, indent=4)
    
    return output_compare_dir

    
def _bootstrap_global_metrics(
        dir_1: str, dir_2: str, tag_1: str, tag_2: str, top_k: int,
        num_catalog_movies: int, n_iterations: int = 1000
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Performs Paired Bootstrap Resampling on global catalog metrics.
    Samples users with replacement, reconstructs the global recommendation distribution,
    and calculates confidence intervals for the paired differences.
    """
    
    #['user_id', 'movie_id', 'movie_tier', 'user_tier']
    path_1 = os.path.join(dir_1, f"user_retrievals_top_{top_k}.parquet")
    path_2 = os.path.join(dir_2, f"user_retrievals_top_{top_k}.parquet")
    
    if not (os.path.exists(path_1) and os.path.exists(path_2)):
        return {}, [
            f"[*] Skipping Bootstrap @{top_k}: user_retrievals parquet missing."]
    
    df1 = pl.read_parquet(path_1).select(["user_id", "movie_id"])
    df2 = pl.read_parquet(path_2).select(["user_id", "movie_id"])
    
    # Extract overlapping users to ensure paired testing
    common_users = np.intersect1d(df1["user_id"].unique().to_numpy(),
        df2["user_id"].unique().to_numpy())
    n_users = len(common_users)
    
    if n_users == 0:
        return {}, []
    
    # Filter DataFrames to only common users
    df1 = df1.filter(pl.col("user_id").is_in(common_users))
    df2 = df2.filter(pl.col("user_id").is_in(common_users))
    
    # Helper: Group movies by user into a list of lists for fast random sampling
    def _build_user_item_matrix(df: pl.DataFrame) -> np.ndarray:
        # Sort by user_id to ensure indices align perfectly between df1 and df2
        grouped = df.sort("user_id").group_by("user_id",
            maintain_order=True).agg(pl.col("movie_id"))
        return grouped["movie_id"].to_numpy()
    
    matrix_1 = _build_user_item_matrix(df1)
    matrix_2 = _build_user_item_matrix(df2)
    
    # Internal Global Metric Calculators
    def calc_coverage(freqs):
        return np.count_nonzero(freqs) / num_catalog_movies
    
    def calc_gini(freqs):
        f = np.sort(freqs[freqs > 0])  # Filter zeros to analyze distribution among retrieved items
        if len(f) == 0: return 0
        cum_sum = np.cumsum(f)
        return (len(f) + 1 - 2 * np.sum(cum_sum) / cum_sum[-1]) / len(f)
    
    def calc_entropy(freqs):
        p = freqs[freqs > 0] / np.sum(freqs)
        return -np.sum(p * np.log(p))
    
    # Output storage
    metrics = {"Coverage": [], "Gini": [], "Uniformity_Entropy": []}
    diffs = {"Coverage": [], "Gini": [], "Uniformity_Entropy": []}
    
    # --- Bootstrap Loop ---
    for _ in range(n_iterations):
        # Sample users with replacement
        idx = np.random.randint(0, n_users, size=n_users)
        
        # Reconstruct global frequencies for this sample
        items_1 = np.concatenate(matrix_1[idx])
        items_2 = np.concatenate(matrix_2[idx])
        
        freqs_1 = np.bincount(items_1, minlength=num_catalog_movies)
        freqs_2 = np.bincount(items_2, minlength=num_catalog_movies)
        
        # Calculate paired metrics
        cov1, cov2 = calc_coverage(freqs_1), calc_coverage(freqs_2)
        gin1, gin2 = calc_gini(freqs_1), calc_gini(freqs_2)
        ent1, ent2 = calc_entropy(freqs_1), calc_entropy(freqs_2)
        
        diffs["Coverage"].append(cov1 - cov2)
        diffs["Gini"].append(gin1 - gin2)
        diffs["Uniformity_Entropy"].append(ent1 - ent2)
    
    # --- Statistical Summaries ---
    results_dict = {}
    conclusions = [
        f"\n--- Bootstrap Global Metrics @{top_k} ({n_iterations} samples) ---"]
    
    for metric_name, diff_array in diffs.items():
        diff_array = np.array(diff_array)
        mean_diff = np.mean(diff_array)
        ci_lower = np.percentile(diff_array, 2.5)
        ci_upper = np.percentile(diff_array, 97.5)
        
        # Empirical P-Value (two-tailed)
        p_val_empirical = min(np.mean(diff_array <= 0),
            np.mean(diff_array >= 0)) * 2
        is_sig = p_val_empirical < 0.05
        
        results_dict[f"Bootstrap_{metric_name}@{top_k}"] = {
            "Mean_Diff": mean_diff,
            "CI_2.5": ci_lower,
            "CI_97.5": ci_upper,
            "Empirical_P_Value": p_val_empirical
        }
        
        direction = "HIGHER" if mean_diff > 0 else "LOWER"
        sig_str = "SIGNIFICANTLY " if is_sig else "Not significantly "
        extra = ""
        if metric_name == "Gini":
            extra = "(lower means weaker predictive power and ability to rank)"
        elif metric_name == "Uniformity_Entropy":
            extra = "(higher means a more efficient use of latent space and has learned diverse, well-spread features)"
        conclusions.append(
            f"[{'*' if is_sig else '='}] {tag_1} has {sig_str}{direction} {metric_name} "
            f"than {tag_2} (P-Val: {p_val_empirical:.4f}, "
            f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]). {extra}")
    
    return results_dict, conclusions

def _get_summary_metrics_dictionary(analysis_dir_path:str):
    file_path = os.path.join(analysis_dir_path, "summary_metrics.json")
    with open(file_path, "r") as f:
        return json.load(f)
    
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
        description="Compare retrieval analysis of 2 models")
    parser.add_argument(
        "--analyze_and_compare",
        action="store_true",
        help="analyze and compare the retrieval from 2 bi-encoders."
             "Note that if set, requires SAVED_MODEL_DIR_1, MODEL_VERSION_1, OUTPUT_BASE_DIR_1 and the same with _2 endings for the 2nd model",
        required=False
    )
    for v in ("1", "2"):
        parser.add_argument(
            f"--SAVED_MODEL_DIR_{v}", type=str,
            help="path to saved_model directory.  path should not contain version.",
            required=False
        )
        parser.add_argument(
            f"--MODEL_VERSION_{v}", type=str,
            help=f"The version for SAVED_MODEL_DIR_{v}",
            required=False
        )
        parser.add_argument(
            f"--OUTPUT_BASE_DIR_{v}", type=str,
            help="the base directory to which a direcotry called post_training_analysis will be written",
            required=False
        )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="compare the retrieval analysis of 2 models"
             "Note that if set, requires INPUT_BASE_DIR_1 and INPUT_BASE_DIR_2 to be set and both must contain a file called 'summary_metrics.json'",
        required=False
    )
    parser.add_argument(
        f"--OUTPUT_COMPARE_DIR", type=str,
        help=f"path to directory to write output comparison files 'summary_compare_metrics.json, summary_compare_conclusions.json'",
        required=True
    )
    for v in ("1", "2"):
        parser.add_argument(
            f"--INPUT_BASE_DIR_{v}", type=str,
            help=f"path to model {v} directory containing a file called 'summary_metrics.json'",
            required=False
        )
        parser.add_argument(
            f"--TAG_{v}", type=str,
            help=f"short string used to identify model {v} in results",
            required=True
        )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress stdout from the test script"
    )
    parser.add_argument(
        f"--NUM_CATALOG_MOVIES", type=str,
        help=f"number of movies in the catalog",
        default=3883,
        required=False
    )
    
    args = parser.parse_args()
    args_dict = vars(args)
    
    quiet = args_dict.get("quiet")
    output_compare_dir = args_dict.get("OUTPUT_COMPARE_DIR")
    tag_1 = args_dict.get("TAG_1")
    tag_2 = args_dict.get("TAG_2")
    num_catalog_movies = args_dict.get("NUM_CATALOG_MOVIES")
    
    if args_dict.get("analyze_and_compare"):
        saved_model_dir_1 = args_dict.get("SAVED_MODEL_DIR_1")
        saved_model_dir_2 = args_dict.get("SAVED_MODEL_DIR_2")
        model_version_1 = args_dict.get("MODEL_VERSION_1")
        model_version_2 = args_dict.get("MODEL_VERSION_2")
        output_dir_1 = args_dict.get("OUTPUT_BASE_DIR_1")
        output_dir_2 = args_dict.get("OUTPUT_BASE_DIR_2")
        
        input_base_dir_1 = run_analysis(saved_model_dir_1, model_version_1, output_dir_1, quiet=quiet)
        print(f'wrote to {input_base_dir_1}')
        input_base_dir_2 = run_analysis(saved_model_dir_2, model_version_2, output_dir_2, quiet=quiet)
        print(f'wrote to {input_base_dir_2}')
    
    elif args_dict.get("compare"):
        input_base_dir_1 = args_dict.get("INPUT_BASE_DIR_1")
        input_base_dir_2 = args_dict.get("INPUT_BASE_DIR_2")
    
    else:
        print("choose --analyze_and_compare or --compare", flush=True)
        exit(0)
    
    print("begin comparison")
    output_dir = run_compare(input_base_dir_1, input_base_dir_2, output_compare_dir=output_compare_dir,
        tag_1=tag_1, tag_2=tag_2, num_catalog_movies=num_catalog_movies, quiet=quiet)
    

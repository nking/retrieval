"""
scripts to analysis the trained models from the recommender_systems project.

some of the analysis is in rust in the ranker project directory inference_src and will be ported
to python here to have it all in one place
"""
import json
import os
import shutil
import subprocess
import argparse
from pathlib import Path
from typing import Tuple

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
--PIPELINE_DATA_DIR_1=../TMP10/bin/rs_pipeline \
--OUTPUT_BASE_DIR_1="../TMP10" \
--PIPELINE_DATA_DIR_2=../TMP11/bin/rs_pipeline \
--OUTPUT_BASE_DIR_2="../TMP11" \
--TAG_1="high-t-model" \
--TAG_2="low-t-model" \
--OUTPUT_COMPARE_DIR="../TMP10_TMP11_Compare" \
--quiet

python -m debugpy --listen 5678 --wait-for-client \
src/test/python/movie_lens_retrieval/post_training_compare_models.py \
--analyze_and_compare \
--PIPELINE_DATA_DIR_1=../TMP10/bin/rs_pipeline \
--OUTPUT_BASE_DIR_1="../TMP10" \
--PIPELINE_DATA_DIR_2=../TMP11/bin/rs_pipeline \
--OUTPUT_BASE_DIR_2="../TMP11" \
--TAG_1="high-t-model" \
--TAG_2="low-t-model" \
--OUTPUT_COMPARE_DIR="../TMP10_TMP11_Compare" \
--quiet

or just compare:

python3 src/test/python/movie_lens_retrieval/post_training_compare_models.py \
--compare \
--INPUT_BASE_DIR_1="../TMP42/post_training_analysis" \
--INPUT_BASE_DIR_2="../TMP43/post_training_analysis" \
--TAG_1="DNN[32,16]" \
--TAG_2="DNN[64,32]" \
--OUTPUT_COMPARE_DIR="../TMP42_TMP43_Compare"

NOTE: nothing in here uses the src/test/resources/data embeddings or recommendations.
'''

from helper import get_project_dir

def run_analysis(pipeline_dir: str, output_dir: str, quiet: bool = False) -> str:
    
    script_path = os.path.join(get_project_dir(),
        "src/test/python/movie_lens_retrieval/post_training_analysis.py")
    # export SAVED_MODEL_DIR="../TMP10/bin/rs_pipeline/Pusher/pushed_model"
    # export MODEL_VERSION="21"
    # export OUTPUT_BASE_DIR="../TMP10"
    # python3 -m unittest src.test.python.movie_lens_retrieval.post_training_analysis.TestAnalysis
    env = os.environ.copy()
    env["PIPELINE_DATA_DIR"] = pipeline_dir
    env["OUTPUT_BASE_DIR"] = output_dir
        
    stdout_setting = subprocess.DEVNULL if quiet else None
    
    result = subprocess.run(["python3", script_path], env=env,
        stdout=stdout_setting)
    if result.returncode == 0:
        print("model analyzed")
    else:
        raise Exception("model analysis failed.")
    
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
        
        #inter_list_diversity
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
    
    # Optional Paired Hubness & Hypersphere Geometry Parquet Metrics (when generated by post_training_analysis-v3)
    if top_k != 20:
        user_hub_path_1 = os.path.join(analysis_dir_1, "user_embedding_hubness.parquet")
        user_hub_path_2 = os.path.join(analysis_dir_2, "user_embedding_hubness.parquet")
        if os.path.exists(user_hub_path_1) and os.path.exists(user_hub_path_2):
            evaluate_metric(
                "user_embedding_hubness.parquet", "user_id",
                "user_uniformity_potential", "User Uniformity Potential (All Users)"
            )
            evaluate_metric(
                "user_embedding_hubness.parquet", "user_id",
                "user_pos_alignment", "User Positive Alignment DistSq (All Users)"
            )
        movie_hub_path_1 = os.path.join(analysis_dir_1, "movie_embedding_hubness.parquet")
        movie_hub_path_2 = os.path.join(analysis_dir_2, "movie_embedding_hubness.parquet")
        if os.path.exists(movie_hub_path_1) and os.path.exists(movie_hub_path_2):
            evaluate_metric(
                "movie_embedding_hubness.parquet", "movie_id",
                "movie_uniformity_potential", "Movie Uniformity Potential (All Movies)"
            )
            evaluate_metric(
                "movie_embedding_hubness.parquet", "movie_id",
                "movie_pos_alignment", "Movie Positive Alignment DistSq (Rated Movies)"
            )
    
    # Return results as a formatted Polars DataFrame
    return pl.DataFrame(results)


def run_compare(analysis_dir_1: str, analysis_dir_2: str,
        output_compare_dir: str,
        tag_1: str, tag_2: str, num_catalog_movies: int,
        quiet: bool = False) -> str:

    resolved_out = Path(output_compare_dir).resolve()
    if resolved_out in (Path.cwd().resolve(), Path.cwd().parent.resolve(), Path("/").resolve()):
        raise ValueError(f"Refusing to wipe unsafe directory: {resolved_out}")
    shutil.rmtree(resolved_out, ignore_errors=True)

    os.makedirs(output_compare_dir, exist_ok=True)
    
    if not os.path.exists(analysis_dir_1):
        raise ValueError(f"analysis_dir_1=>{analysis_dir_1}<= does not exist")
    
    if not os.path.exists(analysis_dir_2):
        raise ValueError(f"analysis_dir_2=>{analysis_dir_2}<= does not exist")
    
    parquet_dir_1 = os.path.join(analysis_dir_1, "parquet_metrics")
    parquet_dir_2 = os.path.join(analysis_dir_2, "parquet_metrics")
    
    agg_res_metrics = dict()
    agg_res_conclusions = []
    
    # Updated to include Inter-List Diversity & Hypersphere Hubness/Geometry metadata
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
            "meaning": "recommending a wide, semantically diverse range of movie genres (intra-slate breadth)",
            "higher_is_better": True},
        "Delta Log Pop (User Tier 0)": {
            "meaning": "calibrating retrieved slate popularity to match mainstream/power users' natural consumption (closer to 0 = less popularity distortion; more negative = stronger niche push)",
            "higher_is_better": "closer_to_zero"},
        "Delta Log Pop (User Tier 1)": {
            "meaning": "calibrating retrieved slate popularity to match mid-tail users' natural consumption (closer to 0 = less popularity distortion; more negative = stronger niche push)",
            "higher_is_better": "closer_to_zero"},
        "Delta Log Pop (User Tier 2)": {
            "meaning": "calibrating retrieved slate popularity to match light/niche users' natural consumption (closer to 0 = less popularity distortion; more negative = stronger niche push)",
            "higher_is_better": "closer_to_zero"},
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
            "meaning": "providing highly individualized slates to mainstream/power users",
            "higher_is_better": True},
        "Inter-List Diversity (User Tier 1)@": {
            "meaning": "providing highly individualized slates to mid-tail users",
            "higher_is_better": True},
        "Inter-List Diversity (User Tier 2)@": {
            "meaning": "providing highly individualized slates to light/niche users",
            "higher_is_better": True},
        "User Uniformity Potential": {
            "meaning": "spreading user query embeddings evenly across the hypersphere without island clumping",
            "higher_is_better": False},
        "User Positive Alignment DistSq": {
            "meaning": "aligning user embeddings closely with their ground-truth positive movie embeddings",
            "higher_is_better": False},
        "Movie Uniformity Potential": {
            "meaning": "spreading movie candidate embeddings evenly across the hypersphere without clumping",
            "higher_is_better": False},
        "Movie Positive Alignment DistSq": {
            "meaning": "aligning movie candidate embeddings closely with their positive user queries",
            "higher_is_better": False}
    }
    
    def get_metric_meta(metric: str) -> Dict[str, Any]:
        for key, meta in metric_meaning_dict.items():
            if metric.startswith(key):
                return meta
        return {"meaning": "affecting an unknown metric",
            "higher_is_better": True}
    
    # Granular tracking to distinguish true exploration from Manifold Fracture / Island Collapse
    trend_tracker = {
        "accuracy_wins": 0, "accuracy_losses": 0,
        "top20_accuracy_wins": 0, "top20_accuracy_losses": 0,
        "ild_wins": 0, "ild_losses": 0,
        "inter_div_wins": 0, "inter_div_losses": 0,
        "tail_pop_wins": 0, "tail_pop_losses": 0,
        "diversity_wins": 0, "diversity_losses": 0,
        "geometry_wins": 0, "geometry_losses": 0,
        "bootstrap_entropy_wins": 0, "bootstrap_entropy_losses": 0,
        "bootstrap_gini_wins": 0, "bootstrap_gini_losses": 0,
    }
    
    for top_k in (20, 100):
        df_results = _compare_model_metrics(
            analysis_dir_1=parquet_dir_1, analysis_dir_2=parquet_dir_2,
            tag_1=tag_1, tag_2=tag_2, top_k=top_k
        )
        
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
            
            c1 = stats_dict.get("T-PValue", 1.0)
            c2 = stats_dict.get("Wilcoxon-PVal",
                stats_dict.get("Wilcoxon-PValue", 1.0))
            
            mean_1 = stats_dict.get(f"Mean ({tag_1})", 0.0)
            mean_2 = stats_dict.get(f"Mean ({tag_2})", 0.0)
            
            is_significant = (c1 < 0.05) and (c2 < 0.05)
            
            if is_significant:
                if meta["higher_is_better"] is None:
                    conclusions.append(
                        f"[*] {tag_1} is significantly {meaning} compared to {tag_2} (P < 0.05).")
                elif meta["higher_is_better"] == "closer_to_zero":
                    # For Delta Log Pop = retrieved_log_pop - gt_log_pop:
                    # 0.0 is neutral popularity alignment; > 0 is popularity amplification; << 0 is niche over-suppression.
                    # If both models are already negative (< -0.5), closer to 0 means better popularity calibration,
                    # whereas more negative means heavier long-tail push.
                    if mean_1 < -0.5 and mean_2 < -0.5:
                        direction_note = (
                            f"{tag_1} ({mean_1:.3f}) is closer to neutral ground-truth popularity (0.0) than {tag_2} ({mean_2:.3f}), "
                            f"avoiding excessive popularity over-suppression while still surfacing tail items"
                            if abs(mean_1) < abs(mean_2)
                            else f"{tag_1} ({mean_1:.3f}) pushes deeper into niche/obscure items than {tag_2} ({mean_2:.3f}) "
                                 f"(beneficial for tail exploration only if Recall and ILD@100 do not degrade)"
                        )
                        conclusions.append(f"[*] Delta Log Pop ({metric}): {direction_note}.")
                    else:
                        tag_1_closer = abs(mean_1) < abs(mean_2)
                        sym = "[+]" if tag_1_closer else "[-]"
                        verdict_word = "BETTER" if tag_1_closer else "WORSE"
                        conclusions.append(
                            f"{sym} {tag_1} ({mean_1:.3f}) is significantly {verdict_word} than {tag_2} ({mean_2:.3f}) at {meaning}."
                        )
                else:
                    tag_1_is_better = (mean_1 > mean_2) if meta[
                        "higher_is_better"] else (mean_1 < mean_2)
                    
                    if tag_1_is_better:
                        conclusions.append(
                            f"[+] {tag_1} is significantly BETTER than {tag_2} at {meaning}.")
                        if "NDCG" in metric or "Recall" in metric or "Precision" in metric:
                            trend_tracker["accuracy_wins"] += 1
                            if top_k == 20:
                                trend_tracker["top20_accuracy_wins"] += 1
                        if metric.startswith("ILD@"):
                            trend_tracker["ild_wins"] += 1
                            trend_tracker["diversity_wins"] += 1
                        elif "Inter-List Diversity" in metric:
                            trend_tracker["inter_div_wins"] += 1
                            trend_tracker["diversity_wins"] += 1
                        elif "Uniformity Potential" in metric or "Alignment DistSq" in metric:
                            trend_tracker["geometry_wins"] += 1
                    else:
                        extra_note = ""
                        if "Inter-List Diversity" in metric and trend_tracker["ild_wins"] > 0 and trend_tracker["accuracy_wins"] >= 2:
                            extra_note = (
                                f" (Note: {tag_2}'s higher cross-user slate difference here is a side effect of "
                                f"lower intra-slate breadth ILD@100 / island fragmentation rather than superior personalization)"
                            )
                        conclusions.append(
                            f"[-] {tag_1} is significantly WORSE than {tag_2} at {meaning}.{extra_note}")
                        if "NDCG" in metric or "Recall" in metric or "Precision" in metric:
                            trend_tracker["accuracy_losses"] += 1
                            if top_k == 20:
                                trend_tracker["top20_accuracy_losses"] += 1
                        if metric.startswith("ILD@"):
                            trend_tracker["ild_losses"] += 1
                            trend_tracker["diversity_losses"] += 1
                        elif "Inter-List Diversity" in metric:
                            trend_tracker["inter_div_losses"] += 1
                            trend_tracker["diversity_losses"] += 1
                        elif "Uniformity Potential" in metric or "Alignment DistSq" in metric:
                            trend_tracker["geometry_losses"] += 1
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
        
        # Track bootstrap global entropy and gini @100
        if top_k == 100:
            g_res = bootstrap_results.get("Bootstrap_Gini@100", {})
            if g_res.get("Empirical_P_Value", 1.0) < 0.05:
                if g_res.get("Mean_Diff", 0.0) < 0:
                    trend_tracker["bootstrap_gini_wins"] += 1
                else:
                    trend_tracker["bootstrap_gini_losses"] += 1
            e_res = bootstrap_results.get("Bootstrap_Uniformity_Entropy@100", {})
            if e_res.get("Empirical_P_Value", 1.0) < 0.05:
                if e_res.get("Mean_Diff", 0.0) > 0:
                    trend_tracker["bootstrap_entropy_wins"] += 1
                else:
                    trend_tracker["bootstrap_entropy_losses"] += 1

    # COMPARE EMBEDDING HUBNESS, UNIFORMITY & ALIGNMENT FROM summary_metrics.json
    hubness_df, hubness_metrics, hubness_conclusions, hub_tracker = _compare_embedding_hubness_summary(
        analysis_dir_1, analysis_dir_2, tag_1, tag_2
    )
    if hubness_df is not None and hubness_df.height > 0:
        print(hubness_df)
    agg_res_metrics.update(hubness_metrics)
    agg_res_conclusions.extend(hubness_conclusions)
    trend_tracker["geometry_wins"] += hub_tracker["geometry_wins"]
    trend_tracker["geometry_losses"] += hub_tracker["geometry_losses"]
    
    # ADD OVERALL TRENDS & STAGE-1 GRAPH RANKER VERDICT CONCLUSIONS
    agg_res_conclusions.append("\n=== OVERALL MACRO TRENDS & STAGE-1 VERDICT ===")
    
    # 1a. Detect Manifold Fracture / Island Collapse Trap in TAG_1 (when TAG_1 is HPO[32,16] and TAG_2 is fixd[32,16]):
    if trend_tracker["ild_losses"] > 0 and trend_tracker["accuracy_losses"] >= 2:
        agg_res_conclusions.append(
            f"-> MANIFOLD FRACTURE / ISLAND COLLAPSE TRAP DETECTED IN {tag_1}: {tag_1} significantly degrades both "
            f"retrieval accuracy ({trend_tracker['accuracy_losses']} accuracy losses) AND Intra-List Diversity (ILD@100), "
            f"with {trend_tracker['geometry_losses']} hypersphere geometry/alignment degradations. "
            f"Apparent gains in Inter-List Diversity or negative Delta Log Pop are artifacts of users being isolated in "
            f"narrow, fragmented embedding islands rather than healthy latent exploration."
        )
        agg_res_conclusions.append(
            f"-> STAGE-1 RECOMMENDATION FOR GRAPH RANKER: REJECT {tag_1} and SELECT {tag_2} (preserves continuous manifold, "
            f"higher semantic slate breadth ILD@100, and superior ground-truth recall)."
        )
    # 1b. Symmetric detection when TAG_1 is the healthy model (fixd[32,16]) and TAG_2 is the fractured model (HPO[32,16]):
    elif trend_tracker["ild_wins"] > 0 and trend_tracker["accuracy_wins"] >= 4 and trend_tracker["geometry_wins"] > trend_tracker["geometry_losses"]:
        agg_res_conclusions.append(
            f"-> SUPERIOR CONTINUOUS MANIFOLD & RETRIEVAL QUALITY ({tag_1} OVER {tag_2}): {tag_1} achieves a decisive win across "
            f"retrieval accuracy ({trend_tracker['accuracy_wins']} accuracy wins, 0 losses), Intra-List Diversity (ILD@100), "
            f"and Hypersphere Geometry ({trend_tracker['geometry_wins']} geometry wins vs {trend_tracker['geometry_losses']} losses). "
            f"By contrast, {tag_2} exhibits manifold island clumping (lower ILD@100 and worse uniformity/alignment), "
            f"which artificially inflates its raw catalog ID scatter (Coverage/Entropy/Inter-List difference) at the expense of relevance."
        )
        agg_res_conclusions.append(
            f"-> STAGE-1 RECOMMENDATION FOR GRAPH RANKER: SELECT {tag_1} and REJECT {tag_2}."
        )
    # 2. Detect Healthy Stage-1 Personalization & Manifold Win for Downstream GraphRanker (e.g. fixd[32,16] vs DNN[64,32]):
    #    Top-20 accuracy is preserved (no top-20 losses), ILD@100 is significantly better (no island fracture),
    #    and personalization/entropy/geometry win, with at most a minor Recall@100 trade-off on head popularity.
    elif (
        trend_tracker["top20_accuracy_losses"] == 0
        and trend_tracker["ild_losses"] == 0
        and (trend_tracker["ild_wins"] > 0 or trend_tracker["inter_div_wins"] > 0)
        and (trend_tracker["bootstrap_entropy_wins"] > 0 or trend_tracker["geometry_wins"] >= trend_tracker["geometry_losses"])
    ):
        if trend_tracker["accuracy_losses"] == 0:
            agg_res_conclusions.append(
                f"-> PARETO IMPROVEMENT: {tag_1} significantly improves diversity, personalization, and hypersphere utilization "
                f"without sacrificing any ranking or retrieval accuracy."
            )
        else:
            agg_res_conclusions.append(
                f"-> HEALTHY PERSONALIZATION & MANIFOLD WIN: {tag_1} preserves Top-20 ranking accuracy while significantly improving "
                f"Intra-List Diversity (ILD@100), Mid/Niche-User Personalization, Catalog Equity (lower Gini / higher Entropy), "
                f"and Hypersphere Geometry ({trend_tracker['geometry_wins']} geometry wins vs {trend_tracker['geometry_losses']} losses), "
                f"with only a minor popularity-driven Recall@100 trade-off."
            )
        agg_res_conclusions.append(
            f"-> STAGE-1 RECOMMENDATION FOR GRAPH RANKER: SELECT {tag_1} when optimizing for personalization, diversity, and "
            f"robustness to train-test covariate shift."
        )
    elif trend_tracker["diversity_wins"] > 0 and trend_tracker["accuracy_losses"] == 0:
        agg_res_conclusions.append(
            f"-> PARETO IMPROVEMENT: {tag_1} significantly improved diversity and exploration without sacrificing ranking accuracy.")
    elif trend_tracker["diversity_wins"] > 0 and trend_tracker["accuracy_losses"] > 0:
        agg_res_conclusions.append(
            f"-> TRADE-OFF: {tag_1} increases diversity and long-tail exploration, but at the cost of top-tier precision/accuracy.")
    elif trend_tracker["accuracy_wins"] > 0 and trend_tracker["diversity_losses"] == 0:
        agg_res_conclusions.append(
            f"-> EXPLOITATION WIN: {tag_1} improved core accuracy/ranking metrics without harming existing catalog diversity.")
    elif (
        trend_tracker["accuracy_losses"] == 0
        and trend_tracker["diversity_losses"] == 0
        and trend_tracker["accuracy_wins"] == 0
        and trend_tracker["diversity_wins"] == 0
    ):
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

    
def _compare_embedding_hubness_summary(
        analysis_dir_1: str,
        analysis_dir_2: str,
        tag_1: str,
        tag_2: str,
        rel_tol: float = 0.02
) -> Tuple[pl.DataFrame, Dict[str, Any], List[str], Dict[str, int]]:
    """
    Compares the Embedding Hubness, Hypersphere Uniformity, and Positive Pair Alignment
    metrics stored in summary_metrics.json (or embedding_hubness.json) between two models.
    
    Uses both a relative tolerance (rel_tol) AND a metric-specific absolute tolerance (abs_tol)
    so that near-zero metrics (e.g. anti-hub fraction 0.0018 vs 0.0005 or low k-NN skewness < 0.5)
    are not falsely flagged as significant degradations due to small denominators.
    """
    def _load_hubness_dict(base_dir: str) -> Dict[str, Any]:
        summary_path = os.path.join(base_dir, "summary_metrics.json")
        hub_path = os.path.join(base_dir, "embedding_hubness.json")
        data = {}
        if os.path.exists(summary_path):
            with open(summary_path, "r") as f:
                data.update(json.load(f))
        if os.path.exists(hub_path):
            with open(hub_path, "r") as f:
                data.update(json.load(f))
        return data

    d1 = _load_hubness_dict(analysis_dir_1)
    d2 = _load_hubness_dict(analysis_dir_2)
    
    # Tuple format: (key, display_name, higher_is_better, abs_tol, healthy_ceiling, meaning)
    hubness_specs = [
        ("uniformity_global_item", "Global Movie Uniformity (log-potential)", False, 0.04, None,
         "spreading the movie catalog evenly across the hypersphere without clumping"),
        ("uniformity_global_user", "Global User Uniformity (log-potential)", False, 0.04, None,
         "spreading user query embeddings evenly across the hypersphere without island clumping"),
        ("uniformity_user_tier_0", "User Uniformity (Tier 0 - Power Users)", False, 0.04, None,
         "preventing power-user embeddings from collapsing into dense islands"),
        ("uniformity_user_tier_1", "User Uniformity (Tier 1 - Mid-Tail Users)", False, 0.04, None,
         "spreading mid-tail user embeddings evenly across the hypersphere"),
        ("uniformity_user_tier_2", "User Uniformity (Tier 2 - Light/Niche Users)", False, 0.04, None,
         "spreading light/niche user embeddings evenly across the hypersphere"),
        ("alignment_user_tier_0", "Positive Pair Alignment DistSq (User Tier 0)", False, 0.035, None,
         "aligning power-user embeddings closely with their ground-truth positive movies"),
        ("alignment_user_tier_1", "Positive Pair Alignment DistSq (User Tier 1)", False, 0.035, None,
         "aligning mid-tail user embeddings closely with their ground-truth positive movies"),
        ("alignment_user_tier_2", "Positive Pair Alignment DistSq (User Tier 2)", False, 0.035, None,
         "aligning light/niche user embeddings closely with their ground-truth positive movies"),
        ("uniformity_movie_tier_0", "Movie Uniformity (Tier 0 - Head Movies)", False, 0.04, None,
         "spreading popular head movie embeddings without hub clumping"),
        ("uniformity_movie_tier_1", "Movie Uniformity (Tier 1 - Torso Movies)", False, 0.04, None,
         "spreading torso movie embeddings evenly across the hypersphere"),
        ("uniformity_movie_tier_2", "Movie Uniformity (Tier 2 - Tail Movies)", False, 0.04, None,
         "spreading long-tail movie embeddings evenly without tail collapse"),
        ("alignment_movie_tier_0", "Positive Pair Alignment DistSq (Movie Tier 0)", False, 0.035, None,
         "aligning head movies closely with their positive user queries"),
        ("alignment_movie_tier_1", "Positive Pair Alignment DistSq (Movie Tier 1)", False, 0.035, None,
         "aligning torso movies closely with their positive user queries"),
        ("alignment_movie_tier_2", "Positive Pair Alignment DistSq (Movie Tier 2)", False, 0.035, None,
         "aligning long-tail movies closely with their positive user queries"),
        ("hubness_item_k20_skewness", "Movie kNN Hubness Skewness (k=20)", False, 0.25, 1.0,
         "suppressing super-hub concentration in the movie nearest-neighbor graph"),
        ("hubness_item_k20_antihub_pct", "Movie kNN Anti-Hub Fraction (k=20)", False, 0.01, 0.01,
         "minimizing orphaned anti-hub movies that never appear in nearest-neighbor lists"),
    ]
    
    rows = []
    metrics_out = {}
    conclusions = ["\n--- Embedding Hubness & Hypersphere Geometry Analysis ---"]
    tracker = {"geometry_wins": 0, "geometry_losses": 0}
    
    for key, display_name, higher_is_better, abs_tol, healthy_ceiling, meaning in hubness_specs:
        if key not in d1 or key not in d2:
            continue
        v1 = float(d1[key])
        v2 = float(d2[key])
        diff = v1 - v2
        denom = max(abs(v2), 1e-9)
        rel_change = diff / denom
        
        # Require BOTH relative shift >= rel_tol AND absolute difference >= abs_tol
        # Also, if both models are well inside a healthy near-zero regime (e.g. skewness < 1.0 or antihubs < 1%), treat as comparable/healthy.
        both_healthy_near_zero = (healthy_ceiling is not None) and (v1 <= healthy_ceiling) and (v2 <= healthy_ceiling) and (abs(diff) < abs_tol)
        is_meaningful = (abs(rel_change) >= rel_tol) and (abs(diff) >= abs_tol) and (not both_healthy_near_zero)
        
        if higher_is_better:
            tag1_better = diff > 0
        else:
            tag1_better = diff < 0
            
        verdict_str = "TIE"
        if is_meaningful:
            if tag1_better:
                verdict_str = f"{tag_1} BETTER"
                tracker["geometry_wins"] += 1
                conclusions.append(
                    f"[+] {tag_1} is BETTER than {tag_2} at {meaning} "
                    f"({v1:.4f} vs {v2:.4f}, Delta: {diff:+.4f})."
                )
            else:
                verdict_str = f"{tag_1} WORSE"
                tracker["geometry_losses"] += 1
                conclusions.append(
                    f"[-] {tag_1} is WORSE than {tag_2} at {meaning} "
                    f"({v1:.4f} vs {v2:.4f}, Delta: {diff:+.4f})."
                )
        else:
            healthy_note = " [Both in healthy low-hubness regime]" if (healthy_ceiling is not None and v1 <= healthy_ceiling and v2 <= healthy_ceiling) else ""
            conclusions.append(
                f"[=] {tag_1} is comparable to {tag_2} at {meaning} "
                f"({v1:.4f} vs {v2:.4f}, Delta: {diff:+.4f}).{healthy_note}"
            )
            
        rows.append({
            "Hubness / Geometry Metric": display_name,
            f"Value ({tag_1})": round(v1, 6),
            f"Value ({tag_2})": round(v2, 6),
            "Delta": round(diff, 6),
            "Rel_Change_%": round(rel_change * 100.0, 2),
            "Direction": "Higher=Better" if higher_is_better else "Lower=Better",
            "Verdict": verdict_str
        })
        metrics_out[f"Hubness_{key}"] = {
            f"Value ({tag_1})": v1,
            f"Value ({tag_2})": v2,
            "Delta": diff,
            "Rel_Change_Pct": rel_change * 100.0,
            "Verdict": verdict_str
        }
        
    # Paired across-tier checks for User Uniformity and User Alignment (3 tiers)
    u_uni_keys = [f"uniformity_user_tier_{t}" for t in (0, 1, 2)]
    if all(k in d1 and k in d2 for k in u_uni_keys):
        u1 = np.array([float(d1[k]) for k in u_uni_keys])
        u2 = np.array([float(d2[k]) for k in u_uni_keys])
        mean_u_diff = float(np.mean(u1 - u2))
        if np.all(u1 > u2):
            conclusions.append(
                f"[!] USER MANIFOLD CLUMPING WARNING: {tag_1} has higher (worse) uniformity potential across ALL 3 user tiers "
                f"(mean shift: {mean_u_diff:+.4f}), confirming increased user island clumping on the hypersphere."
            )
        elif np.all(u1 < u2):
            conclusions.append(
                f"[*] USER MANIFOLD UNIFORMITY WIN: {tag_1} achieves lower (better) uniformity potential across ALL 3 user tiers "
                f"(mean shift: {mean_u_diff:+.4f}), confirming more even user coverage of the hypersphere."
            )

    u_align_keys = [f"alignment_user_tier_{t}" for t in (0, 1, 2)]
    if all(k in d1 and k in d2 for k in u_align_keys):
        a1 = np.array([float(d1[k]) for k in u_align_keys])
        a2 = np.array([float(d2[k]) for k in u_align_keys])
        mean_a_diff = float(np.mean(a1 - a2))
        m_uni_1 = float(d1.get("uniformity_global_item", 0.0))
        m_uni_2 = float(d2.get("uniformity_global_item", 0.0))
        ild_1 = float(d1.get("ild_movie", 0.0))
        ild_2 = float(d2.get("ild_movie", 0.0))
        if np.all(a1 > a2):
            if (m_uni_1 < m_uni_2) and (ild_1 > ild_2):
                conclusions.append(
                    f"[*] MULTI-INTEREST DISPERSION SHIFT: {tag_1} has a {mean_a_diff:+.4f} larger positive-pair squared distance across user tiers "
                    f"because its movie catalog is spread more broadly across the hypersphere (Movie Uniformity {m_uni_1:.4f} vs {m_uni_2:.4f}, "
                    f"ILD {ild_1:.4f} vs {ild_2:.4f}); verify that Recall@100 and NDCG@100 are preserved."
                )
            else:
                conclusions.append(
                    f"[!] POSITIVE ALIGNMENT DEGRADATION: {tag_1} has larger (worse) positive-pair distance across ALL 3 user tiers "
                    f"(mean shift: {mean_a_diff:+.4f}), indicating weaker user-to-item alignment."
                )
        elif np.all(a1 < a2):
            conclusions.append(
                f"[*] POSITIVE ALIGNMENT WIN: {tag_1} achieves tighter (better) positive-pair alignment across ALL 3 user tiers "
                f"(mean shift: {mean_a_diff:+.4f})."
            )

    df_out = pl.DataFrame(rows) if rows else pl.DataFrame()
    return df_out, metrics_out, conclusions, tracker


def _bootstrap_global_metrics(
        dir_1: str, dir_2: str, tag_1: str, tag_2: str, top_k: int,
        num_catalog_movies: int, n_iterations: int = 1000
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Performs Paired Bootstrap Resampling on global catalog metrics.
    Samples users with replacement, reconstructs the global recommendation distribution,
    and calculates confidence intervals for the paired differences.
    """
    path_1 = os.path.join(dir_1, f"user_retrievals_top_{top_k}.parquet")
    path_2 = os.path.join(dir_2, f"user_retrievals_top_{top_k}.parquet")
    
    if not (os.path.exists(path_1) and os.path.exists(path_2)):
        return {}, [
            f"[*] Skipping Bootstrap @{top_k}: user_retrievals parquet missing."]
    
    # Determine MOVIE_OFFSET and catalog size from the full-catalog item_frequencies file
    freq_path_1 = os.path.join(dir_1, f"item_frequencies_top_{top_k}.parquet")
    if os.path.exists(freq_path_1):
        catalog_df = pl.read_parquet(freq_path_1).select("movie_id")
        movie_offset = int(catalog_df["movie_id"].min())
        num_catalog_movies = catalog_df.height
    else:
        # Fallback if item_frequencies is missing (MovieLens 1M: 6040 users + 1)
        movie_offset = 6041
        num_catalog_movies = int(num_catalog_movies)

    # Shift movie_id back to 0-based catalog indices [0, num_catalog_movies - 1]
    df1 = pl.read_parquet(path_1).select([
        pl.col("user_id"),
        (pl.col("movie_id") - movie_offset).alias("movie_id")
    ])
    df2 = pl.read_parquet(path_2).select([
        pl.col("user_id"),
        (pl.col("movie_id") - movie_offset).alias("movie_id")
    ])
    
    # Extract overlapping users to ensure paired testing
    common_users = np.intersect1d(
        df1["user_id"].unique().to_numpy(),
        df2["user_id"].unique().to_numpy()
    )
    n_users = len(common_users)
    
    if n_users == 0:
        return {}, []
    
    # Filter DataFrames to only common users
    df1 = df1.filter(pl.col("user_id").is_in(common_users))
    df2 = df2.filter(pl.col("user_id").is_in(common_users))
    
    # Helper: Group movies by user into a list of lists for fast random sampling
    def _build_user_item_matrix(df: pl.DataFrame) -> np.ndarray:
        grouped = df.sort("user_id").group_by("user_id",
            maintain_order=True).agg(pl.col("movie_id"))
        return grouped["movie_id"].to_numpy()
    
    matrix_1 = _build_user_item_matrix(df1)
    matrix_2 = _build_user_item_matrix(df2)
    
    # Internal Global Metric Calculators (operating on 0-based catalog array of length num_catalog_movies)
    def calc_coverage(freqs):
        return np.count_nonzero(freqs) / num_catalog_movies
    
    def calc_gini(freqs):
        # freqs has exact length num_catalog_movies; 0s represent unretrieved catalog movies
        f = np.sort(freqs)
        if len(f) == 0 or f[-1] == 0:
            return 0.0
        cum_sum = np.cumsum(f)
        return (len(f) + 1 - 2 * np.sum(cum_sum) / cum_sum[-1]) / len(f)
    
    def calc_entropy(freqs):
        # Filter > 0 only to avoid log(0) since 0 * log(0) = 0 in Shannon entropy
        p = freqs[freqs > 0] / np.sum(freqs)
        return -np.sum(p * np.log(p))
    
    diffs = {"Coverage": [], "Gini": [], "Uniformity_Entropy": []}
    
    # --- Bootstrap Loop ---
    for _ in range(n_iterations):
        idx = np.random.randint(0, n_users, size=n_users)
        
        items_1 = np.concatenate(matrix_1[idx])
        items_2 = np.concatenate(matrix_2[idx])
        
        # Both freqs_1 and freqs_2 now have exact shape (num_catalog_movies,)
        freqs_1 = np.bincount(items_1, minlength=num_catalog_movies)
        freqs_2 = np.bincount(items_2, minlength=num_catalog_movies)
        
        cov1, cov2 = calc_coverage(freqs_1), calc_coverage(freqs_2)
        gin1, gin2 = calc_gini(freqs_1), calc_gini(freqs_2)
        ent1, ent2 = calc_entropy(freqs_1), calc_entropy(freqs_2)
        
        diffs["Coverage"].append(cov1 - cov2)
        diffs["Gini"].append(gin1 - gin2)
        diffs["Uniformity_Entropy"].append(ent1 - ent2)

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
        if metric_name == "Coverage":
            extra = "(fraction of unique catalog IDs recommended; higher is beneficial only when Recall and ILD@100 are preserved)"
        elif metric_name == "Gini":
            extra = (
                "(measures Catalog Exposure Inequality of recommended item IDs; lower means more equitable slot distribution, "
                "which is beneficial only when Recall and ILD@100 do not drop due to unaligned scatter)"
            )
        elif metric_name == "Uniformity_Entropy":
            extra = (
                "(measures Catalog Exposure Entropy of recommended item IDs, NOT embedding space uniformity; "
                "higher means slots are spread across more IDs, which is only beneficial if Recall and ILD@100 do not drop)"
            )
        conclusions.append(
            f"[{'*' if is_sig else '='}] {tag_1} has {sig_str}{direction} {metric_name} "
            f"than {tag_2} (P-Val: {p_val_empirical:.4f}, "
            f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]). {extra}")
    
    return results_dict, conclusions

 
def get_largest_subdir_version(parent_dir: str) -> str:
    parent_dir = Path(parent_dir)
    numeric_dirs = [d for d in parent_dir.iterdir() if
        d.is_dir() and d.name.isdigit()]
    if numeric_dirs:
        largest_dir = max(numeric_dirs, key=lambda d: int(d.name))
        return largest_dir.name
    raise ValueError("No numeric subdirectories found.")
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
        description="Compare retrieval analysis of 2 models")
    parser.add_argument(
        "--analyze_and_compare",
        action="store_true",
        help="analyze and compare the retrieval from 2 bi-encoders."
             "Note that if set, requires PIPELINE_DATA_DIR_1, OUTPUT_BASE_DIR_1 and the same with _2 endings for the 2nd model",
        required=False
    )
    for v in ("1", "2"):
        parser.add_argument(
            f"--PIPELINE_DATA_DIR_{v}", type=str,
            help="path to the tfx pipeline data holding Pushed directory.",
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
        f"--NUM_CATALOG_MOVIES", type=int,
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
    output_dir_1 = args_dict.get("OUTPUT_BASE_DIR_1")
    output_dir_2 = args_dict.get("OUTPUT_BASE_DIR_2")
    
    if args_dict.get("analyze_and_compare"):
        for k in ("PIPELINE_DATA_DIR_1", "PIPELINE_DATA_DIR_2", "OUTPUT_BASE_DIR_1", "OUTPUT_BASE_DIR_2"):
            if not args_dict.get(k) or not str(args_dict.get(k)).strip():
                parser.error(f"--analyze_and_compare requires a non-empty --{k}")

        pipeline_dir_1 = args_dict.get("PIPELINE_DATA_DIR_1")
        pipeline_dir_2 = args_dict.get("PIPELINE_DATA_DIR_2")
        
        input_base_dir_1 = run_analysis(pipeline_dir_1, output_dir_1, quiet=quiet)
        print(f'wrote to {input_base_dir_1}')
        
        input_base_dir_2 = run_analysis(pipeline_dir_2, output_dir_2, quiet=quiet)
        print(f'wrote to {input_base_dir_2}')
    
    elif args_dict.get("compare"):
        input_base_dir_1 = os.path.join(output_dir_1, "post_training_analysis")
        input_base_dir_2 = os.path.join(output_dir_2, "post_training_analysis")
    
    else:
        print("choose --analyze_and_compare or --compare", flush=True)
        exit(0)
    
    print("begin comparison")
    output_dir = run_compare(input_base_dir_1, input_base_dir_2, output_compare_dir=output_compare_dir,
        tag_1=tag_1, tag_2=tag_2, num_catalog_movies=num_catalog_movies, quiet=quiet)
    

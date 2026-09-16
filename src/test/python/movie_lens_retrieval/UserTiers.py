from typing import Dict

import polars as pl

def get_user_tiers_from_df(ratings_df: pl.DataFrame) -> Dict[int, int]:
    """
    Given a Polars DataFrame with ['user_id', ...],
    returns a HashMap with key='user_id', value=tier where tier is 0, 1, or 2 for
         head, torso, and tail of the distribution of the number of users ratings.
    """
    #Count history length per user
    user_counts = ratings_df.group_by("user_id").agg(
        pl.len().alias("history_length")
    )
    
    # Find the exact cutoff lengths based on quantiles
    tail_cutoff_val = user_counts["history_length"].quantile(0.20,
        interpolation="nearest")
    head_cutoff_val = user_counts["history_length"].quantile(0.80,
        interpolation="nearest")
    
    # Map to tiers based on the cutoffs
    user_tiers_df = user_counts.with_columns(
        pl.when(pl.col("history_length") <= tail_cutoff_val)
        .then(2)  # Tail
        .when(pl.col("history_length") >= head_cutoff_val)
        .then(0)  # Head
        .otherwise(1)  # Torso
        .alias("user_tier")
    ).select(["user_id", "user_tier"])
    
    return dict(user_tiers_df.iter_rows())
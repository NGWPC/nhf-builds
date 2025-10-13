"""A file for all graph related internal functions"""

from collections import defaultdict
from typing import Any

import pandas as pd
import polars as pl


def _build_graph(reference_flowpaths: pd.DataFrame) -> dict[str, Any]:
    """
    The hydrofabric-related functions for building a graph of upstream flowpath connections

    Parameters
    ----------
    reference_flowpaths : pd.DataFrame
        The reference flowpaths

    Returns
    -------
    dict[str, Any]
        The upstream dictionary containing upstream and downstream connections
    """
    upstream_network = defaultdict(list)

    df_subset = reference_flowpaths[["flowpath_id", "hydroseq", "dnhydroseq"]].copy()

    pl_reference_flowpaths = pl.from_pandas(df_subset)

    df = (
        pl_reference_flowpaths.select(
            [
                pl.col("flowpath_id").cast(pl.Float64).cast(pl.Int64).cast(pl.Utf8).alias("flowpath_id_str"),
                pl.col("hydroseq").cast(pl.Utf8).alias("hydroseq_str"),
                pl.col("dnhydroseq"),
            ]
        )
        .filter(pl.col("dnhydroseq").is_not_null() & (pl.col("dnhydroseq") != 0))
        .with_columns(pl.col("dnhydroseq").cast(pl.Utf8).alias("dnhydroseq_str"))
    )

    hydroseq_lookup = pl_reference_flowpaths.select(
        [
            pl.col("hydroseq").cast(pl.Utf8).alias("hydroseq_str"),
            pl.col("flowpath_id").cast(pl.Float64).cast(pl.Int64).cast(pl.Utf8).alias("flowpath_id_str"),
        ]
    )

    merged = df.join(hydroseq_lookup, left_on="dnhydroseq_str", right_on="hydroseq_str", how="inner").select(
        [
            pl.col("flowpath_id_str").alias("upstream_fp"),
            pl.col("flowpath_id_str_right").alias("downstream_fp"),
        ]
    )

    upstream_network = (
        merged.group_by("downstream_fp")
        .agg(pl.col("upstream_fp").alias("upstream_list"))
        .select([pl.col("downstream_fp"), pl.col("upstream_list")])
    )

    upstream_dict = dict(
        zip(
            upstream_network["downstream_fp"].to_list(),
            upstream_network["upstream_list"].to_list(),
            strict=False,
        )
    )  # key is the downstream flowpath ID, the values are the upstream flowpath IDs

    return upstream_dict

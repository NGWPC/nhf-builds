"""A file for all graph related internal functions"""

from collections import defaultdict
from typing import Any

import pandas as pd
import polars as pl


def _detect_cycles(upstream_network: dict[str, list[str]]) -> None:
    """Detect cycles in the upstream network graph.

    Parameters
    ----------
    upstream_network : dict[str, list[str]]
        Dictionary mapping downstream flowpath IDs to lists of upstream flowpath IDs

    Returns
    -------
    list[list[str]]
        List of cycles found, where each cycle is a list of flowpath IDs
        Returns empty list if no cycles found

    """
    cycles = []
    visited = set()
    rec_stack = set()  # Recursion stack to track current path

    def dfs(node: str, path: list[str]) -> None:
        """Depth-first search to detect cycles."""
        visited.add(node)
        rec_stack.add(node)
        path.append(node)

        # Check all upstream neighbors
        for upstream in upstream_network.get(node, []):
            upstream_str = str(upstream)

            if upstream_str not in visited:
                # Continue DFS
                dfs(upstream_str, path.copy())
            elif upstream_str in rec_stack:
                # Found a cycle - extract the cycle path
                cycle_start = path.index(upstream_str)
                cycle = path[cycle_start:] + [upstream_str]
                cycles.append(cycle)

        # Backtrack
        rec_stack.remove(node)

    # Check all nodes (including disconnected components)
    all_nodes = set(upstream_network.keys())
    for upstream_list in upstream_network.values():
        all_nodes.update(str(u) for u in upstream_list)

    for node in all_nodes:
        if node not in visited:
            dfs(node, [])

    if cycles:
        raise ValueError(
            f"Cycles detected in network! Found {len(cycles)} cycle(s). "
            f"This likely indicates an error in the reference flowpath data."
            f"cycles: {cycles}"
        )


def _find_outlets_by_hydroseq(reference_flowpaths: pd.DataFrame) -> list[str]:
    """Find outlets for the river using hydroseq

    Parameters
    ----------
    reference_flowpaths : pd.DataFrame
        the flowpath reference

    Returns
    -------
    list[str]
        all outlets from the reference
    """
    df_subset = reference_flowpaths[["flowpath_id", "hydroseq", "dnhydroseq"]].copy()
    df_pl = pl.from_pandas(df_subset)

    df_with_str_id = df_pl.with_columns(
        pl.col("flowpath_id").cast(pl.Float64).cast(pl.Int64).cast(pl.Utf8).alias("flowpath_id_str")
    )

    hydroseq_set = set(df_pl["hydroseq"].to_list())

    outlets_df = df_with_str_id.filter(
        (pl.col("dnhydroseq") == 0) | ~pl.col("dnhydroseq").is_in(hydroseq_set)
    )  # dnhydroseq is 0, or doesn't exist in hydroseq

    outlets = outlets_df["flowpath_id_str"].to_list()

    return outlets


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

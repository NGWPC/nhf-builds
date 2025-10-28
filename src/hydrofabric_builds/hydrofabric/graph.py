"""A file for all graph related internal functions"""

from collections import defaultdict
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import polars as pl
import rustworkx as rx
from scipy import sparse


def _validate_and_fix_geometries(gdf: gpd.GeoDataFrame, geom_type: str) -> gpd.GeoDataFrame:
    """
    Validate and fix invalid geometries in a GeoDataFrame.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame to validate
    geom_type : str
        Description for logging (e.g., "flowpaths", "divides")

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with fixed geometries
    """
    invalid_mask = ~gdf.geometry.is_valid
    invalid_count = invalid_mask.sum()

    if invalid_count == 0:
        return gdf  # No invalid geometries

    geometries = gdf[invalid_mask].geometry
    gdf.loc[invalid_mask, "geometry"] = geometries.make_valid()

    if len(gdf[~gdf.geometry.is_valid]) > 0:
        raise ValueError(f"Could not fix invalid geometries in {geom_type}")

    still_invalid = (~gdf.geometry.is_valid).sum()
    if still_invalid > 0:
        raise ValueError(f"Invalid Geometries remain: {gdf[~gdf.geometry.is_valid]}")

    return gdf


def _detect_cycles(upstream_network: dict[str, list[str]]) -> None:
    """Detect cycles in the upstream network graph.

    Parameters
    ----------
    upstream_network : dict[str, list[str]]
        Dictionary mapping downstream flowpath IDs to lists of upstream flowpath IDs
    """
    graph = rx.PyDiGraph(check_cycle=True)
    node_indices = {}
    for to_edge, from_edges in upstream_network.items():
        if to_edge not in node_indices:
            node_indices[to_edge] = graph.add_node(to_edge)
        for from_edge in from_edges:
            if from_edge not in node_indices:
                node_indices[from_edge] = graph.add_node(from_edge)
    for to_edge, from_edges in upstream_network.items():
        for from_edge in from_edges:
            graph.add_edge(node_indices[from_edge], node_indices[to_edge], None)
    ts_order = rx.topological_sort(graph)  # Reindex the flowpaths based on the topo order
    id_order = [graph.get_node_data(gidx) for gidx in ts_order]
    idx_map = {id: idx for idx, id in enumerate(id_order)}

    col = []
    row = []

    for node in ts_order:
        if graph.out_degree(node) == 0:  # terminal node
            continue
        id = graph.get_node_data(node)
        assert len(graph.successors(node)) == 1, f"Node {id} has multiple successors, not dendritic"
        id_ds = graph.successors(node)[0]
        col.append(idx_map[id])
        row.append(idx_map[id_ds])

    matrix = sparse.coo_matrix(
        (np.ones(len(row), dtype=np.uint8), (row, col)), shape=(len(ts_order), len(ts_order)), dtype=np.uint8
    )

    # Ensure matrix is lower triangular
    assert np.all(matrix.row >= matrix.col), "Matrix is not lower triangular"


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

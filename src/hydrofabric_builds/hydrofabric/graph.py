"""A file for all graph related internal functions"""

from collections import defaultdict
from typing import Any

import geopandas as gpd
import numpy as np
import polars as pl
import rustworkx as rx
from scipy import sparse


def _validate_and_fix_geometries(gdf: gpd.GeoDataFrame, geom_type: str) -> gpd.GeoDataFrame:
    """Validate and fix invalid geometries in a GeoDataFrame.

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


def _build_rustworkx_object(upstream_network: dict[str, list[str]]) -> tuple[rx.PyDiGraph, dict]:
    """Detect cycles in the upstream network graph.

    Parameters
    ----------
    upstream_network : dict[str, list[str]]
        Dictionary mapping downstream flowpath IDs to lists of upstream flowpath IDs

    Returns
    -------
    rx.PyDiGraph
        The flowpaths object in graph form
    dict
        The node indices for each object in the graph

    Raises
    ------
    AssertionError
        The flowpaths are not in a lower triangular ordering
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
    return graph, node_indices


def _detect_cycles(graph: rx.PyDiGraph) -> None:
    """Detects any cycles in the rustworkx graph

    Parameters
    ----------
    graph : rx.PyDiGraph
        the DiGraph object
    """
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


def _find_outlets_by_hydroseq(reference_flowpaths: pl.DataFrame) -> list[str]:
    """Find outlets for the river using hydroseq

    Parameters
    ----------
    reference_flowpaths : pl.DataFrame
        the flowpath reference

    Returns
    -------
    list[str]
        all outlets from the reference
    """
    df_pl = reference_flowpaths.select(pl.col(["flowpath_id", "hydroseq", "dnhydroseq", "totdasqkm"]))

    df_with_str_id = df_pl.with_columns(
        pl.col("flowpath_id").cast(pl.Float64).cast(pl.Int64).cast(pl.Utf8).alias("flowpath_id_str")
    )

    hydroseq_set = set(df_pl["hydroseq"].to_list())

    outlets_df = df_with_str_id.filter(
        (pl.col("dnhydroseq") == 0) | ~pl.col("dnhydroseq").is_in(hydroseq_set)
    )  # dnhydroseq is 0, or doesn't exist in hydroseq

    # outlets_sorted = outlets_df.sort("totdasqkm", descending=True) # Commenting out until production
    outlets = outlets_df["flowpath_id_str"].to_list()

    return outlets


def _build_graph(reference_flowpaths: pl.DataFrame) -> dict[str, Any]:
    """The hydrofabric-related functions for building a graph of upstream flowpath connections

    Parameters
    ----------
    reference_flowpaths : pl.DataFrame
        The reference flowpaths

    Returns
    -------
    dict[str, Any]
        The upstream dictionary containing upstream and downstream connections
    """
    upstream_network = defaultdict(list)

    pl_reference_flowpaths = reference_flowpaths.select(
        pl.col(["flowpath_id", "hydroseq", "dnhydroseq", "totdasqkm"])
    )

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


def _extract_outlet_subgraph(
    outlet: str,
    graph: rx.PyDiGraph,
    node_indices: dict[str, int],
) -> tuple[rx.PyDiGraph, dict[str, int], set[int]]:
    """Extract just the subgraph for this outlet.

    Parameters
    ----------
    outlet : str
        Outlet flowpath ID
    graph : rx.PyDiGraph
        Full network graph
    node_indices : dict[str, int]
        Mapping of flowpath IDs to node indices in full graph

    Returns
    -------
    tuple
        (subgraph, subgraph_node_indices, upstream_node_set)
        - subgraph: rx.PyDiGraph containing only this outlet's tree
        - subgraph_node_indices: dict mapping flowpath_id -> node index in subgraph
        - upstream_node_set: set of node indices from original graph
    """
    if outlet not in node_indices:
        subgraph = rx.PyDiGraph()
        sub_indices = {outlet: subgraph.add_node(outlet)}
        return subgraph, sub_indices, {0}

    start_node = node_indices[outlet]
    upstream_nodes = rx.ancestors(graph, start_node)
    upstream_nodes.add(start_node)

    subgraph = graph.subgraph(list(upstream_nodes))

    # Create new node_indices mapping for subgraph
    # Map flowpath_id -> new subgraph node index
    sub_indices = {}
    for node_idx in range(len(subgraph)):
        flowpath_id = subgraph.get_node_data(node_idx)
        sub_indices[flowpath_id] = node_idx

    return subgraph, sub_indices, upstream_nodes


def _partition_all_outlet_subgraphs(
    outlets: list[str],
    graph: rx.PyDiGraph,
    node_indices: dict[str, int],
    reference_flowpaths: pl.DataFrame,
    reference_divides: pl.DataFrame,
) -> dict[str, dict[str, Any]]:
    """Pre-partition subgraphs and filter data for all outlets.

    Parameters
    ----------
    outlets : list[str]
        List of outlet flowpath IDs
    graph : rx.PyDiGraph
        Full network graph
    node_indices : dict[str, int]
        Mapping of flowpath IDs to node indices
    reference_flowpaths : pl.DataFrame
        Full reference flowpaths DataFrame
    reference_divides : pl.DataFrame
        Full reference divides DataFrame

    Returns
    -------
    dict[str, dict[str, Any]]
        Dictionary mapping outlet -> {
            "subgraph": rx.PyDiGraph (only this outlet's tree),
            "node_indices": dict (for subgraph),
            "flowpaths": pl.DataFrame (filtered to this outlet),
            "divides": pl.DataFrame (filtered to this outlet),
        }
    """
    partitions = {}

    for outlet in outlets:
        subgraph, sub_indices, _ = _extract_outlet_subgraph(outlet, graph, node_indices)

        relevant_ids = {subgraph.get_node_data(i) for i in range(len(subgraph))}

        filtered_flowpaths = reference_flowpaths.filter(pl.col("flowpath_id").is_in(relevant_ids))
        filtered_divides = reference_divides.filter(pl.col("divide_id").is_in(relevant_ids))

        partitions[outlet] = {
            "subgraph": subgraph,
            "node_indices": sub_indices,
            "flowpaths": filtered_flowpaths,
            "divides": filtered_divides,
        }

    return partitions

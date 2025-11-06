"""A utils file for common hydrofabric building functions"""

from typing import Any

import geopandas as gpd
import pandas as pd
import rustworkx as rx


def _get_upstream_ids_for_outlet(
    outlet: str,
    graph: rx.PyDiGraph,
    node_indices: dict[str, int],
) -> set[str]:
    """Get all upstream flowpath IDs for an outlet using rustworkx graph traversal.

    Parameters
    ----------
    outlet : str
        The outlet flowpath ID
    graph : rx.PyDiGraph
        The rustworkx graph object
    node_indices : dict[str, int]
        Mapping of flowpath_id to graph node index

    Returns
    -------
    set[str]
        Set of all flowpath IDs in this outlet's upstream tree
    """
    if outlet not in node_indices:
        return {outlet}

    outlet_node = node_indices[outlet]
    upstream_nodes = rx.ancestors(graph, outlet_node)
    upstream_nodes.add(outlet_node)
    upstream_ids = {graph.get_node_data(node) for node in upstream_nodes}

    return upstream_ids


def _calculate_id_ranges_pure(outlet_aggregations: dict) -> dict[str, Any]:
    """Pure function to calculate ID ranges (easily testable).

    Parameters
    ----------
    outlet_aggregations : dict
        Dictionary mapping outlet_id -> outlet data with num_features

    Returns
    -------
    dict[str, Any]
        Dictionary with outlet_id_ranges and total_ids_allocated

    Raises
    ------
    ValueError
        If outlet_aggregations is empty or None
    """
    if not outlet_aggregations:
        raise ValueError("No outlet aggregations provided")

    current_id = 1
    outlet_id_ranges = {}

    for outlet, outlet_data in outlet_aggregations.items():
        num_features = outlet_data["num_features"]
        outlet_id_ranges[outlet] = {
            "id_offset": current_id,
            "id_max": current_id + num_features - 1,
            "num_features": num_features,
        }
        current_id += num_features

    total_ids = current_id - 1
    return {
        "outlet_id_ranges": outlet_id_ranges,
        "total_ids_allocated": total_ids,
    }


def _combine_hydrofabrics(
    built_hydrofabrics: dict[str, dict[str, Any]], crs: str
) -> dict[str, (gpd.GeoDataFrame | pd.DataFrame)]:
    """Function to combine multiple outlet hydrofabrics.

    Parameters
    ----------
    built_hydrofabrics : dict[str, dict[str, Any]]
        Dictionary mapping outlet_id -> hydrofabric data containing
        "flowpaths", "divides", and "nexus" GeoDataFrames
    crs : str
        Coordinate reference system for output GeoDataFrames

    Returns
    -------
    dict[str, gpd.GeoDataFrame]
        Dictionary with keys:
        - "flowpaths": Combined GeoDataFrame of all flowpaths
        - "divides": Combined GeoDataFrame of all divides
        - "nexus": Combined GeoDataFrame of all nexus points

    Raises
    ------
    ValueError
        If built_hydrofabrics is empty or None
    KeyError
        If required keys missing from hydrofabric data
    """
    if not built_hydrofabrics:
        raise ValueError("No built hydrofabrics provided")

    all_flowpaths = []
    all_divides = []
    all_nexus = []
    all_reference_flowpaths = []

    valid_hydrofabrics = {k: v for k, v in built_hydrofabrics.items() if v is not None}
    if not valid_hydrofabrics:
        raise ValueError("No valid hydrofabrics to combine (all results were None)")

    for outlet_id, hf_data in valid_hydrofabrics.items():
        if "flowpaths" not in hf_data:
            raise KeyError(f"Missing 'flowpaths' for outlet {outlet_id}")
        if "divides" not in hf_data:
            raise KeyError(f"Missing 'divides' for outlet {outlet_id}")
        if "nexus" not in hf_data:
            raise KeyError(f"Missing 'nexus' for outlet {outlet_id}")
        if "reference_flowpaths" not in hf_data:
            raise KeyError(f"Missing 'reference_flowpaths' for outlet {outlet_id}")

        if hf_data["flowpaths"] is not None:
            if not hf_data["flowpaths"].empty:
                all_flowpaths.append(hf_data["flowpaths"])
        if hf_data["nexus"] is not None:
            if not hf_data["nexus"].empty:
                all_nexus.append(hf_data["nexus"])
        if hf_data["reference_flowpaths"] is not None:
            if not hf_data["reference_flowpaths"].empty:
                all_reference_flowpaths.append(hf_data["reference_flowpaths"])
        if hf_data["divides"] is not None:
            # Making sure that the divides layer geometry isn't empty
            if not hf_data["divides"].empty and not hf_data["divides"].geometry.is_empty.iloc[0]:
                all_divides.append(hf_data["divides"])

    if not all_flowpaths:
        raise ValueError("No non-empty flowpaths to combine across all outlets")

    combined_flowpaths = pd.concat(all_flowpaths, ignore_index=True)
    combined_divides = pd.concat(all_divides, ignore_index=True)
    combined_nexus = pd.concat(all_nexus, ignore_index=True)
    combined_reference_flowpaths = pd.concat(all_reference_flowpaths, ignore_index=True)

    final_flowpaths = gpd.GeoDataFrame(combined_flowpaths)
    final_divides = gpd.GeoDataFrame(combined_divides)
    final_nexus = gpd.GeoDataFrame(combined_nexus)

    if final_flowpaths.crs is not None and final_flowpaths.crs != crs:
        final_flowpaths = final_flowpaths.to_crs(crs)
        final_divides = final_divides.to_crs(crs)
        final_nexus = final_nexus.to_crs(crs)
    elif final_flowpaths.crs is None:
        final_flowpaths = final_flowpaths.set_crs(crs)
        final_divides = final_divides.set_crs(crs)
        final_nexus = final_nexus.set_crs(crs)

    final_flowpaths = final_flowpaths.assign(
        geometry=final_flowpaths.line_merge()
    )  # removing multilinestring geometries
    return {
        "flowpaths": final_flowpaths,
        "divides": final_divides,
        "nexus": final_nexus,
        "reference_flowpaths": combined_reference_flowpaths,
    }

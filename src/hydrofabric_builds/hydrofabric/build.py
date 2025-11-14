"""A file to contain any hydrofabric building functions"""

import logging
from collections import deque
from typing import Any

import geopandas as gpd
import pandas as pd
import rustworkx as rx
from shapely import Point
from shapely.geometry.base import BaseGeometry

from hydrofabric_builds.config import HFConfig
from hydrofabric_builds.schemas.hydrofabric import Aggregations, Classifications

logger = logging.getLogger(__name__)


def _order_aggregates_base(aggregate_data: Aggregations) -> dict[str, dict[str, Any]]:
    """Take multiple aggregations and order them to build a base hydrofabric for NGEN.

    NOTE: the aggregations used are only aggregates, independents, and connectors as Minor
    flowpaths and subdivide-connectors are for routing

    Parameters
    ----------
    aggregate_data : Aggregations
        The Aggregations BaseModel created by _aggregate_geometries()

    Returns
    -------
    dict[str, dict[str, Any]]
        All required aggregations for the base hydrofabric schema, mapping ref_id to unit info
    """
    ref_id_to_unit: dict[str, dict[str, Any]] = {}
    for unit in aggregate_data.aggregates:
        unit_info: dict[str, Any] = {
            "unit": unit,
            "type": "aggregate",
            "vpu_id": unit["vpu_id"],
            "hydroseq": unit["hydroseq"],
            "length_km": unit["length_km"],
            "area_sqkm": unit["area_sqkm"],
            "div_area_sqkm": unit["div_area_sqkm"],
            "up_id": str(unit["up_id"]),
            "dn_id": str(unit["dn_id"]),
            "all_ref_ids": [str(rid) for rid in unit["ref_ids"]],  # Store all ref_ids
        }
        for ref_id in unit["ref_ids"]:
            ref_id_to_unit[str(ref_id)] = unit_info

    for unit in aggregate_data.independents:
        ref_id = unit["ref_ids"]
        ref_id_to_unit[ref_id] = {
            "unit": unit,
            "type": "independent",
            "vpu_id": unit["vpu_id"],
            "hydroseq": unit["hydroseq"],
            "length_km": unit["length_km"],
            "area_sqkm": unit["area_sqkm"],
            "div_area_sqkm": unit["div_area_sqkm"],
            "all_ref_ids": [ref_id],
        }

    for unit in aggregate_data.connectors:
        ref_id = unit["ref_ids"]
        ref_id_to_unit[ref_id] = {
            "unit": unit,
            "type": "connectors",
            "vpu_id": unit["vpu_id"],
            "hydroseq": unit["hydroseq"],
            "length_km": unit["length_km"],
            "area_sqkm": unit["area_sqkm"],
            "div_area_sqkm": unit["div_area_sqkm"],
            "all_ref_ids": [ref_id],
        }
    return ref_id_to_unit


def _queue_all_unit_upstreams(
    unit_info: dict[str, Any],
    ref_ids: list[str],
    current_ref_id: str,
    graph: rx.PyDiGraph,
    node_indices: dict[str, int],
    to_process: deque[str],
) -> None:
    """Queue all upstream flowpaths for a unit.

    For aggregates, checks all ref_ids to capture all upstream branches.
    For independents/connectors, just checks the current flowpath.

    Parameters
    ----------
    unit_info : dict[str, Any]
        Info for the unit
    ref_ids : list[str]
        All reference flowpath IDs in this unit
    current_ref_id : str
        Current flowpath ID being processed
    graph : rx.PyDiGraph
        Network graph
    node_indices : dict[str, int]
        Mapping of flowpath IDs to node indices
    to_process : deque[str]
        Queue to add upstream IDs to
    """
    if unit_info["type"] == "aggregate":
        # For aggregates, check all ref_ids for upstreams
        all_upstream_ids = []
        if "0" in ref_ids:
            ref_ids.remove("0")
        for ref_id in ref_ids:
            if ref_id in node_indices:
                ref_idx = node_indices[ref_id]
                upstream_ids = [graph[idx] for idx in graph.predecessor_indices(ref_idx)]
                all_upstream_ids.extend(upstream_ids)
        unique_upstream_ids = list(set(all_upstream_ids))
        to_process.extend(unique_upstream_ids)
    else:
        # For independents/connectors, just use current_ref_id
        if current_ref_id in node_indices:
            current_idx = node_indices[current_ref_id]
            upstream_ids = [graph[idx] for idx in graph.predecessor_indices(current_idx)]
            to_process.extend(upstream_ids)


def _build_base_hydrofabric(
    start_id: str,
    aggregate_data: Aggregations,
    classifications: Classifications,
    partition_data: dict[str, Any],
    cfg: HFConfig,
    id_offset: int = 0,
) -> dict[str, (gpd.GeoDataFrame | pd.DataFrame) | list[dict[str, Any]] | None]:
    """Build the base hydrofabric layers.

    Parameters
    ----------
    start_id : str
        The outlet flowpath ID
    aggregate_data : Aggregations
        Aggregation data
    classifications : Classifications
        Classification data
    partition_data : dict[str, Any]
        Contains fp_lookup, subgraph, node_indices
    cfg : HFConfig
        Hydrofabric build config
    id_offset : int, optional
        Starting ID offset for this outlet, by default 0

    Returns
    -------
    dict[str, (gpd.GeoDataFrame | pd.DataFrame) | list[dict[str, Any]] | None]
        Built hydrofabric data with keys: flowpaths, divides, nexus, reference_flowpaths
    """
    fp_lookup: dict[str, dict[str, Any]] = partition_data["fp_lookup"]
    graph = partition_data["subgraph"]
    node_indices: dict[str, int] = partition_data["node_indices"]

    # Build hydroseq lookup for O(1) downstream lookups
    fp_by_hydroseq: dict[Any, dict[str, Any]] = {row["hydroseq"]: row for row in fp_lookup.values()}

    # Build ref_id -> unit lookup
    ref_id_to_unit: dict[str, dict[str, Any]] = _order_aggregates_base(aggregate_data)

    # Initialize processing
    to_process: deque[str] = deque([start_id])
    visited_ref_ids: set[str] = set()
    processed_units: set[frozenset[str]] = set()

    fp_data: list[dict[str, Any]] = []
    div_data: list[dict[str, Any]] = []
    nexus_data: list[dict[str, Any]] = []
    reference_flowpaths_data: list[dict[str, Any]] = []
    base_minor_data: list[dict[str, Any]] = []

    ref_id_to_new_id: dict[str, int] = {}
    downstream_fp_to_nexus: dict[int, int] = {}

    new_id = 1 + id_offset
    nexus_counter = 1 + id_offset

    to_process.extend(list(classifications.force_queue_flowpaths))

    # Main processing loop
    while to_process:
        current_ref_id = to_process.popleft()

        if current_ref_id in visited_ref_ids:
            continue
        visited_ref_ids.add(current_ref_id)

        # Skip minor flowpaths
        if current_ref_id in classifications.minor_flowpaths:
            continue

        # Validate current flowpath
        if current_ref_id not in ref_id_to_unit:
            if not fp_data:  # Starting outlet case
                continue
            if current_ref_id in classifications.aggregation_set:  # headwater case
                continue
            logger.error(f"Flowpath {current_ref_id} not found in any unit and not minor")
            raise ValueError(f"Flowpath {current_ref_id} not found in any unit")

        unit_info = ref_id_to_unit[current_ref_id]
        unit_ref_ids = frozenset(unit_info["all_ref_ids"])

        # Skip if already processed
        if unit_ref_ids in processed_units:
            continue
        processed_units.add(unit_ref_ids)

        # Assign new ID to all refs in this unit
        for ref_id in unit_ref_ids:
            ref_id_to_new_id[ref_id] = new_id

        # Get unit data
        unit: dict[str, Any] = unit_info["unit"]
        unit_type: str = unit_info["type"]
        ref_ids: list[str] = list(unit_info["all_ref_ids"])

        # Get original flowpaths
        original_fps: list[dict[str, Any]] = [fp_lookup[ref_id] for ref_id in ref_ids if ref_id in fp_lookup]

        if not original_fps:
            logger.debug(f"No flowpaths found for unit {ref_ids}")
            # Queue upstream
            lookup_id = unit_info.get("up_id", current_ref_id)
            if lookup_id in node_indices:
                lookup_idx = node_indices[lookup_id]
                upstream_ids = [graph[idx] for idx in graph.predecessor_indices(lookup_idx)]
                to_process.extend(upstream_ids)
            continue

        # Find outlet flowpath (lowest hydroseq)
        if unit_type == "aggregate" and unit_info.get("dn_id"):
            outlet_fp = fp_lookup.get(unit_info["dn_id"]) or min(original_fps, key=lambda x: x["hydroseq"])
        else:
            outlet_fp = (
                original_fps[0] if len(original_fps) == 1 else min(original_fps, key=lambda x: x["hydroseq"])
            )

        # Find downstream unit
        dn_hydroseq = outlet_fp.get("dnhydroseq", 0)
        downstream_unit_id = None

        if dn_hydroseq and dn_hydroseq != 0:
            downstream_fp = fp_by_hydroseq.get(dn_hydroseq)
            if downstream_fp:
                downstream_ref_id = str(downstream_fp["flowpath_id"])
                downstream_unit_id = ref_id_to_new_id.get(downstream_ref_id)

        # Check for divide polygon
        polygon_geom: BaseGeometry | None = unit.get("polygon_geometry")
        if polygon_geom is None or polygon_geom.is_empty:
            logger.error(f"Unit {ref_ids} has no divide polygon")
            # _queue_all_unit_upstreams(unit_info, ref_ids, current_ref_id, graph, node_indices, to_process)
            # continue

        # Get or create nexus
        if downstream_unit_id is not None and downstream_unit_id in downstream_fp_to_nexus:
            nexus_id = downstream_fp_to_nexus[downstream_unit_id]
        else:
            nexus_id = nexus_counter
            nexus_counter += 1

            # Create nexus at downstream end of flowpath
            line_geom = unit["line_geometry"]
            if line_geom.geom_type == "MultiLineString":
                end_coord = list(line_geom.geoms)[-1].coords[-1]
            else:
                end_coord = line_geom.coords[-1]

            nexus_data.append(
                {
                    "nex_id": nexus_id,
                    "dn_fp_id": downstream_unit_id,
                    "geometry": Point(end_coord),
                }
            )

            if downstream_unit_id is not None:
                downstream_fp_to_nexus[downstream_unit_id] = nexus_id

        # Create flowpath
        fp_data.append(
            {
                "fp_id": new_id,
                "dn_nex_id": nexus_id,
                "up_nex_id": None,  # Will fix later
                "div_id": new_id,
                "vpu_id": unit["vpu_id"],
                "hydroseq": unit["hydroseq"],
                "length_km": unit["length_km"],
                "area_sqkm": unit["area_sqkm"],
                "geometry": unit["line_geometry"],
            }
        )

        # Create divide
        div_data.append(
            {
                "div_id": new_id,
                "vpu_id": unit["vpu_id"],
                "type": unit_type,
                "area_sqkm": unit["div_area_sqkm"],
                "geometry": polygon_geom,
            }
        )

        # Create reference entries
        for ref_id in ref_ids:
            reference_flowpaths_data.append(
                {
                    "ref_fp_id": int(ref_id),
                    "fp_id": new_id,
                }
            )

        # Track aggregates for minor flowpaths
        if unit_type == "aggregate" and unit_info.get("dn_id"):
            base_minor_data.append(
                {
                    "fp_id": new_id,
                    "dn_ref_id": unit_info["dn_id"],
                    "up_ref_id": unit_info["up_id"],
                }
            )

        new_id += 1

        _queue_all_unit_upstreams(unit_info, ref_ids, current_ref_id, graph, node_indices, to_process)

    # Fix up_nex_id references
    nexus_by_downstream_fp: dict[int, int] = {
        nex["dn_fp_id"]: nex["nex_id"] for nex in nexus_data if nex.get("dn_fp_id") is not None
    }
    for fp_entry in fp_data:
        fp_id = fp_entry["fp_id"]
        if fp_id in nexus_by_downstream_fp:
            fp_entry["up_nex_id"] = nexus_by_downstream_fp[fp_id]

    # Create GeoDataFrames
    try:
        flowpaths_gdf = gpd.GeoDataFrame(fp_data, crs=cfg.crs)
        flowpaths_gdf["fp_id"] = flowpaths_gdf["fp_id"].astype("Int64")
        flowpaths_gdf["dn_nex_id"] = flowpaths_gdf["dn_nex_id"].astype("Int64")
        flowpaths_gdf["up_nex_id"] = flowpaths_gdf["up_nex_id"].astype("Int64")

        divides_gdf = gpd.GeoDataFrame(div_data, crs=cfg.crs)
        nexus_gdf = gpd.GeoDataFrame(nexus_data, crs=cfg.crs)
        reference_flowpaths_df = pd.DataFrame(reference_flowpaths_data)
    except ValueError:
        flowpaths_gdf = None
        divides_gdf = None
        nexus_gdf = None
        reference_flowpaths_df = None

    return {
        "flowpaths": flowpaths_gdf,
        "divides": divides_gdf,
        "nexus": nexus_gdf,
        "base_minor_flowpaths": base_minor_data,
        "reference_flowpaths": reference_flowpaths_df,
    }

"""A file to contain any hydrofabric building functions"""

import logging
from collections import deque
from typing import Any

import geopandas as gpd
import pandas as pd
from shapely import Point

from hydrofabric_builds.config import HFConfig
from hydrofabric_builds.schemas.hydrofabric import Aggregations, Classifications

logger = logging.getLogger(__name__)


def _order_aggregates_base(aggregate_data: Aggregations) -> dict[str, dict]:
    """Takes multiple aggregations and orders them to build a base hydrofabric for NGEN

    NOTE: the aggregations used are only aggregates, independents, and connectors as Minor flowpaths and subdivide-connectors are for routing

    Parameters
    ----------
    aggregate_data: Aggregations
        The Aggregations BaseModel created by _aggregate_geometries()

    Returns
    -------
    dict[str, dict]
        All required aggregations for the base hydrofabric schema
    """
    ref_id_to_unit = {}
    for unit in aggregate_data.aggregates:
        ref_id = unit["dn_id"]
        ref_id_to_unit[ref_id] = {
            "unit": unit,
            "type": "aggregate",
            "up_id": unit["up_id"],
            "dn_id": unit["dn_id"],
            "all_ref_ids": [ref_id],
        }

    for unit in aggregate_data.independents:
        ref_id = unit["ref_ids"]
        ref_id_to_unit[ref_id] = {"unit": unit, "type": "independent", "all_ref_ids": [ref_id]}

    for unit in aggregate_data.connectors:
        ref_id = unit["ref_ids"]
        ref_id_to_unit[ref_id] = {"unit": unit, "type": "small_scale_connector", "all_ref_ids": [ref_id]}
    return ref_id_to_unit


def _build_base_hydrofabric(
    start_id: str,
    aggregate_data: Aggregations,
    classifications: Classifications | None,
    reference_divides: pd.DataFrame,
    reference_flowpaths: pd.DataFrame,
    upstream_network: dict[str, Any],
    cfg: HFConfig,
    id_offset: int = 0,  # NEW PARAMETER
) -> dict[str, pd.DataFrame] | dict[str, list]:
    """
    Builds the base hydrofabric layers containing all primary and foreign key relationships

    Parameters
    ----------
    start_id: str
        The outlet point used for creating the hydrofabric layer
    aggregate_data: Aggregations
        The Aggregations BaseModel created by _aggregate_geometries()
    classifications: Classifications
        The Classifications BaseModel created by _trace_stack()
    reference_divides: pd.DataFrame
        The reference divides base dataset
    reference_flowpaths: pd.DataFrame
        The reference flowpaths base dataset
    upstream_network: dict[str, Any]
        A dictionary mapping downstream to upstream connectivity
    id_offset: int, default=0
        Starting ID offset for this outlet. Used in parallel processing to ensure
        unique IDs across all outlets.

    Returns
    -------
    dict[str, list]
        The flowpaths, divides, nexus DataFrames, and base_minor_flowpaths for the specific outlet
    """
    ref_id_to_unit = _order_aggregates_base(aggregate_data)
    to_process = deque([start_id])

    visited_ref_ids = set()
    processed_units = set()
    fp_data = []
    div_data = []
    nexus_data = []
    base_minor_data = []
    ref_id_to_new_id = {}
    downstream_fp_to_nexus: dict = {}  # downstream_fp_id -> nexus_id to track nexus points

    new_id = 1 + id_offset
    nexus_counter = 1 + id_offset

    while to_process:
        current_ref_id = to_process.popleft()

        if current_ref_id in visited_ref_ids:
            continue

        if current_ref_id not in ref_id_to_unit:
            logger.warning(f"{current_ref_id} not found in any unit")
            visited_ref_ids.add(current_ref_id)
            continue

        unit_info = ref_id_to_unit[current_ref_id]
        unit_ref_ids = frozenset(unit_info["all_ref_ids"])

        if unit_ref_ids in processed_units:
            continue

        processed_units.add(unit_ref_ids)

        for ref_id in unit_ref_ids:
            visited_ref_ids.add(ref_id)
            ref_id_to_new_id[ref_id] = new_id

        unit = unit_info["unit"]
        unit_type = unit_info["type"]
        ref_ids = list(unit_ref_ids)

        original_fps = reference_flowpaths[
            reference_flowpaths["flowpath_id"].isin([float(id_str) for id_str in ref_ids])
        ]

        if len(original_fps) == 0:
            logger.warning(f"No flowpaths found for unit {ref_ids}")
            continue

        if unit_type == "aggregate" and unit_info["dn_id"] is not None:
            dn_id = unit_info["dn_id"]
            outlet_fp = original_fps[original_fps["flowpath_id"] == float(dn_id)]
            if len(outlet_fp) == 0:
                outlet_fp = original_fps.loc[original_fps["hydroseq"].idxmin()]
            else:
                outlet_fp = outlet_fp.iloc[0]
        else:
            outlet_fp = (
                original_fps.iloc[0]
                if len(original_fps) == 1
                else original_fps.loc[original_fps["hydroseq"].idxmin()]
            )

        dn_hydroseq = outlet_fp["dnhydroseq"]
        downstream_unit_id = None

        if pd.notna(dn_hydroseq) and dn_hydroseq != 0:
            downstream_fp = reference_flowpaths[reference_flowpaths["hydroseq"] == dn_hydroseq]
            if len(downstream_fp) > 0:
                downstream_ref_id = str(int(downstream_fp.iloc[0]["flowpath_id"]))

                if downstream_ref_id in ref_id_to_new_id:
                    downstream_unit_id = ref_id_to_new_id[downstream_ref_id]

        # Get or create nexus at the pour point
        if downstream_unit_id is not None and downstream_unit_id in downstream_fp_to_nexus:
            nexus_id = downstream_fp_to_nexus[downstream_unit_id]
        else:
            nexus_id = nexus_counter
            nexus_counter += 1

            fp_geom = unit["line_geometry"]
            if fp_geom.geom_type == "MultiLineString":
                last_line = list(fp_geom.geoms)[-1]
                end_coordinate = last_line.coords[-1]
            else:
                end_coordinate = fp_geom.coords[-1]

            nexus_data.append(
                {
                    "nex_id": nexus_id,
                    "downstream_fp_id": downstream_unit_id,
                    "geometry": Point(end_coordinate),
                }
            )

            if downstream_unit_id is not None:
                downstream_fp_to_nexus[downstream_unit_id] = nexus_id

        fp_entry = {
            "fp_id": new_id,
            "dn_nex_id": nexus_id,
            "up_nex_id": None,
            "div_id": new_id,
            "geometry": unit["line_geometry"],
        }

        if unit_type == "aggregate" and unit_info["dn_id"] is not None:
            base_minor_data.append(
                {
                    "fp_id": new_id,
                    "dn_ref_id": unit_info["dn_id"],
                    "up_ref_id": unit_info["up_id"],
                }
            )

        fp_data.append(fp_entry)
        div_data.append(
            {"div_id": new_id, "type": unit_type, "ref_ids": ref_ids, "geometry": unit["polygon_geometry"]}
        )

        new_id += 1

        # Queue upstream segments
        if unit_info["type"] == "aggregate":
            upstream_ids = upstream_network.get(unit_info["up_id"], [])
        else:
            upstream_ids = upstream_network.get(current_ref_id, [])

        for upstream_id in upstream_ids:
            to_process.append(upstream_id)

    for fp_entry in fp_data:
        my_nexus = fp_entry["dn_nex_id"]
        flowpaths_using_nexus = [
            other_fp["fp_id"] for other_fp in fp_data if other_fp["dn_nex_id"] == my_nexus
        ]
        if len(flowpaths_using_nexus) > 1:
            fp_entry["up_nex_id"] = my_nexus

    flowpaths_gdf = gpd.GeoDataFrame(fp_data, crs=cfg.crs)
    divides_gdf = gpd.GeoDataFrame(div_data, crs=cfg.crs)
    nexus_gdf = gpd.GeoDataFrame(nexus_data, crs=cfg.crs)

    return {
        "flowpaths": flowpaths_gdf,
        "divides": divides_gdf,
        "nexus": nexus_gdf,
        "base_minor_flowpaths": base_minor_data,
    }

"""A file to contain any hydrofabric building functions"""

import logging
from collections import deque
from typing import Any

import geopandas as gpd
import polars as pl
from shapely import Point
from shapely.ops import unary_union

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
        unit_info = {
            "unit": unit,
            "type": "aggregate",
            "up_id": str(unit["up_id"]),
            "dn_id": str(unit["dn_id"]),
            "all_ref_ids": [str(rid) for rid in unit["ref_ids"]],  # Store all ref_ids
        }
        for ref_id in unit["ref_ids"]:
            ref_id_to_unit[str(ref_id)] = unit_info

    for unit in aggregate_data.independents:
        ref_id = unit["ref_ids"]
        ref_id_to_unit[ref_id] = {"unit": unit, "type": "independent", "all_ref_ids": [ref_id]}

    for unit in aggregate_data.connectors:
        ref_id = unit["ref_ids"]
        ref_id_to_unit[ref_id] = {"unit": unit, "type": "connectors", "all_ref_ids": [ref_id]}
    return ref_id_to_unit


def _build_base_hydrofabric(
    start_id: str,
    aggregate_data: Aggregations,
    classifications: Classifications,
    reference_divides: pl.DataFrame,
    reference_flowpaths: pl.DataFrame,
    upstream_network: dict[str, Any],
    cfg: HFConfig,
    id_offset: int = 0,
) -> dict[str, gpd.GeoDataFrame | list[dict] | None]:
    """Builds the base hydrofabric layers with no-divide connector patching

    No-divide connectors are flowpaths without catchment polygons that connect
    upstream to downstream segments. We "patch" them by:
    1. Merging their line geometry into the downstream flowpath
    2. Redirecting upstream connections through them
    3. Not creating separate units for them
    """
    ref_id_to_unit = _order_aggregates_base(aggregate_data)

    # Create mapping of no-divide connectors to their downstream targets
    connector_to_downstream = {}
    connector_line_geoms = {}
    connector_to_upstreams = {}

    for connector in aggregate_data.no_divide_connectors:
        connector_id = connector["ref_ids"]
        dn_id = connector.get("dn_id")
        up_ids = connector.get("up_id", [])

        if dn_id:
            connector_to_downstream[connector_id] = dn_id
            connector_line_geoms[connector_id] = connector["line_geometry"]
            connector_to_upstreams[connector_id] = up_ids

    to_process = deque([start_id])
    visited_ref_ids = set()
    processed_units = set()
    fp_data: list = []
    div_data = []
    nexus_data = []
    base_minor_data: list[dict] = []
    ref_id_to_new_id = {}
    downstream_fp_to_nexus: dict = {}

    new_id = 1 + id_offset
    nexus_counter = 1 + id_offset

    while to_process:
        current_ref_id = to_process.popleft()

        if current_ref_id in visited_ref_ids:
            continue

        # Skip no-divide connectors - they'll be patched into downstream
        if current_ref_id in connector_to_downstream:
            visited_ref_ids.add(current_ref_id)
            # Queue the downstream target instead
            downstream_id = connector_to_downstream[current_ref_id]
            if downstream_id not in visited_ref_ids:
                to_process.append(downstream_id)
            # Queue upstream IDs that connect to this connector
            for up_id in connector_to_upstreams.get(current_ref_id, []):
                if up_id not in visited_ref_ids:
                    to_process.append(up_id)
            continue

        if current_ref_id not in ref_id_to_unit:
            if not fp_data:
                # no flowpaths. Just a random linestring
                visited_ref_ids.add(current_ref_id)
                continue
            elif current_ref_id in classifications.minor_flowpaths:
                # a minor flowpath, we can ignore for the base
                visited_ref_ids.add(current_ref_id)
                continue
            else:
                logger.error(f"{current_ref_id} not found in any unit")
                raise ValueError(f"{current_ref_id} not found in any unit")

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

        # Check if any upstream connectors should be patched into this unit
        line_geoms_to_merge = [unit["line_geometry"]]
        for connector_id, dn_id in connector_to_downstream.items():
            if dn_id == current_ref_id:
                # This connector patches into current unit
                line_geoms_to_merge.append(connector_line_geoms[connector_id])
                logger.debug(f"Patching no-divide connector {connector_id} into {current_ref_id}")

        merged_line_geom = (
            unary_union(line_geoms_to_merge) if len(line_geoms_to_merge) > 1 else unit["line_geometry"]
        )

        original_fps = reference_flowpaths.filter(pl.col("flowpath_id").cast(pl.Utf8).is_in(ref_ids))

        if original_fps.height == 0:
            logger.warning(f"No flowpaths found for unit {ref_ids}")
            continue

        # Find outlet flowpath
        if unit_type == "aggregate" and unit_info.get("dn_id") is not None:
            dn_id = unit_info["dn_id"]
            outlet_fp = original_fps.filter(pl.col("flowpath_id") == dn_id)
            if outlet_fp.height == 0:
                # Get row with minimum hydroseq
                min_hydroseq_idx = original_fps["hydroseq"].arg_min()
                outlet_fp = original_fps[min_hydroseq_idx]
            else:
                outlet_fp = outlet_fp[0]
        else:
            if original_fps.height == 1:
                outlet_fp = original_fps[0]
            else:
                # Get row with minimum hydroseq
                min_hydroseq_idx = original_fps["hydroseq"].arg_min()
                outlet_fp = original_fps[min_hydroseq_idx]

        dn_hydroseq = outlet_fp["dnhydroseq"][0] if outlet_fp.height > 0 else None
        downstream_unit_id = None

        if dn_hydroseq is not None and dn_hydroseq != 0 and not pl.Series([dn_hydroseq]).is_null()[0]:
            downstream_fp = reference_flowpaths.filter(pl.col("hydroseq") == dn_hydroseq)
            if downstream_fp.height > 0:
                downstream_ref_id = str(int(downstream_fp["flowpath_id"][0]))

                if downstream_ref_id in ref_id_to_new_id:
                    downstream_unit_id = ref_id_to_new_id[downstream_ref_id]

        # Get or create nexus at the pour point
        if downstream_unit_id is not None and downstream_unit_id in downstream_fp_to_nexus:
            nexus_id = downstream_fp_to_nexus[downstream_unit_id]
        else:
            nexus_id = nexus_counter
            nexus_counter += 1

            # Use merged geometry for nexus point
            fp_geom = merged_line_geom
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
            "geometry": merged_line_geom,  # Use merged geometry
        }

        if unit_type == "aggregate" and unit_info.get("dn_id") is not None:
            base_minor_data.append(
                {
                    "fp_id": new_id,
                    "dn_ref_id": unit_info["dn_id"],
                    "up_ref_id": unit_info["up_id"],
                }
            )

        fp_data.append(fp_entry)

        polygon_geom = unit.get("polygon_geometry")
        if polygon_geom is not None and not polygon_geom.is_empty:
            div_data.append({"div_id": new_id, "type": unit_type, "geometry": polygon_geom})
        new_id += 1

        # Queue upstream segments
        if unit_info["type"] == "aggregate":
            upstream_ids = upstream_network.get(unit_info["up_id"], [])
        else:
            upstream_ids = upstream_network.get(current_ref_id, [])

        for upstream_id in upstream_ids:
            to_process.append(upstream_id)

    # Fix up_nex_id references
    for fp_entry in fp_data:
        my_nexus = fp_entry["dn_nex_id"]
        flowpaths_using_nexus = [
            other_fp["fp_id"] for other_fp in fp_data if other_fp["dn_nex_id"] == my_nexus
        ]
        if len(flowpaths_using_nexus) > 1:
            fp_entry["up_nex_id"] = my_nexus
    try:
        flowpaths_gdf = gpd.GeoDataFrame(fp_data, crs=cfg.crs)
        divides_gdf = gpd.GeoDataFrame(div_data, crs=cfg.crs)
        nexus_gdf = gpd.GeoDataFrame(nexus_data, crs=cfg.crs)
    except ValueError:
        flowpaths_gdf = None
        divides_gdf = None
        nexus_gdf = None
    return {
        "flowpaths": flowpaths_gdf,  # TODO add ref_id crosswalk table
        "divides": divides_gdf,
        "nexus": nexus_gdf,
        "base_minor_flowpaths": base_minor_data,
    }

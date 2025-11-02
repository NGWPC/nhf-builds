"""A file to contain any hydrofabric building functions"""

import logging
from collections import deque
from typing import Any

import geopandas as gpd
from shapely import Point
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

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


def _prepare_connector_mappings(
    aggregate_data: Aggregations,
) -> tuple[dict[str, str | None], dict[str, list[BaseGeometry]], dict[str, list[str]]]:
    """Prepare connector mappings for downstream patching.

    Parameters
    ----------
    aggregate_data : Aggregations
        The aggregation data containing no-divide connectors

    Returns
    -------
    tuple[dict[str, str | None], dict[str, list[BaseGeometry]], dict[str, list[str]]]
        - connector_to_downstream: Maps connector ID to ultimate downstream target
        - connector_line_geoms: Maps connector ID to chain of line geometries
        - connector_to_upstreams: Maps connector ID to list of upstream IDs
    """
    connector_to_downstream: dict[str, str | None] = {}
    connector_line_geoms: dict[str, list[BaseGeometry]] = {}
    connector_to_upstreams: dict[str, list[str]] = {}

    # First pass: collect all connectors with their immediate downstream
    temp_connector_map: dict[str, dict[str, Any]] = {}
    for connector in aggregate_data.no_divide_connectors:
        connector_id: str = connector["ref_ids"]
        dn_id: str | None = connector.get("dn_id")
        up_ids: list[str] = connector.get("up_id", [])

        temp_connector_map[connector_id] = {
            "dn_id": dn_id,
            "up_ids": up_ids,
            "line_geometry": connector["line_geometry"],
        }

    # Second pass: resolve chains to find ultimate downstream target
    for connector_id, connector_data in temp_connector_map.items():
        dn_id = connector_data["dn_id"]
        chain_geometries: list[BaseGeometry] = [connector_data["line_geometry"]]

        # Follow the chain of connectors to find the final downstream
        visited_in_chain: set[str] = {connector_id}
        while dn_id and dn_id in temp_connector_map and dn_id not in visited_in_chain:
            visited_in_chain.add(dn_id)
            chain_geometries.append(temp_connector_map[dn_id]["line_geometry"])
            dn_id = temp_connector_map[dn_id]["dn_id"]

        # dn_id is now the ultimate downstream (not a connector) or None (terminal)
        if dn_id:
            connector_to_downstream[connector_id] = dn_id
            connector_line_geoms[connector_id] = chain_geometries
            connector_to_upstreams[connector_id] = connector_data["up_ids"]
        else:
            logger.warning(f"No-divide connector {connector_id} has no downstream target (terminal segment)")
            connector_to_downstream[connector_id] = None
            connector_line_geoms[connector_id] = chain_geometries
            connector_to_upstreams[connector_id] = connector_data["up_ids"]

    return connector_to_downstream, connector_line_geoms, connector_to_upstreams


def _find_downstream_unit(
    outlet_fp: dict[str, Any],
    fp_by_hydroseq: dict[Any, dict[str, Any]],
    connector_to_downstream: dict[str, str | None],
    ref_id_to_new_id: dict[str, int],
) -> int | None:
    """Find the downstream unit ID for a given outlet flowpath.

    Parameters
    ----------
    outlet_fp : dict[str, Any]
        The outlet flowpath dictionary with dnhydroseq attribute
    fp_by_hydroseq : dict[Any, dict[str, Any]]
        Mapping of hydroseq values to flowpath dictionaries
    connector_to_downstream : dict[str, str | None]
        Mapping of connector IDs to their ultimate downstream targets
    ref_id_to_new_id : dict[str, int]
        Mapping of reference IDs to new hydrofabric IDs

    Returns
    -------
    int | None
        Downstream unit ID if found, else None
    """
    dn_hydroseq = outlet_fp.get("dnhydroseq")

    if dn_hydroseq is None or dn_hydroseq == 0:
        return None

    downstream_fp = fp_by_hydroseq.get(dn_hydroseq)

    if not downstream_fp:
        return None

    downstream_ref_id = str(downstream_fp["flowpath_id"])

    # Follow connector chain to find ultimate downstream
    while downstream_ref_id in connector_to_downstream:
        next_dn = connector_to_downstream[downstream_ref_id]
        if next_dn:
            downstream_ref_id = next_dn
        else:
            break

    return ref_id_to_new_id.get(downstream_ref_id)


def _create_nexus_point(
    merged_line_geom: BaseGeometry,
    nexus_counter: int,
    downstream_unit_id: int | None,
) -> dict[str, Any]:
    """Create a nexus point at the downstream end of a flowpath.

    Parameters
    ----------
    merged_line_geom : BaseGeometry
        The merged line geometry (LineString or MultiLineString)
    nexus_counter : int
        The nexus ID counter
    downstream_unit_id : int | None
        The downstream flowpath ID, or None if terminal

    Returns
    -------
    dict[str, Any]
        Nexus data dictionary with nex_id, downstream_fp_id, and geometry
    """
    # Use merged geometry for nexus point
    if merged_line_geom.geom_type == "MultiLineString":
        last_line = list(merged_line_geom.geoms)[-1]
        end_coordinate = last_line.coords[-1]
    else:
        end_coordinate = merged_line_geom.coords[-1]

    return {
        "nex_id": nexus_counter,
        "downstream_fp_id": downstream_unit_id,
        "geometry": Point(end_coordinate),
    }


def _process_unit(
    current_ref_id: str,
    unit_info: dict[str, Any],
    fp_lookup: dict[str, dict[str, Any]],
    fp_by_hydroseq: dict[Any, dict[str, Any]],
    connector_to_downstream: dict[str, str | None],
    connector_line_geoms: dict[str, list[BaseGeometry]],
    ref_id_to_new_id: dict[str, int],
    downstream_fp_to_nexus: dict[int, int],
    new_id: int,
    nexus_counter: int,
    fp_data: list[dict[str, Any]],
    div_data: list[dict[str, Any]],
    nexus_data: list[dict[str, Any]],
    base_minor_data: list[dict[str, Any]],
) -> tuple[int, int]:
    """Process a single unit and create flowpath/divide/nexus.

    Parameters
    ----------
    current_ref_id : str
        The current reference ID being processed
    unit_info : dict[str, Any]
        Unit information including type, unit data, and ref_ids
    fp_lookup : dict[str, dict[str, Any]]
        Flowpath lookup dictionary
    fp_by_hydroseq : dict[Any, dict[str, Any]]
        Flowpath lookup by hydroseq
    connector_to_downstream : dict[str, str | None]
        Connector to downstream mapping
    connector_line_geoms : dict[str, list[BaseGeometry]]
        Connector line geometries
    ref_id_to_new_id : dict[str, int]
        Reference ID to new ID mapping
    downstream_fp_to_nexus : dict[int, int]
        Downstream flowpath to nexus mapping
    new_id : int
        Current new ID counter
    nexus_counter : int
        Current nexus ID counter
    fp_data : list[dict[str, Any]]
        Flowpath data accumulator
    div_data : list[dict[str, Any]]
        Divide data accumulator
    nexus_data : list[dict[str, Any]]
        Nexus data accumulator
    base_minor_data : list[dict[str, Any]]
        Base minor flowpath data accumulator

    Returns
    -------
    tuple[int, int]
        Updated (new_id, nexus_counter) after processing
    """
    unit: dict[str, Any] = unit_info["unit"]
    unit_type: str = unit_info["type"]
    ref_ids: list[str] = list(unit_info["all_ref_ids"])

    # Check if any upstream connectors should be patched into this unit
    line_geoms_to_merge: list[BaseGeometry] = [unit["line_geometry"]]
    upstream_connector_ids: list[str] = []

    for connector_id, dn_id in connector_to_downstream.items():
        if dn_id == current_ref_id:
            line_geoms_to_merge.extend(connector_line_geoms[connector_id])
            upstream_connector_ids.append(connector_id)

            if len(connector_line_geoms[connector_id]) > 1:
                logger.debug(
                    f"Patching connector chain (length={len(connector_line_geoms[connector_id])}) "
                    f"ending with {connector_id} into {current_ref_id}"
                )
            else:
                logger.debug(f"Patching no-divide connector {connector_id} into {current_ref_id}")

    merged_line_geom: BaseGeometry = (
        unary_union(line_geoms_to_merge) if len(line_geoms_to_merge) > 1 else unit["line_geometry"]
    )

    # Get original flowpaths
    original_fps: list[dict[str, Any]] = [fp_lookup[ref_id] for ref_id in ref_ids if ref_id in fp_lookup]

    if not original_fps:
        logger.warning(f"No flowpaths found for unit {ref_ids}")
        return new_id, nexus_counter

    # Find outlet flowpath
    outlet_fp: dict[str, Any]
    if unit_type == "aggregate" and unit_info.get("dn_id") is not None:
        dn_id_str: str = unit_info["dn_id"]
        outlet_fp_temp = fp_lookup.get(dn_id_str)

        if outlet_fp_temp is None:
            outlet_fp = min(original_fps, key=lambda x: x["hydroseq"])
        else:
            outlet_fp = outlet_fp_temp
    else:
        if len(original_fps) == 1:
            outlet_fp = original_fps[0]
        else:
            outlet_fp = min(original_fps, key=lambda x: x["hydroseq"])

    # Find downstream unit
    downstream_unit_id = _find_downstream_unit(
        outlet_fp, fp_by_hydroseq, connector_to_downstream, ref_id_to_new_id
    )

    # Get or create nexus
    nexus_id: int
    if downstream_unit_id is not None and downstream_unit_id in downstream_fp_to_nexus:
        nexus_id = downstream_fp_to_nexus[downstream_unit_id]
    else:
        nexus_id = nexus_counter
        nexus_counter += 1

        nexus = _create_nexus_point(merged_line_geom, nexus_id, downstream_unit_id)
        nexus_data.append(nexus)

        if downstream_unit_id is not None:
            downstream_fp_to_nexus[downstream_unit_id] = nexus_id

    # Check for divide polygon
    polygon_geom: BaseGeometry | None = unit.get("polygon_geometry")
    if polygon_geom is None or polygon_geom.is_empty:
        logger.debug(f"Skipping flowpath creation for unit {ref_ids} - no divide polygon")

        # Register connector IDs
        for connector_id in upstream_connector_ids:
            ref_id_to_new_id[connector_id] = new_id

        ref_id_to_new_id[current_ref_id] = new_id
        return new_id, nexus_counter

    # Create flowpath and divide
    fp_entry: dict[str, Any] = {
        "fp_id": new_id,
        "dn_nex_id": nexus_id,
        "up_nex_id": None,
        "div_id": new_id,
        "geometry": merged_line_geom,
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
    div_data.append({"div_id": new_id, "type": unit_type, "geometry": polygon_geom})

    # Register connector IDs
    for connector_id in upstream_connector_ids:
        ref_id_to_new_id[connector_id] = new_id

    new_id += 1

    return new_id, nexus_counter


def _build_base_hydrofabric(
    start_id: str,
    aggregate_data: Aggregations,
    classifications: Classifications,
    partition_data: dict[str, Any],
    cfg: HFConfig,
    id_offset: int = 0,
) -> dict[str, gpd.GeoDataFrame | list[dict[str, Any]] | None]:
    """Build the base hydrofabric layers with no-divide connector patching using dictionary lookups.

    No-divide connectors are flowpaths without catchment polygons that connect
    upstream to downstream segments. We "patch" them by:
    1. Merging their line geometry into the downstream flowpath
    2. Redirecting upstream connections through them
    3. Not creating separate units for them

    Parameters
    ----------
    start_id : str
        The outlet flowpath ID
    aggregate_data : Aggregations
        Aggregation data
    classifications : Classifications
        Classification data
    partition_data : dict[str, Any]
        Contains fp_lookup, div_lookup, subgraph, node_indices, and DataFrames
    cfg : HFConfig
        Hydrofabric build config
    id_offset : int, optional
        Starting ID offset for this outlet, by default 0

    Returns
    -------
    dict[str, gpd.GeoDataFrame | list[dict[str, Any]] | None]
        Built hydrofabric data with keys: flowpaths, divides, nexus, base_minor_flowpaths
    """
    # Extract from partition_data
    fp_lookup: dict[str, dict[str, Any]] = partition_data["fp_lookup"]
    graph = partition_data["subgraph"]
    node_indices: dict[str, int] = partition_data["node_indices"]

    # Build hydroseq lookup once for O(1) downstream lookups
    fp_by_hydroseq: dict[Any, dict[str, Any]] = {row["hydroseq"]: row for row in fp_lookup.values()}

    ref_id_to_unit = _order_aggregates_base(aggregate_data)

    # Prepare connector mappings
    connector_to_downstream, connector_line_geoms, connector_to_upstreams = _prepare_connector_mappings(
        aggregate_data
    )

    # Initialize processing state
    to_process: deque[str] = deque([start_id])
    visited_ref_ids: set[str] = set()
    processed_units: set[frozenset[str]] = set()
    fp_data: list[dict[str, Any]] = []
    div_data: list[dict[str, Any]] = []
    nexus_data: list[dict[str, Any]] = []
    base_minor_data: list[dict[str, Any]] = []
    ref_id_to_new_id: dict[str, int] = {}
    downstream_fp_to_nexus: dict[int, int] = {}

    new_id = 1 + id_offset
    nexus_counter = 1 + id_offset

    # Main processing loop
    while to_process:
        current_ref_id = to_process.popleft()

        if current_ref_id in visited_ref_ids:
            continue

        # Skip no-divide connectors - they'll be patched into downstream
        if current_ref_id in connector_to_downstream:
            visited_ref_ids.add(current_ref_id)
            downstream_id = connector_to_downstream[current_ref_id]
            if downstream_id and downstream_id not in visited_ref_ids:
                to_process.append(downstream_id)
            for up_id in connector_to_upstreams.get(current_ref_id, []):
                if up_id not in visited_ref_ids:
                    to_process.append(up_id)
            continue

        # Validate current flowpath
        if current_ref_id not in ref_id_to_unit:
            if not fp_data:
                visited_ref_ids.add(current_ref_id)
                continue
            elif current_ref_id in classifications.minor_flowpaths:
                visited_ref_ids.add(current_ref_id)
                continue
            else:
                if current_ref_id in list(classifications.no_divide_connectors):
                    logger.warning(
                        f"No-divide connector {current_ref_id} was in classifications but not in "
                        f"aggregate_data. Skipping."
                    )
                    visited_ref_ids.add(current_ref_id)
                    continue

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

        # Process the unit
        new_id, nexus_counter = _process_unit(
            current_ref_id,
            unit_info,
            fp_lookup,
            fp_by_hydroseq,
            connector_to_downstream,
            connector_line_geoms,
            ref_id_to_new_id,
            downstream_fp_to_nexus,
            new_id,
            nexus_counter,
            fp_data,
            div_data,
            nexus_data,
            base_minor_data,
        )

        # Queue upstream segments
        lookup_id: str
        if unit_info["type"] == "aggregate":
            lookup_id = unit_info["up_id"]
        else:
            lookup_id = current_ref_id

        if lookup_id in node_indices:
            lookup_idx = node_indices[lookup_id]
            upstream_ids = [graph[idx] for idx in graph.predecessor_indices(lookup_idx)]
            for upstream_id in upstream_ids:
                to_process.append(upstream_id)

    # Fix up_nex_id references
    nexus_by_downstream_fp: dict[int, int] = {
        nexus["downstream_fp_id"]: nexus["nex_id"]
        for nexus in nexus_data
        if nexus.get("downstream_fp_id") is not None
    }
    for fp_entry in fp_data:
        my_fp_id: int = fp_entry["fp_id"]
        if my_fp_id in nexus_by_downstream_fp:
            fp_entry["up_nex_id"] = nexus_by_downstream_fp[my_fp_id]

    # Create GeoDataFrames
    flowpaths_gdf: gpd.GeoDataFrame | None
    divides_gdf: gpd.GeoDataFrame | None
    nexus_gdf: gpd.GeoDataFrame | None
    try:
        flowpaths_gdf = gpd.GeoDataFrame(fp_data, crs=cfg.crs)
        divides_gdf = gpd.GeoDataFrame(div_data, crs=cfg.crs)
        nexus_gdf = gpd.GeoDataFrame(nexus_data, crs=cfg.crs)
    except ValueError:
        flowpaths_gdf = None
        divides_gdf = None
        nexus_gdf = None

    return {
        "flowpaths": flowpaths_gdf,
        "divides": divides_gdf,
        "nexus": nexus_gdf,
        "base_minor_flowpaths": base_minor_data,
    }

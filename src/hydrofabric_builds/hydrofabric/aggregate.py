"""Geometry aggregation module for hydrofabric builds."""

import logging
from typing import Any

from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from hydrofabric_builds.schemas.hydrofabric import Aggregations, Classifications

logger = logging.getLogger(__name__)


def _merge_tuples_with_common_values(tuples_list: list[tuple[str, ...]]) -> list[list[str]]:
    """Merge tuples that share any common values using a Union-Find algorithm.

    Parameters
    ----------
    tuples_list : list[tuple[str, ...]]
        List of tuples, each containing values

    Returns
    -------
    list[list[str]]
        Each list contains all connected values
    """
    if not tuples_list:
        return []

    parent: dict[str, str] = {}

    def find(x: str) -> str:
        if x not in parent:
            parent[x] = x
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x: str, y: str) -> None:
        root_x = find(x)
        root_y = find(y)
        if root_x != root_y:
            parent[root_x] = root_y

    for tup in tuples_list:
        if len(tup) > 0:
            first = tup[0]
            for value in tup[1:]:
                union(first, value)

    groups: dict[str, set[str]] = {}
    for tup in tuples_list:
        for value in tup:
            root = find(value)
            if root not in groups:
                groups[root] = set()
            groups[root].add(value)

    return [list(group) for group in groups.values()]


def _process_aggregation_pairs(
    classifications: Classifications,
    fp_lookup: dict[str, dict[str, Any]],
    div_lookup: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Process aggregation pairs using dictionary lookups.

    Parameters
    ----------
    classifications : Classifications
        Classification results
    fp_lookup : dict[str, dict[str, Any]]
        Flowpath lookup dict with attributes and shapely_geometry
    div_lookup : dict[str, dict[str, Any]]
        Divide lookup dict with shapely_geometry

    Returns
    -------
    list[dict[str, Any]]
        Processed aggregation data
    """
    groups = _merge_tuples_with_common_values(classifications.aggregation_pairs)

    results: list[dict[str, Any]] = []
    virtual_flowpaths_set: set[str] = set(classifications.virtual_flowpaths)

    for group_ids in groups:
        # Filter out virtual flowpaths
        fp_ids = [fp_id for fp_id in group_ids if fp_id not in virtual_flowpaths_set]

        if not fp_ids:
            logger.debug(f"Skipping group {group_ids} - all flowpaths are virtual")
            continue

        try:
            # Get flowpath data from lookup dict
            fp_data = [fp_lookup[fp_id] for fp_id in fp_ids if fp_id in fp_lookup]
            div_data = [div_lookup[fp_id] for fp_id in fp_ids if fp_id in div_lookup]

            if not fp_data:
                logger.debug(f"Cannot find flowpaths for {group_ids}")
                continue

            # Sort by hydroseq
            sorted_fps_asc = sorted(fp_data, key=lambda x: x["hydroseq"])
            sorted_fps_desc = sorted(fp_data, key=lambda x: x["hydroseq"], reverse=True)

            # Extract IDs in sorted order
            sorted_ids_asc = [str(fp["flowpath_id"]) for fp in sorted_fps_asc]
            fp_geometry_ids = [str(fp["flowpath_id"]) for fp in sorted_fps_desc]

            # Compute aggregates
            length_km = sum(float(fp["lengthkm"]) for fp in fp_data)
            area_sqkm = sum(float(fp["areasqkm"]) for fp in fp_data)
            hydroseq = max(int(fp["hydroseq"]) for fp in fp_data)
            div_area_sqkm = sum(float(div["areasqkm"]) for div in div_data)
            vpu_id = fp_data[0]["VPUID"]

            # Get geometries from lookup dicts
            line_geoms: list[BaseGeometry] = [
                fp_lookup[fp_id]["shapely_geometry"] for fp_id in fp_geometry_ids if fp_id in fp_lookup
            ]
            polygon_geoms: list[BaseGeometry] = [
                div_lookup[fp_id]["shapely_geometry"] for fp_id in group_ids if fp_id in div_lookup
            ]

            results.append(
                {
                    "ref_ids": fp_ids,
                    "dn_id": sorted_ids_asc[0],
                    "up_id": sorted_ids_asc[-1],
                    "vpu_id": vpu_id,
                    "hydroseq": hydroseq,
                    "length_km": length_km,
                    "area_sqkm": area_sqkm,
                    "div_area_sqkm": div_area_sqkm,
                    "line_geometry": unary_union(line_geoms),
                    "polygon_geometry": unary_union(polygon_geoms) if polygon_geoms else None,
                }
            )

        except KeyError:
            logger.debug(f"Missing flowpath / divide path data for fp_ids {fp_ids}")
            continue
    return results


def _process_independent_flowpaths(
    classifications: Classifications,
    fp_lookup: dict[str, dict[str, Any]],
    div_lookup: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Process independent flowpaths using dictionary lookups.

    Parameters
    ----------
    classifications : Classifications
        Classification results
    fp_lookup : dict[str, dict[str, Any]]
        Flowpath lookup dict with shapely_geometry
    div_lookup : dict[str, dict[str, Any]]
        Divide lookup dict with shapely_geometry

    Returns
    -------
    list[dict[str, Any]]
        Independent flowpath data
    """
    results: list[dict[str, Any]] = []
    for fp in classifications.independent_flowpaths:
        try:
            fp_data = fp_lookup[fp]
            div_data = div_lookup[fp]

            length_km = float(fp_data["lengthkm"])
            area_sqkm = float(fp_data["areasqkm"])
            hydroseq = int(fp_data["hydroseq"])
            div_area_sqkm = float(div_data["areasqkm"])
            vpu_id = fp_data["VPUID"]
            results.append(
                {
                    "ref_ids": fp,
                    "vpu_id": vpu_id,
                    "hydroseq": hydroseq,
                    "length_km": length_km,
                    "area_sqkm": area_sqkm,
                    "div_area_sqkm": div_area_sqkm,
                    "line_geometry": fp_lookup[fp]["shapely_geometry"],
                    "polygon_geometry": div_lookup[fp]["shapely_geometry"] if fp in div_lookup else None,
                }
            )
        except KeyError:
            logger.debug(f"Missing flowpath / divide path data for fp_id {fp}")
            continue

    return results


def _process_virtual_flowpaths(
    classifications: Classifications,
    fp_lookup: dict[str, dict[str, Any]],
    div_lookup: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Process virtual flowpaths using dictionary lookups.

    Parameters
    ----------
    classifications : Classifications
        Classification results
    fp_lookup : dict[str, dict[str, Any]]
        Flowpath lookup dict with shapely_geometry
    div_lookup : dict[str, dict[str, Any]]
        Divide lookup dict with shapely_geometry

    Returns
    -------
    list[dict[str, Any]]
        virtual flowpath data
    """
    results: list[dict[str, Any]] = []
    for fp in classifications.virtual_flowpaths:
        if fp in fp_lookup:
            results.append(
                {
                    "ref_ids": fp,
                    "line_geometry": fp_lookup[fp]["shapely_geometry"],
                    "polygon_geometry": div_lookup[fp]["shapely_geometry"] if fp in div_lookup else None,
                }
            )

    return results


def _process_small_scale_connectors(
    classifications: Classifications,
    fp_lookup: dict[str, dict[str, Any]],
    div_lookup: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Process small scale connectors using dictionary lookups.

    Parameters
    ----------
    classifications : Classifications
        Classification results
    fp_lookup : dict[str, dict[str, Any]]
        Flowpath lookup dict with shapely_geometry
    div_lookup : dict[str, dict[str, Any]]
        Divide lookup dict with shapely_geometry

    Returns
    -------
    list[dict[str, Any]]
        Small scale connector data
    """
    results: list[dict[str, Any]] = []
    for fp in classifications.subdivide_candidates:
        try:
            fp_data = fp_lookup[fp]
            div_data = div_lookup[fp]

            length_km = float(fp_data["lengthkm"])
            area_sqkm = float(fp_data["areasqkm"])
            hydroseq = int(fp_data["hydroseq"])
            div_area_sqkm = float(div_data["areasqkm"])
            vpu_id = fp_data["VPUID"]
            results.append(
                {
                    "ref_ids": fp,
                    "vpu_id": vpu_id,
                    "hydroseq": hydroseq,
                    "length_km": length_km,
                    "area_sqkm": area_sqkm,
                    "div_area_sqkm": div_area_sqkm,
                    "line_geometry": fp_lookup[fp]["shapely_geometry"],
                    "polygon_geometry": div_lookup[fp]["shapely_geometry"] if fp in div_lookup else None,
                }
            )
        except KeyError:
            logger.debug(f"Missing flowpath / divide path data for fp_id {fp}")
            continue

    return results


def _process_connectors(
    classifications: Classifications,
    fp_lookup: dict[str, dict[str, Any]],
    div_lookup: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Process connectors using dictionary lookups.

    Parameters
    ----------
    classifications : Classifications
        Classification results
    fp_lookup : dict[str, dict[str, Any]]
        Flowpath lookup dict with shapely_geometry
    div_lookup : dict[str, dict[str, Any]]
        Divide lookup dict with shapely_geometry

    Returns
    -------
    list[dict[str, Any]]
        Connector data
    """
    results: list[dict[str, Any]] = []
    for fp in classifications.connector_segments:
        try:
            fp_data = fp_lookup[fp]
            div_data = div_lookup[fp]

            length_km = float(fp_data["lengthkm"])
            area_sqkm = float(fp_data["areasqkm"])
            hydroseq = int(fp_data["hydroseq"])
            div_area_sqkm = float(div_data["areasqkm"])
            vpu_id = fp_data["VPUID"]
            results.append(
                {
                    "ref_ids": fp,
                    "vpu_id": vpu_id,
                    "hydroseq": hydroseq,
                    "length_km": length_km,
                    "area_sqkm": area_sqkm,
                    "div_area_sqkm": div_area_sqkm,
                    "line_geometry": fp_lookup[fp]["shapely_geometry"],
                    "polygon_geometry": div_lookup[fp]["shapely_geometry"] if fp in div_lookup else None,
                }
            )
        except KeyError:
            logger.debug(f"Missing flowpath / divide path data for fp_id {fp}")
            continue
    return results


def _aggregate_geometries(
    classifications: Classifications,
    partition_data: dict[str, Any],
) -> Aggregations:
    """Aggregate geometries using dictionary lookups.

    Parameters
    ----------
    classifications : Classifications
        Classification results
    partition_data : dict[str, Any]
        Contains fp_lookup and div_lookup with shapely geometries

    Returns
    -------
    Aggregations
        Aggregated geometry data
    """
    fp_lookup: dict[str, dict[str, Any]] = partition_data["fp_lookup"]
    div_lookup: dict[str, dict[str, Any]] = partition_data["div_lookup"]

    aggregates = _process_aggregation_pairs(classifications, fp_lookup, div_lookup)

    independents = _process_independent_flowpaths(classifications, fp_lookup, div_lookup)

    small_scale_connectors = _process_small_scale_connectors(classifications, fp_lookup, div_lookup)

    virtual_flowpaths = _process_virtual_flowpaths(classifications, fp_lookup, div_lookup)

    connectors = _process_connectors(classifications, fp_lookup, div_lookup)

    return Aggregations(
        aggregates=aggregates,
        independents=independents,
        virtual_flowpaths=virtual_flowpaths,
        small_scale_connectors=small_scale_connectors,
        connectors=connectors,
    )

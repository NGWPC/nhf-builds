"""Geometry aggregation module for hydrofabric builds."""

import logging

import geopandas as gpd
import pandas as pd
from shapely.ops import unary_union

from hydrofabric_builds.schemas.hydrofabric import Aggregations, Classifications

logger = logging.getLogger(__name__)


def _prepare_dataframes(
    _flowpaths_gdf: gpd.GeoDataFrame, _divides_gdf: gpd.GeoDataFrame
) -> tuple[gpd.GeoDataFrame, dict, dict]:
    flowpaths_gdf = _flowpaths_gdf.copy()
    flowpaths_gdf["flowpath_id"] = flowpaths_gdf["flowpath_id"].astype(str).str.replace(".0", "", regex=False)

    divides_gdf = _divides_gdf.copy()
    divides_gdf["divide_id"] = divides_gdf["divide_id"].astype(str).str.replace(".0", "", regex=False)

    fp_geom_lookup = dict(
        zip(flowpaths_gdf["flowpath_id"].astype(str), flowpaths_gdf["geometry"], strict=False)
    )

    div_geom_lookup = dict(zip(divides_gdf["divide_id"].astype(str), divides_gdf["geometry"], strict=False))

    return flowpaths_gdf, fp_geom_lookup, div_geom_lookup


def _merge_tuples_with_common_values(tuples_list: list[tuple[str, ...]]) -> list[list[str]]:
    """Merge tuples that share any common values using a Union-Find algorithm.

    Parameters
    ----------
    tuples_list: list[tuple[str, ...]]
        List of tuples, each containing values

    Returns
    -------
    list[list[str]]
        each list contains all connected values
    """
    if not tuples_list:
        return []

    parent = {}

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

    groups: dict[str, set] = {}
    for tup in tuples_list:
        for value in tup:
            root = find(value)
            if root not in groups:
                groups[root] = set()
            groups[root].add(value)

    return [list(group) for group in groups.values()]


def _process_aggregation_pairs(
    classifications: Classifications,
    flowpaths_gdf: gpd.GeoDataFrame,
    fp_geom_lookup: dict,
    div_geom_lookup: dict,
) -> list[dict]:
    groups = _merge_tuples_with_common_values(classifications.aggregation_pairs)

    results = []
    for group_ids in groups:
        group_fps: pd.Series = flowpaths_gdf[flowpaths_gdf["flowpath_id"].isin(group_ids)]
        if group_fps.empty:
            logger.warning(f"Cannot find flowpaths for {group_fps}")
            continue

        fp_ids = group_fps.copy()
        fp_ids = fp_ids[~fp_ids["flowpath_id"].isin(classifications.minor_flowpaths)]
        sorted_fps = fp_ids.sort_values("hydroseq")
        fp_geometry_ids = fp_ids.sort_values("hydroseq", ascending=False)["flowpath_id"].tolist()
        sorted_ids = sorted_fps["flowpath_id"].tolist()

        if len(sorted_ids) >= 2:
            last_order = sorted_fps.iloc[-1]["streamorder"]
            second_last_order = sorted_fps.iloc[-2]["streamorder"]

            # If last is order 1 but second-to-last is not, swap them to make sure that the upstream-ID is connected to a river network
            if last_order == 1 and second_last_order != 1:
                sorted_ids[-1], sorted_ids[-2] = sorted_ids[-2], sorted_ids[-1]

        results.append(
            {
                "ref_ids": group_ids,
                "dn_id": sorted_ids[0],
                "up_id": sorted_ids[-1],
                "vpu_id": fp_ids.iloc[0]["VPUID"],
                "length_km": fp_ids["lengthkm"].sum().item(),
                "total_da_sqkm": fp_ids["totdasqkm"].sum().item(),
                "line_geometry": unary_union(
                    [fp_geom_lookup[id] for id in fp_geometry_ids if id in fp_geom_lookup]
                ),
                "polygon_geometry": unary_union(
                    [div_geom_lookup[id] for id in group_ids if id in div_geom_lookup]
                ),
            }
        )

    return results


def _process_independent_flowpaths(
    classifications: Classifications, fp_geom_lookup: dict, div_geom_lookup: dict
) -> list[dict]:
    """Process independent flowpaths (no aggregation needed).

    Parameters
    ----------
    classifications : Classifications
        Classification results
    flowpaths_gdf : gpd.GeoDataFrame
        Reference flowpaths
    divides_gdf : gpd.GeoDataFrame
        Reference divides
    fp_geom_lookup : dict
        Flowpath geometry lookup
    div_geom_lookup : dict
        Divide geometry lookup

    Returns
    -------
    tuple[list[dict], list[dict]]
        Independent flowpaths and divides
    """
    results = []
    for fp in classifications.independent_flowpaths:
        results.append(
            {"ref_ids": fp, "line_geometry": fp_geom_lookup[fp], "polygon_geometry": div_geom_lookup[fp]}
        )

    return results


def _process_minor_flowpaths(
    classifications: Classifications, fp_geom_lookup: dict, div_geom_lookup: dict
) -> list[dict]:
    results = []
    for fp in classifications.minor_flowpaths:
        results.append(
            {"ref_ids": fp, "line_geometry": fp_geom_lookup[fp], "polygon_geometry": div_geom_lookup[fp]}
        )

    return results


def _process_small_scale_connectors(
    classifications: Classifications, fp_geom_lookup: dict, div_geom_lookup: dict
) -> list[dict]:
    results = []
    for fp in classifications.subdivide_candidates:
        results.append(
            {"ref_ids": fp, "line_geometry": fp_geom_lookup[fp], "polygon_geometry": div_geom_lookup[fp]}
        )

    return results


def _process_connectors(
    classifications: Classifications, fp_geom_lookup: dict, div_geom_lookup: dict
) -> list[dict]:
    results = []
    for fp in classifications.connector_segments:
        results.append(
            {"ref_ids": fp, "line_geometry": fp_geom_lookup[fp], "polygon_geometry": div_geom_lookup[fp]}
        )

    return results


def _aggregate_geometries(
    classifications: Classifications,
    reference_divides: pd.DataFrame,
    reference_flowpaths: pd.DataFrame,
) -> Aggregations:
    flowpaths_gdf, fp_geom_lookup, div_geom_lookup = _prepare_dataframes(
        reference_flowpaths, reference_divides
    )

    aggregates = _process_aggregation_pairs(classifications, flowpaths_gdf, fp_geom_lookup, div_geom_lookup)

    independents = _process_independent_flowpaths(classifications, fp_geom_lookup, div_geom_lookup)

    small_scale_connectors = _process_small_scale_connectors(classifications, fp_geom_lookup, div_geom_lookup)

    minor_flowpaths = _process_minor_flowpaths(classifications, fp_geom_lookup, div_geom_lookup)

    connectors = _process_connectors(classifications, fp_geom_lookup, div_geom_lookup)

    return Aggregations(
        aggregates=aggregates,
        independents=independents,
        minor_flowpaths=minor_flowpaths,
        small_scale_connectors=small_scale_connectors,
        connectors=connectors,
    )

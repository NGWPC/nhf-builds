"""Geometry aggregation module for hydrofabric builds."""

import logging
from typing import Any

import geopandas as gpd
import polars as pl
from shapely.ops import unary_union

from hydrofabric_builds.schemas.hydrofabric import Aggregations, Classifications

logger = logging.getLogger(__name__)


def _prepare_dataframes(
    flowpaths_df: pl.DataFrame, divides_df: pl.DataFrame
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Prepare shapely geometry lookup dictionaries from Polars DataFrames.

    Converts WKB geometries to shapely objects once for reuse in all aggregations.
    This is much faster than converting WKB to shapely repeatedly in loops.

    Parameters
    ----------
    flowpaths_df : pl.DataFrame
        Flowpaths with WKB geometry column
    divides_df : pl.DataFrame
        Divides with WKB geometry column

    Returns
    -------
    tuple[dict[str, Any], dict[str, Any]]
        Shapely geometry lookup dicts for flowpaths and divides
    """
    fp_ids = flowpaths_df["flowpath_id"].cast(pl.Utf8).to_list()
    fp_shapes = gpd.GeoSeries.from_wkb(flowpaths_df["geometry"])
    fp_geom_lookup = dict(zip(fp_ids, fp_shapes, strict=False))

    div_ids = divides_df["divide_id"].cast(pl.Utf8).to_list()
    div_shapes = gpd.GeoSeries.from_wkb(divides_df["geometry"].to_list())
    div_geom_lookup = dict(zip(div_ids, div_shapes, strict=False))

    return fp_geom_lookup, div_geom_lookup


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
    flowpaths_df: pl.DataFrame,
    fp_geom_lookup: dict[str, Any],
    div_geom_lookup: dict[str, Any],
) -> list[dict]:
    """Process aggregation pairs

    Parameters
    ----------
    classifications : Classifications
        Classification results
    flowpaths_df : pl.DataFrame
        Reference flowpaths with WKB geometries
    fp_geom_lookup : dict[str, Any]
        Flowpath shapely geometry lookup
    div_geom_lookup : dict[str, Any]
        Divide shapely geometry lookup

    Returns
    -------
    list[dict]
        Processed aggregation data
    """
    groups = _merge_tuples_with_common_values(classifications.aggregation_pairs)

    results = []
    for group_ids in groups:
        group_fps = flowpaths_df.filter(pl.col("flowpath_id").cast(pl.Utf8).is_in(group_ids))
        if group_fps.height == 0:
            logger.warning(f"Cannot find flowpaths for {group_ids}")
            continue

        # Filter out minor flowpaths
        fp_ids = group_fps.filter(~pl.col("flowpath_id").cast(pl.Utf8).is_in(classifications.minor_flowpaths))

        # Sort by hydroseq
        sorted_fps = fp_ids.sort("hydroseq")
        sorted_fps_desc = fp_ids.sort("hydroseq", descending=True)

        # Get IDs in sorted order
        fp_geometry_ids = sorted_fps_desc["flowpath_id"].cast(pl.Utf8).to_list()
        sorted_ids = sorted_fps["flowpath_id"].cast(pl.Utf8).to_list()

        length_km = fp_ids["lengthkm"].sum()
        total_da_sqkm = fp_ids["totdasqkm"].sum()
        vpu_id = fp_ids["VPUID"][0]

        line_geoms = [fp_geom_lookup[id] for id in fp_geometry_ids if id in fp_geom_lookup]
        polygon_geoms = [div_geom_lookup[id] for id in group_ids if id in div_geom_lookup]

        results.append(
            {
                "ref_ids": group_ids,
                "dn_id": sorted_ids[0],
                "up_id": sorted_ids[-1],
                "vpu_id": vpu_id,
                "length_km": length_km,
                "total_da_sqkm": total_da_sqkm,
                "line_geometry": unary_union(line_geoms),
                "polygon_geometry": unary_union(polygon_geoms) if polygon_geoms else None,
            }
        )

    return results


def _process_no_divide_connectors(
    classifications: Classifications,
    flowpaths_df: pl.DataFrame,
    fp_geom_lookup: dict[str, Any],
    div_geom_lookup: dict[str, Any],
) -> list[dict]:
    """Process no-divide connectors

    Parameters
    ----------
    classifications : Classifications
        Classification results
    flowpaths_df : pl.DataFrame
        Reference flowpaths with WKB geometries
    fp_geom_lookup : dict[str, Any]
        Flowpath shapely geometry lookup
    div_geom_lookup : dict[str, Any]
        Divide shapely geometry lookup

    Returns
    -------
    list[dict]
        Processed connector data
    """
    results = []

    for fp in classifications.no_divide_connectors:
        # Find upstream flowpaths
        fp_float = float(fp)
        upstream_ids = (
            flowpaths_df.filter(pl.col("flowpath_toid") == fp_float)
            .select("flowpath_id")
            .cast(pl.Utf8)
            .to_series()
            .to_list()
        )

        # Find downstream ID
        downstream_row = flowpaths_df.filter(pl.col("flowpath_id").cast(pl.Utf8) == fp)
        if downstream_row.height > 0:
            downstream_id = str(int(downstream_row["flowpath_toid"][0]))

            results.append(
                {
                    "ref_ids": fp,
                    "dn_id": downstream_id,
                    "up_id": upstream_ids,
                    "line_geometry": fp_geom_lookup[fp],
                }
            )

    return results


def _process_independent_flowpaths(
    classifications: Classifications, fp_geom_lookup: dict[str, Any], div_geom_lookup: dict[str, Any]
) -> list[dict]:
    """Process independent flowpaths.

    Parameters
    ----------
    classifications : Classifications
        Classification results
    fp_geom_lookup : dict[str, Any]
        Flowpath shapely geometry lookup
    div_geom_lookup : dict[str, Any]
        Divide shapely geometry lookup

    Returns
    -------
    list[dict]
        Independent flowpath data
    """
    results = []
    for fp in classifications.independent_flowpaths:
        results.append(
            {
                "ref_ids": fp,
                "line_geometry": fp_geom_lookup[fp],
                "polygon_geometry": div_geom_lookup.get(fp, None),
            }
        )

    return results


def _process_minor_flowpaths(
    classifications: Classifications, fp_geom_lookup: dict[str, Any], div_geom_lookup: dict[str, Any]
) -> list[dict]:
    """Process minor flowpaths.

    Parameters
    ----------
    classifications : Classifications
        Classification results
    fp_geom_lookup : dict[str, Any]
        Flowpath shapely geometry lookup
    div_geom_lookup : dict[str, Any]
        Divide shapely geometry lookup

    Returns
    -------
    list[dict]
        Minor flowpath data
    """
    results = []
    for fp in classifications.minor_flowpaths:
        results.append(
            {
                "ref_ids": fp,
                "line_geometry": fp_geom_lookup[fp],
                "polygon_geometry": div_geom_lookup.get(fp, None),
            }
        )

    return results


def _process_small_scale_connectors(
    classifications: Classifications, fp_geom_lookup: dict[str, Any], div_geom_lookup: dict[str, Any]
) -> list[dict]:
    """Process small scale connectors.

    Parameters
    ----------
    classifications : Classifications
        Classification results
    fp_geom_lookup : dict[str, Any]
        Flowpath shapely geometry lookup
    div_geom_lookup : dict[str, Any]
        Divide shapely geometry lookup

    Returns
    -------
    list[dict]
        Small scale connector data
    """
    results = []
    for fp in classifications.subdivide_candidates:
        results.append(
            {
                "ref_ids": fp,
                "line_geometry": fp_geom_lookup[fp],
                "polygon_geometry": div_geom_lookup.get(fp, None),
            }
        )

    return results


def _process_connectors(
    classifications: Classifications, fp_geom_lookup: dict[str, Any], div_geom_lookup: dict[str, Any]
) -> list[dict]:
    """Process connectors.

    Parameters
    ----------
    classifications : Classifications
        Classification results
    fp_geom_lookup : dict[str, Any]
        Flowpath shapely geometry lookup
    div_geom_lookup : dict[str, Any]
        Divide shapely geometry lookup

    Returns
    -------
    list[dict]
        Connector data
    """
    results = []
    for fp in classifications.connector_segments:
        results.append(
            {
                "ref_ids": fp,
                "line_geometry": fp_geom_lookup[fp],
                "polygon_geometry": div_geom_lookup.get(fp, None),
            }
        )

    return results


def _aggregate_geometries(
    classifications: Classifications,
    reference_flowpaths: pl.DataFrame,
    fp_geom_lookup: dict[str, Any],
    div_geom_lookup: dict[str, Any],
) -> Aggregations:
    """Aggregate geometries

    Parameters
    ----------
    classifications : Classifications
        Classification results
    reference_flowpaths : pl.DataFrame
        Reference flowpaths with WKB geometries
    fp_geom_lookup : dict[str, Any]
        Flowpath shapely geometry lookup
    div_geom_lookup : dict[str, Any]
        Divide shapely geometry lookup

    Returns
    -------
    Aggregations
        Aggregated geometry data
    """
    aggregates = _process_aggregation_pairs(
        classifications, reference_flowpaths, fp_geom_lookup, div_geom_lookup
    )

    no_divide_connectors = _process_no_divide_connectors(
        classifications, reference_flowpaths, fp_geom_lookup, div_geom_lookup
    )

    independents = _process_independent_flowpaths(classifications, fp_geom_lookup, div_geom_lookup)

    small_scale_connectors = _process_small_scale_connectors(classifications, fp_geom_lookup, div_geom_lookup)

    minor_flowpaths = _process_minor_flowpaths(classifications, fp_geom_lookup, div_geom_lookup)

    connectors = _process_connectors(classifications, fp_geom_lookup, div_geom_lookup)

    return Aggregations(
        aggregates=aggregates,
        independents=independents,
        minor_flowpaths=minor_flowpaths,
        no_divide_connectors=no_divide_connectors,
        small_scale_connectors=small_scale_connectors,
        connectors=connectors,
    )

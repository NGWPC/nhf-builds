"""Contains all code for processing hydrofabric data"""

import logging
from typing import Any, cast

import geopandas as gpd
import pandas as pd
from tqdm import tqdm

from hydrofabric_builds.config import HFConfig
from hydrofabric_builds.hydrofabric.aggregate import _aggregate_geometries
from hydrofabric_builds.hydrofabric.build import _build_hydrofabric
from hydrofabric_builds.hydrofabric.trace import _trace_stack
from hydrofabric_builds.hydrofabric.utils import (
    _check_network_cycles,
    _combine_hydrofabrics,
)
from hydrofabric_builds.schemas.hydrofabric import Aggregations, Classifications
from hydrofabric_builds.task_instance import TaskInstance

logger = logging.getLogger(__name__)


def _process_single_outlet(
    outlet: str,
    partition_data: dict[str, Any],
    cfg: HFConfig,
) -> dict[str, Any]:
    """Process outlet with pre-partitioned subgraph and data.

    Parameters
    ----------
    outlet : str
        Outlet ID
    partition_data : dict[str, Any]
        Contains:
        - "subgraph": rx.PyDiGraph (minimal, only this outlet)
        - "node_indices": dict (for subgraph)
        - "fp_lookup": dict (flowpath attributes + shapely_geometry)
        - "div_lookup": dict (divide attributes + shapely_geometry)
        - "flowpaths": pl.DataFrame (for fallback operations)
        - "divides": pl.DataFrame (for fallback operations)
    cfg : HFConfig
        Config

    Returns
    -------
    dict[str, Any]
        Dictionary containing outlet, classifications, aggregate_data, and num_features
    """
    filtered_divides = partition_data["divides"]
    valid_divide_ids: set[str] = set(filtered_divides["divide_id"].to_list())

    # Trace with subgraph
    classifications = _trace_stack(
        start_id=outlet,
        div_ids=valid_divide_ids,
        cfg=cfg,
        partition_data=partition_data,
    )

    # Aggregate geometries
    aggregate_data = _aggregate_geometries(
        classifications=classifications,
        partition_data=partition_data,
    )

    return {
        "outlet": outlet,
        "classifications": classifications.model_dump(),
        "aggregate_data": aggregate_data.model_dump(),
        "num_features": len(partition_data["subgraph"].nodes()),
    }


def _build_single_hydrofabric(
    outlet: str,
    outlet_data: dict[str, Any],
    id_offset: int,
    partition_data: dict[str, Any],
    cfg: HFConfig,
) -> dict[str, Any]:
    """Build a single outlet's hydrofabric with pre-partitioned subgraph and data.

    Parameters
    ----------
    outlet : str
        Outlet ID
    outlet_data : dict[str, Any]
        Outlet aggregation data from map phase
    id_offset : int
        Starting ID for this outlet
    partition_data : dict[str, Any]
        Contains:
        - "subgraph": rx.PyDiGraph (minimal, only this outlet)
        - "node_indices": dict (for subgraph)
        - "fp_lookup": dict (flowpath attributes + shapely_geometry)
        - "div_lookup": dict (divide attributes + shapely_geometry)
        - "flowpaths": pl.DataFrame (for fallback operations)
        - "divides": pl.DataFrame (for fallback operations)
    cfg : HFConfig
        Hydrofabric build config

    Returns
    -------
    dict[str, Any]
        Built hydrofabric data for this outlet
    """
    classifications = Classifications(**outlet_data["classifications"])
    aggregate_data = Aggregations(**outlet_data["aggregate_data"])

    hydrofabric = _build_hydrofabric(
        start_id=outlet,
        aggregate_data=aggregate_data,
        classifications=classifications,
        partition_data=partition_data,
        cfg=cfg,
        id_offset=id_offset,
    )

    return {
        "outlet": outlet,
        "flowpaths": hydrofabric["flowpaths"],
        "divides": hydrofabric["divides"],
        "nexus": hydrofabric["nexus"],
        "reference_flowpaths": hydrofabric["reference_flowpaths"],
        "virtual_flowpaths": hydrofabric["virtual_flowpaths"],
        "virtual_nexus": hydrofabric["virtual_nexus"],
        # "reference_virtual_flowpaths": hydrofabric["reference_virtual_flowpaths"],
        "next_available_id": hydrofabric["next_available_id"],
    }


def map_trace_and_aggregate(**context: dict[str, Any]) -> dict[str, Any]:
    """Execute MAP PHASE: Trace and aggregate flowpaths using pre-partitioned subgraphs.

    This task processes each outlet independently using pre-partitioned subgraphs
    and filtered data from the build_graph task. No large DataFrames are broadcasted.

    Parameters
    ----------
    **context : dict[str, Any]
        Airflow context

    Returns
    -------
    dict[str, Any]
        Dictionary with keys:
        - "outlet_aggregations": dict mapping outlet_id -> outlet data
        - "total_outlets": int total number of outlets

    Raises
    ------
    ValueError
        If no outlets found
    """
    ti = cast(TaskInstance, context["ti"])
    cfg = cast(HFConfig, context["config"])

    outlets: list[str] = ti.xcom_pull(task_id="build_graph", key="outlets")
    outlet_subgraphs: dict[str, dict[str, Any]] = ti.xcom_pull(task_id="build_graph", key="outlet_subgraphs")

    if not outlets:
        raise ValueError("No outlets found. Aborting run")

    # Apply debug limit if configured
    outlets_to_process = outlets[: cfg.build.debug_outlet_count] if cfg.build.debug_outlet_count else outlets

    results: list[dict[str, Any]] = []

    logger.info(f"map_flowpaths task: Processing {len(outlets_to_process)} outlets sequentially")
    for outlet in tqdm(outlets_to_process, desc="Processing outlets"):
        result = _process_single_outlet(
            outlet,
            outlet_subgraphs[outlet],
            cfg,
        )
        results.append(result)

    outlet_aggregations: dict[str, dict[str, Any]] = {result["outlet"]: result for result in results}

    return {
        "outlet_aggregations": outlet_aggregations,
        "total_outlets": len(outlets),
    }


def map_build_hydrofabric(**context: dict[str, Any]) -> dict[str, Any]:
    """Execute MAP PHASE: Build base hydrofabric layers with assigned ID ranges.

    Each outlet's classifications and aggregations are converted into
    flowpaths, divides, and nexus layers with unique IDs using pre-partitioned
    subgraphs and filtered data.

    Parameters
    ----------
    **context : dict[str, Any]
        Airflow context

    Returns
    -------
    dict[str, Any]
        Dictionary with keys:
        - "built_hydrofabrics": dict mapping outlet_id -> hydrofabric data

    Raises
    ------
    ValueError
        If required data from previous phases not found
    """
    ti = cast(TaskInstance, context["ti"])
    cfg = cast(HFConfig, context["config"])

    outlet_subgraphs: dict[str, dict[str, Any]] = ti.xcom_pull(task_id="build_graph", key="outlet_subgraphs")
    outlet_aggregations: dict[str, dict[str, Any]] = ti.xcom_pull(
        task_id="map_flowpaths", key="outlet_aggregations"
    )
    results: list[dict[str, Any]] = []

    if not outlet_aggregations:
        raise ValueError("Missing outlet aggregations")

    logger.info(f"map_build_base task: Building {len(outlet_aggregations)} hydrofabrics sequentially")
    global_nhf_id = 0
    results = []
    for outlet, outlet_data in tqdm(outlet_aggregations.items(), desc="Building hydrofabrics"):
        result = _build_single_hydrofabric(
            outlet,
            outlet_data,
            global_nhf_id,
            outlet_subgraphs[outlet],
            cfg,
        )
        results.append(result)
        global_nhf_id = result["next_available_id"]

    built_hydrofabrics: dict[str, dict[str, Any]] = {result["outlet"]: result for result in results}

    return {
        "built_hydrofabrics": built_hydrofabrics,
    }


def _bulk_replace(df, cols, replace_map: [int, int]):
    """Replace several columns in a DataFrame by looking their values up in a dict.

    There are a few odd decisions here that are more or less necessary due to how NULL/NA/NaN is handled in DataFrames
    as well as the need to comply with assumptions about datatypes made in later pipeline stages.

    Parameters
    ----------
    df: DataFrame/GeoDataFrame whose columns will be transformed
    cols: list of columns to replace
    replace_map: dict for lookup-based transformation

    Returns
    -------
    The transformed DataFrame/GeoDataFrame
    """
    for col in cols:
        df[col] = df[col].apply(lambda e: None if pd.isna(e) else replace_map[int(e)])
        df[col] = df[col].astype("float64") if df[col].isnull().values.any() else df[col].astype("int64")

    return df


def reassign_ids(
    base_hf: dict[str, (gpd.GeoDataFrame | pd.DataFrame)],
) -> dict[str, (gpd.GeoDataFrame | pd.DataFrame)]:
    """Reassign fp_ids based on spatial sorting"""
    flowpaths = base_hf["flowpaths"]
    flowpaths_wgs84 = gpd.GeoDataFrame(flowpaths.copy()).to_crs("EPSG:4326")
    nexuses = base_hf["nexus"]
    nexuses_wgs84 = gpd.GeoDataFrame(nexuses.copy()).to_crs("EPSG:4326")
    divides = base_hf["divides"]

    fp_ids = set(flowpaths["fp_id"].unique())
    leftover_nex_ids = set(nexuses["nex_id"].unique()) - fp_ids

    fp_ids_sorted = []

    for _, row in flowpaths_wgs84.iterrows():
        # fp_centroid = flowpaths_wgs84[idx]["geometry"].centroid
        fp_centroid = row["geometry"].centroid
        fp_ids_sorted.append((row["fp_id"], (-fp_centroid.x, fp_centroid.y)))

    for _, row in nexuses_wgs84.loc[nexuses_wgs84["nex_id"].isin(leftover_nex_ids)].iterrows():
        # nex_centroid = nexuses_wgs84[idx]["geometry"].centroid
        nex_centroid = row["geometry"].centroid
        fp_ids_sorted.append((row["nex_id"], (-nex_centroid.x, nex_centroid.y)))

    fp_ids_sorted = sorted(fp_ids_sorted, key=lambda e: e[1])

    fp_reassign_map = {}
    for idx, fp_id in enumerate(fp_ids_sorted):
        fp_reassign_map[fp_id[0]] = idx

    flowpaths = _bulk_replace(
        flowpaths, ["fp_id", "fp_to_id", "div_id", "up_nex_id", "dn_nex_id"], fp_reassign_map
    )
    nexuses = _bulk_replace(nexuses, ["nex_id", "dn_fp_id"], fp_reassign_map)
    divides = _bulk_replace(divides, ["div_id"], fp_reassign_map)

    base_hf["flowpaths"] = flowpaths
    base_hf["nexus"] = nexuses
    base_hf["divides"] = divides

    virt_fp_id_offset = len(fp_ids_sorted)

    virt_flowpaths = base_hf["virtual_flowpaths"]
    virt_nexuses = base_hf["virtual_nexus"]

    virt_fp_ids = set(virt_flowpaths["virtual_fp_id"].unique())
    leftover_virt_nex_ids = set(virt_nexuses["virtual_nex_id"].unique()) - virt_fp_ids

    virt_fp_ids_sorted = []

    for _, row in virt_flowpaths.copy().to_crs("EPSG:4326").iterrows():
        # fp_centroid = flowpaths_wgs84[idx]["geometry"].centroid
        fp_centroid = row["geometry"].centroid
        virt_fp_ids_sorted.append((row["virtual_fp_id"], (-fp_centroid.x, fp_centroid.y)))

    for _, row in (
        virt_nexuses.copy()
        .to_crs("EPSG:4326")
        .loc[virt_nexuses["virtual_nex_id"].isin(leftover_virt_nex_ids)]
        .iterrows()
    ):
        # nex_centroid = nexuses_wgs84[idx]["geometry"].centroid
        nex_centroid = row["geometry"].centroid
        virt_fp_ids_sorted.append((row["virtual_nex_id"], (-nex_centroid.x, nex_centroid.y)))

    virt_fp_ids_sorted = sorted(virt_fp_ids_sorted, key=lambda e: e[1])

    virt_fp_reassign_map = {}
    for idx, virt_fp_id in enumerate(virt_fp_ids_sorted):
        virt_fp_reassign_map[virt_fp_id[0]] = idx + virt_fp_id_offset

    base_hf["virtual_flowpaths"] = _bulk_replace(
        virt_flowpaths, ["virtual_fp_id", "up_virtual_nex_id", "dn_virtual_nex_id"], virt_fp_reassign_map
    )
    base_hf["virtual_nexus"] = _bulk_replace(
        virt_nexuses, ["virtual_nex_id", "dn_virtual_fp_id"], virt_fp_reassign_map
    )

    return base_hf


def reduce_combine_base_hydrofabric(**context: dict[str, Any]) -> dict[str, Any]:
    """Execute REDUCE PHASE: Combine all built hydrofabric layers into an aggregated dataset.

    All outlet hydrofabrics are concatenated into single unified layers
    for flowpaths, divides, and nexus points.

    Parameters
    ----------
    **context : dict[str, Any]
        Airflow context

    Returns
    -------
    dict[str, Any]
        Dictionary with keys:
        - "flowpaths": GeoDataFrame of combined flowpaths
        - "divides": GeoDataFrame of combined divides
        - "nexus": GeoDataFrame of combined nexus points

    Raises
    ------
    ValueError
        If no built hydrofabrics found from build phase
    """
    ti = cast(TaskInstance, context["ti"])
    cfg = cast(HFConfig, context["config"])
    built_hydrofabrics: dict[str, dict[str, Any]] = ti.xcom_pull(
        task_id="map_build_base", key="built_hydrofabrics"
    )

    if not built_hydrofabrics:
        raise ValueError("No built hydrofabrics found from build phase")

    result = _combine_hydrofabrics(built_hydrofabrics, cfg.crs)

    result = reassign_ids(result)

    _check_network_cycles(
        fp_gdf=result["flowpaths"],
        nex_gdf=result["nexus"],
        vfp_gdf=result.get("virtual_flowpaths"),
        vnex_gdf=result.get("virtual_nexus"),
    )

    return result

"""Contains all code for processing hydrofabric data"""

import logging
from typing import Any, cast

from dask.diagnostics import ProgressBar
from dask.distributed import Client
from tqdm import tqdm

from hydrofabric_builds.config import HFConfig
from hydrofabric_builds.hydrofabric.aggregate import _aggregate_geometries, _prepare_dataframes
from hydrofabric_builds.hydrofabric.build import _build_base_hydrofabric, _order_aggregates_base
from hydrofabric_builds.hydrofabric.trace import _trace_stack
from hydrofabric_builds.hydrofabric.utils import (
    _calculate_id_ranges_pure,
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
    partition_data : dict
        Contains:
        - "subgraph": rx.PyDiGraph (minimal, only this outlet)
        - "node_indices": dict (for subgraph)
        - "flowpaths": pl.DataFrame (pre-filtered)
        - "divides": pl.DataFrame (pre-filtered)
    cfg : HFConfig
        Config

    Returns
    -------
    dict[str, Any]
        Dictionary containing outlet, classifications, aggregate_data, and num_features
    """
    subgraph = partition_data["subgraph"]
    node_indices = partition_data["node_indices"]
    filtered_flowpaths = partition_data["flowpaths"]
    filtered_divides = partition_data["divides"]

    valid_divide_ids = set(filtered_divides["divide_id"].to_list())
    fp_geom_lookup, div_geom_lookup = _prepare_dataframes(filtered_flowpaths, filtered_divides)

    # Trace with subgraph
    classifications = _trace_stack(
        start_id=outlet,
        fp=filtered_flowpaths,
        div_ids=valid_divide_ids,
        cfg=cfg,
        digraph=subgraph,
        node_indices=node_indices,
    )

    # Aggregate geometries
    aggregate_data = _aggregate_geometries(
        classifications=classifications,
        reference_flowpaths=filtered_flowpaths,
        fp_geom_lookup=fp_geom_lookup,
        div_geom_lookup=div_geom_lookup,
    )

    ordered_aggregates = _order_aggregates_base(aggregate_data)

    return {
        "outlet": outlet,
        "classifications": classifications.model_dump(),
        "aggregate_data": aggregate_data.model_dump(),
        "num_features": len(ordered_aggregates),
    }


def _build_single_hydrofabric(
    outlet: str,
    outlet_data: dict[str, Any],
    id_config: dict[str, Any],
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
    id_config : dict[str, Any]
        ID range configuration for this outlet
    partition_data : dict[str, Any]
        Contains:
        - "subgraph": rx.PyDiGraph (minimal, only this outlet)
        - "node_indices": dict (for subgraph)
        - "flowpaths": pl.DataFrame (pre-filtered)
        - "divides": pl.DataFrame (pre-filtered)
    cfg : HFConfig
        Hydrofabric build config

    Returns
    -------
    dict[str, Any]
        Built hydrofabric data for this outlet
    """
    classifications = Classifications(**outlet_data["classifications"])
    aggregate_data = Aggregations(**outlet_data["aggregate_data"])

    subgraph = partition_data["subgraph"]
    node_indices = partition_data["node_indices"]
    filtered_flowpaths = partition_data["flowpaths"]
    filtered_divides = partition_data["divides"]

    hydrofabric = _build_base_hydrofabric(
        start_id=outlet,
        aggregate_data=aggregate_data,
        classifications=classifications,
        reference_divides=filtered_divides,
        reference_flowpaths=filtered_flowpaths,
        graph=subgraph,
        node_indices=node_indices,
        cfg=cfg,
        id_offset=id_config["id_offset"],
    )

    return {
        "outlet": outlet,
        "flowpaths": hydrofabric["flowpaths"],
        "divides": hydrofabric["divides"],
        "nexus": hydrofabric["nexus"],
        "id_range": (id_config["id_offset"], id_config["id_max"]),
    }


def map_trace_and_aggregate(**context: dict[str, Any]) -> dict:
    """MAP PHASE: Trace and aggregate flowpaths using pre-partitioned subgraphs.

    This task processes each outlet independently using pre-partitioned subgraphs
    and filtered data from the build_graph task. No large DataFrames are broadcasted.

    Parameters
    ----------
    **context : dict
        Airflow context

    Returns
    -------
    dict
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
    client = cast(Client, context["dask_client"])

    outlets = ti.xcom_pull(task_id="build_graph", key="outlets")
    outlet_subgraphs = ti.xcom_pull(task_id="build_graph", key="outlet_subgraphs")

    if not outlets:
        raise ValueError("No outlets found. Aborting run")

    # Apply debug limit if configured
    outlets_to_process = outlets[: cfg.debug_outlet_count] if cfg.debug_outlet_count else outlets

    results: list[dict] = []

    if client is not None:
        logger.info(
            f"map_flowpaths task: Processing {len(outlets_to_process)} outlets using Dask Distributed"
        )

        futures = []
        with ProgressBar():
            for idx, outlet in enumerate(outlets_to_process):
                future = client.submit(
                    _process_single_outlet,
                    outlet,
                    outlet_subgraphs[outlet],
                    cfg,
                )
                futures.append(future)
                if idx % 500 == 0:
                    logger.info(f"map_flowpaths task: Processing checkpoint. Completed {idx} outlets")
        results = client.gather(futures)
    else:
        logger.info(f"map_flowpaths task: Processing {len(outlets_to_process)} outlets sequentially")
        for outlet in tqdm(outlets_to_process, desc="Processing outlets"):
            result = _process_single_outlet(
                outlet,
                outlet_subgraphs[outlet],
                cfg,
            )
            results.append(result)

    outlet_aggregations = {result["outlet"]: result for result in results}

    return {
        "outlet_aggregations": outlet_aggregations,
        "total_outlets": len(outlets),
    }


def reduce_calculate_id_ranges(**context: dict[str, Any]) -> dict[str, Any]:
    """REDUCE PHASE: Calculate ID ranges based on feature counts.

    Combines the results from all outlets to calculate non-overlapping
    ID ranges for each outlet's hydrofabric.

    Parameters
    ----------
    **context : dict
        Airflow context

    Returns
    -------
    dict[str, Any]
        Dictionary with keys:
        - "outlet_id_ranges": dict mapping outlet_id -> {id_offset, id_max, num_features}
        - "total_ids_allocated": int total number of IDs allocated

    Raises
    ------
    ValueError
        If no outlet aggregations found from map phase
    """
    ti = cast(TaskInstance, context["ti"])
    outlet_aggregations = ti.xcom_pull(task_id="map_flowpaths", key="outlet_aggregations")

    if not outlet_aggregations:
        raise ValueError("No outlet aggregations found from map phase")

    return _calculate_id_ranges_pure(outlet_aggregations)


def map_build_base_hydrofabric(**context: dict[str, Any]) -> dict[str, Any]:
    """MAP PHASE: Build base hydrofabric layers with assigned ID ranges.

    Each outlet's classifications and aggregations are converted into
    flowpaths, divides, and nexus layers with unique IDs using pre-partitioned
    subgraphs and filtered data.

    Parameters
    ----------
    **context : dict
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
    client = cast(Client, context["dask_client"])

    outlets = ti.xcom_pull(task_id="build_graph", key="outlets")
    outlet_subgraphs = ti.xcom_pull(task_id="build_graph", key="outlet_subgraphs")
    outlet_aggregations = ti.xcom_pull(task_id="map_flowpaths", key="outlet_aggregations")
    outlet_id_ranges = ti.xcom_pull(task_id="reduce_flowpaths", key="outlet_id_ranges")
    results: list[dict] = []

    if not outlet_aggregations:
        raise ValueError("Missing outlet aggregations")
    if not outlet_id_ranges:
        raise ValueError("Missing ID ranges for outlets")

    outlets_to_process = outlets[: cfg.debug_outlet_count] if cfg.debug_outlet_count else outlets

    if client is not None:
        logger.info(
            f"map_build_base task: Building {len(outlet_aggregations)} hydrofabrics using Dask Distributed"
        )

        futures = []
        with ProgressBar():
            for idx, outlet in enumerate(outlets_to_process):
                future = client.submit(
                    _build_single_hydrofabric,
                    outlet,
                    outlet_aggregations[outlet],
                    outlet_id_ranges[outlet],
                    outlet_subgraphs[outlet],
                    cfg,
                )
                futures.append(future)
                if idx % 500 == 0:
                    logger.info(f"map_build_base task: Processing checkpoint. Completed {idx} outlets")
        results = client.gather(futures)

    else:
        logger.info(f"map_build_base task: Building {len(outlet_aggregations)} hydrofabrics sequentially")
        results = []
        for outlet, outlet_data in tqdm(outlet_aggregations.items(), desc="Building hydrofabrics"):
            result = _build_single_hydrofabric(
                outlet,
                outlet_data,
                outlet_id_ranges[outlet],
                outlet_subgraphs[outlet],
                cfg,
            )
            results.append(result)

    built_hydrofabrics = {result["outlet"]: result for result in results}

    return {
        "built_hydrofabrics": built_hydrofabrics,
    }


def reduce_combine_base_hydrofabric(**context: dict[str, Any]) -> dict[str, Any]:
    """REDUCE PHASE: Combine all built hydrofabric layers into an aggregated dataset.

    All outlet hydrofabrics are concatenated into single unified layers
    for flowpaths, divides, and nexus points.

    Parameters
    ----------
    **context : dict
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
    built_hydrofabrics = ti.xcom_pull(task_id="map_build_base", key="built_hydrofabrics")

    if not built_hydrofabrics:
        raise ValueError("No built hydrofabrics found from build phase")

    return _combine_hydrofabrics(built_hydrofabrics, cfg.crs)

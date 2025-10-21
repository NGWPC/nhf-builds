"""Contains all code for processing hydrofabric data"""

import logging
from typing import Any, cast

from hydrofabric_builds.config import HFConfig
from hydrofabric_builds.hydrofabric.aggregate import _aggregate_geometries
from hydrofabric_builds.hydrofabric.build import _build_base_hydrofabric, _order_aggregates_base
from hydrofabric_builds.hydrofabric.trace import _trace_stack
from hydrofabric_builds.hydrofabric.utils import _calculate_id_ranges_pure, _combine_hydrofabrics_pure
from hydrofabric_builds.schemas.hydrofabric import Aggregations, Classifications
from hydrofabric_builds.task_instance import TaskInstance

logger = logging.getLogger(__name__)


def map_trace_and_aggregate(**context: dict[str, Any]) -> dict:
    """MAP PHASE: Trace and aggregate each outlet independently.

    Parameters
    ----------
    **context : dict
        Airflow context with outlet_id passed via task mapping

    Returns
    -------
    dict
        outlet aggregation results (one per outlet)
    """
    ti = cast(TaskInstance, context["ti"])
    cfg = cast(HFConfig, context["config"])

    reference_flowpaths = ti.xcom_pull(task_id="download", key="reference_flowpaths")
    reference_divides = ti.xcom_pull(task_id="download", key="reference_divides")
    upstream_network = ti.xcom_pull(task_id="build_graph", key="upstream_network")
    outlets = ti.xcom_pull(task_id="build_graph", key="outlets")

    if not outlets:
        raise ValueError("No outlets found. Aborting run")

    outlet_aggregations = {}
    for outlet in outlets:
        classifications = _trace_stack(
            start_id=outlet, network_graph=upstream_network, fp=reference_flowpaths, cfg=cfg
        )
        aggregate_data = _aggregate_geometries(
            classifications=classifications,
            reference_divides=reference_divides,
            reference_flowpaths=reference_flowpaths,
        )
        ordered_aggregates = _order_aggregates_base(aggregate_data)
        num_features = len(ordered_aggregates)

        outlet_aggregations[outlet] = {
            "outlet": outlet,
            "classifications": classifications.model_dump(),  # Serialize for XCom
            "aggregate_data": {
                "aggregates": aggregate_data.aggregates,
                "independents": aggregate_data.independents,
                "connectors": aggregate_data.connectors,
                "minor_flowpaths": aggregate_data.minor_flowpaths,
                "small_scale_connectors": aggregate_data.small_scale_connectors,
            },
            "num_features": num_features,
        }

    return {
        "outlet_aggregations": outlet_aggregations,
        "total_outlets": len(outlets),
    }


def reduce_calculate_id_ranges(**context: dict[str, Any]) -> dict[str, Any]:
    """Calculate ID ranges based on feature counts (Reduce).

    Action: Combines the results from all outlets to calculate non-overlapping
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
    """Build base hydrofabric layers (flowpaths, divides, nexus) with assigned ID ranges (Map).

    Each outlet's classifications and aggregations are converted into
    flowpaths, divides, and nexus layers with unique IDs.

    Parameters
    ----------
    **context : dict
        Airflow context

    Returns
    -------
    dict[str, Any]
        Dictionary with keys:
        - "built_hydrofabrics": dict mapping outlet_id -> hydrofabric data
        - "total_flowpaths": int total flowpaths across all outlets
        - "total_divides": int total divides across all outlets
        - "total_nexus": int total nexus points across all outlets

    Raises
    ------
    ValueError
        If required data from previous phases not found
    """
    ti = cast(TaskInstance, context["ti"])
    cfg = cast(HFConfig, context["config"])
    reference_flowpaths = ti.xcom_pull(task_id="download", key="reference_flowpaths")
    reference_divides = ti.xcom_pull(task_id="download", key="reference_divides")
    upstream_network = ti.xcom_pull(task_id="build_graph", key="upstream_network")
    outlet_aggregations = ti.xcom_pull(task_id="map_flowpaths", key="outlet_aggregations")
    outlet_id_ranges = ti.xcom_pull(task_id="reduce_flowpaths", key="outlet_id_ranges")
    if not outlet_aggregations:
        raise ValueError("Missing outlet aggregations")
    if not outlet_id_ranges:
        raise ValueError("Missing ID ranges for outlets")

    built_hydrofabrics = {}

    for outlet, outlet_data in outlet_aggregations.items():
        id_config = outlet_id_ranges[outlet]

        classifications = Classifications(**outlet_data["classifications"])
        aggregate_data = Aggregations(**outlet_data["aggregate_data"])
        hydrofabric = _build_base_hydrofabric(
            start_id=outlet,
            aggregate_data=aggregate_data,
            classifications=classifications,
            reference_divides=reference_divides,
            reference_flowpaths=reference_flowpaths,
            upstream_network=upstream_network,
            cfg=cfg,
            id_offset=id_config["id_offset"],
        )

        built_hydrofabrics[outlet] = {
            "outlet": outlet,
            "flowpaths": hydrofabric["flowpaths"],
            "divides": hydrofabric["divides"],
            "nexus": hydrofabric["nexus"],
            "id_range": (id_config["id_offset"], id_config["id_max"]),
        }

    return {
        "built_hydrofabrics": built_hydrofabrics,
    }


def reduce_combine_base_hydrofabric(**context: dict[str, Any]) -> dict[str, Any]:
    """Combine all built hydrofabric layers into an aggregated dataset (Reduce).

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

    return _combine_hydrofabrics_pure(built_hydrofabrics, cfg.crs)

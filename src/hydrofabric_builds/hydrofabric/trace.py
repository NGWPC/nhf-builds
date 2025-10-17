"""A file to hold all tracing/Hydrofabric stack-related codes"""

import logging
from collections import deque
from typing import Any

import pandas as pd

from hydrofabric_builds.config import HFConfig
from hydrofabric_builds.hydrofabric.schemas import Classifications

logger = logging.getLogger(__name__)


def _get_flowpath_info(fp_id: str, fp_indexed: pd.DataFrame) -> dict[str, Any]:
    """Gets necessary information from the flowpath requested from the reference in dictionary format

    Parameters
    ----------
    fp_id : str
        The current flowpath ID
    fp_indexed : pd.DataFrame
        The flowpath dataframe indexed by ID

    Returns
    -------
    dict[str, Any]
        The required information from the flowpath
    """
    fp_row = fp_indexed.loc[int(float(fp_id))]

    return {
        "flowpath_id": fp_id,
        "total_drainage_area_sqkm": fp_row["totdasqkm"],
        "areasqkm": fp_row["areasqkm_left"],
        "length_km": fp_row["lengthkm"],
        "stream_order": fp_row["streamorder"],
        "hydroseq": fp_row["hydroseq"],
        "dnhydroseq": fp_row["dnhydroseq"],
    }


def _get_unprocessed_upstream_info(
    upstream_ids: list[str], fp_indexed: pd.DataFrame, processed_flowpaths: set[str]
) -> list[dict]:
    """Helper function to get info for unprocessed upstream flowpaths.

    Parameters
    ----------
    upstream_ids : list[str]
        upstream flowpath ids
    fp_indexed : pd.DataFrame
        the reference flowpaths dataframe
    processed_flowpaths : set[str]
        the set of flowpaths that have been processed

    Returns
    -------
    list[dict]
        a list of all upstream info / segments to be processed
    """
    upstream_info = []
    for upstream_id in upstream_ids:
        if upstream_id not in processed_flowpaths:
            info = _get_flowpath_info(upstream_id, fp_indexed)
            upstream_info.append(info)
    return upstream_info


def _queue_upstream(
    upstream_ids: list[str], to_process: deque, processed_flowpaths: set[str], unprocessed_only: bool = False
) -> None:
    """Helper function to add unprocessed upstream flowpaths to processing queue.

    Parameters
    ----------
    upstream_ids : list[str]
        upstream flowpath ids
    to_process : deque
        the stack data structure
    processed_flowpaths : set[str]
        the set of flowpaths that have been processed
    unprocessed_only: bool
        will queue only unprocessed segments
    """
    for upstream_id in upstream_ids:
        if unprocessed_only:
            if upstream_id not in processed_flowpaths:
                to_process.append(upstream_id)
        else:
            to_process.append(upstream_id)


def _rule_independent_large_area(
    current_id: str, fp_info: dict, cfg: HFConfig, result: Classifications
) -> bool:
    """A Hydrofabric Rule for establishing Independent Divides based on a Large Contributing Area

    Parameters
    ----------
    current_id : str
        The current flowpath ID
    fp_info : dict
        All of the catchment information contained within a flowpath ID
    cfg : HFConfig
        The config file
    result : Classifications
        A container for storing classification references

    Returns
    -------
    bool
        True means this rule can be applied and classification was done. False if the rule cannot be applied to this flowpath
    """
    if fp_info["areasqkm"] > cfg.divide_aggregation_threshold:
        result.independent_flowpaths.append(current_id)
        return True
    return False


def _rule_independent_connector(
    current_id: str,
    fp_info: dict,
    upstream_info: list[dict],
    cfg: HFConfig,
    network_graph: dict,
    result: Classifications,
) -> bool:
    """Rule: areasqkm < threshold AND two OR MORE upstream flowpaths where ALL have stream_order > 1 OR: areasqkm < threshold AND has upstream with stream_order == 1 where that upstream's areasqkm > threshold

    Action: Mark as connector segment, aggregate any small order-1 upstreams

    Parameters
    ----------
    current_id : str
        The current flowpath ID
    fp_info : dict
        All of the catchment information contained within a flowpath ID
    upstream_info : list[dict]
        A list of all information for upstream catchments
    cfg : HFConfig
        The config file
    network_graph : dict
        The network graph
    result : Classifications
        A container for storing classification references

    Returns
    -------
    bool
        True means this rule can be applied and classification was done. False if the rule cannot be applied to this flowpath
    """
    if (
        fp_info["areasqkm"] < cfg.divide_aggregation_threshold
        and len(upstream_info) >= 2
        and all(info["stream_order"] > 1 for info in upstream_info)
    ):
        result.connector_segments.append(current_id)
        return True

    if fp_info["areasqkm"] < cfg.divide_aggregation_threshold and len(upstream_info) >= 2:
        # Check if any order-1 upstream is large enough to be independent
        large_order1_upstreams = [
            info
            for info in upstream_info
            if info["stream_order"] == 1 and info["areasqkm"] > cfg.divide_aggregation_threshold
        ]

        # Get small order-1 upstreams that should be aggregated
        small_order1_upstreams = [
            info
            for info in upstream_info
            if info["stream_order"] == 1 and info["areasqkm"] <= cfg.divide_aggregation_threshold
        ]

        if large_order1_upstreams:
            result.connector_segments.append(current_id)

            # Trace and aggregate any small order-1 upstreams into current
            for small_order1 in small_order1_upstreams:
                _trace_and_aggregate_upstream(
                    current_id=small_order1["flowpath_id"],
                    target_id=current_id,
                    network_graph=network_graph,
                    result=result,
                    is_minor=True,
                )

            return True
    return False


def _rule_aggregate_single_upstream(
    current_id: str, fp_info: dict, upstream_info: list[dict], cfg: HFConfig, result: Classifications
) -> bool:
    """Rule: areasqkm < threshold AND one upstream flowpath. Action: Mark to merge into upstream catchment

    Parameters
    ----------
    current_id : str
        The current flowpath ID
    fp_info : dict
        All of the catchment information contained within a flowpath ID
    upstream_info : list[dict]
        A list of all information for upstream catchments
    cfg : HFConfig
        The config file
    result : Classifications
        A container for storing classification references

    Returns
    -------
    bool
        True means this rule can be applied and classification was done. False if the rule cannot be applied to this flowpath
    """
    if fp_info["areasqkm"] < cfg.divide_aggregation_threshold and len(upstream_info) == 1:
        upstream = upstream_info[0]
        result.aggregation_pairs.append((current_id, upstream["flowpath_id"]))

        cumulative_area = result.cumulative_merge_areas.get(current_id, fp_info["areasqkm"])
        if cumulative_area is None:
            cumulative_area = 0
        result.cumulative_merge_areas[upstream["flowpath_id"]] = (
            cumulative_area + result.cumulative_merge_areas.get(upstream["flowpath_id"], 0)
        )
        return True
    return False


def _aggregate_all_upstream_recursive(
    start_id: str, target_id: str, network_graph: dict, result: Classifications
) -> None:
    """Helper to recursively aggregate all upstream flowpaths into a target. Used for order 1 streams to collect entire headwater network

    Parameters
    ----------
    start_id : str
        the upstream COMID
    target_id : dict
        the current_id that upstream segments are aggregated to
    network_graph : dict
        A dictionary containing downstream : [upstream] connections
    result : Classifications
        A container for storing classification references
    """
    upstream_ids = network_graph.get(start_id, [])

    for upstream_id in upstream_ids:
        if upstream_id not in result.processed_flowpaths:
            result.aggregation_pairs.append((upstream_id, target_id))
            result.processed_flowpaths.add(upstream_id)
            _aggregate_all_upstream_recursive(upstream_id, target_id, network_graph, result)


def _trace_and_aggregate_upstream(
    current_id: str,
    target_id: str,
    network_graph: dict,
    result: Classifications,
    is_minor: bool = False,
) -> None:
    """Recursively trace upstream from a minor flowpath and aggregate all upstream segments into the target. Marks all as processed to prevent independent classification.

    Parameters
    ----------
    current_id : str
        The current_flowpath_id
    target_id : str
        The target flowpath that everything aggregates into
    network_graph : dict
        Network graph of upstream connections
    result : Classifications
        Classification results to update
    """
    upstream_ids = network_graph.get(current_id, [])
    if current_id != target_id:
        result.aggregation_pairs.append((current_id, target_id))
    result.processed_flowpaths.add(current_id)
    if is_minor:
        result.minor_flowpaths.append(current_id)
    for upstream_id in upstream_ids:
        if upstream_id not in result.processed_flowpaths:
            _trace_and_aggregate_upstream(upstream_id, target_id, network_graph, result, is_minor)


def _rule_aggregate_order1_all_upstream(
    current_id: str, fp_info: dict, upstream_info: list[dict], network_graph: dict, result: Classifications
) -> bool:
    """Rule: stream_order == 1 Action: Aggregate all upstream flowpaths into current flowpath

    Parameters
    ----------
    current_id : str
        The current flowpath ID
    fp_info : dict
        All of the catchment information contained within a flowpath ID
    upstream_info : list[dict]
        A list of all information for upstream catchments
    cfg : HFConfig
        The config file
    result : Classifications
        A container for storing classification references

    Returns
    -------
    bool
        True means this rule can be applied and classification was done. False if the rule cannot be applied to this flowpath
    """
    if fp_info["stream_order"] == 1 and len(upstream_info) > 0:
        for upstream in upstream_info:
            result.aggregation_pairs.append((upstream["flowpath_id"], current_id))
            result.processed_flowpaths.add(upstream["flowpath_id"])

            # Recursively aggregate all upstream of this upstream
            _aggregate_all_upstream_recursive(upstream["flowpath_id"], current_id, network_graph, result)

        return True
    return False


def _rule_aggregate_order2_with_order1s(
    current_id: str, fp_info: dict, upstream_info: list[dict], network_graph: dict, result: Classifications
) -> bool:
    """Rule: stream_order == 2 AND two OR MORE upstream flowpaths where ALL are stream_order == 1

    Action:
    - Aggregate largest drainage area upstream to current stream
    - Mark all other upstreams as minor flowpaths to be aggregated to current
    - Mark for subdivide preservation

    Parameters
    ----------
    current_id : str
        The current flowpath ID
    fp_info : dict
        All of the catchment information contained within a flowpath ID
    upstream_info : list[dict]
        A list of all information for upstream catchments
    network_graph : dict
        The network graph
    result : Classifications
        A container for storing classification references

    Returns
    -------
    bool
        True means this rule can be applied and classification was done. False if the rule cannot be applied to this flowpath
    """
    if (
        fp_info["stream_order"] == 2
        and len(upstream_info) >= 2  # Changed from == 2 to >= 2
        and all(info["stream_order"] == 1 for info in upstream_info)
    ):
        # Find the largest upstream by drainage area
        largest_upstream = max(upstream_info, key=lambda x: x["total_drainage_area_sqkm"])
        minor_upstreams = [
            info for info in upstream_info if info["flowpath_id"] != largest_upstream["flowpath_id"]
        ]

        # Trace and aggregate the largest (non-minor) upstream
        _trace_and_aggregate_upstream(
            current_id=largest_upstream["flowpath_id"],
            target_id=current_id,
            network_graph=network_graph,
            result=result,
            is_minor=False,
        )

        # Trace and aggregate all minor upstreams
        for minor in minor_upstreams:
            _trace_and_aggregate_upstream(
                current_id=minor["flowpath_id"],
                target_id=current_id,
                network_graph=network_graph,
                result=result,
                is_minor=True,
            )

        result.processed_flowpaths.add(current_id)
        result.subdivide_candidates.append(current_id)
        result.upstream_merge_points.append(current_id)
        return True
    return False


def _rule_aggregate_mixed_upstream_orders(
    current_id: str, fp_info: dict, upstream_info: list[dict], cfg: HFConfig, result: Classifications
) -> bool:
    """Rule: stream_order > 1 AND multiple upstream segments where one is stream_order == 1 and another is stream_order > 1

    Action: Mark SMALL order 1 (areasqkm < threshold) as minor flowpath to aggregate into current
    Note: Large order-1 segments (areasqkm > threshold) are left independent

    Parameters
    ----------
    current_id : str
        The current flowpath ID
    fp_info : dict
        All of the catchment information contained within a flowpath ID
    upstream_info : list[dict]
        A list of all information for upstream catchments
    cfg : HFConfig
        The config file
    result : Classifications
        A container for storing classification references

    Returns
    -------
    bool
        True means this rule can be applied and classification was done. False if the rule cannot be applied to this flowpath
    """
    if fp_info["stream_order"] > 1 and len(upstream_info) >= 2:
        small_order_1_upstreams = [
            info
            for info in upstream_info
            if info["stream_order"] == 1 and info["areasqkm"] < cfg.divide_aggregation_threshold
        ]
        large_order_1_upstreams = [
            info
            for info in upstream_info
            if info["stream_order"] == 1 and info["areasqkm"] >= cfg.divide_aggregation_threshold
        ]
        higher_order_upstreams = [info for info in upstream_info if info["stream_order"] > 1]

        if small_order_1_upstreams and (higher_order_upstreams or large_order_1_upstreams):
            for order_1 in small_order_1_upstreams:
                result.minor_flowpaths.append(order_1["flowpath_id"])
                result.aggregation_pairs.append((order_1["flowpath_id"], current_id))
            # Note upstream merge point for subdivides
            result.upstream_merge_points.append(current_id)
            return True
    return False


def _rule_aggregate_same_order_small_area(
    current_id: str, fp_info: dict, upstream_info: list[dict], cfg: HFConfig, result: Classifications
) -> bool:
    """Rule: stream_order > 1 AND multiple upstream segments where one is stream_order == 1 and another is stream_order == current AND areasqkm < threshold

    Action:
    - Mark order 1 as minor flowpath to aggregate into current
    - Current flowpath merges into same-order upstream

    Parameters
    ----------
    current_id : str
        The current flowpath ID
    fp_info : dict
        All of the catchment information contained within a flowpath ID
    upstream_info : list[dict]
        A list of all information for upstream catchments
    cfg : HFConfig
        The config file
    result : Classifications
        A container for storing classification references

    Returns
    -------
    bool
        True means this rule can be applied and classification was done. False if the rule cannot be applied to this flowpath
    """
    if (
        fp_info["stream_order"] > 1
        and fp_info["areasqkm"] < cfg.divide_aggregation_threshold
        and len(upstream_info) >= 2
    ):
        order_1_upstreams = [info for info in upstream_info if info["stream_order"] == 1]
        same_order_upstreams = [
            info for info in upstream_info if info["stream_order"] == fp_info["stream_order"]
        ]

        if order_1_upstreams and same_order_upstreams:
            for order_1 in order_1_upstreams:
                result.minor_flowpaths.append(order_1["flowpath_id"])
                result.aggregation_pairs.append((order_1["flowpath_id"], current_id))
            same_order = same_order_upstreams[0]
            result.aggregation_pairs.append((current_id, same_order["flowpath_id"]))

            cumulative_area = result.cumulative_merge_areas.get(current_id, fp_info["areasqkm"])
            if cumulative_area is None:
                cumulative_area = 0
            result.cumulative_merge_areas[same_order["flowpath_id"]] = (
                cumulative_area + result.cumulative_merge_areas.get(same_order["flowpath_id"], 0)
            )
            result.upstream_merge_points.append(current_id)
            return True
    return False


def _trace_stack(start_id: str, network_graph: dict, fp: pd.DataFrame, cfg: HFConfig) -> Classifications:
    """
    Trace upstream from a starting flowpath and classify segments according to aggregation rules.

    Rules are applied in order:
    1. Independent - Large Area (>3km²)
    2. Independent - Connector Segment (small area, 2 upstream with order > 1)
    3. Aggregate - Single Upstream (headwater aggregation)
    4. Aggregate - Order 1 Stream (aggregate all upstream into current)
    5. Aggregate - Order 2 with Two Order 1s (subdivide candidate)
    6. Aggregate - Mixed Upstream Orders (order 1 becomes minor)
    7. Aggregate - Same Order with Small Area (current merges upstream)

    Parameters
    ----------
    start_id : str
        the outlet flowpath ID
    network_graph : dict
        the graph of all network connections between flowpaths
    fp : pd.DataFrame
        the reference flowpaths dataframe
    cfg : HFConfig
        the Hydrofabric config file

    Returns
    -------
    Classifications
        A Pydantic BaseModel containing all of the flowpaths and their classifications

    Raises
    ------
    ValueError
        If one of the flowpaths doesn't pass a rule the workflow will be stopped
    """
    fp_indexed = fp.set_index("flowpath_id")
    result = Classifications()
    to_process = deque([start_id])

    while to_process:
        current_id = to_process.popleft()

        fp_info = _get_flowpath_info(current_id, fp_indexed)
        upstream_ids = network_graph.get(current_id, [])
        upstream_info = _get_unprocessed_upstream_info(upstream_ids, fp_indexed, result.processed_flowpaths)

        if not upstream_ids:
            if current_id not in result.processed_flowpaths:
                result.independent_flowpaths.append(current_id)
                result.processed_flowpaths.add(current_id)
            continue

        result.processed_flowpaths.add(current_id)

        # Rule 1: Independent - Large Area
        if _rule_independent_large_area(current_id, fp_info, cfg, result):
            _queue_upstream(upstream_ids, to_process, result.processed_flowpaths, unprocessed_only=True)
            continue

        # Rule 2: Independent - Connector Segment
        if _rule_independent_connector(current_id, fp_info, upstream_info, cfg, network_graph, result):
            _queue_upstream(upstream_ids, to_process, result.processed_flowpaths, unprocessed_only=True)
            continue

        # Rule 3: Aggregate - Single Upstream
        if _rule_aggregate_single_upstream(current_id, fp_info, upstream_info, cfg, result):
            _queue_upstream(upstream_ids, to_process, result.processed_flowpaths, unprocessed_only=True)
            continue

        # Rule 4: Aggregate - Order 1 Stream (All Upstream). This case should be covered by Rule 3
        if _rule_aggregate_order1_all_upstream(current_id, fp_info, upstream_info, network_graph, result):
            continue

        # Rule 5: Aggregate - Order 2 with Two Order 1s
        if _rule_aggregate_order2_with_order1s(current_id, fp_info, upstream_info, network_graph, result):
            _queue_upstream(upstream_ids, to_process, result.processed_flowpaths, unprocessed_only=True)
            continue

        # Rule 6: Aggregate - Mixed Upstream Orders
        if _rule_aggregate_mixed_upstream_orders(current_id, fp_info, upstream_info, cfg, result):
            _queue_upstream(upstream_ids, to_process, result.processed_flowpaths, unprocessed_only=True)
            continue

        # Rule 7: Aggregate - Same Order with Small Area
        if _rule_aggregate_same_order_small_area(current_id, fp_info, upstream_info, cfg, result):
            _queue_upstream(upstream_ids, to_process, result.processed_flowpaths, unprocessed_only=True)
            continue

        raise ValueError(f"No Rule Matched. Please debug flowpath_id: {current_id}")

    return result

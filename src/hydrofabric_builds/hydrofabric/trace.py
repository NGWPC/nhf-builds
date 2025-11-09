"""Tracing and classification module for hydrofabric builds"""

import logging
from collections import deque
from typing import Any

import polars as pl
import rustworkx as rx

from hydrofabric_builds.config import HFConfig
from hydrofabric_builds.schemas.hydrofabric import Classifications

logger = logging.getLogger(__name__)


def _get_unprocessed_upstream_info(
    upstream_ids: list[str], fp_lookup: dict[str, dict[str, Any]], processed: set[str]
) -> list[dict[str, Any]]:
    """Get info for unprocessed upstream flowpaths.

    Parameters
    ----------
    upstream_ids : list[str]
        List of upstream flowpath IDs
    fp_lookup : dict[str, dict[str, Any]]
        Dictionary mapping flowpath_id -> flowpath attributes
    processed : set[str]
        Set of already processed flowpath IDs

    Returns
    -------
    list[dict[str, Any]]
        List of flowpath info dictionaries for unprocessed upstreams
    """
    if not upstream_ids:
        return []

    result: list[dict[str, Any]] = []
    for uid in upstream_ids:
        uid_str = str(uid)
        if uid_str in processed or uid_str not in fp_lookup:
            continue

        row = fp_lookup[uid_str]
        result.append(
            {
                "flowpath_id": uid_str,
                "areasqkm": float(row["areasqkm"]),
                "streamorder": int(row["streamorder"]),
                "lengthkm": float(row["lengthkm"]),
                "totdasqkm": float(row["totdasqkm"]),
                "mainstemlp": float(row.get("mainstemlp", 0)),
            }
        )

    return result


def _queue_upstream(
    upstream_ids: list[str], to_process: deque[str], processed: set[str], unprocessed_only: bool = False
) -> None:
    """Queue upstream flowpaths for processing.

    Parameters
    ----------
    upstream_ids : list[str]
        List of upstream flowpath IDs
    to_process : deque[str]
        Queue of flowpaths to process
    processed : set[str]
        Set of already processed flowpath IDs
    unprocessed_only : bool, optional
        If True, only queue unprocessed flowpaths, by default False
    """
    for uid in upstream_ids:
        uid_str = str(uid)
        if not unprocessed_only or uid_str not in processed:
            to_process.append(uid_str)


def _get_upstream_ids(flowpath_id: str, graph: rx.PyDiGraph, node_indices: dict[str, int]) -> list[str]:
    """Get upstream flowpath IDs from the graph.

    Parameters
    ----------
    flowpath_id : str
        The flowpath ID to get upstreams for
    graph : rx.PyDiGraph
        Network graph
    node_indices : dict[str, int]
        Mapping of flowpath IDs to node indices

    Returns
    -------
    list[str]
        List of upstream flowpath IDs
    """
    if flowpath_id not in node_indices:
        return []

    fp_idx = node_indices[flowpath_id]
    return [str(graph[idx]) for idx in graph.predecessor_indices(fp_idx)]


def _traverse_and_aggregate_all_upstream(
    upstream_id: str,
    downstream_id: str,
    result: Classifications,
    graph: rx.PyDiGraph,
    node_indices: dict[str, int],
) -> None:
    """Traverse all upstream flowpaths and aggregate them to a downstream target.

    This creates aggregation pairs for every flowpath encountered upstream.
    Used for simple upstream aggregation (e.g., order 1 streams).

    Parameters
    ----------
    start_id : str
        Starting flowpath ID
    downstream_id : str
        Target downstream flowpath ID for aggregation
    result : Classifications
        Results container to update
    graph : rx.PyDiGraph
        Network graph
    node_indices : dict[str, int]
        Mapping of flowpath IDs to node indices
    """
    while True:
        result.aggregation_pairs.append((upstream_id, downstream_id))
        result.processed_flowpaths.add(upstream_id)

        # Get next upstream
        next_upstream_ids = _get_upstream_ids(upstream_id, graph, node_indices)

        # Stop if no upstream (headwater)
        if not next_upstream_ids:
            break

        next_upstream_id = next_upstream_ids[0]
        upstream_id = next_upstream_id


def _traverse_and_mark_as_minor(
    start_id: str,
    downstream_id: str,
    result: Classifications,
    graph: rx.PyDiGraph,
    node_indices: dict[str, int],
) -> None:
    """Traverse all upstream flowpaths and mark them as minor, aggregating to downstream.

    This marks flowpaths as minor and creates aggregation pairs.
    Used for small tributaries that should be aggregated into larger streams.
    Continues traversing through ALL upstreams regardless of divides.

    Parameters
    ----------
    start_id : str
        Starting flowpath ID
    downstream_id : str
        Target downstream flowpath ID for aggregation
    result : Classifications
        Results container to update
    graph : rx.PyDiGraph
        Network graph
    node_indices : dict[str, int]
        Mapping of flowpath IDs to node indices
    """
    upstream_id = start_id

    while True:
        result.minor_flowpaths.add(upstream_id)
        result.aggregation_pairs.append((upstream_id, downstream_id))
        result.processed_flowpaths.add(upstream_id)

        # Get next upstream
        next_upstream_ids = _get_upstream_ids(upstream_id, graph, node_indices)

        # Stop if no upstream (headwater)
        if not next_upstream_ids:
            break

        # If multiple upstreams, traverse all of them
        if len(next_upstream_ids) > 1:
            for next_id in next_upstream_ids:
                if next_id not in result.processed_flowpaths:
                    _traverse_and_mark_as_minor(next_id, downstream_id, result, graph, node_indices)
            break

        # Single upstream - continue
        next_upstream_id = next_upstream_ids[0]
        upstream_id = next_upstream_id


def _traverse_chain_with_area_threshold(
    start_id: str,
    current_id: str,
    initial_area: float,
    threshold: float,
    fp_lookup: dict[str, dict[str, Any]],
    div_ids: set[str],
    result: Classifications,
    graph: rx.PyDiGraph,
    node_indices: dict[str, int],
    to_process: deque[str],
    check_stream_order: bool = False,
    expected_order: int | None = None,
) -> None:
    """Traverse upstream chain aggregating until area threshold is met.

    Creates aggregation pairs while tracking cumulative area. Stops when:
    - Cumulative area exceeds threshold
    - Multiple upstream (confluence)
    - No upstream (headwater)
    - Stream order changes (if check_stream_order=True)
    - Next upstream lacks divide

    Parameters
    ----------
    start_id : str
        Starting upstream flowpath ID
    current_id : str
        Current/downstream flowpath ID for aggregation pairs
    initial_area : float
        Initial cumulative area (km²)
    threshold : float
        Area threshold (km²) to stop aggregation
    fp_lookup : dict[str, dict[str, Any]]
        Flowpath lookup dictionary
    div_ids : set[str]
        Set of divide IDs
    result : Classifications
        Results container to update
    graph : rx.PyDiGraph
        Network graph
    node_indices : dict[str, int]
        Mapping of flowpath IDs to node indices
    to_process : deque[str]
        Queue for processing flowpaths
    check_stream_order : bool, optional
        If True, stop when stream order changes, by default False
    expected_order : int | None, optional
        Expected stream order (only used if check_stream_order=True), by default None
    """
    upstream_id = start_id
    cumulative_area = initial_area

    while True:
        # Check if upstream exists
        if upstream_id not in fp_lookup:
            break

        upstream_fp_info = fp_lookup[upstream_id]
        cumulative_area += float(upstream_fp_info["areasqkm"])

        # Create aggregation pair
        result.aggregation_pairs.append((current_id, upstream_id))
        result.processed_flowpaths.add(upstream_id)

        # Check if we've exceeded threshold - stop and queue upstream
        if cumulative_area >= threshold:
            next_upstream_ids = _get_upstream_ids(upstream_id, graph, node_indices)
            for uid in next_upstream_ids:
                if uid not in result.processed_flowpaths:
                    to_process.append(uid)
            break

        # Get next upstream
        next_upstream_ids = _get_upstream_ids(upstream_id, graph, node_indices)

        # Stop if no upstream (headwater)
        if not next_upstream_ids:
            break

        # Stop if multiple upstream (connector/confluence) - queue all
        if len(next_upstream_ids) > 1:
            for uid in next_upstream_ids:
                if uid not in result.processed_flowpaths:
                    to_process.append(uid)
            break

        # Get next upstream info
        next_upstream_id = next_upstream_ids[0]

        # Check stream order if required
        if check_stream_order and expected_order is not None:
            if next_upstream_id not in fp_lookup:
                break

            next_fp_info = fp_lookup[next_upstream_id]
            if next_fp_info["streamorder"] != expected_order:
                # Different order - queue it for processing
                if next_upstream_id not in result.processed_flowpaths:
                    to_process.append(next_upstream_id)
                break

        # Stop if next upstream lacks divide - queue it
        if next_upstream_id not in div_ids:
            if next_upstream_id not in result.processed_flowpaths:
                to_process.append(next_upstream_id)
            break

        # Continue chaining upstream divides
        upstream_id = next_upstream_id


def _traverse_chain_through_no_divides(
    start_id: str,
    current_id: str,
    div_ids: set[str],
    result: Classifications,
    graph: rx.PyDiGraph,
    node_indices: dict[str, int],
    to_process: deque[str],
) -> None:
    """Traverse upstream chain through flowpaths without divides.

    Aggregates flowpaths that lack divides into the current flowpath.
    Stops when encountering a divide or confluence.

    Parameters
    ----------
    start_id : str
        Starting upstream flowpath ID (without divide)
    current_id : str
        Current/downstream flowpath ID for aggregation pairs
    div_ids : set[str]
        Set of divide IDs
    result : Classifications
        Results container to update
    graph : rx.PyDiGraph
        Network graph
    node_indices : dict[str, int]
        Mapping of flowpath IDs to node indices
    to_process : deque[str]
        Queue for processing flowpaths
    """
    upstream_id = start_id

    while True:
        # Chain through this no-divide flowpath
        result.aggregation_pairs.append((current_id, upstream_id))
        result.processed_flowpaths.add(upstream_id)

        # Get next upstream
        next_upstream_ids = _get_upstream_ids(upstream_id, graph, node_indices)

        # Stop if no upstream (headwater)
        if not next_upstream_ids:
            break

        # Stop if multiple upstream (connector/confluence) - queue all
        if len(next_upstream_ids) > 1:
            for uid in next_upstream_ids:
                if uid not in result.processed_flowpaths:
                    to_process.append(uid)
            break

        # Get next upstream
        next_upstream_id = next_upstream_ids[0]

        # Stop if next upstream has a divide - queue it
        if next_upstream_id in div_ids:
            if next_upstream_id not in result.processed_flowpaths:
                to_process.append(next_upstream_id)
            break

        # Continue chaining through no-divide flowpaths
        upstream_id = next_upstream_id


def _check_for_any_upstream_divides(
    start_id: str,
    div_ids: set[str],
    graph: rx.PyDiGraph,
    node_indices: dict[str, int],
) -> bool:
    """Check if there are ANY divides anywhere in the upstream network.

    Uses BFS to search entire upstream network for any segment with a divide.

    Parameters
    ----------
    start_id : str
        Starting flowpath ID
    div_ids : set[str]
        Set of all divide IDs
    graph : rx.PyDiGraph
        Network graph
    node_indices : dict[str, int]
        Mapping of flowpath IDs to node indices

    Returns
    -------
    bool
        True if any upstream segment has a divide, False otherwise
    """
    to_check: deque[str] = deque([start_id])
    visited: set[str] = set()

    while to_check:
        current_fp_id = to_check.popleft()

        if current_fp_id in visited:
            continue

        visited.add(current_fp_id)

        # Check if this segment has a divide
        if current_fp_id in div_ids:
            return True

        # Get upstream and continue searching
        upstream_ids = _get_upstream_ids(current_fp_id, graph, node_indices)
        for up_id in upstream_ids:
            to_check.append(up_id)

    # No divides found anywhere upstream
    return False


def _check_and_aggregate_same_order_no_divide_chain(
    start_id: str,
    current_order: int,
    fp_lookup: dict[str, dict[str, Any]],
    div_ids: set[str],
    result: Classifications,
    graph: rx.PyDiGraph,
    node_indices: dict[str, int],
) -> bool:
    """Check if entire same-order upstream chain has no divides, and aggregate if so.

    Traverses upstream through segments of the same order as start_id.
    If ANY segment in the chain has a divide, returns False.
    If a no-divide connector is encountered, returns False to let other rules handle it.
    If NO segments have divides and no connectors, aggregates all same-order segments,
    then marks ALL upstream (any order) as minor flowpaths.

    Parameters
    ----------
    start_id : str
        Starting flowpath ID (no divide, order 2+)
    current_order : int
        Stream order to follow
    fp_lookup : dict[str, dict[str, Any]]
        Flowpath lookup dictionary
    div_ids : set[str]
        Set of divide IDs
    result : Classifications
        Results container to update
    graph : rx.PyDiGraph
        Network graph
    node_indices : dict[str, int]
        Mapping of flowpath IDs to node indices

    Returns
    -------
    bool
        True if entire chain has no divides and no connectors (aggregated as minor)
        False if any segment in chain has a divide OR if any connector detected
    """
    to_check: deque[str] = deque([start_id])
    chain_ids: list[str] = []
    visited: set[str] = set()
    upstream_boundary: list[str] = []  # Upstream IDs at the boundary (different order)

    while to_check:
        current_fp_id = to_check.popleft()

        if current_fp_id in visited:
            continue

        visited.add(current_fp_id)

        # Get flowpath info
        if current_fp_id not in fp_lookup:
            continue

        stream_order = int(fp_lookup[current_fp_id]["streamorder"])

        # Only follow same order
        if stream_order != current_order:
            # Different order - add to boundary for later processing
            upstream_boundary.append(current_fp_id)
            continue

        # Check if this segment has a divide
        if current_fp_id in div_ids:
            # Found a divide in the chain - don't aggregate
            return False

        # Get upstream to check if this is a no-divide connector
        upstream_ids = _get_upstream_ids(current_fp_id, graph, node_indices)

        # Check if this is a no-divide connector (2+ upstreams)
        if len(upstream_ids) >= 2:
            # This is a connector - return False so other rules can handle it
            # Don't process it here to avoid double-queuing
            return False

        chain_ids.append(current_fp_id)

        # Get upstream and continue checking (single upstream only at this point)
        for up_id in upstream_ids:
            to_check.append(up_id)

    # If we get here, NO segments in the same-order chain have divides or connectors
    # Aggregate all same-order segments and mark as minor
    for chain_id in chain_ids:
        if chain_id != start_id:
            result.aggregation_pairs.append((chain_id, start_id))
        result.minor_flowpaths.add(chain_id)
        result.processed_flowpaths.add(chain_id)

    # Now mark ALL upstream (from boundary) as minor - regardless of order
    for boundary_id in upstream_boundary:
        _mark_all_upstream_as_minor_iterative(boundary_id, result, graph, node_indices)

    return True


def _mark_all_upstream_as_minor_iterative(
    start_id: str,
    result: Classifications,
    graph: rx.PyDiGraph,
    node_indices: dict[str, int],
) -> None:
    """Iteratively mark a flowpath and all upstream as minor flowpaths.

    This uses BFS to mark entire upstream network as minor.
    Used when entire upstream network should be excluded from base hydrofabric.

    Parameters
    ----------
    start_id : str
        The flowpath ID to start marking as minor
    result : Classifications
        Results container to update
    graph : rx.PyDiGraph
        Network graph
    node_indices : dict[str, int]
        Mapping of flowpath IDs to node indices
    """
    to_mark: deque[str] = deque([start_id])

    while to_mark:
        current_fp_id = to_mark.popleft()

        if current_fp_id in result.processed_flowpaths:
            continue

        result.minor_flowpaths.add(current_fp_id)
        result.processed_flowpaths.add(current_fp_id)

        # Get upstream and queue them
        upstream_ids = _get_upstream_ids(current_fp_id, graph, node_indices)
        for up_id in upstream_ids:
            to_mark.append(up_id)


def _rule_independent_large_area(
    current_id: str, fp_info: dict[str, Any], cfg: HFConfig, result: Classifications
) -> bool:
    """Apply rule: Large Area (>threshold) remains independent.

    Large catchments remain independent regardless of stream order
    because they represent significant hydrologic features.

    Parameters
    ----------
    current_id : str
        Current flowpath ID
    fp_info : dict[str, Any]
        Flowpath information
    cfg : HFConfig
        Configuration
    result : Classifications
        Results container

    Returns
    -------
    bool
        True if flowpath is independent due to large area
    """
    if fp_info["areasqkm"] > cfg.divide_aggregation_threshold:
        result.independent_flowpaths.append(current_id)
        result.processed_flowpaths.add(current_id)
        return True

    return False


def _rule_independent_connector(
    current_id: str,
    upstream_info: list[dict[str, Any]],
    cfg: HFConfig,
    graph: rx.PyDiGraph,
    node_indices: dict[str, int],
    result: Classifications,
    div_ids: set[str],
    fp_lookup: dict,
    to_process: deque[str],
) -> bool:
    """Apply rule: Connector where 2+ HIGHER-ORDER streams meet.

    True connectors are confluences of significant streams (order 2+).
    They remain independent regardless of area - critical for network topology.
    All other order 1 streams here become minor flowpaths

    Criteria:
    - Has 2+ upstream segments with order > 1
    - May also have order 1 tributaries (which get aggregated if small)

    Parameters
    ----------
    current_id : str
        Current flowpath ID
    upstream_info : list[dict[str, Any]]
        Upstream flowpath information
    cfg : HFConfig
        Configuration
    graph : rx.PyDiGraph
        Network graph
    node_indices : dict[str, int]
        Mapping of flowpath IDs to node indices
    result : Classifications
        Results container
    div_ids : set[str]
        Set of all divide IDs

    Returns
    -------
    bool
        True if this is a connector segment
    """
    # Separate upstream by stream order
    order_1_upstreams = [info for info in upstream_info if info["streamorder"] == 1]
    higher_order_upstreams = [info for info in upstream_info if info["streamorder"] > 1]

    # Only create independent connectors for >=2 higher-order streams meeting
    if len(higher_order_upstreams) >= 2:
        fp_info = fp_lookup[current_id]
        ds_id = str(int(fp_info["flowpath_toid"]))
        lateral_ids: list = _get_upstream_ids(ds_id, graph, node_indices)
        unprocessed_lateral_info = _get_unprocessed_upstream_info(
            lateral_ids, fp_lookup, result.processed_flowpaths
        )
        all_lateral_info = [fp_lookup[_id] for _id in lateral_ids]
        # if there is a lateral stream that is order 1, we should merge downstream and make the other order 1s minor
        if len(unprocessed_lateral_info) == 1 and len(all_lateral_info) == 2:
            # only works if this is from a two-stream intersection. Can't guarantee basins are next to each other otherwise
            if int(unprocessed_lateral_info[0]["streamorder"]) == 1:
                result.aggregation_pairs.append((current_id, ds_id))
                result.subdivide_candidates.append(current_id)
                lat_id: str = unprocessed_lateral_info[0]["flowpath_id"]
                _traverse_and_mark_as_minor(lat_id, current_id, result, graph, node_indices)
                if lat_id in to_process:
                    to_process.remove(lat_id)
                return True

        # Mark as connector
        result.connector_segments.append(current_id)

        # # Aggregate small order 1 tributaries as their own reaches. Can't me made minor flowpaths given the divide belonging to the flowpath may not be next to the current_id's divide
        # for order1_info in order_1_upstreams:
        #     upstream_id = order1_info["flowpath_id"]
        #     _traverse_and_aggregate_all_upstream(upstream_id, current_id, result, graph, node_indices)

        return True

    return False


def _rule_aggregate_order2_with_order1s(
    current_id: str,
    fp_info: dict[str, Any],
    upstream_info: list[dict[str, Any]],
    graph: rx.PyDiGraph,
    node_indices: dict[str, int],
    result: Classifications,
    div_ids: set[str],
    fp_lookup: dict[str, dict[str, Any]],
    to_process: deque[str],
) -> bool:
    """Apply rule: Order 2 with all Order 1 upstreams (Subdivide Candidate).

    When an order 2 stream has multiple order 1 tributaries joining,
    we aggregate them but mark as a potential subdivision point.

    Parameters
    ----------
    current_id : str
        Current flowpath ID
    fp_info : dict[str, Any]
        Flowpath information
    upstream_info : list[dict[str, Any]]
        Upstream flowpath information
    graph : rx.PyDiGraph
        Network graph
    node_indices : dict[str, int]
        Mapping of flowpath IDs to node indices
    result : Classifications
        Results container
    div_ids : set[str]
        Set of all divide IDs
    fp_lookup : dict[str, dict[str, Any]]
        Flowpath lookup dictionary
    to_process : deque[str]
        Queue for processing flowpaths

    Returns
    -------
    bool
        True if rule applies
    """
    if fp_info["streamorder"] == 2 and all(info["streamorder"] == 1 for info in upstream_info):
        # Check if current has divide
        if current_id not in div_ids:
            # result.no_divide_connectors.append(current_id)
            result.processed_flowpaths.add(current_id)
            return True

        # Find the upstream on the same mainstem
        current_mainstem = fp_info["mainstemlp"]
        on_mainstem = [info for info in upstream_info if info["mainstemlp"] == current_mainstem]
        off_mainstem = [info for info in upstream_info if info["mainstemlp"] != current_mainstem]

        # Aggregate off-mainstem tributaries as minor
        for up_info in off_mainstem:
            upstream_id = up_info["flowpath_id"]
            _traverse_and_mark_as_minor(upstream_id, current_id, result, graph, node_indices)

        # Aggregate on-mainstem flowpaths
        for up_info in on_mainstem:
            upstream_id = up_info["flowpath_id"]
            _traverse_and_aggregate_all_upstream(upstream_id, current_id, result, graph, node_indices)

        return True

    return False


def _rule_aggregate_mixed_upstream_orders(
    current_id: str,
    fp_info: dict[str, Any],
    upstream_info: list[dict[str, Any]],
    cfg: HFConfig,
    result: Classifications,
    div_ids: set[str],
    graph: rx.PyDiGraph,
    node_indices: dict[str, int],
    fp_lookup: dict[str, dict[str, Any]],
    to_process: deque[str],
) -> bool:
    """Apply rule: Mainstem with Order 1 tributaries.

    When a mainstem segment has order 1 tributaries joining,
    aggregate small order 1s as minor flowpaths.

    This is a base case for any connectors that are not handled

    Parameters
    ----------
    current_id : str
        Current flowpath ID
    fp_info : dict[str, Any]
        Flowpath information
    upstream_info : list[dict[str, Any]]
        Upstream flowpath information
    cfg : HFConfig
        Configuration
    result : Classifications
        Results container
    div_ids : set[str]
        Set of all divide IDs
    graph : rx.PyDiGraph
        Network graph
    node_indices : dict[str, int]
        Mapping of flowpath IDs to node indices
    fp_lookup : dict[str, dict[str, Any]]
        Flowpath lookup dictionary
    to_process : deque[str]
        Queue for processing flowpaths

    Returns
    -------
    bool
        True if rule applies
    """
    if fp_info["streamorder"] > 1:
        # Check if current has divide
        if current_id not in div_ids:
            # result.no_divide_connectors.append(current_id)
            result.processed_flowpaths.add(current_id)

            # Get upstream using graph
            upstream_ids = _get_upstream_ids(current_id, graph, node_indices)
            _queue_upstream(upstream_ids, to_process, result.processed_flowpaths, unprocessed_only=True)
            return True

        order_1_upstreams = [info for info in upstream_info if info["streamorder"] == 1]
        same_order_upstreams = [
            info for info in upstream_info if info["streamorder"] == fp_info["streamorder"]
        ]

        # Only apply if we have both order 1 and same-order upstreams
        if not (order_1_upstreams and same_order_upstreams):
            return False

        # Aggregate order 1 tributaries as minor
        for order_1 in order_1_upstreams:
            upstream_id = order_1["flowpath_id"]
            _traverse_and_mark_as_minor(upstream_id, current_id, result, graph, node_indices)

        # Aggregate same-order upstream with area threshold
        for order_n in same_order_upstreams:
            upstream_id = order_n["flowpath_id"]
            cumulative_area = fp_info["areasqkm"]

            _traverse_chain_with_area_threshold(
                start_id=upstream_id,
                current_id=current_id,
                initial_area=cumulative_area,
                threshold=cfg.divide_aggregation_threshold,
                fp_lookup=fp_lookup,
                div_ids=div_ids,
                result=result,
                graph=graph,
                node_indices=node_indices,
                to_process=to_process,
                check_stream_order=True,
                expected_order=fp_info["streamorder"],
            )

        result.upstream_merge_points.append(current_id)
        return True

    return False


# def _rule_aggregate_single_upstream(
#     current_id: str,
#     fp_info: dict[str, Any],
#     upstream_info: list[dict[str, Any]],
#     cfg: HFConfig,
#     result: Classifications,
#     div_ids: set[str],
#     graph: rx.PyDiGraph,
#     node_indices: dict[str, int],
#     fp_lookup: dict[str, dict[str, Any]],
#     to_process: deque[str],
# ) -> bool:
#     """Apply rule: Single upstream aggregation with three behaviors.

#     Handles all single upstream cases:
#     - Current lacks divide → check if upstream is a connector, otherwise aggregate
#     - Order 1 → aggregate all upstream
#     - Higher order → cumulative area threshold aggregation (only if below threshold)
#       - Chains through no-divides automatically
#       - Continues aggregating until area threshold is met

#     Parameters
#     ----------
#     current_id : str
#         Current flowpath ID
#     fp_info : dict[str, Any]
#         Flowpath information
#     upstream_info : list[dict[str, Any]]
#         Upstream flowpath information (should have length 1)
#     cfg : HFConfig
#         Configuration
#     result : Classifications
#         Results container
#     div_ids : set[str]
#         Set of all divide IDs
#     graph : rx.PyDiGraph
#         Network graph
#     node_indices : dict[str, int]
#         Mapping of flowpath IDs to node indices
#     fp_lookup : dict[str, dict[str, Any]]
#         Flowpath lookup dictionary
#     to_process : deque[str]
#         Queue for processing flowpaths

#     Returns
#     -------
#     bool
#         True if rule applies
#     """
#     # Get upstream
#     upstream_ids = _get_upstream_ids(current_id, graph, node_indices)
#     if not upstream_ids:
#         return False

#     upstream_id = upstream_ids[0]

#     # Check if current lacks divide
#     if current_id not in div_ids:
#         # Current has no divide - check if upstream is a connector
#         upstream_upstream_ids = _get_upstream_ids(upstream_id, graph, node_indices)

#         if len(upstream_upstream_ids) >= 2:
#             # Upstream is a connector - aggregate current into it and queue the connector
#             result.aggregation_pairs.append((current_id, upstream_id))
#             result.processed_flowpaths.add(current_id)

#             # Queue the upstream connector for processing
#             if upstream_id not in result.processed_flowpaths:
#                 to_process.append(upstream_id)

#             return True
#         else:
#             # Upstream is not a connector - aggregate both current and upstream
#             result.aggregation_pairs.append((current_id, upstream_id))
#             result.processed_flowpaths.add(current_id)
#             result.processed_flowpaths.add(upstream_id)

#             # Queue upstream's upstream
#             _queue_upstream(
#                 upstream_upstream_ids, to_process, result.processed_flowpaths, unprocessed_only=True
#             )

#             return True

#     # Order 1: Aggregate all upstream
#     if fp_info["streamorder"] == 1:
#         result.processed_flowpaths.add(current_id)

#         # Get upstream
#         if upstream_ids:
#             _traverse_and_aggregate_all_upstream(upstream_id, current_id, result, graph, node_indices)

#         return True

#     # Higher order (>1): Check area threshold before aggregating
#     cumulative_area = fp_info["areasqkm"]

#     # If current segment already exceeds threshold, don't aggregate upstream
#     if cumulative_area >= cfg.divide_aggregation_threshold:
#         result.processed_flowpaths.add(current_id)
#         # Queue upstream for processing as independent units
#         _queue_upstream(upstream_ids, to_process, result.processed_flowpaths, unprocessed_only=True)
#         return True

#     # Current is below threshold - check upstream
#     upstream = upstream_info[0]
#     upstream_id = upstream["flowpath_id"]

#     # Check if upstream has divide
#     if upstream_id not in div_ids:
#         # No divide - chain through it and continue aggregating
#         result.aggregation_pairs.append((current_id, upstream_id))
#         result.processed_flowpaths.add(upstream_id)

#         # Get the next upstream after the no-divide
#         next_upstream_ids = _get_upstream_ids(upstream_id, graph, node_indices)

#         if not next_upstream_ids:
#             # Headwater - done
#             return True

#         if len(next_upstream_ids) > 1:
#             # Multiple upstream (confluence) - queue all
#             for uid in next_upstream_ids:
#                 if uid not in result.processed_flowpaths:
#                     to_process.append(uid)
#             return True

#         # Single upstream - continue checking with area threshold
#         next_upstream_id = next_upstream_ids[0]

#         # Check if next upstream has divide
#         if next_upstream_id not in div_ids:
#             # Another no-divide - recursively handle through chain
#             _traverse_chain_through_no_divides(
#                 start_id=next_upstream_id,
#                 current_id=current_id,
#                 div_ids=div_ids,
#                 result=result,
#                 graph=graph,
#                 node_indices=node_indices,
#                 to_process=to_process,
#             )
#             return True
#         else:
#             # Next upstream has divide - continue with area-based aggregation
#             _traverse_chain_with_area_threshold(
#                 start_id=next_upstream_id,
#                 current_id=current_id,
#                 initial_area=cumulative_area,
#                 threshold=cfg.divide_aggregation_threshold,
#                 fp_lookup=fp_lookup,
#                 div_ids=div_ids,
#                 result=result,
#                 graph=graph,
#                 node_indices=node_indices,
#                 to_process=to_process,
#                 check_stream_order=False,
#                 expected_order=None,
#             )
#             return True

    # # Upstream has divide - chain with cumulative area threshold
    # _traverse_chain_with_area_threshold(
    #     start_id=upstream_id,
    #     current_id=current_id,
    #     initial_area=cumulative_area,
    #     threshold=cfg.divide_aggregation_threshold,
    #     fp_lookup=fp_lookup,
    #     div_ids=div_ids,
    #     result=result,
    #     graph=graph,
    #     node_indices=node_indices,
    #     to_process=to_process,
    #     check_stream_order=False,
    #     expected_order=None,
    # )

    # return True

def _trace_stack(
    start_id: str,
    div_ids: set[str],
    cfg: HFConfig,
    partition_data: dict[str, Any],
) -> Classifications:
    """Trace upstream from a starting flowpath and classify segments according to aggregation rules.

    Rules are applied in order:
    1. No upstream → Independent headwater
    2. Order 1 without divide → No-divide connector (queue upstream)
    3. Order 2+ without divide → Enhanced to handle multiple upstream cases
       - Multiple upstreams with divides (connector):
         a. Mix of order 1 and higher-order → aggregate with higher-order, mark order 1 as minor
         b. All order 2+ → add to no_divide_connectors list
       - Single upstream or no special connector case → check same-order chain logic
    4. Large area → Independent
    5. Nested no-divide connector → Aggregate based on drainage area
    6. Multiple upstream with 2+ higher-order → Connector
    7. Order 2 with all Order 1s → Subdivide candidate
    8. Mixed upstream orders (higher-order + order 1) → Aggregate with rules
    9. Single upstream → Aggregate based on order and area threshold

    Parameters
    ----------
    start_id : str
        The outlet flowpath ID
    div_ids : set[str]
        All IDs from the reference divides dataframe
    cfg : HFConfig
        The Hydrofabric config file
    partition_data : dict[str, Any]
        Contains subgraph, node_indices, and fp_lookup

    Returns
    -------
    Classifications
        A Pydantic BaseModel containing all of the flowpaths and their classifications

    Raises
    ------
    ValueError
        If one of the flowpaths doesn't pass a rule the workflow will be stopped
    """
    digraph: rx.PyDiGraph = partition_data["subgraph"]
    node_indices: dict[str, int] = partition_data["node_indices"]
    fp_lookup: dict[str, dict[str, Any]] = partition_data["fp_lookup"]

    result = Classifications()
    to_process: deque[str] = deque([start_id])
    updated_cumulative_areas: dict[str, float] = {}

    while to_process:
        current_id = to_process.popleft()
        
        if current_id in result.processed_flowpaths:
            continue
            
        fp_info: dict[str, Any] = fp_lookup[current_id]
        upstream_ids = _get_upstream_ids(current_id, digraph, node_indices)
        upstream_info = _get_unprocessed_upstream_info(upstream_ids, fp_lookup, result.processed_flowpaths)
        
        result.processed_flowpaths.add(current_id)

        # Rule 1: No upstream (headwater)
        if not upstream_ids:
            if current_id in div_ids:
                result.independent_flowpaths.append(current_id)
            else:
                result.minor_flowpaths.add(current_id)
            continue

        # Rule 2: Single upstream
        if len(upstream_ids) == 1:
            upstream_id = upstream_ids[0]
            upstream_data = upstream_info[0] if upstream_info else fp_lookup[upstream_id]
            
            # Get cumulative area
            current_area = fp_info["areasqkm"]
            cumulative = updated_cumulative_areas.get(current_id, 0.0) + current_area
            
            # Check if we should aggregate
            if cumulative < cfg.divide_aggregation_threshold:
                # Too small - aggregate
                result.aggregation_pairs.append((current_id, upstream_id))
                updated_cumulative_areas[upstream_id] = cumulative + upstream_data["areasqkm"]
            else:
                # Big enough - independent
                if current_id in div_ids:
                    result.independent_flowpaths.append(current_id)
                # If no divide and big, still aggregate to avoid orphans
                else:
                    result.aggregation_pairs.append((current_id, upstream_id))
            
            _queue_upstream([upstream_id], to_process, result.processed_flowpaths, unprocessed_only=True)
            continue

        # Rule 3: Multiple upstream (connector case)
        if len(upstream_ids) > 1:

            # Case A: Current HAS divide - Connector logic based on stream order
            if current_id in div_ids:
                # Separate by stream order
                order_1_upstreams = [info for info in upstream_info if info["streamorder"] == 1]
                higher_order_upstreams = [info for info in upstream_info if info["streamorder"] > 1]

                if len(higher_order_upstreams) == 0:
                    best_upstream = max(order_1_upstreams, key=lambda x: (x["streamorder"], x["areasqkm"]))
                    result.aggregation_pairs.append((current_id, best_upstream["flowpath_id"]))
                    _queue_upstream([best_upstream["flowpath_id"]], to_process, result.processed_flowpaths, unprocessed_only=True)

                    order_1_upstreams.remove(best_upstream)
                    _ids = [_info["flowpath_id"] for _info in order_1_upstreams]
                    for _id in _ids:
                        _traverse_and_mark_as_minor(_id, current_id, result, digraph, node_indices)
                    continue
                else:
                    if len(upstream_ids) == 2:
                        # Mark all order 1 upstreams as minor
                        for order_1 in order_1_upstreams:
                            upstream_id = order_1["flowpath_id"]
                            _traverse_and_mark_as_minor(upstream_id, current_id, result, digraph, node_indices)

                        # If 2+ higher-order streams meet, this is a connector. else we aggregate to the higher-order
                        if len(higher_order_upstreams) > 1:
                            result.connector_segments.append(current_id)
                        else:
                            result.aggregation_pairs.append((current_id, higher_order_upstreams[0]["flowpath_id"]))
                        # Queue higher-order upstreams
                        higher_order_ids = [info["flowpath_id"] for info in higher_order_upstreams]
                        _queue_upstream(higher_order_ids, to_process, result.processed_flowpaths, unprocessed_only=True)
                        continue
                    else:
                        # 3+ upstream IDs. Mark as connector
                        result.connector_segments.append(current_id)

                        # # Handling the case of where there are 2+ order 1s going into connector
                        # if len(order_1_upstreams) == 1:
                        #     for order_1 in order_1_upstreams:
                        #         upstream_id = order_1["flowpath_id"]
                        #         _traverse_and_mark_as_minor(upstream_id, current_id, result, digraph, node_indices)
                        #     higher_order_ids = [info["flowpath_id"] for info in higher_order_upstreams]
                        #     _queue_upstream(higher_order_ids, to_process, result.processed_flowpaths, unprocessed_only=True)
                        # else:
                        _queue_upstream(upstream_ids, to_process, result.processed_flowpaths, unprocessed_only=True)
                        continue

            # Case B: Current LACKS divide - Rule 3 logic
            else:
                # Get downstream to see if we should aggregate downstream or to best upstream
                ds_id = str(int(fp_info["flowpath_toid"]))
                
                # Check divide status of upstreams
                upstreams_with_divide = [uid for uid in upstream_ids if uid in div_ids]
                upstreams_without_divide = [uid for uid in upstream_ids if uid not in div_ids]
                
                # Subcase 1: All upstreams have divides
                if len(upstreams_without_divide) == 0:
                    # Check if we can aggregate downstream
                    lateral_ids = _get_upstream_ids(ds_id, digraph, node_indices)
                    other_laterals = [lid for lid in lateral_ids if lid != current_id]
                    
                    if len(other_laterals) == 0:
                        # No competition downstream - aggregate down
                        result.aggregation_pairs.append((current_id, ds_id))
                    else:
                        # Competition downstream - aggregate to best upstream
                        best_upstream = max(upstream_info, key=lambda x: (x["streamorder"], x["areasqkm"]))
                        result.aggregation_pairs.append((current_id, best_upstream["flowpath_id"]))
                    
                    # Mark order 1 upstreams as minor
                    for up_info in upstream_info:
                        if up_info["streamorder"] == 1:
                            _traverse_and_mark_as_minor(up_info["flowpath_id"], current_id, result, digraph, node_indices)
                
                # Subcase 2: Mixed divide status
                elif len(upstreams_with_divide) > 0 and len(upstreams_without_divide) > 0:
                    # Aggregate to best upstream WITH divide
                    upstreams_with_divide_info = [info for info in upstream_info if info["flowpath_id"] in upstreams_with_divide]
                    best_upstream = max(upstreams_with_divide_info, key=lambda x: (x["streamorder"], x["areasqkm"]))
                    result.aggregation_pairs.append((current_id, best_upstream["flowpath_id"]))
                    
                    # Mark order 1 upstreams as minor
                    for up_info in upstream_info:
                        if up_info["streamorder"] == 1:
                            _traverse_and_mark_as_minor(up_info["flowpath_id"], current_id, result, digraph, node_indices)
                
                # Subcase 3: All upstreams lack divides
                else:
                    # Check if ANY divides exist 2 layers upstream
                    has_divide_upstream = False
                    for uid in upstream_ids:
                        layer2_ids = _get_upstream_ids(uid, digraph, node_indices)
                        if any(l2_id in div_ids for l2_id in layer2_ids):
                            has_divide_upstream = True
                            break
                    
                    if has_divide_upstream:
                        # Divides exist upstream - aggregate to best upstream
                        best_upstream = max(upstream_info, key=lambda x: (x["streamorder"], x["areasqkm"]))
                        result.aggregation_pairs.append((current_id, best_upstream["flowpath_id"]))
                    else:
                        # No divides anywhere - bad reference data
                        # Check if this is the outlet
                        if start_id == current_id:
                            logger.warning(f"Outlet {start_id} and entire upstream network lack divides - skipping")
                            return result
                        # Not outlet - aggregate everything to downstream
                        result.aggregation_pairs.append((current_id, ds_id))
                        _traverse_and_mark_as_minor(current_id, ds_id, result, digraph, node_indices)
                
                # Queue all upstreams
                _queue_upstream(upstream_ids, to_process, result.processed_flowpaths, unprocessed_only=True)
                continue

        raise ValueError(f"No Rule Matched. Please debug flowpath_id: {current_id}")

    return result


def _trace_single_flowpath_attributes(
    outlet_fp_id: str,
    partition_data: dict[str, Any],
    id_offset: int,
) -> pl.DataFrame:
    """Trace flowpath attributes for a single outlet's drainage basin.

    Parameters
    ----------
    outlet_fp_id : str
        The outlet flowpath ID for this basin
    partition_data : dict[str, Any]
        Contains:
        - "subgraph": rx.PyDiGraph (only this outlet's tree)
        - "node_indices": dict (fp_id -> node index in subgraph)
        - "flowpaths": pl.DataFrame (filtered to this outlet)
        - "fp_lookup": dict (flowpath attributes)
    id_offset : int
        Starting ID for mainstem numbering

    Returns
    -------
    pl.DataFrame
        Updated flowpaths with total_da_sqkm, mainstem_lp, path_length, and dn_hydroseq columns
    """
    basin_graph = partition_data["subgraph"]
    basin_node_indices = partition_data["node_indices"]
    fp_lookup = partition_data["fp_lookup"]

    # Initialize node data in the basin graph
    for node_idx in basin_graph.node_indices():
        fp_id = str(basin_graph[node_idx])

        basin_graph[node_idx] = {
            "fp_id": fp_id,
            "area_sqkm": fp_lookup[fp_id]["area_sqkm"],
            "length_km": fp_lookup[fp_id]["length_km"],
            "hydroseq": fp_lookup[fp_id]["hydroseq"],
            "total_da_sqkm": 0.0,
            "mainstem_lp": None,
            "path_length": 0.0,
            "dn_hydroseq": None,
        }

    outlet_idx = basin_node_indices[outlet_fp_id]

    # Get topological order for this basin
    try:
        topo_order = rx.topological_sort(basin_graph)
    except rx.DAGHasCycle as e:
        raise AssertionError(f"Basin {outlet_fp_id} contains cycles") from e

    # PASS 1: Traverse from OUTLET to ANCESTORS (reverse topo order)
    current_mainstem_id = id_offset

    # Initialize outlet
    basin_graph[outlet_idx]["path_length"] = 0.0
    basin_graph[outlet_idx]["dn_hydroseq"] = 0

    # Calculate path lengths (reverse topo order)
    for node_idx in reversed(topo_order):
        if node_idx == outlet_idx:
            continue

        out_edges = basin_graph.out_edges(node_idx)

        if out_edges:
            downstream_nodes = [tgt_idx for _, tgt_idx, _ in out_edges]

            if downstream_nodes:
                downstream_idx = max(downstream_nodes, key=lambda idx: basin_graph[idx]["path_length"])
                basin_graph[node_idx]["path_length"] = (
                    basin_graph[downstream_idx]["path_length"] + basin_graph[downstream_idx]["length_km"]
                )

    # Trace main mainstem (longest path from outlet to headwater)
    current_idx = outlet_idx
    mainstem_nodes = []

    while True:
        mainstem_nodes.append(current_idx)
        basin_graph[current_idx]["mainstem_lp"] = current_mainstem_id

        in_edges = list(basin_graph.in_edges(current_idx))

        if not in_edges:
            break

        upstream_candidates = [src_idx for src_idx, _, _ in in_edges]

        if not upstream_candidates:
            break

        upstream_idx = max(
            upstream_candidates,
            key=lambda idx: (basin_graph[idx]["path_length"], basin_graph[idx]["total_da_sqkm"]),
        )
        current_idx = upstream_idx

    # Assign tributary mainstems
    tributary_offset = current_mainstem_id + 1
    processed = set(mainstem_nodes)

    for node_idx in basin_graph.node_indices():
        if node_idx not in processed:
            tributary_id = tributary_offset
            tributary_offset += 1

            trib_current = node_idx
            while trib_current not in processed:
                basin_graph[trib_current]["mainstem_lp"] = tributary_id
                processed.add(trib_current)

                in_edges = list(basin_graph.in_edges(trib_current))
                upstream_in_basin = [src_idx for src_idx, _, _ in in_edges]

                if not upstream_in_basin:
                    break

                trib_current = max(
                    upstream_in_basin,
                    key=lambda idx: (basin_graph[idx]["path_length"], basin_graph[idx]["total_da_sqkm"]),
                )

    # Assign dn_hydroseq based on graph edges
    for node_idx in basin_graph.node_indices():
        if node_idx == outlet_idx:
            continue

        out_edges = basin_graph.out_edges(node_idx)
        downstream_nodes = [tgt_idx for _, tgt_idx, _ in out_edges]

        if downstream_nodes:
            downstream_idx = downstream_nodes[0]
            basin_graph[node_idx]["dn_hydroseq"] = basin_graph[downstream_idx]["hydroseq"]
        else:
            basin_graph[node_idx]["dn_hydroseq"] = 0

    # PASS 2: Traverse from ROOT to OUTLET (forward topo order)
    for node_idx in topo_order:
        in_edges = basin_graph.in_edges(node_idx)

        upstream_total = sum(basin_graph[src_idx]["total_da_sqkm"] for src_idx, _, _ in in_edges)

        basin_graph[node_idx]["total_da_sqkm"] = upstream_total + basin_graph[node_idx]["area_sqkm"]

    # Extract results from graph into lists
    fp_ids = []
    total_das = []
    mainstems = []
    path_lengths = []
    dn_hydroseqs = []

    for node_idx in basin_graph.node_indices():
        node_data = basin_graph[node_idx]
        fp_ids.append(node_data["fp_id"])
        total_das.append(node_data["total_da_sqkm"])
        mainstems.append(node_data["mainstem_lp"])
        path_lengths.append(node_data["path_length"])
        dn_hydroseqs.append(node_data["dn_hydroseq"])

    # Create Polars DataFrame with results
    traced_df = pl.DataFrame(
        {
            "fp_id": fp_ids,
            "total_da_sqkm": total_das,
            "mainstem_lp": mainstems,
            "path_length": path_lengths,
            "dn_hydroseq": dn_hydroseqs,
        }
    )

    return traced_df

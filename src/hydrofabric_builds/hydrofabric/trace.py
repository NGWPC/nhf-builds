"""Tracing and classification module for hydrofabric builds."""

import logging
from collections import deque

import polars as pl
import rustworkx as rx

from hydrofabric_builds.config import HFConfig
from hydrofabric_builds.schemas.hydrofabric import Classifications

logger = logging.getLogger(__name__)


def _get_flowpath_info(flowpath_id: str, fp: pl.DataFrame) -> dict:
    """Get flowpath information from the Polars dataframe.

    Parameters
    ----------
    flowpath_id : str
        The flowpath ID to look up
    fp : pl.DataFrame
        The flowpaths Polars dataframe

    Returns
    -------
    dict
        Dictionary containing flowpath attributes
    """
    fp_row = fp.filter(pl.col("flowpath_id") == flowpath_id)

    if fp_row.height == 0:
        raise ValueError(f"Flowpath {flowpath_id} not found in reference data")

    return {
        "flowpath_id": flowpath_id,
        "areasqkm": float(fp_row["areasqkm"][0]),
        "streamorder": int(fp_row["streamorder"][0]),
        "length_km": float(fp_row["lengthkm"][0]),
        "mainstemlp": float(fp_row["mainstemlp"][0] if "mainstemlp" in fp_row.columns else 0),
    }


def _get_unprocessed_upstream_info(upstream_ids: list, fp: pl.DataFrame, processed: set) -> list[dict]:
    """Get info for unprocessed upstream flowpaths.

    Parameters
    ----------
    upstream_ids : list
        List of upstream flowpath IDs
    fp : pl.DataFrame
        The flowpaths Polars dataframe
    processed : set
        Set of already processed flowpath IDs

    Returns
    -------
    list[dict]
        List of flowpath info dictionaries for unprocessed upstreams
    """
    if not upstream_ids:
        return []

    upstream_ids_str = [str(uid) for uid in upstream_ids if str(uid) not in processed]

    if not upstream_ids_str:
        return []

    unprocessed_df = fp.filter(pl.col("flowpath_id").is_in(upstream_ids_str)).select(
        [
            pl.col("flowpath_id"),
            pl.col("areasqkm"),
            pl.col("streamorder"),
            pl.col("lengthkm"),
            pl.when(pl.col("mainstemlp").is_not_null())
            .then(pl.col("mainstemlp"))
            .otherwise(0.0)
            .alias("mainstemlp"),
        ]
    )

    return unprocessed_df.to_dicts()


def _queue_upstream(
    upstream_ids: list, to_process: deque, processed: set, unprocessed_only: bool = False
) -> None:
    """Queue upstream flowpaths for processing.

    Parameters
    ----------
    upstream_ids : list
        List of upstream flowpath IDs
    to_process : deque
        Queue of flowpaths to process
    processed : set
        Set of already processed flowpath IDs
    unprocessed_only : bool
        If True, only queue unprocessed flowpaths
    """
    for uid in upstream_ids:
        uid_str = str(uid)
        if not unprocessed_only or uid_str not in processed:
            to_process.append(uid_str)


def _get_upstream_ids(flowpath_id: str, graph: rx.PyDiGraph, node_indices: dict) -> list[str]:
    """Get upstream flowpath IDs from the graph.

    Parameters
    ----------
    flowpath_id : str
        The flowpath ID to get upstreams for
    graph : rx.PyDiGraph
        Network graph
    node_indices : dict
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
    start_id: str,
    downstream_id: str,
    result: Classifications,
    graph: rx.PyDiGraph,
    node_indices: dict,
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
    node_indices : dict
        Mapping of flowpath IDs to node indices
    """
    upstream_id = start_id

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
    node_indices: dict,
) -> None:
    """Traverse all upstream flowpaths and mark them as minor, aggregating to downstream.

    This marks flowpaths as minor and creates aggregation pairs.
    Used for small tributaries that should be aggregated into larger streams.

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
    node_indices : dict
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

        next_upstream_id = next_upstream_ids[0]
        upstream_id = next_upstream_id


def _traverse_chain_with_area_threshold(
    start_id: str,
    current_id: str,
    initial_area: float,
    threshold: float,
    fp: pl.DataFrame,
    div_ids: set,
    result: Classifications,
    graph: rx.PyDiGraph,
    node_indices: dict,
    to_process: deque,
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
    fp : pl.DataFrame
        Flowpaths dataframe
    div_ids : set
        Set of divide IDs
    result : Classifications
        Results container to update
    graph : rx.PyDiGraph
        Network graph
    node_indices : dict
        Mapping of flowpath IDs to node indices
    to_process : deque
        Queue for processing flowpaths
    check_stream_order : bool
        If True, stop when stream order changes
    expected_order : int | None
        Expected stream order (only used if check_stream_order=True)
    """
    upstream_id = start_id
    cumulative_area = initial_area

    while True:
        # Check if upstream exists
        fp_row = fp.filter(pl.col("flowpath_id") == upstream_id)
        if fp_row.height == 0:
            break

        upstream_fp_info = _get_flowpath_info(upstream_id, fp)
        cumulative_area += upstream_fp_info["areasqkm"]

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
            next_fp_row = fp.filter(pl.col("flowpath_id") == next_upstream_id)
            if next_fp_row.height == 0:
                break

            next_fp_info = _get_flowpath_info(next_upstream_id, fp)
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
    div_ids: set,
    result: Classifications,
    graph: rx.PyDiGraph,
    node_indices: dict,
    to_process: deque,
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
    div_ids : set
        Set of divide IDs
    result : Classifications
        Results container to update
    graph : rx.PyDiGraph
        Network graph
    node_indices : dict
        Mapping of flowpath IDs to node indices
    to_process : deque
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
    div_ids: set,
    graph: rx.PyDiGraph,
    node_indices: dict,
) -> bool:
    """Check if there are ANY divides anywhere in the upstream network.

    Uses BFS to search entire upstream network for any segment with a divide.

    Parameters
    ----------
    start_id : str
        Starting flowpath ID
    div_ids : set
        Set of all divide IDs
    graph : rx.PyDiGraph
        Network graph
    node_indices : dict
        Mapping of flowpath IDs to node indices

    Returns
    -------
    bool
        True if any upstream segment has a divide, False otherwise
    """
    to_check = deque([start_id])
    visited = set()

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
    fp: pl.DataFrame,
    div_ids: set,
    result: Classifications,
    graph: rx.PyDiGraph,
    node_indices: dict,
) -> bool:
    """Check if entire same-order upstream chain has no divides, and aggregate if so.

    Traverses upstream through segments of the same order as start_id.
    If ANY segment in the chain has a divide, returns False.
    If NO segments have divides, aggregates all same-order segments,
    then marks ALL upstream (any order) as minor flowpaths.

    Parameters
    ----------
    start_id : str
        Starting flowpath ID (no divide, order 2+)
    current_order : int
        Stream order to follow
    fp : pl.DataFrame
        Flowpaths dataframe
    div_ids : set
        Set of divide IDs
    result : Classifications
        Results container to update
    graph : rx.PyDiGraph
        Network graph
    node_indices : dict
        Mapping of flowpath IDs to node indices

    Returns
    -------
    bool
        True if entire chain has no divides (aggregated as minor)
        False if any segment in chain has a divide
    """
    to_check = deque([start_id])
    chain_ids = []
    visited = set()
    upstream_boundary = []  # Upstream IDs at the boundary (different order)

    while to_check:
        current_fp_id = to_check.popleft()

        if current_fp_id in visited:
            continue

        visited.add(current_fp_id)

        # Get flowpath info
        fp_row = fp.filter(pl.col("flowpath_id") == current_fp_id)
        if fp_row.height == 0:
            continue

        stream_order = int(fp_row["streamorder"][0])

        # Only follow same order
        if stream_order != current_order:
            # Different order - add to boundary for later processing
            upstream_boundary.append(current_fp_id)
            continue

        # Check if this segment has a divide
        if current_fp_id in div_ids:
            # Found a divide in the chain - don't aggregate
            return False

        chain_ids.append(current_fp_id)

        # Get upstream and continue checking
        upstream_ids = _get_upstream_ids(current_fp_id, graph, node_indices)
        for up_id in upstream_ids:
            to_check.append(up_id)

    # If we get here, NO segments in the same-order chain have divides
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
    node_indices: dict,
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
    node_indices : dict
        Mapping of flowpath IDs to node indices
    """
    to_mark = deque([start_id])

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


def _check_if_order2_chain_has_divides(
    flowpath_id: str,
    fp: pl.DataFrame,
    div_ids: set,
    graph: rx.PyDiGraph,
    node_indices: dict,
) -> tuple[bool, str | None]:
    """Check if an order 2 stream chain has any segments with divides.

    Traverses upstream through order 2 segments to see if any have divides.

    Parameters
    ----------
    flowpath_id : str
        Starting flowpath ID (order 2 without divide)
    fp : pl.DataFrame
        The flowpaths dataframe
    div_ids : set
        Set of all divide IDs
    graph : rx.PyDiGraph
        Network graph
    node_indices : dict
        Mapping of flowpath IDs to node indices

    Returns
    -------
    tuple[bool, str | None]
        (has_divide_in_chain, first_divide_id_found)
        - has_divide_in_chain: True if any order 2 segment upstream has a divide
        - first_divide_id_found: The ID of the first order 2 segment with a divide, or None
    """
    to_check = deque([flowpath_id])
    visited = set()

    while to_check:
        current_fp_id = to_check.popleft()

        if current_fp_id in visited:
            continue

        visited.add(current_fp_id)

        # Get flowpath info
        fp_row = fp.filter(pl.col("flowpath_id") == current_fp_id)
        if fp_row.height == 0:
            continue

        stream_order = int(fp_row["streamorder"][0])

        # Only follow order 2 streams
        if stream_order != 2:
            continue

        # Check if this order 2 segment has a divide
        if current_fp_id in div_ids:
            return (True, current_fp_id)

        # Get upstream and continue checking
        upstream_ids = _get_upstream_ids(current_fp_id, graph, node_indices)
        for up_id in upstream_ids:
            to_check.append(up_id)

    # No order 2 segments with divides found
    return (False, None)


def _rule_independent_large_area(
    current_id: str, fp_info: dict, cfg: HFConfig, result: Classifications
) -> bool:
    """Rule: Large Area (>threshold) remains independent.

    Large catchments remain independent regardless of stream order
    because they represent significant hydrologic features.

    Parameters
    ----------
    current_id : str
        Current flowpath ID
    fp_info : dict
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
        return True

    return False


def _rule_independent_connector(
    current_id: str,
    upstream_info: list[dict],
    cfg: HFConfig,
    graph: rx.PyDiGraph,
    node_indices: dict,
    result: Classifications,
    div_ids: set,
) -> bool:
    """Rule: Connector where 2+ HIGHER-ORDER streams meet.

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
    upstream_info : list[dict]
        Upstream flowpath information
    cfg : HFConfig
        Configuration
    graph : rx.PyDiGraph
        Network graph
    node_indices : dict
        Mapping of flowpath IDs to node indices
    result : Classifications
        Results container
    div_ids : set
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
        if current_id not in div_ids:
            # Classify for later aggregation fixes
            result.no_divide_connectors.append(current_id)
            result.processed_flowpaths.add(current_id)
            return True

        # Mark as connector
        result.connector_segments.append(current_id)

        # Aggregate small order 1 tributaries into the connector
        for order1_info in order_1_upstreams:
            upstream_id = order1_info["flowpath_id"]
            _traverse_and_mark_as_minor(upstream_id, current_id, result, graph, node_indices)

        return True

    return False


def _rule_aggregate_order2_with_order1s(
    current_id: str,
    fp_info: dict,
    upstream_info: list[dict],
    graph: rx.PyDiGraph,
    node_indices: dict,
    result: Classifications,
    div_ids: set,
    fp: pl.DataFrame,
    to_process: deque,
) -> bool:
    """Rule: Order 2 with all Order 1 upstreams (Subdivide Candidate).

    When an order 2 stream has multiple order 1 tributaries joining,
    we aggregate them but mark as a potential subdivision point.

    Parameters
    ----------
    current_id : str
        Current flowpath ID
    fp_info : dict
        Flowpath information
    upstream_info : list[dict]
        Upstream flowpath information
    graph : rx.PyDiGraph
        Network graph
    node_indices : dict
        Mapping of flowpath IDs to node indices
    result : Classifications
        Results container
    div_ids : set
        Set of all divide IDs
    fp : pl.DataFrame
        Flowpaths Polars dataframe
    to_process : deque
        Queue for processing flowpaths

    Returns
    -------
    bool
        True if rule applies
    """
    if fp_info["streamorder"] == 2 and all(info["streamorder"] == 1 for info in upstream_info):
        # Check if current has divide
        if current_id not in div_ids:
            result.no_divide_connectors.append(current_id)
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
    fp_info: dict,
    upstream_info: list[dict],
    cfg: HFConfig,
    result: Classifications,
    div_ids: set,
    graph: rx.PyDiGraph,
    node_indices: dict,
    fp: pl.DataFrame,
    to_process: deque,
) -> bool:
    """Rule: Mainstem with Order 1 tributaries.

    When a mainstem segment has order 1 tributaries joining,
    aggregate small order 1s as minor flowpaths.

    This is a base case for any connectors that are not handled

    Parameters
    ----------
    current_id : str
        Current flowpath ID
    fp_info : dict
        Flowpath information
    upstream_info : list[dict]
        Upstream flowpath information
    cfg : HFConfig
        Configuration
    result : Classifications
        Results container
    div_ids : set
        Set of all divide IDs
    graph : rx.PyDiGraph
        Network graph
    node_indices : dict
        Mapping of flowpath IDs to node indices
    fp : pl.DataFrame
        Flowpaths Polars dataframe
    to_process : deque
        Queue for processing flowpaths

    Returns
    -------
    bool
        True if rule applies
    """
    if fp_info["streamorder"] > 1:
        # Check if current has divide
        if current_id not in div_ids:
            result.no_divide_connectors.append(current_id)
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
                fp=fp,
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


def _rule_aggregate_single_upstream(
    current_id: str,
    fp_info: dict,
    upstream_info: list[dict],
    cfg: HFConfig,
    result: Classifications,
    div_ids: set,
    graph: rx.PyDiGraph,
    node_indices: dict,
    fp: pl.DataFrame,
    to_process: deque,
) -> bool:
    """Rule: Single upstream aggregation with three behaviors.

    Handles all single upstream cases:
    - Current lacks divide → mark as minor
    - Order 1 → aggregate all upstream
    - Higher order → cumulative area threshold aggregation (only if below threshold)
      - Chains through no-divides automatically
      - Continues aggregating until area threshold is met

    Parameters
    ----------
    current_id : str
        Current flowpath ID
    fp_info : dict
        Flowpath information
    upstream_info : list[dict]
        Upstream flowpath information (should have length 1)
    cfg : HFConfig
        Configuration
    result : Classifications
        Results container
    div_ids : set
        Set of all divide IDs
    graph : rx.PyDiGraph
        Network graph
    node_indices : dict
        Mapping of flowpath IDs to node indices
    fp : pl.DataFrame
        Flowpaths Polars dataframe
    to_process : deque
        Queue for processing flowpaths

    Returns
    -------
    bool
        True if rule applies
    """
    # Order 1: Aggregate all upstream
    if fp_info["streamorder"] == 1:
        result.processed_flowpaths.add(current_id)

        # Get upstream
        upstream_ids = _get_upstream_ids(current_id, graph, node_indices)
        if upstream_ids:
            upstream_id = upstream_ids[0]
            _traverse_and_aggregate_all_upstream(upstream_id, current_id, result, graph, node_indices)

        return True

    # Higher order (>1): Check area threshold before aggregating
    cumulative_area = fp_info["areasqkm"]

    # If current segment already exceeds threshold, don't aggregate upstream
    if cumulative_area >= cfg.divide_aggregation_threshold:
        result.processed_flowpaths.add(current_id)
        # Queue upstream for processing as independent units
        upstream_ids = _get_upstream_ids(current_id, graph, node_indices)
        _queue_upstream(upstream_ids, to_process, result.processed_flowpaths, unprocessed_only=True)
        return True

    # Current is below threshold - check upstream
    upstream = upstream_info[0]
    upstream_id = upstream["flowpath_id"]

    # Check if upstream has divide
    if upstream_id not in div_ids:
        # No divide - chain through it and continue aggregating
        result.aggregation_pairs.append((current_id, upstream_id))
        result.processed_flowpaths.add(upstream_id)

        # Get the next upstream after the no-divide
        next_upstream_ids = _get_upstream_ids(upstream_id, graph, node_indices)

        if not next_upstream_ids:
            # Headwater - done
            return True

        if len(next_upstream_ids) > 1:
            # Multiple upstream (confluence) - queue all
            for uid in next_upstream_ids:
                if uid not in result.processed_flowpaths:
                    to_process.append(uid)
            return True

        # Single upstream - continue checking with area threshold
        next_upstream_id = next_upstream_ids[0]

        # Check if next upstream has divide
        if next_upstream_id not in div_ids:
            # Another no-divide - recursively handle through chain
            _traverse_chain_through_no_divides(
                start_id=next_upstream_id,
                current_id=current_id,
                div_ids=div_ids,
                result=result,
                graph=graph,
                node_indices=node_indices,
                to_process=to_process,
            )
            return True
        else:
            # Next upstream has divide - continue with area-based aggregation
            _traverse_chain_with_area_threshold(
                start_id=next_upstream_id,
                current_id=current_id,
                initial_area=cumulative_area,
                threshold=cfg.divide_aggregation_threshold,
                fp=fp,
                div_ids=div_ids,
                result=result,
                graph=graph,
                node_indices=node_indices,
                to_process=to_process,
                check_stream_order=False,
                expected_order=None,
            )
            return True

    # Upstream has divide - chain with cumulative area threshold
    _traverse_chain_with_area_threshold(
        start_id=upstream_id,
        current_id=current_id,
        initial_area=cumulative_area,
        threshold=cfg.divide_aggregation_threshold,
        fp=fp,
        div_ids=div_ids,
        result=result,
        graph=graph,
        node_indices=node_indices,
        to_process=to_process,
        check_stream_order=False,
        expected_order=None,
    )

    return True


def _trace_stack(
    start_id: str,
    fp: pl.DataFrame,
    div_ids: set,
    cfg: HFConfig,
    digraph: rx.PyDiGraph,
    node_indices: dict,
) -> Classifications:
    """Trace upstream from a starting flowpath and classify segments according to aggregation rules.

    Rules are applied in order:
    1. No upstream → Independent headwater
    2. Order 1 without divide → No-divide connector (queue upstream)
    3. Order 2 without divide → Check if order 2 chain has divides upstream
       - If yes: aggregate to upstream with divide
       - If no: mark entire upstream as minor
    4. Order 3+ without divide → No-divide connector (queue upstream)
    5. Large area → Independent
    6. Multiple upstream with 2+ higher-order → Connector
    7. Order 2 with all Order 1s → Subdivide candidate
    8. Mixed upstream orders (higher-order + order 1) → Aggregate with rules
    9. Single upstream → Aggregate based on order and area threshold

    Parameters
    ----------
    start_id : str
        the outlet flowpath ID
    fp : pl.DataFrame
        the reference flowpaths Polars dataframe
    div_ids: set
        all IDs from the reference divides dataframe
    cfg : HFConfig
        the Hydrofabric config file
    digraph : rx.PyDiGraph
        the rustworkx directed graph
    node_indices : dict
        mapping of flowpath IDs to node indices

    Returns
    -------
    Classifications
        A Pydantic BaseModel containing all of the flowpaths and their classifications

    Raises
    ------
    ValueError
        If one of the flowpaths doesn't pass a rule the workflow will be stopped
    """
    result = Classifications()
    to_process = deque([start_id])

    while to_process:
        current_id = to_process.popleft()
        fp_info = _get_flowpath_info(current_id, fp)
        upstream_ids = _get_upstream_ids(current_id, digraph, node_indices)
        upstream_info = _get_unprocessed_upstream_info(upstream_ids, fp, result.processed_flowpaths)

        if len(upstream_info) == 0 and len(upstream_ids) > 0:
            raise ValueError("no upstream info since this segment was mistakingly queued")

        # Rule 1: No Upstream - Independent Headwater
        if not upstream_ids:
            if current_id not in result.processed_flowpaths:
                # Check if has divide, otherwise mark as minor
                if current_id in div_ids:
                    result.independent_flowpaths.append(current_id)
                else:
                    result.minor_flowpaths.add(current_id)
                result.processed_flowpaths.add(current_id)
            continue

        result.processed_flowpaths.add(current_id)

        # Rule 2: Order 1 without divide - aggregate entire upstream chain
        # This is NOT a connector, just an order 1 segment without a divide
        if fp_info["streamorder"] == 1 and current_id not in div_ids:
            result.processed_flowpaths.add(current_id)

            # Aggregate all upstream into this segment
            if upstream_ids:
                upstream_id = upstream_ids[0]
                _traverse_and_aggregate_all_upstream(upstream_id, current_id, result, digraph, node_indices)

            continue

        # Rule 3: Order 2+ without divide - check if entire same-order chain has no divides
        if fp_info["streamorder"] >= 2 and current_id not in div_ids:
            # Check if entire same-order upstream chain has no divides
            chain_has_no_divides = _check_and_aggregate_same_order_no_divide_chain(
                current_id,
                fp_info["streamorder"],
                fp,
                div_ids,
                result,
                digraph,
                node_indices,
            )

            if chain_has_no_divides:
                # Entire chain has no divides - already aggregated and marked as minor
                continue
            else:
                # Some upstream in chain has divide - mark current as connector
                result.no_divide_connectors.append(current_id)
                # Queue upstream for normal processing
                _queue_upstream(upstream_ids, to_process, result.processed_flowpaths, unprocessed_only=True)
                continue

        # Rule 4: Independent - Large Area (regardless of upstream count)
        if _rule_independent_large_area(current_id, fp_info, cfg, result):
            _queue_upstream(upstream_ids, to_process, result.processed_flowpaths, unprocessed_only=True)
            continue

        # Rule 6: Multiple Upstream - Connector Check (2+ higher-order streams meet)
        if len(upstream_info) >= 2:
            if _rule_independent_connector(
                current_id, upstream_info, cfg, digraph, node_indices, result, div_ids
            ):
                _queue_upstream(upstream_ids, to_process, result.processed_flowpaths, unprocessed_only=True)
                continue

        # Rule 7: Order 2 stream with Multiple Order 1 Upstreams (Subdivide Candidate)
        if len(upstream_info) >= 2:
            if _rule_aggregate_order2_with_order1s(
                current_id, fp_info, upstream_info, digraph, node_indices, result, div_ids, fp, to_process
            ):
                _queue_upstream(upstream_ids, to_process, result.processed_flowpaths, unprocessed_only=True)
                continue

        # Rule 8: Mixed Upstream Orders (One upstream is >=2 order, one upstream is order 1)
        if len(upstream_info) >= 2:
            if _rule_aggregate_mixed_upstream_orders(
                current_id,
                fp_info,
                upstream_info,
                cfg,
                result,
                div_ids,
                digraph,
                node_indices,
                fp,
                to_process,
            ):
                continue

        # Rule 9: Single Upstream Aggregation
        if len(upstream_info) == 1:
            if _rule_aggregate_single_upstream(
                current_id,
                fp_info,
                upstream_info,
                cfg,
                result,
                div_ids,
                digraph,
                node_indices,
                fp,
                to_process,
            ):
                continue

        raise ValueError(f"No Rule Matched. Please debug flowpath_id: {current_id}")

    return result

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
    fp_info : dict
        Flowpath information
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
            while True:
                result.minor_flowpaths.add(upstream_id)
                result.aggregation_pairs.append((upstream_id, current_id))
                result.processed_flowpaths.add(upstream_id)

                # Get next upstream using graph
                upstream_idx = node_indices[upstream_id]
                next_upstream_ids = [graph[idx] for idx in graph.predecessor_indices(upstream_idx)]

                # Stop if no upstream (headwater)
                if not next_upstream_ids:
                    break

                next_upstream_id = str(next_upstream_ids[0])
                upstream_id = next_upstream_id

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

        for up_info in off_mainstem:
            upstream_id = up_info["flowpath_id"]
            while True:
                result.minor_flowpaths.add(upstream_id)
                result.aggregation_pairs.append((upstream_id, current_id))
                result.processed_flowpaths.add(upstream_id)

                # Get next upstream using graph
                upstream_idx = node_indices[upstream_id]
                next_upstream_ids = [graph[idx] for idx in graph.predecessor_indices(upstream_idx)]

                # Stop if no upstream (headwater)
                if not next_upstream_ids:
                    break

                next_upstream_id = str(next_upstream_ids[0])
                upstream_id = next_upstream_id

        for up_info in on_mainstem:
            upstream_id = up_info["flowpath_id"]
            while True:
                # Check if upstream exists in dataframe
                fp_row = fp.filter(pl.col("flowpath_id") == upstream_id)
                if fp_row.height == 0:
                    break

                # Create aggregation pair
                result.aggregation_pairs.append((current_id, upstream_id))
                result.processed_flowpaths.add(upstream_id)

                # Get next upstream using graph
                upstream_idx = node_indices[upstream_id]
                next_upstream_ids = [graph[idx] for idx in graph.predecessor_indices(upstream_idx)]

                # Stop if no upstream (headwater)
                if not next_upstream_ids:
                    break

                next_upstream_id = str(next_upstream_ids[0])
                upstream_id = next_upstream_id

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
            id_idx = node_indices[current_id]
            upstream_ids = [graph[idx] for idx in graph.predecessor_indices(id_idx)]
            _queue_upstream(upstream_ids, to_process, result.processed_flowpaths, unprocessed_only=True)
            return True

        order_1_upstreams = [info for info in upstream_info if info["streamorder"] == 1]
        same_order_upstreams = [
            info for info in upstream_info if info["streamorder"] == fp_info["streamorder"]
        ]

        # Only apply if we have both order 1 and same-order upstreams
        if not (order_1_upstreams and same_order_upstreams):
            return False

        for order_1 in order_1_upstreams:
            upstream_id = order_1["flowpath_id"]

            while True:
                result.minor_flowpaths.add(upstream_id)
                result.aggregation_pairs.append((upstream_id, current_id))
                result.processed_flowpaths.add(upstream_id)

                # Get next upstream using graph
                upstream_idx = node_indices[upstream_id]
                next_upstream_ids = [graph[idx] for idx in graph.predecessor_indices(upstream_idx)]

                # Stop if no upstream (headwater)
                if not next_upstream_ids:
                    break

                next_upstream_id = str(next_upstream_ids[0])
                upstream_id = next_upstream_id

        for order_n in same_order_upstreams:
            upstream_id = order_n["flowpath_id"]
            cumulative_area = fp_info["areasqkm"]

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

                # Check if we've exceeded threshold - if so, stop and queue upstream
                if cumulative_area >= cfg.divide_aggregation_threshold:
                    upstream_idx = node_indices[upstream_id]
                    next_upstream_ids = [graph[idx] for idx in graph.predecessor_indices(upstream_idx)]
                    for uid in next_upstream_ids:
                        uid_str = str(uid)
                        if uid_str not in result.processed_flowpaths:
                            to_process.append(uid_str)
                    break

                # Get next upstream using graph
                upstream_idx = node_indices[upstream_id]
                next_upstream_ids = [graph[idx] for idx in graph.predecessor_indices(upstream_idx)]

                # Stop if no upstream (headwater)
                if not next_upstream_ids:
                    break

                # Stop if multiple upstream (connector/confluence) and queue them
                if len(next_upstream_ids) > 1:
                    for uid in next_upstream_ids:
                        uid_str = str(uid)
                        if uid_str not in result.processed_flowpaths:
                            to_process.append(uid_str)
                    break

                # Get next upstream info
                next_upstream_id = str(next_upstream_ids[0])

                # Stop if next upstream has different order
                next_fp_row = fp.filter(pl.col("flowpath_id") == next_upstream_id)
                if next_fp_row.height == 0:
                    break

                next_fp_info = _get_flowpath_info(next_upstream_id, fp)
                if next_fp_info["streamorder"] != fp_info["streamorder"]:
                    # Different order - queue it for processing
                    if next_upstream_id not in result.processed_flowpaths:
                        to_process.append(next_upstream_id)
                    break

                # Stop if next upstream lacks divide
                if next_upstream_id not in div_ids:
                    # Queue it for processing
                    if next_upstream_id not in result.processed_flowpaths:
                        to_process.append(next_upstream_id)
                    break

                # Continue chaining upstream divides
                upstream_id = next_upstream_id

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
    - Higher order → cumulative area threshold aggregation

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

        # Get upstream using graph
        id_idx = node_indices[current_id]
        upstream_ids = [graph[idx] for idx in graph.predecessor_indices(id_idx)]
        upstream_id = str(upstream_ids[0])

        while True:
            result.aggregation_pairs.append((upstream_id, current_id))
            result.processed_flowpaths.add(upstream_id)

            # Get next upstream using graph
            upstream_idx = node_indices[upstream_id]
            next_upstream_ids = [graph[idx] for idx in graph.predecessor_indices(upstream_idx)]

            # Stop if no upstream (headwater)
            if not next_upstream_ids:
                break

            next_upstream_id = str(next_upstream_ids[0])
            upstream_id = next_upstream_id
        return True

    # Higher order: Chain with cumulative area threshold OR until connector
    upstream = upstream_info[0]
    upstream_id = upstream["flowpath_id"]

    # Check if upstream has divide
    if upstream_id not in div_ids:
        # No divide - chain through it
        result.aggregation_pairs.append((current_id, upstream_id))
        result.processed_flowpaths.add(upstream_id)

        # Continue chaining upstream divides through non-divide flowpaths
        while True:
            # Get next upstream using graph
            upstream_idx = node_indices[upstream_id]
            next_upstream_ids = [graph[idx] for idx in graph.predecessor_indices(upstream_idx)]

            # Stop if no upstream (headwater)
            if not next_upstream_ids:
                break

            # Stop if multiple upstream (connector/confluence) - queue all
            if len(next_upstream_ids) > 1:
                for uid in next_upstream_ids:
                    uid_str = str(uid)
                    if uid_str not in result.processed_flowpaths:
                        to_process.append(uid_str)
                break

            # Get next upstream
            next_upstream_id = str(next_upstream_ids[0])

            # Stop if next upstream has a divide - queue it
            if next_upstream_id in div_ids:
                if next_upstream_id not in result.processed_flowpaths:
                    to_process.append(next_upstream_id)
                break

            # Continue chaining upstream divides
            result.aggregation_pairs.append((current_id, next_upstream_id))
            result.processed_flowpaths.add(next_upstream_id)
            upstream_id = next_upstream_id

        return True

    # Upstream has divide - chain with cumulative area threshold
    cumulative_area = fp_info["areasqkm"]

    while True:
        # Get info for current upstream
        fp_row = fp.filter(pl.col("flowpath_id") == upstream_id)
        if fp_row.height == 0:
            break

        upstream_fp_info = _get_flowpath_info(upstream_id, fp)
        cumulative_area += upstream_fp_info["areasqkm"]

        # Create aggregation pair
        result.aggregation_pairs.append((current_id, upstream_id))
        result.processed_flowpaths.add(upstream_id)

        # Check if we've exceeded threshold - stop and queue upstream
        if cumulative_area >= cfg.divide_aggregation_threshold:
            upstream_idx = node_indices[upstream_id]
            next_upstream_ids = [graph[idx] for idx in graph.predecessor_indices(upstream_idx)]
            for uid in next_upstream_ids:
                uid_str = str(uid)
                if uid_str not in result.processed_flowpaths:
                    to_process.append(uid_str)
            break

        # Get next upstream using graph
        upstream_idx = node_indices[upstream_id]
        next_upstream_ids = [graph[idx] for idx in graph.predecessor_indices(upstream_idx)]

        # Stop if no upstream (headwater)
        if not next_upstream_ids:
            break

        # Stop if multiple upstream (connector/confluence) - queue all
        if len(next_upstream_ids) > 1:
            for uid in next_upstream_ids:
                uid_str = str(uid)
                if uid_str not in result.processed_flowpaths:
                    to_process.append(uid_str)
            break

        # Get next upstream info
        next_upstream_id = str(next_upstream_ids[0])

        # Stop if next upstream lacks divide - queue it
        if next_upstream_id not in div_ids:
            if next_upstream_id not in result.processed_flowpaths:
                to_process.append(next_upstream_id)
            break

        # Continue chaining upstream divides
        upstream_id = next_upstream_id

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
    1. Topology-based rules (check upstream count)
       - No upstream → Independent headwater
       - Large area → Independent
       - Multiple higher-order upstream → Connector

    2. Stream order-based rules (specific to general)
       - Order 2 with all Order 1s → Subdivide candidate
       - Same order small area + Order 1s → Edge case
       - Mixed orders (mainstem + tributaries) → Base case

    3. Simple aggregation
       - Single upstream with small area → Aggregate

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
        id_idx = node_indices[current_id]
        upstream_ids = [digraph[idx] for idx in digraph.predecessor_indices(id_idx)]
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

        # Rule 2: Independent - Large Area (regardless of upstream count)
        if _rule_independent_large_area(current_id, fp_info, cfg, result):
            _queue_upstream(upstream_ids, to_process, result.processed_flowpaths, unprocessed_only=True)
            continue

        # Rule 3: Multiple Upstream - Connector Check (2+ higher-order streams meet. May or may not have a stream order 1)
        if len(upstream_info) >= 2:
            if _rule_independent_connector(
                current_id, upstream_info, cfg, digraph, node_indices, result, div_ids
            ):
                _queue_upstream(upstream_ids, to_process, result.processed_flowpaths, unprocessed_only=True)
                continue

        # Rule 4: Order 2 stream with Multiple Order 1 Upstreams (Subdivide Candidate)
        if len(upstream_info) >= 2:
            if _rule_aggregate_order2_with_order1s(
                current_id, fp_info, upstream_info, digraph, node_indices, result, div_ids, fp, to_process
            ):
                _queue_upstream(upstream_ids, to_process, result.processed_flowpaths, unprocessed_only=True)
                continue

        # Rule 5: Mixed Upstream Orders (One upstream is >=2 order, one upstream is order 1)
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

        # Rule 6: Single Upstream Aggregation
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

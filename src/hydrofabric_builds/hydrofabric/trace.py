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
                    _queue_upstream(
                        [best_upstream["flowpath_id"]],
                        to_process,
                        result.processed_flowpaths,
                        unprocessed_only=True,
                    )

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
                            _traverse_and_mark_as_minor(
                                upstream_id, current_id, result, digraph, node_indices
                            )

                        # If 2+ higher-order streams meet, this is a connector. else we aggregate to the higher-order
                        if len(higher_order_upstreams) > 1:
                            result.connector_segments.append(current_id)
                        else:
                            result.aggregation_pairs.append(
                                (current_id, higher_order_upstreams[0]["flowpath_id"])
                            )
                        # Queue higher-order upstreams
                        higher_order_ids = [info["flowpath_id"] for info in higher_order_upstreams]
                        _queue_upstream(
                            higher_order_ids, to_process, result.processed_flowpaths, unprocessed_only=True
                        )
                        continue
                    else:
                        # 3+ upstream IDs. Mark as connector
                        result.connector_segments.append(current_id)
                        _queue_upstream(
                            upstream_ids, to_process, result.processed_flowpaths, unprocessed_only=True
                        )
                        continue

            # Case B: Current LACKS divide - Rule 3 logic
            else:
                # Get downstream to see if we should aggregate downstream or to best upstream
                ds_id = str(int(fp_info["flowpath_toid"]))

                # Step 1: Check if we can aggregate downstream
                lateral_ids = _get_upstream_ids(ds_id, digraph, node_indices)
                other_laterals = [lid for lid in lateral_ids if lid != current_id]
                if len(other_laterals) == 0:
                    result.aggregation_pairs.append((current_id, ds_id))
                    result.connector_segments.append(current_id)
                    for up_info in upstream_info:
                        if up_info["streamorder"] == 1:
                            _traverse_and_mark_as_minor(
                                up_info["flowpath_id"], current_id, result, digraph, node_indices
                            )
                    # Queue non-order-1 upstreams
                    higher_order_ids = [
                        info["flowpath_id"] for info in upstream_info if info["streamorder"] > 1
                    ]
                    _queue_upstream(
                        higher_order_ids, to_process, result.processed_flowpaths, unprocessed_only=True
                    )
                    continue

                # Step 2: Check if both upstreams have no divides in many layers
                all_upstreams_lack_deep_divides = True
                for uid in upstream_ids:
                    layer2_ids = _get_upstream_ids(uid, digraph, node_indices)
                    has_divide_layer2 = any(l2_id in div_ids for l2_id in layer2_ids)
                    if has_divide_layer2:
                        all_upstreams_lack_deep_divides = False
                        break
                    # Check layer 3
                    has_divide_layer3 = False
                    for l2_id in layer2_ids:
                        layer3_ids = _get_upstream_ids(l2_id, digraph, node_indices)
                        if any(l3_id in div_ids for l3_id in layer3_ids):
                            has_divide_layer3 = True
                            break
                    if has_divide_layer3:
                        all_upstreams_lack_deep_divides = False
                        break
                if all_upstreams_lack_deep_divides:
                    result.aggregation_pairs.append((current_id, ds_id))
                    result.minor_flowpaths.add(current_id)
                    # Mark all upstreams as minor
                    for uid in upstream_ids:
                        _traverse_and_mark_as_minor(uid, current_id, result, digraph, node_indices)
                    continue

                # Step 3: Check if there are order 1s or 2s that can be made minor
                order_1_2_upstreams = [info for info in upstream_info if info["streamorder"] <= 2]
                higher_order_upstreams = [info for info in upstream_info if info["streamorder"] > 2]
                if len(order_1_2_upstreams) > 0:
                    # Aggregate to best upstream
                    best_upstream = max(upstream_info, key=lambda x: (x["streamorder"], x["areasqkm"]))
                    result.aggregation_pairs.append((current_id, best_upstream["flowpath_id"]))
                    if len(higher_order_upstreams) > 0:
                        for up_info in order_1_2_upstreams:
                            result.force_queue_flowpaths.add(up_info["flowpath_id"])
                            _traverse_and_mark_as_minor(
                                up_info["flowpath_id"], current_id, result, digraph, node_indices
                            )
                        # Queue non-minor upstreams
                        non_minor_ids = [
                            info["flowpath_id"] for info in upstream_info if info["streamorder"] > 2
                        ]
                        _queue_upstream(
                            non_minor_ids, to_process, result.processed_flowpaths, unprocessed_only=True
                        )
                        continue
                    else:
                        # no higher order upstreams
                        best_upstream = max(
                            order_1_2_upstreams, key=lambda x: (x["streamorder"], x["areasqkm"])
                        )
                        result.aggregation_pairs.append((current_id, best_upstream["flowpath_id"]))
                        order_1_2_upstreams.remove(best_upstream)
                        for up_info in order_1_2_upstreams:
                            _traverse_and_mark_as_minor(
                                up_info["flowpath_id"], current_id, result, digraph, node_indices
                            )
                        _queue_upstream(
                            [best_upstream["flowpath_id"]],
                            to_process,
                            result.processed_flowpaths,
                            unprocessed_only=True,
                        )
                        continue
                else:
                    # This is an awkward connector. Two divides upstream, two downstream, no divide in the flowpath, all upstream are high order
                    result.aggregation_pairs.append((current_id, ds_id))
                    result.connector_segments.append(ds_id)
                    _queue_upstream(
                        upstream_ids, to_process, result.processed_flowpaths, unprocessed_only=True
                    )
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

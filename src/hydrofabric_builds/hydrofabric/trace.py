"""Tracing and classification module for hydrofabric builds."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import polars as pl
import rustworkx as rx

from hydrofabric_builds.config import HFConfig
from hydrofabric_builds.schemas.hydrofabric import Classifications

logger = logging.getLogger(__name__)


class FPInfo(NamedTuple):
    """Flowpath info extracted from lookup."""

    id: str
    order: int
    area: float
    to_id: str


@dataclass
class Context:
    """Immutable trace context shared across processing."""

    graph: rx.PyDiGraph
    node_indices: dict[str, int]
    fp_lookup: dict[str, dict[str, Any]]
    div_ids: set[str]
    threshold: float


@dataclass
class State:
    """Mutable trace state accumulated during processing."""

    queue: list[str] = field(default_factory=list)
    processed: set[str] = field(default_factory=set)
    cumulative_areas: dict[str, float] = field(default_factory=dict)
    independent: set[str] = field(default_factory=set)
    connectors: set[str] = field(default_factory=set)
    non_nextgen: set[str] = field(default_factory=set)
    aggregations: set[tuple[str, str]] = field(default_factory=set)
    aggregation_set: set[str] = field(default_factory=set)
    non_nextgen_virtual_pairs: list[tuple[str, ...]] = field(default_factory=list)
    non_nextgen_virtual_sources: set[str] = field(default_factory=set)
    force_queue: set[str] = field(default_factory=set)


def get_info(ctx: Context, fid: str) -> FPInfo:
    """Get flowpath info from lookup."""
    row = ctx.fp_lookup[fid]
    return FPInfo(
        id=fid,
        order=int(row["streamorder"]),
        area=float(row["areasqkm"]),
        to_id=str(int(row["flowpath_toid"])),
    )


def get_upstreams(ctx: Context, fid: str) -> list[str]:
    """Get immediate upstream flowpath IDs."""
    if fid not in ctx.node_indices:
        return []
    idx = ctx.node_indices[fid]
    return [str(ctx.graph[i]) for i in ctx.graph.predecessor_indices(idx)]


def get_ancestors(ctx: Context, fid: str) -> set[str]:
    """Get all ancestor flowpath IDs."""
    if fid not in ctx.node_indices:
        return set()
    idx = ctx.node_indices[fid]
    return {ctx.graph[i] for i in rx.ancestors(ctx.graph, idx)}


def lacks_deep_divides(ctx: Context, upstreams: list[str]) -> bool:
    """Check if Layer 1 and Layer 2 upstreams lack divides."""
    if any(u in ctx.div_ids for u in upstreams):
        return False
    return not any(u2 in ctx.div_ids for u in upstreams for u2 in get_upstreams(ctx, u))


def find_longest_path_ids(ctx: Context, start_id: str) -> set[str]:
    """Find IDs on the longest path from ancestors to start_id."""
    if start_id not in ctx.node_indices:
        return set()

    idx = ctx.node_indices[start_id]
    anc_indices = rx.ancestors(ctx.graph, idx)
    subgraph = ctx.graph.subgraph(list(anc_indices) + [idx])
    simple_paths = rx.all_pairs_all_simple_paths(subgraph)
    longest = max(
        (p for y in simple_paths.values() for z in y.values() for p in z),
        key=len,
    )
    return {subgraph[i] for i in longest}


def enqueue(st: State, ids: list[str]) -> None:
    """Append unprocessed IDs to the queue."""
    for fid in ids:
        if fid not in st.processed:
            st.queue.append(fid)


def aggregate(st: State, src: str, tgt: str) -> None:
    """Record an aggregation pair."""
    st.aggregations.add((src, tgt))
    st.aggregation_set.update([src, tgt])


def mark_virtual_tree(ctx: Context, st: State, start_id: str, tgt_id: str) -> None:
    """Iteratively mark tree as virtual/aggregated."""
    stack = [start_id]
    while stack:
        curr = stack.pop()
        st.non_nextgen.add(curr)
        aggregate(st, curr, tgt_id)
        st.processed.add(curr)
        if curr not in st.non_nextgen_virtual_sources:
            st.non_nextgen_virtual_sources.add(curr)
            st.non_nextgen_virtual_pairs.append((curr, tgt_id))
        stack.extend(get_upstreams(ctx, curr))


def traverse_and_aggregate(ctx: Context, st: State, start_id: str) -> None:
    """Find longest path; mark others virtual."""
    if start_id not in ctx.node_indices:
        return

    idx = ctx.node_indices[start_id]
    anc_indices = rx.ancestors(ctx.graph, idx)
    anc_ids = [ctx.graph[i] for i in anc_indices] + [start_id]
    longest_ids = find_longest_path_ids(ctx, start_id)
    all_non_nextgen = set(anc_ids).isdisjoint(ctx.div_ids)

    if all_non_nextgen:
        st.non_nextgen.add(start_id)

    queue = [start_id]
    while queue:
        curr = queue.pop(0)
        fp = get_info(ctx, curr)
        st.processed.add(curr)

        if curr not in longest_ids or all_non_nextgen:
            st.non_nextgen.add(curr)
            if curr not in st.non_nextgen_virtual_sources:
                st.non_nextgen_virtual_sources.add(curr)
                st.non_nextgen_virtual_pairs.append((curr, fp.to_id))

        aggregate(st, curr, start_id)
        queue.extend(get_upstreams(ctx, curr))


def handle_headwater(ctx: Context, st: State, curr_id: str, fp: FPInfo) -> None:
    """Handle a headwater flowpath with no upstreams."""
    if curr_id in ctx.div_ids:
        if curr_id not in st.aggregation_set:
            st.independent.add(curr_id)
    else:
        st.non_nextgen.add(curr_id)
        if curr_id not in st.non_nextgen_virtual_sources:
            st.non_nextgen_virtual_sources.add(curr_id)
            st.non_nextgen_virtual_pairs.append((curr_id, fp.to_id))


def handle_single_upstream_flowpath(ctx: Context, st: State, curr_id: str, fp: FPInfo, up_id: str) -> None:
    """Handle a flowpath with exactly one upstream."""
    if curr_id not in ctx.div_ids:
        if fp.order <= 2:
            traverse_and_aggregate(ctx, st, curr_id)
            return

        if lacks_deep_divides(ctx, [up_id]):
            if get_ancestors(ctx, curr_id).isdisjoint(ctx.div_ids):
                if fp.to_id in st.connectors:
                    traverse_and_aggregate(ctx, st, curr_id)
                else:
                    mark_virtual_tree(ctx, st, curr_id, fp.to_id)
                    st.independent.discard(fp.to_id)
                return

    cum_area = st.cumulative_areas.get(curr_id, 0.0) + fp.area

    if fp.area < 0.005:
        aggregate(st, curr_id, up_id)
        if fp.to_id in ctx.fp_lookup:
            st.cumulative_areas[up_id] = ctx.fp_lookup[fp.to_id]["areasqkm"] + fp.area
    elif cum_area < ctx.threshold:
        aggregate(st, curr_id, up_id)
        st.cumulative_areas[up_id] = cum_area
    else:
        if curr_id in ctx.div_ids:
            if up_id not in ctx.div_ids:
                aggregate(st, curr_id, up_id)
            elif curr_id not in st.aggregation_set:
                st.independent.add(curr_id)
        else:
            aggregate(st, curr_id, up_id)

    enqueue(st, [up_id])


def handle_multi_upstream_flowpath(
    ctx: Context, st: State, curr_id: str, fp: FPInfo, upstreams: list[str]
) -> None:
    """Handle a flowpath with multiple upstreams that has a divide."""
    ups_info = [get_info(ctx, u) for u in upstreams if u in ctx.fp_lookup and u not in st.processed]
    order_1 = [u for u in ups_info if u.order == 1]
    higher = [u for u in ups_info if u.order > 1]
    cum_area = st.cumulative_areas.get(curr_id, 0.0) + fp.area

    if not higher:
        best = max(order_1, key=lambda x: (x.order, x.area, x.id))
        if fp.area < 0.005:
            tgt = best.id if fp.to_id == "0" else fp.to_id
            aggregate(st, curr_id, tgt)
            st.independent.discard(tgt)
            if fp.to_id != "0":
                if fp.to_id in st.connectors:
                    st.connectors.remove(fp.to_id)
            enqueue(st, upstreams)
        elif cum_area < ctx.threshold:
            aggregate(st, curr_id, best.id)
            st.cumulative_areas[best.id] = cum_area
            for u in order_1:
                if u.id != best.id:
                    mark_virtual_tree(ctx, st, u.id, curr_id)
            enqueue(st, [best.id])
        else:
            if curr_id not in st.aggregation_set:
                st.connectors.add(curr_id)
            enqueue(st, upstreams)
        return

    if len(upstreams) == 2:
        if len(higher) > 1:
            if curr_id not in st.aggregation_set:
                st.connectors.add(curr_id)
            enqueue(st, upstreams)
        else:
            high_id, o1_id = higher[0].id, order_1[0].id
            if fp.area < 0.005:
                if fp.to_id == "0":
                    aggregate(st, curr_id, high_id)
                    st.independent.discard(high_id)
                    mark_virtual_tree(ctx, st, o1_id, curr_id)
                    enqueue(st, [high_id])
                else:
                    if fp.to_id in st.connectors:
                        st.connectors.remove(fp.to_id)
                    aggregate(st, curr_id, fp.to_id)
                    st.independent.discard(fp.to_id)
                    enqueue(st, upstreams)
            elif cum_area < ctx.threshold:
                aggregate(st, curr_id, high_id)
                st.cumulative_areas[high_id] = cum_area
                mark_virtual_tree(ctx, st, o1_id, curr_id)
                enqueue(st, [h.id for h in higher])
            else:
                if curr_id not in st.aggregation_set:
                    st.connectors.add(curr_id)
                enqueue(st, upstreams)
        return

    # 3+ upstreams
    if fp.area < 0.005:
        if fp.to_id in st.connectors:
            st.connectors.remove(fp.to_id)
        aggregate(st, curr_id, fp.to_id)
        st.independent.discard(fp.to_id)
    elif curr_id not in st.aggregation_set:
        st.connectors.add(curr_id)
    enqueue(st, upstreams)


def handle_multi_upstream_flowpath_no_divide(
    ctx: Context, st: State, curr_id: str, fp: FPInfo, upstreams: list[str]
) -> None:
    """Handle a flowpath with multiple upstreams that lacks a divide."""
    ups_info = [get_info(ctx, u) for u in upstreams if u in ctx.fp_lookup and u not in st.processed]

    if fp.order == 1:
        if fp.to_id in st.connectors:
            traverse_and_aggregate(ctx, st, curr_id)
        else:
            mark_virtual_tree(ctx, st, curr_id, fp.to_id)
            st.independent.discard(fp.to_id)
        return

    if fp.to_id == "0":
        best = max(ups_info, key=lambda x: (x.order, x.area, x.id))
        aggregate(st, curr_id, best.id)
        for u in ups_info:
            if u.id != best.id:
                mark_virtual_tree(ctx, st, u.id, curr_id)
        enqueue(st, [best.id])
        return

    ds_ups = get_upstreams(ctx, fp.to_id)
    other_laterals = [uid for uid in ds_ups if uid != curr_id]

    if not other_laterals:
        if fp.to_id in st.connectors:
            st.connectors.remove(fp.to_id)
        aggregate(st, curr_id, fp.to_id)
        st.independent.discard(fp.to_id)

        all_ups_no_div = all(uid not in ctx.div_ids for uid in upstreams)
        some_ups_no_div = any(uid not in ctx.div_ids for uid in upstreams)

        if all_ups_no_div:
            if get_ancestors(ctx, curr_id).isdisjoint(ctx.div_ids):
                traverse_and_aggregate(ctx, st, curr_id)
                return
            best = max(ups_info, key=lambda x: (x.order, x.area, x.id))
            aggregate(st, curr_id, best.id)
            for uid in upstreams:
                if uid != best.id:
                    st.force_queue.add(uid)
                    mark_virtual_tree(ctx, st, uid, curr_id)
            enqueue(st, [best.id])
            return

        if some_ups_no_div:
            best = max(ups_info, key=lambda x: (x.order, x.area, x.id))
            if best.id not in ctx.div_ids:
                aggregate(st, curr_id, best.id)
                for uid in upstreams:
                    if uid != best.id:
                        st.force_queue.add(uid)
                        mark_virtual_tree(ctx, st, uid, curr_id)
                enqueue(st, [best.id])
                return
            if curr_id not in st.aggregation_set:
                st.connectors.add(curr_id)
            enqueue(st, upstreams)
            return

        # All upstreams have divides
        if curr_id not in st.aggregation_set:
            st.connectors.add(curr_id)
        for u in ups_info:
            if u.order == 1:
                mark_virtual_tree(ctx, st, u.id, curr_id)
        enqueue(st, [u.id for u in ups_info if u.order > 1])
        return

    if lacks_deep_divides(ctx, upstreams):
        if fp.to_id in st.connectors:
            traverse_and_aggregate(ctx, st, curr_id)
        else:
            mark_virtual_tree(ctx, st, curr_id, fp.to_id)
            st.independent.discard(fp.to_id)
        return

    o_1_2 = [u for u in ups_info if u.order <= 2]
    higher = [u for u in ups_info if u.order > 2]

    if o_1_2:
        best = max(ups_info, key=lambda x: (x.order, x.area, x.id))
        aggregate(st, curr_id, best.id)

        if higher:
            for u in o_1_2:
                st.force_queue.add(u.id)
                mark_virtual_tree(ctx, st, u.id, curr_id)
            enqueue(st, [u.id for u in higher])
        else:
            best_sub = max(o_1_2, key=lambda x: (x.order, x.area, x.id))
            aggregate(st, curr_id, best_sub.id)
            for u in o_1_2:
                if u.id != best_sub.id:
                    st.force_queue.add(u.id)
                    mark_virtual_tree(ctx, st, u.id, curr_id)
            enqueue(st, [best_sub.id])
    else:
        if fp.to_id in st.connectors:
            st.connectors.remove(fp.to_id)
        aggregate(st, curr_id, fp.to_id)
        st.independent.discard(fp.to_id)
        enqueue(st, upstreams)


# anomalies


def _anomaly_9272756(ctx: Context, st: State, curr_id: str, ds_id: str) -> None:
    st.aggregation_set.add(curr_id)
    mark_virtual_tree(ctx, st, "9272732", curr_id)
    st.aggregation_set.add("9272732")
    aggregate(st, curr_id, "9272706")
    mark_virtual_tree(ctx, st, "9272688", curr_id)
    aggregate(st, "9272706", "9272686")
    aggregate(st, "9272686", "9270812")
    aggregate(st, "9270812", "9272318")
    st.independent.discard(ds_id)
    enqueue(st, ["9270644", "9272308"])


def _anomaly_7262417(ctx: Context, st: State, curr_id: str, ds_id: str) -> None:
    traverse_and_aggregate(ctx, st, "7262465")
    st.independent.add(curr_id)
    enqueue(st, ["7262413"])


def _anomaly_7262801(ctx: Context, st: State, curr_id: str, ds_id: str) -> None:
    aggregate(st, "7262727", curr_id)
    st.non_nextgen.add("7262683")
    if "7262683" not in st.non_nextgen_virtual_sources:
        st.non_nextgen_virtual_sources.add("7262683")
        st.non_nextgen_virtual_pairs.append(("7262683", "7262727"))
    aggregate(st, "7262727", "7262683")
    st.aggregation_set.update(["7262683", "7262727", curr_id])

    st.non_nextgen.add("7262819")
    aggregate(st, "7262819", "7262727")
    st.aggregation_set.add("7262819")
    if "7262819" not in st.non_nextgen_virtual_sources:
        st.non_nextgen_virtual_sources.add("7262819")
        st.non_nextgen_virtual_pairs.append(("7262819", "7262727"))

    for src, tgt in [("7262727", "7262805"), ("7262805", "7262887"), ("7262887", "7262959")]:
        aggregate(st, src, tgt)
        st.aggregation_set.add(tgt)

    mark_virtual_tree(ctx, st, "7262803", "7262805")
    mark_virtual_tree(ctx, st, "7262933", "7262887")
    enqueue(st, ["940200288"])


def _anomaly_7261789(ctx: Context, st: State, curr_id: str, ds_id: str) -> None:
    mark_virtual_tree(ctx, st, "7261795", curr_id)

    chain = [(curr_id, "7262239"), ("7262239", "7262253"), ("7262253", "7262291")]
    for src, tgt in chain:
        aggregate(st, src, tgt)
        st.aggregation_set.add(tgt)
    st.aggregation_set.update([curr_id, ds_id])

    mark_virtual_tree(ctx, st, "7262255", "7262253")
    enqueue(st, ["7262417"])


def _anomaly_7264167(ctx: Context, st: State, curr_id: str, ds_id: str) -> None:
    aggregate(st, curr_id, "7260693")
    st.aggregation_set.update([curr_id, "7260693"])
    mark_virtual_tree(ctx, st, "7264125", curr_id)

    for src, tgt in [("7260693", "7260373"), ("7260373", "7260303")]:
        aggregate(st, src, tgt)
        st.aggregation_set.add(tgt)

    aggregate(st, "7260373", "7264103")
    st.aggregation_set.add("7264103")
    if "7264103" not in st.non_nextgen_virtual_sources:
        st.non_nextgen_virtual_sources.add("7264103")
        st.non_nextgen_virtual_pairs.append(("7264103", "7260373"))

    enqueue(st, ["7264107"])


def _anomaly_mark_virtual_tree(ctx: Context, st: State, curr_id: str, ds_id: str) -> None:
    mark_virtual_tree(ctx, st, curr_id, ds_id)
    st.independent.discard(ds_id)


def _anomaly_22769238(ctx: Context, st: State, curr_id: str, ds_id: str) -> None:
    mark_virtual_tree(ctx, st, "22769236", curr_id)
    aggregate(st, curr_id, "22769244")
    st.aggregation_set.update([curr_id, "22769244"])
    enqueue(st, ["22769244"])


def _anomaly_7257691(ctx: Context, st: State, curr_id: str, ds_id: str) -> None:
    aggregate(st, curr_id, "7258923")
    aggregate(st, "7258923", "7257829")
    st.aggregation_set.update([curr_id, "7258923", "7257829"])
    enqueue(st, ["7257829"])


def _anomaly_19058436(ctx: Context, st: State, curr_id: str, ds_id: str) -> None:
    mark_virtual_tree(ctx, st, "19058434", curr_id)
    aggregate(st, curr_id, "19058438")
    st.aggregation_set.update([curr_id, "19058438"])
    mark_virtual_tree(ctx, st, "19058412", curr_id)

    for src, tgt in [("19058438", "19058408"), ("19058408", "940180112")]:
        aggregate(st, src, tgt)
        st.aggregation_set.add(tgt)

    traverse_and_aggregate(ctx, st, "19058240")
    traverse_and_aggregate(ctx, st, "940180111")


def _anomaly_21532894(ctx: Context, st: State, curr_id: str, ds_id: str) -> None:
    aggregate(st, curr_id, "21534286")
    st.aggregation_set.update([curr_id, "21534286"])
    mark_virtual_tree(ctx, st, "21532928", curr_id)

    chain = [("21534286", "21532958"), ("21532958", "21532956"), ("21532956", "21533002")]
    trees = ["21532950", "21532954"]
    for (src, tgt), tree in zip(chain[:2], trees, strict=False):
        aggregate(st, src, tgt)
        st.aggregation_set.add(tgt)
        mark_virtual_tree(ctx, st, tree, curr_id)

    aggregate(st, "21532956", "21533002")
    st.aggregation_set.add("21533002")

    aggregate(st, "21532956", "21534452")
    st.aggregation_set.add("21534452")
    st.connectors.add("21534452")
    enqueue(st, ["21534456", "21534462", "21534454"])


def _anomaly_12745197(ctx: Context, st: State, curr_id: str, ds_id: str) -> None:
    for tgt in ["12744163", "12744931", "12745039"]:
        aggregate(st, curr_id, tgt)
        st.aggregation_set.add(tgt)
    st.aggregation_set.add(curr_id)
    enqueue(st, ["12745041"])


def _anomaly_3254269(ctx: Context, st: State, curr_id: str, ds_id: str) -> None:
    aggregate(st, curr_id, ds_id)
    st.aggregation_set.update([curr_id, ds_id])
    st.independent.discard(ds_id)
    aggregate(st, "3254167", "3258317")
    st.aggregation_set.update(["3254167", "3258317"])
    enqueue(st, ["3258317", "3254159"])


def _anomaly_7264077(ctx: Context, st: State, curr_id: str, ds_id: str) -> None:
    aggregate(st, curr_id, "7259909")
    st.aggregation_set.add(curr_id)
    mark_virtual_tree(ctx, st, "7259793", curr_id)
    traverse_and_aggregate(ctx, st, "7259795")


def _anomaly_17493533(ctx: Context, st: State, curr_id: str, ds_id: str) -> None:
    aggregate(st, curr_id, "17493529")
    st.aggregation_set.update([curr_id, "17493529"])
    mark_virtual_tree(ctx, st, "17493279", "17493533")
    aggregate(st, "17493529", "17493261")
    st.aggregation_set.add("17493261")
    enqueue(st, ["17493321", "17493245"])


def _anomaly_12327133(ctx: Context, st: State, curr_id: str, ds_id: str) -> None:
    mark_virtual_tree(ctx, st, "12327137", curr_id)
    aggregate(st, curr_id, "12327125")
    st.aggregation_set.update(["12327125", curr_id])
    enqueue(st, ["12327119"])


def _anomaly_3023064(ctx: Context, st: State, curr_id: str, ds_id: str) -> None:
    mark_virtual_tree(ctx, st, "3023066", curr_id)
    aggregate(st, curr_id, "3023062")
    st.aggregation_set.update([curr_id, "3023062"])
    aggregate(st, "3023012", "3023062")
    st.aggregation_set.add("3023012")
    aggregate(st, "3022994", "3022998")
    st.aggregation_set.update(["3022994", "3022998"])
    enqueue(st, ["3022996", "3022994"])


def _anomaly_5353277(ctx: Context, st: State, curr_id: str, ds_id: str) -> None:
    mark_virtual_tree(ctx, st, "5353283", curr_id)
    st.aggregation_set.add("5353277")
    aggregate(st, curr_id, "5353281")
    st.aggregation_set.add("5353281")
    enqueue(st, ["5352717"])


def _anomaly_traverse_only(ctx: Context, st: State, curr_id: str, ds_id: str) -> None:
    traverse_and_aggregate(ctx, st, curr_id)


def _anomaly_mark_with_agg(ctx: Context, st: State, curr_id: str, ds_id: str) -> None:
    mark_virtual_tree(ctx, st, curr_id, ds_id)
    st.aggregation_set.add(curr_id)


ANOMALY_HANDLERS = {
    "9272756": _anomaly_9272756,
    "7262417": _anomaly_7262417,
    "7262801": _anomaly_7262801,
    "7261789": _anomaly_7261789,
    "7264167": _anomaly_7264167,
    "13257313": _anomaly_mark_virtual_tree,
    "4342468": _anomaly_mark_virtual_tree,
    "22769238": _anomaly_22769238,
    "7257691": _anomaly_7257691,
    "19058436": _anomaly_19058436,
    "21532894": _anomaly_21532894,
    "12745197": _anomaly_12745197,
    "3254269": _anomaly_3254269,
    "7264077": _anomaly_7264077,
    "17493533": _anomaly_17493533,
    "12327133": _anomaly_12327133,
    "3023064": _anomaly_3023064,
    "5353277": _anomaly_5353277,
    **dict.fromkeys(
        [
            "17245948",
            "4386267",
            "8367918",
            "7195127",
            "12778333",
            "12615764",
            "12444133",
            "3053522",
            "18852694",
            "7312267",
        ],
        _anomaly_traverse_only,
    ),
    "8367540": _anomaly_mark_with_agg,
    "14625948": _anomaly_mark_with_agg,
}


def _trace_stack(
    start_id: str,
    div_ids: set[str],
    cfg: HFConfig,
    partition_data: dict[str, Any],
) -> Classifications:
    """Classify hydrofabric flowpaths.

    Args:
        start_id: Starting flowpath ID
        div_ids: Set of divide flowpath IDs
        cfg: Configuration
        partition_data: Dict with subgraph, node_indices, fp_lookup

    Returns
    -------
        Classifications result
    """
    ctx = Context(
        graph=partition_data["subgraph"],
        node_indices=partition_data["node_indices"],
        fp_lookup=partition_data["fp_lookup"],
        div_ids=div_ids,
        threshold=cfg.build.divide_aggregation_threshold,
    )

    st = State(queue=[start_id])

    if div_ids:
        while st.queue:
            curr_id = st.queue.pop(0)
            if curr_id in st.processed:
                continue
            st.processed.add(curr_id)

            if handler := ANOMALY_HANDLERS.get(curr_id):
                handler(ctx, st, curr_id, get_info(ctx, curr_id).to_id)
                continue

            fp = get_info(ctx, curr_id)
            upstreams = get_upstreams(ctx, curr_id)

            match (len(upstreams), curr_id in ctx.div_ids):
                case (0, _):
                    handle_headwater(ctx, st, curr_id, fp)
                case (1, _):
                    handle_single_upstream_flowpath(ctx, st, curr_id, fp, upstreams[0])
                case (_, True):
                    handle_multi_upstream_flowpath(ctx, st, curr_id, fp, upstreams)
                case (_, False):
                    handle_multi_upstream_flowpath_no_divide(ctx, st, curr_id, fp, upstreams)

    res = Classifications()
    res.processed_flowpaths = st.processed
    res.independent_flowpaths = st.independent
    res.connector_segments = list(st.connectors)
    res.non_nextgen_flowpaths = st.non_nextgen
    res.aggregation_pairs = list(st.aggregations)
    res.aggregation_set = st.aggregation_set
    res.non_nextgen_virtual_flowpath_pairs = st.non_nextgen_virtual_pairs
    res.force_queue_flowpaths = st.force_queue
    return res


def _trace_single_flowpath_attributes(
    outlet_fp_id: str, partition_data: dict[str, Any], id_offset: int, hydroseq_offset: int
) -> tuple[pl.DataFrame, int, int]:
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
    hydroseq_offset : int
        Starting hydroseq value

    Returns
    -------
    tuple[pl.DataFrame, int, int]
        Updated flowpaths with total_da_sqkm, mainstem_lp, path_length,
        dn_hydroseq, hydroseq, and stream_order columns; next tributary offset;
        next hydroseq value.
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
            "total_da_sqkm": 0.0,
            "mainstem_lp": None,
            "path_length": 0.0,
            "dn_hydroseq": None,
            "hydroseq": None,
            "streamorder": None,
        }

    outlet_idx = basin_node_indices[outlet_fp_id]

    # Get topological order for this basin
    try:
        topo_order = rx.topological_sort(basin_graph)
    except rx.DAGHasCycle as e:
        raise AssertionError(f"Basin {outlet_fp_id} contains cycles") from e

    # PASS 1: Traverse from OUTLET to ANCESTORS (reverse topo order)
    current_mainstem_id = id_offset
    current_hydroseq = hydroseq_offset

    # Initialize outlet
    basin_graph[outlet_idx]["path_length"] = 0.0
    basin_graph[outlet_idx]["dn_hydroseq"] = 0
    basin_graph[outlet_idx]["hydroseq"] = current_hydroseq
    current_hydroseq += 1

    # Calculate path lengths and hydroseq (reverse topo order)
    for node_idx in reversed(topo_order):
        if node_idx == outlet_idx:
            continue

        # Assign hydroseq (increases going upstream)
        basin_graph[node_idx]["hydroseq"] = current_hydroseq
        current_hydroseq += 1

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

    # Assign dn_hydroseq based on graph edges (now that hydroseq is calculated)
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

    # PASS 2: Traverse from ancestors to OUTLET (forward topo order)
    for node_idx in topo_order:
        in_edges = basin_graph.in_edges(node_idx)

        upstream_total = sum(basin_graph[src_idx]["total_da_sqkm"] for src_idx, _, _ in in_edges)

        basin_graph[node_idx]["total_da_sqkm"] = upstream_total + basin_graph[node_idx]["area_sqkm"]

        # Calculate stream order (Strahler order)
        if not in_edges:
            # Headwater - order 1
            basin_graph[node_idx]["streamorder"] = 1
        else:
            upstream_orders = [basin_graph[src_idx]["streamorder"] for src_idx, _, _ in in_edges]
            max_order = max(upstream_orders)
            count_max = upstream_orders.count(max_order)

            # If two or more streams of same order meet, increment order
            if count_max >= 2:
                basin_graph[node_idx]["streamorder"] = max_order + 1
            else:
                basin_graph[node_idx]["streamorder"] = max_order

    # Extract results from graph into lists
    fp_ids = []
    total_das = []
    mainstems = []
    path_lengths = []
    dn_hydroseqs = []
    hydroseqs = []
    streamorders = []

    for node_idx in basin_graph.node_indices():
        node_data = basin_graph[node_idx]
        fp_ids.append(node_data["fp_id"])
        total_das.append(node_data["total_da_sqkm"])
        mainstems.append(node_data["mainstem_lp"])
        path_lengths.append(node_data["path_length"])
        dn_hydroseqs.append(node_data["dn_hydroseq"])
        hydroseqs.append(node_data["hydroseq"])
        streamorders.append(node_data["streamorder"])

    traced_df = pl.DataFrame(
        {
            "fp_id": fp_ids,
            "total_da_sqkm": total_das,
            "mainstem_lp": mainstems,
            "path_length": path_lengths,
            "dn_hydroseq": dn_hydroseqs,
            "hydroseq": hydroseqs,
            "stream_order": streamorders,
        }
    )

    return traced_df, tributary_offset + 1, current_hydroseq

from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from shapely.ops import unary_union


def build_hydroseq_network(df: pd.DataFrame) -> dict[str, list[str]]:
    """Build network using hydroseq ordering for reliable connectivity

    Parameters
    ----------
    df : pd.DataFrame
        the flowpaths reference

    Returns
    -------
    dict[str, list[str]]
        the reference network connectivity
    """
    upstream_network = defaultdict(list)

    for _, row in df.iterrows():
        fp_id = str(int(float(row["flowpath_id"])))
        dnhydroseq = row["dnhydroseq"]

        # Skip if no downstream connection
        if pd.isna(dnhydroseq) or dnhydroseq == 0:
            continue

        # Find the downstream flowpath
        downstream_mask = df["hydroseq"] == dnhydroseq
        if downstream_mask.any():
            downstream_fp_id = str(int(float(df.loc[downstream_mask, "flowpath_id"].iloc[0])))
            upstream_network[downstream_fp_id].append(fp_id)

    return dict(upstream_network)


def find_outlets_by_hydroseq(df: pd.DataFrame) -> list[str]:
    """Find outlets using hydroseq

    Parameters
    ----------
    df : pd.DataFrame
        the flowpath reference

    Returns
    -------
    list[str]
        all outlets from the reference
    """
    outlets = []

    for _, row in df.iterrows():
        fp_id = str(int(float(row["flowpath_id"])))
        dnhydroseq = row.get("dnhydroseq", 0)

        # Outlet if dnhydroseq is 0 or NaN, or doesn't exist in dataset
        if pd.isna(dnhydroseq) or dnhydroseq == 0:
            outlets.append(fp_id)
        else:
            # Check if downstream flowpath exists in dataset
            downstream_exists = (df["hydroseq"] == dnhydroseq).any()
            if not downstream_exists:
                outlets.append(fp_id)

    return outlets


def get_flowpath_info(fp_id: str, fp_indexed: pd.DataFrame) -> dict:
    """Get basic flowpath information"""
    try:
        fp_row = fp_indexed.loc[int(float(fp_id))]

        return {
            "flowpath_id": fp_id,
            "total_drainage_area_sqkm": fp_row["totdasqkm"],
            "areasqkm": fp_row["areasqkm_left"],  # Local/incremental drainage area
            "length_km": fp_row["lengthkm"],
            "stream_order": fp_row["streamorder"],
            "hydroseq": fp_row["hydroseq"],
            "dnhydroseq": fp_row["dnhydroseq"],
            "geometry": fp_row["geometry"],
        }
    except (KeyError, ValueError) as e:
        print(f"Error getting info for flowpath {fp_id}: {e}")
        # Re-raise the exception to fail fast on bad data
        raise


def find_headwater_group(
    start_id: str, network_graph: dict, fp_indexed: pd.DataFrame, processed_flowpaths: set[str] | None = None
) -> list[str]:
    """Find all flowpaths in a headwater branch starting from a given flowpath.

    Parameters
    ----------
    start_id : str
        the outlet starting flowpath ID
    network_graph : dict
        the network graph object
    fp_indexed : pd.DataFrame
        indexed flowpaths
    processed_flowpaths : set[str], optional
        all processed flowpaths, by default None

    Returns
    -------
    list[str]
        a list of flowpaths
    """
    if processed_flowpaths is None:
        processed_flowpaths = set()

    headwater_group = []
    to_process = deque([start_id])
    local_processed = set()

    while to_process:
        current_id = to_process.popleft()

        # Skip if already processed globally or locally
        if current_id in processed_flowpaths or current_id in local_processed:
            continue

        local_processed.add(current_id)
        headwater_group.append(current_id)

        # Add all upstream segments to the group
        upstream_ids = network_graph.get(current_id, [])
        for upstream_id in upstream_ids:
            if upstream_id not in processed_flowpaths and upstream_id not in local_processed:
                to_process.append(upstream_id)

    return headwater_group


def trace_upstream_with_rules(
    start_id: str,
    network_graph: dict,
    fp_indexed: pd.DataFrame,
    segment_length_threshold: float = 4.0,
    small_catchment_threshold: float = 0.1,
) -> dict:
    """Stack-based upstream trace that identifies single aggregation pairs without chains.

    Parameters
    ----------
    start_id : str
        the outlet flowpath id to start refactoring from
    network_graph : dict[str, list[str]]
        the network connectivity of the reference
    fp : gpd.GeoDataFrame
        the flowpath reference file
    start_id : str
        the outlet flowpath id to start refactoring from
    small_catchment_threshold : float, optional
        the area of the catchment to aggregate into a larger upstream, by default 0.1
    segment_length_threshold : float, optional
        the flowpath length to aggregate by, by default 4.0

    Returns
    -------
    dict
        _description_
    """
    traced_flowpaths = []
    aggregation_pairs: list[tuple[str, ...]] = []
    headwater_groups = []
    independent_flowpaths = []
    minor_flowpaths = []

    # Use a set to track processed flowpaths to prevent chains
    processed_flowpaths = set()
    to_process = deque([start_id])

    print(f"Starting stack-based trace from {start_id}")

    while to_process:
        current_id = to_process.popleft()

        # Skip if already processed
        if current_id in processed_flowpaths:
            print(f"  Skipping {current_id} - already processed")
            continue

        # Mark as processed immediately
        processed_flowpaths.add(current_id)

        fp_info = get_flowpath_info(current_id, fp_indexed)
        traced_flowpaths.append(fp_info)

        print(
            f"  Processing {current_id}: length={fp_info['length_km']:.2f}km, area={fp_info['total_drainage_area_sqkm']:.3f}km², order={fp_info['stream_order']}"
        )

        # Get upstream flowpaths
        upstream_ids = network_graph.get(current_id, [])
        print(f"    Found {len(upstream_ids)} upstream: {upstream_ids}")

        if fp_info["length_km"] >= segment_length_threshold:
            print(
                f"    Segment Length RULE: {fp_info['length_km']:.2f}km >= {segment_length_threshold}km - INDEPENDENT"
            )
            independent_flowpaths.append(current_id)

            # Add all upstream to stack for continued processing
            for upstream_id in upstream_ids:
                if upstream_id not in processed_flowpaths:
                    to_process.append(upstream_id)
            continue

        # If no upstream, it's a headwater
        if not upstream_ids:
            print("    No upstream - adding to headwater group")
            headwater_groups.append([current_id])
            continue

        print(
            f"    Current segment {fp_info['length_km']:.2f}km < {segment_length_threshold}km - applying aggregation rules"
        )

        # making sure we aggregate to the larger upstream segment
        if fp_info["areasqkm"] < small_catchment_threshold:
            print(f"    area {fp_info['areasqkm']:.3f}km² < {small_catchment_threshold}")

            # Get upstream info
            upstream_info: list[dict[str, Any]] = []
            for upstream_id in upstream_ids:
                if upstream_id not in processed_flowpaths:
                    info = get_flowpath_info(upstream_id, fp_indexed)
                    upstream_info.append(info)

            # Find order 2 upstream (preferred)
            order_2_upstream = [info for info in upstream_info if info["stream_order"] == 2]
            order_1_upstream = [info for info in upstream_info if info["stream_order"] == 1]

            if order_2_upstream:
                # Pair with first order 2 upstream
                chosen_upstream = order_2_upstream[0]
                print(f"    PAIR: {current_id} -> {chosen_upstream['flowpath_id']} (order 2)")
                aggregation_pairs.append((current_id, chosen_upstream["flowpath_id"]))

                # Mark chosen upstream as processed and add upstream to stack
                processed_flowpaths.add(chosen_upstream["flowpath_id"])
                chosen_upstream_upstreams = network_graph.get(chosen_upstream["flowpath_id"], [])
                for upstream_of_chosen in chosen_upstream_upstreams:
                    if upstream_of_chosen not in processed_flowpaths:
                        to_process.append(upstream_of_chosen)

                # Handle remaining upstream as headwater groups
                remaining_upstream = [
                    info for info in upstream_info if info["flowpath_id"] != chosen_upstream["flowpath_id"]
                ]
                for info in remaining_upstream:
                    if info["stream_order"] == 1:
                        headwater_group = find_headwater_group(
                            info["flowpath_id"], network_graph, fp_indexed, processed_flowpaths
                        )
                        headwater_groups.append(headwater_group)
                        # Mark all in headwater group as processed
                        for hw_id in headwater_group:
                            processed_flowpaths.add(hw_id)
                    else:
                        # Add to stack for independent processing
                        if info["flowpath_id"] not in processed_flowpaths:
                            to_process.append(info["flowpath_id"])

            elif order_1_upstream:
                # Merge with largest order 1 upstream
                chosen_upstream = max(order_1_upstream, key=lambda x: x["total_drainage_area_sqkm"])
                if chosen_upstream["stream_order"] == 1:
                    chosen_upstream_upstreams = network_graph.get(chosen_upstream["flowpath_id"], [])
                    all_upstream_segments = []
                    while chosen_upstream_upstreams != []:
                        all_upstream_segments.append(chosen_upstream_upstreams)
                        chosen_upstream_upstreams = network_graph.get(chosen_upstream_upstreams[0], [])
                        print(
                            f"    PAIR: {current_id} -> {chosen_upstream['flowpath_id']} + {all_upstream_segments} (largest order 1)"
                        )
                    aggregation_pairs.append(
                        (current_id, chosen_upstream["flowpath_id"]) + tuple(all_upstream_segments[0])
                    )
                else:
                    print(f"    PAIR: {current_id} -> {chosen_upstream['flowpath_id']} (largest order 1)")
                    aggregation_pairs.append((current_id, chosen_upstream["flowpath_id"]))

                    # Mark chosen as processed and add its upstream
                    processed_flowpaths.add(chosen_upstream["flowpath_id"])
                    chosen_upstream_upstreams = network_graph.get(chosen_upstream["flowpath_id"], [])
                    for upstream_of_chosen in chosen_upstream_upstreams:
                        if upstream_of_chosen not in processed_flowpaths:
                            to_process.append(upstream_of_chosen)

                # Handle remaining as headwater groups
                remaining_upstream = [
                    info for info in upstream_info if info["flowpath_id"] != chosen_upstream["flowpath_id"]
                ]
                for info in remaining_upstream:
                    if info["stream_order"] == 1:
                        headwater_group = find_headwater_group(
                            info["flowpath_id"], network_graph, fp_indexed, processed_flowpaths
                        )
                        if headwater_group[0] not in [item for tpl in aggregation_pairs for item in tpl]:
                            headwater_groups.append(headwater_group)
                            for hw_id in headwater_group:
                                processed_flowpaths.add(hw_id)
                                minor_flowpaths.append(hw_id)
            continue

        # RULE 3: DRAINAGE AREA AGGREGATION (normal case)
        if upstream_ids:
            # Get info for unprocessed upstream flowpaths
            upstream_candidates = []
            for upstream_id in upstream_ids:
                if upstream_id not in processed_flowpaths:
                    upstream_info = get_flowpath_info(upstream_id, fp_indexed)
                    upstream_candidates.append(upstream_info)

            if not upstream_candidates:
                print("    All upstream already processed - skipping")
                continue

            # If current flowpath is stream order 1, aggregate entire headwater branch
            if fp_info["stream_order"] == 1:
                print("    ORDER 1 AGGREGATION: Current is order 1 - aggregating entire headwater branch")

                # Find the complete headwater branch starting from current flowpath
                headwater_branch = find_headwater_group(current_id, network_graph, fp_indexed)

                # Remove current_id from the branch since it's already processed
                headwater_branch = [hw_id for hw_id in headwater_branch if hw_id != current_id]

                if headwater_branch:
                    headwater_groups.append([current_id] + headwater_branch)
                    print(f"    HEADWATER BRANCH GROUP: {[current_id] + headwater_branch}")

                    # Mark all as processed
                    for hw_id in headwater_branch:
                        processed_flowpaths.add(hw_id)
                else:
                    # Single headwater - add as group of one
                    headwater_groups.append([current_id])
                    print(f"    SINGLE HEADWATER: {[current_id]}")

                continue

            # Find upstream with largest drainage area
            chosen_upstream = max(upstream_candidates, key=lambda x: x["total_drainage_area_sqkm"])
            print(
                f"    DRAINAGE AREA PAIR: {current_id} -> {chosen_upstream['flowpath_id']} (area: {chosen_upstream['total_drainage_area_sqkm']:.3f}km²)"
            )

            # Record the pair
            aggregation_pairs.append((current_id, chosen_upstream["flowpath_id"]))

            # Mark chosen as processed and add its upstream to stack
            processed_flowpaths.add(chosen_upstream["flowpath_id"])
            chosen_upstream_upstreams = network_graph.get(chosen_upstream["flowpath_id"], [])
            for upstream_of_chosen in chosen_upstream_upstreams:
                if upstream_of_chosen not in processed_flowpaths:
                    to_process.append(upstream_of_chosen)

            # Handle remaining upstream flowpaths
            remaining_upstream = [
                info for info in upstream_candidates if info["flowpath_id"] != chosen_upstream["flowpath_id"]
            ]

            for info in remaining_upstream:
                if info["stream_order"] == 1:
                    # Create headwater group
                    headwater_group = find_headwater_group(
                        info["flowpath_id"], network_graph, fp_indexed, processed_flowpaths
                    )
                    headwater_groups.append(headwater_group)
                    print(f"    HEADWATER GROUP: {headwater_group}")
                    # Mark all as processed
                    for hw_id in headwater_group:
                        processed_flowpaths.add(hw_id)
                else:
                    # Add to stack for independent processing
                    if info["flowpath_id"] not in processed_flowpaths:
                        to_process.append(info["flowpath_id"])
        else:
            print("    No upstream - reached headwater")
            headwater_groups.append([current_id])

    # printing useful trace information for debugging
    print("\n=== TRACE COMPLETE ===")
    print(f"Aggregation pairs: {len(aggregation_pairs)}")
    for pair in aggregation_pairs:
        print(", ".join(item for item in pair))
    print(f"Headwater groups: {len(headwater_groups)}")
    for i, group in enumerate(headwater_groups):
        print(f"  Group {i}: {group}")
    print(f"Independent flowpaths: {len(independent_flowpaths)}")
    print(f"  {independent_flowpaths}")

    return {
        "traced_flowpaths": traced_flowpaths,
        "aggregation_pairs": aggregation_pairs,
        "headwater_groups": headwater_groups,
        "independent_flowpaths": independent_flowpaths,
        "minor_flowpaths": minor_flowpaths,
    }


def aggregate_with_all_rules(
    network_graph: dict[str, list[str]],
    fp: gpd.GeoDataFrame,
    start_id: str,
    small_catchment_threshold: float = 0.1,
    segment_length_threshold: float = 4.0,
) -> dict:
    """Stack-based network aggregation of reference flowpaths

    Parameters
    ----------
    network_graph : dict[str, list[str]]
        the network connectivity of the reference
    fp : gpd.GeoDataFrame
        the flowpath reference file
    start_id : str
        the outlet flowpath id to start refactoring from
    small_catchment_threshold : float, optional
        the area of the catchment to aggregate into a larger upstream, by default 0.1
    segment_length_threshold : float, optional
        the flowpath length to aggregate by, by default 4.0

    Returns
    -------
    dict
        output data
    """
    fp_indexed = fp.set_index("flowpath_id")

    print(f"\n=== Starting aggregation with all rules from outlet {start_id} ===")
    print(f"Small catchment threshold: {small_catchment_threshold} km²")
    print(f"Segment length threshold: {segment_length_threshold} km")

    result = trace_upstream_with_rules(
        start_id, network_graph, fp_indexed, segment_length_threshold, small_catchment_threshold
    )

    traced_flowpaths = result["traced_flowpaths"]

    # Create aggregated unit summary
    total_area = sum(fp["total_drainage_area_sqkm"] for fp in traced_flowpaths)
    total_length = sum(fp["length_km"] for fp in traced_flowpaths)
    flowpath_ids = [fp["flowpath_id"] for fp in traced_flowpaths]

    # Get combined geometry
    geometries = [fp["geometry"] for fp in traced_flowpaths if fp["geometry"] is not None]
    combined_geom = unary_union(geometries) if geometries else None

    # Count stream orders and categorize flowpaths
    order_counts = {}
    small_catchments = []
    long_segments = []
    for fp in traced_flowpaths:
        order = fp["stream_order"]
        order_counts[order] = order_counts.get(order, 0) + 1
        if fp["stream_order"] == 2 and fp["areasqkm"] < small_catchment_threshold:
            small_catchments.append(fp["flowpath_id"])
        if fp["length_km"] >= segment_length_threshold:
            long_segments.append(fp["flowpath_id"])

    final_result = {
        "aggregated_unit": {
            "flowpath_ids": flowpath_ids,
            "flowpath_count": len(flowpath_ids),
            "total_drainage_area_sqkm": total_area,
            "total_length_km": total_length,
            "stream_order_counts": order_counts,
            "small_order_2_catchments": small_catchments,
            "long_segments": long_segments,
            "geometry": combined_geom,
        },
        "flowpath_details": traced_flowpaths,
        "aggregation_pairs": result["aggregation_pairs"],
        "headwater_groups": result["headwater_groups"],
        "independent_flowpaths": result["independent_flowpaths"],
        "minor_flowpaths": result["minor_flowpaths"],
    }

    print("=== Aggregation complete ===")
    print(f"  Processed {len(flowpath_ids)} flowpaths")
    print(f"  Found {len(result['aggregation_pairs'])} aggregation pairs")
    print(f"  Found {len(result['headwater_groups'])} headwater groups")
    print(f"  Found {len(result['independent_flowpaths'])} independent flowpaths")
    print(f"  Total length: {total_length:.2f} km")
    print(f"  Total drainage area: {total_area:.2f} km²")

    return final_result


def debug_network_structure(flowlines: gpd.GeoDataFrame) -> None:
    """Prints information about the network structure for debugging

    Parameters
    ----------
    flowlines : gpd.GeoDataFrame
        the reference flowlines
    """
    print("=== NETWORK STRUCTURE DEBUG ===")

    # Show sample of data
    print("\nSample flowpath data:")
    cols_to_show = ["flowpath_id", "hydroseq", "dnhydroseq", "totdasqkm", "lengthkm", "streamorder"]
    available_cols = [col for col in cols_to_show if col in flowlines.columns]
    print(flowlines[available_cols].head(10))

    # Build network
    network = build_hydroseq_network(flowlines)
    print(f"\nBuilt network with {len(network)} nodes")

    # Show some network connections
    print("\nSample network connections:")
    for _, (downstream_id, upstream_ids) in enumerate(list(network.items())[:5]):
        print(f"  {downstream_id} <- {upstream_ids}")

    # Find outlets
    outlets = find_outlets_by_hydroseq(flowlines)
    print(f"\nFound outlets: {outlets}")


def aggregate_geometries_from_pairs_and_groups(
    flowpaths_gdf: gpd.GeoDataFrame,
    divides_gdf: gpd.GeoDataFrame,
    aggregation_pairs: list[tuple[str, str]],
    headwater_groups: list[list[str]],
    independent_flowpaths: list[str],
    minor_flowpaths: list[str],
) -> dict:
    """Create aggregated geometries from the identified pairs and groups.

    Parameters
    ----------
    flowpaths_gdf : gpd.GeoDataFrame
        GeoDataFrame with flowpath linestrings
    divides_gdf : gpd.GeoDataFrame
        GeoDataFrame with divide polygons/multipolygons
    aggregation_pairs : list[tuple[str, str]]
        List of (downstream_id, upstream_id) pairs
    headwater_groups : list[list[str]]
        List of flowpath ID lists that form complete branches
    independent_flowpaths : list[str]
        List of flowpath IDs that remain independent
    minor_flowpaths : list[str]
        List of flowpath IDs that are to be turned into flowlines

    Returns
    -------
    dict
        output metadata
    """
    print("=== AGGREGATING GEOMETRIES ===")

    # Ensure flowpath_id is string for consistent matching
    flowpaths_gdf = flowpaths_gdf.copy()
    flowpaths_gdf["flowpath_id"] = flowpaths_gdf["flowpath_id"].astype(str).str.replace(".0", "", regex=False)

    divides_gdf = divides_gdf.copy()
    # Assuming divides have a matching ID field - adjust field name as needed
    divide_id_field = "divide_id" if "divide_id" in divides_gdf.columns else "flowpath_id"
    if divide_id_field in divides_gdf.columns:
        divides_gdf[divide_id_field] = (
            divides_gdf[divide_id_field].astype(str).str.replace(".0", "", regex=False)
        )

    # Create lookup dictionaries for fast geometry access
    fp_geom_lookup = {str(row["flowpath_id"]): row["geometry"] for _, row in flowpaths_gdf.iterrows()}

    if divide_id_field in divides_gdf.columns:
        div_geom_lookup = {str(row[divide_id_field]): row["geometry"] for _, row in divides_gdf.iterrows()}
    else:
        print(
            f"Warning: No matching ID field found in divides. Available columns: {list(divides_gdf.columns)}"
        )
        div_geom_lookup = {}

    def create_hydroseq_ordered_geometry(fp_ids, flowpaths_gdf, fp_geom_lookup):
        """Helper function to create unary_union with hydroseq ordering"""
        # Get geometries with their hydroseq for sorting
        fp_geometries_with_hydroseq = []
        for fp_id in fp_ids:
            if fp_id in fp_geom_lookup:
                # Get hydroseq for this flowpath
                fp_row = flowpaths_gdf[flowpaths_gdf["flowpath_id"] == fp_id]
                if len(fp_row) > 0:
                    hydroseq = fp_row["hydroseq"].iloc[0]
                    fp_geometries_with_hydroseq.append((hydroseq, fp_geom_lookup[fp_id]))

        if not fp_geometries_with_hydroseq:
            return None

        # Sort by hydroseq (descending, so lowest hydroseq/most downstream is last)
        fp_geometries_with_hydroseq.sort(key=lambda x: x[0], reverse=True)

        # Extract geometries in correct flow order
        fp_geometries = [geom for hydroseq, geom in fp_geometries_with_hydroseq]

        print(
            f"    Ordering geometries by hydroseq: {[hydroseq for hydroseq, geom in fp_geometries_with_hydroseq]}"
        )

        return unary_union(fp_geometries)

    aggregated_flowpaths = []
    aggregated_divides = []
    aggregation_metadata = {
        "pairs_processed": 0,
        "groups_processed": 0,
        "independent_processed": 0,
        "total_original_flowpaths": len(flowpaths_gdf),
        "total_aggregated_units": 0,
    }

    # Process aggregation pairs (which can now have multiple elements)
    print(f"Processing {len(aggregation_pairs)} aggregation pairs...")
    for i, pair_tuple in enumerate(aggregation_pairs):
        # Handle both 2-element pairs and multi-element groups
        if isinstance(pair_tuple, tuple | list):
            combined_ids = list(pair_tuple)
        else:
            combined_ids = [pair_tuple]

        print(f"  Group {i + 1}: Aggregating {len(combined_ids)} flowpaths: {combined_ids}")

        # Get geometries for all flowpaths in the group
        div_geometries = []

        if div_geom_lookup:
            for fp_id in combined_ids:
                if fp_id in div_geom_lookup:
                    div_geometries.append(div_geom_lookup[fp_id])
                else:
                    print(f"    Warning: Divide {fp_id} not found in geometries")

        # Create aggregated flowpath with hydroseq ordering
        aggregated_fp_geom = create_hydroseq_ordered_geometry(combined_ids, flowpaths_gdf, fp_geom_lookup)

        if aggregated_fp_geom:
            # Create appropriate ID based on group size
            if len(combined_ids) == 2:
                new_fp_id = f"agg_pair_{i + 1}_{combined_ids[0]}_{combined_ids[1]}"
            else:
                new_fp_id = (
                    f"agg_group_{i + 1}_{'_'.join(combined_ids[:3])}{'_etc' if len(combined_ids) > 3 else ''}"
                )

            # Calculate aggregated attributes
            original_fps = flowpaths_gdf[flowpaths_gdf["flowpath_id"].isin(combined_ids)]
            total_length = original_fps["lengthkm"].sum()
            total_drainage = original_fps["totdasqkm"].sum()
            max_stream_order = original_fps["streamorder"].max()

            aggregated_flowpaths.append(
                {
                    "flowpath_id": new_fp_id,
                    "original_ids": combined_ids,
                    "aggregation_type": "pair" if len(combined_ids) == 2 else "multi_group",
                    "lengthkm": total_length,
                    "totdasqkm": total_drainage,
                    "streamorder": max_stream_order,
                    "flowpath_count": len(combined_ids),
                    "geometry": aggregated_fp_geom,
                }
            )

            print(
                f"    Created aggregated flowpath: {total_length:.2f}km, {total_drainage:.2f}km², {len(combined_ids)} segments"
            )

        if div_geometries:
            # Create aggregated divide
            aggregated_div_geom = unary_union(div_geometries)

            # Create appropriate divide ID
            if len(combined_ids) == 2:
                new_div_id = f"agg_div_pair_{i + 1}_{combined_ids[0]}_{combined_ids[1]}"
            else:
                new_div_id = f"agg_div_group_{i + 1}_{'_'.join(combined_ids[:3])}{'_etc' if len(combined_ids) > 3 else ''}"

            aggregated_divides.append(
                {
                    "divide_id": new_div_id,
                    "original_ids": combined_ids,
                    "aggregation_type": "pair" if len(combined_ids) == 2 else "multi_group",
                    "divide_count": len(combined_ids),
                    "geometry": aggregated_div_geom,
                }
            )

        aggregation_metadata["pairs_processed"] += 1

    # Process headwater groups
    print(f"\nProcessing {len(headwater_groups)} headwater groups...")
    for i, group_ids in enumerate(headwater_groups):
        if len(group_ids) == 1:
            print(f"  Group {i + 1}: Single headwater {group_ids[0]} - keeping as individual")
            # Keep single headwaters as individual units
            fp_id = group_ids[0]
            if fp_id in fp_geom_lookup:
                original_fp = flowpaths_gdf[flowpaths_gdf["flowpath_id"] == fp_id].iloc[0]
                aggregated_flowpaths.append(
                    {
                        "flowpath_id": fp_id,
                        "original_ids": [fp_id],
                        "aggregation_type": "headwater_single",
                        "lengthkm": original_fp["lengthkm"],
                        "totdasqkm": original_fp["totdasqkm"],
                        "streamorder": original_fp["streamorder"],
                        "flowpath_count": 1,
                        "geometry": original_fp["geometry"],
                    }
                )
            if div_geom_lookup and fp_id in div_geom_lookup:
                aggregated_divides.append(
                    {
                        "divide_id": fp_id,
                        "original_ids": group_ids,
                        "aggregation_type": "headwater_group",
                        "divide_count": len(group_ids),
                        "geometry": div_geom_lookup[fp_id],
                    }
                )
        else:
            print(f"  Group {i + 1}: Aggregating {len(group_ids)} headwater flowpaths: {group_ids}")

            # Get divide geometries for group
            div_geometries = []
            for fp_id in group_ids:
                if div_geom_lookup and fp_id in div_geom_lookup:
                    div_geometries.append(div_geom_lookup[fp_id])

            # Create aggregated headwater flowpath with hydroseq ordering
            aggregated_fp_geom = create_hydroseq_ordered_geometry(group_ids, flowpaths_gdf, fp_geom_lookup)

            if aggregated_fp_geom:
                new_fp_id = f"agg_headwater_{i + 1}"

                # Calculate aggregated attributes
                original_fps = flowpaths_gdf[flowpaths_gdf["flowpath_id"].isin(group_ids)]
                total_length = original_fps["lengthkm"].sum()
                total_drainage = original_fps["totdasqkm"].sum()
                max_stream_order = original_fps["streamorder"].max()

                aggregated_flowpaths.append(
                    {
                        "flowpath_id": new_fp_id,
                        "original_ids": group_ids,
                        "aggregation_type": "headwater_group",
                        "lengthkm": total_length,
                        "totdasqkm": total_drainage,
                        "streamorder": max_stream_order,
                        "flowpath_count": len(group_ids),
                        "geometry": aggregated_fp_geom,
                    }
                )

                print(f"    Created aggregated headwater: {total_length:.2f}km, {total_drainage:.2f}km²")

            if div_geometries:
                # Create aggregated divide
                aggregated_div_geom = unary_union(div_geometries)
                new_div_id = f"agg_div_headwater_{i + 1}"

                aggregated_divides.append(
                    {
                        "divide_id": new_div_id,
                        "original_ids": group_ids,
                        "aggregation_type": "headwater_group",
                        "divide_count": len(group_ids),
                        "geometry": aggregated_div_geom,
                    }
                )

        aggregation_metadata["groups_processed"] += 1

    # Process independent flowpaths (keep as-is)
    print(f"\nProcessing {len(independent_flowpaths)} independent flowpaths...")
    for fp_id in independent_flowpaths:
        print(f"  Independent: {fp_id}")
        if fp_id in fp_geom_lookup:
            original_fp = flowpaths_gdf[flowpaths_gdf["flowpath_id"] == fp_id].iloc[0]
            aggregated_flowpaths.append(
                {
                    "flowpath_id": f"indep_{fp_id}",
                    "original_ids": [fp_id],
                    "aggregation_type": "independent",
                    "lengthkm": original_fp["lengthkm"],
                    "totdasqkm": original_fp["totdasqkm"],
                    "streamorder": original_fp["streamorder"],
                    "flowpath_count": 1,
                    "geometry": original_fp["geometry"],
                }
            )

        if div_geom_lookup and fp_id in div_geom_lookup:
            original_div = divides_gdf[divides_gdf[divide_id_field] == fp_id].iloc[0]
            aggregated_divides.append(
                {
                    "divide_id": fp_id,
                    "original_ids": [fp_id],
                    "aggregation_type": "independent",
                    "divide_count": 1,
                    "geometry": original_div["geometry"],
                }
            )

        aggregation_metadata["independent_processed"] += 1

    # Process minor flowpaths (keep as-is)
    print(f"\nProcessing {len(minor_flowpaths)} minor flowpaths...")
    for fp_id in minor_flowpaths:
        print(f"  Minor: {fp_id}")
        if fp_id in fp_geom_lookup:
            downstream_hydroseq = flowpaths_gdf[flowpaths_gdf["flowpath_id"] == fp_id]["dnhydroseq"].iloc[0]
            downstream_fp = flowpaths_gdf[flowpaths_gdf["hydroseq"] == downstream_hydroseq][
                "flowpath_id"
            ].iloc[0]
            to_drop = []
            for flowpath in aggregated_flowpaths:
                if flowpath["flowpath_id"] == fp_id:
                    to_drop.append(flowpath)
            for _drop in to_drop:
                aggregated_flowpaths.remove(_drop)
            div_geometries = []
            div_geometries.append(divides_gdf[divides_gdf["flowpath_id"] == fp_id]["geometry"].iloc[0])
            to_drop = []
            original_ids = [fp_id]
            for divide in aggregated_divides:
                if downstream_fp in divide["original_ids"]:
                    original_ids.extend(divide["original_ids"].copy())
                    div_geometries.append(divide["geometry"])
                    to_drop.append(divide)
            for _drop in to_drop:
                aggregated_divides.remove(_drop)
            aggregated_div_geom = unary_union(div_geometries)
            aggregated_divides.append(
                {
                    "divide_id": "fp_id",
                    "original_ids": original_ids,
                    "aggregation_type": "independent",
                    "divide_count": 1,
                    "geometry": aggregated_div_geom,
                }
            )

    # Convert to GeoDataFrames
    aggregated_flowpaths_gdf = gpd.GeoDataFrame(aggregated_flowpaths, crs=flowpaths_gdf.crs)
    aggregated_divides_gdf = (
        gpd.GeoDataFrame(aggregated_divides, crs=divides_gdf.crs)
        if aggregated_divides
        else gpd.GeoDataFrame()
    )

    aggregation_metadata["total_aggregated_units"] = len(aggregated_flowpaths_gdf)

    print("\n=== AGGREGATION SUMMARY ===")
    print(f"Original flowpaths: {aggregation_metadata['total_original_flowpaths']}")
    print(f"Aggregated units: {aggregation_metadata['total_aggregated_units']}")
    print(f"Pairs processed: {aggregation_metadata['pairs_processed']}")
    print(f"Groups processed: {aggregation_metadata['groups_processed']}")
    print(f"Independent processed: {aggregation_metadata['independent_processed']}")
    print(
        f"Reduction: {aggregation_metadata['total_original_flowpaths'] - aggregation_metadata['total_aggregated_units']} flowpaths"
    )

    return {
        "aggregated_flowpaths": aggregated_flowpaths_gdf,
        "aggregated_divides": aggregated_divides_gdf,
        "aggregation_metadata": aggregation_metadata,
    }


def reindex_layers_with_topology(
    flowpaths_gdf: gpd.GeoDataFrame, divides_gdf: gpd.GeoDataFrame, geometry_result: dict
) -> dict[str, gpd.GeoDataFrame]:
    """Reindex aggregated layers and create nexus topology

    Parameters
    ----------
    flowpaths_gdf : _type_
        The flowpath reference
    divides_gdf : _type_
        The divide reference
    geometry_result : dict
        the output of the geometry aggregations

    Returns
    -------
    dict[str, gpd.GeoDataFrame]
        a hydrofabric output
    """
    flowpaths = geometry_result["aggregated_flowpaths"]
    divides = geometry_result["aggregated_divides"]

    print(f"Processing {len(flowpaths)} aggregated flowpaths")
    print(f"Processing {len(divides)} aggregated divides")

    fp_dict = []
    div_dict = []
    nexus_dict = []
    network_dict = []

    _id = 1

    # Create a list of (min_hydroseq, aggregated_unit) for proper ordering
    units_with_hydroseq = []

    for idx, fp_row in flowpaths.iterrows():
        original_ids = fp_row["original_ids"]

        # Get minimum hydroseq for this aggregated unit (most downstream)
        original_fps = flowpaths_gdf[
            flowpaths_gdf["flowpath_id"].isin([float(id_str) for id_str in original_ids])
        ]
        min_hydroseq = original_fps["hydroseq"].min()

        units_with_hydroseq.append((min_hydroseq, idx, fp_row))

    # Sort by hydroseq (most downstream first)
    units_with_hydroseq.sort(key=lambda x: x[0])

    # Pre-analyze to identify confluence points (where multiple flowpaths flow to same downstream)
    confluence_map = {}  # downstream_fp_id -> list of upstream units
    downstream_targets = {}  # unit_index -> downstream_fp_id

    print("Pre-analyzing confluence points...")
    for i, (min_hydroseq, idx, fp_row) in enumerate(units_with_hydroseq):
        original_ids = fp_row["original_ids"]
        original_fps = flowpaths_gdf[
            flowpaths_gdf["flowpath_id"].isin([float(id_str) for id_str in original_ids])
        ]

        if len(original_fps) == 0:
            continue

        # For aggregated units, we need to find the OUTLET of the aggregated unit
        # This is the flowpath within the unit that flows OUTSIDE the unit
        unit_id = i + 1  # This will be the wb-X id

        # Find which original flowpath in this unit flows to a flowpath NOT in this unit
        outlet_fp = None
        for _, orig_fp in original_fps.iterrows():
            dn_hydroseq = orig_fp["dnhydroseq"]
            if pd.isna(dn_hydroseq) or dn_hydroseq == 0:
                # This is a terminal flowpath
                outlet_fp = orig_fp
                break
            else:
                # Check if the downstream flowpath is OUTSIDE this aggregated unit
                downstream_fp = flowpaths_gdf[flowpaths_gdf["hydroseq"] == dn_hydroseq]
                if len(downstream_fp) > 0:
                    downstream_id = str(int(downstream_fp.iloc[0]["flowpath_id"]))
                    if downstream_id not in [str(int(float(x))) for x in original_ids]:
                        # This flowpath flows outside the unit - it's the outlet
                        outlet_fp = orig_fp
                        break

        if outlet_fp is None:
            # Fallback: use most downstream flowpath in unit
            outlet_fp = original_fps.loc[original_fps["hydroseq"].idxmin()]

        dn_hydroseq = outlet_fp["dnhydroseq"]

        # Determine downstream target
        if unit_id == 1 or pd.isna(dn_hydroseq) or dn_hydroseq == 0:
            downstream_fp_id = "wb-0"
        else:
            # Find which unit contains the downstream flowpath
            downstream_original_fp = flowpaths_gdf[flowpaths_gdf["hydroseq"] == dn_hydroseq]
            if len(downstream_original_fp) > 0:
                downstream_original_id = str(int(downstream_original_fp.iloc[0]["flowpath_id"]))

                # Find which unit this belongs to
                found_downstream = False
                for j, (_, _, check_fp_row) in enumerate(units_with_hydroseq):
                    if downstream_original_id in [str(int(float(x))) for x in check_fp_row["original_ids"]]:
                        downstream_unit_position = j + 1  # wb-id position
                        downstream_fp_id = f"wb-{downstream_unit_position}"
                        found_downstream = True
                        break

                if not found_downstream:
                    downstream_fp_id = "wb-0"
            else:
                downstream_fp_id = "wb-0"

        downstream_targets[idx] = downstream_fp_id

        # Group by downstream target
        if downstream_fp_id not in confluence_map:
            confluence_map[downstream_fp_id] = []
        confluence_map[downstream_fp_id].append((min_hydroseq, idx, fp_row, unit_id))

        print(f"    Unit {unit_id} ({fp_row['flowpath_id']}) outlet flows to {downstream_fp_id}")

    # Create shared nexus points for each downstream target
    nexus_assignments = {}  # downstream_target -> nexus_id
    nexus_counter = 1

    print("Creating shared nexus assignments...")
    for downstream_target, upstream_units in confluence_map.items():
        nexus_id = f"nex-{nexus_counter}"
        nexus_assignments[downstream_target] = nexus_id
        print(f"  {nexus_id} -> {downstream_target} (serves {len(upstream_units)} flowpaths)")
        nexus_counter += 1

    print("Processing aggregated units in hydroseq order:")

    for min_hydroseq, idx, fp_row in units_with_hydroseq:
        original_ids = fp_row["original_ids"]

        print(
            f"  Unit {_id}: {fp_row['flowpath_id']} (original IDs: {original_ids}, min hydroseq: {min_hydroseq})"
        )

        # Get original flowpath data for the unit
        original_fps = flowpaths_gdf[
            flowpaths_gdf["flowpath_id"].isin([float(id_str) for id_str in original_ids])
        ]

        if len(original_fps) == 0:
            print(f"    Warning: No original flowpaths found for IDs {original_ids}")
            continue

        # Use the most downstream flowpath for reference attributes
        ref_fp = original_fps.loc[original_fps["hydroseq"].idxmin()]

        # Find corresponding divide(s) - look for the divide that contains these original IDs
        matching_divide = None
        for _div_idx, div_row in divides.iterrows():
            div_original_ids = div_row["original_ids"]
            # Check if there's any overlap between the flowpath original IDs and divide original IDs
            if any(fp_id in div_original_ids for fp_id in original_ids):
                matching_divide = div_row
                break

        if matching_divide is None:
            print(f"    Warning: No matching divide found for flowpath IDs {original_ids}")
            # Create a fallback divide using the original divides
            original_divs = divides_gdf[
                divides_gdf["flowpath_id"].isin([str(int(float(id_str))) for id_str in original_ids])
            ]
            if len(original_divs) > 0:
                total_area = original_divs["areasqkm_left"].sum()
                # Use the geometry from the flowpath aggregation result
                div_geometry = fp_row["geometry"]  # This might not be ideal, but it's a fallback
            else:
                print(f"    Error: No divide data found for {original_ids}")
                continue
        else:
            # Get area from original divides
            original_divs = divides_gdf[
                divides_gdf["flowpath_id"].isin([str(int(float(id_str))) for id_str in original_ids])
            ]
            total_area = original_divs["areasqkm_left"].sum() if len(original_divs) > 0 else 0.0
            div_geometry = matching_divide["geometry"]

        # Get the downstream target and nexus assignment
        downstream_fp_id = downstream_targets[idx]
        nexus_id = nexus_assignments[downstream_fp_id]

        # Create flowpath entry
        current_fp_id = f"wb-{_id}"
        fp_dict.append(
            {
                "fid": _id,
                "flowpath_id": current_fp_id,
                "flowpath_toid": nexus_id,  # Flowpath points to shared nexus
                "reference_ids": original_ids,
                "hydroseq": min_hydroseq,
                "mainstem": ref_fp["mainstemlp"],
                "lengthkm": fp_row["lengthkm"],
                "divide_id": f"cat-{_id}",
                "poi_id": "NULL",
                "vpuid": ref_fp["VPUID"],
                "geometry": fp_row["geometry"],
            }
        )

        # Create divide entry
        div_dict.append(
            {
                "fid": _id,
                "divide_id": f"cat-{_id}",
                "divide_toid": nexus_id,
                "type": "network",
                "ds_id": "NULL",
                "areasqkm": total_area,
                "vpuid": ref_fp["VPUID"],
                "flowpath_id": current_fp_id,
                "lengthkm": fp_row["lengthkm"],
                "has_flowline": True,
                "geometry": div_geometry,
            }
        )

        # Create network entry
        network_dict.append(
            {
                "fid": _id,
                "flowpath_id": current_fp_id,
                "flowpath_toid": nexus_id,
                "mainstem": ref_fp["mainstemlp"],
                "hydroseq": min_hydroseq,
                "lengthkm": fp_row["lengthkm"],
                "divide_id": f"cat-{_id}",
                "poi_id": f"cat-{_id}",
                "vpuid": ref_fp["VPUID"],
                "divide_toid": downstream_fp_id if downstream_fp_id != "wb-0" else f"cat-{_id}",
                "type": "network",
                "areasqkm": total_area,
                "flowline_id": original_ids,
                "hf_part": "NULL",
                "hf_id": "NULL",
                "hf_source": "MVP",
            }
        )

        print(
            f"    Created wb-{_id}, cat-{_id} -> {nexus_id} -> {downstream_fp_id} (area: {total_area:.3f} km²)"
        )
        _id += 1

    # Now create the shared nexus points
    print("\nCreating shared nexus points...")
    nexus_id_counter = 1
    for downstream_target, nexus_id in nexus_assignments.items():
        # Find a representative flowpath that flows to this nexus for geometry
        representative_fp = None
        for fp_entry in fp_dict:
            if fp_entry["flowpath_toid"] == nexus_id:
                representative_fp = fp_entry
                break

        if representative_fp is None:
            print(f"    Warning: No flowpath found for nexus {nexus_id}")
            continue

        # Get the end point of the representative flowpath for nexus location
        fp_geom = representative_fp["geometry"]
        if fp_geom.geom_type == "MultiLineString":
            # Get the last linestring and its last coordinate
            last_line = list(fp_geom.geoms)[-1]
            end_coordinate = last_line.coords[-1]
        else:  # LineString
            end_coordinate = fp_geom.coords[-1]

        nexus_dict.append(
            {
                "fid": nexus_id_counter,
                "nexus_id": nexus_id,
                "nexus_toid": downstream_target,
                "poi_id": "NULL",
                "type": "nexus",
                "vpu_id": "01",
                "geometry": Point(end_coordinate),
            }
        )

        print(f"    Created {nexus_id} -> {downstream_target}")
        nexus_id_counter += 1

    # Create GeoDataFrames
    indexed_flowpaths_gdf = gpd.GeoDataFrame(fp_dict, crs=flowpaths_gdf.crs)
    indexed_divides_gdf = gpd.GeoDataFrame(div_dict, crs=divides_gdf.crs)
    indexed_nexus_gdf = gpd.GeoDataFrame(nexus_dict, crs=flowpaths_gdf.crs)
    indexed_network_gdf = gpd.GeoDataFrame(network_dict)

    print(f"\nCreated {len(indexed_flowpaths_gdf)} indexed flowpaths")
    print(f"Created {len(indexed_divides_gdf)} indexed divides")
    print(f"Created {len(indexed_nexus_gdf)} nexus points")
    print(f"Created {len(indexed_network_gdf)} network entries")

    # Validation checks
    if len(indexed_flowpaths_gdf) != len(indexed_divides_gdf):
        print(
            f"WARNING: Mismatch between flowpaths ({len(indexed_flowpaths_gdf)}) and divides ({len(indexed_divides_gdf)})"
        )

    # Return complete hydrofabric structure
    return {
        "flowpaths": indexed_flowpaths_gdf.drop(columns="reference_ids"),
        "nexus": indexed_nexus_gdf,
        "network": indexed_network_gdf,
        "divides": indexed_divides_gdf,
    }


def test_complete_aggregation_workflow(
    flowpaths_gdf: gpd.GeoDataFrame,
    divides_gdf: gpd.GeoDataFrame,
    segment_length_threshold: float = 4.0,
    small_catchment_threshold: float = 0.1,
) -> dict[str, gpd.GeoDataFrame]:
    """Runs all of the hydrofabric build functions to make a complete end-to-end run

    Parameters
    ----------
    flowpaths_gdf : gpd.GeoDataFrame
        the flowpaths from the v3 reference
    divides_gdf : gpd.GeoDataFrame
        the divides from the v3 reference
    segment_length_threshold : float, optional
        A lengths threshold to determine when to merge upstream
    small_catchment_threshold : float, optional
        a small catchment threshold to determine when to merge

    Returns
    -------
    dict[str, gpd.GeoDataFrame]
        the hydrofabric layers
    """
    print("=== COMPLETE AGGREGATION WORKFLOW ===")

    # Step 1: Build network and find outlets
    print("\nStep 1: Building network graph object...")
    network = build_hydroseq_network(flowpaths_gdf)
    outlets = find_outlets_by_hydroseq(flowpaths_gdf)
    print(f"Found {len(outlets)} outlets")

    # Step 2: Identify aggregation relationships
    print("\nStep 2: Identifying aggregation relationships...")
    all_aggregation_pairs = []
    all_headwater_groups = []
    all_independent_flowpaths = []
    all_minor_flowpaths = []

    for outlet in outlets:
        print(f"  Processing outlet {outlet}...")
        result = aggregate_with_all_rules(
            network_graph=network,
            fp=flowpaths_gdf,
            start_id=outlet,
            segment_length_threshold=segment_length_threshold,
            small_catchment_threshold=small_catchment_threshold,
        )

        all_aggregation_pairs.extend(result["aggregation_pairs"])
        all_headwater_groups.extend(result["headwater_groups"])
        all_independent_flowpaths.extend(result["independent_flowpaths"])
        all_minor_flowpaths.extend(result["minor_flowpaths"])

    print("\nTotal aggregation relationships identified:")
    print(f"  Pairs: {len(all_aggregation_pairs)}")
    print(f"  Headwater groups: {len(all_headwater_groups)}")
    print(f"  Independent: {len(all_independent_flowpaths)}")

    # Step 3: Create aggregated geometries
    print("\nStep 3: Creating aggregated geometries...")
    geometry_result = aggregate_geometries_from_pairs_and_groups(
        flowpaths_gdf=flowpaths_gdf,
        divides_gdf=divides_gdf,
        aggregation_pairs=all_aggregation_pairs,
        headwater_groups=all_headwater_groups,
        independent_flowpaths=all_independent_flowpaths,
        minor_flowpaths=all_minor_flowpaths,
    )

    print("\nStep 3: Creating Nexus Topology and re-index")
    hf = reindex_layers_with_topology(flowpaths_gdf, divides_gdf, geometry_result)
    return hf


if __name__ == "__main__":
    """An entrypoint for running the hydrofabric build workflow using the provided sample data
    """
    fp_gdf = gpd.read_file("cookbooks/hydrofabric/sample_flowpaths.gpkg")
    div_gdf = gpd.read_file("cookbooks/hydrofabric/sample_divides.gpkg")
    output_file = Path("cookbooks/hydrofabric/MVP_NGWPC_hydrofabric.gpkg")

    # optional, but a function that gives information to the user about the structure of the reference
    print("Step 1: Debug network structure")
    debug_network_structure(fp_gdf)

    # the workflow to build the hydrofabric from the sample data
    print("\nStep 2: Test complete aggregation workflow")
    hydrofabric = test_complete_aggregation_workflow(
        flowpaths_gdf=fp_gdf, divides_gdf=div_gdf, segment_length_threshold=4.0, small_catchment_threshold=0.1
    )
    for table_name, _layer in hydrofabric.items():
        if len(_layer) > 0:
            gpd.GeoDataFrame(_layer).to_file(filename=output_file, layer=table_name, driver="GPKG")
        else:
            print(f"Warning: {table_name} layer is empty")

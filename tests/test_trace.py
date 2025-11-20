"""Test cases to ensure functionality works for the trace functions"""

from typing import Any

import pandas as pd
import polars as pl
import rustworkx as rx
from pyprojroot import here

from hydrofabric_builds import (
    HFConfig,
    build_graph,
    download_reference_data,
    map_build_base_hydrofabric,
    map_trace_and_aggregate,
    reduce_calculate_id_ranges,
    reduce_combine_base_hydrofabric,
)
from hydrofabric_builds.hydrofabric.aggregate import _aggregate_geometries
from hydrofabric_builds.hydrofabric.graph import (
    _build_rustworkx_object,
    _build_upstream_dict_from_nexus,
)
from hydrofabric_builds.hydrofabric.trace import _trace_stack
from scripts.hf_runner import LocalRunner


def test_no_divide_fp_upstream_most_reach(trace_case_upstream_no_divide_config: HFConfig) -> None:
    """Testing the tracing output for when there is a no-divide connector at the upstream-most point of a divide"""
    runner = LocalRunner(trace_case_upstream_no_divide_config)
    runner.run_task("download", download_reference_data)
    runner.run_task("build_graph", build_graph)

    outlets: list[str] = runner.ti.xcom_pull(task_id="build_graph", key="outlets")
    outlet_subgraphs: dict[str, dict[str, Any]] = runner.ti.xcom_pull(
        task_id="build_graph", key="outlet_subgraphs"
    )
    outlet = outlets[0]
    partition_data = outlet_subgraphs[outlet]
    filtered_divides = partition_data["divides"]
    valid_divide_ids: set[str] = set(filtered_divides["divide_id"].to_list())
    classifications = _trace_stack(
        start_id=outlet,
        div_ids=valid_divide_ids,
        cfg=trace_case_upstream_no_divide_config,
        partition_data=partition_data,
    )

    aggregate_data = _aggregate_geometries(
        classifications=classifications,
        partition_data=partition_data,
    )

    assert len(aggregate_data.aggregates) == 3, "Incorrect number of aggregates"
    assert len(aggregate_data.independents) == 1, "Incorrect number of independents"
    assert len(aggregate_data.connectors) == 0, "Incorrect number of connectors"
    assert len(aggregate_data.non_nextgen_flowpaths) == 3, "Incorrect number of non nextgen flowpaths"
    assert len(aggregate_data.non_nextgen_virtual_flowpaths) == 1, (
        "Incorrect number of non nextgen virtual flowpaths"
    )
    assert len(aggregate_data.virtual_flowpaths) == 11, "Incorrect number of virtual flowpaths"

    runner.run_task(task_id="map_flowpaths", python_callable=map_trace_and_aggregate, op_kwargs={})
    runner.run_task(task_id="reduce_flowpaths", python_callable=reduce_calculate_id_ranges, op_kwargs={})
    runner.run_task(task_id="map_build_base", python_callable=map_build_base_hydrofabric, op_kwargs={})
    runner.run_task(task_id="reduce_base", python_callable=reduce_combine_base_hydrofabric, op_kwargs={})
    final_flowpaths = runner.ti.xcom_pull(task_id="reduce_base", key="flowpaths")
    final_nexus = runner.ti.xcom_pull(task_id="reduce_base", key="nexus")
    final_virtual_flowpaths = runner.ti.xcom_pull(task_id="reduce_base", key="virtual_flowpaths")
    final_virtual_nexus = runner.ti.xcom_pull(task_id="reduce_base", key="virtual_nexus")

    fp_pl = pl.from_pandas(final_flowpaths.to_wkb())
    upstream_dict = _build_upstream_dict_from_nexus(fp_pl)
    graph, _ = _build_rustworkx_object(upstream_dict)
    assert rx.is_directed_acyclic_graph(graph), "Graph must be acyclic"
    assert rx.is_weakly_connected(graph), "All flowpaths should connect to single outlet"
    assert not rx.is_strongly_connected(graph), "DAG cannot be strongly connected"
    graph_outlets: list[int] = [node for node in graph.node_indices() if graph.out_degree(node) == 0]
    assert len(graph_outlets) == 1, f"Should have exactly 1 outlet, found {len(graph_outlets)}"
    strong_components = rx.strongly_connected_components(graph)
    assert len(strong_components) == graph.num_nodes(), "Each node should be its own SCC in a DAG"

    virtual_fp_pl = pl.from_pandas(final_virtual_flowpaths.to_wkb())
    virtual_upstream_dict = _build_upstream_dict_from_nexus(
        virtual_fp_pl, edge_id="virtual_fp_id", node_id="virtual_nex_id"
    )
    virtual_graph, _ = _build_rustworkx_object(virtual_upstream_dict)
    assert rx.is_directed_acyclic_graph(virtual_graph), "Virtual graph must be acyclic"
    assert rx.is_weakly_connected(virtual_graph), "All virtual flowpaths should connect to single outlet"
    assert not rx.is_strongly_connected(virtual_graph), "Virtual DAG cannot be strongly connected"
    virtual_outlets = [node for node in virtual_graph.node_indices() if virtual_graph.out_degree(node) == 0]
    assert len(virtual_outlets) == 1, f"Should have exactly 1 virtual outlet, found {len(virtual_outlets)}"
    virtual_strong_components = rx.strongly_connected_components(virtual_graph)
    assert len(virtual_strong_components) == virtual_graph.num_nodes(), (
        "Each virtual node should be its own SCC"
    )

    df = pd.DataFrame(
        {
            "nex_id": pd.Series([2, 3, 4], dtype="int64"),
            "dn_fp_id": pd.Series([pd.NA, 2, 3], dtype="Int64"),
        }
    )
    pd.testing.assert_frame_equal(final_nexus.drop(columns=["geometry"]), df)

    df = pd.DataFrame(
        {
            "virtual_nex_id": pd.Series([2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype="Int64"),
            "dn_virtual_fp_id": pd.Series([pd.NA, 2, 3, 4, 5, 6, 7, 8, 9, 11], dtype="Int64"),
        }
    )
    pd.testing.assert_frame_equal(final_virtual_nexus.drop(columns=["geometry"]), df)


def test_no_divide_coastal_outlet(trace_case_no_divide_coastal_outlet: HFConfig) -> None:
    """Testing the tracing output for when there is a no-divide connector at the upstream-most point of a divide"""
    runner = LocalRunner(trace_case_no_divide_coastal_outlet)
    runner.run_task("download", download_reference_data)
    runner.run_task("build_graph", build_graph)

    outlets: list[str] = runner.ti.xcom_pull(task_id="build_graph", key="outlets")
    outlet_subgraphs: dict[str, dict[str, Any]] = runner.ti.xcom_pull(
        task_id="build_graph", key="outlet_subgraphs"
    )
    outlet = outlets[0]
    partition_data = outlet_subgraphs[outlet]
    filtered_divides = partition_data["divides"]
    valid_divide_ids: set[str] = set(filtered_divides["divide_id"].to_list())
    classifications = _trace_stack(
        start_id=outlet,
        div_ids=valid_divide_ids,
        cfg=trace_case_no_divide_coastal_outlet,
        partition_data=partition_data,
    )

    aggregate_data = _aggregate_geometries(
        classifications=classifications,
        partition_data=partition_data,
    )

    assert len(aggregate_data.aggregates) == 2, "Incorrect number of aggregates"
    assert len(aggregate_data.independents) == 0, "Incorrect number of independents"
    assert len(aggregate_data.connectors) == 0, "Incorrect number of connectors"
    assert len(aggregate_data.non_nextgen_flowpaths) == 7, "Incorrect number of non nextgen flowpaths"
    assert len(aggregate_data.non_nextgen_virtual_flowpaths) == 3, (
        "Incorrect number of non nextgen virtual flowpaths"
    )
    assert len(aggregate_data.virtual_flowpaths) == 11, "Incorrect number of virtual flowpaths"

    runner.run_task(task_id="map_flowpaths", python_callable=map_trace_and_aggregate, op_kwargs={})
    runner.run_task(task_id="reduce_flowpaths", python_callable=reduce_calculate_id_ranges, op_kwargs={})
    runner.run_task(task_id="map_build_base", python_callable=map_build_base_hydrofabric, op_kwargs={})
    runner.run_task(task_id="reduce_base", python_callable=reduce_combine_base_hydrofabric, op_kwargs={})
    final_flowpaths = runner.ti.xcom_pull(task_id="reduce_base", key="flowpaths")
    final_nexus = runner.ti.xcom_pull(task_id="reduce_base", key="nexus")
    final_virtual_flowpaths = runner.ti.xcom_pull(task_id="reduce_base", key="virtual_flowpaths")
    final_virtual_nexus = runner.ti.xcom_pull(task_id="reduce_base", key="virtual_nexus")

    fp_pl = pl.from_pandas(final_flowpaths.to_wkb())
    upstream_dict = _build_upstream_dict_from_nexus(fp_pl)
    graph, _ = _build_rustworkx_object(upstream_dict)
    assert rx.is_directed_acyclic_graph(graph), "Graph must be acyclic"
    assert rx.is_weakly_connected(graph), "All flowpaths should connect to single outlet"
    assert not rx.is_strongly_connected(graph), "DAG cannot be strongly connected"
    graph_outlets: list[int] = [node for node in graph.node_indices() if graph.out_degree(node) == 0]
    assert len(graph_outlets) == 1, f"Should have exactly 1 outlet, found {len(graph_outlets)}"
    strong_components = rx.strongly_connected_components(graph)
    assert len(strong_components) == graph.num_nodes(), "Each node should be its own SCC in a DAG"

    virtual_fp_pl = pl.from_pandas(final_virtual_flowpaths.to_wkb())
    virtual_upstream_dict = _build_upstream_dict_from_nexus(
        virtual_fp_pl, edge_id="virtual_fp_id", node_id="virtual_nex_id"
    )
    virtual_graph, _ = _build_rustworkx_object(virtual_upstream_dict)
    assert rx.is_directed_acyclic_graph(virtual_graph), "Virtual graph must be acyclic"
    assert rx.is_weakly_connected(virtual_graph), "All virtual flowpaths should connect to single outlet"
    assert not rx.is_strongly_connected(virtual_graph), "Virtual DAG cannot be strongly connected"
    virtual_outlets = [node for node in virtual_graph.node_indices() if virtual_graph.out_degree(node) == 0]
    assert len(virtual_outlets) == 1, f"Should have exactly 1 virtual outlet, found {len(virtual_outlets)}"
    virtual_strong_components = rx.strongly_connected_components(virtual_graph)
    assert len(virtual_strong_components) == virtual_graph.num_nodes(), (
        "Each virtual node should be its own SCC"
    )

    df = pd.DataFrame(
        {
            "nex_id": pd.Series([2, 3], dtype="int64"),
            "dn_fp_id": pd.Series([pd.NA, 2], dtype="Int64"),
        }
    )
    pd.testing.assert_frame_equal(final_nexus.drop(columns=["geometry"]), df)

    df = pd.DataFrame(
        {
            "virtual_nex_id": pd.Series([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype="Int64"),
            "dn_virtual_fp_id": pd.Series([pd.NA, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype="Int64"),
        }
    )
    pd.testing.assert_frame_equal(final_virtual_nexus.drop(columns=["geometry"]), df)


def test_connector_no_divide_upstream(trace_case_bad_connector_no_divide_config: HFConfig) -> None:
    """Testing the tracing output for when there is a no-divide connector at the upstream-most point of a divide"""
    runner = LocalRunner(trace_case_bad_connector_no_divide_config)
    runner.run_task("download", download_reference_data)
    runner.run_task("build_graph", build_graph)

    outlets: list[str] = runner.ti.xcom_pull(task_id="build_graph", key="outlets")
    outlet_subgraphs: dict[str, dict[str, Any]] = runner.ti.xcom_pull(
        task_id="build_graph", key="outlet_subgraphs"
    )
    outlet = outlets[0]
    partition_data = outlet_subgraphs[outlet]
    filtered_divides = partition_data["divides"]
    valid_divide_ids: set[str] = set(filtered_divides["divide_id"].to_list())
    classifications = _trace_stack(
        start_id=outlet,
        div_ids=valid_divide_ids,
        cfg=trace_case_bad_connector_no_divide_config,
        partition_data=partition_data,
    )

    aggregate_data = _aggregate_geometries(
        classifications=classifications,
        partition_data=partition_data,
    )

    assert len(aggregate_data.aggregates) == 2, "Incorrect number of aggregates"
    assert len(aggregate_data.independents) == 0, "Incorrect number of independents"
    assert len(aggregate_data.connectors) == 1, "Incorrect number of connectors"
    assert len(aggregate_data.non_nextgen_flowpaths) == 7, "Incorrect number of non nextgen flowpaths"
    assert len(aggregate_data.non_nextgen_virtual_flowpaths) == 3, (
        "Incorrect number of non nextgen virtual flowpaths"
    )
    assert len(aggregate_data.virtual_flowpaths) == 6, "Incorrect number of virtual flowpaths"

    runner.run_task(task_id="map_flowpaths", python_callable=map_trace_and_aggregate, op_kwargs={})
    runner.run_task(task_id="reduce_flowpaths", python_callable=reduce_calculate_id_ranges, op_kwargs={})
    runner.run_task(task_id="map_build_base", python_callable=map_build_base_hydrofabric, op_kwargs={})
    runner.run_task(task_id="reduce_base", python_callable=reduce_combine_base_hydrofabric, op_kwargs={})
    final_flowpaths = runner.ti.xcom_pull(task_id="reduce_base", key="flowpaths")
    final_nexus = runner.ti.xcom_pull(task_id="reduce_base", key="nexus")
    final_virtual_flowpaths = runner.ti.xcom_pull(task_id="reduce_base", key="virtual_flowpaths")
    final_virtual_nexus = runner.ti.xcom_pull(task_id="reduce_base", key="virtual_nexus")

    fp_pl = pl.from_pandas(final_flowpaths.to_wkb())
    upstream_dict = _build_upstream_dict_from_nexus(fp_pl)
    graph, _ = _build_rustworkx_object(upstream_dict)
    assert rx.is_directed_acyclic_graph(graph), "Graph must be acyclic"
    assert rx.is_weakly_connected(graph), "All flowpaths should connect to single outlet"
    assert not rx.is_strongly_connected(graph), "DAG cannot be strongly connected"
    graph_outlets: list[int] = [node for node in graph.node_indices() if graph.out_degree(node) == 0]
    assert len(graph_outlets) == 1, f"Should have exactly 1 outlet, found {len(graph_outlets)}"
    strong_components = rx.strongly_connected_components(graph)
    assert len(strong_components) == graph.num_nodes(), "Each node should be its own SCC in a DAG"

    virtual_fp_pl = pl.from_pandas(final_virtual_flowpaths.to_wkb())
    virtual_upstream_dict = _build_upstream_dict_from_nexus(
        virtual_fp_pl, edge_id="virtual_fp_id", node_id="virtual_nex_id"
    )
    virtual_graph, _ = _build_rustworkx_object(virtual_upstream_dict)
    assert rx.is_directed_acyclic_graph(virtual_graph), "Virtual graph must be acyclic"
    assert rx.is_weakly_connected(virtual_graph), "All virtual flowpaths should connect to single outlet"
    assert not rx.is_strongly_connected(virtual_graph), "Virtual DAG cannot be strongly connected"
    virtual_outlets = [node for node in virtual_graph.node_indices() if virtual_graph.out_degree(node) == 0]
    assert len(virtual_outlets) == 1, f"Should have exactly 1 virtual outlet, found {len(virtual_outlets)}"
    virtual_strong_components = rx.strongly_connected_components(virtual_graph)
    assert len(virtual_strong_components) == virtual_graph.num_nodes(), (
        "Each virtual node should be its own SCC"
    )

    df = pd.DataFrame(
        {
            "nex_id": pd.Series([2, 3], dtype="int64"),
            "dn_fp_id": pd.Series([pd.NA, 2], dtype="Int64"),
        }
    )
    pd.testing.assert_frame_equal(final_nexus.drop(columns=["geometry"]), df)

    df = pd.DataFrame(
        {
            "virtual_nex_id": pd.Series([2, 3, 4, 5, 6, 7, 8], dtype="Int64"),
            "dn_virtual_fp_id": pd.Series([pd.NA, 2, 4, 5, 6, 3, 3], dtype="Int64"),
        }
    )
    pd.testing.assert_frame_equal(final_virtual_nexus.drop(columns=["geometry"]), df)


def test_hudson_river_large_scale(trace_case_hudson_river_large_scale: HFConfig) -> None:
    """Testing the tracing output for when there is a no-divide connector at the upstream-most point of a divide"""
    runner = LocalRunner(trace_case_hudson_river_large_scale)
    runner.run_task("download", download_reference_data)
    runner.run_task("build_graph", build_graph)

    outlets: list[str] = runner.ti.xcom_pull(task_id="build_graph", key="outlets")
    outlet_subgraphs: dict[str, dict[str, Any]] = runner.ti.xcom_pull(
        task_id="build_graph", key="outlet_subgraphs"
    )
    outlet = outlets[0]
    partition_data = outlet_subgraphs[outlet]
    filtered_divides = partition_data["divides"]
    valid_divide_ids: set[str] = set(filtered_divides["divide_id"].to_list())
    classifications = _trace_stack(
        start_id=outlet,
        div_ids=valid_divide_ids,
        cfg=trace_case_hudson_river_large_scale,
        partition_data=partition_data,
    )

    aggregate_data = _aggregate_geometries(
        classifications=classifications,
        partition_data=partition_data,
    )

    assert len(aggregate_data.aggregates) == 2319, "Incorrect number of aggregates"
    assert len(aggregate_data.independents) == 1277, "Incorrect number of independents"
    assert len(aggregate_data.connectors) == 1001, "Incorrect number of connectors"
    assert len(aggregate_data.non_nextgen_flowpaths) == 3431, "Incorrect number of non nextgen flowpaths"
    assert len(aggregate_data.non_nextgen_virtual_flowpaths) == 1907, (
        "Incorrect number of non nextgen virtual flowpaths"
    )
    assert len(aggregate_data.virtual_flowpaths) == 9355, "Incorrect number of virtual flowpaths"

    runner.run_task(task_id="map_flowpaths", python_callable=map_trace_and_aggregate, op_kwargs={})
    runner.run_task(task_id="reduce_flowpaths", python_callable=reduce_calculate_id_ranges, op_kwargs={})
    runner.run_task(task_id="map_build_base", python_callable=map_build_base_hydrofabric, op_kwargs={})
    runner.run_task(task_id="reduce_base", python_callable=reduce_combine_base_hydrofabric, op_kwargs={})
    final_flowpaths = runner.ti.xcom_pull(task_id="reduce_base", key="flowpaths")
    final_nexus = runner.ti.xcom_pull(task_id="reduce_base", key="nexus")
    final_virtual_flowpaths = runner.ti.xcom_pull(task_id="reduce_base", key="virtual_flowpaths")
    final_virtual_nexus = runner.ti.xcom_pull(task_id="reduce_base", key="virtual_nexus")

    fp_pl = pl.from_pandas(final_flowpaths.to_wkb())
    upstream_dict = _build_upstream_dict_from_nexus(fp_pl)
    graph, _ = _build_rustworkx_object(upstream_dict)
    assert rx.is_directed_acyclic_graph(graph), "Graph must be acyclic"
    assert rx.is_weakly_connected(graph), "All flowpaths should connect to single outlet"
    assert not rx.is_strongly_connected(graph), "DAG cannot be strongly connected"
    graph_outlets: list[int] = [node for node in graph.node_indices() if graph.out_degree(node) == 0]
    assert len(graph_outlets) == 1, f"Should have exactly 1 outlet, found {len(graph_outlets)}"
    strong_components = rx.strongly_connected_components(graph)
    assert len(strong_components) == graph.num_nodes(), "Each node should be its own SCC in a DAG"

    virtual_fp_pl = pl.from_pandas(final_virtual_flowpaths.to_wkb())
    virtual_upstream_dict = _build_upstream_dict_from_nexus(
        virtual_fp_pl, edge_id="virtual_fp_id", node_id="virtual_nex_id"
    )
    virtual_graph, _ = _build_rustworkx_object(virtual_upstream_dict)
    assert rx.is_directed_acyclic_graph(virtual_graph), "Virtual graph must be acyclic"
    assert rx.is_weakly_connected(virtual_graph), "All virtual flowpaths should connect to single outlet"
    assert not rx.is_strongly_connected(virtual_graph), "Virtual DAG cannot be strongly connected"
    virtual_outlets = [node for node in virtual_graph.node_indices() if virtual_graph.out_degree(node) == 0]
    assert len(virtual_outlets) == 1, f"Should have exactly 1 virtual outlet, found {len(virtual_outlets)}"
    virtual_strong_components = rx.strongly_connected_components(virtual_graph)
    assert len(virtual_strong_components) == virtual_graph.num_nodes(), (
        "Each virtual node should be its own SCC"
    )

    expected_df = pd.read_csv(
        here() / "tests/data/trace_cases/hudson_river_nexus.csv",
        dtype={"nex_id": "int64", "dn_fp_id": "Int64"},
    )
    pd.testing.assert_frame_equal(final_nexus.drop(columns=["geometry"]), expected_df)

    expected_df = pd.read_csv(
        here() / "tests/data/trace_cases/hudson_river_virtual_nexus.csv",
        dtype={"virtual_nex_id": "Int64", "dn_virtual_fp_id": "Int64"},
    )
    pd.testing.assert_frame_equal(final_virtual_nexus.drop(columns=["geometry"]), expected_df)


def test_sioux_falls(trace_case_sioux_falls: HFConfig) -> None:
    """Testing the tracing output for when there is a no-divide connector at the upstream-most point of a divide"""
    runner = LocalRunner(trace_case_sioux_falls)
    runner.run_task("download", download_reference_data)
    runner.run_task("build_graph", build_graph)

    outlets: list[str] = runner.ti.xcom_pull(task_id="build_graph", key="outlets")
    outlet_subgraphs: dict[str, dict[str, Any]] = runner.ti.xcom_pull(
        task_id="build_graph", key="outlet_subgraphs"
    )
    outlet = outlets[0]
    partition_data = outlet_subgraphs[outlet]
    filtered_divides = partition_data["divides"]
    valid_divide_ids: set[str] = set(filtered_divides["divide_id"].to_list())
    classifications = _trace_stack(
        start_id=outlet,
        div_ids=valid_divide_ids,
        cfg=trace_case_sioux_falls,
        partition_data=partition_data,
    )

    aggregate_data = _aggregate_geometries(
        classifications=classifications,
        partition_data=partition_data,
    )

    assert len(aggregate_data.aggregates) == 1771, "Incorrect number of aggregates"
    assert len(aggregate_data.independents) == 1448, "Incorrect number of independents"
    assert len(aggregate_data.connectors) == 1070, "Incorrect number of connectors"
    assert len(aggregate_data.non_nextgen_flowpaths) == 7072, "Incorrect number of non nextgen flowpaths"
    assert len(aggregate_data.non_nextgen_virtual_flowpaths) == 2254, (
        "Incorrect number of non nextgen virtual flowpaths"
    )
    assert len(aggregate_data.virtual_flowpaths) == 7420, "Incorrect number of virtual flowpaths"

    runner.run_task(task_id="map_flowpaths", python_callable=map_trace_and_aggregate, op_kwargs={})
    runner.run_task(task_id="reduce_flowpaths", python_callable=reduce_calculate_id_ranges, op_kwargs={})
    runner.run_task(task_id="map_build_base", python_callable=map_build_base_hydrofabric, op_kwargs={})
    runner.run_task(task_id="reduce_base", python_callable=reduce_combine_base_hydrofabric, op_kwargs={})
    final_flowpaths = runner.ti.xcom_pull(task_id="reduce_base", key="flowpaths")
    final_nexus = runner.ti.xcom_pull(task_id="reduce_base", key="nexus")
    final_virtual_flowpaths = runner.ti.xcom_pull(task_id="reduce_base", key="virtual_flowpaths")
    final_virtual_nexus = runner.ti.xcom_pull(task_id="reduce_base", key="virtual_nexus")

    fp_pl = pl.from_pandas(final_flowpaths.to_wkb())
    upstream_dict = _build_upstream_dict_from_nexus(fp_pl)
    graph, _ = _build_rustworkx_object(upstream_dict)
    assert rx.is_directed_acyclic_graph(graph), "Graph must be acyclic"
    assert rx.is_weakly_connected(graph), "All flowpaths should connect to single outlet"
    assert not rx.is_strongly_connected(graph), "DAG cannot be strongly connected"
    graph_outlets: list[int] = [node for node in graph.node_indices() if graph.out_degree(node) == 0]
    assert len(graph_outlets) == 1, f"Should have exactly 1 outlet, found {len(graph_outlets)}"
    strong_components = rx.strongly_connected_components(graph)
    assert len(strong_components) == graph.num_nodes(), "Each node should be its own SCC in a DAG"

    virtual_fp_pl = pl.from_pandas(final_virtual_flowpaths.to_wkb())
    virtual_upstream_dict = _build_upstream_dict_from_nexus(
        virtual_fp_pl, edge_id="virtual_fp_id", node_id="virtual_nex_id"
    )
    virtual_graph, _ = _build_rustworkx_object(virtual_upstream_dict)
    assert rx.is_directed_acyclic_graph(virtual_graph), "Virtual graph must be acyclic"
    assert rx.is_weakly_connected(virtual_graph), "All virtual flowpaths should connect to single outlet"
    assert not rx.is_strongly_connected(virtual_graph), "Virtual DAG cannot be strongly connected"
    virtual_outlets = [node for node in virtual_graph.node_indices() if virtual_graph.out_degree(node) == 0]
    assert len(virtual_outlets) == 1, f"Should have exactly 1 virtual outlet, found {len(virtual_outlets)}"
    virtual_strong_components = rx.strongly_connected_components(virtual_graph)
    assert len(virtual_strong_components) == virtual_graph.num_nodes(), (
        "Each virtual node should be its own SCC"
    )

    expected_df = pd.read_csv(
        here() / "tests/data/trace_cases/10L_U_nexus.csv",
        dtype={"nex_id": "int64", "dn_fp_id": "Int64"},
    )
    pd.testing.assert_frame_equal(final_nexus.drop(columns=["geometry"]), expected_df)

    expected_df = pd.read_csv(
        here() / "tests/data/trace_cases/10L_U_virtual_nexus.csv",
        dtype={"virtual_nex_id": "Int64", "dn_virtual_fp_id": "Int64"},
    )
    pd.testing.assert_frame_equal(final_virtual_nexus.drop(columns=["geometry"]), expected_df)


def test_braided_river(trace_case_braided: HFConfig) -> None:
    """Testing the tracing output for when there is a no-divide connector at the upstream-most point of a divide"""
    runner = LocalRunner(trace_case_braided)
    runner.run_task("download", download_reference_data)
    runner.run_task("build_graph", build_graph)

    outlets: list[str] = runner.ti.xcom_pull(task_id="build_graph", key="outlets")
    outlet_subgraphs: dict[str, dict[str, Any]] = runner.ti.xcom_pull(
        task_id="build_graph", key="outlet_subgraphs"
    )
    correct_aggregates = [124, 6736]
    correct_independents = [84, 3857]
    correct_connectors = [65, 2930]
    correct_non_nextgen_flowpaths = [204, 11037]
    correct_non_nextgen_virtual_flowpaths = [174, 7705]
    correct_virtual_flowpaths = [498, 27632]
    for idx, outlet in enumerate(outlets):
        partition_data = outlet_subgraphs[outlet]
        filtered_divides = partition_data["divides"]
        valid_divide_ids: set[str] = set(filtered_divides["divide_id"].to_list())
        classifications = _trace_stack(
            start_id=outlet,
            div_ids=valid_divide_ids,
            cfg=trace_case_braided,
            partition_data=partition_data,
        )

        aggregate_data = _aggregate_geometries(
            classifications=classifications,
            partition_data=partition_data,
        )

        assert len(aggregate_data.aggregates) == correct_aggregates[idx], "Incorrect number of aggregates"
        assert len(aggregate_data.independents) == correct_independents[idx], (
            "Incorrect number of independents"
        )
        assert len(aggregate_data.connectors) == correct_connectors[idx], "Incorrect number of connectors"
        assert len(aggregate_data.non_nextgen_flowpaths) == correct_non_nextgen_flowpaths[idx], (
            "Incorrect number of non nextgen flowpaths"
        )
        assert (
            len(aggregate_data.non_nextgen_virtual_flowpaths) == correct_non_nextgen_virtual_flowpaths[idx]
        ), "Incorrect number of non nextgen virtual flowpaths"
        assert len(aggregate_data.virtual_flowpaths) == correct_virtual_flowpaths[idx], (
            "Incorrect number of virtual flowpaths"
        )

    runner.run_task(task_id="map_flowpaths", python_callable=map_trace_and_aggregate, op_kwargs={})
    runner.run_task(task_id="reduce_flowpaths", python_callable=reduce_calculate_id_ranges, op_kwargs={})
    runner.run_task(task_id="map_build_base", python_callable=map_build_base_hydrofabric, op_kwargs={})
    runner.run_task(task_id="reduce_base", python_callable=reduce_combine_base_hydrofabric, op_kwargs={})
    final_flowpaths = runner.ti.xcom_pull(task_id="reduce_base", key="flowpaths")
    final_nexus = runner.ti.xcom_pull(task_id="reduce_base", key="nexus")
    final_virtual_flowpaths = runner.ti.xcom_pull(task_id="reduce_base", key="virtual_flowpaths")
    final_virtual_nexus = runner.ti.xcom_pull(task_id="reduce_base", key="virtual_nexus")

    fp_pl = pl.from_pandas(final_flowpaths.to_wkb())
    upstream_dict = _build_upstream_dict_from_nexus(fp_pl)
    graph, _ = _build_rustworkx_object(upstream_dict)
    assert rx.is_directed_acyclic_graph(graph), "Graph must be acyclic"
    assert not rx.is_strongly_connected(graph), "DAG cannot be strongly connected"
    strong_components = rx.strongly_connected_components(graph)
    assert len(strong_components) == graph.num_nodes(), "Each node should be its own SCC in a DAG"

    virtual_fp_pl = pl.from_pandas(final_virtual_flowpaths.to_wkb())
    virtual_upstream_dict = _build_upstream_dict_from_nexus(
        virtual_fp_pl, edge_id="virtual_fp_id", node_id="virtual_nex_id"
    )
    virtual_graph, _ = _build_rustworkx_object(virtual_upstream_dict)
    assert rx.is_directed_acyclic_graph(virtual_graph), "Virtual graph must be acyclic"
    assert not rx.is_strongly_connected(virtual_graph), "Virtual DAG cannot be strongly connected"
    virtual_strong_components = rx.strongly_connected_components(virtual_graph)
    assert len(virtual_strong_components) == virtual_graph.num_nodes(), (
        "Each virtual node should be its own SCC"
    )

    expected_df = pd.read_csv(
        here() / "tests/data/trace_cases/10L_braided_nexus.csv",
        dtype={"nex_id": "int64", "dn_fp_id": "Int64"},
    )
    pd.testing.assert_frame_equal(final_nexus.drop(columns=["geometry"]), expected_df)

    expected_df = pd.read_csv(
        here() / "tests/data/trace_cases/10L_braided_virtual_nexus.csv",
        dtype={"virtual_nex_id": "Int64", "dn_virtual_fp_id": "Int64"},
    )
    pd.testing.assert_frame_equal(final_virtual_nexus.drop(columns=["geometry"]), expected_df)

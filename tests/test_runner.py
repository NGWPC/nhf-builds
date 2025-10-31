"""Tests for the local runner"""

from typing import Any

import numpy as np
import pandas as pd
import rustworkx as rx
from scipy import sparse

from hydrofabric_builds import (
    HFConfig,
    build_graph,
    download_reference_data,
    map_build_base_hydrofabric,
    map_trace_and_aggregate,
    reduce_calculate_id_ranges,
    reduce_combine_base_hydrofabric,
    write_base_hydrofabric,
)
from scripts.hf_runner import LocalRunner, TaskInstance


class TestTaskInstance:
    """Tests for TaskInstance XCom functionality."""

    def test_xcom_push_and_pull(self, task_instance: TaskInstance) -> None:
        """Test basic push and pull operations."""
        task_instance.xcom_push("task1.result", {"data": 42})
        result = task_instance.xcom_pull("task1", key="result")
        assert result == {"data": 42}

    def test_xcom_pull_default_key(self, task_instance: TaskInstance) -> None:
        """Test pulling with default 'return_value' key."""
        task_instance.xcom_push("task1.return_value", "output.gpkg")
        result = task_instance.xcom_pull("task1")
        assert result == "output.gpkg"

    def test_xcom_pull_nonexistent(self, task_instance: TaskInstance) -> None:
        """Test pulling non-existent key returns None."""
        result = task_instance.xcom_pull("nonexistent_task")
        assert result is None

    def test_xcom_push_overwrite(self, task_instance: TaskInstance) -> None:
        """Test that pushing to same key overwrites previous value."""
        task_instance.xcom_push("task1.data", "first")
        task_instance.xcom_push("task1.data", "second")
        result = task_instance.xcom_pull("task1", key="data")
        assert result == "second"

    def test_xcom_multiple_tasks(self, task_instance: TaskInstance) -> None:
        """Test XCom with multiple tasks."""
        task_instance.xcom_push("download.return_value", "/tmp/data.zip")
        task_instance.xcom_push("process.return_value", "/tmp/data.gpkg")
        task_instance.xcom_push("export.return_value", "/tmp/output.json")

        assert task_instance.xcom_pull("download") == "/tmp/data.zip"
        assert task_instance.xcom_pull("process") == "/tmp/data.gpkg"
        assert task_instance.xcom_pull("export") == "/tmp/output.json"

    def test_xcom_complex_types(self, task_instance: TaskInstance) -> None:
        """Test XCom with complex data types."""
        complex_data = {
            "paths": ["/path/1", "/path/2"],
            "config": {"divide_aggregation_threshold": 3.0, "enabled": True},
            "stats": {"count": 100, "area": 45.6},
        }

        task_instance.xcom_push("task1.return_value", complex_data)
        result = task_instance.xcom_pull("task1")
        assert result == complex_data
        assert result["stats"]["count"] == 100


class TestLocalRunner:
    """Tests for LocalRunner task execution."""

    def test_initialization(self, sample_config: HFConfig) -> None:
        """Test LocalRunner initialization."""
        with LocalRunner(sample_config) as runner:
            assert runner.config == sample_config
            assert runner.run_id is not None
            assert isinstance(runner.ti, TaskInstance)
            assert runner.results == {}

    def test_custom_run_id(self, sample_config: HFConfig) -> None:
        """Test LocalRunner with custom run_id."""
        custom_id = "test_run_12345"
        with LocalRunner(sample_config, run_id=custom_id) as runner:
            assert runner.run_id == custom_id

    def test_run_simple_task(self, sample_config: HFConfig) -> None:
        """Test running a simple task."""

        def simple_task(value: int, **context: dict[str, Any]) -> dict[str, Any]:
            return {"value": value * 2}

        with LocalRunner(sample_config) as runner:
            _ = runner.run_task(task_id="simple", python_callable=simple_task, op_kwargs={"value": 21})

            assert runner.get_result("simple")["status"] == "success"
            assert runner.get_result("simple")["result"]["value"] == 42

    def test_task_context_injection(self, sample_config: HFConfig) -> None:
        """Test that context is properly injected into tasks."""

        def task_with_context(**context: dict[str, Any]) -> dict[str, Any]:
            return {
                "has_ti": "ti" in context,
                "has_task_id": "task_id" in context,
                "has_run_id": "run_id" in context,
                "has_config": "config" in context,
                "has_ds": "ds" in context,
                "has_execution_date": "execution_date" in context,
            }

        with LocalRunner(sample_config) as runner:
            result = runner.run_task("check_context", task_with_context)

            assert all(result.values()), "Missing context keys"
            assert result["has_ti"]
            assert result["has_config"]

    def test_task_accesses_config(self, sample_config: HFConfig) -> None:
        """Test that tasks can access config from context."""

        def task_using_config(**context: dict[str, Any]) -> dict:
            config = context["config"]
            assert config.divide_aggregation_threshold == 3.0  # type: ignore
            return {}

        with LocalRunner(sample_config) as runner:
            _ = runner.run_task("use_config", task_using_config)

    def test_task_uses_xcom(self, sample_config: HFConfig) -> None:
        """Test tasks using XCom to pass data."""

        def task1(**context: dict[str, Any]) -> dict[str, Any]:
            return {"path": "/tmp/output.gpkg"}

        def task2(**context: dict[str, Any]) -> dict:
            ti = context["ti"]
            input_file = ti.xcom_pull("task1", key="path")  # type: ignore
            assert input_file == "/tmp/output.gpkg"
            return {}

        with LocalRunner(sample_config) as runner:
            runner.run_task("task1", task1)
            _ = runner.run_task("task2", task2)

    def test_multiple_tasks_sequential(self, sample_config: HFConfig) -> None:
        """Test running multiple tasks sequentially."""

        def task1(value: int, **context: dict[str, Any]) -> dict[str, Any]:
            return {"value": value + 10}

        def task2(**context: dict[str, Any]) -> dict[str, Any]:
            ti = context["ti"]
            prev = ti.xcom_pull("task1", key="value")  # type: ignore
            return {"value": prev * 2}

        def task3(**context: dict[str, Any]) -> dict[str, Any]:
            ti = context["ti"]
            prev = ti.xcom_pull("task2", key="value")  # type: ignore
            return {"value": prev * 5}

        with LocalRunner(sample_config) as runner:
            runner.run_task("task1", task1, {"value": 5})
            runner.run_task("task2", task2)
            _ = runner.run_task("task3", task3)
            assert len(runner.results) == 3
            assert all(r["status"] == "success" for r in runner.results.values())

    def test_task_return_value_stored_in_xcom(self, sample_config: HFConfig) -> None:
        """Test that task return values are automatically stored in XCom."""

        def task_with_return(**context: dict[str, Any]) -> dict[str, Any]:
            return {"output": "test.gpkg", "count": 100}

        with LocalRunner(sample_config) as runner:
            runner.run_task("test", task_with_return)
            assert runner.ti.xcom_pull("test", key="output") == "test.gpkg"
            assert runner.ti.xcom_pull("test", key="count") == 100


class TestIntegration:
    """Integration tests simulating real pipeline scenarios."""

    def test_full_pipeline(self, sample_config: HFConfig, expected_graph: dict[str, Any]) -> None:
        """Test full pipeline with real download and build_graph functions."""

        with LocalRunner(sample_config) as runner:
            runner.run_task("download", download_reference_data)
            runner.run_task("build_graph", build_graph)
            runner.run_task(task_id="map_flowpaths", python_callable=map_trace_and_aggregate, op_kwargs={})
            runner.run_task(
                task_id="reduce_flowpaths", python_callable=reduce_calculate_id_ranges, op_kwargs={}
            )
            runner.run_task(
                task_id="map_build_base", python_callable=map_build_base_hydrofabric, op_kwargs={}
            )
            runner.run_task(
                task_id="reduce_base", python_callable=reduce_combine_base_hydrofabric, op_kwargs={}
            )
            runner.run_task(task_id="write_base", python_callable=write_base_hydrofabric, op_kwargs={})

            assert all(r["status"] == "success" for r in runner.results.values())

            download_result = runner.get_result("download")
            assert download_result["status"] == "success"

            reference_flowpaths = runner.ti.xcom_pull("download", key="reference_flowpaths")
            reference_divides = runner.ti.xcom_pull("download", key="reference_divides")

            assert reference_flowpaths is not None
            assert reference_divides is not None
            assert len(reference_flowpaths) > 0
            assert len(reference_divides) > 0

            graph_result = runner.get_result("build_graph")
            assert graph_result["status"] == "success"

            outlets = runner.ti.xcom_pull("build_graph", key="outlets")
            assert "6720879" in outlets  # expected outlet

            final_flowpaths = runner.ti.xcom_pull(task_id="reduce_base", key="flowpaths")
            final_divides = runner.ti.xcom_pull(task_id="reduce_base", key="divides")
            final_nexus = runner.ti.xcom_pull(task_id="reduce_base", key="nexus")

            assert len(final_flowpaths) == 50
            assert len(final_divides) == 50
            assert len(final_nexus) == 38

            flowpath_ids = set(final_flowpaths["fp_id"])
            divide_ids = set(final_divides["div_id"])
            assert flowpath_ids == divide_ids, (
                f"Flowpath IDs and Divide IDs don't match.\n"
                f"Missing in divides: {flowpath_ids - divide_ids}\n"
                f"Missing in flowpaths: {divide_ids - flowpath_ids}"
            )
            # Test connectivity using nexus points
            self._verify_nexus_connectivity(final_flowpaths, final_nexus)

    def _verify_nexus_connectivity(self, flowpaths_gdf: pd.DataFrame, nexus_gdf: pd.DataFrame) -> None:
        """Verify that nexus connectivity forms a valid dendritic network.

        Parameters
        ----------
        flowpaths_gdf : pd.DataFrame (GeoDataFrame)
            Flowpaths with fp_id, up_nex_id, dn_nex_id columns

        Raises
        ------
        AssertionError
            If connectivity is invalid (cycles, non-dendritic, or broken links)
        """
        graph = rx.PyDiGraph(check_cycle=True)
        fp_id_to_node = {}

        # Add all flowpaths as nodes
        for fp_id in flowpaths_gdf["fp_id"]:
            fp_id_to_node[fp_id] = graph.add_node(fp_id)

        # Verify all flowpaths are represented as nodes
        assert len(graph) == len(flowpaths_gdf), (
            f"Graph has {len(graph)} nodes but there are {len(flowpaths_gdf)} flowpaths"
        )

        # Build edges based on shared nexus points
        # If flowpath A's dn_nex_id == flowpath B's up_nex_id, then A flows into B
        edge_count = 0
        for _, row in flowpaths_gdf.iterrows():
            fp_id = row["fp_id"]
            dn_nex_id = row["dn_nex_id"]

            # Skip if no downstream nexus
            if pd.isna(dn_nex_id):
                continue

            # Find downstream flowpath(s) that have this nexus as their up_nex_id
            downstream_fps = flowpaths_gdf[flowpaths_gdf["up_nex_id"] == dn_nex_id]

            for _, dn_row in downstream_fps.iterrows():
                dn_fp_id = dn_row["fp_id"]
                # Add edge from current flowpath to downstream flowpath
                graph.add_edge(fp_id_to_node[fp_id], fp_id_to_node[dn_fp_id], None)
                edge_count += 1

        all_nexus_ids = set()
        for val in flowpaths_gdf["dn_nex_id"]:
            if not pd.isna(val):
                all_nexus_ids.add(val)
        for val in flowpaths_gdf["up_nex_id"]:
            if not pd.isna(val):
                all_nexus_ids.add(val)

        # Number of edges should equal number of non-terminal flowpaths
        # (i.e., flowpaths with at least one downstream connection)
        assert graph.num_edges() == edge_count, (
            f"Graph has {graph.num_edges()} edges but created {edge_count} edges"
        )

        print(
            f"Graph statistics: {len(graph)} nodes, {graph.num_edges()} edges, "
            f"{len(all_nexus_ids)} unique nexus points"
        )

        self._verify_graph_structure(graph)

    def _verify_graph_structure(self, graph: rx.PyDiGraph) -> None:
        """Verify dendritic structure and topological ordering.

        Parameters
        ----------
        graph : rx.PyDiGraph
            Graph representing flowpath connectivity

        Raises
        ------
        AssertionError
            If the network has cycles, is not dendritic, or is not in lower triangular ordering
        """
        # Check for cycles
        try:
            ts_order = rx.topological_sort(graph)
        except rx.DAGHasCycle as e:
            raise AssertionError("Network contains cycles - not a valid dendritic network") from e

        id_order = [graph.get_node_data(gidx) for gidx in ts_order]
        idx_map = {id_val: idx for idx, id_val in enumerate(id_order)}

        col = []
        row = []

        for node in ts_order:
            if graph.out_degree(node) == 0:  # terminal node
                continue

            node_id = graph.get_node_data(node)
            successor_indices = graph.successor_indices(node)

            # Verify dendritic structure (each flowpath has at most one downstream)
            assert len(successor_indices) == 1, (
                f"Flowpath {node_id} has {len(successor_indices)} successors, "
                f"network is not dendritic (should have exactly 1 downstream)"
            )

            downstream_node_idx = successor_indices[0]
            downstream_id = graph.get_node_data(downstream_node_idx)

            col.append(idx_map[node_id])
            row.append(idx_map[downstream_id])

        matrix = sparse.coo_matrix(
            (np.ones(len(row), dtype=np.uint8), (row, col)),
            shape=(len(ts_order), len(ts_order)),
            dtype=np.uint8,
        )
        # Ensure matrix is lower triangular (proper topological ordering)
        assert np.all(matrix.row >= matrix.col), (
            "Adjacency matrix is not lower triangular - flowpaths are not in proper topological order"
        )

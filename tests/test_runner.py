"""Tests for the local runner"""

from typing import Any

from hydrofabric_builds import HFConfig
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
            "config": {"dx": 3000, "enabled": True},
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
        runner = LocalRunner(sample_config)
        assert runner.config == sample_config
        assert runner.run_id is not None
        assert isinstance(runner.ti, TaskInstance)
        assert runner.results == {}

    def test_custom_run_id(self, sample_config: HFConfig) -> None:
        """Test LocalRunner with custom run_id."""
        custom_id = "test_run_12345"
        runner = LocalRunner(sample_config, run_id=custom_id)

        assert runner.run_id == custom_id

    def test_run_simple_task(self, runner: LocalRunner) -> None:
        """Test running a simple task."""

        def simple_task(value: int, **context: dict[str, Any]) -> dict[str, Any]:
            return {"value": value * 2}

        _ = runner.run_task(task_id="simple", python_callable=simple_task, op_kwargs={"value": 21})

        assert runner.get_result("simple")["status"] == "success"
        assert runner.get_result("simple")["result"]["value"] == 42

    def test_task_context_injection(self, runner: LocalRunner) -> None:
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

        result = runner.run_task("check_context", task_with_context)

        assert all(result.values()), "Missing context keys"
        assert result["has_ti"]
        assert result["has_config"]

    def test_task_accesses_config(self, runner: LocalRunner) -> None:
        """Test that tasks can access config from context."""

        def task_using_config(**context: dict[str, Any]) -> dict:
            config = context["config"]
            assert config.dx == 3000.0  # type: ignore
            return {}

        _ = runner.run_task("use_config", task_using_config)

    def test_task_uses_xcom(self, runner: LocalRunner) -> None:
        """Test tasks using XCom to pass data."""

        def task1(**context: dict[str, Any]) -> dict[str, Any]:
            return {"path": "/tmp/output.gpkg"}

        def task2(**context: dict[str, Any]) -> dict:
            ti = context["ti"]
            input_file = ti.xcom_pull("task1", key="path")  # type: ignore
            assert input_file == "/tmp/output.gpkg"
            return {}

        runner.run_task("task1", task1)
        _ = runner.run_task("task2", task2)

    def test_multiple_tasks_sequential(self, runner: LocalRunner) -> None:
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

        runner.run_task("task1", task1, {"value": 5})
        runner.run_task("task2", task2)
        _ = runner.run_task("task3", task3)

        assert len(runner.results) == 3
        assert all(r["status"] == "success" for r in runner.results.values())

    def test_task_return_value_stored_in_xcom(self, runner: LocalRunner) -> None:
        """Test that task return values are automatically stored in XCom."""

        def task_with_return(**context: dict[str, Any]) -> dict[str, Any]:
            return {"output": "test.gpkg", "count": 100}

        runner.run_task("test", task_with_return)
        assert runner.ti.xcom_pull("test", key="output") == "test.gpkg"
        assert runner.ti.xcom_pull("test", key="count") == 100


class TestIntegration:
    """Integration tests simulating real pipeline scenarios."""

    def test_full_pipeline(self, sample_config: HFConfig, expected_graph: dict[str, Any]) -> None:
        """Test full pipeline with real download and build_graph functions."""
        from hydrofabric_builds import build_graph, download_reference_data
        from scripts.hf_runner import LocalRunner

        runner = LocalRunner(sample_config)

        runner.run_task("download", download_reference_data)

        runner.run_task("build_graph", build_graph)

        # Verify all tasks succeeded
        assert all(r["status"] == "success" for r in runner.results.values())

        # Verify download results
        download_result = runner.get_result("download")
        assert download_result["status"] == "success"

        reference_flowpaths = runner.ti.xcom_pull("download", key="reference_flowpaths")
        reference_divides = runner.ti.xcom_pull("download", key="reference_divides")

        assert reference_flowpaths is not None
        assert reference_divides is not None
        assert len(reference_flowpaths) > 0
        assert len(reference_divides) > 0

        # Verify build_graph results
        graph_result = runner.get_result("build_graph")
        assert graph_result["status"] == "success"

        upstream_network = runner.ti.xcom_pull("build_graph", key="upstream_network")
        # Check that all expected keys are present
        assert set(upstream_network.keys()) == set(expected_graph.keys())

        # Check that each downstream has the correct upstream flowpaths
        for downstream_id, expected_upstreams in expected_graph.items():
            actual_upstreams = upstream_network[downstream_id]
            assert set(actual_upstreams) == set(expected_upstreams), (
                f"Mismatch for downstream {downstream_id}: expected {expected_upstreams}, got {actual_upstreams}"
            )

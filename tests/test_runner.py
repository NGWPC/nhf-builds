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

        def simple_task(value: int, **context: dict[str, Any]) -> int:
            return value * 2

        result = runner.run_task(task_id="simple", python_callable=simple_task, op_kwargs={"value": 21})

        assert result == 42
        assert runner.get_result("simple")["status"] == "success"
        assert runner.get_result("simple")["result"] == 42

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

        def task_using_config(**context: dict[str, Any]) -> float:
            config = context["config"]
            return config.dx  # type: ignore

        result = runner.run_task("use_config", task_using_config)

        assert result == 3000.0

    def test_task_uses_xcom(self, runner: LocalRunner) -> None:
        """Test tasks using XCom to pass data."""

        def task1(**context: dict[str, Any]) -> str:
            return "/tmp/output.gpkg"

        def task2(**context: dict[str, Any]) -> str:
            ti = context["ti"]
            input_file = ti.xcom_pull("task1")  # type: ignore
            return f"Processed: {input_file}"

        runner.run_task("task1", task1)
        result = runner.run_task("task2", task2)

        assert result == "Processed: /tmp/output.gpkg"

    def test_multiple_tasks_sequential(self, runner: LocalRunner) -> None:
        """Test running multiple tasks sequentially."""

        def task1(value: int, **context: dict[str, Any]) -> int:
            return value + 10

        def task2(**context: dict[str, Any]) -> int:
            ti = context["ti"]
            prev = ti.xcom_pull("task1")  # type: ignore
            return prev * 2

        def task3(**context: dict[str, Any]) -> int:
            ti = context["ti"]
            prev = ti.xcom_pull("task2")  # type: ignore
            return prev - 5

        runner.run_task("task1", task1, {"value": 5})
        runner.run_task("task2", task2)
        result = runner.run_task("task3", task3)

        assert result == 25
        assert len(runner.results) == 3
        assert all(r["status"] == "success" for r in runner.results.values())

    def test_task_return_value_stored_in_xcom(self, runner: LocalRunner) -> None:
        """Test that task return values are automatically stored in XCom."""

        def task_with_return(**context: dict[str, Any]) -> dict[str, Any]:
            return {"output": "test.gpkg", "count": 100}

        runner.run_task("test", task_with_return)
        xcom_result = runner.ti.xcom_pull("test")

        assert xcom_result == {"output": "test.gpkg", "count": 100}


class TestIntegration:
    """Integration tests simulating real pipeline scenarios."""

    def test_full_pipeline_mock(self, sample_config: HFConfig) -> None:
        """Test full pipeline with mock functions."""

        def download_data(source: str, **context: dict[str, Any]) -> str:
            return f"/tmp/{source}.zip"

        def process_data(**context: dict[str, Any]) -> str:
            ti = context["ti"]
            input_file = ti.xcom_pull("download")  # type: ignore
            return f"{input_file}.processed.gpkg"

        def export_data(**context: dict[str, Any]) -> dict[str, Any]:
            ti = context["ti"]
            processed = ti.xcom_pull("process")  # type: ignore
            return {"output": f"{processed}.exported.json", "status": "complete"}

        runner = LocalRunner(sample_config)
        runner.run_task("download", download_data, {"source": "hydrofabric"})
        runner.run_task("process", process_data)
        runner.run_task("export", export_data)

        assert all(r["status"] == "success" for r in runner.results.values())

        download_result = runner.get_result("download")["result"]
        assert "hydrofabric.zip" in download_result

        process_result = runner.get_result("process")["result"]
        assert "processed.gpkg" in process_result

        export_result = runner.get_result("export")["result"]
        assert export_result["status"] == "complete"

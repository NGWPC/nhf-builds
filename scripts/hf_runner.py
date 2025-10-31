"""Local runner for building the NGWPC hydrofabric"""

import argparse
import logging
import os
from collections.abc import Callable
from datetime import datetime
from typing import Any, Self

from dask.distributed import Client
from dotenv import load_dotenv
from pydantic import ValidationError
from pyprojroot import here

from hydrofabric_builds import HFConfig, TaskInstance
from hydrofabric_builds.pipeline.build_divide_attributes import build_divide_attributes
from hydrofabric_builds.pipeline.build_graph import build_graph
from hydrofabric_builds.pipeline.download import download_reference_data
from hydrofabric_builds.pipeline.processing import (
    map_build_base_hydrofabric,
    map_trace_and_aggregate,
    reduce_calculate_id_ranges,
    reduce_combine_base_hydrofabric,
)
from hydrofabric_builds.pipeline.write import write_base_hydrofabric

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
load_dotenv(here() / ".env")


class LocalRunner:
    """Execute pipeline tasks locally with Airflow-like interface.

    Parameters
    ----------
    config : HFConfig
        Pipeline configuration containing build settings and parameters.
    run_id : str or None, default=None
        Unique identifier for this pipeline run. If None, generated from
        current timestamp in format 'YYYYMMDD_HHMMSS'.
    setup_dask : bool, default=True
        Whether to set up a Dask distributed client for parallel processing.

    Attributes
    ----------
    config : HFConfig
        The pipeline configuration.
    run_id : str
        Unique identifier for this run.
    ti : TaskInstance
        TaskInstance for XCom operations.
    results : dict[str, dict[str, Any]]
        Execution results for each task, keyed by task_id.
    dask_client : Client or None
        Dask distributed client, if enabled.
    """

    def __init__(
        self,
        config: HFConfig,
        run_id: str | None = None,
        setup_dask: bool = True,  # ADD THIS PARAMETER
    ) -> None:
        """Initialize the LocalRunner.

        Parameters
        ----------
        config : HFConfig
            Pipeline configuration.
        run_id : str or None, default=None
            Optional run identifier. Auto-generated if not provided.
        setup_dask : bool, default=True
            Whether to set up Dask distributed client.
        """
        self.config: HFConfig = config
        self.run_id: str = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.ti: TaskInstance = TaskInstance()
        self.results: dict[str, dict[str, Any]] = {}
        self.dask_client: Client | None = None

        if self.config.enable_dask_dashboard:
            self._setup_dask_client()

    def _setup_dask_client(self) -> None:
        """Set up Dask distributed client for parallel processing."""
        n_workers = self.config.num_agg_workers if self.config.num_agg_workers else os.cpu_count()

        self.dask_client = Client(
            n_workers=n_workers,
            threads_per_worker=1,
            processes=True,
            memory_limit="16GB",
        )
        logger.info(f"runner: Dask client initialized with {n_workers} workers")
        logger.info(f"runner: Dask dashboard available at: {self.dask_client.dashboard_link}")

    def cleanup(self) -> None:
        """Clean up resources, including Dask client if initialized."""
        if self.dask_client is not None:
            logger.info("runner: Closing Dask client...")
            self.dask_client.close()
            self.dask_client = None
            logger.info("runner: Dask client closed")

    def __enter__(self: Self) -> Self:
        """Context manager entry."""
        return self

    def __exit__(self: Self, *args: str, **kwargs: str) -> None:
        """Context manager exit - ensures cleanup."""
        self.cleanup()

    def run_task(
        self,
        task_id: str,
        python_callable: Callable[..., Any],
        op_kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """Execute a single task.

        Parameters
        ----------
        task_id : str
            Unique identifier for this task. Used in XCom keys and result tracking.
        python_callable : Callable[..., Any]
            The function to execute. Must accept **kwargs to receive context.
        op_kwargs : dict[str, Any] or None, default=None
            Additional keyword arguments to pass to the callable.

        Returns
        -------
        Any
            The return value from the callable.
        """
        logger.info(f"Running task: {task_id}")

        context: dict[str, Any] = {
            "ti": self.ti,
            "task_id": task_id,
            "run_id": self.run_id,
            "ds": datetime.now().strftime("%Y-%m-%d"),
            "execution_date": datetime.now(),
            "config": self.config,
            "dask_client": self.dask_client,  # This can now be None
        }

        kwargs = {**(op_kwargs or {}), **context}

        result = python_callable(**kwargs)

        for k, v in result.items():
            self.ti.xcom_push(f"{task_id}.{k}", v)
        self.results[task_id] = {"status": "success", "result": result}

        logger.info(f"✓ Task {task_id} completed")
        return result

    def get_result(self, task_id: str) -> dict[str, Any]:
        """Retrieve execution results for a specific task.

        Parameters
        ----------
        task_id : str
            The identifier of the task to get results for.

        Returns
        -------
        dict[str, Any] or None
            Dictionary containing 'status' and either 'result' (on success)
            or 'error' (on failure). Returns None if task_id not found.
        """
        result = self.results.get(task_id)
        if result is None:
            raise ValueError("Cannot find result from task")
        return result


def main() -> int:
    """Main entry point for the hydrofabric-build pipeline CLI.

    Returns
    -------
    int
        Exit code: 0 for success, 1 for failure.
    """
    parser = argparse.ArgumentParser(description="A local runner for hydrofabric data processing")
    parser.add_argument("--config", required=False, help="Config file")
    parser.add_argument("--no-dask", action="store_true", help="Disable Dask parallel processing")  # ADD THIS
    args = parser.parse_args()

    try:
        config = HFConfig.from_yaml(args.config)
    except ValidationError as e:
        print("Configuration validation failed:")
        for error in e.errors():
            print(f"  {error['loc']}: {error['msg']}")
        return 1
    except FileNotFoundError:
        logger.error(f"Config file not found: {args.config}")
        return 1
    except TypeError:
        logger.warning("Config file not specified. Using default config settings")
        config = HFConfig()

    with LocalRunner(config) as runner:
        runner.run_task(task_id="download", python_callable=download_reference_data, op_kwargs={})
        runner.run_task(task_id="build_graph", python_callable=build_graph, op_kwargs={})
        runner.run_task(task_id="map_flowpaths", python_callable=map_trace_and_aggregate, op_kwargs={})
        runner.run_task(task_id="reduce_flowpaths", python_callable=reduce_calculate_id_ranges, op_kwargs={})
        runner.run_task(task_id="map_build_base", python_callable=map_build_base_hydrofabric, op_kwargs={})
        runner.run_task(task_id="reduce_base", python_callable=reduce_combine_base_hydrofabric, op_kwargs={})
        runner.run_task(task_id="write_base", python_callable=write_base_hydrofabric, op_kwargs={})
        if config.run_divide_attributes_task:
            runner.run_task(
                task_id="divide_attributes", python_callable=build_divide_attributes, op_kwargs={}
            )

        print("\n" + "=" * 60)
        print("Pipeline completed")
        print("=" * 60)
        for task_id, info in runner.results.items():
            status = "✓" if info["status"] == "success" else "✗"
            print(f"  {status} {task_id}: {info['status']}")
        print("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())

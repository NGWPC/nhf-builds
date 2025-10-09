"""Contains all code for processing hydrofabric data"""

from typing import Any, cast

import geopandas as gpd

from hydrofabric_builds import TaskInstance
from hydrofabric_builds.config import HFConfig


def download_reference_data(**context: dict[str, Any]) -> None:
    """
    Processes hydrofabric data.

    Parameters
    ----------
    **context : dict
        Airflow-compatible context containing:
        - ti : TaskInstance for XCom operations
        - config : HFConfig with pipeline configuration
        - task_id : str identifier for this task
        - run_id : str identifier for this pipeline run
        - ds : str execution date
        - execution_date : datetime object
    """
    cfg = cast(HFConfig, context["config"])
    ti = cast(TaskInstance, context["ti"])
    task_id = cast(str, context["task_id"])
    reference_flowpaths = gpd.read_file(cfg.reference_fabric_path, layer="reference_flowpaths")
    reference_divides = gpd.read_file(cfg.reference_fabric_path, layer="reference_divides")
    ti.xcom_push(f"{task_id}.reference_flowpaths", reference_flowpaths)
    ti.xcom_push(f"{task_id}.reference_divides", reference_divides)

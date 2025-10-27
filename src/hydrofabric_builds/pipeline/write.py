"""Contains all code for writing hydrofabric data"""

import logging
from typing import Any, cast

from hydrofabric_builds._version import __version__
from hydrofabric_builds.config import HFConfig
from hydrofabric_builds.task_instance import TaskInstance

logger = logging.getLogger(__name__)


def write_base_hydrofabric(**context: dict[str, Any]) -> dict:
    """Writes the base hydrofabric layers to disk

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

    Returns
    -------
    dict
        The reference flowpath and divides references in memory
    """
    cfg = cast(HFConfig, context["config"])
    ti = cast(TaskInstance, context["ti"])
    final_flowpaths = ti.xcom_pull(task_id="reduce_base", key="flowpaths")
    final_divides = ti.xcom_pull(task_id="reduce_base", key="divides")
    final_nexus = ti.xcom_pull(task_id="reduce_base", key="nexus")
    final_divides.to_file(
        cfg.output_dir / f"base_hydrofabric_{__version__}.gpkg", layer="divides", driver="GPKG"
    )
    final_flowpaths.to_file(
        cfg.output_dir / f"base_hydrofabric_{__version__}.gpkg", layer="flowpaths", driver="GPKG"
    )
    final_nexus.to_file(cfg.output_dir / f"base_hydrofabric_{__version__}.gpkg", layer="nexus", driver="GPKG")

    return {}

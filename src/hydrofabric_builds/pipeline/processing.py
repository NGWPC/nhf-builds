"""Contains all code for processing hydrofabric data"""

import logging
from typing import Any, cast

from hydrofabric_builds.config import HFConfig
from hydrofabric_builds.hydrofabric.schemas import Classifications
from hydrofabric_builds.hydrofabric.trace import _trace_stack
from hydrofabric_builds.task_instance import TaskInstance

logger = logging.getLogger(__name__)


def aggregate_data(**context: dict[str, Any]) -> dict[str, Classifications]:
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

    Returns
    -------
    dict[str, Classifications]
        The classifications for each flowpath in the reference

    Raise
    -----
    ValueError
        There are no outlets to classify
    """
    ti = cast(TaskInstance, context["ti"])
    cfg = cast(HFConfig, context["config"])

    reference_flowpaths = ti.xcom_pull(task_id="download", key="reference_flowpaths")
    # reference_divides = ti.xcom_pull(task_id="download", key="reference_divides")

    upstream_network = ti.xcom_pull(task_id="build_graph", key="upstream_network")
    outlets = ti.xcom_pull(task_id="build_graph", key="outlets")

    classifications = None
    for outlet in outlets:
        classifications = _trace_stack(
            start_id=outlet, network_graph=upstream_network, fp=reference_flowpaths, cfg=cfg
        )

    if classifications is None:
        raise ValueError("No outlets found. Aborting run")

    return {"classifications": classifications}

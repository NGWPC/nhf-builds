"""Contains all code for building a graph based on flowpath ids"""

import logging
from typing import Any, cast

from hydrofabric_builds.hydrofabric.graph import _build_graph, _detect_cycles, _find_outlets_by_hydroseq
from hydrofabric_builds.task_instance import TaskInstance

logger = logging.getLogger(__name__)


def build_graph(**context: dict[str, Any]) -> dict[str, dict[str, Any] | list[str]]:
    """
    Builds a graph of all downstream to upstream connectivity

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
    dict[str, dict[str, Any] | list[str]]
        - All outlets from the hydrofabric to trace upstream for connectivity
        - The upstream dictionary containing upstream and downstream connections
    """
    ti = cast(TaskInstance, context["ti"])

    reference_flowpaths = ti.xcom_pull(task_id="download", key="reference_flowpaths")

    upstream_dict = _build_graph(reference_flowpaths)

    _detect_cycles(upstream_dict)
    logger.info("Build Graph Task: No cycles detected when creating upstream network connections")

    outlets = _find_outlets_by_hydroseq(reference_flowpaths)

    return {"outlets": outlets, "upstream_network": upstream_dict}

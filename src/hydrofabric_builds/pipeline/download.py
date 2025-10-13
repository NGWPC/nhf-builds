"""Contains all code for processing hydrofabric data"""

from typing import Any, cast

import geopandas as gpd

from hydrofabric_builds.config import HFConfig


def download_reference_data(**context: dict[str, Any]) -> dict[str, gpd.GeoDataFrame]:
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
    dict[str, gpd.GeoDataFrame]
        The reference flowpath and divides references in memory
    """
    cfg = cast(HFConfig, context["config"])

    open_options = {"IMMUTABLE": "YES"}
    reference_flowpaths = gpd.read_file(
        cfg.reference_fabric_path, layer="reference_flowpaths", **open_options
    )
    reference_divides = gpd.read_file(cfg.reference_fabric_path, layer="reference_divides", **open_options)
    return {"reference_flowpaths": reference_flowpaths, "reference_divides": reference_divides}

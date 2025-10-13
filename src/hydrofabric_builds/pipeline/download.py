"""Contains all code for processing hydrofabric data"""

import logging
from typing import Any, cast

import geopandas as gpd

from hydrofabric_builds.config import HFConfig

logger = logging.getLogger(__name__)


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

    reference_divides = gpd.read_parquet(cfg.reference_divides_path)
    logger.info(f"Download Task: Ingested Reference Divides from: {cfg.reference_divides_path}")
    reference_flowpaths = gpd.read_parquet(cfg.reference_flowlines_path)
    logger.info(f"Download Task: Ingested Reference Flowpaths from: {cfg.reference_flowlines_path}")
    return {"reference_flowpaths": reference_flowpaths, "reference_divides": reference_divides}

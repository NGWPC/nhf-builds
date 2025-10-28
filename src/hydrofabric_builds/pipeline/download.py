"""Contains all code for downloading hydrofabric data"""

import logging
from typing import Any, cast

import geopandas as gpd

from hydrofabric_builds.config import HFConfig
from hydrofabric_builds.hydrofabric.graph import _validate_and_fix_geometries

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
    reference_divides["divide_id"] = reference_divides["divide_id"].astype("int").astype("str")
    reference_divides = _validate_and_fix_geometries(reference_divides, geom_type="divides")
    logger.info(f"Download Task: Ingested Reference Divides from: {cfg.reference_divides_path}")

    reference_flowpaths = gpd.read_parquet(cfg.reference_flowpaths_path)
    reference_flowpaths["flowpath_id"] = reference_flowpaths["flowpath_id"].astype("int").astype("str")
    logger.info(f"Download Task: Ingested Reference Flowpaths from: {cfg.reference_flowpaths_path}")

    reference_flowpaths = _validate_and_fix_geometries(reference_flowpaths, geom_type="flowpaths")

    return {"reference_flowpaths": reference_flowpaths, "reference_divides": reference_divides}

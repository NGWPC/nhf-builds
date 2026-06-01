"""Contains all code for building active NWM lakes in task"""

import logging
from typing import Any, cast

from hydrofabric_builds.config import HFConfig
from hydrofabric_builds.hydrofabric.lakes import lakes_pipeline

logger = logging.getLogger(__name__)


def build_lakes(**context: dict[str, Any]) -> dict[str, Any]:
    """Builds lakes layer from all NWM lake sources

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

    lakes_pipeline(cfg)

    # TODO
    # if cfg.lakes.fp_lk_crosswalk:
    #     gdf_fp_lk_crosswalk = _crosswalk_fp_lk(cfg)
    #     gdf_fp_lk_crosswalk.to_file(
    #         cfg.output_file_path, layer="lakes_flowpaths", driver="GPKG", overwrite=True
    #     )

    return {"lakes": "done"}

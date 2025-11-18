"""Contains all code for building divide attributes in task"""

import logging
from pathlib import Path
from typing import Any, cast

from hydrofabric_builds.config import HYDROFABRIC_OUTPUT_FILE, HFConfig
from hydrofabric_builds.helpers.io import load_config
from hydrofabric_builds.hydrofabric.waterbodies import crosswalk_waterbodies, rfc_da_pipeline

logger = logging.getLogger(__name__)


def build_waterbodies(**context: dict[str, Any]) -> dict[str, Any]:
    """Builds waterbodies table

    Builds RFC-DA dataset if not found in output location in reservoir configs

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

    res_cfg = load_config(cfg.reservoirs_config_path)
    rfcda_file = Path(res_cfg["rfc_da"]["outputs"]["out_gpkg"])

    if not rfcda_file.exists():
        logger.info(f"RFC-DA file not found at {rfcda_file}. Running RFC-DA pipeline.")
        rfc_da_pipeline(cfg.reservoirs_config_path)

        assert rfcda_file.exists(), "RFC-DA pipeline was run, but output file not found"

        logger.info(f"RFC-DA file created: {rfcda_file}.")

    logger.info(f"Using RFC-DA file found at {rfcda_file}.")
    gdf = crosswalk_waterbodies(HYDROFABRIC_OUTPUT_FILE, rfcda_file)

    gdf.to_file(HYDROFABRIC_OUTPUT_FILE, layer="waterbodies", driver="GPKG", overwrite=True)
    logger.info("Saved waterbodies layer")

    return {"waterodies": "done"}

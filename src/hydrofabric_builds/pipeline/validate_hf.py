import logging
from pathlib import Path
from typing import Any, cast

import geopandas as gpd
import pandas as pd
from pandantic import Pandantic
from pydantic import ValidationError

from hydrofabric_builds.config import HFConfig
from hydrofabric_builds.schemas.validate_hydrofabric import (
    Divides,
    Flowpaths,
    Gages_AK,
    Gages_CONUS,
    Gages_HI,
    Gages_PRVI,
)

logger = logging.getLogger(__name__)


def validate_divides(gpkg_path_filename: Path) -> None:
    """Validate the divides layer using Pydantic and check for NaNs

    Parameters
    ----------
    gpkg_path_filename : Path
        full path and filename of the NHF geopackage

    Returns
    -------
    None
    """
    try:
        divides = gpd.read_file(gpkg_path_filename, layer="divides")
    except FileNotFoundError:
        logger.warning(f"Error: The file {gpkg_path_filename} was not found.")

    divides = pd.DataFrame(divides)

    logger.info("Validate divide attributes")
    logger.info(f"Total number of rows: {len(divides)}\n")

    rows_with_nan = divides[divides.isna().any(axis=1)]
    logger.info(f"Total number of rows with NaNs: {len(rows_with_nan)}\n")

    nan_counts = divides.isna().sum().to_dict()
    logger.info("Number of NaNs per attribute")
    for key, value in nan_counts.items():
        logger.info(f"{key}: {value}")

    logger.info("Number of NaNs by VPU")
    nan_by_vpu = divides.isna().groupby(divides["vpu_id"]).sum()
    with pd.option_context("display.max_columns", None):
        logger.info(nan_by_vpu)

    validator = Pandantic(schema=Divides)

    logger.info("Validate divide attributes format")
    try:
        validator.validate(dataframe=divides, errors="raise")
    except ValidationError as e:
        error_details = e.errors()
        for error in error_details:
            logger.info(f"{error['loc']}: {error['msg']}; value is {error['input']} ")


def validate_flowpaths(gpkg_path_filename: Path) -> None:
    """Validate the flowpath layer using Pydantic and check for NaNs

    Parameters
    ----------
    gpkg_path_filename : Path
        full path and filename of the NHF geopackage

    Returns
    -------
    None
    """
    try:
        flowpaths = gpd.read_file(gpkg_path_filename, layer="flowpaths")
    except FileNotFoundError:
        logger.warning(f"Error: The file {gpkg_path_filename} was not found.")
    flowpaths = pd.DataFrame(flowpaths)

    logger.info("Validate flowpaths attributes")
    rows_with_nan = flowpaths[flowpaths.isna().any(axis=1)]
    logger.info(f"Total number of rows with NaNs: {len(rows_with_nan)}")

    nan_counts = flowpaths.isna().sum().to_dict()
    logger.info("Number of NaNs per attribute")
    for key, value in nan_counts.items():
        logger.info(f"{key}: {value}")

    logger.info("Number of NaNs by VPU")
    nan_by_vpu = flowpaths.isna().groupby(flowpaths["vpu_id"]).sum()
    with pd.option_context("display.max_columns", None):
        logger.info(nan_by_vpu)

    validator = Pandantic(schema=Flowpaths)

    logger.info("Validate Flowpath Attributes format")
    try:
        validator.validate(dataframe=flowpaths, errors="raise")
    except ValidationError as e:
        error_details = e.errors()
        for error in error_details:
            logger.info(f"{error['loc']}: {error['msg']}; value is {error['input']} ")


def validate_gages(gpkg_path_filename: Path, crs: str) -> None:
    """Check NHF gages against the list of ~1600 gages

    Parameters
    ----------
    gpkg_path_filename : Path
        full path and filename of the NHF geopackage
    crs: str
         CRS EPSG string

    Returns
    -------
    None
    """
    try:
        gages_layer = gpd.read_file(gpkg_path_filename, layer="Gages")
    except FileNotFoundError:
        logger.warning(f"Error: The file {gpkg_path_filename} was not found.")
    gages_layer = pd.DataFrame(gages_layer)

    gages_nwm = gages_layer["site_no"].to_list()

    if crs == "EPSG:5070":
        gages_calibratable = Gages_CONUS.gages
    elif crs == "EPSG:3338":
        gages_calibratable = Gages_AK.gages
    elif crs == "EPSG:32604":
        gages_calibratable = Gages_HI.gages
    elif crs == "EPSG:6566":
        gages_calibratable = Gages_PRVI.gages

    diff = set(gages_calibratable) - set(gages_nwm)

    if not diff:
        logger.info("All calibratable gages are in NHF")
    else:
        logger.info("The following gages are missing from NHF: \n")
        for gage in diff:
            logger.info(gage)


def validate_hf(**context: dict[str, Any]) -> dict[str, Any]:
    """Validates the divides and flowpath layers

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
        validation status
    """
    cfg = cast(HFConfig, context["config"])
    file_name = cfg.output_file_path
    crs = cfg.crs
    validate_divides(file_name)
    validate_flowpaths(file_name)
    validate_gages(file_name, crs)
    return {"validation": "done"}

import json
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


def validate_divides(gpkg_path_filename: Path) -> list[dict[Any, Any]]:
    """Validate the divides layer using Pydantic and check for NaNs

    Parameters
    ----------
    gpkg_path_filename : Path
        full path and filename of the NHF geopackage

    Returns
    -------
    list
        list of dictionaries containing divide validation
    """
    try:
        divides = gpd.read_file(gpkg_path_filename, layer="divides")
    except FileNotFoundError:
        logger.warning(f"Error: The file {gpkg_path_filename} was not found.")

    divides = pd.DataFrame(divides)

    divides_out: list[dict[str, int | list[str]]] = []

    divides_out.append({"Total number of divide rows": len(divides)})

    rows_with_nan = divides[divides.isna().any(axis=1)]
    divides_out.append({"Total number of rows with NaNs": len(rows_with_nan)})

    nan_counts = divides.isna().sum().to_dict()
    divides_out.append({"Number of NaNs per attribute": nan_counts})

    validator = Pandantic(schema=Divides)

    logger.info("Validate divide attributes format")
    validation_errors = []
    try:
        validator.validate(dataframe=divides, errors="raise")
    except ValidationError as e:
        error_details = e.errors()
        for error in error_details:
            validation_errors.append(f"{error['loc']}: {error['msg']}; value is {error['input']}")
    divides_out.append({"Validate divide attributes format": validation_errors})
    return divides_out


def validate_flowpaths(gpkg_path_filename: Path) -> list[dict[Any, Any]]:
    """Validate the flowpath layer using Pydantic and check for NaNs

    Parameters
    ----------
    gpkg_path_filename : Path
        full path and filename of the NHF geopackage

    Returns
    -------
    list
        list of dictionaries containing divide validation
    """
    try:
        flowpaths = gpd.read_file(gpkg_path_filename, layer="flowpaths")
    except FileNotFoundError:
        logger.warning(f"Error: The file {gpkg_path_filename} was not found.")
    flowpaths = pd.DataFrame(flowpaths)

    flowpaths_out: list[dict[str, int | list[str]]] = []

    flowpaths_out.append({"Total number of flowpaths rows": len(flowpaths)})

    rows_with_nan = flowpaths[flowpaths.isna().any(axis=1)]
    flowpaths_out.append({"Total number of flowpaths rows with NaNs": len(rows_with_nan)})

    nan_counts = flowpaths.isna().sum().to_dict()
    flowpaths_out.append({"Number of NaNs per attribute": nan_counts})

    validator = Pandantic(schema=Flowpaths)

    validation_errors = []
    try:
        validator.validate(dataframe=flowpaths, errors="raise")
    except ValidationError as e:
        error_details = e.errors()
        for error in error_details:
            validation_errors.append(f"{error['loc']}:{error['msg']}; value is {error['input']}")

    flowpaths_out.append({"Validate flowpaths attributes format": validation_errors})
    return flowpaths_out


def validate_gages(gpkg_path_filename: Path, crs: str) -> list:
    """Check NHF gages against the list of ~1600 gages

    Parameters
    ----------
    gpkg_path_filename : Path
        full path and filename of the NHF geopackage
    crs: str
        CRS EPSG string

    Returns
    -------
    list
        a list of missing gages
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
    diffs_list = list(diff)
    return diffs_list


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
    path = cfg.output_dir
    gpkg_file_root = cfg.output_name.stem
    crs = cfg.crs

    divides = validate_divides(file_name)
    flowpaths = validate_flowpaths(file_name)
    gages = validate_gages(file_name, crs)
    output_items = {"Divides": divides, "Flowpaths": flowpaths, "Missing Gages": gages}

    with open(f"{path}/{gpkg_file_root}_validation.json", "w") as file:
        json.dump(output_items, file, indent=4)

    return {"validation": "done"}

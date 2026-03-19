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
    CRS,
    Divides,
    Domain,
    Flowpaths,
    Layer,
)

logger = logging.getLogger(__name__)


def validate_layer(gpkg_path_filename: Path, layer_name: Layer, crs: CRS) -> list[dict[Any, Any]]:
    """Validate the divides layer using Pydantic and check for NaNs

    Parameters
    ----------
    gpkg_path_filename : Path
        full path and filename of the NHF geopackage

    layer_name : Layer
        Name of geopackage layer for divides or flowpaths

    crs : CRS
        EPSG string for the CRS

    Returns
    -------
    list
        list of dictionaries containing divide validation
    """
    # Read layer and convert from geo data frame to a Pandas data frame
    try:
        layer = gpd.read_file(gpkg_path_filename, layer=layer_name.value)
    except FileNotFoundError:
        error_str = {"Error": f"The file {gpkg_path_filename} was not found."}
        logger.warning(error_str)
        return [error_str]
    except ValueError:
        error_str = {"Error": f"Unable to read {layer_name.value} layer from {gpkg_path_filename}"}
        logger.warning(error_str)
        return [error_str]
    layer = pd.DataFrame(layer)

    # Get CRS and set the domain name
    if crs == CRS.CONUS:
        domain = Domain.CONUS.value
    elif crs == CRS.AK:
        domain = Domain.AK.value
    elif crs == CRS.HI:
        domain = Domain.HI.value
    elif crs == CRS.PRVI:
        domain = Domain.PRVI.value

    # Create empty list to store output items
    layer_out: list[dict[str, int | list[str]]] = []

    # Get total number of rows in layer
    layer_out.append({f"Total number of {layer_name.value} rows": len(layer)})

    # Get total number of rows with NaNs
    rows_with_nan = layer[layer.isna().any(axis=1)]
    layer_out.append({f"Total number of {layer_name.value} rows with NaNs": len(rows_with_nan)})

    # Get NaN counts by attribute
    nan_counts = layer.isna().sum().to_dict()
    layer_out.append({"Number of NaNs per attribute": nan_counts})

    # Use Pandantic to validate the data frames with the Pydantic schema.
    if layer_name == Layer.DIVIDES:
        # Add the domain name to the dataframe to select the proper lat/lon ranges per domain
        layer["domain"] = domain
        pydantic_schema = Divides
        logger.info("Validate divide attributes format")
    elif layer_name == Layer.FLOWPATHS:
        pydantic_schema = Flowpaths
        logger.info("Validate flowpaths attributes format")

    # Empty list to collect validation errors
    validation_errors = []

    validator = Pandantic(schema=pydantic_schema)

    try:
        validator.validate(dataframe=layer, errors="raise")
    except ValidationError as e:
        error_details = e.errors()
        for error in error_details:
            validation_errors.append(f"{error['loc']}: {error['msg']}; value is {error['input']}")
    layer_out.append({f"Validate {layer_name.value} attributes format": validation_errors})

    return layer_out


def validate_gages(gpkg_path_filename: Path, crs: CRS, gages_list: Path) -> list:
    """Check NHF gages against the list of ~1600 gages

    Parameters
    ----------
    gpkg_path_filename : Path
        full path and filename of the NHF geopackage
    crs : CRS
        CRS EPSG string
    gages_list : Path
        path and filename for calibratable gages list in csv format

    Returns
    -------
    list
        a list of missing gages
    """
    try:
        gages_layer = gpd.read_file(gpkg_path_filename, layer="Gages")
    except FileNotFoundError:
        error_str = f"Error: The file {gpkg_path_filename} was not found."
        logger.warning(error_str)
        return [error_str]
    except ValueError:
        error_str = f"Error reading gages layer from {gpkg_path_filename}"
        logger.warning(error_str)
        return [error_str]

    gages_layer = pd.DataFrame(gages_layer)

    gages_nhf = gages_layer["site_no"].to_list()

    calibratable_gages = pd.read_csv(gages_list)

    if crs.value == CRS.CONUS.value:
        calibratable_gages_domain: list = calibratable_gages[
            calibratable_gages["domain"] == Domain.CONUS.value
        ]["gage_id"].to_list()
    elif crs.value == CRS.AK.value:
        calibratable_gages_domain = calibratable_gages[calibratable_gages["domain"] == Domain.AK.value][
            "gage_id"
        ].to_list()
    elif crs.value == CRS.HI.value:
        calibratable_gages_domain = calibratable_gages[calibratable_gages["domain"] == Domain.HI.value][
            "gage_id"
        ].to_list()
    elif crs.value == CRS.PRVI.value:
        calibratable_gages_domain = calibratable_gages[calibratable_gages["domain"] == Domain.PRVI.value][
            "gage_id"
        ].to_list()

    gages_out = []

    diff = set(calibratable_gages_domain) - set(gages_nhf)
    diffs_list = list(diff)
    gages_out.append({"Missing Gages": diffs_list})

    missing_fp = gages_layer[gages_layer["fp_id"].isna() & gages_layer["virtual_fp_id"].isna()][
        "site_no"
    ].to_list()
    gages_out.append({"Gages with no flowpath or vitual flowpath": missing_fp})

    return gages_out


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
    calibration_gages = cfg.validate_hf.calibration_gages_path

    # Check if CRS is valid
    try:
        crs_enum = CRS(crs)
    except:
        error_str = f"CRS {crs} is not valid"
        logger.warning(error_str)
        return {"validation": error_str}

    divides = validate_layer(file_name, Layer.DIVIDES, crs_enum)
    flowpaths = validate_layer(file_name, Layer.FLOWPATHS, crs_enum)
    gages = validate_gages(file_name, crs_enum, calibration_gages)
    output_items = {"Divides": divides, "Flowpaths": flowpaths, "Gages": gages}

    with open(f"{path}/{gpkg_file_root}_validation.json", "w") as file:
        json.dump(output_items, file, indent=4)

    return {"validation": "done"}

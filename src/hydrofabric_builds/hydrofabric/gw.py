import logging

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from rasterio import features
from rasterio.transform import from_origin

from hydrofabric_builds.schemas.hydrofabric import (
    DivideAttributesModelConfig,
)

logger = logging.getLogger(__name__)


def groundwater_attributes(model_cfg: DivideAttributesModelConfig) -> None:
    """A pipeline to calculate divide attributes for a single divides file

    Parameters
    ----------
    model_cfg : DivideAttributesModelConfig
        model configuration parameters
    """
    # set projection based on input CRS string.  The NWM domain projections are used for groundwater parameters not
    # the EDFS projections.
    crs = model_cfg.crs
    if crs == "EPSG:5070":
        prjstr = "+proj=lcc +lat_1=30 +lat_2=60 +lat_0=40.0000076293945 +lon_0=-97 +x_0=0 +y_0=0 +a=6370000 +b=6370000 +units=m +no_defs"
    elif crs == "EPSG:3338":
        prjstr = "+proj=stere +lat_0=90 +lat_ts=60 +lon_0=-135"
    elif crs == "EPSG:32604":
        prjstr = "+proj=lcc +units=m +a=6370000.0 +b=6370000.0 +lat_1=10.0 +lat_2=30.0 +lat_0=20.6 +lon_0=-157.42 +x_0=0 +y_0=0 +k_0=1.0 +nadgrids=@null +wktext +no_defs"
    elif crs == "EPSG:6566":
        prjstr = "+proj=lcc +units=m +a=6370000.0 +b=6370000.0 +lat_1=18.1 +lat_2=18.1 +lat_0=18.1 +lon_0=-65.91 +x_0=0 +y_0=0 +k_0=1.0 +nadgrids=@null +wktext  +no_defs"

    # Get paths from the divide attributes section of the config file
    gw_attributes = [cfg for cfg in model_cfg.attributes if "GWBUCKPARM" in cfg.file_name.name][0]
    gwbuckparm_filename = gw_attributes.file_name
    if gw_attributes.file_name2 and gw_attributes.file_name3:
        spatial_weights_filename = gw_attributes.file_name2
        geogrid_raster_filename = gw_attributes.file_name3
        tmp_geogrid_path = gw_attributes.file_name3.parent / "tmp_geogrid.tif"
        tmp_raster_path = gw_attributes.file_name3.parent / "tmp_gw_features.tif"
    else:
        error_str = {
            "Error": "Groundwater divide attribute requires a 'file_name2' and 'file_name2' to be specified with spatial weights and geogrid files. Groundwater will not be calculated."
        }
        logger.warning(error_str)
        return

    # Read divides from the hydrofabric that was just built
    try:
        divides = gpd.read_file(model_cfg.hf_path, layer="divides")
    except FileNotFoundError:
        error_str = {"Error": f"The file {model_cfg.hf_path} was not found."}
        logger.warning(error_str)
        return
    except ValueError:
        error_str = {"Error": f"Unable to read divides layer from {model_cfg.hf_path}"}
        logger.warning(error_str)
        return

    # Reproject divides to the NWM domain CRS.
    divides_reproj = divides.to_crs(prjstr)

    # Number the divides for matching to pixel in the spatial weights file.
    # This can be changed to use the div_id and a separate numbering isn't necessary.
    divides_reproj["cat_id"] = np.arange(len(divides_reproj)) + 1

    # Creating blank raster for catchments
    logger.info("Creating blank geogrid raster")
    ds = xr.open_dataset(geogrid_raster_filename, engine="netcdf4")
    transform = from_origin(ds["x"][0], ds["y"][0], 1000, 1000)

    with rasterio.open(
        tmp_geogrid_path,
        "w",
        driver="GTiff",
        height=len(ds["y"]),
        width=len(ds["x"]),
        count=1,
        dtype=rasterio.uint8,
        crs=ds.attrs["proj4"],
        transform=transform,
    ) as _dst:
        pass
    del ds, transform

    # Using the empty NWM domain raster, burn in the divide polygons and save in the temp dir.
    logger.info("Rasterizing catchments")
    with rasterio.open(tmp_geogrid_path) as src:
        out_shape = src.shape
        transform = src.transform
        crs = src.crs

        shapes = (
            (geom, value)
            for geom, value in zip(divides_reproj.geometry, divides_reproj["cat_id"], strict=False)
        )

        rasterized = features.rasterize(
            shapes,
            out_shape=out_shape,
            transform=transform,
            fill=np.nan,
            all_touched=True,
            dtype=rasterio.float32,
        )

        with rasterio.open(
            tmp_raster_path,
            "w",
            driver="GTiff",
            height=rasterized.shape[0],
            width=rasterized.shape[1],
            count=1,
            dtype=rasterized.dtype,
            crs=crs,
            transform=transform,
        ) as dst:
            dst.write(rasterized, 1)

    # Read the NWM spatial weights file
    try:
        wts = xr.open_dataset(spatial_weights_filename)
    except FileNotFoundError:
        error_str = {"Error": f"The file {spatial_weights_filename} was not found. Skipping groundwater"}
        logging.warning(error_str)
        return
    wts = wts[["IDmask", "weight", "i_index", "j_index"]].to_dataframe()

    # Read the NWM GWBUCKPARM file
    try:
        gwbuckparm = xr.open_dataset(gwbuckparm_filename)
    except FileNotFoundError:
        error_str = {"Error": f"The file {gwbuckparm_filename} was not found. Skipping groundwater."}
        logging.warning(error_str)
        return
    gwbuckparm = gwbuckparm[["ComID", "Expon", "Zmax", "Coeff"]].to_dataframe()

    # Open the raster containing the divides and create a dataframe
    # containing the row, column and divide ID.
    logger.info("computing groundwater parameters")
    with rasterio.open(tmp_raster_path) as src:
        band_data = src.read(1)
        height = src.height
        width = src.width

        cols, rows = np.meshgrid(np.arange(width), np.arange(height))
        rows_flat = rows.flatten()
        cols_flat = cols.flatten()
        values_flat = band_data.flatten()

        raster_matrix = pd.DataFrame({"row": rows_flat, "col": cols_flat, "value": values_flat})
        raster_matrix = raster_matrix.dropna()

    # Create i_index and j_index columns, reversing the j index so that it starts at 1,1
    # in the lower left corner.
    raster_matrix["i_index"] = raster_matrix["col"].copy()
    raster_matrix["j_index"] = height + 1 - raster_matrix["row"].copy()

    # Merge the raster dataframe to the weights on i_index and j_index
    raster_matrix.set_index(["i_index", "j_index"], drop=True, inplace=True)
    wts.set_index(["i_index", "j_index"], drop=True, inplace=True)
    wts_div = raster_matrix.merge(wts, on=["i_index", "j_index"], how="left")

    # For each ComID (IDmask) in the spatial weights, add up the weights for each
    # overlapping divide.  This creates rows for divides that contain the contribution
    # from each ComID.
    wts_div_agg = wts_div.groupby(["IDmask", "value"])[["weight"]].sum()
    wts_div_agg = wts_div_agg.reset_index()
    wts_div_agg = wts_div_agg.rename(columns={"IDmask": "ComID", "value": "cat_id", "weight": "sumwt"})
    # Merge with gwbuckparm data on the ComID to get GW attribute values by
    # divide with contribution from each overlapping ComID
    gwparm = wts_div_agg.merge(gwbuckparm, on="ComID", how="left")

    # Merge divide IDs -- This can be removed if div_id is used instead of
    # creating new numbers for each divide.
    columns_to_merge = divides_reproj[["cat_id", "div_id"]]
    gwparm = pd.merge(gwparm, columns_to_merge, on="cat_id", how="left")

    # For each divide, compute attributes by summing the weights from each
    # contributing ComID and dividing by the total contributing weight.

    gwparm = (
        gwparm.groupby("div_id")
        .apply(
            lambda x: pd.Series(
                {
                    "Coeff": (x["Coeff"] * x["sumwt"]).sum() / x["sumwt"].sum(),
                    "Expon": (x["Expon"] * x["sumwt"]).sum() / x["sumwt"].sum(),
                    "Zmax": (x["Zmax"] * x["sumwt"]).sum() / x["sumwt"].sum(),
                }
            )
        )
        .reset_index()
    )

    # Merge groundwater attributes to the divides layer in the hydrofabric.
    divides = divides.merge(gwparm, on=model_cfg.divide_id, how="left")
    divides.to_file(model_cfg.hf_path, layer="divides", driver="GPKG", overwrite=True)

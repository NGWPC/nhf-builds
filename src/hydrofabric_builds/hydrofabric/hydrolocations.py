import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd
from pyogrio.errors import DataLayerError

logger = logging.getLogger(__name__)


def hydrolocations_pipeline(hf_path: Path) -> None:
    """Creates hydrolocations from any subset of gages, lakes, and waterbodies

    Creates a new index for hy_id from present tables
    Overwrites present tables to include hy_id
    Creates new hydrolocations table with hy_id and dn_nex_id
    Hydrolocations table is empty if no layers were present

    Parameters
    ----------
    hf_path : Path
        Path to hydrofabric to use
    """
    # NOTE: This function is WET but works

    # This patterns handles null layers and assigns an incremental hy_id
    # cycle through opening layers and checking if present
    # assign a hy_id based on the index + 1
    # start the next hy_id where you left off (last value + 1)
    # end_hy starts at 1 until overwritten by finding a present layer
    # hold list of present layers for later use
    end_hy = 1
    present_layers = []

    # waterbodies
    try:
        gdf_wb = gpd.read_file(hf_path, layer="waterbodies")
        hy_ids = gdf_wb.index + end_hy
        gdf_wb.insert(2, "hy_id", hy_ids)
        end_hy = gdf_wb["hy_id"].iloc[-1] + 1
        present_layers.append(gdf_wb)
    except DataLayerError:
        gdf_wb = None

    # gages
    try:
        gdf_gages = gpd.read_file(hf_path, layer="gages")
        hy_ids = gdf_gages.index + end_hy
        gdf_gages.insert(2, "hy_id", hy_ids)
        end_hy = gdf_gages["hy_id"].iloc[-1] + 1
        present_layers.append(gdf_gages)
    except DataLayerError:
        gdf_gages = None

    # lakes
    try:
        gdf_lakes = gpd.read_file(hf_path, layer="lakes")
        hy_ids = gdf_lakes.index + end_hy
        gdf_lakes.insert(2, "hy_id", hy_ids)
        end_hy = gdf_lakes["hy_id"].iloc[-1] + 1
        present_layers.append(gdf_lakes)
    except DataLayerError:
        gdf_lakes = None

    # if no layers were present, write an empty hydrolocations table with correct columns and return
    if not present_layers:
        df_hl = gpd.GeoDataFrame(columns=["hy_id", "dn_nex_id"], data=[])
        df_hl.to_file(hf_path, layer="hydrolocations", driver="GPKG", overwrite=True)
        logger.info("Wrote empty hydrolocations table beacuse gages, lakes, and waterbodies were not present")
        return

    # concat what is available for a full hydrolocations list
    gdfs = [gdf_wb, gdf_gages, gdf_lakes]
    concat_list = [gdf[["hy_id"]] for gdf in gdfs if isinstance(gdf, gpd.GeoDataFrame)]
    gdf_hl = pd.concat(concat_list, ignore_index=True)

    # save with hy_id
    gdf_wb.to_file(hf_path, layer="waterbodies", driver="GPKG", overwrite=True) if isinstance(
        gdf_wb, gpd.GeoDataFrame
    ) else None
    gdf_gages.to_file(hf_path, layer="gages", driver="GPKG", overwrite=True) if isinstance(
        gdf_gages, gpd.GeoDataFrame
    ) else None
    gdf_lakes.to_file(hf_path, layer="lakes", driver="GPKG", overwrite=True) if isinstance(
        gdf_lakes, gpd.GeoDataFrame
    ) else None
    logger.info("Wrote layers with updated with hy_id")

    # join fp_ids to get downstream nex_id
    gdf_fp = gpd.read_file(hf_path, layer="flowpaths")

    if isinstance(gdf_wb, gpd.GeoDataFrame):
        gdf_wb = gdf_wb.merge(gdf_fp[["fp_id", "dn_nex_id"]], on="fp_id", how="left")

    if isinstance(gdf_gages, gpd.GeoDataFrame):
        gdf_gages = gdf_gages.merge(gdf_fp[["fp_id", "dn_nex_id"]], on="fp_id", how="left")

    if isinstance(gdf_lakes, gpd.GeoDataFrame):
        gdf_lakes = gdf_lakes.merge(gdf_fp[["fp_id", "dn_nex_id"]], on="fp_id", how="left")

    # recreate the concat list with the updated dataframes
    gdfs = [gdf_wb, gdf_gages, gdf_lakes]
    concat_list = [gdf[["hy_id", "dn_nex_id"]] for gdf in gdfs if isinstance(gdf, gpd.GeoDataFrame)]

    # concat the HL with nexus
    df_dnnex = pd.concat(concat_list, axis=0, ignore_index=True)
    gdf_hl = gpd.GeoDataFrame(gdf_hl.merge(df_dnnex, on="hy_id", how="left"))

    # save final hl
    gdf_hl.to_file(hf_path, layer="hydrolocations", driver="GPKG", overwrite=True)
    logger.info("Wrote hydrolocations table")

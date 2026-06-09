import logging

import geopandas as gpd
import pandas as pd
import xarray as xr
from pyogrio.errors import DataLayerError

from hydrofabric_builds.config import HFConfig
from hydrofabric_builds.lakes.da import _generate_additional_crosswalk, _merge, _read_adhoc, _read_res_index

logger = logging.getLogger(__name__)


def res_da_pipeline(cfg: HFConfig) -> pd.DataFrame:
    try:
        lakes = gpd.read_file(cfg.output_file_path, layer="lakes")
    except DataLayerError:
        logger.info("Lakes layer not available for Reservoir DA. Skipping Reservoir DA table.")
        return
    try:
        gages = gpd.read_file(cfg.output_file_path, layer="gages")
    except DataLayerError:
        logger.info("Gages layer not available for Reservoir DA. Skipping Reservoir DA table.")
        return

    df_list = []

    logger.info("Retrieving reservoirs from crosswalk")
    ds = xr.open_dataset(cfg.res_da.res_crosswalk.path)
    df_list.append(_read_res_index(ds=ds, **cfg.res_da.res_crosswalk.fields.model_dump()))
    del ds

    if cfg.res_da.adhoc.run:
        logger.info("Retrieving reservoirs from adhoc table")
        gdf = gpd.read_file(cfg.res_da.adhoc.path, layer=cfg.res_da.adhoc.layer)
        df_list.append(_read_adhoc(gdf=gdf))
        del gdf

    if cfg.res_da.generate_additional_crosswalk:
        logger.info("Generating reservoir:gage crosswalks from data")
        fp = gpd.read_file(cfg.output_file_path, layer="flowpaths")
        df_list.append(_generate_additional_crosswalk(fp, gages, lakes))
        del fp

    logger.info("Merging reservoir DA tables")
    return _merge(lakes, df_list)

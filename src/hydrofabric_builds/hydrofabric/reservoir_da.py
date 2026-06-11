import logging

import geopandas as gpd
import pandas as pd
import xarray as xr
from pyogrio.errors import DataLayerError, DataSourceError

from hydrofabric_builds.config import HFConfig
from hydrofabric_builds.lakes.da import (
    _add_great_lakes,
    _all_level_pool,
    _generate_additional_crosswalk,
    _merge,
    _read_adhoc,
    _read_res_index,
)
from hydrofabric_builds.schemas.hydrofabric import GREAT_LAKES_MAPPING

logger = logging.getLogger(__name__)


def res_da_pipeline(cfg: HFConfig) -> pd.DataFrame:
    """Runs the reservoir data assimilation pipeline

    Reads from reservoir index, adhoc lakes file, and creates additional gage:lake crosswalk.
    If reservoir index is not available, returns all level pool.
    If lakes layer is not available, returns empty table.
    If gages are not available when additional gage:lake crosswalk is requested, crosswalk will not be run.

    Parameters
    ----------
    cfg : HFConfig
        HF Config

    Returns
    -------
    pd.DataFrame
        Res DA dataframe
    """
    try:
        lakes = gpd.read_file(cfg.output_file_path, layer="lakes")
    except DataLayerError:
        logger.info("Lakes layer not available for Reservoir DA. Returning empty dataframe.")
        return pd.DataFrame(
            columns=[
                "nhf_lake_id",
                cfg.res_da.lake_id_field,
                cfg.res_da.gage_id_field,
                cfg.res_da.da_type_field,
            ]
        )

    if cfg.res_da.all_level_pool:
        logger.info("Setting all reservoir DA to level pool")
        return _all_level_pool(
            df_lakes=lakes,
            gage_id_field=cfg.res_da.gage_id_field,
            lake_id_field=cfg.res_da.lake_id_field,
            res_da_field=cfg.res_da.da_type_field,
        )

    df_list = []

    logger.info("Retrieving reservoirs from crosswalk")
    ds = xr.open_dataset(cfg.res_da.res_crosswalk.path)
    df_list.append(
        _read_res_index(
            ds=ds, output_gage_field=cfg.res_da.gage_id_field, **cfg.res_da.res_crosswalk.fields.model_dump()
        )
    )
    del ds

    if cfg.res_da.great_lakes:
        logger.info("Adding Great Lakes")
        df_list.append(_add_great_lakes(mapping=GREAT_LAKES_MAPPING.copy()))

    if cfg.res_da.adhoc.run:
        logger.info("Retrieving reservoirs from adhoc table")
        gdf = gpd.read_file(cfg.res_da.adhoc.path, layer=cfg.res_da.adhoc.layer)
        df_list.append(
            _read_adhoc(
                gdf=gdf,
                rfc_field=cfg.res_da.adhoc.rfc_field,
                gage_id_field=cfg.res_da.gage_id_field,
                lake_id_field=cfg.res_da.lake_id_field,
                res_da_field=cfg.res_da.da_type_field,
                null_value=cfg.res_da.adhoc.null_value,
            )
        )
        del gdf

    if cfg.res_da.generate_additional_crosswalk:
        logger.info("Generating reservoir:gage crosswalks from data")
        try:
            gages = gpd.read_file(cfg.output_file_path, layer="gages")
        except (DataLayerError, DataSourceError):
            logger.info("Gages layer not available for Reservoir DA. Skipping additional crosswalking.")
            gages = gpd.GeoDataFrame()
        if gages.any():
            fp = gpd.read_file(cfg.output_file_path, layer="flowpaths")
            df_list.append(_generate_additional_crosswalk(fp, gages, lakes))
            del fp

    logger.info("Merging reservoir DA tables")
    return _merge(
        lakes, df_list, res_da_field=cfg.res_da.da_type_field, lake_id_field=cfg.res_da.lake_id_field
    )

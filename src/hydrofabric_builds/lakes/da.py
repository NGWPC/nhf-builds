import logging

import geopandas as gpd
import pandas as pd
import xarray as xr

from hydrofabric_builds.schemas.hydrofabric import ResDAMapping

logger = logging.getLogger(__name__)

DA_MAPPING = ResDAMapping()


def _read_res_index(
    ds: xr.Dataset,
    res_da_field: str = "da_type",
    index_field: str = "feature_id",
    lake_id_field: str = "lake_id",
    usgs_gage_id_field: str = "usgs_gage_id",
    usgs_lake_id_field: str = "usgs_lake_id",
    usace_gage_id_field: str = "usace_gage_id",
    usace_lake_id_field: str = "usace_lake_id",
    rfc_gage_id_field: str = "rfc_gage_id",
    rfc_lake_id_field: str = "rfc_lake_id",
    output_gage_field: str = "site_no",
) -> pd.DataFrame:
    """Extract crosswalk values and add DA scheme"""
    for field in [
        lake_id_field,
        usgs_gage_id_field,
        usgs_lake_id_field,
        usace_gage_id_field,
        usace_lake_id_field,
        rfc_gage_id_field,
        rfc_lake_id_field,
    ]:
        assert field in ds.variables

    # rfc
    rfc_crosswalk = pd.DataFrame(
        data={
            rfc_gage_id_field: ds[rfc_gage_id_field].to_numpy(),
            rfc_lake_id_field: ds[rfc_lake_id_field].to_numpy(),
        }
    )
    rfc_crosswalk[rfc_gage_id_field] = (
        rfc_crosswalk[rfc_gage_id_field].apply(lambda x: x.decode("utf-8")).str.strip()
    )
    rfc_crosswalk[res_da_field] = DA_MAPPING.rfc_forecast
    rfc_crosswalk.rename(
        columns={rfc_gage_id_field: output_gage_field, rfc_lake_id_field: lake_id_field}, inplace=True
    )

    # usgs
    usgs_crosswalk = pd.DataFrame(
        data={
            usgs_gage_id_field: ds[usgs_gage_id_field].to_numpy(),
            usgs_lake_id_field: ds[usgs_lake_id_field].to_numpy(),
        }
    )
    usgs_crosswalk[usgs_gage_id_field] = (
        usgs_crosswalk[usgs_gage_id_field].apply(lambda x: x.decode("utf-8")).str.strip()
    )
    usgs_crosswalk[res_da_field] = DA_MAPPING.usgs_persistence
    usgs_crosswalk.rename(
        columns={usgs_gage_id_field: output_gage_field, usgs_lake_id_field: lake_id_field}, inplace=True
    )

    # usace
    usace_crosswalk = pd.DataFrame(
        data={
            usace_gage_id_field: ds[usace_gage_id_field].to_numpy(),
            usace_lake_id_field: ds[usace_lake_id_field].to_numpy(),
        }
    )
    usace_crosswalk[usace_gage_id_field] = (
        usace_crosswalk[usace_gage_id_field].apply(lambda x: x.decode("utf-8")).str.strip()
    )
    usace_crosswalk[res_da_field] = DA_MAPPING.usace_persistence
    usace_crosswalk.rename(
        columns={usace_gage_id_field: output_gage_field, usace_lake_id_field: lake_id_field}, inplace=True
    )

    df_out = pd.concat([rfc_crosswalk, usgs_crosswalk, usace_crosswalk], ignore_index=True)

    return df_out


def _read_adhoc(
    gdf: gpd.GeoDataFrame,
    rfc_field: str = "locationId",
    lake_id_field: str = "lake_id",
    res_da_field: str = "da_type",
    null_value: int = -99999,
) -> pd.DataFrame:
    df = gdf.loc[gdf[lake_id_field] != null_value, [lake_id_field, rfc_field]].copy()
    df[res_da_field] = DA_MAPPING.rfc_forecast
    return df


def _generate_additional_crosswalk(
    fp: gpd.GeoDataFrame, gages: gpd.GeoDataFrame, lakes: gpd.GeoDataFrame
) -> pd.DataFrame:
    """Stub to generate more crosswalk"""
    pass


def _merge(
    df_lakes: gpd.GeoDataFrame,
    df_list: list[pd.DataFrame],
    res_da_field: str = "da_type",
    gid_field: str = "nhf_lake_id",
    lake_id_field: str = "lake_id",
) -> pd.DataFrame:
    df_lakes = df_lakes[[gid_field, lake_id_field]].copy()

    for df in df_list:
        df[lake_id_field] = df[lake_id_field].astype(int).astype(str)

    df_all = pd.concat(df_list)
    df_lakes = df_lakes.merge(df_all, how="left", on=lake_id_field)

    df_lakes.reset_index(drop=True, inplace=True)
    df_lakes.loc[df_lakes[res_da_field].isnull(), res_da_field] = DA_MAPPING.level_pool
    return df_lakes

import logging

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from hydrofabric_builds.schemas.hydrofabric import ResDAMapping

logger = logging.getLogger(__name__)

DA_MAPPING = ResDAMapping()


def _read_res_index(
    ds: xr.Dataset,
    res_da_field: str = "da_type",
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
        assert field in ds.variables, "Fields missing from reservoir index crosswalk"

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
    gage_id_field: str = "site_no",
    lake_id_field: str = "lake_id",
    res_da_field: str = "da_type",
    null_value: int = -99999,
) -> pd.DataFrame:
    """Read adhoc lakes file and give RFC forecast type"""
    df = gdf.loc[gdf[lake_id_field] != null_value, [lake_id_field, rfc_field]].copy()
    df[res_da_field] = DA_MAPPING.rfc_forecast
    df.rename(columns={rfc_field: gage_id_field}, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def _add_great_lakes(
    mapping: dict,
    gage_id_field: str = "site_no",
    lake_id_field: str = "lake_id",
    res_da_field: str = "da_type",
) -> pd.DataFrame:
    lake_id = []
    site_no = []
    for k, v in mapping.items():
        lake_id.append(k)
        site_no.append(v["site_no"])

    return pd.DataFrame(
        data={
            lake_id_field: lake_id,
            gage_id_field: site_no,
            res_da_field: [DA_MAPPING.great_lakes] * len(mapping.keys()),
        }
    )


def _generate_additional_crosswalk(
    fp: gpd.GeoDataFrame, gages: gpd.GeoDataFrame, lakes: gpd.GeoDataFrame
) -> pd.DataFrame:
    """Stub to generate more crosswalk"""
    # TODO
    pass


def _merge(
    df_lakes: gpd.GeoDataFrame,
    df_list: list[pd.DataFrame],
    res_da_field: str = "da_type",
    gid_field: str = "nhf_lake_id",
    lake_id_field: str = "lake_id",
) -> pd.DataFrame:
    """Merge all dataframe sources and de-duplicate lake_id"""
    df_lakes = df_lakes[[gid_field, lake_id_field]].copy()

    for df in df_list:
        df[lake_id_field] = df[lake_id_field].astype(int).astype(str)

    df_all = pd.concat(df_list)

    # de-dupe 1: drop true duplicates of lake_id and same res type
    df_all = df_all.drop_duplicates(subset=[lake_id_field, res_da_field], keep=False)

    # de-dupe 2: choose the duplicate with greater res_da field (non-level pool) and prefer RFC over all
    dupe = df_all.loc[df_all.duplicated(subset=lake_id_field, keep=False)].copy().reset_index(drop=True)
    df_all = df_all.loc[~df_all.duplicated(subset=lake_id_field, keep=False)].copy().reset_index(drop=True)

    dupe["priority"] = 0
    dupe["priority"] = np.where(dupe[res_da_field] > 1, 1, dupe["priority"])  # anything non-LP
    dupe["priority"] = np.where(dupe[res_da_field] == 4, 2, dupe["priority"])  # RFC
    dupe["priority"] = np.where(dupe[res_da_field] == 6, 3, dupe["priority"])  # Great Lakes
    idx = dupe.groupby(lake_id_field)["priority"].idxmax()
    dupe = dupe.loc[idx].copy().drop(columns=["priority"])

    # if there are still duplicates, take the first as they have same res DA value
    dupe = dupe.drop_duplicates(subset=lake_id_field, keep="first")

    df_all = pd.concat([df_all, dupe], ignore_index=True)
    df_all = df_all.reset_index(drop=True)

    # merge back to lakes
    df_lakes = df_lakes.merge(df_all, how="left", on=lake_id_field)
    df_lakes.reset_index(drop=True, inplace=True)
    df_lakes.loc[df_lakes[res_da_field].isnull(), res_da_field] = DA_MAPPING.level_pool
    df_lakes[res_da_field] = df_lakes[res_da_field].astype(int)

    assert ~df_lakes.duplicated(subset=lake_id_field).any(), f"Duplicate {lake_id_field} detected"
    assert ~df_lakes.duplicated(subset=gid_field).any(), f"Duplicate {gid_field} detected"

    return df_lakes

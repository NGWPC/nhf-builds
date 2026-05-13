"""Contains all code for building active NWM lakes in task"""

import logging
from pathlib import Path

from hydrofabric_builds.config import HFConfig
from hydrofabric_builds.helpers.flowpath_association import (
    associate_flowpaths_nearest_point,
    associate_flowpaths_polygon_outlet,
    join_attributes,
)
import geopandas as gpd
from hydrofabric_builds.hydrofabric.utils import _crosswalk_nexus, _crosswalk_reference
import pandas as pd
import numpy as np
logger = logging.getLogger(__name__)


def _associate_lake_flowpaths(main_cfg: HFConfig, lake_type) -> gpd.GeoDataFrame:
    """Associate flowpaths and join attributes from source file if requested.

    """
    cfg = getattr(main_cfg.lakes, lake_type)

    # Preprocess lakes by associating with flowpaths if requested or if processed path does not exist
    if cfg.associate_flowpaths or not cfg.tmp_path.exists():
        # Use nearest point association method
        if cfg.flowpath_association_method == "nearest_point":
            logger.info("Associating flowpath with points")
            gdf = associate_flowpaths_nearest_point(
                points_path=cfg.input_path,
                points_layer=cfg.input_layer,
                flowpaths_path=Path(main_cfg.build.reference_flowpaths_path),
                search_radius_m=cfg.search_radius_m,
                point_id=cfg.id_field,
                flowpath_id=cfg.fp_id,
                flowpath_id_out_field=cfg.fp_out_id,
            )
        # use polygon flowpath outlet method
        elif cfg.flowpath_association_method == "polygon_outlet":
            logger.info("Associating flowpaths with polygons")
            gdf = associate_flowpaths_polygon_outlet(
                polygon_path=cfg.input_path,
                polygon_layer=cfg.lakes.input_layer,
                flowpaths_path=Path(main_cfg.build.reference_flowpaths_path),
                search_radius_m=cfg.search_radius_m,
                min_preferred_intersection_len_m=cfg.min_preferred_intersection_len_m,
                flowpath_id=cfg.id_field,
                flowpath_id_out_field=cfg.fp_out_id,
            )

        # invalid method
        else:
            raise ValueError("Config contained invalid Lakes flowpath association method")

        if cfg.attrib_src_path:
                gdf = join_attributes(
                    gdf,
                    attrib_dst_key=cfg.id_field,
                    attrib_src_path=cfg.attrib_src_path,
                    attrib_src_layer=cfg.attrib_src_layer,
                    attrib_src_key=cfg.attrib_src_key,
                    attrib_src_fields=cfg.attrib_fields.copy(),
                    rename=True,
                )

    # Save nwm_lakes layer to NHF
    gdf.to_file(cfg.tmp_path, layer="lakes", driver="GPKG", overwrite=True)

    return gdf

# maybe this can be one function just with separate cfg for IDs?

def _improve_placement(cfg, gdf):
    gdf['nid'] = None
    gdf.to_file(cfg.lakes.nwm_lakes_tmp_path, driver='GPKG', overwrite=True)

def _concat_lakes(gdf_nwm, gdf_adhoc, gdf_ref_res):




# TODO: Finish
def _join_nid(nid_path, res_df):
    """Join National Inventory Dams data to reservoirs"""
    nid_path = Path(nid_path)
    if nid_path.suffix.lower() == ".gpkg":
        nid_df = gpd.read_file(nid_path)
    elif nid_path.suffix.lower() in {".parquet", ".pq"}:
        nid_df = pd.read_parquet(nid_path)
    else:
        nid_df = pd.read_csv(nid_path)

    nid_df.columns = [col.lower() for col in nid_df.columns]

    # cast types like R
    for col in ("spillway_type", "dam_type"):
        if col in nid_df.columns:
            nid_df[col] = nid_df[col].astype("string")

    for col in ("structural_height", "dam_height", "nid_height", "surface_area", "hydraulic_height"):
        if col in nid_df.columns:
            nid_df[col] = pd.to_numeric(nid_df[col], errors="coerce")

    if "surface_area" not in nid_df.columns:
        nid_df["surface_area"] = np.nan

    # keep only needed columns (loosely matching R)
    keep_cols = [
        "nidid",
        "dam_name",
        "dam_type",
        "spillway_type",
        "spillway_width",
        "dam_length",
        "dam_height",
        "structural_height",
        "hydraulic_height",
        "nid_height",
        "surface_area",
        "wb_areasqkm",
        "nid_storage",
        "normal_storage",
        "max_storage",
        "hazard",
        "purposes",
    ]

    keep_cols = [c for c in keep_cols if c in nid_df.columns]
    nid_df = nid_df[keep_cols].copy()

    # restrict NID to NID IDs in da$nid
    if "nid" not in res_df.columns:
        raise ValueError("Expected 'nid' column in reference reservoirs (da).")
    nid_ids = res_df["nid"].dropna().unique()
    nid_df = nid_df[nid_df["nidid"].isin(nid_ids)].copy()

    res_df = res_df.rename(columns={"nid": "nidid"}).copy()

    res_df = res_df.merge(nid_df, on='nid', how='left')

    return res_df

def _filter_ref_res(cfg):
    res = gpd.read_file(cfg.lakes.ref_reservoirs_path)

    gdf = res[
        ((res["distance_to_fp_m"] < cfg.lakes.max_waterbody_nearest_dist_m) & (res["wb_areasqkm"] >= cfg.lakes.min_area_sqkm))
        | (res["dam_id"].isin(cfg.lakes.res_keep))
    ].copy()

    try:
        nwm_lakes = gpd.read_file(cfg.lakes.nwm_lakes_tmp_path)
        gdf = gdf.loc[~gdf['nid'].isin(nwm_lakes['nid'])].copy()

    except Exception as e:
        logger.warning("Could not read nwm_lakes file. Reference reservoirs will not be filtered to exclude nwm_lakes with same NID")

    return gdf



def _filter_columns(gdf, fields):
    # add nulls for any missing columns requested
    for f in fields:
        if f not in gdf.columns:
            gdf[f] = None

    out_columns = (
        ["nhf_lake_id", "ref_fp_id", "fp_id", "virtual_fp_id", "dn_nex_id", "dn_virtual_nex_id", "div_id"]
        + fields
        + ["geometry"]
    )

    # select final attribute list
    return gdf[out_columns]

def _create_ids(gdf):
    gdf["nhf_lake_id"] = range(1, gdf.shape[0] + 1)
    return gdf

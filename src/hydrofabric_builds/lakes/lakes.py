"""Contains all code for building active NWM lakes in task"""

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from hydrofabric_builds.config import HFConfig
from hydrofabric_builds.helpers.flowpath_association import (
    associate_flowpaths_nearest_point,
    associate_flowpaths_polygon_outlet,
    join_attributes,
)
from hydrofabric_builds.lakes.helpers import point_elevation, polygon_elevation

logger = logging.getLogger(__name__)


def _associate_lake_flowpaths(main_cfg: HFConfig, lake_type) -> gpd.GeoDataFrame:
    """Associate flowpaths and join attributes from source file if requested."""
    cfg = getattr(main_cfg.lakes, lake_type)

    try:
        gdf = gpd.read_file(cfg.path, layer=cfg.layer)
        del gdf
    except Exception as e:
        logger.info(
            f"Input file for lake type {lake_type} could not be read. Skipping flowpath association for {lake_type}. Exception: {e}"
        )
        # Return empty dataframe to cause rest of pipeline to work
        return gpd.GeoDataFrame(columns=["geometry", "lake_id"])

    # Preprocess lakes by associating with flowpaths if requested or if processed path does not exist
    if cfg.associate_flowpaths or not cfg.tmp_path.exists():
        # Use nearest point association method
        if cfg.flowpath_association_method == "nearest_point":
            logger.info(f"Associating {lake_type} flowpath with points")
            gdf = associate_flowpaths_nearest_point(
                points_path=cfg.path,
                points_layer=cfg.layer,
                flowpaths_path=Path(main_cfg.build.reference_flowpaths_path),
                search_radius_m=cfg.search_radius_m,
                point_id=cfg.id_field,
                flowpath_id="flowpath_id",
                flowpath_id_out_field="ref_fp_id",
            )
        # use polygon flowpath outlet method
        elif cfg.flowpath_association_method == "polygon_outlet":
            logger.info(f"Associating {lake_type} flowpaths with polygons")
            gdf = associate_flowpaths_polygon_outlet(
                polygon_path=cfg.path,
                polygon_layer=cfg.layer,
                flowpaths_path=Path(main_cfg.build.reference_flowpaths_path),
                search_radius_m=cfg.search_radius_m,
                min_preferred_intersection_len_m=cfg.min_preferred_intersection_len_m,
                flowpath_id="flowpath_id",
                flowpath_id_out_field="ref_fp_id",
            )

        # invalid method
        else:
            raise ValueError(f"Config contained invalid flowpath association method for {lake_type}")

        if cfg.attrib_src_path:
            gdf = join_attributes(
                gdf,
                attrib_dst_key=cfg.id_field,
                attrib_src_path=cfg.attrib_src_path,
                attrib_src_layer=cfg.attrib_src_layer,
                attrib_src_key=cfg.attrib_src_key,
                attrib_src_fields=cfg.fields.copy(),
                rename=True,
            )
        if cfg.output_id_field not in gdf.columns:
            gdf = gdf.rename(columns={cfg.id_field: cfg.output_id_field})

        # Save nwm_lakes layer to NHF
        gdf.to_file(cfg.tmp_path, layer="lakes", driver="GPKG", overwrite=True)

    else:
        # read the pre-processed file to return
        gdf = gpd.read_file(cfg.tmp_path)

    return gdf


def _merge_ref_wb(cfg: HFConfig, gdf_ref_wb):
    gdf_ref_res = gpd.read_file(cfg.lakes.ref_res.path)
    gdf_ref_wb = gdf_ref_wb.merge(
        gdf_ref_res[["dam_id", "ref_fab_fp", "ref_fab_wb", "nid", "wb_areasqkm"]],
        left_on="comid",
        right_on="ref_fab_wb",
    )
    del gdf_ref_res
    return gdf_ref_wb


def _concat_lakes(gdf_nwm, gdf_adhoc, gdf_ref_wb, gdf_ref_res):
    return pd.concat([gdf_nwm, gdf_adhoc, gdf_ref_wb, gdf_ref_res], ignore_index=True)


def _calculate_elevation(gdf, dem_path, calculate: bool = True):
    if calculate:
        # NOTE: it would be nicer if these both returned series
        gdf_all_lks = polygon_elevation(dem_path, gdf_all_lks)
        gdf_all_lks["dam_elev"] = point_elevation(dem_path, gdf_all_lks)
    else:
        gdf_all_lks["dam_elev"] = np.nan
        gdf_all_lks["ref_elev"] = np.nan

    return gdf_all_lks


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

    res_df = res_df.merge(nid_df, on="nidid", how="left")

    return res_df


def _filter_adhoc_lakes(cfg: HFConfig):
    gdf = gpd.read_file(cfg.lakes.adhoc.path)
    missing = gdf.loc[gdf[cfg.lakes.adhoc.id_field] == -99999, :].copy()
    ref_wb = gdf.loc[gdf[cfg.lakes.adhoc.ref_wb_field] is True, :].copy()
    return missing, ref_wb


def _filter_ref_res(cfg, gdf_nwm, gdf_ref_wb):
    res = gpd.read_file(cfg.lakes.ref_reservoirs_path)

    # filter to exclude dam_id that have already been joined to nwm lakes or included wb
    res = res.loc[
        ~res["dam_id"].isin(gdf_nwm["dam_id"]) & ~res["dam_id"].isin(gdf_ref_wb["dam_id"]),
        ["dam_id", "ref_fab_fp", "ref_fab_wb", "nid", "wb_areasqkm"],
    ].copy()

    # filter to criteria
    res = res[
        (
            (res["distance_to_fp_m"] < cfg.lakes.max_waterbody_nearest_dist_m)
            & (res["wb_areasqkm"] >= cfg.lakes.min_area_sqkm)
        )
        | (res["dam_id"].isin(cfg.lakes.res_keep))
    ].copy()

    return res


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


def _crosswalk_fp_lk():
    pass


def _fold_ref_res_to_nwm_lakes(config: HFConfig, nwm_lakes_pt) -> gpd.GeoDataFrame:
    """Takes in nwm_lakes GeoDataFrame with polygons and and returns a GeoDataFrame with points derived from reference reservoirs OR centroid if no reference available.

    Attempts to find the most downstream reference reservoir candidate by searching in proximity of the most downstream
    intersecting flowpath rather than searching in proximity of lake polygon.

    If reference point is available, also retains "dam_id", "dam_name" and "nid" columns from reference datapoint.
    """
    nwm_lakes = gpd.read_file(config.lakes.nwm.path, layer=config.lakes.nwm.layer).to_crs(config.crs)
    fp = gpd.read_parquet(config.build.reference_flowpaths_path).to_crs(config.crs)
    ref_res = gpd.read_file(config.lakes.ref_res.path).to_crs(config.crs)

    max_distance = config.lakes.nwm.max_search_distance_m

    nwm_lakes[["dam_name", "dam_id", "nid"]] = [pd.NA, pd.NA, pd.NA]
    nwm_lakes.rename(columns={config.lakes.nwm.id_field: config.lakes.nwm.output_id_field}, inplace=True)

    # for each lake
    for idx, _ in nwm_lakes.iterrows():
        # Limit FPs to those that intersect with *this* lake
        fps = fp[nwm_lakes["geometry"][idx].intersects(fp.geometry)]
        # Skip and replace w/ centroid if no intersections
        if len(fps) == 0:
            nwm_lakes.loc[idx, "geometry"] = nwm_lakes["geometry"][idx].centroid
            continue
        # Keep min hydroseq of all intersected FPs
        outlet_fp_id = fps[config.lakes.nwm.fp_id_field][fps["hydroseq"].idxmin()]
        nwm_lakes.loc[idx, "outlet_fp_id"] = outlet_fp_id
        # Find nearest ref_res to most downstream fp_id
        candidates = ref_res.sindex.nearest(
            fps["geometry"][fps["hydroseq"].idxmin()], max_distance=max_distance
        )
        # If we found a candidate, copy over all of (dam_name, nid, dam_id, geometry). Otherwise, replace w/ centroid
        if candidates.shape[1] != 0:
            nwm_lakes.loc[idx, ["dam_name", "nid", "dam_id", "geometry"]] = ref_res.loc[
                candidates[1, 0], ["dam_name", "nid", "dam_id", "geometry"]
            ]
        else:
            nwm_lakes.loc[idx, "geometry"] = nwm_lakes["geometry"][idx].centroid

    # join updated geometries and reference reservoir info back to nwm lakes points with associate flowpaths dataframe
    nwm_lakes_pt.drop(columns=["geometry"], inplace=True)
    nwm_lakes_pt = nwm_lakes_pt.merge(
        nwm_lakes[
            ["geometry", config.lakes.nwm.output_id_field, "dam_name", "nid", "dam_id", "outlet_fp_id"]
        ],
        on=config.lakes.nwm.output_id_field,
        how="left",
    )

    # replace associated fp id with better match
    nwm_lakes_pt.loc[~nwm_lakes_pt["outlet_fp_id"].isna(), config.lakes.nwm.fp_id_out_field] = nwm_lakes_pt[
        "outlet_fp_id"
    ]

    return gpd.GeoDataFrame(nwm_lakes_pt, crs=config.crs)

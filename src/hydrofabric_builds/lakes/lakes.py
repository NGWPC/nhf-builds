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


def build_nwm_lakes(cfg) -> gpd.GeoDataFrame:
    """Processes NWM lakes

    """
    # Preprocess lakes by associating with flowpaths if requested or if processed path does not exist
    if cfg.lakes.associate_flowpaths or not cfg.lakes.processed_path.exists():
        # Use nearest point association method
        if cfg.lakes.flowpath_association_method == "nearest_point":
            logger.info("Associating flowpath with points")
            gdf = associate_flowpaths_nearest_point(
                points_path=cfg.lakes.input_path,
                flowpaths_path=Path(cfg.build.reference_flowpaths_path),
                search_radius_m=cfg.lakes.search_radius_m,
                point_id=cfg.lakes.id_field,
                flowpath_id="flowpath_id",
                flowpath_id_out_field="ref_fp_id",
                points_layer=cfg.lakes.input_layer,
            )
        # use polygon flowpath outlet method
        elif cfg.lakes.flowpath_association_method == "polygon_outlet":
            logger.info("Associating flowpaths with polygons")
            gdf = associate_flowpaths_polygon_outlet(
                polygon_path=cfg.lakes.input_path,
                flowpaths_path=Path(cfg.build.reference_flowpaths_path),
                search_radius_m=cfg.lakes.search_radius_m,
                min_preferred_intersection_len_m=cfg.lakes.min_preferred_intersection_len_m,
                flowpath_id="flowpath_id",
                flowpath_id_out_field="ref_fp_id",
                polygon_layer=cfg.lakes.input_layer,
            )
            if cfg.lakes.attrib_src_path:
                gdf = join_attributes(
                    gdf,
                    attrib_dst_key=cfg.lakes.id_field,
                    attrib_src_path=cfg.lakes.attrib_src_path,
                    attrib_src_layer=cfg.lakes.attrib_src_layer,
                    attrib_src_key=cfg.lakes.attrib_src_key,
                    attrib_src_fields=cfg.lakes.fields.copy(),
                    rename=True,
                )

        # invalid method
        else:
            raise ValueError("Config contained invalid Lakes flowpath association method")

    # Save nwm_lakes layer to NHF
    gdf.to_file(cfg.lakes.nwm_lake_file_path, layer="lakes", driver="GPKG", overwrite=True)

    return gdf

# maybe this can be one function just with separate cfg for IDs?

def build_misc_lake(src, cfg):
    """Associates flowpaths for a lake file"""
    gdf = gpd.read_file(src)

    # Use nearest point association method
    if cfg.lakes.flowpath_association_method == "nearest_point":
            logger.info(f"Associating flowpath with points for {src}")
            gdf = associate_flowpaths_nearest_point(
                points_path=cfg.lakes.input_path,
                flowpaths_path=Path(cfg.build.reference_flowpaths_path),
                search_radius_m=cfg.lakes.search_radius_m,
                point_id=cfg.lakes.id_field,
                flowpath_id="flowpath_id",
                flowpath_id_out_field="ref_fp_id",
                points_layer=cfg.lakes.input_layer,
            )
        # use polygon flowpath outlet method
    elif cfg.lakes.flowpath_association_method == "polygon_outlet":
        logger.info(f"Associating flowpaths with polygons for {src}")
        gdf = associate_flowpaths_polygon_outlet(
            polygon_path=cfg.lakes.input_path,
            flowpaths_path=Path(cfg.build.reference_flowpaths_path),
            search_radius_m=cfg.lakes.search_radius_m,
            min_preferred_intersection_len_m=cfg.lakes.min_preferred_intersection_len_m,
            flowpath_id="flowpath_id",
            flowpath_id_out_field="ref_fp_id",
            polygon_layer=cfg.lakes.input_layer,
        )
        if cfg.lakes.attrib_src_path:
            gdf = join_attributes(
                gdf,
                attrib_dst_key=cfg.lakes.id_field,
                attrib_src_path=cfg.lakes.attrib_src_path,
                attrib_src_layer=cfg.lakes.attrib_src_layer,
                attrib_src_key=cfg.lakes.attrib_src_key,
                attrib_src_fields=cfg.lakes.fields.copy(),
                rename=True,
            )

    gdf.to_file()


def join_nid(nid_path, res_df):
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

    res_df = res_df[["dam_id", "nid", "ref_fab_wb", "x", "y"]].rename(columns={"nid": "nidid"}).copy()

    res_df = res_df.merge(nid_df, on='nidid', how='left')
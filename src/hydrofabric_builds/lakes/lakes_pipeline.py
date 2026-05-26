import logging

import geopandas as gpd

from hydrofabric_builds.config import HFConfig
from hydrofabric_builds.hydrofabric.utils import _crosswalk_nexus, _crosswalk_reference
from hydrofabric_builds.lakes.hydraulics import _populate_hydraulics
from hydrofabric_builds.lakes.lakes import (
    _associate_lake_flowpaths,
    _calculate_elevation__adhoc,
    _calculate_elevation__nwm,
    _calculate_elevation__refres,
    _calculate_elevation__refwb,
    _concat_lakes,
    _create_ids,
    _filter_adhoc_lakes,
    _filter_columns,
    _filter_ref_res,
    _fold_ref_res_to_nwm_lakes,
    _join_nid,
    _merge_ref_wb,
)

logger = logging.getLogger(__name__)


def lakes_pipeline(cfg: HFConfig) -> None:
    # use cached lakes if requested
    if cfg.lakes.use_cached_lakes:
        logger.info("Using cached lakes file to build lakes layer.")
        gdf = gpd.read_file(cfg.lakes.lakes_path)
        gdf.to_file(cfg.output_file_path, layer="lakes", driver="GPKG", overwrite=True)

    # if not lake types were selected to run, write blank layer
    elif (cfg.lakes.nwm.run or cfg.lakes.adhoc.run or cfg.lakes.ref_wb.run or cfg.lakes.ref_res.run) is False:
        logger.info("No lake types were selected to run. Writing blank layer to lakes.")
        gdf = gpd.GeoDataFrame(columns=cfg.lakes.fields, crs=cfg.crs)
        gdf.to_file(cfg.output_file_path, layer="lakes", driver="GPKG", overwrite=True)

    # else run pipeline
    else:
        # ---NWM lakes
        if cfg.lakes.nwm.run:
            logger.info("Running NWM lakes")
            gdf_nwm_lakes = _associate_lake_flowpaths(cfg, "nwm")
            # improve placement
            gdf_nwm_lakes = _fold_ref_res_to_nwm_lakes(cfg, gdf_nwm_lakes)
            gdf_nwm_lakes = _calculate_elevation__nwm(cfg, gdf_nwm_lakes)
        else:
            gdf_nwm_lakes = gpd.GeoDataFrame(columns=["geometry", "dam_id"], crs=cfg.crs)

        # # ---Adhoc lakes - optional: these are point geometries to be associated with flowpath and added to lakes layer
        if cfg.lakes.adhoc.run:
            logger.info("Running adhoc lakes")
            gdf_missing_adhoc, gdf_adhoc_ref_wb = _filter_adhoc_lakes(cfg)
            gdf_missing_adhoc = _associate_lake_flowpaths(cfg, "adhoc", gdf=gdf_missing_adhoc)
            gdf_missing_adhoc = _calculate_elevation__adhoc(cfg, gdf_missing_adhoc)
            gdf_missing_adhoc.to_file("data/lakes/tmp_adhoc.gpkg")
        else:
            gdf_missing_adhoc = gpd.GeoDataFrame({"geometry": []}, crs=cfg.crs)

        # # --ref wb - any lakes needed in the reference waterbodies dataset
        if cfg.lakes.ref_wb.run:
            logger.info("Running adhoc lakes found only in reference waterbodies")
            gdf_ref_wb = _associate_lake_flowpaths(cfg, "ref_wb", gdf=gdf_adhoc_ref_wb)
            gdf_ref_wb = _merge_ref_wb(cfg, gdf_ref_wb)
            gdf_ref_wb = _calculate_elevation__refwb(cfg, gdf_ref_wb)
            gdf_ref_wb.to_file("data/lakes/tmp_refwb.gpkg")
        else:
            gdf_ref_wb = gpd.GeoDataFrame({"geometry": []}, crs=cfg.crs)

        # ----Filter ref res to candidates and exclude res already used by nwm lakes and waterbodies
        if cfg.lakes.ref_res.run:
            logger.info("Running reference reservoirs")
            gdf_ref_res = _filter_ref_res(cfg, gdf_nwm_lakes, gdf_ref_wb)
            gdf_ref_res.to_file("data/lakes/tmp_refres1.gpkg")
            gdf_ref_res = _calculate_elevation__refres(cfg, gdf_ref_res)
            gdf_ref_res.to_file("data/lakes/tmp_refres2.gpkg")
        else:
            gdf_ref_res = gpd.GeoDataFrame({"geometry": []}, crs=cfg.crs)

        # ----CONCAT concat all tables
        logger.info("All lakes source files run. Concatenating lakes.")
        gdf_all_lks = _concat_lakes(cfg, gdf_nwm_lakes, gdf_missing_adhoc, gdf_ref_wb, gdf_ref_res)
        gdf_all_lks.to_file("data/lakes/tmp_all_lks33.gpkg")

        # --- NID attributes
        # join to NID based on NID_ID
        # keep NID cols / process [build_rfc_da_locs]
        logger.info("Joining lakes to NID")
        gdf_all_lks = _join_nid(cfg, gdf_all_lks)
        # gdf_all_lks.to_file("data/lakes/tmp_all_lks4444.csv")
        # gdf_all_lks.to_file("data/lakes/tmp_all_lks44443.gpkg", driver='GPKG')

        # --- Hydraulics
        # hydraulics
        # fill only what's missing
        # defaults if not enough data
        logger.info("Calculating hydraulic parameters")
        gdf_all_lks = _populate_hydraulics(gdf_all_lks)

        # # --- FINALIZE
        logger.info("Finalizing lakes layer")
        gdf_all_lks = _crosswalk_reference(hf_path=cfg.output_file_path, gdf=gdf_all_lks)
        gdf_all_lks = _crosswalk_nexus(hf_path=cfg.output_file_path, gdf=gdf_all_lks)
        gdf_all_lks = _create_ids(gdf=gdf_all_lks)
        gdf_all_lks = _filter_columns(gdf=gdf_all_lks, fields=cfg.lakes.fields)

        # cache lakes file and save to NHF
        gdf_all_lks.to_file(cfg.lakes.lakes_path, layer="lakes", driver="GPKG", overwrite=True)
        gdf_all_lks.to_file(cfg.output_file_path, layer="lakes", driver="GPKG", overwrite=True)

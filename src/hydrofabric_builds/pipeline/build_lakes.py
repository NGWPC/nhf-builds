"""Contains all code for building active NWM lakes in task"""

import logging
from typing import Any, cast

import geopandas as gpd

from hydrofabric_builds.config import HFConfig
from hydrofabric_builds.hydrofabric.utils import _crosswalk_nexus, _crosswalk_reference
from hydrofabric_builds.lakes.hydraulics import _populate_hydraulics
from hydrofabric_builds.lakes.lakes import (
    _associate_lake_flowpaths,
    _calculate_elevation,
    _concat_lakes,
    _create_ids,
    _crosswalk_fp_lk,
    _filter_columns,
    _filter_ref_res,
    _improve_placement,
    _join_nid,
    _merge_ref_wb,
)

logger = logging.getLogger(__name__)


def build_lakes(**context: dict[str, Any]) -> dict[str, Any]:
    """Builds lakes layer from all NWM lake sources

    Parameters
    ----------
    **context : dict
        Airflow-compatible context containing:
        - ti : TaskInstance for XCom operations
        - config : HFConfig with pipeline configuration
        - task_id : str identifier for this task
        - run_id : str identifier for this pipeline run
        - ds : str execution date
        - execution_date : datetime object

    """
    cfg = cast(HFConfig, context["config"])

    if cfg.lakes.use_cached_lakes:
        gdf = gpd.read_file(cfg.lakes.lakes_path)
        gdf.to_file(cfg.output_file_path, layer="lakes", driver="GPKG", overwrite=True)

    else:
        # ---NWM lakes - fp association
        # read in NWM lakes polygon
        # associate flowpaths with nwm_lakes - poly or point option
        # retain attributes
        gdf_nwm_lakes = _associate_lake_flowpaths(cfg, "nwm")

        # ---IMPROVE NWM LAKES PLACEMENT
        # read reference reservoirs
        # will return gdf with all sources and improved placements and ref res fields
        gdf_nwm_lakes = _improve_placement(gdf_nwm_lakes, cfg.lakes.ref_res)

        # ---Reference Waterbodies - optional: these are reference waterbody polygons to be included if IDs are requested
        gdf_ref_wb = _associate_lake_flowpaths(cfg, "ref_wb")
        gdf_ref_wb = _merge_ref_wb(gdf_ref_wb, cfg.lakes.ref_res.input_file)

        # ---Adhoc lakes - optional: these are point geometries to be associated with flowpath and added to lakes layer
        gdf_adhoc_lakes = _associate_lake_flowpaths(cfg, "adhoc")

        # ----Filter ref res to candidates and exclude res already used by nwm lakes and waterbodies
        gdf_ref_res = _filter_ref_res(cfg, gdf_nwm_lakes, gdf_ref_wb)

        # ----CONCAT concat all tables
        gdf_all_lks = _concat_lakes(gdf_nwm_lakes, gdf_adhoc_lakes, gdf_ref_wb, gdf_ref_res)

        # --- ELEVATION
        # Includes the elevation config setting to toggle if elevation will be calculated or if nulls returned
        gdf_all_lks = _calculate_elevation(gdf_all_lks, cfg.lakes.dem_path, cfg.lakes.calculate_elevation)

        # --- NID attributes
        # join to NID based on NID_ID
        # keep NID cols / process [build_rfc_da_locs]
        gdf_all_lks = _join_nid(gdf_all_lks, cfg.lakes.nid_path)

        # --- Hydraulics
        # hydraulics
        # fill only what's missing
        # defaults if not enough data
        gdf_all_lks = _populate_hydraulics(cfg, gdf_all_lks)

        # --- FINALIZE
        # finalize
        # keep columns
        # drop duplicates
        # create new ID
        _filter_columns(gdf=gdf_all_lks, fields=cfg.lakes.fields)
        _crosswalk_reference(hf_path=cfg.output_file_path, gdf=gdf_all_lks)
        _crosswalk_nexus(hf_path=cfg.output_file_path, gdf=gdf_all_lks)
        _create_ids(gdf=gdf_all_lks)

        # cache lakes file and save to NHF
        gdf_all_lks.to_file(cfg.lakes.lakes_path, layer="lakes", driver="GPKG", overwrite=True)
        gdf_all_lks.to_file(cfg.output_file_path, layer="lakes", driver="GPKG", overwrite=True)

    if cfg.lakes.fp_lk_crosswalk:
        gdf_fp_lk_crosswalk = _crosswalk_fp_lk(cfg)
        gdf_fp_lk_crosswalk.to_file(
            cfg.output_file_path, layer="lakes_flowpaths", driver="GPKG", overwrite=True
        )

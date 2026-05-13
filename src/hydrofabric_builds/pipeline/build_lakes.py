"""Contains all code for building active NWM lakes in task"""

import logging
from typing import Any, cast

import geopandas as gpd
import numpy as np

from hydrofabric_builds.config import HFConfig
from hydrofabric_builds.hydrofabric.utils import _crosswalk_nexus, _crosswalk_reference
from hydrofabric_builds.lakes.helpers import point_elevation, polygon_elevation
from hydrofabric_builds.lakes.hydraulics import populate_hydraulics
from hydrofabric_builds.lakes.lakes import (
    _associate_lake_flowpaths,
    _create_ids,
    _crosswalk_fp_lk,
    _filter_columns,
    _join_nid,
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
        if cfg.lakes.nwm.associate_flowpaths:
            gdf_nwm_lakes = _associate_lake_flowpaths(cfg, "nwm")
        else:
            gdf_nwm_lakes = gpd.read_file(cfg.lakes.nwm.input_file, layer=cfg.lakes.nwm.input_layer)

        # ---Adhoc lake file
        if cfg.lakes.adhoc.associate_flowpaths:
            gdf_adhoc_lakes = _associate_lake_flowpaths(cfg)
        else:
            gdf_adhoc_lakes = gpd.read_file(cfg.lakes.adhoc.input_file, layer=cfg.lakes.adhoc.input_layer)

        # ---IMPROVE PLACEMENT - insert here
        # read reference reservoirs
        # will return gdf with all sources and improved placements
        gdf_nwm_lakes = _improve_placement(gdf_nwm_lakes, cfg.lakes.ref_res)

        # ----FILTER
        # filter out only nwm_lakes and reference reservoirs that meet criteria
        # TODO: cleanup
        gdf_ref_res = _filter_ref_reservoirs(cfg.lakes.ref_res)

        # ----CONCAT
        gdf_all_lks = _concat_lakes(gdf_nwm_lakes, gdf_adhoc_lakes, gdf_ref_res)

        # --- ELEVATION
        # elevations functions - toggleable
        # polygon area
        # dam points
        if cfg.lakes.calculate_elevation:
            gdf = polygon_elevation(cfg.lakes.dem_path, gdf)
            gdf["dam_elev"] = point_elevation(cfg.lakes.dem_path, gdf)
        else:
            gdf["dam_elev"] = np.nan
            gdf["ref_elev"] = np.nan

        # --- NID attributes
        # join to NID based on NID_ID
        # keep NID cols / process [build_rfc_da_locs]
        gdf = _join_nid(gdf, cfg.lakes.nid_path)

        # --- Hydraulics
        # hydraulics
        # fill only what's missing
        # defaults if not enough data
        populate_hydraulics(cfg, gdf)

        # --- FINALIZE
        # finalize
        # keep columns
        # drop duplicates
        # create new ID
        _filter_columns(gdf=gdf, fields=cfg.lakes.fields)
        _crosswalk_reference(hf_path=cfg.output_file_path, gdf=gdf)
        _crosswalk_nexus(hf_path=cfg.output_file_path, gdf=gdf)
        _create_ids(gdf=gdf)

        # cache lakes file and save to NHF
        gdf.to_file(cfg.lakes.lakes_path, layer="lakes", driver="GPKG", overwrite=True)
        gdf.to_file(cfg.output_file_path, layer="lakes", driver="GPKG", overwrite=True)

    if cfg.lakes.fp_lk_crosswalk:
        gdf = _crosswalk_fp_lk(cfg)
        gdf.to_file(cfg.output_file_path, layer="lakes_flowpaths", driver="GPKG", overwrite=True)

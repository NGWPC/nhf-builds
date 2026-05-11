"""Contains all code for building active NWM lakes in task"""

import logging
from pathlib import Path
from typing import Any, cast

from hydrofabric_builds.config import HFConfig
from hydrofabric_builds.lakes.lakes import build_nwm_lakes, build_misc_lake
from hydrofabric_builds.lakes.helpers import polygon_elevation, extract_elev_at_points


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

    # ---NWM lakes - fp association
    # read in NWM lakes polygon
    # associate flowpaths with nwm_lakes - poly or point option
    # retain attributes
    build_nwm_lakes(cfg)

    # ---OPTIONAL NEW LAKES
    # read in new lakes loop of sources
    # associate flowpaths with new lakes - if lake_id in NWM lakes, join rfc-da info
    # TODO: maybe this isn't a separate task... need to join rfcda to nwm lakes?
    for src in cfg.lakes.additional_lake_paths:
        build_misc_lake(src, cfg)

    # ---IMPROVE PLACEMENT - insert here
    # read reference reservoirs
    # will return gdf with all sources and improved placements

    # ----FILTER
    # filter out only nwm_lakes and reference reservoirs that meet criteria
    # TODO: cleanup
    da = res[
        ((res["distance_to_fp_m"] < max_waterbody_nearest_dist_m) & (res["wb_areasqkm"] >= min_area_sqkm))
        | (res["dam_id"].isin(res_keep))
    ].copy()

    ids = da["ref_fab_wb"].dropna().unique().tolist()
    if not ids:
        raise ValueError("No ref_fab_wb IDs found in candidate reservoirs.")

    # --- ELEVATION
    # elevations functions - toggleable
    # polygon area
    # dam points
    gdf = polygon_elevation(dem, gdf)
    gdf = point_elevation(dem, gdf)

    # --- NID attributes
    # join to NID based on NID_ID
    # keep NID cols / process [build_rfc_da_locs]
    gdf = join_nid(gdf, cfg.lakes.nid_path)

    # --- Hydraulics
    # hydraulics
    # fill only what's missing
    # defaults if not enough data
    populate_hydaulics

    # --- FINALIZE
    # finalize
    # keep columns
    # drop duplicates
    # create new ID
    filter_columns()
    _crosswalk_reference
    _crosswalk_nexus()
    create_id()

    # --- crosswalk waterbodies and flowpaths [separate task?]


     

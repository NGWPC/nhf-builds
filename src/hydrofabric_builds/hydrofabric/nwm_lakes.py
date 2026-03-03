from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def crosswalk_nwm_lakes(hf_path: Path, nwm_lakes_path: Path) -> gpd.GeoDataFrame:
    """Crosswalks RFC-DA reservoirs with reference flowpaths and NHF flowpaths. Saves waterbodies table.

    Parameters
    ----------
    hf_path : Path
        Path to input hydrofabric gpkg
    rfcda_path : Path
        Path to RFC-DA gpkg

    """
    # read rfcda
    gdf_res = gpd.read_file(nwm_lakes_path)

    # read HF ref ID cross walk table
    hf_ref = gpd.read_file(hf_path, layer="reference_flowpaths")
    hf_fp = gpd.read_file(hf_path, layer="flowpaths")

    # join on cross walk table
    logger.info("Crosswalking reference flowpath IDs")
    gdf_res = gdf_res.loc[~gdf_res["ref_fab_fp"].isnull(), :].copy()
    gdf_res["ref_fab_fp"] = pd.to_numeric(gdf_res["ref_fab_fp"]).astype(np.int64)
    gdf_res = gdf_res.merge(hf_ref, left_on="ref_fab_fp", right_on="ref_fp_id", how="left")
    gdf_res = gdf_res.merge(hf_fp[["fp_id"]], on="fp_id", how="left")
    gdf_res = gdf_res.loc[~gdf_res["fp_id"].isnull(), :].copy()
    gdf_res["nwm_lake_id"] = range(1, gdf_res.shape[0] + 1)

    # select final attribute list
    gdf_res = gdf_res[
        [
            "nwm_lake_id",
            "lake_id",
            "fp_id",
            "ref_fp_id",
            # "dam_id",
            # "dam_name",
            # "dam_type",
            "LkArea",
            "LkMxE",
            "WeirC",
            "WeirL",
            "WeirE",
            # "OrficeC",
            # "OrficeA",
            # "OrficeE",
            "Dam_Length",
            "ifd",
            "geometry",
        ]
    ]

    return gdf_res

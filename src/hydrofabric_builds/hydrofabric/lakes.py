from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd

logger = logging.getLogger(__name__)


def complete_lakes(nwm_lakes_path: Path, fields: list[str]) -> gpd.GeoDataFrame:
    """Complete requested columns

    Parameters
    ----------
    nwm_lakes_path : Path
        Path to input hydrofabric gpkg
    rfcda_path : Path
        Path to RFC-DA gpkg

    """
    gdf_lakes = gpd.read_file(nwm_lakes_path)
    gdf_lakes["nwm_lake_id"] = range(1, gdf_lakes.shape[0] + 1)

    # add nulls for any missing columns requested
    for f in fields:
        if f not in gdf_lakes.columns:
            gdf_lakes[f] = None

    out_columns = ["nwm_lake_id", "fp_id"] + fields + ["geometry"]

    # select final attribute list
    return gdf_lakes[out_columns]

from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pandas as pd

from hydrofabric_builds.hydrolocations.usgs_NLDI_API import (
    nldi_basin_by_position,
    nldi_basin_by_site,
)

# ---- Config ----
AREA_CRS = "EPSG:6933"  # equal-area for basin area
WORK_CRS = "EPSG:5070"  # projection for buffers & intersections


@dataclass
class BasinResult:
    """sample basin class"""

    basin_gdf: gpd.GeoDataFrame  # CRS=EPSG:4326
    area_km2: float
    source: str  # 'nldi_site'|'nldi_position'|'none'


def basin_area_km2(basin_4326: gpd.GeoDataFrame) -> float:
    """Compute basin area in km² using an equal-area projection."""
    if basin_4326.empty:
        return 0.0
    basin_eq = basin_4326.to_crs(AREA_CRS)
    dissolved = basin_eq.dissolve().geometry.iloc[0]
    return float(dissolved.area / 1_000_000.0)


def choose_flowpath_for_gage(
    in_buf: gpd.GeoDataFrame,
    site_no: str | None,
    lon: float,
    lat: float,
    flow_id_col: str = "flowpath_id",
    area_col: str = "totdasqkm",
    area_match_pct: float = 0.15,
) -> tuple[str | None, BasinResult, gpd.GeoDataFrame]:
    """
    Given pre-filtered candidate flowpaths (in_buf) and a gage (lon/lat, optional site_no),

    pick the candidate whose 'totdasqkm' best matches the basin area from USGS NLDI.
    :param in_buf: flowpaths in the buffered area of the gage
    :param site_no: sit number of the gage
    :param lon: longitude of the gage
    :param lat: latitude of the gage
    :param flow_id_col: column name of flowpath_id
    :param area_col: column name of area
    :param area_match_pct: error percentage between the areas compared to each other
    :return: The best flowpath matched to the gage in GeoDtaFrame
    """
    # 1) basin from USGS NLDI
    basin_gdf = None
    source = "none"
    if site_no:
        basin_gdf = nldi_basin_by_site(site_no)
        if basin_gdf is not None:
            source = "nldi_site"
    if basin_gdf is None:
        basin_gdf = nldi_basin_by_position(lon, lat)
        if basin_gdf is not None:
            source = "nldi_position"

    if basin_gdf is None:
        basin_result = BasinResult(
            basin_gdf=gpd.GeoDataFrame(geometry=[], crs="EPSG:4326"),
            area_km2=0.0,
            source=source,
        )
    else:
        area_km2 = basin_area_km2(basin_gdf)
        basin_result = BasinResult(basin_gdf=basin_gdf, area_km2=area_km2, source=source)

    # helper: choose by area
    def _best_by_area(df: gpd.GeoDataFrame, target_km2: float) -> str | None:
        """
        Matches flowpath and gages

        :param df: gage geodataframe
        :param target_km2: usgs area in km2 as groundtruth
        :return: flowpath id that matches with the gage
        """
        if not pd.notna(target_km2) or target_km2 <= 0.0:
            return None
        if df.empty:
            return None
        if area_col not in df.columns:
            raise ValueError(f"'{area_col}' not found in flowpaths.")
        vals = pd.to_numeric(df[area_col], errors="coerce")
        diffs = (vals - target_km2).abs()
        if diffs.isna().all():
            return None
        rel_err = diffs / target_km2
        best_idx = rel_err.idxmin()
        best_rel_err = float(rel_err.loc[best_idx])
        if not np.isfinite(best_rel_err):
            return None
        return str(df.loc[best_idx, flow_id_col]) if best_rel_err <= area_match_pct else None

    selected_id = _best_by_area(in_buf, basin_result.area_km2) if not in_buf.empty else None
    return selected_id, basin_result, in_buf

import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd

logger = logging.getLogger(__name__)


def append_from_routelink(
    gdf: gpd.GeoDataFrame, routelink: Path, id_col_name: str, shape: Path | None
) -> gpd.GeoDataFrame:
    """Append gages from RouteLink file to GeoDataFrame

    Use ogr2ogr to convert NC file to GPKG and add EPSG:4326 georef i.e. ogr2ogr RouteLink.gpkg RouteLink.nc -t_srs EPSG:4326 -s_srs EPSG:4326

    Parameters
    ----------
    routelink : Path
        RouteLink file to extract from
    output_csv: Path
        Path to write gage list CSV
    shape: Path | None
        Shapefile to use for clipping
    """
    gages = gpd.read_file(routelink).to_crs(gdf.crs)

    # first get gages only
    gages = gages.loc[gages[id_col_name].str.strip() != ""].copy()

    # then check intersection if requested
    if shape:
        # Get boundary to clip to
        shp = gpd.read_file(shape).to_crs(gdf.crs)
        merged_geom = shp["geometry"].union_all()
        gages = gages.loc[gages["geometry"].intersects(merged_geom), :].copy()

    gages = gages.rename(columns={id_col_name: "site_no"})
    gages["site_no"] = gages["site_no"].str.strip()

    gages = gpd.GeoDataFrame(gages[["geometry", "site_no"]][~gages["site_no"].isin(gdf["site_no"])].copy())
    gages = pd.concat([gdf, gages])
    gages["geometry"] = gages["geometry"].force_2d()
    gages["status"] = gages["status"].fillna("-")
    logger.info("gages: added gages from RouteLink not already present in dataset")
    return gages

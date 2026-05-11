
import rasterio
from rasterstats import zonal_stats   
import numpy as np
from pathlib import Path
import geopandas as gpd

def polygon_elevation(dem_path):
    with rasterio.open(dem_path) as src:
            dem_crs = src.crs
            if dem_crs is None:
                raise ValueError("DEM has no CRS.")
            if ref_wbs.crs is None:
                raise ValueError("ref_wbs has no CRS; cannot reproject.")

            if ref_wbs.crs.to_string().upper() != dem_crs.to_string().upper():
                ref_wbs = ref_wbs.to_crs(dem_crs)
                stats = zonal_stats(
                    vectors=ref_wbs,  # GeoDataFrame or shapes
                    raster=str(dem_path),  # path to your DEM
                    stats="mean",
                    nodata=src.nodata if src.nodata is not None else None,
                    all_touched=False,  # or True if you want a more inclusive mask
                )
                ref_wbs["ref_elev"] = [s["mean"] for s in stats]
            else:
                ref_wbs["ref_elev"] = np.nan

def extract_elev_at_points(dem_path: str | Path, pts: gpd.GeoDataFrame) -> np.ndarray:
    """Sample DEM at point locations; returns 1D array of elevations."""
    pts = pts.copy()
    with rasterio.open(dem_path) as src:
        if pts.crs is None:
            raise ValueError("points must have a CRS")
        if src.crs is not None and pts.crs.to_string().upper() != src.crs.to_string().upper():
            pts = pts.to_crs(src.crs)
        coords = [(geom.x, geom.y) for geom in pts.geometry]
        samples = list(src.sample(coords))
        arr = np.array([s[0] if len(s) else np.nan for s in samples], dtype=float)
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
        return arr

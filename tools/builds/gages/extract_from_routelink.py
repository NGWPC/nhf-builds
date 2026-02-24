import argparse
from pathlib import Path

import geopandas
import pandas as pd


def merge_shape_geometries(path: str):
    """Merge several vector geometries into one for clipping

    Parameters
    ----------
    path: str
    """
    shp = geopandas.read_file(path).to_crs(epsg=4326)
    merged = shp["geometry"][0]
    for geom in shp["geometry"]:
        merged = merged.union(geom)
    return merged


def extract_from_routelink(routelink: Path, output_csv: Path, shape: Path | None):
    """Extract gages from RouteLink file

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
    gages = geopandas.read_file(routelink).to_crs(epsg=4326)
    coords = gages.get_coordinates()
    coords_x = coords["x"]
    coords_y = coords["y"]

    # Test for 'gages' field to be non empty and, if shapefile given, test for geometry intersection
    merged_geom = merge_shape_geometries(shape) if shape else None
    gage_filter = (
        lambda idx: gages["geometry"][idx].intersects(merged_geom) and gages["gages"][idx].strip() != ""
        if shape
        else lambda idx: gages["gages"][idx].strip() != ""
    )

    # Collect all gages that pass filter check into a DataFrame
    filtered_gages = pd.concat(
        pd.DataFrame(
            [[gages["gages"][idx].strip(), coords_x[idx], coords_y[idx]]], columns=["gageid", "lon", "lat"]
        )
        for idx in filter(
            gage_filter,
            (idx for idx, _ in gages.iterrows()),
        )
    )

    filtered_gages.to_csv(output_csv, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "routelink",
        type=Path,
        help="Path to RouteLink file",
    )
    parser.add_argument(
        "output_csv",
        type=Path,
        help="Output path of gages CSV",
    )
    parser.add_argument(
        "--shape",
        type=Path,
        default=None,
        help="Path of shapefile to filter gages with",
    )

    args = parser.parse_args()
    extract_from_routelink(args.routelink, args.output_csv, args.shape)

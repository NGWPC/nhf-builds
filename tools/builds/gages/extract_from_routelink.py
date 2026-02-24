import argparse
from pathlib import Path

import geopandas
import pandas as pd


def extract_from_routelink(routelink: Path, output_csv: Path, shape: Path | None) -> None:
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

    # first get gages only
    gages = gages.loc[gages["gages"].str.strip() != ""].copy()

    # then check intersection if requested
    if shape:
        # Get boundary to clip to
        shp = geopandas.read_file(shape).to_crs(epsg=4326)
        merged_geom = shp["geometry"].union_all()
        gages = gages.loc[gages["geometry"].intersects(merged_geom), :].copy()

    gages["lat"], gages["lon"] = gages["geometry"].y, gages["geometry"].x
    gages = gages.rename(columns={"gages": "gageid"})
    gages["gageid"] = gages["gageid"].str.strip()

    gages[["gageid", "lat", "lon"]].to_csv(output_csv, index=False)
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

    # Test for "gages" field to be non empty and, if shapefile given, test for geometry intersection
    merged_geom = geopandas.read_file(shape).to_crs(epsg=4326)["geometry"].union_all() if shape else None
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

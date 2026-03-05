"""Helpers for associating geometries with reference flowpaths"""

from pathlib import Path

import geopandas as gpd
import pandas as pd


def associate_flowpaths_nearest_point(
    points_path: Path,
    flowpaths_path: Path,
    search_radius_m: int | float,
    point_id: str,
    flowpath_id: str,
    flowpath_id_out_field: str,
    flowpath_layer: str | None = None,
    points_layer: str | None = None,
) -> gpd.GeoDataFrame:
    """Associate point geometries with flowpath lines by buffering by a search radius and selecting mimnium distance

    Adapted from gage nearest fp code

    Parameters
    ----------
    points_path : Path
        Points to associate with flowpath
    flowpaths_path : Path
        Flowpath linestrings
    search_radius_m : int | float
        Buffer radius for matching flowpaths
    point_id : str
        Column name for ID in points gdf
    flowpath_id : str
        Column name for ID in flowpath gdf
    flowpath_id_out_field: str
        Column name for the flowpath ID in output file, by default 'ref_fp_id'
    flowpath_layer: str | None
        Layer name if reading from a GPKG, by default None
    flowpath_layer: str | None
        Layer name if reading from a GPKG, by default None

    Returns
    -------
    gpd.GeoDataFrame
        Original point gdf including associated flowpath
    """
    # load
    gdf_flowpaths = (
        gpd.read_parquet(flowpaths_path)
        if ".parquet" in flowpaths_path.name
        else gpd.read_file(flowpaths_path, layer=flowpath_layer)
    )
    gdf_points = (
        gpd.read_file(points_path, layer=points_layer) if points_layer else gpd.read_file(points_path)
    )

    # coerce geometry to 2D linestings
    gdf_flowpaths["geometry"] = gdf_flowpaths["geometry"].line_merge()
    gdf_flowpaths["geometry"] = gdf_flowpaths["geometry"].force_2d()

    # change to reference flowpath CRS if not matching
    if gdf_points.crs != gdf_flowpaths.crs:
        gdf_points = gdf_points.to_crs(gdf_flowpaths.crs)
        assert gdf_points.crs == gdf_flowpaths.crs, "CRS does not match for flowpaths and points"

    # buffer points with search radius
    gdf_points_buffer = gdf_points.copy()
    gdf_points_buffer["geometry"] = gdf_points["geometry"].buffer(float(search_radius_m))

    # intersect points buffer with flowpaths
    joined = gpd.sjoin(gdf_points_buffer, gdf_flowpaths, predicate="intersects", how="left")

    # prepare matches - based on gages nearest fp
    out = {}
    # cycle through each point ID and its subset
    for pt_row, sub in joined.groupby(point_id):
        # get point geometry
        pt = gdf_points.loc[gdf_points[point_id] == pt_row, "geometry"].values[0]

        # select flowpath geometry and ID from flowpath table for each candidate list
        candidates = gdf_flowpaths.loc[
            gdf_flowpaths[flowpath_id].isin(sub[flowpath_id].values), [flowpath_id, "geometry"]
        ].copy()

        # if no flowpaths in buffer, skip
        if candidates.shape[0] == 0:
            continue

        # calculate distance from point
        candidates["dist"] = candidates["geometry"].distance(pt)

        # select first minimum distance
        # TODO: take lower hydrosequence if tie?
        best_fp = candidates.loc[candidates["dist"] == min(candidates["dist"]), flowpath_id].values[0]
        out[pt_row] = str(best_fp)

    # assign dict of points and flowpaths to point gdf
    for k, v in out.items():
        gdf_points.loc[gdf_points[point_id] == k, flowpath_id_out_field] = v

    # TODO: when fp is missing

    return gdf_points


def associate_flowpaths_polygon_outlet(
    polygon_path: Path,
    flowpaths_path: Path,
    search_radius_m: int | float,
    polygon_id: str,
    flowpath_id: str,
    flowpath_id_out_field: str = "fp_id",
    flowpath_layer: str | None = None,
    polygon_layer: str | None = None,
    attrib_source_file: Path |  None = None,
    attrib_source_layer: str |  None = None,
    attrib_source_key: str |  None = None,
) -> gpd.GeoDataFrame:
    """Associate the intersection of waterbody polygons and their flowpath outlets

    Adapted from above

    Parameters
    ----------
    polygon_path : Path
        Polygons to associate with flowpath
    flowpaths_path : Path
        Flowpath linestrings
    search_radius_m : int | float
        Buffer radius for matching flowpaths
    polygon_id : str
        Column name for ID in lakes gdf
    flowpath_id : str
        Column name for ID in flowpath gdf
    flowpath_id_out_field: str
        Column name for the flowpath ID in output file, by default 'fp_id'
    flowpath_layer: str | None
        Layer name if reading from a GPKG, by default None

    Returns
    -------
    gpd.GeoDataFrame
    Original lake gdf including associated flowpath

    """
    gdf_flowpaths = (
        gpd.read_parquet(flowpaths_path)
        if ".parquet" in flowpaths_path.name
        else gpd.read_file(flowpaths_path, layer=flowpath_layer)
    )
    gdf_poly = (
        gpd.read_file(polygon_path, layer=polygon_layer) if polygon_layer else gpd.read_file(polygon_path)
    )

    gdf_source = (
        gpd.read_file(attrib_source_files)
        if attrib_source_files is not None
        else None
    )
    if 

    # coerce geometry to 2D linestings
    gdf_flowpaths["geometry"] = gdf_flowpaths["geometry"].line_merge()
    gdf_flowpaths["geometry"] = gdf_flowpaths["geometry"].force_2d()

    # change to reference flowpath CRS if not matching
    if gdf_poly.crs != gdf_flowpaths.crs:
        gdf_points = gdf_poly.to_crs(gdf_flowpaths.crs)
        assert gdf_points.crs == gdf_flowpaths.crs, "CRS does not match for flowpaths and points"

    for idx, _lake in gdf_poly.iterrows():
        for search_radius in range(0, search_radius_m, search_radius_m / 10):
            buffered = (
                gdf_poly["geometry"][idx]
                if search_radius == 0
                else gdf_poly["geometry"][idx].buffer(search_radius)
            )
            candidates = gdf_flowpaths["geometry"].intersects(buffered, [flowpath_id, "hydroseq"])
            candidates = gdf_flowpaths.loc[candidates]
            if len(candidates["hydroseq"]) == 0:
                continue
            min_idx = candidates["hydroseq"].idxmin()
            gdf_poly.loc[idx, flowpath_id_out_field] = candidates[flowpath_id][min_idx]
            gdf_poly.loc[idx, "buffer_radius"] = search_radius
            break

    return gdf_poly

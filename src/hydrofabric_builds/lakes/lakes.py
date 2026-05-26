"""Contains all code for building active NWM lakes in task"""

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from hydrofabric_builds.config import HFConfig
from hydrofabric_builds.helpers.flowpath_association import (
    associate_flowpaths_nearest_point,
    associate_flowpaths_polygon_outlet,
    join_attributes,
)
from hydrofabric_builds.lakes.helpers import point_elevation, polygon_elevation

logger = logging.getLogger(__name__)


def _associate_lake_flowpaths(
    main_cfg: HFConfig, lake_type: str, gdf: gpd.GeoDataFrame | None = None
) -> gpd.GeoDataFrame:
    """Associate flowpaths and join attributes from source file if requested."""
    cfg = getattr(main_cfg.lakes, lake_type)

    # if a gdf is passed in, use it, if not read from config
    if gdf is None:
        try:
            gdf = gpd.read_file(cfg.path, layer=cfg.layer)
        except Exception as e:  # noqa: BLE001
            logger.info(
                f"Input file for lake type {lake_type} could not be read. Skipping flowpath association for {lake_type}. Exception: {e}"
            )
            # Return empty dataframe to cause rest of pipeline to work
            return gpd.GeoDataFrame(columns=["geometry", "lake_id"])

    # Preprocess lakes by associating with flowpaths if requested or if processed path does not exist
    if cfg.associate_flowpaths or not cfg.tmp_path.exists():
        # Use nearest point association method
        if cfg.flowpath_association_method == "nearest_point":
            logger.info(f"Associating {lake_type} flowpath with points")
            gdf = associate_flowpaths_nearest_point(
                # points_path=cfg.path,
                # points_layer=cfg.layer,
                gdf_points=gdf,
                flowpaths_path=Path(main_cfg.build.reference_flowpaths_path),
                search_radius_m=cfg.search_radius_m,
                point_id=cfg.id_field,
                flowpath_id="flowpath_id",
                flowpath_id_out_field="ref_fp_id",
            )
        # use polygon flowpath outlet method
        elif cfg.flowpath_association_method == "polygon_outlet":
            logger.info(f"Associating {lake_type} flowpaths with polygons")
            gdf = associate_flowpaths_polygon_outlet(
                # polygon_path=cfg.path,
                # polygon_layer=cfg.layer,
                gdf_poly=gdf,
                flowpaths_path=Path(main_cfg.build.reference_flowpaths_path),
                search_radius_m=cfg.search_radius_m,
                min_preferred_intersection_len_m=cfg.min_preferred_intersection_len_m,
                flowpath_id="flowpath_id",
                flowpath_id_out_field="ref_fp_id",
            )

        # invalid method
        else:
            raise ValueError(f"Config contained invalid flowpath association method for {lake_type}")

        if cfg.attrib_src_path:
            gdf = join_attributes(
                gdf,
                attrib_dst_key=cfg.id_field,
                attrib_src_path=cfg.attrib_src_path,
                attrib_src_layer=cfg.attrib_src_layer,
                attrib_src_key=cfg.attrib_src_key,
                attrib_src_fields=cfg.fields.copy(),
                rename=True,
            )
        if cfg.output_id_field not in gdf.columns:
            gdf = gdf.rename(columns={cfg.id_field: cfg.output_id_field})

        # Save nwm_lakes layer to NHF
        gdf.to_file(cfg.tmp_path, layer="lakes", driver="GPKG", overwrite=True)

    else:
        # read the pre-processed file to return
        gdf = gpd.read_file(cfg.tmp_path)

    return gdf


def _fold_ref_res_to_nwm_lakes(cfg: HFConfig, nwm_lakes_pt: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Takes in nwm_lakes GeoDataFrame with polygons and and returns a GeoDataFrame with points derived from reference reservoirs OR centroid if no reference available.

    Attempts to find the most downstream reference reservoir candidate by searching in proximity of the most downstream
    intersecting flowpath rather than searching in proximity of lake polygon.

    If reference point is available, also retains "dam_id", "dam_name" and "nid" columns from reference datapoint.
    """
    nwm_lakes = gpd.read_file(cfg.lakes.nwm.path, layer=cfg.lakes.nwm.layer).to_crs(cfg.crs)
    fp = gpd.read_parquet(cfg.build.reference_flowpaths_path).to_crs(cfg.crs)
    ref_res = gpd.read_file(cfg.lakes.ref_res.path).to_crs(cfg.crs)

    max_distance = cfg.lakes.nwm.max_search_distance_m

    nwm_lakes[["dam_name", "dam_id", "nid"]] = [pd.NA, pd.NA, pd.NA]
    nwm_lakes.rename(columns={cfg.lakes.nwm.id_field: cfg.lakes.nwm.output_id_field}, inplace=True)

    # for each lake
    for idx, _ in nwm_lakes.iterrows():
        # Limit FPs to those that intersect with *this* lake
        fps = fp[nwm_lakes["geometry"][idx].intersects(fp.geometry)]
        # Skip and replace w/ centroid if no intersections
        if len(fps) == 0:
            nwm_lakes.loc[idx, "geometry"] = nwm_lakes["geometry"][idx].centroid
            continue
        # Keep min hydroseq of all intersected FPs
        outlet_fp_id = fps[cfg.lakes.nwm.fp_id_field][fps["hydroseq"].idxmin()]
        nwm_lakes.loc[idx, "outlet_fp_id"] = outlet_fp_id
        # Find nearest ref_res to most downstream fp_id
        candidates = ref_res.sindex.nearest(
            fps["geometry"][fps["hydroseq"].idxmin()], max_distance=max_distance
        )
        # If we found a candidate, copy over all of (dam_name, nid, dam_id, geometry). Otherwise, replace w/ centroid
        if candidates.shape[1] != 0:
            nwm_lakes.loc[idx, ["dam_name", "nid", "dam_id", "geometry"]] = ref_res.loc[
                candidates[1, 0], ["dam_name", "nid", "dam_id", "geometry"]
            ]
        else:
            # TODO: most downstream flowpath interesction
            nwm_lakes.loc[idx, "geometry"] = nwm_lakes["geometry"][idx].centroid

    # join updated geometries and reference reservoir info back to nwm lakes points with associate flowpaths dataframe
    nwm_lakes_pt.drop(columns=["geometry"], inplace=True)
    nwm_lakes_pt = nwm_lakes_pt.merge(
        nwm_lakes[["geometry", cfg.lakes.nwm.output_id_field, "dam_name", "nid", "dam_id", "outlet_fp_id"]],
        on=cfg.lakes.nwm.output_id_field,
        how="left",
    )

    # replace associated fp id with better match
    nwm_lakes_pt.loc[~nwm_lakes_pt["outlet_fp_id"].isna(), cfg.lakes.nwm.fp_id_out_field] = nwm_lakes_pt[
        "outlet_fp_id"
    ]

    return gpd.GeoDataFrame(nwm_lakes_pt, crs=cfg.crs)


def _calculate_elevation__nwm(cfg: HFConfig, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Calculate elevation for NWM Lakes.

    Polygon elevation is used for `ref_elev`. Point elevation is used for `dam_elev`
    """
    if cfg.lakes.calculate_elevation:
        # polygons - join nwm lake polygons for polygon elevation (ref_elev)
        gdf_nwm_poly = gpd.read_file(cfg.lakes.nwm.path, layer=cfg.lakes.nwm.layer).to_crs(cfg.crs)
        gdf_nwm_poly = polygon_elevation(cfg.lakes.dem.path, gdf_nwm_poly, "ref_elev")
        gdf_nwm_poly = gdf_nwm_poly.rename(columns={cfg.lakes.nwm.id_field: cfg.lakes.nwm.output_id_field})
        gdf = gdf.merge(
            gdf_nwm_poly[[cfg.lakes.nwm.output_id_field, "ref_elev"]].copy(),
            on=cfg.lakes.nwm.output_id_field,
            how="left",
        )

        # point
        gdf["dam_elev"] = point_elevation(cfg.lakes.dem.path, gdf)
    else:
        gdf["dam_elev"] = np.nan
        gdf["ref_elev"] = np.nan

    return gdf


def _calculate_elevation__adhoc(cfg: HFConfig, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Calculate elevation for Adhoc Lakes

    Adhoc lakes only have points. Point elevation is used for both `dam_elev` and `ref_elev`
    """
    if cfg.lakes.calculate_elevation:
        # adhoc lakes are only points so use point elevation for both
        gdf["dam_elev"] = point_elevation(cfg.lakes.dem.path, gdf)
        gdf["ref_elev"] = gdf["dam_elev"].copy()
    else:
        gdf["dam_elev"] = np.nan
        gdf["ref_elev"] = np.nan

    return gdf


def _calculate_elevation__refwb(cfg: HFConfig, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Calculate elevation for Reference Waterbodies

    Polygon elevation is used for `ref_elev`.
    """
    if cfg.lakes.calculate_elevation:
        # TODO: give ref wbes better point placement
        # refwb are only polygon right now so use polygon for both
        gdf = polygon_elevation(cfg.lakes.dem.path, gdf, "ref_elev")
        gdf["dam_elev"] = gdf["ref_elev"].copy()
    else:
        gdf["dam_elev"] = np.nan
        gdf["ref_elev"] = np.nan

    return gdf


def _calculate_elevation__refres(cfg: HFConfig, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Calculate elevation for Reference Reservoirs

    Reference reservoirs are joined to reference waterbodies (polygon) to get polygon elevation (`ref_elev`)
    Point elevation is used for `dam_elev`
    """
    if cfg.lakes.calculate_elevation:
        # polygons - join ref wb polygons for polygon elevation (ref_elev)
        gdf_wb_poly = gpd.read_file(cfg.lakes.ref_wb.path).to_crs(cfg.crs)
        gdf_wb_poly = gdf_wb_poly.rename(columns={cfg.lakes.ref_wb.id_field: cfg.lakes.ref_res.ref_wb_id_col})
        gdf_wb_poly = gdf_wb_poly.loc[
            gdf_wb_poly[cfg.lakes.ref_res.ref_wb_id_col].isin(gdf[cfg.lakes.ref_res.ref_wb_id_col])
        ].copy()
        gdf_wb_poly = polygon_elevation(cfg.lakes.dem.path, gdf_wb_poly, "ref_elev")
        gdf = gdf.merge(
            gdf_wb_poly[[cfg.lakes.ref_res.ref_wb_id_col, "ref_elev"]].copy(),
            on=cfg.lakes.ref_res.ref_wb_id_col,
            how="left",
        )

        # point
        gdf["dam_elev"] = point_elevation(cfg.lakes.dem.path, gdf)
    else:
        gdf["dam_elev"] = np.nan
        gdf["ref_elev"] = np.nan

    return gdf


def _filter_adhoc_lakes(cfg: HFConfig) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Filters adhoc lakes to 1) null lake ID and 2) lake ID only in reference waterbodies

    Returns tuple of 1) missing null lakes and 2) required reference waterbodies
    """
    gdf = gpd.read_file(cfg.lakes.adhoc.path)
    missing = gdf.loc[gdf[cfg.lakes.adhoc.id_field] == -99999, :].copy()
    ref_wb = gdf.loc[gdf[cfg.lakes.adhoc.ref_wb_field] == True, :].copy()
    return missing, ref_wb


def _merge_ref_wb(cfg: HFConfig, gdf_ref_wb: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Merge reference reservoirs data into reference waterbodies (e.g. flowpath IDs, NID, area)"""
    gdf_ref_res = gpd.read_file(cfg.lakes.ref_res.path)

    if pd.api.types.is_object_dtype(gdf_ref_res[cfg.lakes.ref_res.ref_wb_id_col]):
        gdf_ref_wb[cfg.lakes.ref_wb.output_id_field] = (
            gdf_ref_wb[cfg.lakes.ref_wb.output_id_field].astype(pd.Int32Dtype()).astype(pd.StringDtype())
        )

    gdf_ref_wb = gdf_ref_wb.merge(
        gdf_ref_res[["ref_fab_fp", "ref_fab_wb", "nid", "wb_areasqkm"]],
        left_on=cfg.lakes.ref_wb.output_id_field,
        right_on=cfg.lakes.ref_res.ref_wb_id_col,
        how="left",
    )

    del gdf_ref_res
    return gdf_ref_wb


def _filter_ref_res(
    cfg: HFConfig, gdf_nwm: gpd.GeoDataFrame, gdf_ref_wb: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Filter reference reservoirs based on criteria

    Criteria for filtering:
    - dam_id is not in NWM lakes (reference reservoirs joined to NWM lakes in previous step)
    - dam_id is not in selected reference waterbodies
    - reference waterbody is present (`ref_fab_wb` is not null)
    - distance to flowpath is less than configured `ref_res_max_distance_m`
    - area is less than configured `min_wb_area_sqkm`

    Fields are renamed (ref_fab_fp : ref_fp_id) and (ref_wb_id : configured lake ID value from nwm.output_id_field)
    """
    res = gpd.read_file(cfg.lakes.ref_res.path)

    # filter to exclude dam_id that have already been joined to nwm lakes or included wb
    res = res.loc[
        ~res["dam_id"].isin(gdf_nwm["dam_id"])
        & ~res["dam_id"].isin(gdf_ref_wb["dam_id"])
        & res["ref_fab_wb"].isnull()
        == False,
        ["geometry", "dam_id", "ref_fab_fp", "ref_fab_wb", "nid", "wb_areasqkm", "distance_to_fp_m"],
    ].copy()

    # filter to criteria
    res = res[
        (
            (res["distance_to_fp_m"] < cfg.lakes.ref_res.max_distance_m)
            & (res["wb_areasqkm"] >= cfg.lakes.ref_res.min_wb_area_sqkm)
        )
        | (res["dam_id"].isin(cfg.lakes.ref_res.ref_res_keep))
    ].copy()

    res = res.rename(columns={"ref_fab_fp": "ref_fp_id", "ref_wb_wb": cfg.lakes.nwm.output_id_field})

    return res


def _concat_lakes(
    cfg: HFConfig,
    gdf_nwm: gpd.GeoDataFrame,
    gdf_adhoc: gpd.GeoDataFrame,
    gdf_ref_wb: gpd.GeoDataFrame,
    gdf_ref_res: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Concat all available lakes and force final CRS."""
    gdf_nwm = gdf_nwm.to_crs(cfg.crs)
    gdf_adhoc = gdf_adhoc.to_crs(cfg.crs)
    gdf_ref_wb = gdf_ref_wb.to_crs(cfg.crs)
    gdf_ref_res = gdf_ref_res.to_crs(cfg.crs)
    return pd.concat([gdf_nwm, gdf_adhoc, gdf_ref_wb, gdf_ref_res], ignore_index=True)


def _join_nid(cfg: HFConfig, res_df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Join National Inventory Dams (NID) data to all lake data

    This follows R code for reference-reservoirs.
    NID can be read from csv, parquet, or gpkg.
    Columns are all cast to snake_case.
    If `nid` is not a column, the process is skipped.
    """
    nid_path = Path(cfg.lakes.nid.path)
    if nid_path.suffix.lower() == ".gpkg":
        nid_df = gpd.read_file(nid_path)
    elif nid_path.suffix.lower() in {".parquet", ".pq"}:
        nid_df = pd.read_parquet(nid_path)
    else:
        nid_df = pd.read_csv(nid_path)

    nid_df.columns = [col.lower() for col in nid_df.columns]

    # cast types like R
    for col in ("spillway_type", "dam_type"):
        if col in nid_df.columns:
            nid_df[col] = nid_df[col].astype("string")

    for col in ("structural_height", "dam_height", "nid_height", "surface_area", "hydraulic_height"):
        if col in nid_df.columns:
            nid_df[col] = pd.to_numeric(nid_df[col], errors="coerce")

    if "surface_area" not in nid_df.columns:
        nid_df["surface_area"] = np.nan

    # keep only needed columns (loosely matching R)
    keep_cols = [
        "nidid",
        "dam_name",
        "dam_type",
        "spillway_type",
        "spillway_width",
        "dam_length",
        "dam_height",
        "structural_height",
        "hydraulic_height",
        "nid_height",
        "surface_area",
        "wb_areasqkm",
        "nid_storage",
        "normal_storage",
        "max_storage",
        "hazard",
        "purposes",
    ]

    keep_cols = [c for c in keep_cols if c in nid_df.columns]
    nid_df = nid_df[keep_cols].copy()

    # restrict NID to NID IDs in da$nid
    if "nid" not in res_df.columns:
        logger.info(
            "'nid'column was not found in lakes during NID join. NID data will not be used in hydraulics calculation"
        )
        res_df[keep_cols] = None
        return res_df

    nid_ids = res_df["nid"].dropna().unique()
    nid_df = nid_df[nid_df["nidid"].isin(nid_ids)].copy()

    res_df = res_df.rename(columns={"nid": "nidid"}).copy()
    res_df = res_df.merge(nid_df, on="nidid", how="left")
    res_df = pd.DataFrame(res_df)

    return res_df


def _filter_columns(gdf: gpd.GeoDataFrame, fields: list[str]) -> gpd.GeoDataFrame:
    """Final filter for columns.

    Include all IDs and requested columns. Fill with nulls if column not present.
    """
    # add nulls for any missing columns requested
    for f in fields:
        if f not in gdf.columns:
            gdf[f] = None

    out_columns = (
        ["nhf_lake_id", "ref_fp_id", "fp_id", "virtual_fp_id", "dn_nex_id", "dn_virtual_nex_id", "div_id"]
        + fields
        + ["geometry"]
    )

    # select final attribute list
    return gpd.GeoDataFrame(gdf[out_columns])


def _create_ids(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Create IDs for NHF"""
    # TODO: change to location-based
    gdf["nhf_lake_id"] = range(1, gdf.shape[0] + 1)
    return gdf


def _crosswalk_fp_lk() -> None:
    pass

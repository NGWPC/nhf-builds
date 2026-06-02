"""Contains all functions for building lakes"""

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
    """Associate flowpaths and join attributes from source file if requested.

    An attribute file can be supplied to join pre-computed attributes to lakes.
    The associated file will be saved so that it can be picked up in pipeline runs.

    Parameters
    ----------
    main_cfg : HFConfig
        Main HF Config
    lake_type : str
        The string representing the name of the lake type in the LakesConfig (e.g. 'nwm', 'ref_wb')
    gdf : gpd.GeoDataFrame | None, optional
        An in memory GFF. If None, will read from the lake type config, by default None

    Returns
    -------
    gpd.GeoDataFrame
        FP-associated geodataframe
    """
    # Get the cfg for the requested lake type
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
                gdf_points=gdf,
                flowpaths_path=Path(main_cfg.build.reference_flowpaths_path),
                search_radius_m=cfg.search_radius_m,
                point_id=cfg.id_field,
                flowpath_id=main_cfg.lakes.fp_id_field,
                flowpath_id_out_field=main_cfg.lakes.fp_id_out_field,
            )
        # use polygon flowpath outlet method
        elif cfg.flowpath_association_method == "polygon_outlet":
            logger.info(f"Associating {lake_type} flowpaths with polygons")
            gdf = associate_flowpaths_polygon_outlet(
                gdf_poly=gdf,
                flowpaths_path=Path(main_cfg.build.reference_flowpaths_path),
                search_radius_m=cfg.search_radius_m,
                min_preferred_intersection_len_m=cfg.min_preferred_intersection_len_m,
                flowpath_id=main_cfg.lakes.fp_id_field,
                flowpath_id_out_field=main_cfg.lakes.fp_id_out_field,
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
        if main_cfg.lakes.output_comid_field not in gdf.columns:
            gdf = gdf.rename(columns={cfg.id_field: main_cfg.lakes.output_comid_field})

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
    # only run if reference reservoirs are present
    if cfg.lakes.nwm.improve_placement_ref_res:
        nwm_lakes = gpd.read_file(cfg.lakes.nwm.path, layer=cfg.lakes.nwm.layer).to_crs(cfg.crs)
        fp = gpd.read_parquet(cfg.build.reference_flowpaths_path).to_crs(cfg.crs)
        try:
            ref_res = gpd.read_file(cfg.lakes.ref_res.path).to_crs(cfg.crs)
        except Exception:  # noqa: BLE001
            logger.info("Reference reservoirs could not be read. NWM lake placement was not adjusted.")
            nwm_lakes_pt[["nid", "dam_id"]] = None
            return gpd.GeoDataFrame(nwm_lakes_pt, crs=cfg.crs)

        max_distance = cfg.lakes.nwm.max_refres_search_distance_m

        nwm_lakes[["dam_name", "dam_id", "nid"]] = [pd.NA, pd.NA, pd.NA]
        nwm_lakes.rename(columns={cfg.lakes.nwm.id_field: cfg.lakes.output_comid_field}, inplace=True)

        # for each lake
        for idx, _ in nwm_lakes.iterrows():
            # Limit FPs to those that intersect with *this* lake
            fps = fp[nwm_lakes["geometry"][idx].intersects(fp.geometry)]

            # Skip and replace w/ centroid if no intersections
            if len(fps) == 0:
                nwm_lakes.loc[idx, "geometry"] = nwm_lakes["geometry"][idx].centroid
                continue

            # Keep min hydroseq of all intersected FPs (ref fp)
            outlet_fp_id = fps[cfg.lakes.fp_id_field][fps["hydroseq"].idxmin()]
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
                nwm_lakes.loc[idx, "geometry"] = nwm_lakes["geometry"][idx].centroid

        # join updated geometries and reference reservoir info back to nwm lakes points with associate flowpaths dataframe
        nwm_lakes_pt.drop(columns=["geometry"], inplace=True)
        nwm_lakes_pt = nwm_lakes_pt.merge(
            nwm_lakes[
                ["geometry", cfg.lakes.output_comid_field, "dam_name", "nid", "dam_id", "outlet_fp_id"]
            ],
            on=cfg.lakes.output_comid_field,
            how="left",
        )

        # replace associated fp id with better match
        nwm_lakes_pt.loc[~nwm_lakes_pt["outlet_fp_id"].isna(), cfg.lakes.fp_id_out_field] = nwm_lakes_pt[
            "outlet_fp_id"
        ]

        return gpd.GeoDataFrame(nwm_lakes_pt, crs=cfg.crs)

    else:
        nwm_lakes_pt[["nid", "dam_id"]] = None
        return gpd.GeoDataFrame(nwm_lakes_pt)


def _calculate_elevation__nwm(cfg: HFConfig, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Calculate elevation for NWM Lakes.

    Polygon elevation is used for `ref_elev`. Point elevation is used for `dam_elev`
    """
    if cfg.lakes.calculate_elevation:
        # polygons - join nwm lake polygons for polygon elevation (ref_elev)
        gdf_nwm_poly = gpd.read_file(cfg.lakes.nwm.path, layer=cfg.lakes.nwm.layer).to_crs(cfg.crs)
        logger.info("Calculating NWM elevations")
        gdf_nwm_poly = polygon_elevation(cfg.lakes.dem.path, gdf_nwm_poly, "ref_elev")
        gdf_nwm_poly = gdf_nwm_poly.rename(columns={cfg.lakes.nwm.id_field: cfg.lakes.output_comid_field})
        gdf = gdf.merge(
            gdf_nwm_poly[[cfg.lakes.output_comid_field, "ref_elev"]].copy(),
            on=cfg.lakes.output_comid_field,
            how="left",
        )
        # point
        gdf["dam_elev"] = point_elevation(cfg.lakes.dem.path, gdf)

    else:
        gdf["dam_elev"] = np.nan
        gdf["ref_elev"] = np.nan

    return gdf


def _calculate_elevation__refwb(cfg: HFConfig, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Calculate elevation for Reference Waterbodies

    Polygon elevation is used for `ref_elev`.
    """
    if cfg.lakes.calculate_elevation:
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
        # ref wb ID was changed to final output ID in filter step
        gdf_wb_poly = gpd.read_file(cfg.lakes.ref_wb.path).to_crs(cfg.crs)
        gdf_wb_poly = gdf_wb_poly.rename(columns={cfg.lakes.ref_wb.id_field: cfg.lakes.output_comid_field})
        gdf_wb_poly = gdf_wb_poly.loc[
            gdf_wb_poly[cfg.lakes.output_comid_field].isin(gdf[cfg.lakes.output_comid_field])
        ].copy()
        gdf_wb_poly = polygon_elevation(cfg.lakes.dem.path, gdf_wb_poly, "ref_elev")
        gdf = gdf.merge(
            gdf_wb_poly[[cfg.lakes.output_comid_field, "ref_elev"]].copy(),
            on=cfg.lakes.output_comid_field,
            how="left",
        )

        # point
        gdf["dam_elev"] = point_elevation(cfg.lakes.dem.path, gdf)
    else:
        gdf["dam_elev"] = np.nan
        gdf["ref_elev"] = np.nan

    return gdf


def _prep_ref_wb(cfg: HFConfig) -> gpd.GeoDataFrame:
    """Filters adhoc lakes to lake ID only in reference waterbodies

    Merge reference reservoirs data into reference waterbodies (e.g. flowpath IDs, NID, area)

    Returns required reference waterbodies with attributes
    """
    gdf_adhoc = gpd.read_file(cfg.lakes.adhoc.path)
    gdf_ref_res = gpd.read_file(cfg.lakes.ref_res.path)
    gdf_wb_polys = gpd.read_file(cfg.lakes.ref_wb.path)

    # select where reference waterbody is required
    gdf_adhoc = gdf_adhoc.loc[gdf_adhoc[cfg.lakes.adhoc.ref_wb_field] == True, :].copy()  # noqa: E712
    # cast ID to string if ref wb to string
    if pd.api.types.is_object_dtype(gdf_ref_res[cfg.lakes.ref_res.ref_wb_id_col]):
        gdf_adhoc[cfg.lakes.ref_wb.output_id_field] = (
            gdf_adhoc[cfg.lakes.ref_wb.output_id_field].astype(pd.Int32Dtype()).astype(pd.StringDtype())
        )

    # merge the ref res data
    gdf_adhoc = gdf_adhoc.merge(
        gdf_ref_res[["ref_fab_fp", "ref_fab_wb", "nid"]],
        left_on=cfg.lakes.ref_wb.output_id_field,
        right_on=cfg.lakes.ref_res.ref_wb_id_col,
        how="left",
    )

    # get geometry from ref wb polygons
    gdf_wb_polys["LkArea"] = gdf_wb_polys.geometry.area
    gdf_adhoc = gdf_adhoc.merge(
        gdf_wb_polys[[cfg.lakes.ref_wb.id_field, "LkArea"]],
        left_on=cfg.lakes.ref_wb.output_id_field,
        right_on=cfg.lakes.ref_wb.id_field,
    ).drop(columns=[cfg.lakes.ref_wb.id_field])

    return gdf_adhoc


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
    res = res.to_crs(cfg.crs)

    # filter to exclude dam_id that have already been joined to nwm lakes or included wb
    res = res.loc[
        ~res["dam_id"].isin(gdf_nwm["dam_id"])
        & ~res["dam_id"].isin(gdf_ref_wb["dam_id"])
        & (res["ref_fab_wb"].isnull() == False),  # noqa: E712
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

    res = res.rename(columns={"ref_fab_fp": "ref_fp_id", "ref_fab_wb": cfg.lakes.output_comid_field})

    return res


def _concat_lakes(cfg: HFConfig, gdfs: list[gpd.GeoDataFrame]) -> gpd.GeoDataFrame:
    """Concat all available lakes and force final CRS."""
    for i, gdf in enumerate(gdfs):
        gdfs[i] = gdf.to_crs(cfg.crs)
    return pd.concat(gdfs, ignore_index=True)


def _join_nid(cfg: HFConfig, res_df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Join National Inventory Dams (NID) data to all lake data

    This follows R code for reference-reservoirs.
    NID can be read from csv, parquet, or gpkg.
    Columns are all cast to snake_case.
    If `nid` is not a column or reference reservoirs is not run, the process is skipped.

    Deduplication:
    There can be multiple NID dams for a single NID ID. There can also be multiple reference reservoirs and NWM lakes
    that share a NID ID.
    An NWM lake point is picked first and exclude from deduplication calculations.
    Choose the dam that is spatially nearest to a NID dam to keep when there are multiple options.
    """
    # Needed columns
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

    # Don't run if reference reservoirs were not run as this is the dataset with the NID ID
    if not cfg.lakes.ref_res.run:
        logger.info("Reference reservoirs not ran so skipping NID join")
        # return with required columns
        res_df[keep_cols] = None
        return res_df

    # Return guard in case NID is not found (e.g. oCONUS)
    try:
        nid_path = Path(cfg.lakes.nid.path)
        nid_df = pd.read_csv(nid_path)
    except Exception as e:  # noqa: BLE001
        logger.info(f"NID table not read, skipping NID. Exception: {e}")
        # return with required columns
        res_df[keep_cols] = None
        return res_df

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

    # Make NID spatial for de-duplicating
    nid_df = nid_df.loc[~nid_df["latitude"].isnull() & ~nid_df["longitude"].isnull()].copy()
    nid_df = gpd.GeoDataFrame(
        nid_df, geometry=gpd.points_from_xy(nid_df["latitude"], nid_df["longitude"]), crs=4326
    )
    nid_df = nid_df.to_crs(res_df.crs)

    # restrict NID to NID IDs in da$nid
    if "nid" not in res_df.columns:
        logger.info(
            "'nid'column was not found in lakes during NID join. NID data will not be used in hydraulics calculation"
        )
        res_df[keep_cols] = None
        return res_df

    # keep only NID in reservoirs
    nid_df = nid_df.rename(columns={"nidid": "nid", "dam_name": "dam_name_nid"})
    nid_ids = res_df["nid"].dropna().unique()
    nid_df = nid_df[nid_df["nid"].isin(nid_ids)].copy()

    # Drop pure duplicates from res before continuing
    res_df = res_df.drop_duplicates(keep="first")

    # remove NWM to be held separately
    nwm_df = res_df.loc[~res_df["attrib_src"].isnull()].copy()
    res_df = res_df.loc[
        (res_df["attrib_src"].isnull())
        & (~res_df["nid"].isin(nwm_df["nid"]))
        & (~res_df["dam_id"].isin(nwm_df["dam_id"]))
    ].copy()

    # merge NID and res
    res_df = res_df.merge(nid_df, on="nid", how="left")

    # Duplicates: There are potential for many dam_id : many NID
    # Get duplicated dam_id and nid: this means there were multiple NIDs for a single dam
    # locate the duplicates, mark them as all true with keep=false, then drop the extras for this subset
    res_df = res_df.replace("<NA>", None)
    res_nid = res_df.loc[~res_df["dam_id"].isnull() & ~res_df["nid"].isnull()].copy()

    # get out duplicated and keep all for now
    duplicated = res_nid.loc[res_nid.duplicated(subset=["dam_id", "nid"], keep=False)].drop_duplicates()
    duplicated = duplicated.set_geometry("geometry_x")

    # remove all duplicates being evaluated from original df, do not keep any (keep=false)
    res_df = res_df.drop_duplicates(subset=["dam_id", "nid"], keep=False)

    # get only the NIDs that are duplicated to reduce spatial search
    nid_subset = nid_df.loc[nid_df["nid"].isin(duplicated["nid"])].copy()

    # groupby each NID from the NID subset table
    # spatial join between the duplicated value (dam_id and nid) and corresponding NID group to get the distance between points
    grouped = nid_subset.groupby("nid")
    nearest_list = []
    for name, group in grouped:
        dupe = duplicated.loc[duplicated["nid"] == name, :].copy()
        group.index.rename("tmp_index", inplace=True)
        nearest_list.append(
            gpd.sjoin_nearest(
                dupe, group[["geometry", "nid", "dam_name_nid"]], distance_col="dist", how="left"
            )
        )

    # gdf_nearest includes all the duplicates with the distance between dam and nearest NID dam
    gdf_nearest = pd.concat(nearest_list).reset_index(drop=True)

    # get the index of the lowest distance per NID and keep only the dam with lowest NID distance even if there are multiple dams
    keep_index = gdf_nearest[["dam_id", "dist", "nid_right"]].groupby("nid_right").idxmin(numeric_only=True)
    gdf_dupes_removed = gdf_nearest.loc[keep_index["dist"]].copy()

    # geomtry x is the original geometry
    gdf_dupes_removed = gdf_dupes_removed.rename(columns={"nid_left": "nid", "geometry_x": "geometry"}).drop(
        columns=["nid_right", "geometry_y"]
    )

    res_df = (
        res_df.rename(columns={"geometry_x": "geometry"})
        .drop(columns=["geometry_y"])
        .set_geometry("geometry")
    )

    # ensure no duplicated geometries though this should  be handled
    res_df = res_df.drop_duplicates(subset=["geometry"], keep="first")

    # add in the desired duplicates and nwm
    output = pd.concat([res_df, gdf_dupes_removed, nwm_df])

    return output


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

    gdf.replace(pd.NA, None, inplace=True)

    # select final attribute list
    return gpd.GeoDataFrame(gdf[out_columns])


def _create_ids(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Create IDs for NHF"""
    # TODO: change to location-based
    gdf["nhf_lake_id"] = range(1, gdf.shape[0] + 1)
    return gdf


def _assert_nwm_lakes(cfg: HFConfig, gdf_all_lks: gpd.GeoDataFrame) -> None:
    """Assert all NWM lakes are present in the final output"""
    gdf_nwm_lakes = gpd.read_file(cfg.lakes.nwm.path, layer=cfg.lakes.nwm.layer)

    # check that the fields are the same name in NWM as output and cast if needed
    if gdf_nwm_lakes[cfg.lakes.nwm.id_field].dtype != gdf_all_lks[cfg.lakes.output_comid_field].dtype:
        gdf_nwm_lakes[cfg.lakes.nwm.id_field] = gdf_nwm_lakes[cfg.lakes.nwm.id_field].astype(pd.StringDtype())
        gdf_all_lks[cfg.lakes.output_comid_field] = gdf_all_lks[cfg.lakes.output_comid_field].astype(
            pd.StringDtype()
        )

    notin = gdf_nwm_lakes.loc[
        ~gdf_nwm_lakes[cfg.lakes.nwm.id_field].isin(gdf_all_lks[cfg.lakes.output_comid_field])
    ]

    if len(notin) == 0:
        logger.info(f"All {len(gdf_nwm_lakes[cfg.lakes.nwm.id_field])} NWM lakes included")
    else:
        notin.to_file(cfg.lakes.lakes_path.parent / "missing_lakes.gpkg")
        raise ValueError(
            f"{len(notin)} missing NWM lakes found. Wrote missing lakes to {(cfg.lakes.lakes_path.parent / 'missing_lakes.gpkg')}"
        )

    return


def _crosswalk_fp_lk() -> None:
    pass

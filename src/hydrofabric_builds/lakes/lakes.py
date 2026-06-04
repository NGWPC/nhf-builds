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
from hydrofabric_builds.pipeline.processing import _encode_unique

logger = logging.getLogger(__name__)


def _read_inputs(cfg: HFConfig) -> dict[str, gpd.GeoDataFrame]:
    """Read commonly used inputs. Return an empty geodataframe if no file to support type checking throughout.

    Functions will test if gdf is empty.
    """
    inputs = {}

    inputs["nwm_lakes"] = (
        gpd.read_file(cfg.lakes.nwm.path, layer=cfg.lakes.nwm.layer).to_crs(cfg.crs)
        if cfg.lakes.nwm.path.exists()
        else gpd.GeoDataFrame(geometry=[], crs=cfg.crs)
    )
    inputs["nid"] = pd.read_csv(cfg.lakes.nid.path) if cfg.lakes.nid.path.exists() else pd.DataFrame()
    inputs["adhoc"] = (
        gpd.read_file(cfg.lakes.adhoc.path).to_crs(cfg.crs)
        if cfg.lakes.adhoc.path.exists()
        else gpd.GeoDataFrame(geometry=[], crs=cfg.crs)
    )
    inputs["ref_res"] = (
        gpd.read_file(cfg.lakes.ref_res.path).to_crs(cfg.crs)
        if cfg.lakes.ref_res.path.exists()
        else gpd.GeoDataFrame(geometry=[], crs=cfg.crs)
    )
    inputs["ref_wb"] = (
        gpd.read_file(cfg.lakes.ref_wb.path).to_crs(cfg.crs)
        if cfg.lakes.ref_wb.path.exists()
        else gpd.GeoDataFrame(geometry=[], crs=cfg.crs)
    )

    return inputs


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
    if cfg.associate_flowpaths or not cfg.fp_associated_path.exists():
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
        gdf.to_file(cfg.fp_associated_path, layer="lakes", driver="GPKG", overwrite=True)

    else:
        # read the pre-processed file to return
        gdf = gpd.read_file(cfg.fp_associated_path)

    return gdf


def _fold_ref_res_to_nwm_lakes(
    cfg: HFConfig, nwm_lakes_pt: gpd.GeoDataFrame, nwm_lakes_orig: gpd.GeoDataFrame, ref_res: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Takes in nwm_lakes GeoDataFrame with polygons and and returns a GeoDataFrame with points derived from reference reservoirs OR centroid if no reference available.

    Attempts to find the most downstream reference reservoir candidate by searching in proximity of the most downstream
    intersecting flowpath rather than searching in proximity of lake polygon.

    If reference point is available, also retains "dam_id", "dam_name" and "nid" columns from reference datapoint.
    """
    # only run if reference reservoirs are present, requested, and nwm lakes have polygons

    if cfg.lakes.nwm.improve_placement_path.exists() and cfg.lakes.nwm.use_cached_improve_placement:
        nwm_lakes_pt = gpd.read_file(cfg.lakes.nwm.improve_placement_path)
        return nwm_lakes_pt

    elif (
        cfg.lakes.nwm.improve_placement_ref_res
        and not ref_res.empty
        and nwm_lakes_orig.geometry.iloc[0].geom_type in ["Polygon", "MultiPolygon"]
    ):
        logger.info("Improving NWM lake placement with reference reservoirs.")
        fp = gpd.read_parquet(cfg.build.reference_flowpaths_path).to_crs(cfg.crs)
        nwm_lakes_poly = nwm_lakes_orig.copy().to_crs(cfg.crs)

        max_distance = cfg.lakes.nwm.max_refres_search_distance_m

        nwm_lakes_poly[["dam_name", "dam_id", "nid"]] = [pd.NA, pd.NA, pd.NA]
        nwm_lakes_poly.rename(columns={cfg.lakes.nwm.id_field: cfg.lakes.output_comid_field}, inplace=True)

        # for each lake
        for idx, _ in nwm_lakes_poly.iterrows():
            # Limit FPs to those that intersect with *this* lake
            fps = fp[nwm_lakes_poly["geometry"][idx].intersects(fp.geometry)]

            # Skip and replace w/ centroid if no intersections
            if len(fps) == 0:
                nwm_lakes_poly.loc[idx, "geometry"] = nwm_lakes_poly["geometry"][idx].centroid
                continue

            # Keep min hydroseq of all intersected FPs (ref fp)
            outlet_fp_id = fps[cfg.lakes.fp_id_field][fps["hydroseq"].idxmin()]
            nwm_lakes_poly.loc[idx, "outlet_fp_id"] = outlet_fp_id

            # Find nearest ref_res to most downstream fp_id
            candidates = ref_res.sindex.nearest(
                fps["geometry"][fps["hydroseq"].idxmin()], max_distance=max_distance
            )
            # If we found a candidate, copy over all of (dam_name, nid, dam_id, geometry). Otherwise, replace w/ centroid
            if candidates.shape[1] != 0:
                nwm_lakes_poly.loc[idx, ["dam_name", "nid", "dam_id", "geometry"]] = ref_res.loc[
                    candidates[1, 0], ["dam_name", "nid", "dam_id", "geometry"]
                ]
            else:
                nwm_lakes_poly.loc[idx, "geometry"] = nwm_lakes_poly["geometry"][idx].centroid

        # join updated geometries and reference reservoir info back to nwm lakes points with associate flowpaths dataframe
        nwm_lakes_pt.drop(columns=["geometry"], inplace=True)
        nwm_lakes_pt = nwm_lakes_pt.merge(
            nwm_lakes_poly[
                ["geometry", cfg.lakes.output_comid_field, "dam_name", "nid", "dam_id", "outlet_fp_id"]
            ],
            on=cfg.lakes.output_comid_field,
            how="left",
        )

        # replace associated fp id with better match
        nwm_lakes_pt.loc[~nwm_lakes_pt["outlet_fp_id"].isna(), cfg.lakes.fp_id_out_field] = nwm_lakes_pt[
            "outlet_fp_id"
        ]

        gdf = gpd.GeoDataFrame(nwm_lakes_pt, crs=cfg.crs)
        gdf.to_file(cfg.lakes.nwm.improve_placement_path)
        return gdf

    else:
        nwm_lakes_pt[["nid", "dam_id"]] = None
        return gpd.GeoDataFrame(nwm_lakes_pt)


def _calculate_elevation__nwm(
    cfg: HFConfig, gdf_nwm_pts: gpd.GeoDataFrame, gdf_nwm_orig: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Calculate elevation for NWM Lakes.

    Polygon elevation is used for `ref_elev`. Point elevation is used for `dam_elev`
    """
    if cfg.lakes.calculate_elevation:
        logger.info("Calculating NWM elevations")
        # if original nwm lakes is polygons - join nwm lake polygons for polygon elevation (ref_elev)
        if gdf_nwm_orig.geometry.iloc[0].geom_type in ["Polygon", "MultiPolygon"]:
            gdf_nwm_poly = polygon_elevation(cfg.lakes.dem.path, gdf_nwm_orig, "ref_elev")
            gdf_nwm_poly = gdf_nwm_poly.rename(columns={cfg.lakes.nwm.id_field: cfg.lakes.output_comid_field})
            gdf_nwm_poly = gdf_nwm_poly.to_crs(cfg.crs)
            gdf_nwm_poly["nwm_lakes_area"] = gdf_nwm_poly.area
            gdf_nwm_pts = gdf_nwm_pts.merge(
                gdf_nwm_poly[[cfg.lakes.output_comid_field, "ref_elev", "nwm_lakes_area"]].copy(),
                on=cfg.lakes.output_comid_field,
                how="left",
            )
            gdf_nwm_pts["dam_elev"] = point_elevation(cfg.lakes.dem.path, gdf_nwm_pts)
            gdf_nwm_pts["LkArea"] = np.where(
                gdf_nwm_pts["LkArea"].isnull(),
                gdf_nwm_pts["nwm_lakes_area"] / 1_000_000.0,
                gdf_nwm_pts["LkArea"],
            )

        # if all points
        else:
            gdf_nwm_pts["dam_elev"] = point_elevation(cfg.lakes.dem.path, gdf_nwm_pts)
            gdf_nwm_pts["ref_elev"] = gdf_nwm_pts["dam_elev"].copy()

    else:
        gdf_nwm_pts["dam_elev"] = np.nan
        gdf_nwm_pts["ref_elev"] = np.nan

    return gdf_nwm_pts


def _calculate_elevation__refwb(
    cfg: HFConfig, gdf_refwb_pts: gpd.GeoDataFrame, gdf_refwb_poly: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Calculate elevation for Reference Waterbodies

    Polygon elevation is used for `ref_elev`.
    """
    if cfg.lakes.calculate_elevation:
        gdf_refwb_poly = gdf_refwb_poly.loc[
            gdf_refwb_poly[cfg.lakes.ref_wb.id_field].isin(gdf_refwb_pts[cfg.lakes.output_comid_field])
        ].copy()
        gdf_refwb_poly = polygon_elevation(cfg.lakes.dem.path, gdf_refwb_poly, "ref_elev")
        gdf_refwb_poly["dam_elev"] = gdf_refwb_poly["ref_elev"].copy()
        gdf_refwb_pts = gdf_refwb_pts.merge(
            gdf_refwb_poly[[cfg.lakes.ref_wb.id_field, "ref_elev", "dam_elev"]],
            left_on=cfg.lakes.output_comid_field,
            right_on=cfg.lakes.ref_wb.id_field,
            how="left",
        )
    else:
        gdf_refwb_pts["dam_elev"] = np.nan
        gdf_refwb_pts["ref_elev"] = np.nan

    return gdf_refwb_pts


def _calculate_elevation__refres(
    cfg: HFConfig, gdf_ref_res: gpd.GeoDataFrame, gdf_wb_poly: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Calculate elevation for Reference Reservoirs

    Reference reservoirs are joined to reference waterbodies (polygon) to get polygon elevation (`ref_elev`)
    Point elevation is used for `dam_elev`
    """
    if cfg.lakes.calculate_elevation:
        # polygons - join ref wb polygons for polygon elevation (ref_elev)
        # ref wb ID was changed to final output ID in filter step
        gdf_wb_poly = gdf_wb_poly.rename(columns={cfg.lakes.ref_wb.id_field: cfg.lakes.output_comid_field})
        gdf_wb_poly = gdf_wb_poly.loc[
            gdf_wb_poly[cfg.lakes.output_comid_field].isin(gdf_ref_res[cfg.lakes.output_comid_field])
        ].copy()
        gdf_wb_poly = polygon_elevation(cfg.lakes.dem.path, gdf_wb_poly, "ref_elev")
        gdf_ref_res = gdf_ref_res.merge(
            gdf_wb_poly[[cfg.lakes.output_comid_field, "ref_elev"]].copy(),
            on=cfg.lakes.output_comid_field,
            how="left",
        )

        # point
        gdf_ref_res["dam_elev"] = point_elevation(cfg.lakes.dem.path, gdf_ref_res)

    else:
        gdf_ref_res["dam_elev"] = np.nan
        gdf_ref_res["ref_elev"] = np.nan

    return gdf_ref_res


def _prep_ref_wb(
    cfg: HFConfig, gdf_adhoc: gpd.GeoDataFrame, gdf_ref_res: gpd.GeoDataFrame, gdf_wb_polys: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Filters adhoc lakes to lake ID only in reference waterbodies

    Merge reference reservoirs data into reference waterbodies (e.g. flowpath IDs, NID, area)

    Returns required reference waterbodies with attributes
    """
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

    # get geometry from ref wb polygons (area in m², convert to km²)
    gdf_wb_polys["LkArea"] = gdf_wb_polys.geometry.area / 1_000_000.0
    gdf_adhoc = gdf_adhoc.merge(
        gdf_wb_polys[[cfg.lakes.ref_wb.id_field, "LkArea"]],
        left_on=cfg.lakes.ref_wb.output_id_field,
        right_on=cfg.lakes.ref_wb.id_field,
        how="inner",
    ).drop(columns=[cfg.lakes.ref_wb.id_field])

    return gdf_adhoc


def _filter_ref_res(
    cfg: HFConfig, gdf_ref_res: gpd.GeoDataFrame, gdf_nwm: gpd.GeoDataFrame, gdf_ref_wb: gpd.GeoDataFrame
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
    # filter to exclude dam_id that have already been joined to nwm lakes or included wb
    gdf_ref_res = gdf_ref_res.loc[
        ~gdf_ref_res["dam_id"].isin(gdf_nwm["dam_id"])
        & ~gdf_ref_res["dam_id"].isin(gdf_ref_wb["dam_id"])
        & (gdf_ref_res["ref_fab_wb"].isnull() == False),  # noqa: E712
        ["geometry", "dam_id", "ref_fab_fp", "ref_fab_wb", "nid", "wb_areasqkm", "distance_to_fp_m"],
    ].copy()

    # filter to criteria
    gdf_ref_res = gdf_ref_res[
        (
            (gdf_ref_res["distance_to_fp_m"] < cfg.lakes.ref_res.max_distance_m)
            & (gdf_ref_res["wb_areasqkm"] >= cfg.lakes.ref_res.min_wb_area_sqkm)
        )
        | (gdf_ref_res["dam_id"].isin(cfg.lakes.ref_res.ref_res_keep))
    ].copy()

    gdf_ref_res = gdf_ref_res.rename(
        columns={"ref_fab_fp": "ref_fp_id", "ref_fab_wb": cfg.lakes.output_comid_field}
    )

    return gdf_ref_res


def _concat_lakes(cfg: HFConfig, gdfs: list[gpd.GeoDataFrame]) -> gpd.GeoDataFrame:
    """Concat all available lakes and force final CRS."""
    for i, gdf in enumerate(gdfs):
        gdfs[i] = gdf.to_crs(cfg.crs)
    return pd.concat(gdfs, ignore_index=True)


def _dedup_lake_id(
    cfg: HFConfig,
    gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Deduplicate by lake_id (COMID) after concatenation, before NID join.

    When the same lake_id appears from multiple sources:
    1. NWM lakes (attrib_src is set) are preferred over ref_res / ref_wb
    2. Among same-priority duplicates, the most downstream (lowest hydroseq) wins
    """
    dupe_mask = gdf[cfg.lakes.output_comid_field].duplicated(keep=False)
    if not dupe_mask.any():
        logger.info("No duplicate lake_id values to resolve")
        return gdf

    n_dupe_groups = gdf.loc[dupe_mask, cfg.lakes.output_comid_field].nunique()
    logger.info(
        f"Deduplicating {cfg.lakes.output_comid_field}: "
        f"{dupe_mask.sum()} rows across {n_dupe_groups} duplicate groups"
    )

    # Build hydroseq lookup: ref_fp_id -> fp_id -> hydroseq
    # The NHF GPKG at cfg.output_file_path already has these layers
    gdf_ref_fp = gpd.read_file(cfg.output_file_path, layer="reference_flowpaths")
    gdf_fp = gpd.read_file(cfg.output_file_path, layer="flowpaths")

    hydro_lookup = gdf_ref_fp.merge(gdf_fp[["fp_id", "hydroseq"]], on="fp_id", how="left").dropna(
        subset=["fp_id"]
    )[["ref_fp_id", "hydroseq"]]
    hydro_lookup["ref_fp_id"] = pd.to_numeric(hydro_lookup["ref_fp_id"], errors="coerce")
    hydro_lookup = hydro_lookup.dropna(subset=["ref_fp_id"]).drop_duplicates(subset="ref_fp_id")

    # Tag source priority: NWM (has attrib_src) = 0, else = 1
    has_attrib = (
        gdf.get("attrib_src", pd.Series([False] * len(gdf), index=gdf.index)).notna()
        if "attrib_src" in gdf.columns
        else pd.Series([False] * len(gdf), index=gdf.index)
    )
    gdf["_priority"] = (~has_attrib).astype(int)  # NWM = 0, non-NWM = 1

    # Add hydroseq for tiebreaking
    gdf["_ref_fp_id_num"] = pd.to_numeric(gdf["ref_fp_id"], errors="coerce")
    gdf = gdf.merge(
        hydro_lookup,
        left_on="_ref_fp_id_num",
        right_on="ref_fp_id",
        how="left",
        suffixes=("", "_hydro"),
    )

    # For each duplicate group, sort and keep the first (best) entry
    keep_indices: list[int] = []
    for _lid, group in gdf[dupe_mask].groupby(cfg.lakes.output_comid_field):
        sorted_group = group.sort_values(["_priority", "hydroseq"], na_position="last")
        keep_indices.append(sorted_group.index[0])

    drop_indices = gdf[dupe_mask].index.difference(keep_indices)
    gdf = gdf.drop(index=drop_indices)
    gdf = gdf.drop(
        columns=["_ref_fp_id_num", "ref_fp_id_hydro", "hydroseq", "_priority"],
        errors="ignore",
    )

    logger.info(f"Dropped {len(drop_indices)} duplicate lake rows; kept {len(keep_indices)}")
    return gdf


def _join_nid(cfg: HFConfig, res_df: gpd.GeoDataFrame, nid_df: pd.DataFrame) -> gpd.GeoDataFrame:
    """Join National Inventory Dams (NID) data to lakes.

    NID is joined by attribute (nidid) first. When multiple NID coordinate
    records exist for the same nidid, the spatially closest one to the lake
    point is kept.

    NWM lakes (attrib_src is set) already have NID info from placement
    improvement and are excluded from the deduplication logic, then
    stitched back.
    """
    # Columns to retain from NID for hydraulics computation
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

    # Early return if NID is not available
    if not cfg.lakes.ref_res.run:
        logger.info("Reference reservoirs not run. Skipping NID join.")
        res_df[keep_cols] = None
        return res_df

    if nid_df.empty:
        logger.info("NID table not available. Skipping NID join.")
        res_df[keep_cols] = None
        return res_df

    if "nid" not in res_df.columns:
        logger.info(
            "'nid' column not found in lakes during NID join. "
            "NID data will not be used in hydraulics calculation."
        )
        res_df[keep_cols] = None
        return res_df

    # Preprocess NID table
    nid_df.columns = [col.lower() for col in nid_df.columns]

    for col in ("spillway_type", "dam_type"):
        if col in nid_df.columns:
            nid_df[col] = nid_df[col].astype("string")
    for col in ("structural_height", "dam_height", "nid_height", "surface_area", "hydraulic_height"):
        if col in nid_df.columns:
            nid_df[col] = pd.to_numeric(nid_df[col], errors="coerce")
    if "surface_area" not in nid_df.columns:
        nid_df["surface_area"] = np.nan

    # Make NID spatial
    nid_df = nid_df.loc[~nid_df["latitude"].isnull() & ~nid_df["longitude"].isnull()].copy()
    nid_gdf = gpd.GeoDataFrame(
        nid_df, geometry=gpd.points_from_xy(nid_df["longitude"], nid_df["latitude"]), crs=4326
    )
    nid_gdf = nid_gdf.to_crs(res_df.crs)

    # Rename for merge
    nid_gdf = nid_gdf.rename(columns={"nidid": "nid", "dam_name": "dam_name_nid"})

    # Filter NID to only the nid values present in our lakes
    nid_vals = res_df["nid"].dropna().unique()
    nid_gdf = nid_gdf[nid_gdf["nid"].isin(nid_vals)].copy()

    # Clean string nulls
    res_df = res_df.replace(["<NA>", "None"], None)
    res_df = res_df.drop_duplicates(keep="first")

    # Separate NWM lakes (already have NID data from placement improvement)
    nwm_df = res_df.loc[res_df["attrib_src"].notna()].copy()
    res_df = res_df.loc[res_df["attrib_src"].isna()].copy()

    # Exclude non-NWM rows that share identifying attributes with NWM lakes.
    if not nwm_df.empty:
        # Cast comparison columns to string for type-safe isin checks
        lake_id_col = cfg.lakes.output_comid_field
        res_df[lake_id_col] = res_df[lake_id_col].astype(str)
        nwm_df[lake_id_col] = nwm_df[lake_id_col].astype(str)
        res_df["nid"] = res_df["nid"].astype(str)
        nwm_df["nid"] = nwm_df["nid"].astype(str)
        res_df["dam_id"] = res_df["dam_id"].astype(str)
        nwm_df["dam_id"] = nwm_df["dam_id"].astype(str)

        res_df = res_df.loc[
            (~res_df["nid"].isin(nwm_df["nid"]))
            & (~res_df["dam_id"].isin(nwm_df["dam_id"]))
            & (~res_df[lake_id_col].isin(nwm_df[lake_id_col]))
        ].copy()

    # Attribute-merge NID onto non-NWM lakes
    res_df = res_df.merge(nid_gdf, on="nid", how="left")

    # Dedup: one (lake_id, nid) -> potentially many NID coordinate records
    dupe_cols = [cfg.lakes.output_comid_field, "nid"]
    dupe_mask = res_df.duplicated(subset=dupe_cols, keep=False)
    if dupe_mask.any():
        logger.info(
            f"Resolving {dupe_mask.sum()} duplicate NID coordinate rows "
            f"for {res_df[dupe_mask][dupe_cols[0]].nunique()} lake(s)"
        )
        dupes = res_df[dupe_mask].copy()
        non_dupes = res_df[~dupe_mask].copy()

        # Compute distance between lake (geometry_x) and NID point (geometry_y)
        dupes["_nid_dist"] = dupes["geometry_x"].distance(dupes["geometry_y"])
        keep_idx = dupes.groupby(dupe_cols)["_nid_dist"].idxmin()
        deduped = dupes.loc[keep_idx].drop(columns=["_nid_dist"])

        # Restore active geometry from geometry_x
        deduped = deduped.rename(columns={"geometry_x": "geometry"}).drop(
            columns=["geometry_y"], errors="ignore"
        )
        non_dupes = non_dupes.rename(columns={"geometry_x": "geometry"}).drop(
            columns=["geometry_y"], errors="ignore"
        )
        res_df = gpd.GeoDataFrame(pd.concat([non_dupes, deduped], ignore_index=True), crs=cfg.crs)
    else:
        # No duplicates: restore active geometry from geometry_x
        res_df = res_df.rename(columns={"geometry_x": "geometry"}).drop(
            columns=["geometry_y"], errors="ignore"
        )
        res_df = gpd.GeoDataFrame(res_df, crs=cfg.crs)

    # Stitch NWM lakes back in
    output = pd.concat([res_df, nwm_df], ignore_index=True)

    # Rename nid -> nidid for output schema consistency
    output = output.rename(columns={"nid": "nidid"})

    output[cfg.lakes.output_comid_field] = output[cfg.lakes.output_comid_field].astype(str).copy()

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
    """Create location-based lake IDs using Open Location Code (Plus Codes).

    Converts the centroid of each lake geometry to a deterministic base-20 integer ID similar to the process for generating the location based fp_id's in ./src/hydrofabric_builds/lakes/lakes.py
    """
    gdf_wgs84 = gdf.to_crs("EPSG:4326")
    used_ints: set[int] = set()
    ids: list[int] = []

    for geom in gdf_wgs84.geometry:
        pt = geom.centroid
        _, olc_int = _encode_unique(pt.y, pt.x, code_length=12, used_ints=used_ints)
        used_ints.add(olc_int)
        ids.append(olc_int)

    gdf["nhf_lake_id"] = ids
    gdf["nhf_lake_id"] = gdf["nhf_lake_id"].astype("Int64")
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

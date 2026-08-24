"""Crosswalk low head dams.

Script to crosswalk low head dams from USACE LHDI to 1 or 2 input lake polygon layers.

New polygons will be created only if the lhd point is NOT within a buffered
distance of either of the lake inputs.

Polygons default to the NHF path for NWM lakes and reference waterbodies
    NWM lakes: ./data/sconus/lakes/input/nwm_lakes_sconus_input.gpkg
    reference-waterbodies: "./data/sconus/lakes/input/reference_waterbodies.gpkg"

Outputs to a GPKG with standard dam information
"""

import argparse

import geopandas as gpd
import numpy as np

"""Read LHDI data and convert to GeoDataFrame

Reads the raw LHDI data and filters on the review status being "Confirmed."  Gathers
necessary column to populate the standard dam information fields and adds default 
values for missing attributes.

Converts values from feet to meters currently.

"""


def lhdi_to_gdf(lhdi_path) -> gpd.GeoDataFrame:
    lhdi_gdf = gpd.read_file(lhdi_path, layer="low_head_dams_raw")
    lhdi_verified = lhdi_gdf[lhdi_gdf.reviewStatusId == "Confirmed"]
    lhdi_verified = lhdi_verified.reset_index(drop=True)
    lhdi_columns_to_keep = [
        "name",
        "lhdId",
        "length",
        "damHeight",
        "hydraulicHeight",
        "nidHeight",
        "damVolume",
        "purposeIds",
        "geometry",
    ]
    lhdi_columns_rename = {
        "name": "dam_name",
        "lhdId": "nidid",
        "length": "dam_length",
        "damHeight": "dam_height",
        "hydraulicHeight": "hydraulic_height",
        "nidHeight": "nid_height",
        "damVolume": "nid_storage",
        "purposeIds": "purposes",
    }
    lhdi_columns_retype = {
        "dam_length": "float64",
        "dam_height": "float64",
        "hydraulic_height": "float64",
        "nid_height": "float64",
        "nid_storage": "float64",
    }
    lhdi_verified = lhdi_verified[lhdi_columns_to_keep]
    lhdi_verified = lhdi_verified.rename(columns=lhdi_columns_rename)
    lhdi_verified = lhdi_verified.astype(lhdi_columns_retype)
    lhdi_verified = lhdi_verified.assign(
        dam_type="low head dam",
        spillway_type="",
        spillway_width="",
        structural_height=np.nan,
        surface_area=np.nan,
        wb_areasqkm=np.nan,
        normal_storage=np.nan,
        max_storage=np.nan,
        hazard="",
    )
    return lhdi_verified


"""Crosswalk low head dams to lake layers.

Creates a crosswalk between low head dams and lake layers, identifying dams that are
 within a specified buffer distance from lakes.  Dams that are less than the buffer
 distance from a lake are excluded from the crosswalk.
"""


def crosswalk_low_head_dams(
    lhd: gpd.GeoDataFrame,
    lakes_1: gpd.GeoDataFrame,
    lakes_2: gpd.GeoDataFrame | None,
    lakes_1_key: str,
    lakes_2_key: str | None,
    buffer: int = 300,
) -> gpd.GeoDataFrame:
    # Join lhd polygons to first lake layer
    lhd = lhd.to_crs(lakes_1.crs)
    lhd_columns = lhd.columns.tolist()
    lhd_join_lakes_1 = gpd.sjoin_nearest(
        lhd, lakes_1, how="left", distance_col="dist_to_lake_1"
    )
    lhd_columns.append("dist_to_lake_1")
    lhd_join_lakes_1 = lhd_join_lakes_1[lhd_columns]

    # If available, join lhd point to second lake layer
    if isinstance(lakes_2, gpd.GeoDataFrame):
        lakes_2 = lakes_2.to_crs(lakes_1.crs)
        lhd_join_lakes_2 = gpd.sjoin_nearest(
            lhd, lakes_2, how="left", distance_col="dist_to_lake_2"
        )
        lhd_join_lakes_2 = lhd_join_lakes_2[["nidid", "dist_to_lake_2"]]
        output = lhd_join_lakes_1.merge(lhd_join_lakes_2, on="nidid")
        output = output[
            (output.dist_to_lake_1 > buffer) & (output.dist_to_lake_2 > buffer)
        ]

    else:
        output = lhd_join_lakes_1[lhd_join_lakes_1.dist_to_lake_1 > buffer]
    return output


"""Create buffered low head dam polygons.

Generates square buffer polygons around low head dam points based on the dam length.
"""


def create_lhd_polygons(crosswalk_gdf):
    lhd_gdf = crosswalk_gdf.to_crs("EPSG:5070")
    lhd_gdf["buffer_dist"] = (lhd_gdf["dam_length"].fillna(20)) / 2
    lhd_gdf["geometry"] = lhd_gdf.buffer(
        distance=(lhd_gdf.buffer_dist), cap_style="square"
    )
    lhd_gdf = lhd_gdf.drop(columns=["dist_to_lake_1", "dist_to_lake_2", "buffer_dist"])
    lhd_gdf.to_file(lhd_polygons_path, layer="low_head_dams_polygons")


lakes_1_path = (
    "/mnt/d/NOAA/EDFS/NGWPC-11532/low_head_dams/Data/nwm_lakes_sconus_input.gpkg"
)
lakes_2_path = (
    "/mnt/d/NOAA/EDFS/NGWPC-11532/low_head_dams/Data/reference_waterbodies.gpkg"
)
lhd_path = "/mnt/d/NOAA/EDFS/NGWPC-11532/low_head_dams/Data/low_head_dams.gpkg"
lhd_polygons_path = (
    "/mnt/d/NOAA/EDFS/NGWPC-11532/low_head_dams/Data/low_head_dams_polygons.gpkg"
)

lakes_1 = gpd.read_file(lakes_1_path)
lakes_2 = gpd.read_file(lakes_2_path)

lhd = lhdi_to_gdf(lhd_path)
lhd_crosswalk = crosswalk_low_head_dams(lhd, lakes_1, lakes_2, "newID", "comid")
create_lhd_polygons(lhd_crosswalk)

"""Contains all code for building active NWM lakes in task"""

import logging
from pathlib import Path

from hydrofabric_builds.config import HFConfig
from hydrofabric_builds.helpers.flowpath_association import (
    associate_flowpaths_nearest_point,
    associate_flowpaths_polygon_outlet,
    join_attributes,
)
import geopandas as gpd
from hydrofabric_builds.hydrofabric.utils import _crosswalk_nexus, _crosswalk_reference
logger = logging.getLogger(__name__)


def build_nwm_lakes(cfg) -> gpd.GeoDataFrame:
    """Processes NWM lakes

    """
    # Preprocess lakes by associating with flowpaths if requested or if processed path does not exist
    if cfg.lakes.associate_flowpaths or not cfg.lakes.processed_path.exists():
        # Use nearest point association method
        if cfg.lakes.flowpath_association_method == "nearest_point":
            logger.info("Associating flowpath with points")
            gdf = associate_flowpaths_nearest_point(
                points_path=cfg.lakes.input_path,
                flowpaths_path=Path(cfg.build.reference_flowpaths_path),
                search_radius_m=cfg.lakes.search_radius_m,
                point_id=cfg.lakes.id_field,
                flowpath_id="flowpath_id",
                flowpath_id_out_field="ref_fp_id",
                points_layer=cfg.lakes.input_layer,
            )
        # use polygon flowpath outlet method
        elif cfg.lakes.flowpath_association_method == "polygon_outlet":
            logger.info("Associating flowpaths with polygons")
            gdf = associate_flowpaths_polygon_outlet(
                polygon_path=cfg.lakes.input_path,
                flowpaths_path=Path(cfg.build.reference_flowpaths_path),
                search_radius_m=cfg.lakes.search_radius_m,
                min_preferred_intersection_len_m=cfg.lakes.min_preferred_intersection_len_m,
                flowpath_id="flowpath_id",
                flowpath_id_out_field="ref_fp_id",
                polygon_layer=cfg.lakes.input_layer,
            )
            if cfg.lakes.attrib_src_path:
                gdf = join_attributes(
                    gdf,
                    attrib_dst_key=cfg.lakes.id_field,
                    attrib_src_path=cfg.lakes.attrib_src_path,
                    attrib_src_layer=cfg.lakes.attrib_src_layer,
                    attrib_src_key=cfg.lakes.attrib_src_key,
                    attrib_src_fields=cfg.lakes.fields.copy(),
                    rename=True,
                )

        # invalid method
        else:
            raise ValueError("Config contained invalid Lakes flowpath association method")

    # Save nwm_lakes layer to NHF
    gdf.to_file(cfg.lakes.nwm_lake_file_path, layer="lakes", driver="GPKG", overwrite=True)

    return gdf

# maybe this can be one function just with separate cfg for IDs?

def build_misc_lake(src, cfg):
    """Associates flowpaths for a lake file"""
    gdf = gpd.read_file(src)

    # Use nearest point association method
    if cfg.lakes.flowpath_association_method == "nearest_point":
            logger.info(f"Associating flowpath with points for {src}")
            gdf = associate_flowpaths_nearest_point(
                points_path=cfg.lakes.input_path,
                flowpaths_path=Path(cfg.build.reference_flowpaths_path),
                search_radius_m=cfg.lakes.search_radius_m,
                point_id=cfg.lakes.id_field,
                flowpath_id="flowpath_id",
                flowpath_id_out_field="ref_fp_id",
                points_layer=cfg.lakes.input_layer,
            )
        # use polygon flowpath outlet method
    elif cfg.lakes.flowpath_association_method == "polygon_outlet":
        logger.info(f"Associating flowpaths with polygons for {src}")
        gdf = associate_flowpaths_polygon_outlet(
            polygon_path=cfg.lakes.input_path,
            flowpaths_path=Path(cfg.build.reference_flowpaths_path),
            search_radius_m=cfg.lakes.search_radius_m,
            min_preferred_intersection_len_m=cfg.lakes.min_preferred_intersection_len_m,
            flowpath_id="flowpath_id",
            flowpath_id_out_field="ref_fp_id",
            polygon_layer=cfg.lakes.input_layer,
        )
        if cfg.lakes.attrib_src_path:
            gdf = join_attributes(
                gdf,
                attrib_dst_key=cfg.lakes.id_field,
                attrib_src_path=cfg.lakes.attrib_src_path,
                attrib_src_layer=cfg.lakes.attrib_src_layer,
                attrib_src_key=cfg.lakes.attrib_src_key,
                attrib_src_fields=cfg.lakes.fields.copy(),
                rename=True,
            )

    gdf.to_file()
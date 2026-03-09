"""Contains all code for building OCONUS waterbodies in task"""

import logging
from typing import Any, cast

from hydrofabric_builds.config import HFConfig
from hydrofabric_builds.helpers.flowpath_association import (
    associate_flowpaths_nearest_point,
    associate_flowpaths_polygon_outlet,
    join_attributes,
)
from hydrofabric_builds.hydrofabric.nwm_lakes import complete_nwm_lakes
from hydrofabric_builds.reservoirs.data_prep.hydraulics import populate_nwm_hydaulics

logger = logging.getLogger(__name__)


def build_nwm_lakes(**context: dict[str, Any]) -> dict[str, Any]:
    """Builds NWM lakes

    Parameters
    ----------
    **context : dict
        Airflow-compatible context containing:
        - ti : TaskInstance for XCom operations
        - config : HFConfig with pipeline configuration
        - task_id : str identifier for this task
        - run_id : str identifier for this pipeline run
        - ds : str execution date
        - execution_date : datetime object

    """
    cfg = cast(HFConfig, context["config"])

    # Preprocess lakes by associating with flowpaths if requested or if processed path does not exist
    if cfg.nwm_lakes.associate_flowpaths or not cfg.nwm_lakes.processed_path.exists():
        # Use nearest point association method
        if cfg.nwm_lakes.flowpath_association_method == "nearest_point":
            logger.info("Associating flowpath points")
            gdf = associate_flowpaths_nearest_point(
                points_path=cfg.nwm_lakes.input_path,
                flowpaths_path=cfg.output_file_path,
                flowpath_layer="flowpaths",
                search_radius_m=cfg.nwm_lakes.search_radius_m,
                point_id=cfg.nwm_lakes.id_field,
                flowpath_id="fp_id",
                flowpath_id_out_field="fp_id",
                points_layer=cfg.nwm_lakes.input_layer,
            )
        # use polygon flowpath outlet method
        elif cfg.nwm_lakes.flowpath_association_method == "polygon_outlet":
            logger.info("Creating waterbodies table based on point layer")
            gdf = associate_flowpaths_polygon_outlet(
                polygon_path=cfg.nwm_lakes.input_path,
                flowpaths_path=cfg.output_file_path,
                flowpath_layer="flowpaths",
                search_radius_m=cfg.nwm_lakes.search_radius_m,
                polygon_id=cfg.nwm_lakes.id_field,
                flowpath_id="fp_id",
                flowpath_id_out_field="fp_id",
                polygon_layer=cfg.nwm_lakes.input_layer,
            )
            if cfg.nwm_lakes.attrib_src_path:
                gdf = join_attributes(
                    gdf,
                    attrib_dst_key=cfg.nwm_lakes.id_field,
                    attrib_src_path=cfg.nwm_lakes.attrib_src_path,
                    attrib_src_layer=cfg.nwm_lakes.attrib_src_layer,
                    attrib_src_key=cfg.nwm_lakes.attrib_src_key,
                    attrib_src_fields=cfg.nwm_lakes.fields,
                    rename=True,
                )

        # invalid method
        else:
            raise ValueError("Config contained invalid NWM Lakes flowpath association method")

        # Save pre-processed file
        gdf.to_file(cfg.nwm_lakes.processed_path, overwrite=True, driver="GPKG")
        del gdf

    # Populate hydraulics if requested - use flowpath associated file
    # If hydaulics is skipped, columns will be assumed to be present in processed file or added as null
    if cfg.nwm_lakes.populate_nwm_hydaulics:
        gdf = populate_nwm_hydaulics(cfg.nwm_lakes.processed_path)
        gdf.to_file(cfg.nwm_lakes.processed_path, overwrite=True, driver="GPKG")

    # Complete the nwm_lakes file with requested columns
    gdf = complete_nwm_lakes(nwm_lakes_path=cfg.nwm_lakes.processed_path, fields=cfg.nwm_lakes.fields)

    # Save nwm_lakes layer to NHF
    gdf.to_file(cfg.output_file_path, layer="nwm_lakes", driver="GPKG", overwrite=True)

    return {"nwm_lakes": "done"}

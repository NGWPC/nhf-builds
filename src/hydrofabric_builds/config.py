"""A pydantic basemodel for setting HFConfig defaults"""

import os
from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, Field, field_validator
from pyprojroot import here

from hydrofabric_builds._version import __version__

HYDROFABRIC_OUTPUT_FILE = here() / f"data/base_hydrofabric_{__version__}.gpkg"


class HFConfig(BaseModel):
    """A config validation class for default build settings"""

    crs: str = Field(
        default="EPSG:5070",
        description="Coordinate Reference System for the hydrofabric builds. Defaults to Conus Albers",
    )

    divide_aggregation_threshold: float = Field(
        default=3.0, description="Threshold for divides to aggreagate into an upstream catchment [km^2]"
    )

    output_dir: Path = Field(
        default=here() / "data/",
        description="The directory for output files to be saved from Hydrofabric builds",
    )

    output_file: Path = Field(default=HYDROFABRIC_OUTPUT_FILE, description="The output file")

    reference_divides_path: str = Field(
        default="s3://edfs-data/reference/super_conus/reference_divides.parquet",
        description="The location of the reference fabric divides. Default is in the NGWPC Test AWS account",
    )
    reference_flowpaths_path: str = Field(
        default="s3://edfs-data/reference/super_conus/reference_flowpaths.parquet",
        description="The location of the reference fabric flowpaths. Default is in the NGWPC Test AWS account",
    )

    debug_outlet_count: int | None = Field(
        default=None,
        description="Debug setting to limit the number of outlets processed. None (default) processes all outlets. Set to a positive integer to limit for testing.",
    )
    divide_attributes_processes: int = Field(
        description="Number of processes to run during multiprocessing for divide attributes",
        default=os.cpu_count(),
    )
    divide_attributes_config_path: str = Field(
        default=here() / "configs/example_divide_attributes_config.yaml",
        description="YAML model definition for building divide attributes",
    )

    run_build_hydrofabric_tasks: bool = Field(
        default=True, description="Decides if we want to run the hydrofabric build tasks"
    )

    run_divide_attributes_task: bool = Field(
        default=True, description="Decides if we want to run the divide attributes task"
    )

    run_flowpath_attributes_task: bool = Field(
        default=True, description="Decides if we want to run the flowpath attributes task"
    )

    flowpath_attributes_config: dict = Field(
        description="Dictionary of flowpath attributes values as found in FlowpathAttributesModelConfig",
        default=None,
    )

    @field_validator("debug_outlet_count")
    @classmethod
    def validate_debug_outlet_count(cls, v: int | None) -> int | None:
        """Validate debug_outlet_count is None or positive."""
        if v is not None and v <= 0:
            raise ValueError("debug_outlet_count must be None (for all outlets) or a positive integer")
        return v

    @classmethod
    def from_yaml(cls, path: str) -> Self:
        """An internal method to read a config from a YAML file

        Parameters
        ----------
        path : str
            The path to the provided YAML file

        Returns
        -------
        HFConfig
            A configuration object validated
        """
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

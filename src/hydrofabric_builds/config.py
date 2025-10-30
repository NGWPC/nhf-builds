"""A pydantic basemodel for setting HFConfig defaults"""

from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, Field
from pyprojroot import here


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

    reference_divides_path: str = Field(
        default="s3://edfs-data/reference/super_conus/reference_divides.parquet",
        description="The location of the reference fabric divides. Default is in the NGWPC Test AWS account",
    )
    reference_flowpaths_path: str = Field(
        default="s3://edfs-data/reference/super_conus/reference_flowpaths.parquet",
        description="The location of the reference fabric flowpaths. Default is in the NGWPC Test AWS account",
    )

    debug_outlet_count: int = Field(
        default=-1,
        description="A debug setting to only run a specified number out outlets through the runner. Setting to -1 as a default to avoidd premature activation",
    )
    divide_attributes_processes: int = Field(
        default=11, description="Number of processes to run during multiprocessing for divide attributes"
    )
    divide_attributes_config_path: str = Field(
        default=here() / "configs/divide_attributes_config.yaml",
        description="YAML model definition for building divide attributes",
    )

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

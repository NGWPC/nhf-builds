"""A pydantic basemodel for setting HFConfig defaults"""

from typing import Self

import yaml
from pydantic import BaseModel, Field


class HFConfig(BaseModel):
    """A config validation class for default build settings"""

    divide_aggregation_threshold: float = Field(
        default=3.0, description="Threshold for divides to aggreagate into an upstream catchment [km^2]"
    )

    reference_divides_path: str = Field(
        default="s3://edfs-data/reference/super_conus/reference_divides.parquet",
        description="The location of the reference fabric divides. Default is in the NGWPC Test AWS account",
    )
    reference_flowpaths_path: str = Field(
        default="s3://edfs-data/reference/super_conus/reference_flowpaths.parquet",
        description="The location of the reference fabric flowpaths. Default is in the NGWPC Test AWS account",
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

"""A pydantic basemodel for setting HFConfig defaults"""

from typing import Self

import yaml
from pydantic import BaseModel, Field


class HFConfig(BaseModel):
    """A config validation class for default build settings"""

    dx: float = Field(default=3000, description="Discretization length for segments")
    reference_fabric_path: str = Field(
        default="/vsis3/edfs-data/reference/sc_reference_fabric.gpkg",
        description="The location of the reference fabric. Default is in the NGWPC Test AWS account",
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

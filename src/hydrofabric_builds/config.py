"""A pydantic basemodel for setting HFConfig defaults"""

from typing import Self

from pydantic import BaseModel, Field


class HFConfig(BaseModel):
    """A config validation class for default build settings"""

    dx: float = Field(3000, description="Discretization length for segments")

    @classmethod
    def from_yaml(cls, path: str) -> Self:
        """An internal method to read a config from a YAML file

        Parameters
        ----------
        path : str
            The path to the provided YAML file

        Returns
        -------
        _type_
            _description_
        """
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

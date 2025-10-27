"""A file to host all Hydrofabric Schemas"""

from enum import Enum, StrEnum

import pyarrow as pa
from pydantic import BaseModel, ConfigDict, Field


class Classifications(BaseModel):
    """A Pydantic BaseModel Container to contain flowpath_id classifications for aggregation"""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    aggregation_pairs: list[tuple[str, ...]] = Field(
        default_factory=list,
        description=(
            "A list of tuples for flowpaths to be aggregated together. Format: (downstream_id, upstream_id, ...) where downstream merges into upstream"
        ),
    )
    no_divide_connectors: list[str] = Field(
        default_factory=list,
        description=(
            "A list of flowpath IDs that do not have divides and connect rivers. These are errors in the reference that will not be carried into the hydrofabric"
        ),
    )
    minor_flowpaths: set[str] = Field(
        default_factory=set,
        description=(
            "Reference flowpaths classified as 'minor' tributaries. These are flowpaths that are stream-order 1, with a total DA of < threshold where routing will not be run"
        ),
    )
    independent_flowpaths: list[str] = Field(
        default_factory=list,
        description=(
            "Flowpaths that remain independent and are NOT aggregated. These are large catchments (areasqkm > threshold) that form their own divides"
        ),
    )
    connector_segments: list[str] = Field(
        default_factory=list,
        description=(
            "Small flowpaths (areasqkm < threshold) that connect two higher-order streams. These have two upstream flowpaths where both have streamorder > 1. They remain independent despite being small because they serve as connectors between large stream branches, and aggregation would present inconsistencies within routing"
        ),
    )
    subdivide_candidates: list[str] = Field(
        default_factory=list,
        description=(
            "Flowpaths marked to be preserved for creating sub-divides later. "
            "These are typically order-2 streams where two order-1 tributaries merge. "
            "While they may be aggregated for regular divides, they represent important "
            "hydrologic points where sub-catchments should be delineated."
        ),
    )
    upstream_merge_points: list[str] = Field(
        default_factory=list,
        description=(
            "Flowpaths where upstream tributaries merge into the mainstem. These locations are significant for sub-divide delineation as they represent points where multiple flow paths converge"
        ),
    )
    processed_flowpaths: set[str] = Field(
        default_factory=set,
        description=(
            "Set of all flowpath IDs that have been processed during the outlet upstream tracing. Used internally to prevent re-processing flowpaths by mistake (which creates cycles)"
        ),
    )
    cumulative_merge_areas: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Tracks cumulative drainage areas being merged into each flowpath. Key: flowpath_id, Value: cumulative area (km²) from all upstream merges. Helps identify when to stop chaining aggregations"
        ),
    )


class Aggregations(BaseModel):
    """A Pydantic BaseModel Container to contain flowpath_id classifications for aggregation"""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    aggregates: list[dict] = Field(
        description=(
            "A list of geometries for aggreagated flowpaths and divides along with upstream and downstream identifiers"
        ),
    )
    independents: list[dict] = Field(
        description=("A list of independent segments and their geometries"),
    )
    minor_flowpaths: list[dict] = Field(
        description=("A list of minor flowpaths and their geometries"),
    )
    no_divide_connectors: list[dict] = Field(
        description=(
            "A list of flowpaths that connect to multiple upstream components that do not have geometries"
        ),
    )
    small_scale_connectors: list[dict] = Field(
        description=("A list of small-scale connection segments and their geometries"),
    )
    connectors: list[dict] = Field(
        description=("A list of connection segments and their geometries"),
    )


class Flowpaths:
    """The schema for flowpaths table"""

    @classmethod
    def columns(cls) -> list[str]:
        """Returns the columns associated with this schema

        Returns
        -------
        list[str]
            The schema columns
        """
        return [
            "fp_id",
            "dn_nex_id",
            "up_nex_id",
            "div_id",
            "geometry",
        ]

    @classmethod
    def arrow_schema(cls) -> pa.Schema:
        """Returns the PyArrow Schema object.

        Returns
        -------
        pa.Schema
            PyArrow schema for flowpaths table
        """
        return pa.schema(
            [
                pa.field("fp_id", pa.int32(), nullable=False),
                pa.field("dn_nex_id", pa.int32(), nullable=False),
                pa.field("up_nex_id", pa.int32(), nullable=True),
                pa.field("div_id", pa.int32(), nullable=False),
                pa.field("geometry", pa.binary(), nullable=False),
            ]
        )


class Divides:
    """The schema for divides table"""

    @classmethod
    def columns(cls) -> list[str]:
        """Returns the columns associated with this schema

        Returns
        -------
        list[str]
            The schema columns
        """
        return ["div_id", "geometry"]

    @classmethod
    def arrow_schema(cls) -> pa.Schema:
        """Returns the PyArrow Schema object.

        Returns
        -------
        pa.Schema
            PyArrow schema for divides table
        """
        return pa.schema(
            [
                pa.field("div_id", pa.int32(), nullable=False),
                pa.field("geometry", pa.binary(), nullable=False),
            ]
        )


class Nexus:
    """The schema for Nexus table"""

    @classmethod
    def columns(cls) -> list[str]:
        """Returns the columns associated with this schema

        Returns
        -------
        list[str]
            The schema columns
        """
        return ["nex_id", "geometry"]

    @classmethod
    def arrow_schema(cls) -> pa.Schema:
        """Returns the PyArrow Schema object.

        Returns
        -------
        pa.Schema
            PyArrow schema for nexus table
        """
        return pa.schema(
            [
                pa.field("nex_id", pa.int32(), nullable=False),
                pa.field("geometry", pa.binary(), nullable=False),
            ]
        )


class HydrofabricDomains(StrEnum):
    """The domains used when querying the hydrofabric

    Attributes
    ----------
    AK : str
        Alaska
    CONUS : str
        Conterminous United States
    GL : str
        The US Great Lakes
    HI : str
        Hawai'i
    PRVI : str
        Puerto Rico, US Virgin Islands
    """

    AK = "ak_hf"
    CONUS = "conus_hf"
    GL = "gl_hf"
    HI = "hi_hf"
    PRVI = "prvi_hf"


class HydrofabricDomainsGPKG(StrEnum):
    """The domains used when querying the hydrofabric

    Attributes
    ----------
    AK : str
        Alaska
    CONUS : str
        Conterminous United States
    GL : str
        The US Great Lakes
    HI : str
        Hawai'i
    PRVI : str
        Puerto Rico, US Virgin Islands
    """

    AK = "AK"
    CONUS = "CONUS"
    GL = "GL"
    HI = "HI"
    PRVI = "PRVI"


class HydrofabricCRS(Enum):
    """The domains used when querying the hydrofabric

    Attributes
    ----------
    AK : str
        Alaska
    CONUS : str
        Conterminous United States
    GL : str
        The US Great Lakes
    HI : str
        Hawai'i
    PRVI : str
        Puerto Rico, US Virgin Islands
    """

    AK = 3338
    CONUS = 5070
    GL = 5070  # TEMP: MAY CHANGE
    HI = 102007
    PRVI = 32161

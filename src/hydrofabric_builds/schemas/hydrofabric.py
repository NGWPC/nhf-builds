"""A file to host all Hydrofabric Schemas"""

from enum import Enum, StrEnum
from pathlib import Path
from typing import Any, Self

import pyarrow as pa
from pydantic import BaseModel, ConfigDict, Field, model_validator

from hydrofabric_builds.helpers.stats import weighted_circular_mean, weighted_geometric_mean


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
    independent_flowpaths: set[str] = Field(
        default_factory=set,
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
    force_queue_flowpaths: set[str] = Field(
        default_factory=set,
        description="flowpaths that are required to be queued. These are only for streams deeply nested in no-divide connectors",
    )
    aggregation_set: set[str] = Field(
        default_factory=set,
        description=("A set flowpaths that have been aggregated together"),
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


class AggTypeEnum(StrEnum):
    """Zonal statistics aggregation types"""

    mean = "mean"
    mode = "mode"
    max = "max"
    circular_mean = "weighted_circular_mean"
    quartile_dist = "quartile_dist"
    geom_mean = "weighted_geometric_mean"


def get_operation(op: str) -> Any:
    """Helper to return zonal statistics aggregation type

    Parameters
    ----------
    op : str
        requested operation

    Returns
    -------
    Any
        string operation, list of string operations, or callable custom function
    """
    assert op in AggTypeEnum, ValueError("Invalid aggregation type")

    # make a mapping from the Enum keys where {key: key}
    mapping = dict(zip(AggTypeEnum.__members__.keys(), AggTypeEnum.__members__.keys(), strict=False))
    mapping.update(
        {
            "weighted_circular_mean": weighted_circular_mean,  # type: ignore[dict-item]
            "weighted_geometric_mean": weighted_geometric_mean,  # type: ignore[dict-item]
            "quartile_dist": ["quantile(q=0.25)", "quantile(q=0.5)", "quantile(q=0.75)", "quantile(q=1)"],  # type: ignore[dict-item]
        }
    )

    return mapping[op]


class DivideAttributeConfig(BaseModel):
    """Pydantic model for divide attributes attribute configuration"""

    agg_type: AggTypeEnum = Field(description="Zonal stats aggregation type")
    field_name: str = Field(description="Output field name for divide attribute")
    file_name: Path = Field(description="File path of attribute raster")
    tmp: Path = Field(
        description="Temp file path for parquet",
        default_factory=lambda data: Path("/tmp/divide-attributes") / f"tmp_{data['field_name']}.parquet",
    )

    @model_validator(mode="after")
    def make_tmp_dir(self: Any) -> Self:  # type: ignore[misc,type-var]
        """Model validator to create a temp directory if it does not exist"""
        self.tmp.parents[0].mkdir(parents=True, exist_ok=True)
        return self


class DivideAttributeModelConfig(BaseModel):
    """Pydantic model for divide attributes model configuration"""

    data_dir: Path = Field(description="Directory of all input data")
    divides_path: Path = Field(description="Divides path for entire domain")
    divide_id: str = Field(description="Field name for unique divide id", default="divide_id")
    attributes: list[DivideAttributeConfig] = Field(
        description="List of attributes to be computed. Specify in DivideAttributeConfig data model."
    )
    output: Path = Field(
        description="Output file path", default_factory=lambda data: data["data_dir"] / "divides_output.gpkg"
    )
    divides_path_list: list[Path] | None | None = Field(
        description="List of divides paths to use for parallel run. ex. list of VPU subsets.",
        default=None,
    )
    tmp_dir: Path = Field(description="Temp path for saving files", default=Path("/tmp/divide-attributes"))
    split_vpu: bool | None = Field(
        description="If running in parallel, this will split the domain divides file into separate files."
        "Each VPU can be run separately and will be stitched at end."
        "This will replace anything input to the `divides_path_list`",
        default=False,
    )
    debug: bool = Field(
        description="Setting debug to true will save all temporary files. Setting to false will delete files if run fails.",
        default=False,
    )

    @model_validator(mode="after")
    def make_tmp_dir(self: Any) -> Self:  # type: ignore[misc,type-var]
        """Model validator to create a temp directory if it does not exist"""
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        return self

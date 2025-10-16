"""A file to host all Hydrofabric Schemas"""

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
    minor_flowpaths: list[str] = Field(
        default_factory=list,
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
            "Small flowpaths (areasqkm < threshold) that connect two higher-order streams. These have two upstream flowpaths where both have stream_order > 1. They remain independent despite being small because they serve as connectors between large stream branches, and aggregation would present inconsistencies within routing"
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

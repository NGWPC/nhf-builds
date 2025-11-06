"""Contains all code for writing hydrofabric data"""

import logging
from typing import Any, cast

import polars as pl
from tqdm import tqdm

from hydrofabric_builds.hydrofabric.graph import (
    _build_rustworkx_object,
    _build_upstream_dict_from_nexus,
    _partition_all_outlet_subgraphs,
)
from hydrofabric_builds.hydrofabric.trace import _trace_single_flowpath_attributes
from hydrofabric_builds.hydrofabric.utils import _calculate_id_ranges_pure
from hydrofabric_builds.task_instance import TaskInstance

logger = logging.getLogger(__name__)


def trace_hydrofabric_attributes(**context: dict[str, Any]) -> dict:
    """Writes attributes to the hydrofabric from graph objects.

    Processes each outlet independently using partitioned subgraphs to avoid
    cycle issues and improve performance. Uses existing graph infrastructure
    by building upstream dictionary from nexus connections.

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

    Returns
    -------
    dict
        The reference flowpath and divides references in memory with traced attributes
    """
    ti = cast(TaskInstance, context["ti"])

    # Get combined hydrofabric data
    fp = ti.xcom_pull(task_id="reduce_base", key="flowpaths")
    nex = ti.xcom_pull(task_id="reduce_base", key="nexus")
    fp_pl = pl.from_pandas(fp.to_wkb())

    logger.info("trace_attributes task: Building upstream connectivity dictionary from nexus connections")
    upstream_dict = _build_upstream_dict_from_nexus(fp_pl)

    logger.info("trace_attributes task: Building rustworkx graph from upstream dictionary")
    graph, node_indices = _build_rustworkx_object(upstream_dict)

    logger.info("trace_attributes task: Partitioning subgraphs by outlet")
    outlet_nexus_ids = set(nex[nex["dn_fp_id"].isna()]["nex_id"])
    outlet_flowpaths = fp[fp["dn_nex_id"].isin(outlet_nexus_ids)]["fp_id"].tolist()
    outlet_subgraphs = _partition_all_outlet_subgraphs(
        outlets=outlet_flowpaths,
        graph=graph,
        node_indices=node_indices,
        reference_flowpaths=fp_pl,
        reference_divides=None,  # Not needed for tracing
        _id="fp_id",
    )

    # Calculate ID ranges for mainstems across all outlets
    outlet_aggregations = {
        outlet_fp_id: {
            "num_features": len(partition_data["subgraph"]),
            "nodes": set(range(len(partition_data["subgraph"]))),
        }
        for outlet_fp_id, partition_data in outlet_subgraphs.items()
    }
    id_ranges = _calculate_id_ranges_pure(outlet_aggregations)

    # Process each outlet basin
    logger.info(f"trace_attributes task: Building attributes for {len(outlet_subgraphs)} drainage basins")
    results = []
    for outlet_fp_id, partition_data in tqdm(outlet_subgraphs.items(), desc="Building Graph Attributes"):
        id_offset = id_ranges["outlet_id_ranges"][outlet_fp_id]["id_offset"]

        traced_basin = _trace_single_flowpath_attributes(
            outlet_fp_id=outlet_fp_id,
            partition_data=partition_data,
            id_offset=id_offset,
        )

        results.append(traced_basin)

    _gdf = pl.concat(results).to_pandas()
    _gdf["fp_id"] = _gdf["fp_id"].astype("Int64")
    updated_flowpaths_gdf = fp.merge(_gdf, on="fp_id", how="left")

    return {"flowpaths_with_attributes": updated_flowpaths_gdf}

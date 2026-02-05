"""Test whether the NHF hydrofabric flowpath network forms a directed acyclic graph.

The fp_id -> to_fp_id relationships are obtained by joining the flowpaths and nexus layers of the NHF GeoPackage.

Example usage:
    python examples/test_dag.py /path/to/nhf.gpkg
"""

from __future__ import annotations

import argparse
import sqlite3

import pandas as pd
import rustworkx as rx


def network_is_dag(edges: pd.DataFrame, id_col: str = "fp_id", to_id_col: str = "to_fp_id") -> bool:
    """Check if a flowpath network forms a DAG.

    Equivalent to hfutils::network_is_dag() in R.

    Parameters
    ----------
    edges : pd.DataFrame
        DataFrame with columns for source and target flowpath IDs.
    id_col : str
        Column name for source flowpath ID (default: "fp_id").
    to_id_col : str
        Column name for target/downstream flowpath ID (default: "to_fp_id").

    Returns
    -------
    bool
        True if network is a DAG, False otherwise.
    """
    graph = rx.PyDiGraph()
    node_indices: dict[int, int] = {}

    # Add nodes and edges from the edge list
    for _, row in edges.iterrows():
        fp_id = int(row[id_col])
        to_fp_id = int(row[to_id_col])

        if fp_id not in node_indices:
            node_indices[fp_id] = graph.add_node(fp_id)
        if to_fp_id not in node_indices:
            node_indices[to_fp_id] = graph.add_node(to_fp_id)

        graph.add_edge(node_indices[fp_id], node_indices[to_fp_id], None)

    return rx.is_directed_acyclic_graph(graph)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test whether NHF hydrofabric flowpath network is a DAG.")
    parser.add_argument(
        "gpkg_path",
        help="Path to the NHF GeoPackage file.",
    )

    args = parser.parse_args()

    print("=== NHF Hydrofabric DAG Test ===\n")
    print(f"GeoPackage: {args.gpkg_path}\n")

    con = sqlite3.connect(args.gpkg_path)

    # Join flowpaths to nexus to resolve fp_id -> to_fp_id relationships
    # WHERE clause filters out outlets (NULL downstream) since rustworkx's network_is_dag
    # doesn't handle NAs internally (unlike accumulate_downstream)
    edges = pd.read_sql_query(
        """
        SELECT f.fp_id, n.dn_fp_id AS to_fp_id
        FROM flowpaths f
        LEFT JOIN nexus n ON f.dn_nex_id = n.nex_id
        WHERE n.dn_fp_id IS NOT NULL
        """,
        con,
    )

    is_dag = network_is_dag(edges)
    print(f"Flowpath network is DAG: {is_dag}\n")

    con.close()


if __name__ == "__main__":
    main()

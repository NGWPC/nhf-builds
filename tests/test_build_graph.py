"""Unit tests for build_graph task"""

import pandas as pd

from hydrofabric_builds.hydrofabric.graph import _build_graph
from scripts.hf_runner import LocalRunner


class TestBuildGraphUnit:
    """Unit tests for build_graph with controlled data."""

    def test_simple_linear_network(self, runner: LocalRunner) -> None:
        """Test a simple linear chain: 1 -> 2 -> 3."""
        flowpaths = pd.DataFrame(
            {
                "flowpath_id": [1.0, 2.0, 3.0],
                "hydroseq": [100.0, 200.0, 300.0],
                "dnhydroseq": [200.0, 300.0, 0.0],  # 1->2->3, 3 is outlet
            }
        )

        network = _build_graph(flowpaths)

        assert len(network) == 2
        assert "2" in network
        assert "3" in network
        assert "1" in network["2"]
        assert "2" in network["3"]

    def test_branching_network(self, runner: LocalRunner) -> None:
        """Test a network with branching: 1,2 -> 3."""
        flowpaths = pd.DataFrame(
            {
                "flowpath_id": [1.0, 2.0, 3.0],
                "hydroseq": [100.0, 200.0, 300.0],
                "dnhydroseq": [300.0, 300.0, 0.0],  # Both 1 and 2 flow to 3
            }
        )

        network = _build_graph(flowpaths)

        assert len(network) == 1  # Only 3 has upstream
        assert "3" in network
        assert set(network["3"]) == {"1", "2"}

    def test_multiple_outlets(self, runner: LocalRunner) -> None:
        """Test network with multiple outlets."""
        flowpaths = pd.DataFrame(
            {
                "flowpath_id": [1.0, 2.0, 3.0, 4.0],
                "hydroseq": [100.0, 200.0, 300.0, 400.0],
                "dnhydroseq": [200.0, 0.0, 400.0, 0.0],  # 1->2(outlet), 3->4(outlet)
            }
        )

        network = _build_graph(flowpaths)

        assert len(network) == 2
        assert "2" in network
        assert "4" in network
        assert "1" in network["2"]
        assert "3" in network["4"]

    def test_filters_nan_dnhydroseq(self, runner: LocalRunner) -> None:
        """Test that NaN dnhydroseq values are filtered out."""
        flowpaths = pd.DataFrame(
            {
                "flowpath_id": [1.0, 2.0, 3.0],
                "hydroseq": [100.0, 200.0, 300.0],
                "dnhydroseq": [200.0, 300.0, float("nan")],  # 3 has NaN
            }
        )

        network = _build_graph(flowpaths)

        assert len(network) == 2
        assert "2" in network
        assert "3" in network

    def test_filters_zero_dnhydroseq(self, runner: LocalRunner) -> None:
        """Test that zero dnhydroseq values are filtered out."""
        flowpaths = pd.DataFrame(
            {
                "flowpath_id": [1.0, 2.0, 3.0],
                "hydroseq": [100.0, 200.0, 300.0],
                "dnhydroseq": [200.0, 0.0, 0.0],  # 2 and 3 have zero (outlets)
            }
        )

        network = _build_graph(flowpaths)

        assert len(network) == 1
        assert "2" in network
        assert "1" in network["2"]

    def test_complex_dendritic_network(self, runner: LocalRunner) -> None:
        """Test a more complex dendritic (tree-like) network."""
        flowpaths = pd.DataFrame(
            {
                "flowpath_id": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "hydroseq": [100.0, 200.0, 300.0, 400.0, 500.0, 600.0],
                "dnhydroseq": [300.0, 300.0, 600.0, 500.0, 600.0, 0.0],
                # 1,2 -> 3 -> 6(outlet)
                #        4 -> 5 -> 6(outlet)
            }
        )

        network = _build_graph(flowpaths)

        assert "3" in network
        assert set(network["3"]) == {"1", "2"}

        assert "5" in network
        assert "4" in network["5"]

        assert "6" in network
        assert set(network["6"]) == {"3", "5"}

    def test_isolated_flowpath(self, runner: LocalRunner) -> None:
        """Test network with an isolated flowpath (no connections)."""
        flowpaths = pd.DataFrame(
            {
                "flowpath_id": [1.0, 2.0, 3.0],
                "hydroseq": [100.0, 200.0, 999.0],  # 3 has unique hydroseq
                "dnhydroseq": [200.0, 0.0, 0.0],  # 1->2, 3 is isolated
            }
        )

        network = _build_graph(flowpaths)

        assert len(network) == 1
        assert "2" in network
        assert "1" in network["2"]
        # 3 should not appear in network at all since there are no upstream connections

    def test_empty_network(self, runner: LocalRunner) -> None:
        """Test with all flowpaths being outlets (no connections)."""
        flowpaths = pd.DataFrame(
            {
                "flowpath_id": [1.0, 2.0, 3.0],
                "hydroseq": [100.0, 200.0, 300.0],
                "dnhydroseq": [0.0, 0.0, 0.0],  # All outlets
            }
        )

        network = _build_graph(flowpaths)

        assert len(network) == 0  # No connections

    def test_string_id_consistency(self, runner: LocalRunner) -> None:
        """Test that flowpath_id float -> str conversions are consistent."""
        flowpaths = pd.DataFrame(
            {
                "flowpath_id": [6720675.0, 6722501.0],
                "hydroseq": [100.0, 200.0],
                "dnhydroseq": [200.0, 0.0],
            }
        )

        network = _build_graph(flowpaths)

        # Check that IDs are strings
        for downstream_id, upstream_ids in network.items():
            assert isinstance(downstream_id, str)
            assert all(isinstance(uid, str) for uid in upstream_ids)

        # Check specific values
        assert "6722501" in network
        assert "6720675" in network["6722501"]

    def test_large_flowpath_ids(self, runner: LocalRunner) -> None:
        """Test with 3 flopaths going to one outlet"""
        flowpaths = pd.DataFrame(
            {
                "flowpath_id": [6720675.0, 6720683.0, 6720773.0, 6720689.0],
                "hydroseq": [100.0, 200.0, 300.0, 400.0],
                "dnhydroseq": [200.0, 0.0, 200.0, 200.0],  # Multiple flowing to 683
            }
        )

        network = _build_graph(flowpaths)

        assert "6720683" in network
        assert set(network["6720683"]) == {"6720675", "6720773", "6720689"}

    def test_no_duplicate_upstream_connections(self, runner: LocalRunner) -> None:
        """Test that each upstream flowpath appears only once per downstream."""
        flowpaths = pd.DataFrame(
            {
                "flowpath_id": [1.0, 2.0, 3.0],
                "hydroseq": [100.0, 200.0, 300.0],
                "dnhydroseq": [300.0, 300.0, 0.0],
            }
        )

        network = _build_graph(flowpaths)

        # Check no duplicates in upstream lists
        for upstream_ids in network.values():
            assert len(upstream_ids) == len(set(upstream_ids))

    def test_return_value_structure(self, runner: LocalRunner) -> None:
        """Test that the return value has the correct structure."""
        flowpaths = pd.DataFrame(
            {
                "flowpath_id": [1.0, 2.0],
                "hydroseq": [100.0, 200.0],
                "dnhydroseq": [200.0, 0.0],
            }
        )

        network = _build_graph(flowpaths)

        assert isinstance(network, dict)

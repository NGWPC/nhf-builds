"""Unit tests for build_graph task"""

import pandas as pd

from hydrofabric_builds.hydrofabric.graph import _build_graph, _find_outlets_by_hydroseq


class TestBuildGraphUnit:
    """Unit tests for build_graph with controlled data."""

    def test_simple_linear_network(self) -> None:
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

    def test_branching_network(self) -> None:
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

    def test_multiple_outlets(self) -> None:
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

    def test_filters_nan_dnhydroseq(self) -> None:
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

    def test_filters_zero_dnhydroseq(self) -> None:
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

    def test_complex_dendritic_network(self) -> None:
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

    def test_isolated_flowpath(self) -> None:
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

    def test_empty_network(self) -> None:
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

    def test_string_id_consistency(self) -> None:
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

    def test_large_flowpath_ids(self) -> None:
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

    def test_no_duplicate_upstream_connections(self) -> None:
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

    def test_return_value_structure(self) -> None:
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


class TestFindOutletsByHydroseq:
    """Unit tests for _find_outlets_by_hydroseq function"""

    def test_single_outlet_with_zero_dnhydroseq(self) -> None:
        """Test finding outlet when dnhydroseq is 0."""
        flowpaths = pd.DataFrame(
            {
                "flowpath_id": [1.0, 2.0],
                "hydroseq": [100.0, 200.0],
                "dnhydroseq": [200.0, 0.0],  # 2 is outlet
            }
        )

        outlets = _find_outlets_by_hydroseq(flowpaths)

        assert len(outlets) == 1
        assert "2" in outlets

    def test_multiple_outlets(self) -> None:
        """Test finding multiple outlets."""
        flowpaths = pd.DataFrame(
            {
                "flowpath_id": [1.0, 2.0, 3.0, 4.0],
                "hydroseq": [100.0, 200.0, 300.0, 400.0],
                "dnhydroseq": [200.0, 0.0, 400.0, 0.0],  # 2 and 4 are outlets
            }
        )

        outlets = _find_outlets_by_hydroseq(flowpaths)

        assert len(outlets) == 2
        assert set(outlets) == {"2", "4"}

    def test_all_outlets(self) -> None:
        """Test when all flowpaths are outlets."""
        flowpaths = pd.DataFrame(
            {
                "flowpath_id": [1.0, 2.0, 3.0],
                "hydroseq": [100.0, 200.0, 300.0],
                "dnhydroseq": [0.0, 0.0, 0.0],  # All outlets
            }
        )

        outlets = _find_outlets_by_hydroseq(flowpaths)

        assert len(outlets) == 3
        assert set(outlets) == {"1", "2", "3"}

    def test_no_outlets_connected_network(self) -> None:
        """Test a fully connected circular network (no true outlets)."""
        # This is theoretical - shouldn't happen in real river networks
        flowpaths = pd.DataFrame(
            {
                "flowpath_id": [1.0, 2.0, 3.0],
                "hydroseq": [100.0, 200.0, 300.0],
                "dnhydroseq": [200.0, 300.0, 100.0],  # Circular: 1->2->3->1
            }
        )

        outlets = _find_outlets_by_hydroseq(flowpaths)

        assert len(outlets) == 0

    def test_linear_chain_single_outlet(self) -> None:
        """Test a simple linear chain with one outlet."""
        flowpaths = pd.DataFrame(
            {
                "flowpath_id": [1.0, 2.0, 3.0],
                "hydroseq": [100.0, 200.0, 300.0],
                "dnhydroseq": [200.0, 300.0, 0.0],  # 1->2->3, 3 is outlet
            }
        )

        outlets = _find_outlets_by_hydroseq(flowpaths)

        assert len(outlets) == 1
        assert "3" in outlets

    def test_mixed_outlet_conditions(self) -> None:
        """Test outlets identified by different conditions."""
        flowpaths = pd.DataFrame(
            {
                "flowpath_id": [1.0, 2.0, 3.0, 4.0],
                "hydroseq": [100.0, 200.0, 300.0, 400.0],
                "dnhydroseq": [200.0, 0.0, float("nan"), 999.0],
                # 1 -> 2 (zero)
                # 3 (NaN)
                # 4 (non-existent downstream)
            }
        )

        outlets = _find_outlets_by_hydroseq(flowpaths)

        assert len(outlets) == 2
        assert set(outlets) == {"2", "4"}

    def test_large_flowpath_ids(self) -> None:
        """Test with realistic large flowpath IDs."""
        flowpaths = pd.DataFrame(
            {
                "flowpath_id": [6720675.0, 6720683.0, 6720797.0],
                "hydroseq": [100.0, 200.0, 300.0],
                "dnhydroseq": [200.0, 300.0, 0.0],
            }
        )

        outlets = _find_outlets_by_hydroseq(flowpaths)

        assert len(outlets) == 1
        assert "6720797" in outlets

    def test_empty_dataframe(self) -> None:
        """Test with empty DataFrame."""
        flowpaths = pd.DataFrame(
            {
                "flowpath_id": [],
                "hydroseq": [],
                "dnhydroseq": [],
            }
        )

        outlets = _find_outlets_by_hydroseq(flowpaths)

        assert len(outlets) == 0

    def test_single_flowpath(self) -> None:
        """Test with a single flowpath (must be an outlet)."""
        flowpaths = pd.DataFrame(
            {
                "flowpath_id": [1.0],
                "hydroseq": [100.0],
                "dnhydroseq": [0.0],
            }
        )

        outlets = _find_outlets_by_hydroseq(flowpaths)

        assert len(outlets) == 1
        assert "1" in outlets
